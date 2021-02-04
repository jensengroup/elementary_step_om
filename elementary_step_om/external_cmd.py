import os
import shutil
import subprocess
import textwrap
import copy
import numpy as np


from xtb.interface import Calculator as CalculatorAPI
from xtb.interface import Molecule as MoleculeAPI
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_solvent, get_method

from rdkit import Chem
from .io_xtb import read_xtb_out

from .xyz2mol_local import xyz2AC


def run_cmd(cmd):
    """ Run cmd to run QM program """
    cmd = cmd.split()
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, err = p.communicate()
    return output.decode("utf-8")


class Calculator:
    def __init__(self, sdf=None, cpus=1):

        self.conformer = sdf
        self._ncores = cpus

        self._inp_filename = None
        self._working_dir = None

    def _make_working_dir(self):
        """ """
        wdir_name = self.conformer.label
        if os.chdir != wdir_name and not os.path.isdir(wdir_name):
            os.makedirs(wdir_name)

        self._make_working_dir = wdir_name
        os.chdir(self._make_working_dir)

    def add_structure(self, conf):
        self.conformer = conf


class xTB(Calculator):
    def __init__(self, *args, **kwds):
        self._xtb_kwds = self._make_cmd(kwds.pop("xtb_args", {"opt": "loose"}))
        super().__init__(*args, **kwds)

    def _make_cmd(self, xtb_args):
        xtb_kwds = "--norestart --strict "
        for arg, value in xtb_args.items():
            if value in [None, ""]:
                xtb_kwds += f"--{arg.lower()} "
            else:
                xtb_kwds += f"--{arg.lower()} {value.lower()} "
        return xtb_kwds

    def write_input(self):
        """ """
        self._inp_filename = f"{self.conformer.label}.xyz"

        with open(f"{ self._inp_filename}", "w") as inp:
            inp.write(self.conformer.write_xyz())

    def calculate(self, cpus=1, properties=["energy", "structure"]):
        """ """
        os.environ["OMP_NUM_THREADS"] = str(cpus)
        os.environ["MKL_NUM_THREADS"] = str(cpus)

        self._make_working_dir()
        self.write_input()

        output = run_cmd(f"xtb {self._inp_filename} {self._xtb_kwds}")
        with open("test.log", "w") as f:
            f.write(output)
        os.chdir("..")

        results = dict()
        if read_xtb_out(output, "converged"):
            results["converged"] = True
            for prop in properties:
                results[prop] = read_xtb_out(output, prop)
        else:
            results["converged"] = False

        return results


class xTBPath:
    """ """

    def __init__(self, reaction, label="reaction", cpus=1, memory=3):

        self.reaction = reaction
        self.nprocs = cpus
        self.memory = memory

        self.reaction_name = label

        self.npaths = 0
        self._initialize_path_search()

    def _initialize_path_search(self):

        self.path_reactant = copy.deepcopy(self.reaction.reactant)
        self.path_product = copy.deepcopy(self.reaction.product)

    def _make_working_dir(self):
        """ Make Workig directory, with the name `reaction_name` """
        if not os.path.isdir(self.reaction_name):
            os.makedirs(self.reaction_name)
        self._working_dir = self.reaction_name
        os.chdir(self._working_dir)

    def _finish_calculation(self):
        """ Clean directory to prepare for new path search """
        if os.path.basename(os.getcwd()) != self._working_dir:
            os.chdir(self._working_dir)

        os.chdir("..")
        os.rename(self._working_dir, f"{self._working_dir}{self._micro_iter_num}")

    def _write_input(self, kpush, kpull, alpha, temp, forward_reaction=True):
        """ """

        if forward_reaction:
            reactant_fname = self.reaction_name + "_r.xyz"
            product_fname = self.reaction_name + "_p.xyz"

        else:
            reactant_fname = self.reaction_name + "_p.xyz"
            product_fname = self.reaction_name + "_r.xyz"

        path_file = """\
        $path
            nrun=1   
            nopt=100    
            anopt=3
            kpush={0}
            kpull={1}
            alp={2}
            product={4}
        $end
        $scc
            temp={3}
        $end
        $opt
            optlevel=2
        $end
        """.format(
            kpush, kpull, alpha, temp, product_fname
        )

        with open("path.inp", "w") as ofile:
            ofile.write(textwrap.dedent(path_file))

        self.path_reactant._conformers[0].write_xyz(file=self.reaction_name + "_r.xyz")
        self.path_product._conformers[0].write_xyz(file=self.reaction_name + "_p.xyz")

        return reactant_fname, product_fname

    def _run_xtb_path(
        self,
        kpush,
        kpull,
        alpha,
        temp,
        solvent=None,
        chrg=0,
        multiplicity=1,
        forward_reaction=True,
    ):

        __XTB_PATH__ = "/home/koerstz/projects/origin_of_life/small_tests/version_tests_xtb/6.1/xtb-190527"

        os.environ["XTBHOME"] = __XTB_PATH__ + "/bin"
        os.environ["OMP_STACKSIZE"] = str(self.memory) + "G"
        os.environ["OMP_NUM_THREADS"] = str(self.nprocs)
        os.environ["MKL_NUM_THREADS"] = str(self.nprocs)

        self._make_working_dir()
        reac_fname, prod_fname = self._write_input(
            kpush, kpull, alpha, temp, forward_reaction=forward_reaction
        )

        os.system("ulimit -s unlimited")
        # cmd = f"xtb {reac_fname} --path {prod_fname} --input path.inp --gfn2")
        cmd = f"{__XTB_PATH__}/bin/xtb {reac_fname} --path --input path.inp --gfn2"
        if solvent is not None:
            cmd += f" --gbsa {solvent}"
        if chrg != 0:
            cmd += f" --chrg {chrg}"
        if multiplicity != 1:
            cmd += f" --uhf {multiplicity-1}"

        output = run_cmd(cmd)
        self._finish_calculation()
        return output

    def _is_reaction_complete(self, output):
        """ Checks that the path have an RMSD below 0.5 AA."""
        output_lines = output.split("\n")
        for line_number, line in enumerate(output_lines):
            if "run 1  barrier" in line:
                try:
                    rmsd = float(output_lines[line_number].split()[-1])
                    # print(rmsd)
                except:
                    print(line)
                if rmsd < 0.5:
                    return True
        return False

    def _read_complete_path(self):
        """ Read coordinates and energies from the final succesfull path """

        with open(f"{self.reaction_name}/xtbpath_1.xyz", "r") as path:
            xtbpath = path.read()

        paths_xyz = xtbpath.split("SCF done")
        natoms = int(paths_xyz[0])
        del paths_xyz[0]

        path_coords = np.zeros((len(paths_xyz), natoms, 3))
        energies = np.zeros(len(paths_xyz))
        for i, path_point_xyz in enumerate(paths_xyz):
            xyz_data = path_point_xyz.split("\n")
            energies[i] = float(xyz_data[0])
            del xyz_data[0]

            coords = np.zeros((natoms, 3))
            for j in range(natoms):
                coords[j] = list(
                    map(float, [coord for coord in xyz_data[j].split()][1:])
                )
            path_coords[i] = coords

        return energies, path_coords

    def _is_reac_prod_identical(self, charge=0, huckel=False):
        """
        This function ensures that if RMSD is above 0.5AA, and something
        happend in the last iteration - it is probably an intermediate.
        """
        pt = Chem.GetPeriodicTable()
        _, path_coords = self._read_complete_path()
        atom_nums = [
            pt.GetAtomicNumber(atom)
            for atom in self.path_reactant._conformers[0]._atom_symbols
        ]

        reactant_ac, _ = xyz2AC(atom_nums, path_coords[0], charge, use_huckel=huckel)
        product_ac, _ = xyz2AC(atom_nums, path_coords[-1], charge, use_huckel=huckel)

        if np.array_equal(reactant_ac, product_ac):  # Nothing happend - reac = prod
            return True
        else:
            return False

    def _find_xtb_path(
        self,
        chrg=0,
        multiplicity=1,
        temp=300,
        solvent=None,
        huckel=False,
        save_paths=True,
    ):
        """"""
        kpull_list = [-0.02, -0.02, -0.02, -0.02, -0.03, -0.03, -0.04, -0.04]
        alp_list = [0.6, 0.6, 0.3, 0.3, 0.6, 0.6, 0.6, 0.4]

        def run_micro_iter(kpush, kpull, alpha, run_num, reac_direction):
            """
            Only saves the initial search structure if the path search
            is uncussesfull.
            """
            if os.path.isdir(f"{self.reaction_name}"):
                shutil.rmtree(f"{self.reaction_name}")

            os.mkdir(f"micro_run{run_num}")
            os.chdir(f"micro_run{run_num}")

            self._micro_iter_num = 0
            for i in range(3):
                kpush = round(kpush, 4)
                kpull = round(kpull, 4)

                output = self._run_xtb_path(
                    kpush,
                    kpull,
                    alpha,
                    temp,
                    solvent=solvent,
                    chrg=chrg,
                    multiplicity=multiplicity,
                    forward_reaction=reac_direction,
                )

                if self._is_reaction_complete(output):
                    shutil.copytree(
                        f"{self._working_dir}{self._micro_iter_num}",
                        f"../{self.reaction_name}",
                    )
                    print(
                        f">> OK! kpush: {kpush:.4f}, kpull: {kpull:.4f}, alpha: {alpha}, temp: {temp}, forward?: {reac_direction}"
                    )
                    os.chdir("..")
                    if not save_paths:
                        shutil.rmtree(f"micro_run{run_num}")
                    return True
                else:
                    print(
                        f">> fail!: kpush: {kpush:.4f}, kpull: {kpull:.4f}, alpha: {alpha}, temp: {temp}, forward?: {reac_direction}"
                    )

                kpush *= float(1.5)
                kpull *= float(1.5)

                self._micro_iter_num += 1

            shutil.copytree(
                f"{self._working_dir}0", f"../{self.reaction_name}"
            )  # TODO move first terminating search.
            os.chdir("..")
            if not save_paths:
                shutil.rmtree(f"micro_run{run_num}")

            return False

        direction_forward = True
        i = 0
        for param_set_idx, (kpull, alpha) in enumerate(zip(kpull_list, alp_list)):
            if param_set_idx == 0:
                kpush = 0.008
            else:
                kpush = 0.01

            if run_micro_iter(kpush, kpull, alpha, param_set_idx, direction_forward):
                return True

            # Setup new path search with new parameter set, and update stuctures.
            if self._is_reac_prod_identical():
                print("Noting happend. Updating strucures.")
                _, path_coords = self._read_complete_path()
                if direction_forward:
                    self.path_reactant._conformers[0]._update_structure(path_coords[-1])
                else:
                    self.path_product._conformers[0]._update_structure(path_coords[-1])
            else:
                print()
                print("Something happend but RMSD not bellow 0.5 RMSD.")
                print("Most likely not a one step reaction.")
                return "intermediate"  # TODO: Do something different if this it hit.

            i += 1
            if i % 2 == 0:
                direction_forward = True
            else:
                direction_forward = False

        return False

    def _get_single_point_energies(self, coords, solvent=None, chrg=0, multiplicity=1):
        """ Compute single point energies"""

        pt = Chem.GetPeriodicTable()
        atom_nums = np.array(
            [
                pt.GetAtomicNumber(atom)
                for atom in self.path_reactant._conformers[0]._atom_symbols
            ]
        )
        single_point_energies = np.zeros(len(coords))
        for i, coord in enumerate(coords):
            if multiplicity != 1:
                calc = CalculatorAPI(
                    param=get_method("GFN2-xTB"),
                    numbers=atom_nums,
                    positions=coord * 1.8897259886,
                    charge=chrg,
                    uhf=multiplicity - 1,
                )
            else:
                calc = CalculatorAPI(
                    param=get_method("GFN2-xTB"),
                    numbers=atom_nums,
                    positions=coord * 1.8897259886,
                    charge=chrg,
                )
            calc.set_verbosity(VERBOSITY_MUTED)
            if solvent is not None:
                calc.set_solvent(get_solvent(solvent))
            res = calc.singlepoint()
            single_point_energies[i] = res.get_energy()
        return single_point_energies * 627.503

    def _interpolate_ts(self, solvent=None, chrg=0, multiplicity=1, npoints=20):
        """ """
        _, coords = self._read_complete_path()
        energies = self._get_single_point_energies(
            coords, solvent=solvent, chrg=chrg, multiplicity=multiplicity
        )

        max_energy_idx = energies.argmax()

        difference_mat = coords[max_energy_idx - 1] - coords[max_energy_idx + 1]
        pt = Chem.GetPeriodicTable()
        atoms = [
            pt.GetAtomicNumber(atom)
            for atom in self.path_reactant._conformers[0]._atom_symbols
        ]
        interpolated_path_coords = np.zeros((npoints, len(atoms), 3))
        for j in range(npoints + 1):
            interpolated_path_coords[j - 1] = (
                coords[max_energy_idx + 1] + j / npoints * difference_mat
            )

        interpolated_energies = self._get_single_point_energies(
            interpolated_path_coords,
            solvent=solvent,
            chrg=chrg,
            multiplicity=multiplicity,
        )

        return interpolated_energies, interpolated_path_coords

    def _run_barrer_scan(
        self, chrg=0, multiplicity=1, solvent=None, huckel=False, save_paths=False
    ):
        """ """
        return_msg_300 = self._find_xtb_path(
            chrg=chrg,
            multiplicity=multiplicity,
            huckel=huckel,
            solvent=solvent,
            temp=300,
            save_paths=save_paths,
        )

        if return_msg_300 is False:
            print("Didn't find a path. Increasing the temperature.")
            return_msg_6000 = self._find_xtb_path(
                chrg=chrg,
                multiplicity=multiplicity,
                huckel=huckel,
                solvent=solvent,
                temp=6000,
                save_paths=save_paths,
            )
            if return_msg_6000 is False:
                print("No path is found!")
                return float("nan"), None

        elif return_msg_300 == "intermediate":
            return float("nan"), None

        # If we found a path interpolate between structures max-1 and max+1.
        interpolated_energies, interpolated_coords = self._interpolate_ts(
            solvent=solvent, chrg=chrg, multiplicity=multiplicity, npoints=20
        )
        ts_idx = interpolated_energies.argmax()

        return interpolated_energies[ts_idx], interpolated_coords[ts_idx]

    def run_barrier_scan(
        self,
        nruns=3,
        chrg=0,
        multiplicity=1,
        solvent=None,
        huckel=False,
        save_paths=False,
    ):
        """Run barrier scan nruns times, and return the minimum energy and corresponding
        coordinates.
        """
        ts_energy, ts_coords = 9999.9, None
        for _ in range(nruns):
            energy, coords = self._run_barrer_scan(
                nruns=nruns,
                chrg=chrg,
                multiplicity=multiplicity,
                solvent=solvent,
                huckel=huckel,
                save_paths=save_paths,
            )

            if energy < ts_energy:
                ts_energy = ts_energy
                ts_coords = coords

        return ts_energy, ts_coords

    def write_xyz(self, atoms, coords, fname=""):
        """ """
        xyz = f"{len(atoms)}\n \n"
        for i, coord in enumerate(coords):
            xyz += f"{atoms[i]}   {coord[0]}  {coord[1]}   {coord[2]} \n"

        with open(fname, "w") as f:
            f.write(xyz)
