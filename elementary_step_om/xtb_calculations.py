import os
import shutil
import textwrap
import subprocess
import numpy as np

from rdkit import Chem

from .xyz2mol_local import xyz2AC_vdW
from elementary_step_om.io import io_xtb
from elementary_step_om.chem import Reaction


class CalculatorError(Exception):
    """ Exemption for calcualtor errors """


def run_cmd(cmd):
    """ Run cmd to run QM program """
    cmd = cmd.split()
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, err = p.communicate()
    return output.decode("utf-8"), err.decode("utf-8")


def get_xtb_version(xtb_path):
    """ """
    output, _ = run_cmd(f"{xtb_path} --version")
    for line in output.split('\n'):
        if "version" in line.lower():
            return line.split()[2]


class Calculator:
    def __init__(self, nprocs: int = 1, location: str = ".", overwrite: bool = True):

        self._nprocs = nprocs
        self._location = location
        self._overwrite = overwrite
        self._root_dir = os.getcwd()

    def _make_working_directory(
        self, namespace: str = None, overwrite: bool = False
    ) -> str:
        """ Make directory to to run calculations in. """
        working_dir = os.path.join(self._location, namespace)
        if os.path.isdir(working_dir) and self._overwrite:
            shutil.rmtree(working_dir, ignore_errors=True)

        elif os.path.isdir(working_dir):
            raise CalculatorError("Working directory allready exists")

        os.makedirs(working_dir)
        os.chdir(working_dir)

        return working_dir

    def _remove_working_dir(self, namespace: str) -> None:
        """ Removes working directory all and files within """
        working_dir = os.path.join(self._location, namespace)
        os.chdir(self._root_dir)
        shutil.rmtree(working_dir)


class xTBCalculator(Calculator):

    allowd_properties = ["energy", "structure"]

    def __init__(
        self,
        xtb_kwds: str = "--opt loose",
        charge: int = 0,
        spin: int = 1,
        properties: list = ["energy", "structure"],
        nprocs: int = 1,
        overwrite: bool = True,
        location: str = ".",
    ):
        # Check that the property reader is implemeted.
        for property in properties:
            if property not in self.allowd_properties:
                raise CalculatorError("Property not implemented.")

        self._charge = charge
        self._spin = spin
        self._properties = properties
        self._xtb_kwds = xtb_kwds

        super().__init__(nprocs=nprocs, location=location, overwrite=overwrite)

        self._setup_xtb_enviroment()

    def _setup_xtb_enviroment(self) -> None:
        """ """
        os.environ["OMP_NUM_THREADS"] = str(self._nprocs)
        os.environ["MKL_NUM_THREADS"] = str(self._nprocs)

        if "XTB_CMD" not in os.environ:
            raise CalculatorError('No XTB_CMD command. export XTB_CMD="path_to_xtb"')
        self._xTB_CMD = os.environ["XTB_CMD"]  # takes path from XTB_CMD

    def _make_cmd(self, input_filename: str) -> None:
        """ """
        xtb_cmd = f"{self._xTB_CMD} {input_filename} {self._xtb_kwds} --norestart"
        xtb_cmd += f" --chrg {self._charge} --uhf {self._spin - 1}"
        return xtb_cmd

    def _write_input(self, atoms, coords, namespace) -> str:
        """ """
        if len(atoms) != len(coords):
            raise CalculatorError("Length of atoms and coords doesn't match.")

        input_filename = namespace + ".xyz"
        with open(input_filename, "w") as inputfile:
            inputfile.write(f"{len(atoms)} \n \n")
            for atom, coord in zip(atoms, coords):
                inputfile.write(f"{atom}  {coord[0]} {coord[1]} {coord[2]} \n")
        return input_filename

    def _clean(self, working_dir: str) -> None:
        """ """
        self._remove_working_dir(working_dir)

    def __call__(self, atoms, coords, label):
        """ """
        results = {}

        working_dir = self._make_working_directory(namespace=label)
        input_filename = self._write_input(atoms, coords, label)
        cmd = self._make_cmd(input_filename)

        output, errmsg = run_cmd(cmd)

        # Extract properties.
        if "normal termination" in errmsg:
            results["normal_termination"] = True
            for property in self._properties + ["converged"]:
                property_value = io_xtb.read_xtb_out(output, property=property)
                
                # Something wrong with output..
                if property_value is None: 
                    results = {}
                    results["normal_termination"] = False
                    break

                results[property] = property_value
        else:
            results["normal_termination"] = False
        self._clean(working_dir)
        return results


class xTBPathSearch:
    """ """

    def __init__(
        self,
        xtb_kwds: str = "",
        nruns: int = 3,
        overwrite: bool = True,
        nprocs: int = 1,
        memory: int = 2,
        location: str = ".",
    ):

        self._xtb_kwds = xtb_kwds
        self._nruns = nruns

        self._overwrite = overwrite
        self._nprocs = nprocs
        self._memory = memory
        self._location = location
        self._root_dir = os.getcwd()

        self._setup_xtb_enviroment()

    def _setup_xtb_enviroment(self) -> None:
        """ """
        os.environ["OMP_STACKSIZE"] = str(self._memory) + "G"
        os.environ["OMP_NUM_THREADS"] = str(self._nprocs)
        os.environ["MKL_NUM_THREADS"] = str(self._nprocs)

        if "XTB_CMD" not in os.environ:
            raise CalculatorError('No XTB_CMD command. export XTB_CMD="path_to_xtb"')

        self._xTB_CMD = os.environ["XTB_CMD"]
        
        # check xTB version.
        if get_xtb_version(self._xTB_CMD) != "6.1.4":
            raise CalculatorError("Path search can only be used with v. 6.1.x")

    @property
    def reaction(self):
        return self._reaction

    @reaction.setter
    def reaction(self, reaction):
        self._reactant = reaction.reactant
        self._product = reaction.product
        self._charge = reaction.charge
        self._spin = reaction.spin
        self._reaction_label = reaction.reaction_label

    def _write_input(self, kpush, kpull, alpha, temp, forward_reaction=True):
        """
        Write "ractant" and "product" files. If forward_reaction = True the
        calculation is run from reactant -> product, if false it is performed
        from product -> reactant.
        """

        self._reactant_fname = self._reaction_label + "_r.xyz"
        self._product_fname = self._reaction_label + "_p.xyz"

        if forward_reaction:
            self._from_file = self._reactant_fname
            self._to_file = self._product_fname
        else:
            self._from_file = self._product_fname
            self._to_file = self._reactant_fname

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
            kpush, kpull, alpha, temp, self._to_file
        )

        with open("path.inp", "w") as ofile:
            ofile.write(textwrap.dedent(path_file))

        self._reactant.conformers[0].write_xyz(filename=self._reactant_fname)
        self._product.conformers[0].write_xyz(filename=self._product_fname)

    def _make_cmd(self):
        """ """
        cmd = f"{self._xTB_CMD} {self._from_file} --path --input path.inp --gfn2 --norestart "
        cmd += f"--chrg {self._charge} --uhf {self._spin} "
        cmd += self._xtb_kwds
        return cmd

    def _run_xtb_path(self, iter_num, kpush, kpull, alpha, temp, forward_reaction=True):
        """
        Run the xTB path search with the given parameters.
        return the xTB output.
        """
        os.makedirs(f"run{iter_num}")
        os.chdir(f"run{iter_num}")

        self._write_input(kpush, kpull, alpha, temp, forward_reaction=forward_reaction)
        cmd = self._make_cmd()
        output, stderr = run_cmd(cmd)

        os.chdir("..")

        return output

    def _get_single_point_energies(self, path_coords):
        """ Compute single point energies for each point in 'path_coords' """

        xtb_calc = xTBCalculator(
            xtb_kwds="--sp", charge=self._charge, spin=self._spin, properties=["energy"]
        )

        atom_symbols = self._reactant.conformers[0].atom_symbols
        sp_energies = np.empty(path_coords.shape[0])
        for point_idx, path_point_coord in enumerate(path_coords):
            results = xtb_calc(atom_symbols, path_point_coord, f"point{point_idx}")
            sp_energies[point_idx] = results["energy"]

        return sp_energies * 627.503

    def _interpolate_ts(self, path_coords, npoints=20):
        """ """
        natoms = path_coords.shape[1]

        energies = self._get_single_point_energies(path_coords)
        max_energy_idx = energies.argmax()

        difference_mat = (
            path_coords[max_energy_idx - 1] - path_coords[max_energy_idx + 1]
        )
        interpolated_path_coords = np.zeros((npoints, natoms, 3))
        for j in range(npoints + 1):
            interpolated_path_coords[j - 1] = (
                path_coords[max_energy_idx + 1] + j / npoints * difference_mat
            )
        interpolated_energies = self._get_single_point_energies(
            interpolated_path_coords
        )

        ts_idx = interpolated_energies.argmax()
        return interpolated_energies[ts_idx], interpolated_path_coords[ts_idx]

    def _is_reaction_complete(self, output):
        """ Checks that the path have an RMSD below 0.5 AA."""

        # TODO Not super efficient, reads all lines in file.

        output_lines = output.split("\n")
        for line in output_lines:
            if "run 1  barrier" in line:
                try:
                    rmsd = float(line.split()[-1])
                except:
                    rmsd = 9999.9  # rmsd above 0.5 reaction not complete.
                    print(line)
                if rmsd < 0.5:
                    return True
        return False

    def _read_path(self, relative_path="."):
        """ Read coordinates and energies from the path search """
        path_filename = os.path.join(relative_path, "xtbpath_1.xyz")
        with open(path_filename, "r") as path_file:
            xtbpath = path_file.read()

        path_xyz_blocks = xtbpath.split("SCF done")
        natoms = int(path_xyz_blocks[0])

        del path_xyz_blocks[0]

        path_coords = np.zeros((len(path_xyz_blocks), natoms, 3))
        relative_energies = np.zeros(len(path_xyz_blocks))
        for structure_idx, path_strucrure in enumerate(path_xyz_blocks):
            xyz_data = path_strucrure.split("\n")
            relative_energies[structure_idx] = float(xyz_data[0])
            del xyz_data[0]

            coords = np.zeros((natoms, 3))
            for j in range(natoms):
                atom_coord = [coord for coord in xyz_data[j].split()][1:]
                coords[j] = np.array(atom_coord).astype(float)
            path_coords[structure_idx] = coords

        return relative_energies, path_coords

    def _is_reac_prod_identical(self, path_coords):
        """
        This function ensures that if RMSD is above 0.5AA, and something
        happend in the last iteration - it is probably an intermediate.
        """
        pt = Chem.GetPeriodicTable()
        atom_nums = [
            pt.GetAtomicNumber(atom)
            for atom in self._reactant.conformers[0].atom_symbols
        ]

        reactant_ac, _ = xyz2AC_vdW(atom_nums, path_coords[0])
        product_ac, _ = xyz2AC_vdW(atom_nums, path_coords[-1])

        if np.array_equal(reactant_ac, product_ac):  # Nothing happend - reac = prod
            return True
        return False

    def _find_xtb_path(self, temp=300):
        """"""
        kpull_list = [-0.02, -0.02, -0.02, -0.02, -0.03, -0.03, -0.04, -0.04]
        alp_list = [0.6, 0.6, 0.3, 0.3, 0.6, 0.6, 0.6, 0.4]

        def run_param_set(kpush, kpull, alpha, run_num, reac_direction):
            """"""
            os.makedirs(f"param_set{run_num}")
            os.chdir(f"param_set{run_num}")

            for iter_num in range(3):
                kpush = round(kpush, 4)
                kpull = round(kpull, 4)

                output = self._run_xtb_path(
                    iter_num, kpush, kpull, alpha, temp, forward_reaction=reac_direction
                )

                if self._is_reaction_complete(output):
                    print(f">> Found a xTB path going forward: {reac_direction}")
                    print(f"    * kpush: {kpush:.4f}")
                    print(f"    * kpull: {kpull:.4f}")
                    print(f"    * alpha: {alpha}")
                    print(f"    * temp: {temp}")

                    # Read output path, and energies.
                    final_energies, final_path_coords = self._read_path(
                        relative_path=f"run{iter_num}"
                    )

                    return True, final_energies, final_path_coords

                kpush *= float(1.5)
                kpull *= float(1.5)

            # if nothing happens, return the path for the first micro run.
            energies, path_coords = self._read_path(relative_path="run0")
            os.chdir("..")
            return False, energies, path_coords

        direction_forward = True
        i = 0
        for param_set_idx, (kpull, alpha) in enumerate(zip(kpull_list, alp_list)):
            if param_set_idx == 0:
                kpush = 0.008
            else:
                kpush = 0.01
            found_path, energies, coords = run_param_set(
                kpush, kpull, alpha, param_set_idx, direction_forward
            )

            if found_path:
                return True, energies, coords

            # if the search was unsucessfull, check if the reactant and product changed.
            # If they didn't update reactant and product structures.
            if self._is_reac_prod_identical(coords):
                if direction_forward:
                    self._reactant.conformers[0].coordinates = coords[-1]
                else:
                    self._product.conformers[0].coordinates = coords[-1]
            else:
                print()
                text = (
                    ">> Something happend but RMSD not bellow 0.5 RMSD. "
                    "Most likely not a one step reaction."
                )
                print(text)
                return "found intermediate", None, None

            # Change direction
            i += 1
            if i % 2 == 0:
                direction_forward = True
            else:
                direction_forward = False

        return "increase temp", None, None

    def _run_barrer_scan(self):
        """ """
        return_msg_temp300, _, path_coords = self._find_xtb_path(temp=300)
        if return_msg_temp300 is True:
            ts_energy, ts_coords = self._interpolate_ts(path_coords, npoints=20) 
            return ts_energy, ts_coords
        
        elif return_msg_temp300 == "found intermediate":
            return float("nan"), None 

        elif return_msg_temp300 == "increase temp":
            print("Didn't find a path. Increasing the temperature to 6000 K.")
            os.makedirs('tmp6000')
            os.chdir('tmp6000')
            return_msg_temp6000, _, path_coords = self._find_xtb_path(temp=6000)
            print("return code: ", return_msg_temp6000)
            if return_msg_temp6000 is True:
                ts_energy, ts_coords = self._interpolate_ts(path_coords, npoints=20) 
                return ts_energy, ts_coords
            else:
                return float("nan"), None

    def _make_root_working_directory(self):
        """ Make the directory the path-search is working in """

        self._root_workind_dir = os.path.join(self._location, self._reaction_label)
        if os.path.isdir(self._root_workind_dir) and self._overwrite:
            shutil.rmtree(self._root_workind_dir)
        elif os.path.isdir(self._root_workind_dir) and not self._overwrite:
            raise CalculatorError("Root directory exists.")

        os.makedirs(self._root_workind_dir)
        os.chdir(self._root_workind_dir)

    def __call__(self, reaction: Reaction):
        """ """
        self.reaction = reaction

        self._make_root_working_directory()

        ts_energy, ts_coords = 99999.9, None
        for i in range(self._nruns):
            print(os.getcwd())
            os.makedirs(f"pathrun{i}")
            os.chdir(f"pathrun{i}")
            energy, coords = self._run_barrer_scan()
            if energy < ts_energy:
                ts_energy = energy
                ts_coords = coords

            os.chdir(self._root_dir)
            os.chdir(self._root_workind_dir)

        if ts_energy == 99999.9:
            ts_energy = np.float("nan")

        os.chdir(self._root_dir)
        return ts_energy, ts_coords
