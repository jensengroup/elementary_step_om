import copy
import itertools
import multiprocessing as mp
import numpy as np

from rdkit.Geometry import Point3D
from rdkit import Chem
from rdkit.Chem import rdmolfiles, AllChem, rdmolops
from rdkit.Chem import rdChemReactions
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions

import hashlib


def reassign_atom_idx(mol):
    """ Reassigns the RDKit mol object atomid to atom mapped id """
    renumber = [(atom.GetIdx(), atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    new_idx = [idx[0] for idx in sorted(renumber, key=lambda x: x[1])]
    mol = Chem.RenumberAtoms(mol, new_idx)
    rdmolops.AssignStereochemistry(mol, force=True)
    return mol


def coords_to_AC(symbols, coords, covalent_factor: float = 1.3):
    """ """
    pt = Chem.GetPeriodicTable()
    new_coords = np.asarray(coords, dtype=np.float32)
    num_atoms = len(symbols)

    new_ac = np.zeros((num_atoms, num_atoms), dtype=np.int8)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i > j:
                atom_num_i = pt.GetAtomicNumber(symbols[i])
                atom_num_j = pt.GetAtomicNumber(symbols[j])

                Rcov_i = pt.GetRcovalent(atom_num_i) * covalent_factor
                Rcov_j = pt.GetRcovalent(atom_num_j) * covalent_factor

                dist = np.linalg.norm(new_coords[i] - new_coords[j])
                if dist < Rcov_i + Rcov_j:
                    new_ac[i, j] = new_ac[j, i] = 1
    return new_ac


class MoleculeException(Exception):
    """ An exception that is raised by the Molecule class """


class ReactionException(Exception):
    """ An exception that is raised by the Reaction class """


class BaseMolecule:
    """
    A base class for molecules. Encapsulates an RDKit mol object.

    The object is hashable by the the SMILES (should be changed to the graph hash)
    and is therefore comparable thorugh the equality operator and in sets.
    """

    def __init__(self, molblock: str = None):
        if molblock is None:
            raise MoleculeException("Need to provide a MolBlock")

        self.rd_mol = None
        self.conformers = []
        self._calculator = None

        self.results = {}
        self._mol_hash = None

        self._inititalize_molecule(molblock)

    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label})"

    def __eq__(self, other):
        if not type(self) is type(other):
            raise TypeError(
                f"Comparing {self.__class__.__name__} to {other.__class__.__name__}"
            )
        return self.__hash__() == other.__hash__()

    def __hash__(self) -> int:
        if self._mol_hash is None:
            m = hashlib.blake2b()
            m.update(Chem.MolToSmiles(self.rd_mol).encode("utf-8"))
            self._mol_hash = int(str(int(m.hexdigest(), 16))[:32])
        return self._mol_hash

    def _inititalize_molecule(self, molblock):
        """ """
        self.rd_mol = Chem.MolFromMolBlock(molblock, sanitize=False)
        Chem.SanitizeMol(self.rd_mol)

        # Does it have a 3D conformer?
        if all(self.rd_mol.GetConformer().GetPositions()[:, 2] == 0.0):
            self.rd_mol.RemoveAllConformers()
        else:
            self.conformers.append(Conformer(molblock=molblock))
            self.rd_mol.RemoveAllConformers()
            AllChem.Compute2DCoords(self.rd_mol)

    @property
    def molblock(self) -> str:
        """ molblock are the MolBlock from self.rd_mol"""
        return Chem.MolToMolBlock(self.rd_mol)

    @property
    def label(self) -> str:
        return self.__class__.__name__ + "_" + str(self.__hash__())

    @property
    def atom_symbols(self) -> list:
        return [atom.GetSymbol() for atom in self.rd_mol.GetAtoms()]

    @property
    def ac_matrix(self):
        return rdmolops.GetAdjacencyMatrix(self.rd_mol)

    @property
    def calculator(self):
        return self._calculator

    @calculator.setter
    def calculator(self, calc_instance):
        for conf in self.conformers:
            conf.calculator = calc_instance

    @classmethod
    def from_molfile(cls, file):
        """
        Initialize Molecule from a Molfile
        """
        if not file.endswith(("mol", "sdf")):
            raise TypeError("Only works with mol/sdf files")

        suppl = rdmolfiles.SDMolSupplier(file, removeHs=False, sanitize=False)
        first_mol = next(suppl)
        obj = cls(molblock=Chem.MolToMolBlock(first_mol))
        return obj

        # AllChem.Compute2DCoords(first_mol)
        # obj.molecule = first_mol
        # for conf_idx in range(len(suppl)):
        #     obj._conformers.append(
        #         Conformer(sdf=suppl.GetItemText(conf_idx),
        #                   label=f"{file.split('.')[0]}-{conf_idx}")
        #     )
        # obj._2d_structure = False
        # return obj

    @classmethod
    def from_rdkit_mol(cls, rdkit_mol):
        """ Initialize Molecule from a RDKit mol object. """

        n_confs = rdkit_mol.GetConformers()
        if len(n_confs) <= 1:  # If no 3D conformers.
            return cls(molblock=Chem.MolToMolBlock(rdkit_mol))
        else:
            raise NotImplementedError("RDKit have more than 1 Conformer.")

    def _remove_pseudo_chirality(self) -> None:
        """ Reassign stereochemistry for the RDKit mol object """
        rdmolops.AssignStereochemistry(
            self.rd_mol, cleanIt=True, flagPossibleStereoCenters=True, force=True
        )

    def _remove_atom_mapping(self) -> None:
        """ Remove atom mapping from RDKit mol object """
        [atom.SetAtomMapNum(0) for atom in self.rd_mol.GetAtoms()]

    def has_atom_mapping(self) -> None:
        """ Determines is the molecule has atom mappings """
        for atom in self.rd_mol.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                return True
        return False

    def get_fragments(self):
        """ Split molecule into fragments """
        fragments = []
        for frag in Chem.GetMolFrags(self.rd_mol, asMols=True, sanitizeFrags=False):
            fragments.append(Fragment.from_rdkit_mol(frag))
        return fragments

    def num_rotatable_bond(self):
        """ Calculates the number of rotatable bonds in the fragment """
        if self.rd_mol is None:
            raise RuntimeError("Fragment has no RDKit mol")

        rot_bonds_smarts = [
            "[!#1]~[!$(*#*)&!D1]-!@[!$(*#*)&!D1]~[!#1]",
            "[*]~[*]-[O,S]-[#1]",
            "[*]~[*]-[NX3;H2]-[#1]",
        ]

        num_dihedral = 0
        for bond_smart in rot_bonds_smarts:
            dihedral_template = Chem.MolFromSmarts(bond_smart)
            dihedrals = self.rd_mol.GetSubstructMatches(dihedral_template)
            num_dihedral += len(dihedrals)

        return num_dihedral

    def run_calculations(self, parallel_confs: int = 1) -> None:
        """
        Run the calculation defined by the calculator object on
        all conformers.
        """
        with mp.Pool(int(parallel_confs)) as pool:
            updated_confs = pool.map(self._calculation_worker, self.conformers)
        self.conformers = updated_confs

    def _calculation_worker(self, conf):
        conf.run_calculation()
        return conf

    def embed_molecule(
        self,
        confs_pr_frag: int = 1,
        seed: int = 42,
        refine_calculator=None,
        overwrite: bool = True,
        direction: list = [0.8, 0, 0],
    ) -> None:
        """
        If more than one fragment, the fragments are embedded individually
        and then merged afterwards.
        """

        def merge_fragments(frag_confs, conf_num: int) -> Conformer:
            """ """
            nfrags = len(frag_confs)
            if nfrags == 0:
                merged_conformer = frag_confs[0]
            else:
                merged_conformer = Chem.MolFromMolBlock(
                    frag_confs[0].molblock, sanitize=False
                )
                for frag_conf in frag_confs[1:]:
                    frag_natoms = len(frag_conf.atom_symbols)
                    new_coords = (
                        frag_conf.coordinates
                        + np.array(direction) * nfrags * frag_natoms
                    )
                    frag_conf.coordinates = new_coords

                    frag_conf = Chem.MolFromMolBlock(frag_conf.molblock, sanitize=False)
                    merged_conformer = Chem.CombineMols(merged_conformer, frag_conf)

            merged_conformer = reassign_atom_idx(merged_conformer)
            merged_conformer = Conformer(molblock=Chem.MolToMolBlock(merged_conformer))

            if refine_calculator is not None:
                merged_conformer.calculator = refine_calculator
                merged_conformer.run_calculation()

            return merged_conformer

        if overwrite:
            self.conformers = []

        fragments = self.get_fragments()
        fragment_confs = []
        for frag in fragments:
            frag.make_fragment_conformers(nconfs=confs_pr_frag, seed=seed)
            fragment_confs.append(frag.conformers)

        for conf_num, frag_confs_set in enumerate(itertools.product(*fragment_confs)):
            self.conformers.append(merge_fragments(frag_confs_set, conf_num))


class Molecule(BaseMolecule):
    """ """

    def __init__(self, molblock: str = None) -> None:
        """ """
        super().__init__(molblock=molblock)

        # Is the graph mapped?
        if self.has_atom_mapping():
            self.unmap_molecule()

    def unmap_molecule(self):
        """ Remove both atom mapping and reassign the sterochemistry."""
        self._remove_atom_mapping()
        self._remove_pseudo_chirality()

    def get_mapped_molecule(self):
        """
        Assign atom mapping as atom idx + 1 and assign random pseudochirality
        returns a MappedMolecule.
        """
        tmp_rdmol = copy.deepcopy(self.rd_mol)
        [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in tmp_rdmol.GetAtoms()]
        rdmolops.AssignStereochemistry(
            tmp_rdmol, cleanIt=True, flagPossibleStereoCenters=True, force=True
        )

        opts = StereoEnumerationOptions(onlyUnassigned=False, unique=False)
        tmp_rdmol = next(EnumerateStereoisomers(tmp_rdmol, options=opts))
        tmp_rdmol = Chem.MolFromMolBlock(Chem.MolToMolBlock(tmp_rdmol), sanitize=False)

        return MappedMolecule(molblock=Chem.MolToMolBlock(tmp_rdmol))


class MappedMolecule(BaseMolecule):
    """ """

    def __init__(self, molblock=None):
        super().__init__(molblock=molblock)

        # Is the graph mapped?
        if self.has_atom_mapping():
            self.rd_mol = reassign_atom_idx(self.rd_mol)
        else:
            raise MoleculeException("Atoms in MolBlock are not mapped")

    def get_unmapped_molecule(self) -> Molecule:
        """
        Remove atom mapping and reassign the sterochemistry.
        Return Unmapped Molecule
        """
        tmp_mol = copy.deepcopy(self)
        tmp_mol._remove_atom_mapping()
        tmp_mol._remove_pseudo_chirality()

        return Molecule(molblock=Chem.MolToMolBlock(tmp_mol.rd_mol))


class Solvent:
    """
    Base class that represents a solvent.
    """

    def __init__(self, smiles=None, n_solvent=0, active=0):
        """ """
        self._smiles = smiles
        self._n_solvent_molecules = n_solvent
        self._nactive = active


class Fragment(BaseMolecule):
    """
    The Fragment class is a special `Molecule` with only one Fragment.
    Doesn't alter the atom mapping.
    """

    # TODO: This doesn't perform an UFF minimization. This is a problem when dealing
    # with organometalics.
    def _embed_fragment(self, frag_rdkit, nconfs=20, seed=20):
        """ """
        p = AllChem.ETKDGv3()
        p.useRandomCoords = True
        p.randomSeed = int(seed)

        # Always assign stereochemistry when embedding.
        rdmolops.AssignStereochemistry(frag_rdkit, force=False)
        try:
            AllChem.EmbedMultipleConfs(frag_rdkit, numConfs=nconfs, params=p)
        except RuntimeError:
            print(f"RDKit Failed to embed: {self.label}.")
            return []

        # # Worst code ever...
        # smarts = Chem.MolToSmarts(frag_rdkit)
        # for patt in ["\d", ""]:
        #     smarts = re.sub(f":\d{patt}", '', smarts)
        # smarts_ob = pybel.Smarts(smarts)
        # ##
        for conf_idx in range(frag_rdkit.GetNumConformers()):
            conf_molblock = Chem.MolToMolBlock(frag_rdkit, confId=conf_idx)
            conf = Conformer(molblock=conf_molblock)

            # coords = np.zeros((frag_rdkit.GetNumAtoms(), 3))
            # obmol = pybel.readstring('mdl', conf.molblock)
            # obmol.localopt(forcefield='uff', steps=uffSteps) # This makes it not work
            # smarts_ob.findall(obmol) # how to i get the atom mapping to work correct??
            # for i, atom in enumerate(obmol):
            #    coords[i] = atom.coords

            self.conformers.append(conf)

    @staticmethod  # Does it need to be static??
    def _dative2covalent(inp_mol):
        """
        This doesn't change the atom order
        can be speed op by only looping over dative bonds.
        """
        # TODO this i wrong for Metals. See Jan messeage on Slack!
        mol = copy.deepcopy(inp_mol)
        for bond in mol.GetBonds():
            if bond.GetBondType() is Chem.rdchem.BondType.DATIVE:
                beginAtom = bond.GetBeginAtom().SetFormalCharge(0)
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        mol.UpdatePropertyCache(strict=False)
        rdmolops.AssignStereochemistry(mol, force=True)
        return mol

    def make_fragment_conformers(
        self,
        nconfs: int = 20,
        seed: int = 42,
        overwrite: bool = True,
    ) -> None:
        """
        Makes conformers by embedding conformers using RDKit and
        refine with openbabel UFF. If molecule is more then one fragment
        they are moved X*n_atoms awat from each other.

        openbabel dependency might be eliminated at a later stage.

        overwrite = True - remove old conformers
        """
        if overwrite:
            self.conformers = []

        rdkit_mol = self._dative2covalent(self.rd_mol)
        self._embed_fragment(rdkit_mol, nconfs=nconfs, seed=seed)


class Conformer:
    """ """

    def __init__(self, molblock=None):
        """ """
        self._molblock = molblock
        self.results = None
        self._calculator = None

        self._set_atom_symbols()
        self._set_init_connectivity()

    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label})"

    @property
    def molblock(self) -> str:
        return self._molblock

    @property
    def label(self) -> str:
        return self.__class__.__name__ + str(hash(self._molblock))

    @property
    def coordinates(self):
        natoms = len(self.atom_symbols)
        info = self._molblock.split("\n")[4 : 4 + natoms]
        coords = np.array([coord.split()[:3] for coord in info], dtype=float)
        return coords

    @coordinates.setter
    def coordinates(self, new_coords):
        self._update_molblock_coords(new_coords)

    @property
    def calculator(self):
        return self._calculator

    @calculator.setter
    def calculator(self, calc_instance):
        """ """
        calc = copy.deepcopy(calc_instance)
        self._calculator = calc

    def _set_init_connectivity(self):
        """ """
        # for organometalics consider removing metal.
        # perhaps move to fragment.

        sdf = self._molblock.split("\n")
        del sdf[:3]  # del header

        info_line = sdf[0].strip().split()
        natoms = int(info_line[0])
        nbonds = int(info_line[1])
        del sdf[: natoms + 1]  # del coord block

        connectivity = np.zeros((natoms, natoms), dtype=np.int8)
        for line_idx in range(nbonds):
            i, j = sdf[line_idx].split()[:2]
            connectivity[int(i) - 1, int(j) - 1] = 1
            connectivity[int(j) - 1, int(i) - 1] = 1

        self._init_connectivity = connectivity

    def _set_atom_symbols(self):
        """ """
        # for organometalics consider removing metal.
        # perhaps move to fragment.

        sdf = self._molblock.split("\n")
        del sdf[:3]  # del header

        info_line = sdf[0].strip().split()
        natoms = int(info_line[0])
        del sdf[0]

        atom_symbols = []
        for line_idx in range(natoms):
            atom_symbols.append(sdf[line_idx].split()[3])

        self.atom_symbols = atom_symbols

    def _check_connectivity(self, new_coords, covalent_factor=1.3):
        """ check that the updated structure is ok. """
        new_ac = coords_to_AC(
            self.atom_symbols, new_coords, covalent_factor=covalent_factor
        )
        if np.array_equal(new_ac, self._init_connectivity):
            return True
        return False

    def _update_molblock_coords(self, new_coords):
        """ """
        tmp_mol = Chem.MolFromMolBlock(self.molblock, sanitize=False)
        conf = tmp_mol.GetConformer()
        for i in range(tmp_mol.GetNumAtoms()):
            x, y, z = new_coords[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))

        self._molblock = Chem.MolToMolBlock(tmp_mol)

    def run_calculation(self, covalent_factor: float = 1.3) -> None:
        """
        Run single calculation on the conformer.
        """
        calc_results = self.calculator(self.atom_symbols, self.coordinates, self.label)

        # Check that calculation succeded.
        if calc_results.pop("normal_termination") and calc_results.pop("converged"):
            self.results = calc_results
            self.results["converged"] = True

            # update structure - if the connectivity is ok.
            if "structure" in self.results:
                new_coords = calc_results.pop("structure")
                if self._check_connectivity(new_coords, covalent_factor):
                    self._update_molblock_coords(new_coords)

                # This is super ugly, you can do better!! :)
                else:
                    self.results = {"converged": False}
        else:
            self.results = {"converged": False}

    def write_xyz(self, filename=None):
        """  """
        xyz_string = ""
        structure_block = self._molblock.split("V2000")[1]
        natoms = 0
        for line in structure_block.strip().split("\n"):
            line = line.split()
            if len(line) < 5:
                break
            coords = line[:3]
            symbol = line[3]

            xyz_string += f"{symbol} {' '.join(coords)}\n"
            natoms += 1
        xyz_string = f"{natoms} \n \n" + xyz_string

        if filename is not None:
            with open(filename, "w") as fout:
                fout.write(xyz_string)
        else:
            return xyz_string


class Reaction:
    """
    Contains atom-mapped reactant and product.
    """

    def __init__(
        self,
        reactant: MappedMolecule,
        product: MappedMolecule,
        charge: int = 0,
        spin: int = 1,
    ):

        if not isinstance(reactant, MappedMolecule) and isinstance(
            product, MappedMolecule
        ):
            raise ReactionException("reactant and product has to be MappedMolecules!")

        self.reactant = reactant
        self.product = product
        self.charge = charge
        self.spin = spin

        self._path_search_calculator = None

        self._ts_path_energies = []
        self._ts_path_coordinates = []
        self._ts_energies = []
        self._ts_coordinates = []
        self.ts_check = None

        self._reaction_hash = None

    def __eq__(self, other):
        """  """
        if not type(self) is type(other):
            raise TypeError(f"Comparing {self.__class__} to {other.__class__}")
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        if self._reaction_hash is None:
            m = hashlib.blake2b()
            m.update(Chem.MolToSmiles(self.reactant.rd_mol).encode("utf-8"))
            m.update(Chem.MolToSmiles(self.product.rd_mol).encode("utf-8"))
            self._reaction_hash = int(str(int(m.hexdigest(), 16))[:32])
        return self._reaction_hash
    
    @property
    def reaction_label(self) -> str:
        return self.__class__.__name__ + "_" + str(self.__hash__())

    @property
    def path_search_calculator(self):
        return self._path_search_calculator

    @path_search_calculator.setter
    def path_search_calculator(self, calc):
        self._path_search_calculator = copy.deepcopy(calc)

    @property
    def ts_guess_energies(self):
        """
        """
        ts_guess_energies = []
        for energies in self._ts_path_energies:
            if energies is None:
                continue
            ts_guess_energies.append(energies.max())
        return np.array(ts_guess_energies)

    @property
    def ts_guess_coordinates(self):
        """
        """
        ts_guess_coordinates = []
        for energies, coords in zip(self._ts_path_energies, self._ts_path_coordinates):
            if energies is None:
                continue
            ts_guess_coordinates.append(coords[energies.argmax()])
        return np.asarray(ts_guess_coordinates)

    @property
    def rd_reaction(self):
        """ """
        rd_reaction = rdChemReactions.ChemicalReaction()
        rd_reaction.AddReactantTemplate(self.reactant.rd_mol)
        rd_reaction.AddProductTemplate(self.product.rd_mol)
        return rd_reaction

    def run_path_search(self, seed=42):  # TODO make it possible to take list og coords and energies
        """
        """
        if self._path_search_calculator is None:
            raise ReactionException("Set the path search calculator")
        self._ts_path_energies, self._ts_path_coordinates = self.path_search_calculator(self, seed=seed)

    def _run_ts_search(self, ts_calculator=None):
        """ Run a transition state search using Gaussian. """

        if len(self.ts_guess_coordinates) == 0:
            raise ReactionException("Run a path search before TS search.")

        if ts_calculator is None:
            raise RuntimeError("Needs a G16 calculator!")

        if "structure" not in ts_calculator._properties:
            print('added "structure" to properties')
            ts_calculator._properties += ["structure"]
        if "frequencies" not in ts_calculator._properties:
            print('added "frequencies" to properties')
            ts_calculator._properties += ["frequencies"]
        
        for coords in self.ts_guess_coordinates:
            ts_results = ts_calculator(
                self.reactant.atom_symbols, coords, label=self.reaction_label
            )
            if ts_results["converged"] is True:
                img_frequencis = [freq for freq in ts_results["frequencies"] if freq < 0.0]
                if len(img_frequencis) == 1:
                    self._ts_coordinates.append(ts_results['structure'])
                    self._ts_energies.append(ts_results['energy'])
                else:
                    print("not one img. frequency.")
        
        self._ts_coordinates = np.array(self._ts_coordinates)
        self._ts_energies = np.array(self._ts_energies)

    def _run_irc(self, irc_calculator = None, refine_calculator = None):
        """
        You do not need to add forward/reverse. This is done automatically.
        TODO: perform reverse and forward in parallel.
        """
        if "structure" in irc_calculator._properties:
            print('replacing "structure" in properties with "irc_structure"')
            structure_idx = irc_calculator._properties.index("structure")
            irc_calculator._properties += ["irc_structure"]
            del irc_calculator._properties[structure_idx]

        all_irc_results = []
        for ts_coords in self._ts_coordinates:
            irc_results = dict()    
            for rev_or_fw in ["reverse", "forward"]:
                tmp_kwds = [kwd.strip() for kwd in irc_calculator._kwds.split(",")]
                if "reverse" in tmp_kwds:
                    del tmp_kwds[tmp_kwds.index("reverse")]
                elif "forward" in tmp_kwds:
                    del tmp_kwds[tmp_kwds.index("forward")]
                tmp_kwds.insert(-1, rev_or_fw)
                irc_calculator._kwds = ", ".join(tmp_kwds)

                results = irc_calculator(
                    self.reactant.atom_symbols, ts_coords, label=f"{self.reaction_label}_{rev_or_fw}"
                )
                results = dict([
                    itm for itm in results.items() if itm[0] in ['converged', 'irc_structure']
                ])
                irc_results[rev_or_fw] = results
        
            all_irc_results.append(irc_results)

        # Refine IRC endpoint with refine calculator.
        if refine_calculator is not None:
            # Loop over each IRC
            for irc_result in all_irc_results: # Loop over each IRC
                for rev_or_fwd, results in irc_result.items():

                    if not results['converged']:
                        continue
                     
                    refine_results = refine_calculator(
                        self.reactant.atom_symbols,
                        results["irc_structure"],
                        label=f"{self.reaction_label}_refine_irc_{rev_or_fwd}",
                    )
                    results['irc_structure'] = refine_results['structure']

        return all_irc_results

    def irc_check_ts(self, ts_calculator, irc_calculator, refine_calculator):
        """ """
        # Run TS optimization
        self._run_ts_search(ts_calculator)

        # Run IRC
        irc_endpoints_results = self._run_irc(irc_calculator, refine_calculator)
        ts_ok = []
        for irc_enpoints in irc_endpoints_results:
            found_ends = {"reactant": False, "product": False}
            for results in irc_enpoints.values():
                endpint_ac = coords_to_AC(self.reactant.atom_symbols, results["irc_structure"])
                found_reactant = np.array_equal(self.reactant.ac_matrix, endpint_ac)
                found_product = np.array_equal(self.product.ac_matrix, endpint_ac)
                if found_reactant:
                    found_ends["reactant"] = True
                elif found_product:
                    found_ends["product"] = True
            ts_ok.append(found_ends)
        
        self.ts_check = ts_ok

    def write_ts(self):
        """ Write xyz file for all TS's """

        symbols = [atom.GetSymbol() for atom in self.reactant.rd_mol.GetAtoms()]
        for i, (energy, coords) in enumerate(zip(self._ts_energies, self._ts_coordinates)):
            name = f"{self.reaction_label}_ts{i}"
            xyz = f"{len(symbols)}\n {name}: {energy:.5f} Hartree \n"
            for symbol, coord in zip(symbols, coords):
                xyz += f"{symbol}  " + " ".join(map(str, coord)) + "\n"
                
            with open(name + ".xyz", 'w') as xyzfile:
                xyzfile.write(xyz)

    def write_ts_guess(self):
        """ """
        symbols = [atom.GetSymbol() for atom in self.reactant.rd_mol.GetAtoms()]
        for i, (energy, coords) in enumerate(zip(self.ts_guess_energies, self.ts_guess_coordinates)):
            name = f"{self.reaction_label}_tsguess{i}"
            xyz = f"{len(symbols)}\n {name}: {energy:.5f} Hartree \n"
            for symbol, coord in zip(symbols, coords):
                xyz += f"{symbol}  " + " ".join(map(str, coord)) + "\n"
                
            with open(name + ".xyz", 'w') as xyzfile:
                xyzfile.write(xyz)
