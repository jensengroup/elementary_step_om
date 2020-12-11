import copy
import multiprocessing as mp
import numpy as np

from rdkit.Geometry import Point3D
from rdkit import Chem
from rdkit.Chem import rdmolfiles, AllChem, rdchem, rdmolops
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions

from openbabel import pybel

from .elementary_step import valid_products


def sdf2xyz(sdftxt):
    """  """
    xyz_string = ''
    structure_block = sdftxt.split('V2000')[1]
    natoms = 0
    for line in structure_block.strip().split('\n'):
        line = line.split()
        if len(line) < 5:
            break

        coords = line[:3] 
        symbol = line[3]

        xyz_string += f"{symbol} {' '.join(coords)}\n"
        natoms += 1

    xyz_string = f"{natoms} \n \n" + xyz_string
    return xyz_string


class Molecule():
    """ """
    def __init__(self, moltxt=None, label=None, removeHs=False,
                 sanitize=False):

        self.label = label

        self.molecule = None   # This is a 2D graph
        self._conformers = []
        self._embed_ok = None

        # Helper varibels
        self._2d_structure = None
        self.removeHs = removeHs
        self.sanitize = sanitize

        if moltxt is not None:
            self._read_moltxt(moltxt)

    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label})"

    def __eq__(self, other):
        """  """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Comparing {self.__class__} to {other.__class__}")

        if (self.molecule.HasSubstructMatch(other.molecule, useChirality=True) and
            other.molecule.HasSubstructMatch(self.molecule, useChirality=True)):
            return True
        return False

    def set_calculator(self, calculator):
        """ """
        for conf in self._conformers:
            conf.set_calculator(calculator)
        self.calculator = calculator

    def _read_moltxt(self, moltxt):
        """ """
        if not all(x in moltxt for x in ['$$$$', 'V2000']):
            raise TypeError('not .sdf/.mol text')
        elif 'V3000' in moltxt:
            NotImplementedError('.sdf V3000 not implemented as txt yet.')

        self._conformers = [f'{x}$$$$' for x in moltxt.split('$$$$')][:-1]

        first_mol = Chem.MolFromMolBlock(self._conformers[0],
                                         sanitize=self.sanitize,
                                         removeHs=self.removeHs)

        if not first_mol.GetConformer().Is3D():  # if 1D of 2D
            self.molecule = first_mol
            self._2d_structure = True
        else:  # if 3D create 2D coords
            AllChem.Compute2DCoords(first_mol)
            self.molecule = first_mol

        if self.label is None:
            self.label = self._conformers[0].split('\n')[0]

    @classmethod
    def from_molfile(cls, file, removeHs=False, sanitize=False):
        """
        Reads Molfile
        """
        if not file.endswith(('mol', 'sdf')):
            raise TypeError('Only works with mol/sdf files')

        obj = cls(moltxt=None, label=file.split('.')[0], removeHs=removeHs,
                  sanitize=False)

        suppl = rdmolfiles.SDMolSupplier(file,
                                         removeHs=removeHs,
                                         sanitize=sanitize)

        first_mol = next(suppl)
        if not first_mol.GetConformer().Is3D():  # if 1D or 2D
            obj.molecule = first_mol
            obj._2d_structure = True
            return obj

        AllChem.Compute2DCoords(first_mol)
        obj.molecule = first_mol
        for conf_idx in range(len(suppl)):
            obj._conformers.append(suppl.GetItemText(conf_idx))
        obj._2d_structure = False
        return obj

    @classmethod
    def from_rdkit_mol(cls, rdkit_mol, removeHs=False, sanitize=False):
        """ """
        try:
            label = rdkit_mol.GetProp('_Name')
        except KeyError:
            label = None

        obj = cls(label=label, removeHs=removeHs, sanitize=sanitize)
        obj.molecule = rdkit_mol

        if len(rdkit_mol.GetConformers()) == 0:  # If no 3D conformers.
            return obj

        # If rdkit mol has 3D confs, add confs to obj.
        if rdkit_mol.GetConformer().Is3D():
            for conf in rdkit_mol.GetConformers():
                mol_block = Chem.MolToMolBlock(rdkit_mol, condId=conf.Getid())
                obj._conformers.append(mol_block)
        return obj

    @staticmethod
    def _dative2covalent(inp_mol):
        """ This doesn't change the atom order
        can be speed op by only looping over dative bonds.
        """
        # TODO this i wrong for Metals. See Jan messeage on Slack!
        mol = copy.deepcopy(inp_mol)
        for bond in mol.GetBonds():
            if bond.GetBondType() is Chem.rdchem.BondType.DATIVE:
                beginAtom = bond.GetBeginAtom().SetFormalCharge(0)
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        mol.UpdatePropertyCache(strict=False)
        return mol


class Conformer:
    """ """
    def __init__(self, sdf=None, label=None, calculator=None):
        """ """
        self.structure = sdf
        self.label = label
        self._converged = None
        self._set_atom_symbols()
        self._set_init_connectivity() # makes connectivty matrix from sdf

        self.set_calculator(calculator)
        self.results = dict()

    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label})"

    def set_calculator(self, calc):
        if calc is not None:
            calc = copy.deepcopy(calc)
            calc.add_structure(self)
            self.__calc = calc
        else:
            self.__calc = None

    def get_calculator(self):
        return self.__calc

    def _set_init_connectivity(self):
        """ """
        # for organometalics consider removing metal.
        # perhaps move to fragment.

        sdf = self.structure.split('\n')
        del sdf[:3]  # del header

        info_line = sdf[0].strip().split()
        natoms = int(info_line[0])
        nbonds = int(info_line[1])
        del sdf[:natoms + 1]  # del coord block

        connectivity = np.zeros((natoms, natoms), dtype=np.int8)
        for line_idx in range(nbonds):
            i, j = sdf[line_idx].split()[:2]
            connectivity[int(i) - 1, int(j)-1] = 1
            connectivity[int(j) - 1, int(i)-1] = 1

        self._init_connectivity = connectivity

    def _set_atom_symbols(self):
        """ """
        # for organometalics consider removing metal.
        # perhaps move to fragment.

        sdf = self.structure.split('\n')
        del sdf[:3]  # del header

        info_line = sdf[0].strip().split()
        natoms = int(info_line[0])
        del sdf[0]

        atom_symbols = []
        for line_idx in range(natoms):
            atom_symbols.append(sdf[line_idx].split()[3])

        self._atom_symbols = atom_symbols

    def _check_structure(self, new_coords, covalent_factor=1.3):
        """ check that the updated structure is ok"""

        pt = Chem.GetPeriodicTable()
        new_coords = np.asarray(new_coords, dtype=np.float32)
        new_ac = np.zeros(self._init_connectivity.shape, dtype=np.int8)
        for i in range(new_coords.shape[0]):
            for j in range(new_coords.shape[1]):
                if i > j:
                    atom_num_i = pt.GetAtomicNumber(self._atom_symbols[i])
                    atom_num_j = pt.GetAtomicNumber(self._atom_symbols[j])

                    Rcov_i = pt.GetRcovalent(atom_num_i) * covalent_factor
                    Rcov_j = pt.GetRcovalent(atom_num_j) * covalent_factor

                    dist = np.linalg.norm(new_coords[i] - new_coords[j])
                    if dist < Rcov_i + Rcov_j:
                        new_ac[i, j] = new_ac[j, i] = 1

        if np.array_equal(new_ac, self._init_connectivity):
            return True

        return False

    def update_structure(self, covalent_factor=1.3):
        """ The sdf write is not pretty can this be done better?"""
        if 'structure' in self.results.keys():
            new_coords = self.results.pop('structure')
        else:
            raise RuntimeError('Compute new coords before update.')

        if self._check_structure(new_coords):
            self._converged = True

            natoms = self._init_connectivity.shape[0]
            init_sdf =  self.structure.split('\n')

            sdf_string = "\n".join(init_sdf[:4]) + "\n"
            for i, line in enumerate(init_sdf[4:4+natoms]):
                line_tmp = line.split()
                sdf_string += "".join([f"{x:10.4f}" for x in new_coords[i]])
                sdf_string += f" {line_tmp[3]}"
                sdf_string += "   " # three spaces?? why?
                sdf_string += " ".join(line_tmp[4:])
                sdf_string += "\n"
            sdf_string += "\n".join(init_sdf[4+natoms:])

            self.structure = sdf_string

        else:
            self._converged = False

    def write_xyz(self):
        return sdf2xyz(self.structure)


class Reagent(Molecule):
    """  """
    pass


class Product(Molecule):
    """ """
    pass


class Solvent(Molecule):
    """ """
    def __init__(self, smiles=None, n_solvent=0, active=0):
        """ """
        self._smiles = smiles
        self._n_solvent_molecules = n_solvent
        self._nactive = active


class Fragment(Molecule):
    """ Fragment class - removes label chiraity and atom map numbers"""

    def __init__(self, **kwds):
        super().__init__(**kwds)

        if self.molecule is not None:
            self._reset_label_chirality()

    @classmethod
    def from_rdkit_mol(cls, rdkit_mol):
        """ """
        try:
            label = rdkit_mol.GetProp('_Name')
        except KeyError:
            label = None

        obj = cls(label=label, removeHs=False, sanitize=False)
        obj.molecule = rdkit_mol

        # Reset atom map num and label chirality
        [atom.SetAtomMapNum(0) for atom in obj.molecule.GetAtoms()]
        obj._reset_label_chirality()

        if len(rdkit_mol.GetConformers()) == 0:  # If no 3D conformers.
            return obj

        # If rdkit mol has 3D confs, add confs to obj.
        if rdkit_mol.GetConformer().Is3D():
            for conf in rdkit_mol.GetConformers():
                mol_block = Chem.MolToMolBlock(rdkit_mol, condId=conf.Getid())
                obj._conformers.append(mol_block)

        return obj

    def _reset_label_chirality(self):
        """ Reset psudo R/S label chirality for fragments """
        patt = Chem.MolFromSmarts('[C^3;H2,H3]')
        atom_matches = self.molecule.GetSubstructMatches(patt)

        if atom_matches is not None:
            for atom_idx in atom_matches:
                self.molecule.GetAtomWithIdx(atom_idx[0]).SetChiralTag(
                                             rdchem.ChiralType.CHI_UNSPECIFIED)

    def num_rotatable_bond(self):
        """ Calculates the number of rotatable bonds in the fragment """
        if self.molecule is None:
            raise RuntimeError("Fragment has no RDKit mol")

        rot_bonds_smarts = [
                "[!#1]~[!$(*#*)&!D1]-!@[!$(*#*)&!D1]~[!#1]",
                "[*]~[*]-[O,S]-[#1]",
                "[*]~[*]-[NX3;H2]-[#1]",
                ]

        # TODO Sanitizing Molecule should be done elsewhere. 
        Chem.SanitizeMol(self.molecule)
        num_dihedral = 0
        for bond_smart in rot_bonds_smarts:
            dihedral_template = Chem.MolFromSmarts(bond_smart)
            dihedrals = self.molecule.GetSubstructMatches(dihedral_template)
            num_dihedral += len(dihedrals)

        return num_dihedral

    def make_conformers_rdkit(self, nConfs=20, uffSteps=5_000, seed=20):
        """
        makes conformers by embedding conformers using RDKit and
        refine with openbabel UFF. openbabel dependency might be
        eliminated at a later stage.
        """
        rdkit_mol = self._dative2covalent(self.molecule)  # needed for obabel

        p = Chem.rdDistGeom.srETKDGv3()
        try:  # Can the fragment actually be embedded?
            Chem.rdDistGeom.EmbedMultipleConfs(rdkit_mol, numConfs=nConfs,
                                               params=p)
        except RuntimeError:
            print(f"Failed to embed: {self.label} in step 1")
            self._embed_ok = False

        # make tmp mol to copy coords back into original connectivity.
        tmp_mol = copy.deepcopy(self.molecule)
        confid = AllChem.EmbedMolecule(tmp_mol)
        if confid >= 0:  # if RDKit can't embed confid = -1
            tmp_conf = tmp_mol.GetConformer()
            self._embed_ok = True
            for conf_idx, embed_conf in enumerate(rdkit_mol.GetConformers()):
                # Run OpenBabel UFF min.
                mol_block = Chem.MolToMolBlock(rdkit_mol,
                                               confId=embed_conf.GetId())
                obmol = pybel.readstring('sdf', mol_block)
                obmol.localopt(forcefield='uff', steps=uffSteps)

                # Update tmp mol conformer with UFF coords.
                for i, atom in enumerate(obmol):
                    x, y, z = atom.coords
                    tmp_conf.SetAtomPosition(i, Point3D(x, y, z))

                conf = Conformer(Chem.MolToMolBlock(tmp_mol),
                                 label=f"{self.label}_c{conf_idx}")
                self._conformers.append(conf)
        else:
            print(f"Failed to embed: {self.label} in step 2")
            self._embed_ok = False

    def relax_conformers(self, nprocs=4):
        """ """
        if self._embed_ok is True:  # Check that embeding is ok
            with mp.Pool(nprocs) as pool:
                results = pool.map(worker, self._conformers)

            for i, conf in enumerate(self._conformers):
                if results[i]['converged'] == True:
                    conf.results = results[i]
                    conf.update_structure()
                else:
                    conf._converged = False


# TODO: perhaps
def worker(conf):
    return conf.get_calculator().calculate()


class Reaction():
    """ """
    def __init__(self, reagents=None, solvent=None, reaction_name=None,
                 removeHs=False, sanitize=False):

        self._reagents = reagents
        self._solvent = solvent

        self._reaction_name = reaction_name
        self._reaction_mol = None

        self.products = []
        self.unique_fragments = []

        # Helper varibels
        self.removeHs = removeHs
        self.sanitize = sanitize

        self._prepare_reaction()

    def _prepare_reaction(self):
        """ Combine reagents and X active atoms into RDKit Mol
        and write mol file.
        """
        # Add Reagents
        if len(self._reagents) == 1:
            self._reaction_mol = self._reagents[0].molecule
        else:
            self._reaction_mol = Chem.CombineMols(
                                        self._reagents[0].molecule,
                                        self._reagents[1].molecule)
            if len(self._reagents) > 2:
                for reag in self._reagents[2:]:
                    self._reaction_mol = Chem.CombineMols(self._reaction_mol,
                                                          reag.molecules)

        # Add active solvent molecules
        for _ in range(self._solvent._nactive):
            sol_mol = Chem.AddHs(Chem.MolFromSmiles(self._solvent._smiles))
            self._reaction_mol = Chem.CombineMols(self._reaction_mol, sol_mol)
        AllChem.Compute2DCoords(self._reaction_mol)

        # Atom mapping, and set random chirality.
        for i, atom in enumerate(self._reaction_mol.GetAtoms()):
            atom.SetAtomMapNum(i+1)

        Chem.SanitizeMol(self._reaction_mol)
        opts = StereoEnumerationOptions(onlyUnassigned=False, unique=False)
        rdmolops.AssignStereochemistry(self._reaction_mol, cleanIt=True,
                                       flagPossibleStereoCenters=True,
                                       force=True)
        self._reaction_mol = next(EnumerateStereoisomers(self._reaction_mol,
                                  options=opts))
        rdmolops.AssignStereochemistry(self._reaction_mol, cleanIt=True,
                                       flagPossibleStereoCenters=True,
                                       force=True)

        # Write input file.
        self._reaction_mol.SetProp('_Name', self._reaction_name)
        Chem.MolToMolFile(self._reaction_mol, f'{self._reaction_name}.mol')

    def shake(self, max_bonds=2, CD=4, get_unique_fragments=True,
              generator=False, nprocs=1):
        """
        Enumerates all possible products where the combinations
        of `max_bonds` are broken/formed. However the maximum
        formed/broken bonds are `CD`.
        """
        self.products = valid_products(self._reaction_mol, n=max_bonds,
                            cd=CD,
                            charge=Chem.GetFormalCharge(self._reaction_mol),
                            n_procs=nprocs)

        if not generator:
            # TODO: give product names
            self.products = [Product.from_rdkit_mol(m) for m in self.products]
            if get_unique_fragments:  # Extract all fragments
                self.get_fragments()

        # TODO: If we have a transition metal add the remaining non active
        # solvent molecules.

    def get_fragments(self, remove_atomMapNum=True):
        """
        Return .csv file with fragments and the corresponding charge
        """
        if len(self.products) == 0:
            raise RuntimeError('No products created yet. Run .shake() .')

        # find fragment label number
        if len(self.unique_fragments) == 0:
            frag_num = 0
        else:
            frag_num = int(self.unique_fragments[-1].label.split('-')[0])

        for mol in self.products:
            fragments = Chem.GetMolFrags(mol.molecule, asMols=True,
                                         sanitizeFrags=False)
            for frag in fragments:
                mfrag = Fragment.from_rdkit_mol(frag)
                if mfrag not in self.unique_fragments:
                    mfrag.label = f'fragment-{frag_num}'
                    self.unique_fragments.append(mfrag)
                    frag_num += 1
