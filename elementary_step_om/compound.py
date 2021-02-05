import shutil
import copy
from joblib import Parallel
import multiprocessing as mp
import numpy as np

from rdkit.Geometry import Point3D
from rdkit import Chem
from rdkit.Chem import rdmolfiles, AllChem, rdmolops, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions

from openbabel import pybel
import hashlib

from .external_cmd import xTBPath, xTB

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


class Molecule:
    """ """

    def __init__(self, moltxt=None, label=None):

        self.label = label

        self.molecule = None   # This is a 2D graph
        self.pseudo_chiral = True

        self._conformers = []

        # Helper varibels
        self._2d_structure = None
        self._embed_ok = None
        self._mol_hash = None

    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label})"
    
    # def _repr_svg_(self):
    #     """ In Jupyter Notebooks show reacting mol """
    #     from rdkit.Chem.Draw import rdMolDraw2D

    #     d2d = rdMolDraw2D.MolDraw2DSVG(200,200)
    #     d2d.DrawMolecule(self.molecule, legend="Reacting Mol")
    #     d2d.FinishDrawing()
    #     return d2d.GetDrawingText()

    def __eq__(self, other):
        """  """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Comparing {self.__class__} to {other.__class__}")
        return self._mol_hash == other._mol_hash
        
        #if (self.molecule.HasSubstructMatch(other.molecule, useChirality=True) and
        #    other.molecule.HasSubstructMatch(self.molecule, useChirality=True)):
        #    return True
        #return False

    def __hash__(self):
        if self._mol_hash is None:
            m = hashlib.blake2b()
            m.update(Chem.MolToSmiles(self.molecule).encode('utf-8'))
            self._mol_hash = int(str(int(m.hexdigest(), 16))[:32])
        #self._mol_hash = Chem.MolToSmiles(self.molecule)
        #if self._mol_hash is None and not self.pseudo_chiral:
            #self._mol_hash = hash(Chem.MolToSmiles(self.molecule))
            #self._mol_hash = make_graph_hash(self.molecule)
        #elif self._mol_hash is None and self.pseudo_chiral:
            #self._mol_hash = make_graph_hash(self.molecule, use_atom_maps=True)
        
        return self._mol_hash

    def set_calculator(self, calculator):
        """ """
        for conf in self._conformers:
            conf.set_calculator(calculator)
        self.calculator = calculator
    
    def _remove_pseudo_chirality(self):
        """ """
        rdmolops.AssignStereochemistry(self.molecule, cleanIt=True,
                                       flagPossibleStereoCenters=True,
                                       force=True)

    def _remove_atom_mapping(self):
        """ Remove atom mapping for molecule """
        [atom.SetAtomMapNum(0) for atom in self.molecule.GetAtoms()]

    def make_canonical(self):
        self._remove_atom_mapping()
        self._remove_pseudo_chirality()
        
        self.pseudo_chiral = False
        return self

#    def _read_moltxt(self, moltxt):
#        """ """
#        if not all(x in moltxt for x in ['$$$$', 'V2000']):
#            raise TypeError('not .sdf/.mol text')
#        elif 'V3000' in moltxt:
#            NotImplementedError('.sdf V3000 not implemented as txt yet.')
#
#        self._conformers = [f'{x}$$$$' for x in moltxt.split('$$$$')][:-1]
#
#        first_mol = Chem.MolFromMolBlock(self._conformers[0],
#                                         sanitize=False,
#                                         removeHs=False)
#
#        if not reactionsfirst_mol.GetConformer().Is3D():  # if 1D of 2D
#            self.molecule = first_mol
#            self._2d_structure = True
#        else:  # if 3D create 2D coords
#            AllChem.Compute2DCoords(first_mol)
#            self.molecule = first_mol
#
#        if self.label is None:
#            self.label = self._conformers[0].split('\n')[0]

    @classmethod
    def from_molfile(cls, file):
        """
        Reads Molfile
        """
        if not file.endswith(('mol', 'sdf')):
            raise TypeError('Only works with mol/sdf files')

        obj = cls(moltxt=None, label=file.split('.')[0])

        suppl = rdmolfiles.SDMolSupplier(file, removeHs=False, sanitize=False)

        first_mol = next(suppl)
        if not first_mol.GetConformer().Is3D():  # if 1D or 2D
            obj.molecule = first_mol
            obj._2d_structure = True
            return obj

        AllChem.Compute2DCoords(first_mol)
        obj.molecule = first_mol
        for conf_idx in range(len(suppl)):
            obj._conformers.append(
                Conformer(sdf=suppl.GetItemText(conf_idx), 
                          label=f"{file.split('.')[0]}-{conf_idx}")
            )
        obj._2d_structure = False
        return obj

    @classmethod
    def from_rdkit_mol(cls, rdkit_mol, label='molecule'):
        """ """
        obj = cls(label=label)
        obj.molecule = copy.deepcopy(rdkit_mol)

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
    
    def get_fragments(self):
        """ """
        fragments = []
        for frag in Chem.GetMolFrags(self.molecule, asMols=True, sanitizeFrags=False):
            fragments.append(Fragment.from_rdkit_mol(frag))
        return fragments

    def _embed_fragment(self, frag_rdkit, nconfs=20, uffSteps=5_000, seed=20):
        """ """
        p = AllChem.ETKDGv3()
        p.useRandomCoords = True
        p.randomSeed=seed
        try:  # Can the fragment actually be embedded?
            AllChem.EmbedMultipleConfs(frag_rdkit, numConfs=nconfs, params=p)
        except RuntimeError:
            print(f"RDKit Failed to embed: {self.label}.")
            return None, False

        tmp_frag_mol = copy.deepcopy(frag_rdkit)
        confid = AllChem.EmbedMolecule(tmp_frag_mol, params=p)
        conformers = []
        if confid >= 0:  # if RDKit can't embed confid = -1
            tmp_frag_conf = tmp_frag_mol.GetConformer()
            embed_ok = True
            for conf_idx, embed_conf in enumerate(frag_rdkit.GetConformers()):
                # Run OpenBabel UFF min.
                mol_block = Chem.MolToMolBlock(frag_rdkit, confId=embed_conf.GetId())
                obmol = pybel.readstring('sdf', mol_block)
                obmol.localopt(forcefield='uff', steps=uffSteps)

                # Update tmp mol conformer with UFF coords.
                for i, atom in enumerate(obmol):
                    x, y, z = atom.coords
                    tmp_frag_conf.SetAtomPosition(i, Point3D(x, y, z))

                conf_name = f"{self.label}_c{conf_idx}"
                conf = Conformer(Chem.MolToMolBlock(tmp_frag_mol), label=conf_name)
                conformers.append(conf)
        else:
            print(f"openbabel failed to UFF optimize: {self.label}.")
            embed_ok = False

        return conformers, embed_ok

class Conformer:
    """ """
    def __init__(self, sdf=None, label=None, calculator=None):
        """ """
        self.structure = sdf
        self.label = label
        self._converged = None
        self._set_atom_symbols()
        self._set_init_connectivity() # makes connectivty matrix from sdf

        if sdf is not None:
            self._embed_ok = True

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
            connectivity[int(i) - 1, int(j) - 1] = 1
            connectivity[int(j) - 1, int(i) - 1] = 1

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
            for j in range(new_coords.shape[0]):
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
    
    def _update_structure(self, new_coords):
        """ """
        natoms = self._init_connectivity.shape[0]
        init_sdf =  self.structure.split('\n')

        sdf_string = "\n".join(init_sdf[:4]) + "\n"
        for i, line in enumerate(init_sdf[4:4+natoms]):
            line_tmp = line.split()
            line_tmp = line.split()
            sdf_string += "".join([f"{x:10.4f}" for x in new_coords[i]])
            sdf_string += f" {line_tmp[3]}"
            sdf_string += "   " # three spaces?? why?
            sdf_string += " ".join(line_tmp[4:])
            sdf_string += "\n"
        sdf_string += "\n".join(init_sdf[4+natoms:])
        
        self.structure = sdf_string

    def update_structure(self, covalent_factor=1.3):
        """ The sdf write is not pretty can this be done better?"""
        if 'structure' in self.results.keys():
            new_coords = self.results.pop('structure')
        else:
            raise RuntimeError('Compute new coords before update.')
        
        if self._check_structure(new_coords):
            self._update_structure(new_coords)
            self._converged = True
        else:
            self._converged = False
    
    def write_xyz(self, file=None):
        xyz = sdf2xyz(self.structure)
        if file is not None:
            with open(file, 'w') as xyz_file:
                xyz_file.write(xyz)
        else:
            return xyz

    def relax_conformer(self):
        """ """
        if self._embed_ok is True:  # Check that embeding is ok
            results = self.__calc.calculate()  
            converged = results.pop('converged')
            if converged == True:
                self.results = results
                self.update_structure()
            else:
                self._converged = False


class Solvent(Molecule):
    """ """
    def __init__(self, smiles=None, n_solvent=0, active=0):
        """ """
        self._smiles = smiles
        self._n_solvent_molecules = n_solvent
        self._nactive = active
        self.label = 'solvent'


class Fragment(Molecule):
    """ Fragment class - removes label chiraity and atom map numbers"""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.pseudo_chiral = False 

    #    if self.molecule is not None:
    #        self._reset_label_chirality()

    #def _reset_label_chirality(self):
    #    """ Reset psudo R/S label chirality for fragments """
    #    patt = Chem.MolFromSmarts('[C^3;H2,H3]')
    #    atom_matches = self.molecule.GetSubstructMatches(patt)
    #
    #    if atom_matches is not None:
    #        for atom_idx in atom_matches:
    #            self.molecule.GetAtomWithIdx(atom_idx[0]).SetChiralTag(
    #                                         rdchem.ChiralType.CHI_UNSPECIFIED)

    @classmethod
    def from_rdkit_mol(cls, rdkit_mol):
        """ """
        try:
            label = rdkit_mol.GetProp('_Name')
        except KeyError:
            label = None

        obj = cls(label=label)
        obj.molecule = rdkit_mol

        # Reset atom map num and label chirality
        #[atom.SetAtomMapNum(0) for atom in obj.molecule.GetAtoms()]
        #obj._reset_label_chirality()

        if len(rdkit_mol.GetConformers()) == 0:  # If no 3D conformers.
            return obj

        # If rdkit mol has 3D confs, add confs to obj.
        if rdkit_mol.GetConformer().Is3D():
            for conf in rdkit_mol.GetConformers():
                mol_block = Chem.MolToMolBlock(rdkit_mol, condId=conf.Getid())
                obj._conformers.append(mol_block)

        return obj

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
    
    def make_conformers_rdkit(self, nconfs=20, uffSteps=5_000, seed=20):
        """
        Makes conformers by embedding conformers using RDKit and
        refine with openbabel UFF. If molecule is more then one fragment
        they are moved X*n_atoms awat from each other.
        
        openbabel dependency might be eliminated at a later stage.
        """
        rdkit_mol = self._dative2covalent(self.molecule)  # copy of rdkit_mol with covalent bonds broken

        self._conformers, self._embed_ok = self._embed_fragment(rdkit_mol, nconfs=nconfs, uffSteps=uffSteps, seed=seed)
        if self._embed_ok is not True:
            self._conformers = []

    def relax_conformers(self, nprocs=4):
        """ """
        if self._embed_ok is True:  # Check that embeding is ok
            with mp.Pool(int(nprocs)) as pool:
                results = pool.map(self.worker, self._conformers)
            for i, conf in enumerate(self._conformers):
                converged = results[i].pop('converged')
                if converged == True:
                    conf.results = results[i]
                    conf.update_structure() # checks if structure changed
                else:
                    conf._converged = False

    def worker(self, conf):
        return conf.get_calculator().calculate()


class Reaction:
    """
    Contains atom-mapped reactant and product.
    """
    def __init__(self, reactant, product, label='reaction'):
        
        self.reactant = reactant
        self.product = product
        self.ts_energy = None

        self._reactant_frags = None
        self._product_frags = None 

        self._reaction_label = label
        self._reaction_hash = None
    
    def __eq__(self, other):
        """  """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Comparing {self.__class__} to {other.__class__}")
        return self._reaction_hash == other._reaction_hash

    def __hash__(self):
        if self._reaction_hash is None:
            #self._reaction_hash = make_graph_hash(self.product.molecule, 
            #                                      use_atom_maps=True)
            m = hashlib.blake2b()
            m.update(Chem.MolToSmiles(self.reactant.molecule).encode('utf-8'))
            m.update(Chem.MolToSmiles(self.product.molecule).encode('utf-8'))
            self._reaction_hash = int(str(int(m.hexdigest(), 16))[:32])
        return self._reaction_hash
    
    def get_fragments(self):
        """ """
        reac_frags = [Fragment.from_rdkit_mol(frag) for frag in Chem.GetMolFrags(self.reactant.molecule, asMols=True, sanitizeFrags=False)]
        prod_frags = [Fragment.from_rdkit_mol(frag) for frag in Chem.GetMolFrags(self.product.molecule, asMols=True, sanitizeFrags=False)]
        
        self._reactant_frags = reac_frags
        self._product_frags = prod_frags

        return self._reactant_frags,  self._product_frags
    
    def _embed_molecules_appart(self, chrg=0, solvent="", multiplicity=1, refine=True):
        """ 
        """
        self.get_fragments() 

        def merge_fragments(frags):
            """ """
            if len(frags) == 1:
                combined_mol = Chem.MolFromMolBlock(frags[0]._conformers[0].structure,  sanitize=False)
            else:
                combined_mol = Chem.MolFromMolBlock(frags[0]._conformers[0].structure, sanitize=False)
                for frag in frags[1:]:
                    new_frag = Chem.MolFromMolBlock(frag._conformers[0].structure, sanitize=False)
                    conf = new_frag.GetConformer()
                    coords = np.array(conf.GetPositions())
                    for i in range(new_frag.GetNumAtoms()):
                        x, y, z = coords[i]
                        x = x + 0.8 * len(self._product_frags) * new_frag.GetNumAtoms() # translate x coordinate.
                        conf.SetAtomPosition(i, Point3D(x, y, z))
                    combined_mol = Chem.CombineMols(combined_mol, new_frag)
            
            atom_map_order = np.zeros(combined_mol.GetNumAtoms()).astype(np.int)
            for atom in combined_mol.GetAtoms():
                map_number = atom.GetAtomMapNum()-1
                atom_map_order[map_number] = atom.GetIdx()
            combined_mol = Chem.RenumberAtoms(combined_mol, atom_map_order.tolist())

            if refine:
                refine_args = {
                    'opt': '', # normal --opt 
                    'gbsa': solvent,
                    'chrg': chrg,
                    'uhf': multiplicity
                }
                xtb = xTB(xtb_args=refine_args) 
                conf = Conformer(sdf=Chem.MolToMolBlock(combined_mol), label='some_label')
                conf.set_calculator(xtb)
                conf.relax_conformer()
                shutil.rmtree('some_label')
                return conf
            else:
                return Conformer(sdf=Chem.MolToMolBlock(combined_mol), label='label-some')
        
        # Embed fragments
        for reac_frag in self._reactant_frags:
            reac_frag.make_conformers_rdkit(nconfs=1)  # Embed each fragment alone
        
        for prod_frag in self._product_frags:
            prod_frag.make_conformers_rdkit(nconfs=1)  # Embed each fragment alone

        self.reactant._conformers = [merge_fragments(self._reactant_frags)]
        self.product._conformers = [merge_fragments(self._product_frags)]

        self.product._embed_ok = True
        self.reactant._embed_ok = True

    def get_ts_estimate(self, nruns=3, solvent="", charge=0, spin=1, refine=True, save_paths=False):
        """
        """
        self._embed_molecules_appart(
            chrg=charge, 
            solvent=solvent,
            multiplicity=spin,
            refine=refine)

        xtb_path = xTBPath(self,
            label=self._reaction_label,
            charge=charge,
            spin=spin,
            solvent=solvent
            )

        self.ts_energy, self.ts_coords = xtb_path.run_barrier_scan_ntimes(nruns=nruns, save_paths=save_paths)

        return self.ts_energy
