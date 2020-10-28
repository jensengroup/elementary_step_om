# import gzip
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdDistGeom, AllChem
from rdkit.Chem import rdmolops
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions

from .elementary_step import valid_products


class Molecule():
    """ """

    def __init__(self, file=None, removeHs=False, sanitize=False):
        if not file.endswith(('mol', 'sdf')):
            raise TypeError('Only works with mol/sdf files')

        self.label = file.split('.')[0]
        self._molfile = file
        self._conformers = []

        # Helper varibels
        self._2d_structure = self._is_2d()
        self.removeHs = removeHs
        self.sanitize = sanitize

        self._read_molfile()

    def _read_molfile(self):
        """ Reads Molfile removeHs and sanitize is set when instance is
        initiated
        """
        suppl = rdmolfiles.ForwardSDMolSupplier(self._molfile,
                                                removeHs=self.removeHs,
                                                sanitize=self.sanitize)
        conformers = []
        for confId, mol in enumerate(suppl):
            mol.SetProp('_Name', f"{self.label}-{confId}")
            conformers.append(mol)
        self._conformers = conformers

    def _write_molecule(self):
        """ Write molecule to mol.gz file """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def _is_2d(self):
        """ Check if molfile is 2D (a graph) """
        with open(self._molfile) as f:
            f.readline()
            if '2D' in f.readline():
                self._2d_structure = True
                return True
            else:
                self._2d_structure = False
                return False
    
    def embed_molecule(self, method='rdkit', seed=31, **kwargs):
        """ Reads conformer 1 from Molfile and reembeds molecule
        kwargs for RDKit EmbedMolecule. """
        if method.lower() == 'rdkit':
            self._read_molfile()
            rdDistGeom.EmbedMolecule(self._conformers[0], randomSeed=seed, **kwargs)
            Chem.MolToMolFile(self._conformers[0], self._molfile)
        self._2d_structure = False


class Reagent(Molecule):
    """  """
    pass


class Solvent(Molecule):
    """ """

    def __init__(self, smiles=None, n_solvent=0, active=0):
        """ """
        self._smiles = smiles
        self._n_solvent_molecules = n_solvent
        self._nactive = active
    

class Reaction():
    """ """

    def __init__(self, reagents=None, solvent=None, reaction_name=None,
                 removeHs=False, sanitize=False):
        
        self._reagents = reagents
        self._solvent = solvent

        self._reaction_name = reaction_name
        self._reaction_mol = None
        self._products = None

        # Helper varibels
        self.removeHs = removeHs
        self.sanitize = sanitize

        mol = self._prepare_reaction()

    def _prepare_reaction(self):
        """ Combine reagents and X active atoms into RDKit Mol
        and write mol file.
        """
        # Add Reagents
        if len(self._reagents) == 1:
            self._reaction_mol = self._reagents[0]._conformers[0]
        else:
            self._reaction_mol = Chem.CombineMols(
                                        self._reagents[0]._conformers[0],
                                        self._reagents[1]._conformers[0]
                                        )
            if len(self._reagents) > 2:
                for reag in self._reagents[2:]:
                    self._reaction_mol = Chem.CombineMols(self._reaction_mol,
                                                          reag._conformers[0])

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
                                   flagPossibleStereoCenters=True, force=True)
        self._reaction_mol = next(EnumerateStereoisomers(self._reaction_mol, options=opts))
        rdmolops.AssignStereochemistry(self._reaction_mol, cleanIt=True,
                                   flagPossibleStereoCenters=True, force=True)
        
        # Write input file.
        self._reaction_mol.SetProp('_Name', self._reaction_name)
        Chem.MolToMolFile(self._reaction_mol, f'{self._reaction_name}.mol')

    def shake(self, max_bonds=2, CD=4, generator=False, nprocs=1):
        """ """ 
        self._products = valid_products(self._reaction_mol, n=max_bonds, cd=CD, 
                                        charge=Chem.GetFormalCharge(self._reaction_mol),
                                        n_procs=nprocs)
        if not generator:
            self._products = list(self._products)

        # TODO: If we have a transition metal add the remaining non active 
        # solvent molecules.
