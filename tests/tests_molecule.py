import unittest
from rdkit import Chem

from elementary_step_om.compound import MoleculeException, Molecule


class TestMolecule(unittest.TestCase):
    
    def setUp(self):
        
        with open("./tests/fixtures/mapped_molfile.mol", 'r') as mapped_molblock:
            self.molblock_mapped = mapped_molblock.read()

        with open("./tests/fixtures/unmapped_molfile.mol", 'r') as unmapped_molblock:
            self.molblock_unmapped = unmapped_molblock.read()

    def test_no_input(self):
        self.assertRaises(MoleculeException, Molecule, )

    def test_init_with_mapped_molblock(self):
        mol = Molecule(moltxt=self.molblock_mapped)
        
        # Atom mapping is correct.
        self.assertEqual(
            [atom.GetAtomMapNum() for atom in mol.rd_mol.GetAtoms()], 
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        )
        
        # atom_mapped key is set
        self.assertTrue(mol.atom_mapped)

    def test_init_with_unmapped_molblock(self):
        mol = Molecule(moltxt=self.molblock_unmapped)
        
        # Atom mapping is correct.
        self.assertEqual(
            [atom.GetAtomMapNum() for atom in mol.rd_mol.GetAtoms()], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
        
        # atom_mapped key is set
        self.assertFalse(mol.atom_mapped)       

    def test_mapped_hash(self):
        mol = Molecule(moltxt=self.molblock_mapped)
        self.assertEqual(hash(mol), 1588624637251409299)

    def test_unmapped_hash(self):
        mol = Molecule(moltxt=self.molblock_unmapped)
        self.assertEqual(hash(mol), 2073239415288407608)
    
    def test_make_canonical(self):
        mol = Molecule(moltxt=self.molblock_mapped)
        mol.make_canonical()
        
        self.assertEqual(hash(mol), 2073239415288407608) # unmapped hash
        self.assertFalse(mol.atom_mapped) 

    

if __name__ == "__main__":
    unittest.main()
