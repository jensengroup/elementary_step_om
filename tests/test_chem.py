from elementary_step_om.external_calculation.gaussian_calculations import GaussianCalculator
import unittest

from rdkit import Chem
import numpy as np

from elementary_step_om.chem import (
    MoleculeException,
    BaseMolecule,
    Molecule,
    MappedMolecule,
    Fragment,
    Conformer,
    Reaction
)


from elementary_step_om.external_calculation.xtb_calculations import (
    xTBCalculator, 
    xTBPathSearch,
)


class TestBaseMolecule(unittest.TestCase):

    def setUp(self) -> None:
        with open("./tests/fixtures/mapped_molfile.mol", 'r') as mapped_molblock:           
            self.molblock_mapped = mapped_molblock.read()
        
        with open("./tests/fixtures/unmapped_molfile.mol", 'r') as mapped_molblock:           
            self.molblock_unmapped = mapped_molblock.read()
    
    def test_empty_initialize(self):
        self.assertRaises(MoleculeException, BaseMolecule, )

    def test_initalize_from_molblock(self):
        pass

    def test_initalize_from_molfile(self):
        pass

    def test_initalize_from_rdkit_mol(self):
        pass

    def test_has_atom_mapping(self):
        pass

    def test_num_rotatable_bonds(self):
        pass

    def test_set_calculator(self):
        pass

    def test_run_calculations_in_parallel(self):
        pass

    def test_embed_molecule(self):
        calc = xTBCalculator()
        basemolecule = BaseMolecule(molblock=self.molblock_mapped, label="test_mol")
        basemolecule.embed_molecule(confs_pr_frag=2, refine_calculator=calc)
        self.assertEqual(len(basemolecule.conformers), 4)


class TestMolecule(unittest.TestCase):
    def test_unmap_molecule(self):
        pass

    def test_make_mapped_molecule(self):
        pass

    def test_unmapped_hash(self):
        pass


class TestMappedMolecule(unittest.TestCase):
    def test_get_unmapped_molecule(self):
        pass

    def test_mapped_hash(self):
        pass


class TestFragment(unittest.TestCase):

    def setUp(self) -> None:
        with open("./tests/fixtures/water.mol", "r") as molfile:
            self.unmapped_molblock = molfile.read()

    def test_embed_fragment(self):
        # Check that the mapping is ok.
        frag = Fragment(molblock=self.unmapped_molblock, label="test")
        frag.make_fragment_conformers(nconfs=2)

        self.assertEqual(len(frag.conformers),  2)

    # Test that the mapping is ok.

class TestConformer(unittest.TestCase):
    def setUp(self) -> None:
        with open("./tests/fixtures/water.mol", "r") as molfile:
            self.molblock = molfile.read()

    def test_conformer_init(self):
        pass

    def test_init_connectivity(self):
        test_conf = Conformer(molblock=self.molblock, label="test_mol")

        true_init_conectivity = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        self.assertTrue(
            np.array_equal(test_conf._init_connectivity, true_init_conectivity)
        )

    def test_atom_symbols(self):
        test_conf = Conformer(molblock=self.molblock, label="test_mol")
        self.assertEqual(test_conf.atom_symbols, ["O", "H", "H"])

    def test_update_molblock(self):
        test_conf = Conformer(molblock=self.molblock, label="test_mol")
        init_coords = np.array(
            [
                [-0.6348, 1.1971, 0.0010],
                [0.2304, 0.9941, 0.6493],
                [-0.2352, 1.9814, -0.6587],
            ]
        )
        self.assertTrue(np.array_equal(test_conf.coordinates, init_coords))

        # Update molblock
        changed_coords = np.array(
            [
                [-2.6348, 1.1971, 0.0010],
                [0.2304, 0.9941, 0.6493],
                [-0.2352, 1.9814, -0.6587],
            ]
        )
        test_conf._update_molblock_coords(changed_coords)
        self.assertTrue(np.array_equal(test_conf.coordinates, changed_coords))

    def test_connectivity_check(self):
        test_conf = Conformer(molblock=self.molblock, label="test_mol")

        # Test that the input coords are the same:
        self.assertTrue(test_conf._check_connectivity(test_conf.coordinates))

        # What if the 1-2 bond is elongated?
        change_b12_length = np.array(
            [
                [-0.6348, 1.1971, 0.0010],
                [1.0366, 0.6931, 1.0015],
                [-0.2352, 1.9814, -0.6587],
            ]
        )
        self.assertFalse(test_conf._check_connectivity(change_b12_length))

    def test_run_calculation_sp(self):
        test_conf = Conformer(molblock=self.molblock, label="test_mol")
        xtb_calc = xTBCalculator(
            xtb_kwds="--sp", properties=["energy"], location="/tmp"
        )

        test_conf.calculator = xtb_calc
        test_conf.run_calculation()
        self.assertAlmostEqual(test_conf.results['energy'], -5.046521661199)

    def test_run_calculation_opt(self):
        test_conf = Conformer(molblock=self.molblock, label="test_mol")
        xtb_calc = xTBCalculator(
            xtb_kwds="--opt loose", properties=["energy", "structure"], location="/tmp"
        )

        test_conf.calculator = xtb_calc
        test_conf.run_calculation()
        self.assertAlmostEqual(test_conf.results['energy'], -5.070542913476)


# class TestReaction(unittest.TestCase):
    
#     def setUp(self) -> None:

#         with open("./tests/fixtures/reaction_test_reac.mol") as reacfile:
#             tmp_mol = Chem.MolFromMolBlock(reacfile.read(), sanitize=False)
#             [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in tmp_mol.GetAtoms()]
#             self.reac_molblock = Chem.MolToMolBlock(tmp_mol)
        
#         with open("./tests/fixtures/reaction_test_prod.mol") as prodfile:
#             tmp_mol = Chem.MolFromMolBlock(prodfile.read(), sanitize=False)
#             [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in tmp_mol.GetAtoms()]
#             self.prod_molblock = Chem.MolToMolBlock(tmp_mol)

#         self.product = MappedMolecule(self.prod_molblock, label="product")
#         self.reactant = MappedMolecule(self.reac_molblock, label="reactant")

#         self.reaction = Reaction(self.reactant, self.product, charge=0, spin=1, label='test_reaction')

#         self.xtb_external_script = "/home/koerstz/github/elementary_step_om/scripts/gaussian_xtb_external.py"

#         self.ts_guess_coords = np.array([
#             [ 2.91627753,  0.18151285,  1.35640364],
#             [ 2.00120470,  0.69177945,  0.39570334],
#             [ 6.30692278,  1.19898649,  0.63177906],
#             [ 5.72205623,  1.87038045,  1.66574245],
#             [ 7.22351241,  1.42596664,  0.2917494 ],
#             [ 2.12037641,  0.31989479, -0.70912777],
#             [ 1.17657684,  1.48504094,  0.62684816],
#             [ 4.12003428,  1.20263387,  2.09896795],
#             [ 2.59496514, -0.02410531,  2.27748232],
#             [ 5.88258895,  0.45889672,  0.11285571],
#             [ 6.28136693,  2.65772506,  2.29375323],
#             [ 4.16504112,  1.34570315,  1.36883216],
#             ])

#         self.ts_coords = np.array(
#             [[-1.381831, -0.627274, -0.106123],
#             [-1.78906 ,  0.624465,  0.056732],
#             [ 1.510003,  0.700091, -0.021235],
#             [ 1.37372 , -0.726196,  0.026563],
#             [ 2.371407,  1.213683,  0.045669],
#             [-1.944015,  1.32255 , -0.884482],
#             [-1.906813,  1.126618,  1.126579],
#             [ 0.587683, -1.081824, -0.850523],
#             [-1.444362, -1.569949,  0.209311],
#             [ 0.695559,  1.279934, -0.14543 ],
#             [ 2.38323 , -1.322021,  0.165086],
#             [ 0.436804, -0.970058,  0.808817]])
    
    # def test_ts_search(self):
    #     """ """
    #     ts_calculator = GaussianCalculator(
    #         kwds="opt=(ts,calcall,noeigentest)",
    #         properties=['structure', 'energy', 'frequencies'],
    #         external_script=self.xtb_external_script
    #     )
    #     results = self.reaction._run_ts_search(ts_calculator = ts_calculator)

    #     self.assertAlmostEqual(results['energy'], -12.2793155)
            
    # def test_irc(self):
    #     """ """
    #     irc_calculator = GaussianCalculator(
    #         kwds="irc=(calcfc, recalc=10, maxpoints=50, stepsize=5)",
    #         properties=['irc_structure', 'energy'],
    #         external_script=self.xtb_external_script
    #     )

    #     results = self.reaction._run_irc(irc_calculator=irc_calculator)

    #     self.assertAlmostEqual(results['forward']['energy'], -12.3820476)
    #     self.assertAlmostEqual(results['reverse']['energy'], -12.3900272)

    # def test_irc_check(self):
    #     """ """
    #     ts_calculator = GaussianCalculator(
    #         kwds="opt=(ts,calcall,noeigentest)",
    #         properties=['structure', 'energy', 'frequencies'],
    #         external_script=self.xtb_external_script
    #     )

    #     irc_calculator = GaussianCalculator(
    #         kwds="irc=(calcfc, recalc=10, maxpoints=50, stepsize=5)",
    #         properties=['irc_structure', 'energy'],
    #         external_script=self.xtb_external_script
    #     )

    #     refine_calculator = xTBCalculator()       

    #     self.reaction._ts_guess_coordinates = self.ts_guess_coords
    #     self.reaction.irc_check_ts(ts_calculator, irc_calculator, refine_calculator)
        
    #     print(self.reaction.__dict__)


if __name__ == "__main__":
    unittest.main()
