import unittest
import os

from rdkit import Chem

from elementary_step_om.external_calculation.calculator import Calculator

from elementary_step_om.external_calculation.xtb_calculations import (
    xTBCalculator, 
    xTBPathSearch,
)

from elementary_step_om.external_calculation.gaussian_calculations import GaussianCalculator


from elementary_step_om.chem import (
    MappedMolecule,
    Reaction
) 

class TestCalculator(unittest.TestCase):        

    def test_calculator_make_and_renove_wdir(self):
        calc = Calculator(location='/tmp')

        # Make working dir
        calc._make_working_directory(namespace='test', overwrite=True)
        self.assertTrue(os.path.isdir(os.path.join('/tmp', 'test')))

        # Remove working dir
        calc._remove_working_dir(namespace='test')
        self.assertFalse(os.path.isdir(os.path.join('/tmp', 'test')))


class TestxTBCalculator(unittest.TestCase):

    def setUp(self) -> None:
        self._initdir = os.getcwd()

        self.xtb_path = os.environ["XTB_CMD"]
        
        self.atoms = ['C', 'H', 'H', 'H', 'H']
        self.coords = [
            [-0.55529, 0.99311, -0.00000],
            [ 0.24471, 0.74474,  0.66575],
            [-0.23269, 1.75744, -0.67574],
            [-0.84058, 0.12377, -0.55475],
            [-1.39259, 1.34650,  0.56475]
        ]   

    def test_make_cmd(self) -> None:
        xtb = xTBCalculator(
                xtb_kwds="--sp loose", charge=1, spin=2, properties=['structure']
            )

        self.assertEqual(xtb._make_cmd("xx"),
            self.xtb_path + " xx --sp loose --norestart --chrg 1 --uhf 1"
        )
    
    def test_write_input(self) -> None:
        xtb = xTBCalculator(
            xtb_kwds="--opt loose", 
            charge=0,
            spin=1,
            properties=['energy', 'structure'],
            location="/tmp"
        )

        # TODO: there might be a problem here.
        _ = xtb._write_input(self.atoms, self.coords, "mol_test")
        self.assertTrue(os.path.exists("./mol_test.xyz"))

        os.remove("./mol_test.xyz")

    
    def test_calculation(self) -> None:
        xtb = xTBCalculator(
            xtb_kwds="--opt loose", 
            charge=0,
            spin=1,
            properties=['energy', 'structure'],
            location="/tmp"
        )

        results = xtb(self.atoms, self.coords, label="mol_test")
        self.assertAlmostEqual(results['energy'], -4.175218505571, places=7)


# class TestxTBPathSearch(unittest.TestCase):
#     """ """
#     def setUp(self) -> None:
#         with open("./tests/fixtures/reaction_test_reac.mol") as reacfile:
#             tmp_mol = Chem.MolFromMolBlock(reacfile.read(), sanitize=False)
#             [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in tmp_mol.GetAtoms()]
#             self.reac_molblock = Chem.MolToMolBlock(tmp_mol)
        
#         with open("./tests/fixtures/reaction_test_prod.mol") as prodfile:
#             tmp_mol = Chem.MolFromMolBlock(prodfile.read(), sanitize=False)
#             [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in tmp_mol.GetAtoms()]
#             self.prod_molblock = Chem.MolToMolBlock(tmp_mol)

#     def test_path_search(self):
#         reactant = MappedMolecule(self.reac_molblock, label="reactant")
#         product = MappedMolecule(self.prod_molblock, label="product")

#         reaction = Reaction(reactant, product)
#         path_search_calc = xTBPathSearch(
#             xtb_kwds="", location="/tmp", nruns=1, overwrite=True
#             )

#         reaction.path_search_calculator = path_search_calc
#         reaction.run_path_search(seed=42)

#         xtb_external_script =  "/home/koerstz/github/elementary_step_om/scripts/gaussian_xtb_external.py"
#         ts_calculator = GaussianCalculator(
#             kwds="opt=(ts, calcall, noeigentest)",
#             properties=['structure', 'energy', 'frequencies'],
#             external_script=xtb_external_script
#         )
        
#         irc_calculator = GaussianCalculator(
#             kwds="irc=(calcfc, recalc=10, maxpoints=50, loose, stepsize=5, recorrect=never)",
#             properties=['irc_structure'],
#             external_script=xtb_external_script
#         )

#         refine_calc = xTBCalculator()

#         reaction.irc_check_ts(ts_calculator, irc_calculator, refine_calc)


class TestGaussianCalculator(unittest.TestCase):
    
    def setUp(self) -> None:
        self._initdir = os.getcwd()

        self.g16_path = os.environ["G16_CMD"]
        
        self.atoms = ['C', 'H', 'H', 'H', 'H']
        self.coords = [
            [-0.55529, 0.99311, -0.00000],
            [ 0.24471, 0.74474,  0.66575],
            [-0.23269, 1.75744, -0.67574],
            [-0.84058, 0.12377, -0.55475],
            [-1.39259, 1.34650,  0.56475]
        ]   

    def test_make_cmd(self) -> None:
        g16 = GaussianCalculator(
                kwds="opt", charge=0, spin=1, properties=['structure'], nprocs=1, memory=2
            )

        self.assertEqual(g16._make_cmd("xx"), f"{self.g16_path} xx")

    def test_write_input(self) -> None:
        xtb = GaussianCalculator(
            kwds="opt", charge=0, spin=1, properties=['energy', 'structure'], location="/tmp"
        )

        inp_file = xtb._write_input(self.atoms, self.coords, "mol_test")
        self.assertTrue(os.path.exists(inp_file))

        os.remove("mol_test.com")
    
    def test_calculation(self) -> None:
        g16 = GaussianCalculator(
            kwds="pm3 opt=loose", charge=0, spin=1, properties=['energy', 'structure'], location="/tmp"
        )

        results = g16(self.atoms, self.coords, label="mol_test1")
        self.assertAlmostEqual(results['energy'], -0.0207267746681, places=7)
    
    def test_external_calculation(self) -> None:
        g16 = GaussianCalculator(
            kwds="opt",
            charge=0, spin=1,
            properties=['energy', 'structure'],
            location="/tmp",
            external_script="/home/koerstz/github/elementary_step_om/scripts/gaussian_xtb_external.py"
        )
        
        results = g16(self.atoms, self.coords, label="mol_test")
        self.assertAlmostEqual(results['energy'], -4.17521842, places=7)
    