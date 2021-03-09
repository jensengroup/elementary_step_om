"""
Run this script on the nodes. 

The script computes the fragment energies, for the reaction products. 
The fragment energies can be used to compute reaction enthalpies.
"""

import sys
import pickle
from tqdm import tqdm

from rdkit import Chem

sys.path.append(sys.path[0].rsplit('/', 1)[0])
from elementary_step_om.chem import Fragment
from elementary_step_om.xtb_calculations import xTBCalculator

if __name__ == '__main__':
    
    input_fragments_file = sys.argv[1]
    ncpus = 2

    with open(input_fragments_file, "rb") as frag_batch:
        fragments = pickle.load(frag_batch)
        
    # 1) Makes n conformers using RDKit and refine using OBabel
    # 2) Setup xTB calculation on fragments.
    # 3) Run the calculations (includes a check of the connectivity). 
    n = 0
    for fragment in tqdm(fragments):
        # 1)
        fragment.label = f"hey-{n}"
        fragment.make_fragment_conformers(nconfs=2) 
        
        # 2)
        formal_charge = Chem.GetFormalCharge(fragment.rd_mol)
        xtb_calc = xTBCalculator(
            xtb_kwds="--opt loose", charge=formal_charge, spin=1, properties=['energy', 'structure']
        )
        fragment.calculator = xtb_calc

        # 3)
        fragment.run_calculations()

    # Save fragments with updated conformers, and corresponding energy.
    with open(sys.argv[1].split('.')[0] + '_output.pkl', "wb") as frag_output:
        pickle.dump(fragments, frag_output)
