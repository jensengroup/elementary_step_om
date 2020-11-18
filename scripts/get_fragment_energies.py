"""
Run this script on the nodes. 

The script computes the fragment energies, for the reaction products. 
The fragment energies can be used to compute reaction enthalpies.
"""

import sys
import pickle

from rdkit import Chem

sys.path.append('/home/koerstz/github/elementary_step_om')
from elementary_step_om import Fragment
from elementary_step_om.external_cmd import xTB

if __name__ == '__main__':
    
    input_fragments_file = sys.argv[1]
    ncpus = 6

    with open(input_fragments_file, "rb") as frag_batch:
        fragments = pickle.load(frag_batch)
        
    # 1) Makes n conformers using RDKit and refine using OBabel
    # 2) Setup xTB calculation on fragments.
    # 3) Run the calculations (includes a check of the connectivity). 
    for fragment in fragments:
        # 1)
        fragment.make_conformers_rdkit() 
        
        # 2)
        formal_charge = Chem.GetFormalCharge(fragment._reaction_mol)
        args = xtb_args={'opt': 'loose', 
                         'alpb': 'water',
                         'chrg': str(formal_charge)} 
        fragment.set_calculator(xTB(xtb_args=args)) 
        
        # 3)
        fragment.relax_conformers(nprocs=ncpus)
    
    # Save fragments with updated conformers, and corresponding energy.
    with open(sys.argv[1].split('.')[0] + '_output.pkl', "wb") as frag_output:
        pickle.dump(fragments, frag_output)
