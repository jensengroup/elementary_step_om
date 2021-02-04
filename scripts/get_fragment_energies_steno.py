"""
Run this script on the nodes. 

The script computes the fragment energies, for the reaction products. 
The fragment energies can be used to compute reaction enthalpies.
"""

import sys
import pickle

from rdkit import Chem

sys.path.append('/groups/kemi/koerstz/git/elementary_step_om')
from elementary_step_om import Fragment
from elementary_step_om.external_cmd import xTB

if __name__ == '__main__':

    input_fragments_file = sys.argv[1]
    ncpus = sys.argv[2]

    with open(input_fragments_file, "rb") as frag_batch:
        fragments = pickle.load(frag_batch)

    # 1) Makes n conformers using RDKit and refine using OBabel
    # 2) Setup xTB calculation on fragments.
    # 3) Run the calculations (includes a check of the connectivity). 
    for i, fragment in enumerate(fragments):
        # 1)
        fragment.label = f"hey-{i}"
        rot_bonds = fragment.num_rotatable_bond()
        print(f"batch fragment {i}: # confs {3+5*rot_bonds}")
        fragment.make_conformers_rdkit(nconfs=3+5*rot_bonds)

        # 2)
        formal_charge = Chem.GetFormalCharge(fragment.molecule)

        # kwds lookup xTB cmd line tool.
        args = xtb_args={'opt': 'loose',
                         'chrg': str(formal_charge)}

        fragment.set_calculator(xTB(xtb_args=args))

        # 3)
        fragment.relax_conformers(nprocs=ncpus)

    # Save fragments with updated conformers, and corresponding energy.
    with open(sys.argv[1].split('.')[0] + '_output.pkl', "wb") as frag_output:
        pickle.dump(fragments, frag_output)
