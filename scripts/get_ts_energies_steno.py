import sys
import pickle

from rdkit import Chem

sys.path.append(sys.path[0].rsplit('/', 1)[0])
from elementary_step_om.chem import Reaction
from elementary_step_om.xtb_calculations import xTBPathSearch


if __name__ == '__main__':

    input_file = sys.argv[1]
    with open(input_file, "rb") as f:
        reactions = pickle.load(f)

    print(f"Total number of reactions: {len(reactions)} \n")
    output_reactions = []
    for i, reaction in enumerate(reactions[14:]):
        reaction.reaction_label = reaction.reaction_label + f"-{i}"
        print(f"Path search for: {reaction.reaction_label}")
        
        xtbpath = xTBPathSearch(xtb_kwds="", nruns=1)
        reaction.path_search_calculator = xtbpath
        
        reaction.run_path_search()
        output_reactions.append(reaction)
        
        print(reaction.__dict__)

    with open(input_file.split('.')[0] + "_output.pkl", 'wb') as out:
        pickle.dump(output_reactions, out)
