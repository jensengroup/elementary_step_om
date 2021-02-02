import sys
import pickle

from rdkit import Chem

sys.path.append('/home/koerstz/github/elementary_step_om')

if __name__ == '__main__':
    
    input_file = sys.argv[1]
    with open(input_file, "rb") as f:
        reactions = pickle.load(f)
    
    print(f"Total number of reactions: {len(reactions)} \n")
    output_reactions = []
    for i, reaction in enumerate(reactions):
        reaction._reaction_label = reaction._reaction_label + f"-{i}"
        print(f"Path search for: {reaction._reaction_label}")
        
        chrg = Chem.GetFormalCharge(reaction.reactant.molecule)
        reaction.get_ts_estimate(solvent='water', charge=chrg, refine=True)
        output_reactions.append(reaction)
        print()
    
    with open(input_file.split('.')[0] + "_output.pkl", 'wb') as out:
        pickle.dump(output_reactions, out)
