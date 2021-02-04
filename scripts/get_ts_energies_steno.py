import sys
import pickle

from rdkit import Chem

sys.path.append('/groups/kemi/koerstz/git/elementary_step_om')


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
        spin = 1
        reaction.get_ts_estimate(solvent=None, charge=chrg, multiplicity=spin, nruns=3, refine=True)
        output_reactions.append(reaction)
        print()

    with open(input_file.split('.')[0] + "_output.pkl", 'wb') as out:
        pickle.dump(output_reactions, out)
