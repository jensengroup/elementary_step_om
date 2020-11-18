""" 
Use script to make batches to compute for fragments energies.
"""

import pickle
import sys

sys.path.append("/home/koerstz/github/elementary_step_om")
from elementary_step_om import Reaction


def make_batches(frag_list, n=5):
    """ Returns n sized batches from frag_list """
    for batch_idx in range(0, len(frag_list), n):
        yield frag_list[batch_idx : batch_idx + n]


if __name__ == "__main__":
    with open("reaction_object_m2_cd4_water1.pkl", "rb") as reaction_file:
        reaction = pickle.load(reaction_file)

    for idx, batch in enumerate(make_batches(reaction.unique_fragments, n=5)):
        with open(f"{reaction._reaction_name}_fragbatch_{idx}.pkl", "wb") as batchp:
            pickle.dump(batch, batchp)

        if batch_idx == 5:
            break
