import sys
import pickle
from tqdm import tqdm 

sys.path.append(sys.path[0].rsplit('/', 1)[0])
from elementary_step_om.chem import Fragment
from elementary_step_om.external_calculation.xtb_calculations import xTBCalculator
from elementary_step_om.external_calculation.gaussian_calculations import GaussianCalculator


if __name__ == "__main__":
    
    exteral_xtb_in_g16 = "/home/koerstz/github/elementary_step_om/scripts/gaussian_xtb_external.py"

    ts_calc = GaussianCalculator(
        kwds="opt=(ts,calcall,noeigentest) pm3",
        properties=['structure', 'energy', 'frequencies'],
        external_script=None,
        charge=0,
        spin=1
    )

    irc_calc = GaussianCalculator(
        kwds="irc=(calcfc, recalc=10, maxpoints=50, stepsize=5) pm3",
        properties=['irc_structure', 'energy'],
        external_script=None,
        charge=0,
        spin=1
    )

    refine_calc = xTBCalculator(
            xtb_kwds="--opt loose",
            charge=0,
            spin=1
    ) 


    input_file = sys.argv[1]
    with open(input_file, "rb") as f:
        reactions = pickle.load(f)
    
    print(f"Total number of reactions: {len(reactions)} \n")
    
    for reaction in tqdm(reactions):
        #print("new")
        reaction.irc_check_ts(ts_calc, irc_calc, refine_calc)
        
        #print(reaction.__dict__)
        #print()
        #print()
    with open(input_file.split('.')[0] + "_output.pkl", 'wb') as f:
        pickle.dump(reactions, f)
