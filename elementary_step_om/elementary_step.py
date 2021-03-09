from itertools import combinations as comb
from itertools import product as prod
import copy
import numpy as np

from functools import partial
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdchem
from rdkit.Chem.rdchem import ResonanceFlags
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions
from rdkit.Chem import rdmolops


from .xyz2mol_local import AC2mol, get_proto_mol


max_valence = {}

# [min not allowed valence, max allwed valence]
max_valence[1] = [0, 1]
max_valence[6] = [0, 4]
max_valence[7] = [0, 4]
max_valence[8] = [0, 3]
max_valence[9] = [0, 1]
max_valence[14] = [0, 4]
max_valence[15] = [0, 5]
max_valence[16] = [0, 6]
max_valence[17] = [0, 1]
max_valence[35] = [0, 1]
max_valence[53] = [0, 1]


# TODO: make a version that can be compiled
def valid_product(product_adj_matrix, atomic_num):
    """ Check that the produces product is valid according to the valence"""
    product_valence = product_adj_matrix.sum(axis=1, dtype=np.int16)
    for atom, valence in zip(atomic_num, product_valence):
        if valence > max_valence[atom][1] or valence == max_valence[atom][0]:
            return False
    return True

# TODO: make a version that can be compiled
def parallel_reactions(I, frag_idx): 
    """ I a parallel reaction? 
    
    A parallel reaction is when two fragments A B becomes A' B'
    but doesn't create a bond between the two fragments.
    However, when A B becomes A' B or A B' it's not a parrallel reaction.
    """
    if len(frag_idx) == 1: # Only one fragment.
        return False

    num_frags = len(frag_idx)
    frag_change_matrix = np.zeros((num_frags, num_frags), dtype=np.int32)
    for i, frag_i in enumerate(frag_idx):
        for j, frag_j in enumerate(frag_idx):
            m = I[np.ix_(frag_i, frag_j)]
            frag_change_matrix[i,j] = not np.all(m == 0)
    
    off_diagonal = frag_change_matrix[~np.eye(frag_change_matrix.shape[0],dtype=bool)].sum()
    diagonal = frag_change_matrix[np.eye(frag_change_matrix.shape[0],dtype=bool)].sum()

    if off_diagonal == 0 and diagonal < 1: 
        return True
    
    return False


def get_most_rigid_resonance(mol):
    """ Return the most rigid resonance structure """
    all_resonance_structures = [res for res in rdchem.ResonanceMolSupplier(mol,
                                ResonanceFlags.UNCONSTRAINED_ANIONS)]

    min_rot_bonds = 9999
    most_rigid_res = copy.deepcopy(mol)
    if len(all_resonance_structures) <=1: # 0 is kind weird
        return most_rigid_res

    for res in all_resonance_structures:
        Chem.SanitizeMol(res)
        num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(res)
        if num_rot_bonds < min_rot_bonds:
            most_rigid_res = copy.deepcopy(res)
            min_rot_bonds = num_rot_bonds

    return most_rigid_res


def reassign_atom_idx(mol):
    """ Assigns RDKit mol atom id to atom mapped id """
    renumber = [(atom.GetIdx(), atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    new_idx = [idx[0] for idx in sorted(renumber, key=lambda x: x[1])]

    return Chem.RenumberAtoms(mol, new_idx)


def set_chirality(product, reactant):
    """ Produce all combinations of isomers (R/S and cis/trans). But force 
    product atoms with unchanged neighbors to the same label chirality as
    the reactant """

    product = reassign_atom_idx(product)
    reactant = reassign_atom_idx(reactant)

    # Find chiral atoms - including label chirality
    chiral_atoms_product = Chem.FindMolChiralCenters(product, includeUnassigned=True)

    unchanged_atoms = []
    for atom, _ in chiral_atoms_product:
        product_neighbors = [a.GetIdx() for a in product.GetAtomWithIdx(atom).GetNeighbors()]
        reactant_neighbors = [a.GetIdx() for a in reactant.GetAtomWithIdx(atom).GetNeighbors()]
        
        if sorted(product_neighbors) == sorted(reactant_neighbors):
            unchanged_atoms.append(atom)

    # make combinations of isomers.
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=False)
    rdmolops.AssignStereochemistry(
        product, cleanIt=True, flagPossibleStereoCenters=True, force=True
    )

    product_isomers = []
    product_isomers_mols = []
    for product_isomer in EnumerateStereoisomers(product, options=opts):
        rdmolops.AssignStereochemistry(product_isomer, force=True)
        for atom in unchanged_atoms:
            reactant_global_tag = reactant.GetAtomWithIdx(atom).GetProp('_CIPCode')

            # TODO make sure that the _CIPRank is the same for atom in reactant and product.
            product_isomer_global_tag = product_isomer.GetAtomWithIdx(atom).GetProp('_CIPCode')
            if reactant_global_tag != product_isomer_global_tag:
                product_isomer.GetAtomWithIdx(atom).InvertChirality()

        # TODO: This is using SMILES.
        if Chem.MolToSmiles(product_isomer) not in product_isomers:
            product_isomers.append(Chem.MolToSmiles(product_isomer))
            product_isomers_mols.append(product_isomer)

    return product_isomers_mols


def conversion_matrix_to_mol(prod_ac_matrix , atomic_num, charge):
    """ Applies the conversion matrix to the reactant AC matrix, and checks
    that the molecule is.

    Also works as helper function for creating mols in parallel.
    """

    proto_mol = get_proto_mol(atomic_num)
    mol = AC2mol(proto_mol, prod_ac_matrix, atomic_num, charge,
                 allow_charged_fragments=True, use_graph=True,
                 use_atom_maps=True)

    if int(Chem.GetFormalCharge(mol)) != charge:
        return None

    return get_most_rigid_resonance(mol)  # molecle with most double bonds.


def valid_products(reactant, n=2, cd=4, charge=0, n_procs=1):
    """ General generator that produces hypothetical valid
    one-step products from reactant.

    * n (int) - number of bonds to break/form
    * cd (int) - max value for #break + #form <= cd
    """
    reactant_adj_matrix = Chem.GetAdjacencyMatrix(reactant)
    atomic_num = [atom.GetAtomicNum() for atom in reactant.GetAtoms()]

    frag_idx = []
    for frag in Chem.GetMolFrags(reactant, asMols=True, sanitizeFrags=False):
        frag_idx.append([atom.GetAtomMapNum()-1 for atom in frag.GetAtoms()])

    # Create the one bond conversion matrices
    adjacency_matrix_shape = reactant_adj_matrix.shape
    make1, break1 = [], []
    for i in range(adjacency_matrix_shape[0]):
        for j in range(adjacency_matrix_shape[1]):
            conversion_matrix = np.zeros(adjacency_matrix_shape, np.int8)
            if j > i:
                if reactant_adj_matrix[i, j] == 0:
                    conversion_matrix[i, j] = conversion_matrix[j, i] = 1
                    make1.append(conversion_matrix)
                else:
                    conversion_matrix[i, j] = conversion_matrix[j, i] = -1
                    break1.append(conversion_matrix)

    # Use one bond conversion matrices to create the remaining
    # conversion matrices
    comb_to_check = [c for c in prod(range(n+1), repeat=2)
                     if sum(c) <= cd and max(c) <= n]

    conv_to_mol = partial(
        conversion_matrix_to_mol, atomic_num=atomic_num, charge=charge
    )

    # TODO: make more mem efficient. Right now it's not ideal
    # all valid prod AC matrices are stored.
    valid_prod_ac_matrices = []
    for num_make, num_brake in comb_to_check[1:]:  # excluding reactant 0,0
        for conv_matrix in prod(comb(make1, num_make), comb(break1, num_brake)):
            conversion_matrix = np.array(sum(conv_matrix, ())).sum(axis=0)
            
            # if it's a parallel reaction, continue.
            if parallel_reactions(conversion_matrix, frag_idx):
                continue

            product_adj_matrix = reactant_adj_matrix + conversion_matrix
            if valid_product(product_adj_matrix, atomic_num):
                valid_prod_ac_matrices.append(product_adj_matrix)

    print(f"# valid product AC matrices: {len(valid_prod_ac_matrices)}")
    with Pool(n_procs) as pool:
        mols = pool.map(conv_to_mol, valid_prod_ac_matrices)
    mols = filter(None, mols)

    for mol in mols:
        for isomer in set_chirality(mol, reactant):
            # Reinitialize the RDKit mol, which ensures it is valid.
            isomer = Chem.MolFromMolBlock(Chem.MolToMolBlock(isomer), sanitize=False)
            Chem.SanitizeMol(isomer)
            yield isomer
