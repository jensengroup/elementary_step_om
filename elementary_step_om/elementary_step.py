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


def valid_product(product_adj_matrix, atomic_num):
    """ Check that the produces product is valid according to the valence"""
    product_valence = product_adj_matrix.sum(axis=1, dtype=np.int16)
    for atom, valence in zip(atomic_num, product_valence):
        if valence > max_valence[atom][1] or valence == max_valence[atom][0]:
            return False
    return True


def get_most_rigid_resonance(mol):
    """ Return the most rigid resonance structure """
    all_resonance_structures = [res for res in rdchem.ResonanceMolSupplier(mol,
                                ResonanceFlags.UNCONSTRAINED_ANIONS)]

    if len(all_resonance_structures) == 1:
        return all_resonance_structures[0]
    else:
        min_rot_bonds = 9999
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

    # TODO move these somewhere it makes more sense.
    product = reassign_atom_idx(product)
    reactant = reassign_atom_idx(reactant)

    Chem.SanitizeMol(product)
    Chem.SanitizeMol(reactant)

    # Find chiral atoms - including label chirality
    chiral_atoms_product = Chem.FindMolChiralCenters(product, includeUnassigned=True)

    unchanged_atoms = []
    for atom, chiral_tag in chiral_atoms_product:
        product_neighbors = [a.GetIdx() for a in product.GetAtomWithIdx(atom).GetNeighbors()]
        reactant_neighbors = [a.GetIdx() for a in reactant.GetAtomWithIdx(atom).GetNeighbors()]
        
        if sorted(product_neighbors) == sorted(reactant_neighbors):
            unchanged_atoms.append(atom)

    # make combinations of isomers.
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=False)
    rdmolops.AssignStereochemistry(product, cleanIt=True,
                                   flagPossibleStereoCenters=True, force=True)

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
    Chem.Kekulize(mol, clearAromaticFlags=True)

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

    conv_to_mol = partial(conversion_matrix_to_mol,
                          atomic_num=atomic_num, charge=charge)

    # TODO: make more mem efficient. Right now it's not ideal
    # all valid prod AC matrices are stored.
    valid_prod_ac_matrices = []
    for num_make, num_brake in comb_to_check[1:]:  # skip (0, 0)
        for conv_matrix in prod(comb(make1, num_make), comb(break1, num_brake)):
            conversion_matrix = np.array(sum(conv_matrix, ())).sum(axis=0)
            product_adj_matrix = reactant_adj_matrix + conversion_matrix

            if valid_product(product_adj_matrix, atomic_num):
                valid_prod_ac_matrices.append(product_adj_matrix)

    print(f"# valid product AC matrices: {len(valid_prod_ac_matrices)}")
    with Pool(n_procs) as pool:
        mols = pool.map(conv_to_mol, valid_prod_ac_matrices)
    mols = filter(None, mols)

    for mol in mols:
        for isomer in set_chirality(mol, reactant):
            yield isomer
