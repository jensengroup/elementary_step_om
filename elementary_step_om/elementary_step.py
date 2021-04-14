import networkx as nx
from elementary_step_om.chem import MappedMolecule, Reaction
from itertools import combinations, product
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

from numba import jit
from numba.typed import Dict
from numba import types

from .xyz2mol_local import AC2mol, get_proto_mol


key_type = types.int64
val_type = types.int32[:]

@jit(nopython=True)
def valid_product_numba(adj_matrix, atomic_num):
    """ Check that the produces product is valid according to the valence"""

    max_valence_numba = Dict.empty(key_type=key_type, value_type=val_type)

    max_valence_numba[1] = np.array([0, 1], dtype=np.int32)
    max_valence_numba[5] = np.array([0, 3], dtype=np.int32)
    max_valence_numba[6] = np.array([0, 4], dtype=np.int32)
    max_valence_numba[7] = np.array([0, 4], dtype=np.int32)
    max_valence_numba[8] = np.array([0, 3], dtype=np.int32)
    max_valence_numba[9] = np.array([0, 1], dtype=np.int32)
    max_valence_numba[14] = np.array([0, 4], dtype=np.int32)
    max_valence_numba[15] = np.array([0, 5], dtype=np.int32)
    max_valence_numba[16] = np.array([0, 6], dtype=np.int32)
    max_valence_numba[17] = np.array([0, 1], dtype=np.int32)
    max_valence_numba[35] = np.array([0, 1], dtype=np.int32)
    max_valence_numba[53] = np.array([0, 1], dtype=np.int32)

    natoms = adj_matrix.shape[0]
    product_valence = np.empty(natoms, dtype=np.int64)
    for i in  range(natoms):
        valence = 0
        for conection in adj_matrix[i]:
            valence += conection
        product_valence[i] = valence

    for atom, valence in zip(atomic_num, product_valence):
        if valence > max_valence_numba[atom][1] or valence == max_valence_numba[atom][0]:
            return False
    return True


# TODO: make a version that can be compiled
def not_parallel_reactions(I, frag_idx):
    """I a parallel reaction?

    A parallel reaction is when two fragments A B becomes A' B'
    but doesn't create a bond between the two fragments.
    However, when A B becomes A' B or A B' it's not a parrallel reaction.

    # TODO: This is not doing what it supposed to do. I think.
    """
    if len(frag_idx) == 1:  # Only one fragment.
        return True

    num_frags = len(frag_idx)
    frag_change_matrix = np.zeros((num_frags, num_frags), dtype=np.int32)
    for i, frag_i in enumerate(frag_idx):
        for j, frag_j in enumerate(frag_idx):
            m = I[np.ix_(frag_i, frag_j)]
            frag_change_matrix[i, j] = not np.all(m == 0)

    off_diagonal = frag_change_matrix[
        ~np.eye(frag_change_matrix.shape[0], dtype=bool)
    ].sum()
    diagonal = frag_change_matrix[np.eye(frag_change_matrix.shape[0], dtype=bool)].sum()
    #print(off_diagonal, diagonal, off_diagonal == 0 and diagonal > 1)
    if off_diagonal == 0 and diagonal > 1: # Think < has to change.
        return False
    
    return True


def reassign_atom_idx(mol):
    """ Assigns RDKit mol atom id to atom mapped id """
    renumber = [(atom.GetIdx(), atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    new_idx = [idx[0] for idx in sorted(renumber, key=lambda x: x[1])]
    mol = Chem.RenumberAtoms(mol, new_idx)
    rdmolops.AssignStereochemistry(mol, force=True)
    return mol


def get_most_rigid_resonance(mol):
    """ Return the most rigid resonance structure """
    all_resonance_structures = [
        res
        for res in rdchem.ResonanceMolSupplier(mol, ResonanceFlags.UNCONSTRAINED_ANIONS)
    ]

    min_rot_bonds = 9999
    most_rigid_res = copy.deepcopy(mol)
    if len(all_resonance_structures) <= 1:  # 0 is kind weird
        return most_rigid_res

    for res in all_resonance_structures:
        Chem.SanitizeMol(res)
        num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(res)
        if num_rot_bonds < min_rot_bonds:
            most_rigid_res = copy.deepcopy(res)
            min_rot_bonds = num_rot_bonds

    return most_rigid_res

class CreateValidIs:

    def __init__(self, mapped_molecule, max_num_bonds: int = 2, cd: int = 4) -> None:

        self._max_bonds = max_num_bonds
        self._cd = cd

        self._mapped_molecule = mapped_molecule
        self._ac_matrix = self._mapped_molecule.ac_matrix
        self._frag_idx = []
        for frag in Chem.GetMolFrags(mapped_molecule.rd_mol, asMols=True, sanitizeFrags=False):
            self._frag_idx.append([atom.GetAtomMapNum() - 1 for atom in frag.GetAtoms()])

        self._atom_number = np.array([atom.GetAtomicNum() for atom in self._mapped_molecule.rd_mol.GetAtoms()])

    def _make_simple_conversion_matrices(self) -> None:
        """ """
        # Create one-bond conversion matrices
        num_atoms = len(self._mapped_molecule.atom_symbols)

        make1, break1 = [], []
        for i in range(num_atoms):
            for j in range(num_atoms):
                conversion_matrix = np.zeros(self._ac_matrix.shape, np.int8)
                if j > i:
                    if self._ac_matrix [i, j] == 0:
                        conversion_matrix[i, j] = conversion_matrix[j, i] = 1
                        make1.append(conversion_matrix)
                    else:
                        conversion_matrix[i, j] = conversion_matrix[j, i] = -1
                        break1.append(conversion_matrix)

        self._make1 = make1
        self._break1 = break1

    def _make_break_combinations(self):
        """
        """
        product_combination = []
        for make_break in product(range(self._max_bonds + 1), repeat=2):
            if sum(make_break) <= self._cd and max(make_break) <= self._max_bonds:
                product_combination.append(make_break)
        return product_combination[1:]  # skip 0,0 = reactant.

    def __iter__(self):
        """
        """
        valid_valence_filter = partial(valid_product_numba, atomic_num=self._atom_number)
        parallel_reaction_filter = partial(not_parallel_reactions, frag_idx=self._frag_idx)

        self._make_simple_conversion_matrices()
        
        for num_make, num_break in self._make_break_combinations():
            make_combs = combinations(self._make1, num_make)
            break_combs = combinations(self._break1, num_break)

            for conv_matrix in product(make_combs, break_combs):
                conv_matrix = np.array(sum(conv_matrix, ())).sum(axis=0)  
                product_ac_matrix = self._ac_matrix  + conv_matrix

                if valid_valence_filter(product_ac_matrix):
                    if parallel_reaction_filter(conv_matrix):
                        yield product_ac_matrix
   
class TakeElementaryStep:

    def __init__(self, mapped_molecule, max_num_bonds: int = 2, cd: int = 4) -> None:
        
        self._max_bonds = max_num_bonds
        self._cd = cd

        self._mapped_molecule = mapped_molecule
        self._atom_number = [atom.GetAtomicNum() for atom in self._mapped_molecule.rd_mol.GetAtoms()]

    def _ac_to_mapped_products(self, product_ac_matrix):
        """ """
        formal_charge = Chem.GetFormalCharge(self._mapped_molecule.rd_mol)

        proto_mol = get_proto_mol(self._atom_number)
        mol = AC2mol(
            proto_mol,
            product_ac_matrix,
            self._atom_number,
            charge=formal_charge,
            allow_charged_fragments=True,
            use_graph=True,
            use_atom_maps=True,
        )

        if int(Chem.GetFormalCharge(mol)) != formal_charge:
           return None
        
        return get_most_rigid_resonance(mol) 

    def get_isomers(self, product):
        """Produce all combinations of isomers (R/S and cis/trans). But force
        product atoms with unchanged neighbors to the same label chirality as
        the reactant"""

        product = reassign_atom_idx(product)
        reactant = reassign_atom_idx(copy.deepcopy(self._mapped_molecule.rd_mol))

        # Find chiral atoms - including label chirality
        chiral_atoms_product = Chem.FindMolChiralCenters(product, includeUnassigned=True)

        unchanged_atoms = []
        for atom, _ in chiral_atoms_product:
            product_neighbors = [
                a.GetIdx() for a in product.GetAtomWithIdx(atom).GetNeighbors()
            ]
            reactant_neighbors = [
                a.GetIdx() for a in reactant.GetAtomWithIdx(atom).GetNeighbors()
            ]

            if sorted(product_neighbors) == sorted(reactant_neighbors):
                unchanged_atoms.append(atom)

        # make combinations of isomers.
        opts = StereoEnumerationOptions(onlyUnassigned=False, unique=False)
        rdmolops.AssignStereochemistry(
            product, cleanIt=True, flagPossibleStereoCenters=True, force=True
        )

        product_isomers_mols = []
        for product_isomer in EnumerateStereoisomers(product, options=opts):
            rdmolops.AssignStereochemistry(product_isomer, force=True)
            for atom in unchanged_atoms:
                reactant_global_tag = reactant.GetAtomWithIdx(atom).GetProp("_CIPCode")

                # TODO make sure that the _CIPRank is the same for atom in reactant and product.
                product_isomer_global_tag = product_isomer.GetAtomWithIdx(atom).GetProp(
                    "_CIPCode"
                )
                if reactant_global_tag != product_isomer_global_tag:
                    product_isomer.GetAtomWithIdx(atom).InvertChirality()
            
            #isomer = Chem.MolFromMolBlock(Chem.MolToMolBlock(product_isomer), sanitize=False)
            #Chem.SanitizeMol(isomer)
            
            mapped_isomer = MappedMolecule(molblock=Chem.MolToMolBlock(product_isomer))
            if mapped_isomer not in product_isomers_mols:
                product_isomers_mols.append(mapped_isomer)

        return product_isomers_mols

    def _AC2mappedisomers(self, AC):
        """ """
        mol = self._ac_to_mapped_products(AC)
        if mol is None:
            return []
        
        return self.get_isomers(mol)

    def get_products(self, nprocs=1):

        valid_Is = CreateValidIs(
            self._mapped_molecule, max_num_bonds=self._max_bonds, cd=self._cd
        )

        with Pool(processes=nprocs) as pool:
            mapped_products = pool.map(self._AC2mappedisomers, valid_Is)
        
        tmp_mapped_products = []
        for sublist in mapped_products:
            tmp_mapped_products += sublist

        return tmp_mapped_products
    
    # TODO: make network nodes/edges in parallel.
    def get_network(self, spin: int = 1, remove_self_loop: bool =True) -> nx.MultiDiGraph:
        """ """
        network = nx.MultiDiGraph()
        start_unmapped_molecule = self._mapped_molecule.get_unmapped_molecule()
        network.add_node(
            start_unmapped_molecule.__hash__(),
            canonical_reactant = start_unmapped_molecule,
            is_run = True
        )

        nodes = []
        edges = []
        # Run this in parallel.
        for mapped_product in self.get_products():
            canonical_product = mapped_product.get_unmapped_molecule()
            reaction = Reaction(
                reactant=self._mapped_molecule,
                product=mapped_product,
                charge=Chem.GetFormalCharge(self._mapped_molecule.rd_mol),
                spin=spin
            )
            
            # Make new node
            node_name = canonical_product.__hash__()
            node_data = {
                "canonical_reactant": canonical_product,
                "mapped_reactant": mapped_product,
                "is_run": False
            }
            nodes.append((node_name, node_data))

            # Make new reaction
            in_node = start_unmapped_molecule.__hash__()
            out_node = canonical_product.__hash__()
            reaction_key = reaction.__hash__()

            edge_data = {"reaction": reaction}
            edges.append((in_node, out_node, reaction_key, edge_data))
        
        #this is not super pretty - but it seems to work.
        add_nodes = []
        for node in nodes:
            if node[0] not in network:
                add_nodes.append(node)

        network.add_nodes_from(add_nodes)
        network.add_edges_from(edges)

        if remove_self_loop:
            network.remove_edges_from(nx.selfloop_edges(network))

        return network
        