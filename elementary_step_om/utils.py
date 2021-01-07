import networkx as nx
from itertools import combinations
import hashlib

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, BondStereo


def is_equal(frag1, frag2):
    if frag1.HasSubstructMatch(frag2) and frag2.HasSubstructMatch(frag1):
        return True
    return False


def equivalent_neighbors(atom, useChirality=True):
    """
    Breaks bonds on 'atom', and compares the resulting fragments.
    If two fragments are identical, the atom isn't chiral.

    I don't think i want to use chirality when comparing the
    fragments. Are the fragment different when it only
    differentiates in the chirality?
    """
    mol = atom.GetOwningMol()

    fragments = []
    for nei in atom.GetNeighbors():
        bond_to_break = mol.GetBondBetweenAtoms(atom.GetIdx(), nei.GetIdx()).GetIdx()
        new_mol = bond = Chem.FragmentOnBonds(
            mol, [bond_to_break], dummyLabels=[(0, 0)]
        )
        new_frags = Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=False)
        fragments += new_frags

        if len(new_frags) > 1:
            if is_equal(new_frags[0], new_frags[1]):
                return False

    for frag1, frag2 in combinations(fragments, 2):
        if is_equal(frag1, frag2):
            return True
    return False


def double_bond_pseudochiral(bond):
    """
    Checks if the double bond is pseudo chiral.
    """
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    if equivalent_neighbors(begin_atom) or equivalent_neighbors(end_atom):
        return True
    return False


def make_graph_hash(mol, use_atom_maps=False):
    """
    As is now. It doesn't recognise symmetric rings of i.e. the type:
    '[H]C([O-])=C([O-])OOC(=C1OO1)C([H])([H])[H]'  as two identical
    neighbours, and are therefore chiral.

    The function return the hash(weisfeiler_lehman_graph_hash).
    That is, the hash of the hash. The __hash__ dundur needs
    a integer which you get from the hash() functions.
    
    The hash is truncated to 16 digits. Is that smart?
    """
    ac_matrix = Chem.GetAdjacencyMatrix(mol)
    graph = nx.from_numpy_matrix(ac_matrix)

    # node labels
    for node_label in graph.nodes():
        atom = mol.GetAtomWithIdx(node_label)
        atom_symbol = atom.GetSymbol()

        # Chirality on atoms
        rdkit_chiral_tag = atom.GetChiralTag()
        if rdkit_chiral_tag == ChiralType.CHI_UNSPECIFIED:
            chiral_tag = ""
        else:
            if equivalent_neighbors(atom):
                chiral_tag = ""
            elif rdkit_chiral_tag == ChiralType.CHI_TETRAHEDRAL_CW:
                chiral_tag = "@"
            elif rdkit_chiral_tag == ChiralType.CHI_TETRAHEDRAL_CCW:
                chiral_tag = "@@"

        # Atomic charges
        formal_atomic_charge = str(atom.GetFormalCharge())

        atom_identifier = atom_symbol + chiral_tag + formal_atomic_charge
        if use_atom_maps:
            atom_identifier += str(atom.GetAtomMapNum())

        graph.nodes[node_label]["atom_identifier"] = atom_identifier

    # Edge labels
    for edge_labels in graph.edges():
        bond = mol.GetBondBetweenAtoms(*edge_labels)
        bond_type = bond.GetSmarts(allBondsExplicit=True)  # This or use bond type

        rdkit_bond_stereo = bond.GetStereo()
        if rdkit_bond_stereo == BondStereo.STEREONONE:
            bond_stereo = ""
        else:
            if double_bond_pseudochiral(bond):
                bond_stereo = ""
            elif rdkit_bond_stereo in [BondStereo.STEREOCIS, BondStereo.STEREOZ]:
                bond_stereo = "\/"
            elif rdkit_bond_stereo in [BondStereo.STEREOTRANS, BondStereo.STEREOE]:
                bond_stereo = "//"

        bond_identifier = bond_type + bond_stereo
        graph.edges[edge_labels]["bond_identifier"] = bond_identifier

    nx_hash_hex = nx.weisfeiler_lehman_graph_hash(graph, node_attr="atom_identifier", edge_attr="bond_identifier")
   
    m = hashlib.sha256()
    m.update(nx_hash_hex.encode('utf-8'))
    
    return int(str(int(m.hexdigest(), 16))[:16])
