import networkx as nx
import pickle
import numpy as np
import copy
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions

from .elementary_step import valid_products
from .compound import Molecule, Fragment, Reaction

class ReactionNetwork:
    def __init__(
        self,
        reagents=None,
        solvent=None,
        max_bonds=2,
        max_chemical_dist=4,
        reaction_energy_cut_off=30.0,
        barrier_cutoff=50.0
    ):

        self.reagents = reagents
        self.solvent = solvent

        self._max_bonds = max_bonds
        self._max_cd = max_chemical_dist
        self._reaction_energy_cut_off = reaction_energy_cut_off
        self._barrier_cutoff = barrier_cutoff

        self._fragment_energies = dict()
        self._unique_reactions = dict()

        self.network = None

        if self.reagents is not None:
            self._initialize_network()

    def _initialize_network(self):
        """ """
#        reaction_cell = ReactionCell(
#            reacting_mol=self._prepare_reacting_mol(self.reagents),
#            solvent=self.solvent,
#            max_bonds=self._max_bonds,
#            max_chemical_dist=self._max_cd,
#        )

        mapped_reactant = self._prepare_reacting_mol(self.reagents)
        mapped_reactant.label = 'initial_reactant'

        canonical_mol = copy.deepcopy(mapped_reactant).make_canonical()

        self.network = nx.MultiDiGraph()
        self.network.add_node(canonical_mol.__hash__(),
            canonical_reactant=canonical_mol,
            mapped_reactant=mapped_reactant,
            is_run=False
        )

    def _prepare_reacting_mol(self, reagents):
        """ Combine reagents and X active atoms into RDKit Mol 
        which is the reacting molecule.

        TODO: move to ReactionCell
        """
        # Add Reagents
        if len(reagents) == 1:
            reactants = reagents[0].molecule
        else:
            reactants = Chem.CombineMols(reagents[0].molecule, reagents[1].molecule)
            if len(reagents) > 2:
                for reag in reagents[2:]:
                    reactants = Chem.CombineMols(reactants, reag.molecule)

        # Add active solvent molecules
        if self.solvent is not None:
            for _ in range(self.solvent._nactive):
                sol_mol = Chem.AddHs(Chem.MolFromSmiles(self.solvent._smiles))
                reactants = Chem.CombineMols(reactants, sol_mol)
            AllChem.Compute2DCoords(reactants)

        # Atom mapping, and set random chirality.
        for i, atom in enumerate(reactants.GetAtoms()):
            atom.SetAtomMapNum(i+1)

        Chem.SanitizeMol(reactants)
        opts = StereoEnumerationOptions(onlyUnassigned=False, unique=False)
        rdmolops.AssignStereochemistry(reactants, cleanIt=True,
                                       flagPossibleStereoCenters=True,
                                       force=True)
        reactants = next(EnumerateStereoisomers(reactants,
                                  options=opts))
        rdmolops.AssignStereochemistry(reactants, cleanIt=True,
                                       flagPossibleStereoCenters=True,
                                       force=True)
        
        return Molecule.from_rdkit_mol(reactants)


    def _get_fragment_energy(self, mol_hash):
        """ """
        if mol_hash in self._fragment_energies:
            return self._fragment_energies[mol_hash]
        return np.float('nan')

    def take_step(self, nprocs=6, remove_self_loop=True):
        """
        Expands the network with an eksta layer.
        """
        def filter_not_run(n1):
            return self.network.nodes[n1]["is_run"] == False

        not_run_nodes = nx.subgraph_view(self.network, filter_node=filter_not_run)
        new_edges = []
        new_nodes = []
        for node_name, node_data in not_run_nodes.nodes(data=True):
            mapped_products = valid_products(node_data['mapped_reactant'].molecule, n=self._max_bonds,
                                cd=self._max_cd,
                                charge=Chem.GetFormalCharge(node_data['mapped_reactant'].molecule),
                                n_procs=nprocs)
            
            for mapped_product in mapped_products:
                mapped_product = Molecule.from_rdkit_mol(mapped_product)
                canonical_product = copy.deepcopy(mapped_product).make_canonical()
                new_nodes.append(
                    (
                        canonical_product.__hash__(), 
                        {"canonical_reactant": canonical_product, "mapped_reactant": mapped_product, "is_run": False}
                    )
                )

                reac = Reaction(node_data['mapped_reactant'], mapped_product)
                new_edges.append(
                    (
                        node_name, canonical_product.__hash__(), reac.__hash__(),
                        {"reaction": reac}
                    )
                )

            self.network.nodes[node_name]["is_run"] = True
            
        #if self._new_nodes(new_nodes): # If no new nodes, don't add them.
        unique_new_nodes = self._new_nodes(new_nodes)
        self.network.add_nodes_from(unique_new_nodes)

        unique_edges = self._new_edges(new_edges) # removes duplicated edges.
        self.network.add_edges_from(unique_edges)

        if remove_self_loop:
            self.network.remove_edges_from(nx.selfloop_edges(self.network))

    def _new_nodes(self, tmp_new_nodes):
        old_nodes = list(self.network.nodes)
        
        new_nodes = []
        for node in tmp_new_nodes:
            if node[0] in old_nodes:
                continue
            new_nodes.append(node)
        return new_nodes

    def _new_edges(self, new_edges):
        """ Rmoves """
        old_edges = self.network.edges(data="reaction_hash")
        new_edges = [
            (prod, reac, key, data)
            for prod, reac, key, data in new_edges
            if (prod, reac, key) not in old_edges
        ]
        
        return new_edges

    def _mol_energy(self, fragments):
        """ """ 
        energy = 0
        for frag in fragments:
            frag_energy = self._get_fragment_energy(frag.__hash__())
            if np.isnan(frag_energy):
                energy = frag_energy
                break
            energy += frag_energy
        
        return energy

    def prune_nodes(self):
        """  """
        for start_node, end_node, edge_key, reaction in self.network.edges(
            data="reaction", keys=True
        ):  
            reac_frags = self.network.nodes[start_node]['canonical_reactant'].get_fragments()
            prod_frags = self.network.nodes[end_node]['canonical_reactant'].get_fragments()

            reac_energy = self._mol_energy(reac_frags)
            prod_energy = self._mol_energy(prod_frags)

            reaction_energy = prod_energy - reac_energy
            self.network[start_node][end_node][edge_key].update(
                reaction_energy=reaction_energy
            )

            # reac_energy = 0
            # for reac_frag in reac_frags:
            #     frag_energy = self._get_fragment_energy(reac_frag.__hash__())
            #     if np.isnan(frag_energy):
            #         reac_energy = frag_energy
            #         break
            #     reac_energy += frag_energy

            # prod_energy = 0
            # for prod_frag in prod_frags:
            #     frag_energy = self._get_fragment_energy(prod_frag.__hash__())
            #     if np.isnan(frag_energy):
            #         prod_energy = frag_energy
            #         break
            #     prod_energy += frag_energy

            # reaction_energy = prod_energy - reac_energy
            # self.network[start_node][end_node][edge_key].update(
            #     reaction_energy=reac_energy
            # )

        # Remove edges with to high reaction energy, and nodes without any reactions.
        edges_to_remove = []
        for edge in self.network.edges(keys=True):
            reac_energy = self.network.get_edge_data(*edge)["reaction_energy"]
            if np.isnan(reac_energy) or reac_energy > self._reaction_energy_cut_off:
                edges_to_remove.append(edge)

        self.network.remove_edges_from(edges_to_remove)
        self.network.remove_nodes_from(list(nx.isolates(self.network)))

    def get_unique_new_fragments(self, filename=None, overwrite=True):
        """ """ 
        fragments = []
        for node_name, reactant in self.network.nodes(data="canonical_reactant"):
            for fragment in reactant.get_fragments():
                if fragment.__hash__() not in self._fragment_energies:
                    fragments.append(fragment)
        unique_fragments = list(set(fragments))
        
        if os.path.exists(filename):
            if overwrite:
                with open(filename, 'wb') as _file:
                    pickle.dump(unique_fragments, _file)
        
        elif not os.path.exists(filename) and filename is not None:
            with open(filename, 'wb') as _file:
                    pickle.dump(unique_fragments, _file)

        return unique_fragments
    
    def load_fragment_energies(self, filename):
        """ """
        with open(filename, 'rb') as _file:
            fragments = pickle.load(_file)
        
        for fragment in fragments:
            min_conf_energy = 9999.9
            for conf in fragment._conformers:
                if conf._converged:
                    conf_energy = conf.results['energy']
                    if min_conf_energy > conf_energy:
                        min_conf_energy = conf_energy
            
            if min_conf_energy == 9999.9:
                min_conf_energy = np.float('nan')
            
            self._fragment_energies[fragment.__hash__()] = min_conf_energy * 627.503

    def node_from_hash(self, hash_key):
        return self.network.nodes[hash_key]

    def get_unique_reactions(self, filename=None, overwrite=True):
        """  """ 
        unique_reactions = []
        for node_in, node_out, reaction in self.network.edges(data='reaction'):
            if reaction.__hash__() not in self._unique_reactions:
                self._unique_reactions[reaction.__hash__()] = reaction
                unique_reactions.append(reaction)
        
        if os.path.exists(filename):
            if overwrite:
                with open(filename, 'wb') as _file:
                    pickle.dump(unique_reactions, _file)
        
        elif not os.path.exists(filename) and filename is not None:
            with open(filename, 'wb') as _file:
                    pickle.dump(unique_reactions, _file)

        return unique_reactions

    def load_reaction_energies(self, filename=None): 
        """ """
        with open(filename, 'rb') as inp:
            output_reactions = pickle.load(inp)

        computed_reactions = dict()
        for reaction in output_reactions:
            computed_reactions[reaction.__hash__()] = reaction

        for node_in, node_out, reaction in self.network.edges(data="reaction"):
            reaction_hash = reaction.__hash__()
            edge_data = self.network[node_in][node_out][reaction_hash]
            
            new_reaction = computed_reactions[reaction_hash]
            reactant = self.network.nodes[node_in]['mapped_reactant']

            reactant_E = self._mol_energy(reactant.get_fragments())
            edge_data['reaction'] = new_reaction
            edge_data['barrier_energy'] = new_reaction.ts_energy - reactant_E

    def prune_reactions(self, remove_isolates=True):
        """ """
        edges_to_remove = []
        for node_in, node_out, data in self.network.edges(data=True):
            reaction_hash = data['reaction'].__hash__()
            barier_energy = data['barrier_energy']
            if barier_energy > self._barrier_cutoff or np.isnan(barier_energy):
                edges_to_remove.append((node_in, node_out, reaction_hash))
        self.network.remove_edges_from(edges_to_remove)

        if remove_isolates:
            self.network.remove_nodes_from(list(nx.isolates(self.network)))

    def save(self, filename):
        """save class as self.name.txt"""
        with open(filename, 'wb') as _file:
            pickle.dump(self.__dict__, _file)
    
    @classmethod
    def load(cls, filename):
        """save class as self.name.txt"""
        obj = cls( )
        with open(filename, 'rb') as _file:
            tmp_dict = pickle.load(_file)
        obj.__dict__.update(tmp_dict)
        return obj


class ReactionCell:
    """ """
    def __init__(self, reacting_mol=None, solvent=None, max_bonds=2,
                 max_chemical_dist=4):

        self.reacting_mol = reacting_mol
        self.solvent = solvent

        self._max_bonds = max_bonds
        self._max_cd = max_chemical_dist

        self._reactions = []
        self._products = []

        self.unique_fragments = []

    def get_reactions(self, get_unique_fragments=True,
                 generator=False, nprocs=1):
        """
        Enumerates all possible products where the combinations
        of `max_bonds` are broken/formed. However the maximum
        formed/broken bonds are `CD`.
        """

        # TODO: If we have a transition metal add the remaining non active
        # solvent molecules.

        mapped_products = valid_products(self.reacting_mol.molecule, n=self._max_bonds,
                                    cd=self._max_cd,
                                    charge=Chem.GetFormalCharge(self.reacting_mol.molecule),
                                    n_procs=nprocs)
        
        #for mapped_product in products:
        #    prod_mapped_molecule = Molecule.from_rdkit_mol(mapped_product, pseudo_chiral=True)
        #    self._reactions.append( Reaction(self.reacting_mol, prod_mapped_molecule))
        #    self._products.append(Molecule.from_rdkit_mol(mapped_product, pseudo_chiral=False))
        
        #return [self._products, list(set(self._reactions))

    def get_fragments(self, ncpus=-1, remove_atomMapNum=True):
        """
        Return .csv file with fragments and the corresponding charge
        """
        if len(self.reactions) == 0:
            raise RuntimeError('No products created yet. Run .shake() .')

        # find fragment label number
        if len(self.unique_fragments) == 0:
            frag_num = 0
        else:
            frag_num = int(self.unique_fragments[-1].label.split('-')[0])

        fragments = Parallel(n_jobs=ncpus)(delayed(
            self._get_fragments_helper)(mol) for mol in self.reactions
            )
        self.unique_fragments = list(set(chain.from_iterable(fragments)))

        n = 0
        for frag in self.unique_fragments:
            frag.label =  f'fragment-{n}'
            n += 1
    
    @staticmethod
    def _get_fragments_helper(mol):
        """
        """
        frags = []
        for frag in Chem.GetMolFrags(mol.molecule, asMols=True,
                                     sanitizeFrags=False):
            frags.append(Fragment.from_rdkit_mol(frag))
        return frags