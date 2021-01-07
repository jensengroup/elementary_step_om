import networkx as nx
import pickle
from collections import defaultdict
import numpy as np

from .compound import ReactionCell, Fragment

class ReactionNetwork:
    def __init__(
        self,
        reagents=None,
        solvent=None,
        max_bonds=2,
        max_chemical_dist=4,
        reaction_energy_cut_off=30.0,
    ):

        self._reagents = reagents
        self._solvent = solvent

        self._max_bonds = max_bonds
        self._max_cd = max_chemical_dist
        self._reaction_energy_cut_off = reaction_energy_cut_off

        self._fragment_energies = dict()

        self.network = None

        if self._reagents is not None:
            self._initialize_network()

    def _initialize_network(self):
        """ """
        reaction_cell = ReactionCell(
            reagents=self._reagents,
            solvent=self._solvent,
            max_bonds=self._max_bonds,
            max_chemical_dist=self._max_cd,
        )

        self.network = nx.MultiDiGraph()
        self.network.add_node(
            reaction_cell._reacting_mol.__hash__(), reagent=self._reagents, is_run=False
        )

    def fragment_energies(self, mol_hash):
        """ """
        if mol_hash in self._fragment_energies:
            return self._fragment_energies[mol_hash]
        return np.float('nan')

    def take_step(self, nprocs=6, remove_self_loop=True):
        """
        Expands the network with an eksta layer.

        N.B. remember to prune the reaction network
        before you take an ekstra step.
        """
        # check that all this is working as expected.
        def filter_not_run(n1):
            return self.network.nodes[n1]["is_run"] == False

        not_run_nodes = nx.subgraph_view(self.network, filter_node=filter_not_run)
        new_edges = []
        new_nodes = []
        for node_name, node_data in not_run_nodes.nodes(data=True):
            node_reagent = node_data["reagent"]
            reaction_cell = ReactionCell(
                reagents=node_reagent,
                solvent=self._solvent,
                max_bonds=self._max_bonds,
                max_chemical_dist=self._max_cd,
            )

            products, reactions = reaction_cell.reactions(nprocs=1)
            new_nodes.append(
                [
                    (prod.__hash__(), {"reagent": [prod], "is_run": False})
                    for prod in products
                ]
            )
            new_edges.append(
                [
                    (
                        reac.reactant.__hash__(),
                        reac.product.__hash__(),
                        {"reaction": reac, "reac_hash": reac.__hash__()},
                    )
                    for reac in reactions
                ]
            )

            node_data["is_run"] = True

        for to_nodes in new_nodes:
            if self._new_nodes(to_nodes):
                self.network.add_nodes_from(to_nodes)

        for edges in new_edges:
            edges = self._new_edges(edges)
            self.network.add_edges_from(edges)

        if remove_self_loop:
            self.network.remove_edges_from(nx.selfloop_edges(self.network))

    def _new_nodes(self, new_nodes):
        old_nodes = set(self.network.nodes())
        new_nodes = set([node_name for node_name, _ in new_nodes])
        if len(new_nodes - old_nodes) == 0:
            return False
        return True

    def _new_edges(self, new_edges):
        """ Rmoves """
        old_edges = self.network.edges(data="reac_hash")
        new_edges = [
            (prod, reac, data)
            for prod, reac, data in new_edges
            if (prod, reac, data["reac_hash"]) not in old_edges
        ]
        return new_edges

    def prune_nodes(self):
        """  """
        for start_node, end_node, edge_key, reaction in self.network.edges(
            data="reaction", keys=True
        ):
            reac_frags, prod_frags = reaction.get_fragments()

            reac_energy = 0
            for reac_frag in reac_frags:
                frag_energy = self.fragment_energies(reac_frag.__hash__())
                if np.isnan(frag_energy):
                    reac_energy = frag_energy
                    break
                reac_energy += frag_energy

            prod_energy = 0
            for prod_frag in prod_frags:
                frag_energy = self.fragment_energies(prod_frag.__hash__())
                if np.isnan(frag_energy):
                    prod_energy = frag_energy
                    break
                prod_energy += frag_energy

            reac_energy = prod_energy - reac_energy
            self.network[start_node][end_node][edge_key].update(
                reaction_energy=reac_energy
            )

        # Remove edges with to high reaction energy, and nodes without any reactions.
        edges_to_remove = []
        for edge in self.network.edges(keys=True):
            reac_energy = self.network.get_edge_data(*edge)["reaction_energy"]
            if np.isnan(reac_energy) or reac_energy > self._reaction_energy_cut_off:
                edges_to_remove.append(edge)

        self.network.remove_edges_from(edges_to_remove)
        self.network.remove_nodes_from(list(nx.isolates(self.network)))

    def get_unique_new_fragments(self, filename=None):
        """ """ 
        fragments = []
        for node_name, node_data in self.network.nodes(data=True):
            for reagent in node_data['reagent']:
                for fragment in reagent.get_fragments():
                    if fragment.__hash__() not in self._fragment_energies:
                        fragments.append(fragment)
        unique_fragments = list(set(fragments))
        
        if filename is not None:
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

    def node_from_hash(self, h):
        return self.network.nodes[h]

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