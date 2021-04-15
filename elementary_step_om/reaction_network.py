#from copy import deepcopy
import os
import networkx as nx
import pickle
#from networkx.algorithms import node_classification
import numpy as np
#from collections import defaultdict
#import hashlib

#from rdkit import Chem
#from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem import rdmolops

from elementary_step_om.elementary_step import TakeElementaryStep
from .chem import MappedMolecule, Reaction


class ReactionNetwork:
    def __init__(self, reactant: MappedMolecule = None, charge: int = 0, spin: int = 1):

        self._initial_reactant = reactant

        self._spin = spin
        self._charge = rdmolops.GetFormalCharge(self._initial_reactant.rd_mol)

        self._fragment_energies = dict()
        self._unique_reactions = dict()

        self._initialize_network()

        print(f">> initializing: charge={self._charge} and spin(2S+1)={self._spin}")

    def _initialize_network(self):
        """"""
        canonical_mol = self._initial_reactant.get_unmapped_molecule()

        self._network = nx.MultiDiGraph()
        self._network.add_node(
            canonical_mol.__hash__(),
            canonical_reactant=canonical_mol,
            mapped_reactant=self._initial_reactant,
            is_run=False,
        )

    def _get_fragment_energy(self, mol_hash):
        """ """
        if mol_hash in self._fragment_energies:
            return self._fragment_energies[mol_hash]
        return np.float("nan")

    def get_energy_filter(self, max_reaction_energy=30.0, max_barrier=None):
        def energy_filter(nin, nout, key):
            edge_data = self._network[nin][nout][key]
            if edge_data['reaction_energy'] <= max_reaction_energy:
                return True
            
            if max_barrier is not None:
                if edge_data['barrier_energy'] <= max_barrier:
                    return True
            return False
        return energy_filter
    
    
    def get_ts_check_filter(self,
        check_reactant = True, check_product = True, reactant_or_product = False
    ):
        def check_filter(nin, nout, key):
            edge_data = self._network[nin][nout][key]
            
            for check in edge_data['ts_ok']:
                reac_check, prod_check = check.values()

                if check_reactant and not check_product: # Check reactant
                    if reac_check is True:
                        return True
                elif check_product and not check_reactant: # Check product
                    if prod_check is True:
                        return False
                elif check_reactant and check_product: # Check reactant and product
                    if reac_check and prod_check:
                        return True
                elif reactant_or_product: # Check reactant or product
                    if reac_check or prod_check:
                        return True
            return False         
        return check_filter

    def network_view(self, ekstra_filters=None):
        """
        Removes all paths not completely computed, and paths that
        doesn't go thorough the filters.
        """
        def filter_network_view(nin, nout, key):
            edge_data = self._network[nin][nout][key]
            # barrier is only computed for some paths.
            if set(['reaction', 'reaction_energy', 'barrier_energy']).issubset(set(edge_data)):
                if edge_data['barrier_energy'] is None:
                    return False
            return True

        tmp = nx.subgraph_view(self._network, filter_edge=filter_network_view).copy()
        if ekstra_filters is not None:
            if not isinstance(ekstra_filters, list):
                ekstra_filters = [ekstra_filters]
            
            for ekstra_filter in ekstra_filters:
                if ekstra_filter is None:
                    continue
                tmp = nx.subgraph_view(tmp, filter_edge=ekstra_filter).copy()
        
        if len(self._network) != 1:
            tmp.remove_nodes_from(list(nx.isolates(tmp)))

        return tmp

    def create_node_subnets(
        self,
        filename: str,
        max_bonds: int = 2,
        max_cd: int = 4,
        energy_filter = None,
        ts_check_filter = None
    ):
        """
        Saves file with take_step classes for each nodes that isn't run.
        """
        def filter_not_run(node_name):
            return self._network.nodes[node_name]["is_run"] == False

        nodes_to_run = self.network_view(ekstra_filters=[energy_filter, ts_check_filter]).copy()
        nodes_to_run = nx.subgraph_view(nodes_to_run, filter_node=filter_not_run).copy()
        nodes_to_step = []
        for node_name, node_data in nodes_to_run.nodes(data=True):
            take_step = TakeElementaryStep(
                mapped_molecule=node_data["mapped_reactant"],
                max_num_bonds=max_bonds,
                cd=max_cd,
            )
            nodes_to_step.append(take_step)
            #node_data['is_run'] = None
            self._network.nodes[node_name]['is_run'] = None
        
        print(f">> # nodes to run: {len(nodes_to_step)}")
        with open(filename, "wb") as step_files:
            pickle.dump(nodes_to_step, step_files)

    def load_subnets(self, filename):
        """
        """
        with open(filename, "rb") as out_file:
            subnets = pickle.load(out_file)

        for i, subnet in enumerate(subnets):
            for node, data in subnet.nodes(data=True):
                if node in self._network.nodes:
                    data.clear()

        self._network = nx.compose_all([self._network] + subnets)

        # Update is run
        def import_run(node):
            return self._network.nodes[node]['is_run'] is None

        for node in nx.subgraph_view(self._network, filter_node=import_run).nodes:
             self._network.nodes[node]['is_run'] = True
 
    def get_unique_fragments(self, filename, overwrite=True):
        """ """
        fragments = []
        for _, unmapped_reactant in self._network.nodes(data="canonical_reactant"):
            for fragment in unmapped_reactant.get_fragments():
                if fragment.__hash__() not in self._fragment_energies:
                    fragments.append(fragment)
        unique_fragments = list(set(fragments))

        if os.path.exists(filename):
            if overwrite:
                with open(filename, "wb") as _file:
                    pickle.dump(unique_fragments, _file)
        else:
            with open(filename, "wb") as _file:
                pickle.dump(unique_fragments, _file)

        return unique_fragments

    def load_fragment_energies(self, filename):
        """ """
        with open(filename, "rb") as _file:
            fragments = pickle.load(_file)

        for fragment in fragments:
            min_conf_energy = 9999.9
            for conf in fragment.conformers:
                if conf.results["converged"]:
                    conf_energy = conf.results["energy"]
                    if min_conf_energy > conf_energy:
                        min_conf_energy = conf_energy

            if min_conf_energy == 9999.9:
                min_conf_energy = np.float("nan")

            self._fragment_energies[fragment.__hash__()] = min_conf_energy * 627.503

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

    def _prune_nan_reaction_energies(self):
        """
        Compute reaction energies, and remove Nan values from network.
        """
        for in_node, out_node, edge_key in self._network.edges(keys=True):
            reac_frags = self._network.nodes[in_node][
                "canonical_reactant"
            ].get_fragments()
            prod_frags = self._network.nodes[out_node][
                "canonical_reactant"
            ].get_fragments()

            reac_energy = self._mol_energy(reac_frags)
            prod_energy = self._mol_energy(prod_frags)

            reaction_energy = prod_energy - reac_energy
            self._network[in_node][out_node][edge_key].update(
                reaction_energy=reaction_energy
            )

        # Remove nan reaction energies.
        def filter_nan_energies(nin, nout, key):
            return not np.isnan(self._network[nin][nout][key]["reaction_energy"])

        self._network = nx.subgraph_view(
            self._network, filter_edge=filter_nan_energies
        ).copy()
        self._network.remove_nodes_from(list(nx.isolates(self._network)))

    #TODO: max_reaction_energy change to filter.
    def get_unique_reactions(
        self, filename: str, max_reaction_energy: float = 30.0, overwrite: bool = True
    ):
        """"""

        def filter_energies(nin, nout, key):
            """Get reactions with reaction energy < max_energy and
            no barrier energy computed.
            """
            ok_energy = (
                self._network[nin][nout][key]["reaction_energy"] <= max_reaction_energy
            )
            barrier_ok = not "barrier_energy" in self._network[nin][nout][key]
            return all([ok_energy, barrier_ok])

        unique_reactions = []
        subgraph = nx.subgraph_view(self._network, filter_edge=filter_energies)
        for _, _, reaction in subgraph.edges(data="reaction"):
            reactant_hash = reaction.reactant.__hash__()
            product_hash = reaction.product.__hash__() 

            # Same hash going from reactant to product.
            reaction_set_hash = hash(frozenset([reactant_hash, product_hash]))
            
            if reaction_set_hash not in self._unique_reactions:
                self._unique_reactions[reaction_set_hash] = reaction
                unique_reactions.append(reaction)

        if os.path.exists(filename) and not overwrite:
            pass
        else:
            with open(filename, "wb") as _file:
                pickle.dump(unique_reactions, _file)

        return unique_reactions

    def load_reaction_energies(self, filename):
        """
        """

        # Load reaction data
        computed_reactions = dict()
        with open(filename, "rb") as inp:
            output_reactions = pickle.load(inp)
            for reaction in output_reactions:
                reactant_hash = reaction.reactant.__hash__()
                product_hash = reaction.product.__hash__() 

                # Same hash going from reactant to product.
                reaction_set_hash = hash(frozenset([reactant_hash, product_hash]))
                self._unique_reactions[reaction_set_hash] = reaction

        # Compute barrier energy.
        for node_in, _, edge_data in self._network.edges(data=True):
            reactant_hash = reaction.reactant.__hash__()
            product_hash = reaction.product.__hash__() 

            # Same hash going from reactant to product.
            reaction_set_hash = hash(frozenset([reactant_hash, product_hash]))
            if (
                reaction_set_hash in  self._unique_reactions.keys()
            ):  # This shouldn't be computed but input to.
                new_reaction = self._unique_reactions[reaction_set_hash]
            else:
                edge_data["barrier_energy"] = None
                continue

            reactant = self._network.nodes[node_in]["canonical_reactant"]
            reactant_E = self._mol_energy(reactant.get_fragments())
            edge_data["reaction"] = new_reaction
            ts_energy = min(new_reaction.ts_guess_energies, default=np.float("nan"))
            edge_data["barrier_energy"] = ts_energy - reactant_E

    def _prune_nan_reaction_paths(self):
        """ """

        def filter_nan_energies(nin, nout, key):
            barrier_energy = self._network[nin][nout][key]["barrier_energy"]
            if barrier_energy is None:
                return True
            return not np.isnan(self._network[nin][nout][key]["barrier_energy"])

        self._network = nx.subgraph_view(
            self._network, filter_edge=filter_nan_energies
        ).copy()
        self._network.remove_nodes_from(list(nx.isolates(self._network)))

    def get_reactions_to_check(self, filename, max_barrier_energy=50.0, overwrite=True):
        """"""
        reactions_to_check = []

        def filter_is_checked(nin, nout, key):
            barrier_energy = self._network[nin][nout][key]["barrier_energy"]
            if barrier_energy is None:
                return False
            ok_barrier = barrier_energy <= max_barrier_energy
            not_checked = not "ts_ok" in self._network[nin][nout][key]
            return all([ok_barrier, not_checked])

        subgraph = nx.subgraph_view(self._network, filter_edge=filter_is_checked)
        reactions_to_check = [reac for _, _, reac in subgraph.edges(data="reaction")]

        if os.path.exists(filename) and not overwrite:
            pass
        else:
            with open(filename, "wb") as _file:
                pickle.dump(reactions_to_check, _file)

        return reactions_to_check

    def load_reaction_after_check(self, filename):
        """"""
        with open(filename, "rb") as inp:
            output_reactions = pickle.load(inp)

        new_reactions = {}
        for out_reaction in output_reactions:
            new_reactions[out_reaction.__hash__()] = out_reaction

        for _, _, key, data in self._network.edges(keys=True, data=True):
            data["ts_ok"] = []
            try:
                data["reaction"] = new_reactions[key]
            except:
                continue

            # load check data
            if data["reaction"].ts_check is None:
                continue

            data["ts_ok"] = data["reaction"].ts_check

    def _prune_nan_ts_check(self):
        """
        Prune computed reactions, where no path converged or no
        path connects to the reactant or product.
        """

        def filter_non_ts(nin, nout, key):
            # Did you compute a barrier
            if self._network[nin][nout][key]["barrier_energy"] is None:
                return True

            # Did any path converge?
            ts_not_ok = len(self._network[nin][nout][key]["ts_ok"]) != 0

            # Check if one path terminates in either reactant or product
            one_true = False
            for check in self._network[nin][nout][key]["ts_ok"]:
                if any(check.values()):
                    one_true = True
                    break
            return all([ts_not_ok, one_true])

        self._network = nx.subgraph_view(self._network, filter_edge=filter_non_ts).copy()
        self._network.remove_nodes_from(list(nx.isolates(self._network))) 

    def save(self, filename):
        """save class as self.name.txt"""
        with open(filename, "wb") as _file:
            pickle.dump(self.__dict__, _file)

    @classmethod
    def load(cls, filename):
        """save class as self.name.txt"""
        obj = cls()
        with open(filename, "rb") as _file:
            tmp_dict = pickle.load(_file)
        obj.__dict__.update(tmp_dict)
        return obj
