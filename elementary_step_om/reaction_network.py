import os
import networkx as nx
import pickle
from networkx.classes.function import all_neighbors
import numpy as np

from rdkit.Chem import rdmolops

from elementary_step_om.elementary_step import TakeElementaryStep
from .chem import MappedMolecule


class ReactionNetwork:
    def __init__(self, reactant: MappedMolecule = None, charge: int = 0, spin: int = 1):

        self._initial_reactant = reactant

        self._spin = spin
        self._charge = rdmolops.GetFormalCharge(self._initial_reactant.rd_mol)

        self._fragment_energies = dict()
        self._unique_reactions = dict()
        self._checked_reactions = dict()

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
    
    def get_reaction_hash(self, reaction):
        """ Same hash going from reactant to product. """
        reaction_hash = reaction.reactant.__hash__()
        product_hash = reaction.product.__hash__()
        return hash(frozenset([reaction_hash, product_hash]))

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
            
            if "ts_ok" not in edge_data:
                return False

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

    def get_all_attibues(self):        
        all_atributes = set()
        for _, _, data in self._network.edges(data=True):
            for att in data.keys():
                all_atributes.add(att)
        return all_atributes
    
    def network_view(self, ekstra_filters=None):
        """
        Removes all paths not completely computed, and paths that
        doesn't go thorough the filters.
        """
        all_atributes = self.get_all_attibues()

        def filter_network_view(nin, nout, key):
            edge_data = self._network[nin][nout][key]

            # barrier is only computed for some paths.
            if all_atributes.issubset(set(edge_data)):
                return True
            return False

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
        inactive_atoms = [],
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
                inactive_atoms = inactive_atoms
            )
            nodes_to_step.append(take_step)
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
            reaction_set_hash = self.get_reaction_hash(reaction)
            
            if reaction_set_hash not in self._unique_reactions:
                self._unique_reactions[reaction_set_hash] = None 
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
                reaction_set_hash = self.get_reaction_hash(reaction)                
                ts_energy = min(reaction.ts_guess_energies, default=np.float("nan"))
                self._unique_reactions[reaction_set_hash] = ts_energy

        # Compute barrier energy.
        for node_in, _, edge_data in self._network.edges(data=True):
            new_reaction = edge_data['reaction']
            reaction_set_hash = self.get_reaction_hash(new_reaction)   

            if reaction_set_hash in  self._unique_reactions.keys():
                ts_energy = self._unique_reactions[reaction_set_hash]
                if ts_energy is None:
                    edge_data["barrier_energy"] = None
                    continue
            else:
                edge_data["barrier_energy"] = None
                continue
            
            reactant = self._network.nodes[node_in]["canonical_reactant"]
            reactant_E = self._mol_energy(reactant.get_fragments())
            edge_data["reaction"] = new_reaction
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
            edge_data = self._network[nin][nout][key]
            barrier_energy = edge_data["barrier_energy"]
            
            # Barrier energy is not computed, no need to check
            if barrier_energy is None:
                return False
            
            # Check if barrier energy is < max_barrier_energy
            ok_barrier = edge_data["barrier_energy"] < max_barrier_energy

            # and the reactions is not a "back reaction"
            reaction_hash = self.get_reaction_hash(edge_data['reaction'])
            not_checked = reaction_hash not in self._checked_reactions.keys()

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

        for out_reaction in output_reactions:
            reaction_hash = self.get_reaction_hash(out_reaction)
            self._checked_reactions[reaction_hash] = out_reaction

        for _, _, data in self._network.edges(data=True):
            reaction_hash = self.get_reaction_hash(data['reaction'])
            if reaction_hash in self._checked_reactions:
                data["ts_ok"] = False
                reaction_check_status = self._checked_reactions[reaction_hash].ts_check
                if reaction_check_status is not None:
                    data["ts_ok"] = reaction_check_status

    def _prune_nan_ts_check(self):
        """
        Prune computed reactions, where no path converged or no
        path connects to the reactant or product.
        """

        def filter_non_ts(nin, nout, key):
            reaction_edge = self._network[nin][nout][key]

            # Did you compute a barrier
            if "ts_ok" not in reaction_edge:
                return True
            
            # Did any path converge?
            if reaction_edge['ts_ok'] is False:
                return False

            if len(reaction_edge['ts_ok']) == 0:
                return False
            
            # Check if one path terminates in either reactant or product
            one_true = False
            for check in self._network[nin][nout][key]["ts_ok"]:
                if any(check.values()):
                    one_true = True
                    break
            if one_true is False:
                return False

            return True

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
