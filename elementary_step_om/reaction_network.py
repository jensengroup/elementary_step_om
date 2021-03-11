import os
import networkx as nx
import pickle
import numpy as np
from collections import defaultdict 

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from .elementary_step import valid_products
from .chem import (
    MappedMolecule, 
    Reaction
)

class ReactionNetwork:
    def __init__(
        self,
        reagents=None,
        solvent=None,
        spin: int = 1,
        max_bonds=2,
        max_chemical_dist=4,
        reaction_energy_cut_off=30.0,
        barrier_cutoff=50.0
    ):

        self.reagents = reagents
        self.solvent = solvent
        self._spin = spin

        self._max_bonds = max_bonds
        self._max_cd = max_chemical_dist
        self._reaction_energy_cut_off = reaction_energy_cut_off
        self._barrier_cutoff = barrier_cutoff

        self._fragment_energies = dict()
        self._unique_reactions = dict()
        self._num_reacs = 0

        self.network = None

        if self.reagents is not None:
            self._initialize_network()

    def _initialize_network(self):
        """ """
        mapped_reactant = self._prepare_reacting_mol(self.reagents)
        mapped_reactant.label = 'initial_mapped_reactant'

        canonical_mol = mapped_reactant.get_unmapped_molecule(label='initial_reactant')
        self._charge = rdmolops.GetFormalCharge(canonical_mol.rd_mol)
        print(f">> initializing: charge={self._charge} and spin(2S+1)={self._spin}")

        self.network = nx.MultiDiGraph()
        self.network.add_node(
            canonical_mol.__hash__(), 
            canonical_reactant=canonical_mol,
            mapped_reactant=mapped_reactant,
            is_run=False
        )

    def _prepare_reacting_mol(self, reagents):
        """ Combine reagents and X active atoms into RDKit Mol 
        which is the reacting molecule.
        """
        # Add Reagents
        if len(reagents) == 1:
            reactants = reagents[0].rd_mol
        else:
            reactants = Chem.CombineMols(reagents[0].rd_mol, reagents[1].rd_mol)
            if len(reagents) > 2:
                for reag in reagents[2:]:
                    reactants = Chem.CombineMols(reactants, reag.rd_mol)

        # Add active solvent molecules
        if self.solvent is not None:
            for _ in range(self.solvent._nactive):
                sol_mol = Chem.AddHs(Chem.MolFromSmiles(self.solvent._smiles))
                reactants = Chem.CombineMols(reactants, sol_mol)
            AllChem.Compute2DCoords(reactants)

        # Atom mapping, and set random chirality.
        return MappedMolecule.from_rdkit_mol(reactants)

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
            mapped_products = valid_products(
                    node_data['mapped_reactant'].rd_mol,
                    n=self._max_bonds,
                    cd=self._max_cd,
                    charge=Chem.GetFormalCharge(node_data['mapped_reactant'].rd_mol),
                    n_procs=nprocs
                )
            
            for mapped_product in mapped_products:
                mapped_product = MappedMolecule.from_rdkit_mol(mapped_product)
                canonical_product = mapped_product.get_unmapped_molecule()
                new_nodes.append(
                    (
                        canonical_product.__hash__(), 
                        {"canonical_reactant": canonical_product,
                         "mapped_reactant": mapped_product,
                         "is_run": False
                         }
                    )
                )

                reac = Reaction(
                    reactant=node_data['mapped_reactant'],
                    product=mapped_product,
                    charge=self._charge,
                    spin=self._spin,
                    label=f"reaction-{self._num_reacs}"
                )

                new_edges.append(
                    (
                        node_name, canonical_product.__hash__(), reac.__hash__(),
                        {"reaction": reac}
                    )
                )

                self._num_reacs += 1

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
        for node_name, unmapped_reactant in self.network.nodes(data="canonical_reactant"):
            for fragment in unmapped_reactant.get_fragments():
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
            for conf in fragment.conformers:
                if conf.results['converged']:
                    conf_energy = conf.results['energy']
                    if min_conf_energy > conf_energy:
                        min_conf_energy = conf_energy
            
            if min_conf_energy == 9999.9:
                min_conf_energy = np.float('nan')
            
            self._fragment_energies[fragment.__hash__()] = min_conf_energy * 627.503

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
        
        # if path crashed (no reaction i.e. missing hash) return None
        computed_reactions = defaultdict(lambda : None) 
        for reaction in output_reactions:
            computed_reactions[reaction.__hash__()] = reaction

        for node_in, node_out, reaction in self.network.edges(data="reaction"):
            reaction_hash = reaction.__hash__()
            edge_data = self.network[node_in][node_out][reaction_hash]
            
            new_reaction = computed_reactions[reaction_hash]
            # If reaction crashed
            if new_reaction is None: 
                edge_data['barrier_energy'] = np.float('nan')
                continue

            reactant = self.network.nodes[node_in]['canonical_reactant']
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
        