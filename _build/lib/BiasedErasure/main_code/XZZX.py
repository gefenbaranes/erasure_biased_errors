from BiasedErasure.main_code.StabilizerCode import StabilizerCode
from BiasedErasure.main_code.xzzx_circuit import xzzx_circuit, data_ids, ancilla_ids, get_data_ancilla_indices
from BiasedErasure.main_code.xzzx_circuit_noisy import xzzx_circuit_noisy
from BiasedErasure.main_code.xzzx_cluster import xzzx_cluster, cluster_data_ids, get_neighbors_next_previous_layers, layer_offset, cluster_qubits_ids, get_num_layers
from BiasedErasure.main_code.xzzx_cluster_noisy import xzzx_cluster_noisy
import numpy as np
from BiasedErasure.main_code.noise_channels import atom_array
# from xzzx_cluster_noisy import xzzx_cluster, cluster_data_ids, get_neighbors_next_previous_layers, layer_offset, cluster_qubits_ids


class XZZX(StabilizerCode):
    def __init__(self, **kwargs):
        self.stabilizers = self.get_stabilizers()

    def get_stabilizers(self):
        # TODO: define these.
        pass
    
    def get_data_ancilla_indices(self, dx, dy, cycles, architecture):
        if architecture == 'MBQC':
            num_layers = get_num_layers(cycles)
            data_qubits, ancilla_qubits = cluster_qubits_ids(dx=dx, dy=dy, num_layers=num_layers)
            return data_qubits,  ancilla_qubits
        elif architecture == 'CBQC':
            data_qubits, ancilla_qubits = data_ids(dx,dy), ancilla_ids(dx,dy)
            return data_qubits, ancilla_qubits
        

    def get_neighbors(self, qbt, dx, dy):
        # Get neighbors of qubit qbt
        # Transform qbt to (i, j) coordinate.
        ValueError('architecture' == 'CBQC')
        alpha = 2*dx - 1
        j = qbt // alpha
        i = qbt % alpha

        neighbors = []
        if j != 2 * dy - 2:
            neighbors.append(int(((j+1)*(2*dx-1)+i)))  # up
        if i != 2 * dx - 2:
            neighbors.append(int((j*(2*dx-1)+(i+1))))  # right
        if i != 0:
            neighbors.append(int((j*(2*dx-1)+(i - 1))))  # left
        if j != 0:
            neighbors.append(int(((j-1)*(2*dx-1)+i)))  # down
        return neighbors
    
    def get_ordered_neighbors(self, qbt: int, dx: int, dy: int):
        # Get neighbors of qubit qbt in the form: {neighbor1: order1, ...}
        # where order_i specifies the time step that qbt interetacts with  neighbor_i.

        # Transform qbt to (i, j) coordinate.
        ValueError('architecture' == 'CBQC')
        alpha = 2*dx - 1
        j = qbt // alpha
        i = qbt % alpha
        neighbors = {}
        if j != 2 * dy - 2:
            neighbors[int(((j+1)*(2*dx-1)+i))] = 1  # up
        if i != 2 * dx - 2:
            neighbors[int((j*(2*dx-1)+(i+1)))] = 2  # right
        if i != 0:
            neighbors[int((j*(2*dx-1)+(i - 1)))] = 3  # left
        if j != 0:
            neighbors[int(((j-1)*(2*dx-1)+i))] = 4  # down
        return neighbors
    
    def get_order(self, direction: str):
        ordering_dict = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
        return ordering_dict[direction]
        # order: up, right, left, down
        
    def get_neighbors_CZ(self, qbt, dx, dy, cycles:int = None, architecture='CBQC', gates_dicts = {}):
        # Get neighbors of qubit qbt
        # CZ neighbors from up and down (in MBQC, only in even layers and also from next and previous layers)
        
        if architecture == 'CBQC':
            # for CBQC we return a dictionary, for every qubit index (key), the order of the gate (value).
            neighbors = {1: [], 2: [], 3:[], 4:[]}
            # Transform qbt to (i, j) coordinate.
            alpha = 2*dx - 1
            j = qbt // alpha
            i = qbt % alpha
            if j != 2 * dy - 2:
                ordering_key = self.get_order('up')
                neighbors[ordering_key].append(int(((j+1)*(2*dx-1)+i)))  # up
            if j != 0:
                ordering_key = self.get_order('down')
                neighbors[ordering_key].append(int(((j-1)*(2*dx-1)+i)))  # down
        elif architecture == 'MBQC': # add neighbors from previous and next layer
            num_layers = get_num_layers(cycles)
            potential_neighbors = []
            m = int(0.5*(2*dx-1)*(2*dy-1))  # num ancilla qubits in a surface code
            n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data qubits in a layer
            single_layer_offset = m + n
            layer_ix = qbt // single_layer_offset
            offset = layer_offset(dx=dx, dy=dy, layer_ix=layer_ix, physical=False)
            alpha = 2*dx - 1
            j = (qbt - offset) // alpha
            i = (qbt - offset) % alpha
            ValueError (qbt != j*(2*dx-1)+i+offset)
            if j != 2 * dy - 2: # north
                potential_neighbors.append((j+1)*(2*dx-1)+i+offset)
            if j != 0: # south
                potential_neighbors.append((j-1)*(2*dx-1)+i+offset)
            # other_layers_neighbors = get_neighbors_next_previous_layers(dx=dx, dy=dy, qubit_index = qbt)
            # for neighbor in other_layers_neighbors:
            #     potential_neighbors.append(neighbor)
            data_qubits, ancilla_qubits = cluster_qubits_ids(dx = dx, dy = dy, num_layers=num_layers)
            all_qubits_indices = data_qubits + ancilla_qubits
            neighbors = list(set(potential_neighbors) & set(all_qubits_indices))

        return neighbors

    def get_neighbors_CX(self, qbt, dx, dy, cycles: int = None, architecture='CBQC', gates_dicts={}):
        # Get neighbors of qubit qbt
        # CX from right and left
        if architecture == 'CBQC':
            # for CBQC we return a dictionary, for every qubit index (key), the order of the gate (value).
            neighbors = {1: [], 2: [], 3:[], 4:[]}
            # Transform qbt to (i, j) coordinate.
            alpha = 2*dx - 1
            j = qbt // alpha
            i = qbt % alpha
            if i != 2 * dx - 2:
                ordering_key = self.get_order('right')
                neighbors[ordering_key].append(int((j*(2*dx-1)+(i+1))))  # right
            if i != 0:
                ordering_key = self.get_order('left')
                neighbors[ordering_key].append(int((j*(2*dx-1)+(i - 1))))  # left
        elif architecture == 'MBQC': # add neighbors from previous and next layer
            num_layers = get_num_layers(cycles)
            potential_neighbors = []
            m = int(0.5*(2*dx-1)*(2*dy-1))  # num ancilla qubits in a surface code
            n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data qubits in a layer
            single_layer_offset = m + n
            layer_ix = qbt // single_layer_offset
            offset = layer_offset(dx=dx, dy=dy, layer_ix=layer_ix, physical=False)
            alpha = 2*dx - 1
            j = (qbt - offset) // alpha
            i = (qbt - offset) % alpha
            ValueError (qbt != j*(2*dx-1)+i+offset)
            if i != 2 * dx - 2: # east
                potential_neighbors.append(j*(2*dx-1)+(i+1)+offset)
            if i != 0: # west
                potential_neighbors.append(j*(2*dx-1)+(i - 1)+offset)            
            # new addition - other layers:
            other_layers_neighbors = get_neighbors_next_previous_layers(dx=dx, dy=dy, qubit_index = qbt, num_layers=num_layers)
            for neighbor in other_layers_neighbors:
                potential_neighbors.append(neighbor)
            data_qubits, ancilla_qubits = cluster_qubits_ids(dx = dx, dy = dy, num_layers=num_layers)
            all_qubits_indices = data_qubits + ancilla_qubits
            # all_qubits_indices = cluster_qubits_ids(dx = dx, dy = dy, num_layers=num_layers)
            neighbors = list(set(potential_neighbors) & set(all_qubits_indices))
        
        return neighbors
    
    
    def transfer_qubit_number_to_ij_index(self, qbt, dx, dy):
        alpha = 2*dx - 1
        j = qbt // alpha
        i = qbt % alpha
        return [i,j]
    
    def get_down_neighbor(self, qbt, dx, dy):
        # Transform qbt to (i, j) coordinate.
        alpha = 2*dx - 1
        j = qbt // alpha
        i = qbt % alpha
        if j != 0:
            return int(((j-1)*(2*dx-1)+i))  # down
        else:
            return None
        
    def get_up_neighbor(self, qbt, dx, dy):
        # Transform qbt to (i, j) coordinate.
        alpha = 2*dx - 1
        j = qbt // alpha
        i = qbt % alpha
        if j != 2 * dy - 2:
            return int(((j+1)*(2*dx-1)+i))  # up
        else:
            return None
        

    def get_circuit(self, cycles, dx: int, dy: int, basis='Z', architecture='CBQC', atom_array_sim = False, **kwargs):
        if architecture == 'MBQC':
            if atom_array_sim:
                return xzzx_cluster_noisy(cycles=cycles, dx=dx, dy=dy, basis=basis, **kwargs)
            else:
                return xzzx_cluster(cycles=cycles, dx=dx, dy=dy, basis=basis, **kwargs)
            
        elif architecture == 'CBQC':
            if atom_array_sim:
                return xzzx_circuit_noisy(cycles=cycles, dx=dx, dy=dy, basis=basis, **kwargs)
            else:
                return xzzx_circuit(cycles=cycles, dx=dx, dy=dy, basis=basis, **kwargs)

        else:
            assert True == False
    
    def get_all_data_ids(self, architecture, dx, dy, cycles:int):
        if architecture == "MBQC":
            return cluster_data_ids(dx,dy,cycles)
        elif architecture == "CBQC":
            return data_ids(dx,dy)


    def get_all_ancilla_ids(self, architecture, dx: int, dy: int, cycles: int = -1):
        if architecture == "MBQC":
            # TODO: implement this.
            return None
        elif architecture == "CBQC":
            return ancilla_ids(dx, dy)


    def get_data_neighbors_of_ancilla_cbqc(self, dx: int, dy: int, **kwargs):
        ancillas = ancilla_ids(dx=dx, dy=dy) # GB: I changed it, to fix the "architecture is missing" bug
        # ancillas = self.get_all_ancilla_ids(dx=dx, dy=dy, architecture=architecture)
        data_neighbors = {ancilla: self.get_ordered_neighbors(ancilla, dx, dy) for ancilla in ancillas}
        return data_neighbors


    
    
    def get_all_qubit_indices(self, dx, dy, cycles, architecture):
        if architecture == 'MBQC':
            num_layers = get_num_layers(cycles)
            data_qubits, ancilla_qubits = cluster_qubits_ids(dx, dy, num_layers=num_layers)
            return data_qubits +  ancilla_qubits
        elif architecture == 'CBQC':
            data_qubits, ancilla_qubits = get_data_ancilla_indices(dx,dy)
            return data_qubits +  ancilla_qubits

    
    def get_all_qubit_indices_layer(self, dx, dy, layer_ix, architecture, single_list = False):
        if architecture == 'MBQC':
            offset = layer_offset(dx, dy, layer_ix)
            ancilla_qubits = [j*(2*dx-1)+i + offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1) and ((i%2 ==0 and layer_ix%2==0) or (i%2 ==1 and layer_ix%2==1))]
            data_qubits = [j*(2*dx-1)+i + offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0)]
            if single_list:
                return data_qubits + ancilla_qubits
            else:
                return [data_qubits, ancilla_qubits]
        elif architecture == 'CBQC':
            print(f"Need to build this feature")
            return None
        
        
        
    def get_gates_dicts(self, architecture, dx, dy):
        if architecture == "MBQC":
            print(f"Need to build this feature")
            return None
        elif architecture == "CBQC":
            print(f"Need to build this feature")
            return None

    def get_initialization_basis(self, qbt, dx, dy, basis='X'):
        alpha = 2*dx - 1
        j = qbt // alpha
        i = qbt % alpha
        (prep1, prep2) = ('RZ', 'RX') if basis == 'X' else ('RX', 'RZ')
        assert (i+j) % 2 == 0
        if i%2 == 0:
            return prep1
        else:
            return prep2
    
    def find_SWAP_undetected_qubits(self, dx, dy, SWAP_round_ix):
        # only relevant for CBQC
        data_qubits = data_ids(dx,dy)
        unheralded_data_qubits = []
        for qbt in data_qubits:
            alpha = 2*dx - 1
            j = qbt // alpha
            i = qbt % alpha
            if (SWAP_round_ix%2 == 0 and j == 0) or (SWAP_round_ix%2 == 1 and j == 2*dy-2):
                unheralded_data_qubits.append(qbt)
        return unheralded_data_qubits
                
        
        