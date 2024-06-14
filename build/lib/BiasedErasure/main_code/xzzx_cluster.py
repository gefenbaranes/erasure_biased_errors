import stim
import numpy as np
from BiasedErasure.main_code.noise_channels import no_noise
"""
This code only works for logical Z.
TODO: logical X.
"""

def get_num_layers(cycles:int):
    num_layers = 2*(cycles) - 1 # every layer checks X or Z and we want cycles XZ layers in total. plus 1 because of the init layer
    # num_layers = cycles # for debugging
    return num_layers
    
    
def cluster_data_ids(dx: int, dy: int, cycles:int):
    # this function returns all the indexes of the data qubits in the cluster
    num_layers = get_num_layers(cycles)
    data_ids_list = []
    for layer_ix in range(num_layers):
        offset = layer_offset(dx, dy, layer_ix)
        for j in range(2*dy-1):
            for i in range(2*dx-1):
                if (i + j) % 2 == 0: #data qubit
                    qubit_ix = j*(2*dx-1)+i+offset
                    data_ids_list.append(qubit_ix)
                    # print(f"{i,j,layer_ix} qubit index is: {qubit_ix}. Now our list has: {data_ids_list}")
                        
    return data_ids_list


def cluster_qubits_ids(dx: int, dy: int, num_layers:int):
    # this function returns all the indexes of the qubits in the cluster
    data_ids = []
    ancilla_ids = []
    qubits_ids_list = []
    for layer_ix in range(num_layers):
        offset = layer_offset(dx, dy, layer_ix)
        for j in range(2*dy-1):
            for i in range(2*dx-1):
                if (i + j) % 2 == 0: #data qubit
                    qubit_ix = j*(2*dx-1)+i+offset
                    data_ids.append(qubit_ix)
                elif (layer_ix % 2 == 0 and i % 2 == 0) or (layer_ix % 2 == 1 and i % 2 == 1): # ancilla qubits
                        qubit_ix = j*(2*dx-1)+i+offset
                        ancilla_ids.append(qubit_ix)
                        
    return data_ids, ancilla_ids


def get_qubit_type(i: int, layer_ix: int):
    # this function gives us the data qubit type: X or Z type.
    if (layer_ix % 2 == 0 and i % 2 == 1) or (layer_ix % 2 == 1 and i % 2 == 0):
        return "X"
    elif (layer_ix % 2 == 0 and i % 2 == 0) or (layer_ix % 2 == 1 and i % 2 == 1):
        return "Z"
    
        

def layer_offset(dx: int, dy: int, layer_ix: int, physical=False) -> int:
    # if physical == True: only the number of physical qubits in the layer (half of the ancillas)
    # else: return the number full number of ancillas + data qubits for the regular circuit-based surface code
    n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data qubit in a layer
    if physical:
        m = int(0.5*(2*dx-1)*(2*dy-1)*0.5)  # num ancilla qubits in a layer
    else:
        m = int(0.5*(2*dx-1)*(2*dy-1))  # num ancilla qubits in a surface code
    return layer_ix * (n + m)

def total_num_qubits(dx: int, dy: int, num_layers:int):
    n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data qubits in each layer
    m = int(0.5*(2*dx-1)*(2*dy-1)*0.5) # num ancilla qubits in each layer
    return (n+m)*num_layers # num qubits in all layer

def qubit_index(dx: int, dy: int, row_index:int, col_index:int, layer_ix: int) -> int:
    outLayers_offset = layer_offset(dx=dx, dy=dy, layer_ix=layer_ix, physical=True)
    inLayer_offset = 0
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if row_index==i and col_index==j:
                return inLayer_offset+outLayers_offset
            if ((i+j)%2 == 0) or (layer_ix % 2 == 0 and i % 2 == 0) or (layer_ix % 2 == 1 and i % 2 == 1):
                inLayer_offset += 1

#####################################


def gates_in_layer(circ: stim.Circuit, dx: int, dy: int, bias_preserving_gates : bool, layer_ix=0):
    # this function should change for different codes
    offset = layer_offset(dx, dy, layer_ix)
    # Entangling gates
    rnd1, rnd2, rnd3, rnd4 = [], [], [], []
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if (i + j) % 2 == 1:  # Iterate over ancilla qubits.
                if (layer_ix%2 == 0 and i%2 == 0) or (layer_ix%2 == 1 and i%2 == 1): # ancilla locations
                    if j != 2 * dy - 2: # north
                        rnd1.extend([j*(2*dx-1)+i+offset, (j+1)*(2*dx-1)+i+offset])
                    if i != 2 * dx - 2: # east
                        rnd2.extend([j*(2*dx-1)+i+offset, j*(2*dx-1)+(i+1)+offset])
                    if i != 0: # west
                        rnd3.extend([j*(2*dx-1)+i+offset, j*(2*dx-1)+(i - 1)+offset])
                    if j != 0: # south
                        rnd4.extend([j*(2*dx-1)+i+offset,  (j-1)*(2*dx-1)+i+offset])
    circ.append('CZ', rnd1)
    # circ.append('TICK')
    if bias_preserving_gates:
        circ.append('CX', rnd2)
    else:
        targets = rnd2[1::2]
        circ.append('H', targets)
        circ.append('CZ', rnd2)
        circ.append('H', targets)
    # circ.append('CX', rnd2)
    # circ.append('TICK')
    if bias_preserving_gates:
        circ.append('CX', rnd3)
    else:
        targets = rnd3[1::2]
        circ.append('H', targets)
        circ.append('CZ', rnd3)
        circ.append('H', targets)
    # circ.append('CX', rnd3)
    # circ.append('TICK')
    circ.append('CZ', rnd4)
    # circ.append('TICK')

def interact_and_measure_all_layers(circ: stim.Circuit, dx: int, dy: int, num_layers: int, bias_preserving_gates:bool, basis='X'):
    # In this function we take each layer from second to last, interact with previous layer (gates between layers) and measure previous layer.
    for layer_ix in range(num_layers):
        # add TICK before every layer
        circ.append('TICK')
        if layer_ix<num_layers-1:# last layer has no gates to the next layer
            # TODO: take the layer out of storage, entangle to the previous layer and measure the previous layer. make sure the measurements are at the same order for the detectors.
            gates_between_layers(circ=circ, dx=dx, dy=dy, layer1_ix=layer_ix, bias_preserving_gates=bias_preserving_gates)
            measure_layer(circ, dx, dy, layer_ix)
        elif layer_ix == num_layers-1: # last layer
            measure_layer(circ, dx, dy, layer_ix)
        # add TICK after every layer
        circ.append('TICK')

def gates_between_layers(circ: stim.Circuit, dx: int, dy: int, bias_preserving_gates : bool, layer1_ix: int):
    # TODO: take the relevant qubits outside of storage into the entangling zone.
    layer2_ix = layer1_ix + 1
    offset1, offset2 = layer_offset(dx, dy, layer1_ix), layer_offset(dx, dy, layer2_ix)

    rnd = []
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if (i + j) % 2 == 0:  # Iterate over data qubits.
                if (layer1_ix % 2 == 0 and i % 2 == 0) or (layer1_ix % 2 == 1 and i % 2 == 1): # data 1: X type qubit
                    data1_ix = j*(2*dx-1)+i+offset1 # X type qubit
                    data2_ix = j*(2*dx-1)+i+offset2 # Z type qubit
                    rnd.extend([data1_ix, data2_ix]) # CX: X type to Z type
                elif (layer1_ix % 2 == 0 and i % 2 == 1) or (layer1_ix % 2 == 1 and i % 2 == 0): # data 1: Z type qubit
                    data1_ix = j*(2*dx-1)+i+offset1 # Z type qubit
                    data2_ix = j*(2*dx-1)+i+offset2 # X type qubit
                    rnd.extend([data2_ix, data1_ix]) # CX: X type to Z type
    
    if bias_preserving_gates:
        circ.append('CX', rnd)
    else:
        targets = rnd[1::2]
        circ.append('H', targets)
        circ.append('CZ', rnd)
        circ.append('H', targets)
    # circ.append('CX', rnd)
    # circ.append('TICK')


def measure_layer(circ, dx: int, dy: int, layer_ix: int):
    # measure all qubits in the layer
    offset = layer_offset(dx, dy, layer_ix)
    for j in range(2*dy-1):
        for i in range(2*dx-1): # X type data or ancilla
            if ((i+j)%2 == 0 and layer_ix % 2 == 0 and i % 2 == 0) \
            or ((i+j)%2 == 0 and layer_ix % 2 == 1 and i % 2 == 1) \
            or ((i+j)%2 == 1 and layer_ix % 2 == 0 and i % 2 == 0) \
            or ((i+j)%2 == 1 and layer_ix % 2 == 1 and i % 2 == 1):
                circ.append('MX', [j*(2*dx-1)+i+offset])
                # or Z type data:
            elif ((i+j)%2 == 0 and layer_ix % 2 == 0 and i % 2 == 1) \
            or ((i+j)%2 == 0 and layer_ix % 2 == 1 and i % 2 == 0):
                circ.append('MZ', [j*(2*dx-1)+i+offset])
                    


def construct_detectors_all_layers(circ, dx: int, dy: int, num_layers:int, basis='X'):
    # measure all the qubits in the cluster in the X basis (X type) or Z basis (Z type)
    # create 6 body operators parity checks for all the unit cells in the cluster
    # TODO: in XZZX cluster [], X type qubits are measured in X, Z type qubits are measured in Z.

    N = total_num_qubits(dx=dx, dy=dy, num_layers=num_layers) # total num qubits in all layers

    # For each qubit, check the 6 body operator where this qubit is the upper face qubit of the relevant unit cell:
    for layer_ix in range(num_layers):
        for j in range(2*dy-1):
            for i in range(2*dx-1):
                if (i + j) % 2 == 1:
                    if (layer_ix % 2 == 0 and i % 2 == 1) or (layer_ix % 2 == 1 and i % 2 == 0):  # (even) or (odd) layer
                        detector_targets = []
                        # offset = layer_offset(dx=dx, dy=dy, layer_ix=layer_ix)
                        
                        if j != 2 * dy - 2: # north
                            #data_ix = int(((j+1)*(2*dx-1)+i+offset)/2)  
                            qubit_ix = qubit_index(dx=dx, dy=dy, row_index=i, col_index=j+1, layer_ix=layer_ix)
                            detector_targets.append(stim.target_rec(-(N-qubit_ix)))
                        if i != 2 * dx - 2:  # east
                            # data_ix = int((j*(2*dx-1)+(i+1))/2)
                            qubit_ix = qubit_index(dx=dx, dy=dy, row_index=i+1, col_index=j, layer_ix=layer_ix)
                            detector_targets.append(stim.target_rec(-(N-qubit_ix)))
                        if i != 0:  # west
                            # data_ix = int((j*(2*dx-1)+(i - 1))/2)
                            qubit_ix = qubit_index(dx=dx, dy=dy, row_index=i-1, col_index=j, layer_ix=layer_ix)
                            detector_targets.append(stim.target_rec(-(N-qubit_ix)))
                        if j != 0:  # south
                            # data_ix = int(((j-1)*(2*dx-1)+i)/2)
                            qubit_ix = qubit_index(dx=dx, dy=dy, row_index=i, col_index=j-1, layer_ix=layer_ix)
                            detector_targets.append(stim.target_rec(-(N-qubit_ix)))
                        if layer_ix != num_layers - 1: # up
                            # data_ix = j*(2*dx-1)+i+layer_offset(dx=dx, dy=dy, layer_ix=layer_ix+1)
                            qubit_ix = qubit_index(dx=dx, dy=dy, row_index=i, col_index=j, layer_ix=layer_ix+1)
                            detector_targets.append(stim.target_rec(-(N-qubit_ix)))
                        if layer_ix != 0: # down
                            # data_ix = j*(2*dx-1)+i+layer_offset(dx=dx, dy=dy, layer_ix=layer_ix-1)
                            qubit_ix = qubit_index(dx=dx, dy=dy, row_index=i, col_index=j, layer_ix=layer_ix-1)
                            detector_targets.append(stim.target_rec(-(N-qubit_ix)))
                        circ.append('DETECTOR', detector_targets)



def initialize_layer(circ: stim.Circuit, dx: int, dy: int, basis='X', layer_ix=0):
    offset = layer_offset(dx, dy, layer_ix)
    # TODO: make this compatible with Z logical basis as well.
    # ValueError(basis != 'X')
        
    # Initialize data qubits
    (prep1, prep2) = ('RX', 'RZ') if basis == 'Z' else ('RZ', 'RX')
    circ.append(prep1, [j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if 
                        ((i+j)%2 == 0 and ((i%2==0 and layer_ix%2==0) or (i%2==1 and layer_ix%2==1)))]) # X type data
    circ.append(prep2, [j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if 
                        ((i+j)%2 == 0 and ((i%2==0 and layer_ix%2==1) or (i%2==1 and layer_ix%2==0)))]) # Z type data

    # Initialize ancilla qubits
    if layer_ix%2 == 0: # even layer
        circ.append('RX', [j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2 == 0)])
    else:
        circ.append('RX', [j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2 == 1)])
    # circ.append('TICK')
    
    
def initialize_all_layers(circ: stim.Circuit, dx: int, dy: int, num_layers: int, bias_preserving_gates : bool, basis='X'):
    # initialize all the qubits in all d layers
    # TODO: in XZZX cluster [], X type qubits are initialize in |+>, Z type qubits in |0>.
    for layer_ix in range(num_layers): # reset all  qubits in all layers
        initialize_layer(circ=circ, dx=dx, dy=dy, basis=basis, layer_ix=layer_ix)
        
    for layer_ix in range(num_layers): # make gates to initialize all layers in parallel
        circ.append('TICK')
        gates_in_layer(circ=circ, dx=dx, dy=dy, layer_ix=layer_ix, bias_preserving_gates=bias_preserving_gates)
        circ.append('TICK')
        # TODO: after initializing all layers, put all of them in the storage besides the first layer.
    

def logical(circ: stim.Circuit, dx: int, dy: int, num_layers:int, basis='X') -> None:
    # Extract the logical qubit information from the data qubits in last layer
    # This function should change for different codes
    N = total_num_qubits(dx=dx, dy=dy, num_layers=num_layers) # total num qubits in all layer
    detector_targets = []
    # offset = layer_offset(dx=dx, dy=dy, layer_ix=max(dx,dy)-1, physical=True)
    if basis == 'Z':
        for layer_ix in range(num_layers):
            for j in range(2*dy-1):
                if layer_ix % 2 == 0 and j % 2 == 0: # left col on even layers
                    data_ix = qubit_index(dx=dx, dy=dy, row_index=0, col_index=j, layer_ix=layer_ix)
                    # print(f"i= {0}, j={j}, data index = {data_ix}")
                    detector_targets.append(stim.target_rec(-(N-data_ix)))
        if num_layers % 2 == 0:  # last layer measure in X if even num of layers (d is even)
            for j in range(2*dy-1):
                data_ix = qubit_index(dx=dx, dy=dy, row_index=0, col_index=j, layer_ix=num_layers-1)
                detector_targets.append(stim.target_rec(-(N-data_ix)))
                # print(f"Last layer! i= {0}, j={j}, data index = {num_layers-1}")
        circ.append('OBSERVABLE_INCLUDE', detector_targets, 0)
    else:
        assert basis == 'X'
        print(f"The circuit is not compatible with Z basis initialization.")
        for layer_ix in range(num_layers):
            for i in range(2*dx-1): # first row on even layers
                if layer_ix % 2 == 0 and i % 2 == 0:
                    data_ix = qubit_index(dx=dx, dy=dy, row_index=i, col_index=0, layer_ix=layer_ix)
                    detector_targets.append(stim.target_rec(-(N-data_ix)))
        if num_layers % 2 == 1:  # last layer measure in Z if odd num of layers (d is even)
            for i in range(2*dx-1):
                data_ix = qubit_index(dx=dx, dy=dy, row_index=i, col_index=0, layer_ix=num_layers-1)
                detector_targets.append(stim.target_rec(-(N-data_ix)))
        circ.append('OBSERVABLE_INCLUDE', detector_targets, 0)
        
    
def xzzx_cluster(cycles, dx: int, dy: int, bias_preserving_gates : bool, basis='X', noise=no_noise, **kwargs):
    num_layers = get_num_layers(cycles)
    circ = stim.Circuit()
    initialize_all_layers(circ, dx, dy, num_layers=num_layers, basis=basis, bias_preserving_gates=bias_preserving_gates)
    interact_and_measure_all_layers(circ, dx, dy, num_layers=num_layers, basis=basis, bias_preserving_gates=bias_preserving_gates)
    # all_gates(circ, dx, dy, num_layers=num_layers, basis=basis)
    construct_detectors_all_layers(circ, dx, dy, num_layers=num_layers, basis=basis)
    # circ = noise(circ, **kwargs)
    logical(circ, dx, dy, basis=basis, num_layers=num_layers)
    return circ

def get_neighbors_next_previous_layers(dx: int, dy: int, qubit_index: int, num_layers: int):
    n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data qubit in a layer
    m = int((2*dx-1)*(2*dy-1)*0.5)  # num ancilla qubits in a surface code (twice num ancilla in MBQC layer)
    offset = n + m
    neighbors_indices = []
    if qubit_index - offset >= 0:
        neighbors_indices.append(qubit_index - offset)
    if qubit_index + offset <= num_layers * offset - 1:
        neighbors_indices.append(qubit_index + offset)
    return neighbors_indices
        

if __name__ == '__main__':
    dx = 2
    dy = 2
    circuit = xzzx_cluster(cycles=max(dx,dy), dx=dx, dy=dy, basis='X', noise=no_noise)
    print(circuit)