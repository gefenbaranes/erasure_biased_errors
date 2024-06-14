import stim
import numpy as np
from BiasedErasure.main_code.noise_channels import no_noise
from BiasedErasure.main_code.LogicalCircuit import LogicalCircuit
from typing import Union, List, Optional

"""
This code only works for logical X.
TODO: logical Z.
"""

def cluster_data_ids(dx: int, dy: int, cycles:int):
    # this function returns all the indexes of the data qubits in the cluster
    # num_layers = max(dx, dy)
    num_layers = 2*cycles - 1
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
    # num_layers = max(dx, dy)
    qubits_ids_list = []
    for layer_ix in range(num_layers):
        offset = layer_offset(dx, dy, layer_ix)
        for j in range(2*dy-1):
            for i in range(2*dx-1):
                if (i + j) % 2 == 0: #data qubit
                    qubit_ix = j*(2*dx-1)+i+offset
                    qubits_ids_list.append(qubit_ix)
                elif (layer_ix % 2 == 0 and i % 2 == 0) or (layer_ix % 2 == 1 and i % 2 == 1): # ancilla qubits
                        qubit_ix = j*(2*dx-1)+i+offset
                        qubits_ids_list.append(qubit_ix)
                        
    return qubits_ids_list


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
    # num_layers = max(dx,dy)
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


def gates_in_layer(circ: stim.Circuit, dx: int, dy: int, bias_preserving_gates : bool, layer_ix=0, move_duration: Optional[float] = 200):
    # this function should change for different codes
    offset = layer_offset(dx, dy, layer_ix)
    data_qubit_x_type = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and ((i%2==0 and layer_ix%2==0) or (i%2==1 and layer_ix%2==1)))] )
    data_qubit_z_type = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and ((i%2==0 and layer_ix%2==1) or (i%2==1 and layer_ix%2==0)))])
    ancilla_qubit_even_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2 == 0)])
    ancilla_qubit_odd_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2 == 1)])
    
    
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
    # round 1 - CZ: (no movement errors in first round, because we counted to movement when we moved the qubits to entangling zone)
    circ.append('CZ', rnd1)
    # circ.append('TICK')
    
    # round 2 - CX:
    if bias_preserving_gates:
        circ.append('CX', rnd2)
    else:
        targets = rnd2[1::2]
        circ.append('H', targets)
        circ.append('CZ', rnd2)
        circ.append('H', targets)
    circ.append('MOVE', (), move_duration) # idle errors during movement
    # circ.append('TICK')
    
    # round 3 - CX:
    if bias_preserving_gates:
        circ.append('CX', rnd3)
    else:
        targets = rnd3[1::2]
        circ.append('H', targets)
        circ.append('CZ', rnd3)
        circ.append('H', targets)
    circ.append('MOVE', (), move_duration) # idle errors during movement
    # circ.append('TICK')
    
    # round 4 - CZ:
    circ.append('CZ', rnd4)
    circ.append('MOVE', (), move_duration) # idle errors during movement
    # circ.append('TICK')

def interact_and_measure_all_layers(circ: stim.Circuit, dx: int, dy: int, num_layers: int, bias_preserving_gates : bool, basis='Z', move_duration: Optional[float] = 200):
    # In this function we take each layer from second to last, interact with previous layer (gates between layers) and measure previous layer.
    
    # take layer0 to the entangling zone:
    all_qubits_in_layer0 = sorted( [ j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ( (i+j)%2 == 0 ) or ( (i+j)%2 == 1 and i%2==0 ) ] )
    circ.append('MOVE_TO_ENTANGLING', all_qubits_in_layer0, 0) # no idling errors because we can assume that we didn't remove this layer from the zone after preparation
    for layer_ix in range(num_layers):
        if layer_ix<num_layers-1:# last layer has no gates to the next layer
            layer2_ix = layer_ix + 1
            # TODO: take layer2 out of storage, entangle layer1 and measure layer1. make sure the measurements are at the same order for the detectors.
            offset = layer_offset(dx, dy, layer2_ix)
            all_qubits_in_layer2 = sorted( [ j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if \
                                    ( (i+j)%2 == 0 ) or \
                                    ( (i+j)%2 == 1 and ((i%2==1 and layer2_ix%2==1) or (i%2==0 and layer2_ix%2==0)) ) ] )
            circ.append('MOVE_TO_ENTANGLING', all_qubits_in_layer2, 0)
            circ.append('MOVE', all_qubits_in_layer2, move_duration) # idle errors during movement
            gates_between_layers(circ=circ, dx=dx, dy=dy, layer1_ix=layer_ix, bias_preserving_gates=bias_preserving_gates)
            measure_layer(circ, dx, dy, layer_ix, num_layers=num_layers)
        elif layer_ix == num_layers-1: # last layer
            measure_layer(circ, dx, dy, layer_ix, num_layers=num_layers)


def gates_between_layers(circ: stim.Circuit, dx: int, dy: int, layer1_ix: int, bias_preserving_gates:bool):
    # TODO: take the relevant qubits outside of storage into the entangling zone.
    layer2_ix = layer1_ix + 1
    offset1, offset2 = layer_offset(dx, dy, layer1_ix), layer_offset(dx, dy, layer2_ix)

    # take layer2 out of storage (layer1 should already be inside the entangling zone)
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
    circ.append('TICK')


def measure_layer(circ, dx: int, dy: int, layer_ix: int, num_layers:int, move_duration: Optional[float] = 200):
    # Masure all qubits in the layer
    # We can only measure in Z and do H before if needed MX.
    offset = layer_offset(dx, dy, layer_ix)
    
    data_qubit_x_type = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and ((i%2==0 and layer_ix%2==0) or (i%2==1 and layer_ix%2==1)))] )
    data_qubit_z_type = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and ((i%2==0 and layer_ix%2==1) or (i%2==1 and layer_ix%2==0)))])
    ancilla_qubits = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and ((i%2==0 and layer_ix%2==0) or (i%2==1 and layer_ix%2==1)))] ) # all ancilla qubits are x-type

    circ.append('H', sorted(data_qubit_x_type + ancilla_qubits))
    if layer_ix != num_layers-1:
        circ.append('MOVE', (), move_duration) # Put idling noise on all active qubits due to movement time. last layer is measured in the entangling zone so there is no extra idling noise.
    circ.append('MZ', sorted(data_qubit_x_type + data_qubit_z_type + ancilla_qubits))
    circ.append('MOVE_TO_NO_NOISE', data_qubit_x_type + data_qubit_z_type + ancilla_qubits, 0) # all measured qubits are inactive now
        

def construct_detectors_all_layers(circ, dx: int, dy: int, num_layers:int, basis='Z'):
    # measure all the qubits in the cluster in the X basis (X type) or Z basis (Z type)
    # create 6 body operators parity checks for all the unit cells in the cluster
    # TODO: in XZZX cluster [], X type qubits are measured in X, Z type qubits are measured in Z.

    # num_layers = max(dx, dy)
    N = total_num_qubits(dx=dx, dy=dy, num_layers=num_layers) # total num qubits in all layers

    # step 1: measure all the qubits in the layer
    # for layer_ix in range(num_layers):
    #     measure_layer(circ, dx, dy, layer_ix)

    # step 2: for each qubit, check the 6 body operator where this qubit is the upper face qubit of the relevant unit cell:
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



def initialize_layer(circ: stim.Circuit, dx: int, dy: int, basis='Z', layer_ix=0, move_duration: Optional[float] = 200):
    offset = layer_offset(dx, dy, layer_ix)
    # TODO: make this compatible with Z logical basis as well.
    ValueError(basis != 'X')

    data_qubit_x_type = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and ((i%2==0 and layer_ix%2==0) or (i%2==1 and layer_ix%2==1)))] )
    data_qubit_z_type = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and ((i%2==0 and layer_ix%2==1) or (i%2==1 and layer_ix%2==0)))])
    ancilla_qubit_even_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2 == 0)])
    ancilla_qubit_odd_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2 == 1)])
    
    # We can initialize only in Z and do H afterwards. also remove zones.
    circ.append('MOVE_TO_ENTANGLING', data_qubit_x_type + data_qubit_z_type, move_duration) # get idling noise
    circ.append('RZ', data_qubit_x_type + data_qubit_z_type) # X type data
    
    # Initialize data qubits
    if basis == 'Z':
        circ.append('H', data_qubit_z_type) # Z type in |+>
    elif basis == 'X':
        circ.append('H', data_qubit_x_type) # X type in |+>

    # Initialize ancilla qubits
    if layer_ix%2 == 0: # even layer
        circ.append('MOVE_TO_ENTANGLING', ancilla_qubit_even_layers, move_duration) # get idling noise
        circ.append('RZ', ancilla_qubit_even_layers)
        circ.append('H', ancilla_qubit_even_layers)
    else:
        circ.append('MOVE_TO_ENTANGLING', ancilla_qubit_odd_layers, move_duration) # get idling noise
        circ.append('RZ', ancilla_qubit_odd_layers)
        circ.append('H', ancilla_qubit_odd_layers)
    circ.append('TICK')
    
    
def initialize_all_layers(circ: stim.Circuit, dx: int, dy: int, num_layers: int, bias_preserving_gates : bool, basis='Z', move_duration: Optional[float] = 200):
    ValueError(basis != 'X')
    
    # initialize and connect all the qubits in all d layers
    
    ### part 1 - initialize all layers in 1 shot: (in XZZX cluster, X type qubits are initialize in |+>, Z type qubits in |0>)
    
    # We can entangle up to 4 layers in one shot, so we have batches of up to 4:
    for batch_start in range(0, num_layers, 4):
        batch_end = min(batch_start + 4, num_layers)
        offsets = []
        even_layers_offsets = []
        odd_layers_offsets = []
        # Inner loop to process each layer in the current batch
        for layer_ix in range(batch_start, batch_end):
            offset = layer_offset(dx, dy, layer_ix)
            offsets.append(offset)
            if layer_ix%2 == 0: # even layer
                even_layers_offsets.append(offset)
            elif layer_ix%2 == 1: # odd layer
                odd_layers_offsets.append(offset)
                    
        data_qubits_x_type_all_even_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) for offset in even_layers_offsets if ( (i+j)%2 == 0 and (i%2==0) )] )
        data_qubits_x_type_all_odd_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) for offset in odd_layers_offsets if  ( (i+j)%2 == 0 and (i%2==1) )] )
        data_qubits_x_type_all_layers = sorted(data_qubits_x_type_all_even_layers + data_qubits_x_type_all_odd_layers)
        
        data_qubits_z_type_all_even_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) for offset in even_layers_offsets if ( (i+j)%2 == 0 and  i%2==1 )])
        data_qubits_z_type_all_odd_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) for offset in odd_layers_offsets if ( (i+j)%2 == 0 and  i%2==0 )])
        data_qubits_z_type_all_layers = sorted(data_qubits_z_type_all_even_layers + data_qubits_z_type_all_odd_layers)

        ancilla_qubit_all_even_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) for offset in even_layers_offsets if ((i+j)%2 == 1 and i%2 == 0)])
        ancilla_qubit_all_odd_layers = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) for offset in odd_layers_offsets if ((i+j)%2 == 1 and i%2 == 1)])
        
        # Initialize data qubits
        circ.append('MOVE_TO_ENTANGLING', data_qubits_x_type_all_layers + data_qubits_z_type_all_layers + ancilla_qubit_all_even_layers + ancilla_qubit_all_odd_layers, 0) # move all qubits to the entangling zone
        circ.append('RZ', sorted(data_qubits_x_type_all_layers + data_qubits_z_type_all_layers + ancilla_qubit_all_even_layers + ancilla_qubit_all_odd_layers)) # X type data  + all ancilla qubits
        circ.append('MOVE', data_qubits_x_type_all_layers + data_qubits_z_type_all_layers + ancilla_qubit_all_even_layers + ancilla_qubit_all_odd_layers, move_duration) # idling errors
        if basis == 'Z':
            circ.append('H', sorted(data_qubits_z_type_all_layers + ancilla_qubit_all_even_layers + ancilla_qubit_all_odd_layers)) # Z type in |+>
        elif basis == 'X':
            circ.append('H', sorted(data_qubits_x_type_all_layers + ancilla_qubit_all_even_layers + ancilla_qubit_all_odd_layers)) # X type in |+>

        circ.append('TICK')
        
        
        ### part 2 - gates in all layers in the batch in 1 shot:
        rnd1, rnd2, rnd3, rnd4 = [], [], [], []
        for layer_ix in range(batch_start, batch_end):
            offset = layer_offset(dx, dy, layer_ix)
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
        
        # round 1 - CZ: (no movement errors in first round, because we counted to movement when we moved the qubits to entangling zone)
        circ.append('CZ', rnd1)
        
        # round 2 - CX:
        if bias_preserving_gates:
            circ.append('CX', rnd2)
        else:
            targets = rnd2[1::2]
            circ.append('H', targets)
            circ.append('CZ', rnd2)
            circ.append('H', targets)
        circ.append('MOVE', (), move_duration) # idle errors during movement
        
        # round 3 - CX:
        if bias_preserving_gates:
            circ.append('CX', rnd3)
        else:
            targets = rnd3[1::2]
            circ.append('H', targets)
            circ.append('CZ', rnd3)
            circ.append('H', targets)
        circ.append('MOVE', (), move_duration) # idle errors during movement
        
        # round 4 - CZ:
        circ.append('CZ', rnd4)
        circ.append('MOVE', (), move_duration) # idle errors during movement
        
        # move all qubits to the storage zone:
        circ.append('MOVE_TO_STORAGE', ancilla_qubit_all_even_layers + ancilla_qubit_all_odd_layers + data_qubits_x_type_all_layers + data_qubits_z_type_all_layers, move_duration) # idle errors during movement



        # # move all qubits (except first layer) to the storage zone:
        # data_qubits_x_type_all_layers_except_0 = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) for offset in offsets if ( (i+j)%2 == 0 and (i%2==0) and offset > 0)] )
        # data_qubits_z_type_all_even_layers_except_0 = sorted([j*(2*dx-1)+i+offset for j in range(2*dy-1) for i in range(2*dx-1) for offset in offsets if ( (i+j)%2 == 0 and  i%2==1 and offset > 0)])

        # circ.append('MOVE_TO_STORAGE', ancilla_qubit_all_even_layers + ancilla_qubit_all_odd_layers + data_qubits_x_type_all_layers_except_0 + data_qubits_z_type_all_even_layers_except_0, move_duration) # idle errors during movement

    

def logical(circ: stim.Circuit, dx: int, dy: int, num_layers:int, basis='Z') -> None:
    # Extract the logical qubit information from the data qubits in last layer
    # This function should change for different codes
    # num_layers = max(dx, dy)
    N = total_num_qubits(dx=dx, dy=dy, num_layers=num_layers) # total num qubits in all layer
    detector_targets = []
    # offset = layer_offset(dx=dx, dy=dy, layer_ix=max(dx,dy)-1, physical=True)
    if basis == 'X':
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
        assert basis == 'Z'
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
        
    
def xzzx_cluster_noisy(cycles, dx: int, dy: int, bias_preserving_gates : bool, basis='Z', noise=no_noise, **kwargs):
    num_layers = 2*cycles - 1 # every layer checks X or Z, and we want cycles XZ layers in total. plus 1 because of the init layer
    error_model = {
        "entangling_zone_error_rate": (.002 / 4, .002 / 4, .005 / 4),
        "entangling_gate_error_rate": (.002 / 4, .002 / 4, .0025 / 4, .002 / 4, 0, 0, 0, .002 / 4, 0, 0, 0, .0025 / 4, 0, 0, .005 / 4),
        "idle_loss_rate": 1e-7,
        "idle_error_rate": (5e-6 / 25, 5e-6 / 25, 2e-5 / 25),
        "entangling_loss_rate": 0.005/4,
        "reset_error_rate": 0.003,
        "measurement_error_rate": 0.004,
        "single_qubit_error_rate": (1e-4, 1e-4, 1e-4),
        "reset_loss_rate": 0.000
    }
    
    move_duration = 200
    qubit_indices = cluster_qubits_ids(dx,dy, num_layers=num_layers)
    circ = LogicalCircuit(qubit_indices = qubit_indices, initialize_circuit=False, **error_model)
        
    initialize_all_layers(circ, dx, dy, num_layers=num_layers, basis=basis, bias_preserving_gates=bias_preserving_gates)
    interact_and_measure_all_layers(circ, dx, dy, num_layers=num_layers, basis=basis, bias_preserving_gates=bias_preserving_gates)
    # all_gates(circ, dx, dy, num_layers=num_layers, basis=basis)
    construct_detectors_all_layers(circ, dx, dy, num_layers=num_layers, basis=basis)
    # circ = noise(circ, **kwargs)
    logical(circ, dx, dy, basis=basis, num_layers=num_layers)
    return circ

def get_neighbors_next_previous_layers(dx: int, dy: int, qubit_index):
    n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data qubit in a layer
    m = int((2*dx-1)*(2*dy-1)*0.5)  # num ancilla qubits in a surface code (twice num ancilla in MBQC layer)
    offset = n + m
    neighbors_indices = []
    if qubit_index - offset >= 0:
        neighbors_indices.append(qubit_index - offset)
    if qubit_index + offset < max(dx, dy) * offset - 1:
        neighbors_indices.append(qubit_index + offset)
    return neighbors_indices
        

if __name__ == '__main__':
    dx = 2
    dy = 2
    circuit = xzzx_cluster(cycles=max(dx,dy), dx=dx, dy=dy, basis='Z', noise=no_noise)
    print(circuit)