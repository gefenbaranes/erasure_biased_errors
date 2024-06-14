import stim
import numpy as np
import random



def biased_erasure_noise_old(phys_err, bias_ratio, erasure_ratio):
    # here we put both loss and Pauli errors on each qubit.
    phys_err_1q = (1- np.sqrt(1 - phys_err))
    px_q1 = phys_err_1q / (2*(1+bias_ratio))
    py_q1 = px_q1
    pz_q1 = bias_ratio*(px_q1 + py_q1)
    entangling_gate_error_rate=(1-erasure_ratio)*np.array([px_q1, py_q1, pz_q1])
    entangling_gate_loss_rate=erasure_ratio*phys_err_1q
    return entangling_gate_error_rate, entangling_gate_loss_rate


def biased_erasure_noise(phys_err, bias_ratio):
    # here entangling_gate_loss_rate, entangling_gate_error_rate don't depend on erasure_ratio because this will be counted inside LogicalCircuit
    phys_err_1q = 1- np.sqrt(1 - phys_err)
    px_q1 = phys_err_1q / (2*(1+bias_ratio))
    py_q1 = px_q1
    pz_q1 = bias_ratio*(px_q1 + py_q1)
    entangling_gate_error_rate=np.array([px_q1, py_q1, pz_q1])
    entangling_gate_loss_rate=phys_err_1q
    return entangling_gate_error_rate, entangling_gate_loss_rate


def biased_erasure_noise_MBQC(
        circuit: stim.Circuit, dx: int = 2, dy: int = 2, 
        entangling_gate_error_rate = np.array([0.0,0.0,0.0]),
        entangling_gate_loss_rate = 0.0,
        erasure_weight: float = 0.0,
        bias_preserving: bool = False,
         **kwargs
        ) -> stim.Circuit:
    

    result = stim.Circuit()
    
    # Handling loss:
    potential_lost_qubits = np.array([], dtype=np.int32)
    loss_probabilities = np.array([], dtype=np.float32)
    
    n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data qubit in a layer
    m = int((2*dx-1)*(2*dy-1)*0.5)  # num ancilla qubits in a surface code (twice num ancilla in MBQC layer)
    single_layer_offset = n + m
    
    for instruction in circuit:
        if (instruction.name in ['CZ', 'CX']):
            qubits = instruction.targets_copy()
            layer_indices = [qbt.value // single_layer_offset for qbt in qubits]
            if set(layer_indices).issubset([0]):  # no error on gates within the first layer
                result.append(instruction)
            else:
                loss_probabilities, potential_lost_qubits = entangling_error(instruction, result, entangling_gate_error_rate, 
                                entangling_gate_loss_rate, erasure_weight, bias_preserving, 
                                loss_probabilities, potential_lost_qubits)

        else:
            result.append(instruction)
    return result, loss_probabilities, potential_lost_qubits



def entangling_error(instruction, circuit: stim.Circuit, entangling_gate_error_rate, entangling_gate_loss_rate, erasure_weight,
                    bias_preserving: bool, loss_probabilities: np.array, potential_lost_qubits: np.array):
    # targets = instruction.targets_copy()
    
    targets = [t.value for t in instruction.targets_copy()]
    
    ### Part 1 - bias preserving gates:
    if instruction.name == 'CX' and not bias_preserving:
        h_targets = targets[1::2]
        circuit.append('H', h_targets)
        circuit.append('CZ', targets)
        
    else:
        circuit.append(instruction)
        

    ### Part 2 - Put noise:
    err_types = random.choices(['loss','pauli'], weights=[erasure_weight,(1-erasure_weight)], k=int(len(targets)))
    
    # Pauli noise channel:
    pauli_targets = [target for target, err_type in zip(targets, err_types) if err_type == 'pauli'] # take all targets with 'pauli' errors:
    pauli_error = np.array(entangling_gate_error_rate)
    if sum(pauli_error) > 0 and len(pauli_targets)> 0:
        if len(pauli_error) == 3:
            circuit.append('PAULI_CHANNEL_1', pauli_targets, pauli_error)
        elif len(pauli_error) == 8:
            circuit.append('PAULI_CHANNEL_2', pauli_targets, pauli_error)
    
    loss_prob = entangling_gate_loss_rate
    if loss_prob>0:
        # Apply I channel to all targets to simulate identity operation
        circuit.append('I', targets)

        # Update the potential lost qubits list and loss probabilities
        for target, err_type in zip(targets, err_types):
            if err_type == 'loss':
                potential_lost_qubits = np.append(potential_lost_qubits, target)
                loss_probabilities = np.append(loss_probabilities, loss_prob)
            else:
                # If it's not a loss, append zero to maintain the correct indices
                potential_lost_qubits = np.append(potential_lost_qubits, target)
                loss_probabilities = np.append(loss_probabilities, 0)



    ### Part 3 - bias preserving gates:
    if instruction.name == 'CX' and not bias_preserving:
        circuit.append('H', h_targets)

    
    return loss_probabilities, potential_lost_qubits
            

def biased_erasure_noise_correlated(phys_err, bias_ratio):
    print("Need to build this feature")
    return None

def no_noise(phys_err, bias_ratio):
    entangling_gate_error_rate = (0,0,0)
    entangling_gate_loss_rate = 0
    return entangling_gate_error_rate, entangling_gate_loss_rate


def atom_array(phys_err, bias_ratio):
    return None, None



if __name__ == '__main__':
    pass

