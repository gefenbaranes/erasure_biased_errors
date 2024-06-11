import stim
import time
from BiasedErasure.main_code.utilities import draw_graph
import numpy as np
from BiasedErasure.main_code.noise_channels import biased_erasure_noise

def get_data_ancilla_indices(dx: int, dy:int):
    ancilla_qubits = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1)]
    data_qubits =    [j*(2*dx-1)+i   for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0)]
    return data_qubits, ancilla_qubits

def data_ids(dx: int, dy: int):
    return [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0)]

def ancilla_ids(dx: int, dy: int):
    return [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1)]

def find_paired_qubit(round_type: str, qubit_index: list, dx: int, dy: int):
    # this code takes a qubit (i,j) and find the paired qubit in the SWAP round.
    i = qubit_index[0] # column
    j = qubit_index[1] # row
    
    # even rounds: up. odd rounds: down.
    paired_qubit_i = i
    if round_type == 'even':
        paired_qubit_j = j+1
    elif round_type == 'odd':
        paired_qubit_j = j-1
        if paired_qubit_j < 0:
            paired_qubit_j += 2*dy
    return [paired_qubit_i, paired_qubit_j]

def get_extra_acillas_SWAP_round(round_type: str, data_index: list, dx: int, dy: int):
    # during a SWAP round, we need to add extra ancillas such that every data qubit will have a pair.
    # even round: up. odd round: down.
    # This function return all the extra ancilla qubits and their data pairs (to do gates with them).
    ancilla_row_index = dy+1 if round_type == 'even' else -1
    new_ancillas_i_indices = range(0,2*dx-1,2)
    new_ancillas_j_indices = [2*dy-1]*dx
    pairs_list = []
    for (ancilla_i, ancilla_j) in zip(new_ancillas_i_indices, new_ancillas_j_indices):
        paired_data_i = ancilla_i
        paired_data_j = ancilla_j - 1 if round_type == 'even' else 0
        paired_data_index  = paired_data_j*(2*dx-1)+paired_data_i
        paired_ancilla_index = ancilla_j*(2*dx-1)+ancilla_i
        pairs_list.append([paired_data_index, paired_ancilla_index])
    return pairs_list
        
    
    
    
    
def ij_index_into_number(i:int, j:int, dx: int, dy: int):
    # takes a (x,y) qubit index (i,j) and return the qubit number on the lattice.
    return j*(2*dx-1)+i
    
    
def initialize(circ: stim.Circuit, dx: int, dy: int, basis='Z'):
    # Initialize data qubits
    (prep1, prep2) = ('RZ', 'RX') if basis == 'X' else ('RX', 'RZ')
    circ.append(prep1, [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and i%2==0)]) # data qubits, even i
    circ.append(prep2, [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and i%2==1)]) # data qubits, odd i

    # Initialize ancilla qubits
    circ.append('RX', [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if (i+j)%2 == 1])
    circ.append('TICK')


def checks(circ: stim.Circuit, dx: int, dy: int):
    # Entangling gates
    rnd1, rnd2, rnd3, rnd4 = [], [], [], []
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if (i + j) % 2 == 1:  # Iterate over ancilla qubits.
                if j != 2 * dy - 2:
                    rnd1.extend([j*(2*dx-1)+i, (j+1)*(2*dx-1)+i])
                if i != 2 * dx - 2:
                    rnd2.extend([j*(2*dx-1)+i, j*(2*dx-1)+(i+1)])
                if i != 0:
                    rnd3.extend([j*(2*dx-1)+i, j*(2*dx-1)+(i - 1)])
                if j != 0:
                    rnd4.extend([j*(2*dx-1)+i, (j-1)*(2*dx-1)+i])
    circ.append('CZ', rnd1)
    circ.append('TICK')
    circ.append('CX', rnd2)
    circ.append('TICK')
    circ.append('CX', rnd3)
    circ.append('TICK')
    circ.append('CZ', rnd4)
    circ.append('TICK')


def measure_checks(circ, dx: int, dy: int, t_compare=False, basis='Z', offset=0):
    m = int(0.5*(2*dx-1)*(2*dy-1))
    circ.append('MRX', [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if (i+j)%2 == 1])
    if t_compare:
        # Compare parity of check at t (now) vs t-1.

        # Add offset equal to number of entangling gates. One detector per gate to deal with erasure.
        # offset = 2*(dx-1)*(2*dy-1) + 2*(dy-1)*(2*dx-1) # num of entangling gates. For erasure after every entangling gate
        # offset = 0 # for erasure only after initialization
        for i in range(m):
            circ.append('DETECTOR', [stim.target_rec(-(i+1)),
                                    stim.target_rec(-(i+1+m+offset))])
    else:
        if basis == 'X':
            check_ixs = [int((j*(2*dx-1)+i)/2) for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==0)]
            for check_ix in check_ixs:
                circ.append('DETECTOR', [stim.target_rec(-(m-check_ix))])
        else:
            check_ixs = [int((j*(2*dx-1)+i)/2) for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==1)]
            for check_ix in check_ixs:
                circ.append('DETECTOR', [stim.target_rec(-(m-check_ix))])


def repeat_block(dx: int, dy: int, offset=0, **kwargs) -> stim.Circuit:
    circ2 = stim.Circuit()
    circ2.append('TICK')
    checks(circ2, dx, dy)
    measure_checks(circ2, dx, dy, t_compare=True, offset=offset)
    return circ2


def measure_data(circ: stim.Circuit, dx: int, dy:int, basis='Z'):
    n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data
    m = int(0.5*(2*dx-1)*(2*dy-1))  # num ancilla
    # Measure data
    (meas1, meas2) = ('MZ', 'MX') if basis == 'X' else ('MX', 'MZ')
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if (i+j)%2 == 0:
                if i%2==0:
                    circ.append(meas1, [j*(2*dx-1)+i])
                else:
                    circ.append(meas2, [j*(2*dx-1)+i])

    anc_ix = 0
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if (i + j) % 2 == 1:
                if (basis == 'X' and i % 2 == 0) or (basis == 'Z' and i % 2 == 1):
                    detector_targets = [stim.target_rec(-(n+m-anc_ix))] # ancilla measurement from previous round
                    # now we will compare it to data measurements in the final round (neighbors of the ancilla)
                    if j != 2 * dy - 2:
                        data_ix = int(((j+1)*(2*dx-1)+i)/2)  # up
                        detector_targets.append(stim.target_rec(-(n-data_ix)))
                    if i != 2 * dx - 2:
                        data_ix = int((j*(2*dx-1)+(i+1))/2)  # right
                        detector_targets.append(stim.target_rec(-(n-data_ix)))
                    if i != 0:
                        data_ix = int((j*(2*dx-1)+(i - 1))/2)  # left
                        detector_targets.append(stim.target_rec(-(n-data_ix)))
                    if j != 0:
                        data_ix = int(((j-1)*(2*dx-1)+i)/2)  # down
                        detector_targets.append(stim.target_rec(-(n-data_ix)))
                    circ.append('DETECTOR', detector_targets)
                anc_ix += 1


def logical(circ: stim.Circuit, dx: int, dy: int, basis='Z') -> None:
    n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data
    detector_targets = []

    if basis == 'X':
        for i in range(2*dx-1):
            if i % 2 == 0:  # data qubit
                data_ix = int(i / 2)
                detector_targets.append(stim.target_rec(-(n-data_ix)))
        circ.append('OBSERVABLE_INCLUDE', detector_targets, 0)
    else:
        assert basis == 'Z'
        for j in range(2*dy - 1):
            if j % 2 == 0:  # data qubit
                data_ix = int(j*(2*dx-1)/2)
                detector_targets.append(stim.target_rec(-(n-data_ix)))
        circ.append('OBSERVABLE_INCLUDE', detector_targets, 0)


def xzzx_circuit(cycles, dx: int, dy: int, basis='Z', **kwargs):
    circ = stim.Circuit()
    initialize(circ, dx, dy, basis=basis)
    checks(circ, dx, dy) # note that we do all gates in the preparation round and don't add noise. If you want to add noise to prep rounds, do only half of the gates!
    measure_checks(circ, dx, dy, basis=basis)
    repeat_circ = repeat_block(dx, dy, **kwargs) * (cycles - 1)
    # repeat_circ = noise(repeat_circ, **kwargs)
    circ += repeat_circ
    measure_data(circ, dx, dy, basis=basis)
    logical(circ, dx, dy, basis=basis)
    return circ


if __name__ == '__main__':
    dx = 3
    dy = 3
    # circuit = xzzx_circuit(max(dx, dy), dx, dy, basis='X', noise=biased_erasure_noise, p2q=0.1, bias=0.5, erasure_weight=0.2, biased_erasure=False, bias_preserving=True)
    # circuit = xzzx_circuit(max(dx, dy), dx, dy, basis='Z', noise=no_noise)
    # dem = circuit.detector_error_model(allow_gauge_detectors=True,decompose_errors=True, approximate_disjoint_errors=True)
    
    
    circuit = xzzx_circuit(cycles=max(dx,dy), dx=dx, dy=dy, basis='Z', noise=biased_erasure_noise,
                                p2q=0.1, bias=0, erasure_weight=1, biased_erasure=False,
                                bias_preserving=False)
    print(circuit)
    dem = circuit.detector_error_model(allow_gauge_detectors=True,decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True)
    #circuit.diagram("timeline-svg")
    # graph = pymatching.Matching.from_detector_error_model(dem)
    # graph = pymatching.Matching.to_networkx(graph)
    # draw_graph(graph)
    #print(dem)
    print(repr(dem))
    #dem.diagram("matchgraph-3d")
    
    # circuit = xzzx_circuit(max(dx, dy), dx, dy, noise=two_q_gate_err, p2q=0.2)
    # circuit = biased_erasure_noise(circuit, p2q=0.01, bias=1, erasure_weight=0, biased_erasure=False, bias_preserving=True)
    