import stim
import numpy as np
from BiasedErasure.main_code.LogicalCircuit import LogicalCircuit
from typing import Union, List, Optional

def get_data_ancilla_indices(dx: int, dy:int):
    ancilla_qubits = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1)]
    data_qubits =    [j*(2*dx-1)+i   for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0)]
    return data_qubits, ancilla_qubits

def data_ids(dx: int, dy: int):
    return [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0)]


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
    
    
def initialize(circ: LogicalCircuit, dx: int, dy: int, bias_preserving_gates: bool, basis='Z', move_duration: Optional[float] = 200):
    # Initialize data qubits
    measure_qubit_indices_odd = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==1)] # odd i
    measure_qubit_indices_even = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==0)] # even i
    data_qubit_indices_even = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and i%2==0)] # data qubits, even i
    data_qubit_indices_odd = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and i%2==1)] # data qubits, odd i
    
    if basis == 'Z':
        circ.append('MOVE_TO_NO_NOISE', measure_qubit_indices_even, 0)
        circ.append('MOVE_TO_ENTANGLING', measure_qubit_indices_odd + data_qubit_indices_even + data_qubit_indices_odd, 0)
        circ.append('RZ', sorted(measure_qubit_indices_odd + measure_qubit_indices_even + data_qubit_indices_even + data_qubit_indices_odd)) # reset all qubits to |0>
        circ.append('MOVE', measure_qubit_indices_odd + data_qubit_indices_even + data_qubit_indices_odd, move_duration) # get idling noise
        circ.append('H', sorted(data_qubit_indices_odd + measure_qubit_indices_odd)) # initialize odd ancillas + off data qubits in |+>
    elif basis == 'X':
        circ.append('MOVE_TO_NO_NOISE', measure_qubit_indices_odd, 0)
        circ.append('MOVE_TO_ENTANGLING', measure_qubit_indices_even + data_qubit_indices_even + data_qubit_indices_odd, 0)
        circ.append('RZ', sorted(measure_qubit_indices_even + measure_qubit_indices_odd + data_qubit_indices_even + data_qubit_indices_odd)) # reset all qubits to |0>
        circ.append('MOVE', measure_qubit_indices_even + data_qubit_indices_even + data_qubit_indices_odd, move_duration) # get idling noise
        circ.append('H', sorted(data_qubit_indices_even + measure_qubit_indices_even)) # initialize odd ancillas + off data qubits in |+>

    # if basis == 'Z':
    #     circ.append('MOVE_TO_NO_NOISE', measure_qubit_indices_even, 0)
    #     circ.append('MOVE_TO_ENTANGLING', measure_qubit_indices_odd + data_qubit_indices_even + data_qubit_indices_odd, 0)
    #     circ.append('RZ', measure_qubit_indices_odd + data_qubit_indices_even + data_qubit_indices_odd) # reset all data qubits + odd ancilla qubits to |0>
    #     circ.append('H', data_qubit_indices_odd + measure_qubit_indices_odd) # initialize odd ancillas + off data qubits in |+>
    # if basis == 'X':
    #     circ.append('MOVE_TO_NO_NOISE', measure_qubit_indices_odd, 0)
    #     circ.append('MOVE_TO_ENTANGLING', measure_qubit_indices_even + data_qubit_indices_even + data_qubit_indices_odd, 0)
    #     circ.append('RZ', measure_qubit_indices_even + data_qubit_indices_even + data_qubit_indices_odd) # reset all data qubits + odd ancilla qubits to |0>
    #     circ.append('H', data_qubit_indices_even + measure_qubit_indices_even) # initialize odd ancillas + off data qubits in |+>
        
    # Entangling gates
    rnd1, rnd2, rnd3, rnd4 = [], [], [], []
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if (i + j) % 2 == 1 and ((basis == 'X' and i % 2 == 0) or (basis == 'Z' and i % 2 == 1)):  # Iterate over ancilla qubits that are relevant.
                if j != 2 * dy - 2:
                    rnd1.extend([j*(2*dx-1)+i, (j+1)*(2*dx-1)+i])
                if i != 2 * dx - 2:
                    rnd2.extend([j*(2*dx-1)+i, j*(2*dx-1)+(i+1)])
                if i != 0:
                    rnd3.extend([j*(2*dx-1)+i, j*(2*dx-1)+(i - 1)])
                if j != 0:
                    rnd4.extend([j*(2*dx-1)+i, (j-1)*(2*dx-1)+i])
    
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
    
    
    
    # measure checks:
    m = int(0.5*(2*dx-1)*(2*dy-1))
    if basis == 'Z':
        circ.append('H', measure_qubit_indices_odd)
        circ.append('MOVE', measure_qubit_indices_odd, move_duration) # Put idling noise on all active qubits due to movement time
        circ.append('M', sorted(measure_qubit_indices_odd + measure_qubit_indices_even)) # measure all ancilla qubits
        circ.append('MOVE_TO_NO_NOISE', measure_qubit_indices_odd + measure_qubit_indices_even, 0) # all measured qubits are inactive now
        check_ixs = [int((j*(2*dx-1)+i)/2) for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==0)] # deterministic checks
        for check_ix in check_ixs:
            circ.append('DETECTOR', [stim.target_rec(-(m-check_ix))])
    else:
        circ.append('H', measure_qubit_indices_even)
        circ.append('MOVE', measure_qubit_indices_even, move_duration) # Put idling noise on all active qubits due to movement time
        circ.append('M', sorted(measure_qubit_indices_odd + measure_qubit_indices_even)) # measure all ancilla qubits
        circ.append('MOVE_TO_NO_NOISE', measure_qubit_indices_odd + measure_qubit_indices_even, 0) # all measured qubits are inactive now
        check_ixs = [int((j*(2*dx-1)+i)/2) for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==1)] # deterministic checks
        for check_ix in check_ixs:
            circ.append('DETECTOR', [stim.target_rec(-(m-check_ix))])
                
                
def measure_stabilizers(circ: LogicalCircuit, dx: int, dy: int, bias_preserving_gates: bool, move_duration: Optional[float] = 200, offset : int = 0):
    circ.offset = 0
    measure_qubit_indices_odd = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==1)]
    measure_qubit_indices_even = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==0)]
    data_qubit_indices_even = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and i%2==0)] # data qubits, even i
    data_qubit_indices_odd = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and i%2==1)] # data qubits, odd i
    
    circ.append('MOVE_TO_ENTANGLING', data_qubit_indices_even + data_qubit_indices_odd, 0)
    circ.append('MOVE_TO_ENTANGLING', measure_qubit_indices_odd + measure_qubit_indices_even, 0)
    circ.append('MOVE', measure_qubit_indices_odd + measure_qubit_indices_even, move_duration) # get idling errors
    circ.append('RZ', sorted(measure_qubit_indices_even + measure_qubit_indices_odd)) # reset all ancilla qubits in |0>
    circ.append('H', sorted(measure_qubit_indices_even + measure_qubit_indices_odd)) # initialize all ancilla qubits in |+>
    
    # Entangling gates
    rnd1, rnd2, rnd3, rnd4 = [], [], [], []
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if (i + j) % 2 == 1:  # Iterate over all ancilla qubits.
                if j != 2 * dy - 2:
                    rnd1.extend([j*(2*dx-1)+i, (j+1)*(2*dx-1)+i])
                if i != 2 * dx - 2:
                    rnd2.extend([j*(2*dx-1)+i, j*(2*dx-1)+(i+1)])
                if i != 0:
                    rnd3.extend([j*(2*dx-1)+i, j*(2*dx-1)+(i - 1)])
                if j != 0:
                    rnd4.extend([j*(2*dx-1)+i, (j-1)*(2*dx-1)+i])
    
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
    circ.append('MOVE', (), move_duration)
    # circ.append('TICK')
    
    # round 3 - CX:
    if bias_preserving_gates:
        circ.append('CX', rnd3)
    else:
        targets = rnd3[1::2]
        circ.append('H', targets)
        circ.append('CZ', rnd3)
        circ.append('H', targets)
    circ.append('MOVE', (), move_duration)
    # circ.append('TICK')
    
    # round 4 - CZ:
    circ.append('CZ', rnd4)
    circ.append('MOVE', (), move_duration)
    # circ.append('TICK')
    
    
    measure_qubit_indices_odd = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==1)] # ancilla qubits, odd i
    measure_qubit_indices_even = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==0)] # ancilla qubits, even i
    m = int(0.5*(2*dx-1)*(2*dy-1))
    circ.append('H', measure_qubit_indices_odd + measure_qubit_indices_even)
    circ.append('MOVE', measure_qubit_indices_odd + measure_qubit_indices_even, move_duration) # Put idling noise due to movement time
    circ.append('M', sorted(measure_qubit_indices_odd + measure_qubit_indices_even))
    circ.append('MOVE_TO_NO_NOISE', measure_qubit_indices_odd + measure_qubit_indices_even, 0) # all measured qubits are inactive now
    
    # Compare parity of check at t (now) vs t-1.
    # Add offset equal to number of entangling gates. One detector per gate to deal with erasure.
    for i in range(m):
        circ.append('DETECTOR', [stim.target_rec(-(i+1)),
                                stim.target_rec(-(i+1+m+circ.offset))])


def repeat_block(dx: int, dy: int, bias_preserving_gates : bool, error_model: dict, offset=0, **kwargs) -> stim.Circuit:
    data_qubits, ancilla_qubits = get_data_ancilla_indices(dx,dy)
    qubit_indices = data_qubits + ancilla_qubits
    circ2 = LogicalCircuit(qubit_indices = qubit_indices, **error_model)
    
    # circ2 = LogicalCircuit(qubit_indices = qubit_indices, initialize_circuit=False,
    #                         idle_loss_rate=error_model["idle_loss_rate"],
    #                         idle_error_rate=error_model["idle_error_rate"],
    #                         entangling_error_rate=error_model["entangling_error_rate"],
    #                         entangling_loss_rate=error_model["entangling_loss_rate"],
    #                         reset_error_rate=error_model["reset_error_rate"], measurement_error_rate=error_model["measurement_error_rate"],
    #                         single_qubit_error_rate=error_model["single_qubit_error_rate"],
    #                         reset_loss_rate=error_model["reset_loss_rate"])
    circ2.append('TICK')
    measure_stabilizers(circ2, dx, dy, offset=offset, bias_preserving_gates=bias_preserving_gates)
    return circ2


def measure_data(circ: LogicalCircuit, dx: int, dy:int, basis='Z', move_duration: Optional[float] = 200):
    # last round, measure data qubits and make them not active anymore
    # measure_qubit_indices_odd = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==1)]
    # measure_qubit_indices_even = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 1 and i%2==0)]
    circ.offset = 0
    data_qubit_indices_even = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and i%2==0)] # data qubits, even i
    data_qubit_indices_odd = [j*(2*dx-1)+i for j in range(2*dy-1) for i in range(2*dx-1) if ((i+j)%2 == 0 and i%2==1)] # data qubits, odd i
    
    n = int(np.ceil(0.5*(2*dx-1)*(2*dy-1)))  # num data
    m = int(0.5*(2*dx-1)*(2*dy-1))  # num ancilla
    circ.append('MOVE_TO_ENTANGLING', data_qubit_indices_even + data_qubit_indices_odd, 0) # make sure they are in the entangling zone for the gate and measurement errors.
    # circ.append('MOVE', data_qubit_indices_even + data_qubit_indices_odd, move_duration) # no idling noise because we take the picture in the entangling zone.
    # Measure data
    # (meas1, meas2) = ('MZ', 'MX') if basis == 'Z' else ('MX', 'MZ')
    if basis == 'Z':
        for j in range(2*dy-1):
            for i in range(2*dx-1):
                if (i+j)%2 == 0: # data qubit
                    if i%2==0: # even data
                        circ.append('MZ', [j*(2*dx-1)+i])
                    else: # odd data
                        circ.append('H', [j*(2*dx-1)+i])
                        circ.append('MZ', [j*(2*dx-1)+i])
    elif basis == 'X':
        for j in range(2*dy-1):
            for i in range(2*dx-1):
                if (i+j)%2 == 0: # data qubit
                    if i%2==0: # even data
                        circ.append('H', [j*(2*dx-1)+i])
                        circ.append('MZ', [j*(2*dx-1)+i])
                    else: # odd data
                        circ.append('MZ', [j*(2*dx-1)+i])

    circ.append('MOVE_TO_NO_NOISE', data_qubit_indices_even + data_qubit_indices_odd, 0) # all measured qubits are inactive now
    
    # take care of detectors:
    anc_ix = 0
    for j in range(2*dy-1):
        for i in range(2*dx-1):
            if (i + j) % 2 == 1:
                if (basis == 'Z' and i % 2 == 0) or (basis == 'X' and i % 2 == 1):
                    detector_targets = [stim.target_rec(-(n+m-anc_ix+circ.offset))] # ancilla measurement from previous round
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

    if basis == 'Z':
        for i in range(2*dx-1):
            if i % 2 == 0:  # data qubit
                data_ix = int(i / 2)
                detector_targets.append(stim.target_rec(-(n-data_ix)))
        circ.append('OBSERVABLE_INCLUDE', detector_targets, 0)
    else:
        assert basis == 'X'
        for j in range(2*dy - 1):
            if j % 2 == 0:  # data qubit
                data_ix = int(j*(2*dx-1)/2)
                detector_targets.append(stim.target_rec(-(n-data_ix)))
        circ.append('OBSERVABLE_INCLUDE', detector_targets, 0)


def xzzx_circuit_noisy(cycles, dx: int, dy: int, bias_preserving_gates:bool, basis='Z', **kwargs):

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
    # entangling_error_rate_yes_gate = (.002 / 4, .002 / 4, .0025 / 4, .002 / 4, 0, 0, 0, .002 / 4, 0, 0, 0, .0025 / 4, 0, 0, .005 / 4)
    move_duration = 200
    data_qubits, ancilla_qubits = get_data_ancilla_indices(dx,dy)
    qubit_indices = data_qubits + ancilla_qubits
    circ = LogicalCircuit(qubit_indices = qubit_indices, initialize_circuit=False, **error_model)
    
    # circ = stim.Circuit()
    offset = 0
    initialize(circ, dx, dy, basis=basis, bias_preserving_gates=bias_preserving_gates, move_duration=move_duration)
    repeat_circ = repeat_block(dx, dy, bias_preserving_gates=bias_preserving_gates, error_model=error_model, **kwargs) * (cycles - 1)
    circ += repeat_circ
    measure_data(circ, dx, dy, basis=basis, move_duration=move_duration)
    logical(circ, dx, dy, basis=basis)
    return circ


if __name__ == '__main__':
    dx = 3
    dy = 3
    # circuit = xzzx_circuit(max(dx, dy), dx, dy, basis='X', noise=biased_erasure_noise, p2q=0.1, bias=0.5, erasure_weight=0.2, biased_erasure=False, bias_preserving=True)
    # circuit = xzzx_circuit(max(dx, dy), dx, dy, basis='Z', noise=no_noise)
    # dem = circuit.detector_error_model(allow_gauge_detectors=True,decompose_errors=True, approximate_disjoint_errors=True)
    
    
    circuit = xzzx_circuit_noisy(cycles=max(dx,dy), dx=dx, dy=dy, basis='Z', bias_preserving_gates = True)
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
    