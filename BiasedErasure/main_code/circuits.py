import numpy as np
import random
import os
import qec
import pymatching, stim
import random
import math
import time
import sinter
import copy
from BiasedErasure.main_code.XZZX import XZZX
from BiasedErasure.main_code.noise_channels import biased_erasure_noise_MBQC
from BiasedErasure.main_code.LogicalCircuitMBQC import LogicalCircuitMBQC
from BiasedErasure.main_code.LogicalCircuit import LogicalCircuit 
from BiasedErasure.main_code.GenerateLogicalCircuit import GenerateLogicalCircuit

def memory_experiment_surface_new(dx, dy, code, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, erasure_ratio, num_logicals=1, 
                                logical_basis='X', biased_pres_gates = False, ordering = 'fowler', loss_detection_method = 'FREE', 
                                loss_detection_frequency = 1, atom_array_sim=False, replace_H_Ry=False, xzzx=False, noise_params={}, printing=False, circuit_index = 0):
    """ This circuit simulated 1 logical qubits, a memory experiment with QEC cycles. We take perfect initialization and measurement and put noise only on the QEC cycles part."""
    assert logical_basis in ['X', 'Z'] # init and measurement basis for the single qubit logical state
    if printing:
        print(f"entangling Pauli error rate = {entangling_gate_error_rate}, entangling loss rate = {entangling_gate_loss_rate}")
    
    ### Added by SG to allow different orderings for different rounds
    if (type(ordering) is list) or (type(ordering) is np.ndarray):
        if len(ordering) != QEC_cycles:
            ordering = [ordering[0]]*QEC_cycles
            if printing:
                print(f"Incorrect number of orderings given. Defaulting to the first value: {ordering}")
        else:
            if printing:
                print(f"Using orderings: {ordering}")
    else:
        ordering = [ordering]*QEC_cycles
        if printing:
            print(f"Using orderings: {ordering}")
    ####

    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(dx, dy) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(dx, dy) for _ in range(num_logicals)]
    
    if atom_array_sim:
        # playing with error model: # it works!!! :)
        # entangling_gate_loss_rate = 0.06
        # single_qubit_loss_rate = 0.06
        # lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
        #                         loss_noise_scale_factor=1, spam_noise_scale_factor=1,
        #                         gate_noise_scale_factor=1, idle_noise_scale_factor=1,
        #                         atom_array_sim = atom_array_sim,
        #                         entangling_gate_loss_rate=entangling_gate_loss_rate,
        #                         single_qubit_loss_rate=single_qubit_loss_rate
        #                         )
        # scale_factor = 1
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                            loss_noise_scale_factor=1, spam_noise_scale_factor=1,
                            gate_noise_scale_factor=1, idle_noise_scale_factor=1,
                            atom_array_sim = atom_array_sim, replace_H_Ry=replace_H_Ry, circuit_index = circuit_index, **noise_params)
        # lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0; lc.spam_noise_scale_factor = 0; lc.idle_noise_scale_factor = 0 # debugging, without noise
        
    else:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=0,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=0,
                                entangling_gate_error_rate=entangling_gate_error_rate, 
                                entangling_gate_loss_rate=entangling_gate_loss_rate,
                                erasure_ratio = erasure_ratio,
                                atom_array_sim = atom_array_sim)
    
    #  initialization step:
    start_time = time.time()
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
    if logical_basis == 'X':
        lc.append(qec.surface_code.prepare_plus_no_gates, list(range(0, len(logical_qubits))), xzzx=xzzx)
    
    elif logical_basis == 'Z':
        lc.append(qec.surface_code.prepare_zero_no_gates, list(range(0, len(logical_qubits))), xzzx=xzzx)
        
    if not atom_array_sim: lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
        
    # print(f"initializing the Logical circuit took: {time.time() - start_time:.6f}s")

    # QEC rounds:
    start_time = time.time()
    SWAP_round_index = 0; SWAP_round_type = 'even'; SWAP_round = False
    for round_ix in range(QEC_cycles):
        if loss_detection_method == 'SWAP' and ((round_ix+1)%loss_detection_frequency == 0): # check if its a SWAP round:
            SWAP_round = True
            SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
            SWAP_round_index += 1
        

        put_detectors = False if round_ix == 0 else True
        init_round = True if round_ix == 0 else False
        # if not init_round:
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round
        if xzzx: # measure xzzx construction, meaning without the H at the end and beginning of round on data qubits.
            
            # new version of stabilizer checks to fix weight=2 checks:
            lc.append(qec.surface_code.measure_stabilizers_xzzx_weight2_new_ver, list(range(len(logical_qubits))), order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type, compare_with_previous=True, put_detectors = put_detectors, logical_basis=logical_basis, init_round=init_round, automatic_detectors=False) # append QEC rounds

            # lc.append(qec.surface_code.measure_stabilizers_xzzx, list(range(len(logical_qubits))), order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type, compare_with_previous=True, put_detectors = put_detectors, logical_basis=logical_basis, init_round=init_round, automatic_detectors=False) # append QEC rounds
        else:
            lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type, compare_with_previous=True, put_detectors = put_detectors, logical_basis=logical_basis, init_round=init_round, automatic_detectors=False) # append QEC rounds
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round
    
    # print(f"adding all QEC rounds took: {time.time() - start_time:.6f}s")

    # logical measurement step:
    start_time = time.time()
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
    no_ancillas = True if QEC_cycles==0 else False # added by SG
    if logical_basis == 'X':
        lc.append(qec.surface_code.measure_x, list(range(len(logical_qubits))), observable_include=True, xzzx=xzzx, automatic_detectors=False, no_ancillas = no_ancillas)
    
    elif logical_basis == 'Z':
        lc.append(qec.surface_code.measure_z, list(range(len(logical_qubits))), observable_include=True, xzzx=xzzx, automatic_detectors=False, no_ancillas = no_ancillas)
    
    # print(f"final measurement round took: {time.time() - start_time:.6f}s")
    # print(lc)

    return lc


def memory_experiment_surface(dx, dy, code, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, erasure_ratio, num_logicals=1, logical_basis='X', biased_pres_gates = False, ordering = 'fowler', loss_detection_method = 'FREE', loss_detection_frequency = 1, atom_array_sim=False):
    """ This circuit simulated 1 logical qubits, a memory experiment with QEC cycles. We take perfect initialization and measurement and put noise only on the QEC cycles part."""
    assert logical_basis in ['X', 'Z'] # init and measurement basis for the single qubit logical state
    print(f"entangling Pauli error rate = {entangling_gate_error_rate}, entangling loss rate = {entangling_gate_loss_rate}")
    
    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(dx, dy) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(dx, dy) for _ in range(num_logicals)]
    
    if atom_array_sim:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=1,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=1,
                                atom_array_sim = atom_array_sim)
    else:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=0,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=0,
                                entangling_gate_error_rate=entangling_gate_error_rate, 
                                entangling_gate_loss_rate=entangling_gate_loss_rate,
                                erasure_ratio = erasure_ratio,
                                atom_array_sim = atom_array_sim)
    
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
    if logical_basis == 'X':
        lc.append(qec.surface_code.prepare_plus, list(range(0, len(logical_qubits))), order=ordering, with_cnot=biased_pres_gates)
        if not atom_array_sim: lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
        
        SWAP_round_index = 0; SWAP_round_type = 'none'; SWAP_round = False
        for round_ix in range(QEC_cycles):
            if loss_detection_method == 'SWAP' and ((round_ix+1)%loss_detection_frequency == 0): # check if its a SWAP round:
                SWAP_round = True
                SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
                SWAP_round_index += 1
            
            lc.append_from_stim_program_text("""TICK""") # starting a QEC round
            lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order=ordering, with_cnot=biased_pres_gates, compare_with_previous=True, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type) # append QEC rounds
            lc.append_from_stim_program_text("""TICK""") # starting a QEC round
        if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
        lc.append(qec.surface_code.measure_x, list(range(len(logical_qubits))), observable_include=True)

    elif logical_basis == 'Z':
        lc.append(qec.surface_code.prepare_zero, list(range(0, len(logical_qubits))), order=ordering, with_cnot=biased_pres_gates)
        if not atom_array_sim: lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
        
        SWAP_round_index = 0; SWAP_round_type = 'none'; SWAP_round = False
        for round_ix in range(QEC_cycles):
            if loss_detection_method == 'SWAP' and ((round_ix+1)%loss_detection_frequency == 0): # check if its a SWAP round:
                SWAP_round = True
                SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
                SWAP_round_index += 1
        
            lc.append_from_stim_program_text("""TICK""") # starting a QEC round
            lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order=ordering, with_cnot=biased_pres_gates, compare_with_previous=True, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type) # append QEC rounds
            lc.append_from_stim_program_text("""TICK""") # starting a QEC round
        if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
        lc.append(qec.surface_code.measure_z, list(range(len(logical_qubits))), observable_include=True)
    
    return lc

def memory_experiment_MBQC(dx, dy, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, erasure_ratio, logical_basis='X', biased_pres_gates = False, atom_array_sim=False):
    """ This circuit simulated 1 logical qubits, a memory experiment with QEC cycles. We take perfect initialization and measurement and put noise only on the QEC cycles part.
    We are using the XZZX cluster state. cycles=c --> 2*c + 1 layers in the cluster state.
    
    """
    xzzx_instance = XZZX()  # Assuming XZZX has a constructor that can be called without arguments

    init_circuit = xzzx_instance.get_circuit(cycles=QEC_cycles, dx=dx, dy=dy, basis=logical_basis,
                                        offset=0, architecture = 'MBQC', bias_preserving_gates = biased_pres_gates, 
                                        atom_array_sim = atom_array_sim)
    
    # TODO: check bug
    if atom_array_sim:
        print("Need to build this feature in the new framework. It works only in the old framework.")
    else:
        noisy_circuit, loss_probabilities, potential_lost_qubits = biased_erasure_noise_MBQC(init_circuit, dx=dx, dy=dy,
                                        entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate, 
                                        erasure_weight=erasure_ratio, bias_preserving = biased_pres_gates)

    lc = LogicalCircuitMBQC(noisy_circuit, loss_probabilities, potential_lost_qubits)

    return lc



    
def Steane_QEC_circuit(dx, dy, code, Steane_type, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, erasure_ratio, logical_basis='X',
                    biased_pres_gates = False, loss_detection_on_all_qubits = True, atom_array_sim=False, ancilla1_for_preselection: bool = False,
                    ancilla2_for_preselection: bool = False, obs_pos: int = None):
    assert logical_basis in ['X', 'Z'] # measurement basis for final logical state
    assert Steane_type in ['Regular', 'SWAP'] # measurement basis for final logical state
    num_logicals = 3
    # Steane_type can be 'Regular' or 'swap'.
    # q0: logical. q1: |+> ancilla. q2: |0> ancilla.
    
    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(dx, dy) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(dx, dy) for _ in range(num_logicals)]
    
    if atom_array_sim:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=1,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=1,
                                atom_array_sim = atom_array_sim)
    else:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=0,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=0,
                                entangling_gate_error_rate=entangling_gate_error_rate, 
                                entangling_gate_loss_rate=entangling_gate_loss_rate,
                                erasure_ratio = erasure_ratio,
                                atom_array_sim = atom_array_sim)
        
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
    # First prepare all logical qubits:
    if logical_basis == 'X':
        lc.append(qec.surface_code.prepare_plus, [0])
    elif logical_basis == 'Z':
        lc.append(qec.surface_code.prepare_zero, [0])
    lc.append(qec.surface_code.prepare_plus, [1])
    lc.append(qec.surface_code.prepare_zero, [2])

    # FT preparation of logical ancilla qubits:
    for logical_ancilla_ix in [1,2]:
        if not atom_array_sim: lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
        for _ in range(obs_pos):
            lc.append_from_stim_program_text("""TICK""") # starting a QEC round
            lc.append(qec.surface_code.measure_stabilizers, [logical_ancilla_ix], order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
            lc.append_from_stim_program_text("""TICK""") # starting a QEC round

        lc.spam_noise_scale_factor = 0  # TODO: check this. also do we need to set "lc.loss_noise_scale_factor = 0" ?
        lc.gate_noise_scale_factor = 0
        lc.loss_noise_scale_factor = 0

        if (ancilla1_for_preselection and logical_ancilla_ix == 1):
            lq = lc.logical_qubits[logical_ancilla_ix]
            physical_data_qubit_layout = lq.data_qubits.reshape(lq.dy, lq.dx)
            logical_x = physical_data_qubit_layout[lq.dy // 2, :]
            logical_x_operator = []
            for _ in range(len(logical_x)):
                logical_x_operator.append(stim.target_x(logical_x[_]))
                if _ != len(logical_x) - 1:
                    logical_x_operator.append(stim.target_combiner())
            lc.append('MPP', logical_x_operator)  # TODO: check this.
            lc.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], lc.num_observables)
        if (ancilla2_for_preselection and logical_ancilla_ix == 2):
            lc.append('MPP', lc.logical_qubits[logical_ancilla_ix].logical_z_operator)  # TODO: check this.
            lc.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], lc.num_observables)

        lc.spam_noise_scale_factor = 0
        lc.gate_noise_scale_factor = 1
        lc.loss_noise_scale_factor = 1

        for _ in range(QEC_cycles - obs_pos):
            lc.append_from_stim_program_text("""TICK""") # starting a QEC round
            lc.append(qec.surface_code.measure_stabilizers, [logical_ancilla_ix], order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
            lc.append_from_stim_program_text("""TICK""") # starting a QEC round

        if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0  # PB: what's the purpose of this line?
    
    
    if not atom_array_sim: lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
    
    
    if Steane_type == 'Regular':
        # Entangle 0 and 1:
        lc.append(qec.surface_code.global_h, [1], move_duration=0)
        lc.append(qec.surface_code.global_cz, [0,1], move_duration=0)
        lc.append(qec.surface_code.global_h, [1], move_duration=0)
        lc.append(qec.surface_code.measure_z, [1], observable_include=False)
        # Entangle 0 and 2:
        lc.append(qec.surface_code.global_h, [0], move_duration=0)
        lc.append(qec.surface_code.global_cz, [0,2], move_duration=0)
        lc.append(qec.surface_code.global_h, [0], move_duration=0)
        lc.append(qec.surface_code.measure_x, [2], observable_include=False)
    
    elif Steane_type == 'SWAP':
        # Entangle 0 and 1:
        # CX 1-->0
        lc.append(qec.surface_code.global_h, [0], move_duration=0)
        lc.append(qec.surface_code.global_cz, [0,1], move_duration=0)
        lc.append(qec.surface_code.global_h, [0], move_duration=0)
        # noiseless CX 0-->1:
        lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
        lc.append(qec.surface_code.global_h, [1], move_duration=0)
        lc.append(qec.surface_code.global_cz, [0,1], move_duration=0)
        lc.append(qec.surface_code.global_h, [1], move_duration=0)
        lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
        
        lc.append(qec.surface_code.measure_z, [0], observable_include=False)
        
        # Entangle 1 and 2 
        # CX 1-->2:
        lc.append(qec.surface_code.global_h, [2], move_duration=0)
        lc.append(qec.surface_code.global_cz, [1,2], move_duration=0)
        lc.append(qec.surface_code.global_h, [2], move_duration=0)
        
        # noiseless CX 2-->1:
        lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
        lc.append(qec.surface_code.global_h, [1], move_duration=0)
        lc.append(qec.surface_code.global_cz, [1,2], move_duration=0)
        lc.append(qec.surface_code.global_h, [1], move_duration=0)
        lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
        
        lc.append(qec.surface_code.measure_x, [1], observable_include=False)
        
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
    # Final measurement of the logical qubit:
    final_logical_ix = 0 if Steane_type == 'Regular' else 2
    observable_include = False if (ancilla1_for_preselection or ancilla2_for_preselection) else True
    if logical_basis == 'X':
        lc.append(qec.surface_code.measure_x, [final_logical_ix], observable_include=observable_include)

    elif logical_basis == 'Z':
        lc.append(qec.surface_code.measure_z, [final_logical_ix], observable_include=observable_include)

    return lc


def steane_ancilla_prep(dx, dy, code, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, erasure_ratio, basis='X',
                    biased_pres_gates = False, atom_array_sim=False, obs_pos: int = None):
    assert basis in ['X', 'Z'] # measurement basis for final logical state
    num_logicals = 3
    logical_ix = 1 if basis == 'X' else 2
    # q0: logical. q1: |+> ancilla. q2: |0> ancilla.
    
    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(dx, dy) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(dx, dy) for _ in range(num_logicals)]
    
    if atom_array_sim:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=1,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=1,
                                atom_array_sim = atom_array_sim)
    else:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=0,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=0,
                                entangling_gate_error_rate=entangling_gate_error_rate, 
                                entangling_gate_loss_rate=entangling_gate_loss_rate,
                                erasure_ratio = erasure_ratio,
                                atom_array_sim = atom_array_sim)
        
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    

    # First prepare all logical qubits:
    lc.append(qec.surface_code.prepare_zero, [0])
    lc.append(qec.surface_code.prepare_plus, [1])
    lc.append(qec.surface_code.prepare_zero, [2])
    # if logical_ix == 1:
    #     lc.append(qec.surface_code.prepare_plus, [1])
    # else:
    #     lc.append(qec.surface_code.prepare_zero, [2])

    # FT preparation of logical ancilla qubits:
    if not atom_array_sim: lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
    for _ in range(obs_pos):
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round
        lc.append(qec.surface_code.measure_stabilizers, [logical_ix], order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round

    lc.spam_noise_scale_factor = 0 
    lc.gate_noise_scale_factor = 0
    lc.loss_noise_scale_factor = 0

    if basis == 'X':
        lq = lc.logical_qubits[logical_ix]
        physical_data_qubit_layout = lq.data_qubits.reshape(lq.dy, lq.dx)
        logical_x = physical_data_qubit_layout[lq.dy // 2, :]
        logical_x_operator = []
        for _ in range(len(logical_x)):
            logical_x_operator.append(stim.target_x(logical_x[_]))
            if _ != len(logical_x) - 1:
                logical_x_operator.append(stim.target_combiner())
        lc.append('MPP', logical_x_operator)
        lc.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], lc.num_observables)
    else:
        lc.append('MPP', lc.logical_qubits[logical_ix].logical_z_operator)
        lc.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], lc.num_observables)

    lc.spam_noise_scale_factor = 0
    lc.gate_noise_scale_factor = 1
    lc.loss_noise_scale_factor = 1

    for _ in range(QEC_cycles - obs_pos):
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round
        lc.append(qec.surface_code.measure_stabilizers, [logical_ix], order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round

    return lc


def GHZ_experiment_Surface(dx, dy, order, num_logicals, code, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, erasure_ratio, logical_basis='X', biased_pres_gates = False, loss_detection_on_all_qubits = True, atom_array_sim=False):
    assert logical_basis in ['X', 'Z'] # measurement basis for final logical state
    print(f"order = {order}")
    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(dx, dy) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(dx, dy) for _ in range(num_logicals)]
    
    if atom_array_sim:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=1,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=1,
                                atom_array_sim = atom_array_sim)
    else:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                                loss_noise_scale_factor=1, spam_noise_scale_factor=0,
                                gate_noise_scale_factor=1, idle_noise_scale_factor=0,
                                entangling_gate_error_rate=entangling_gate_error_rate, 
                                entangling_gate_loss_rate=entangling_gate_loss_rate,
                                erasure_ratio = erasure_ratio,
                                atom_array_sim = atom_array_sim)
        
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
    # First prepare all logical qubits in |0>
    lc.append(qec.surface_code.prepare_zero, list(range(0, len(logical_qubits))))

    # Rotate noiselessly
    lc.append(qec.surface_code.rotate_code, 0)

    # Hadamard all qubits
    lc.append(qec.surface_code.global_h, list(range(len(logical_qubits))), move_duration=0)

    # Entangle all qubits:
    for _ in range(len(order)):
        lc.append(qec.surface_code.global_cz, [order[_][0], order[_][1]], move_duration=0)
        lc.append(qec.surface_code.global_h, order[_][1], move_duration=0)
        
    # add QEC cycles:
    if not atom_array_sim: lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
    for cycle_num in range(QEC_cycles):
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round
        if loss_detection_on_all_qubits: # LD rounds on all logical qubits after every gate
            lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
        else: # only on the logical qubits who did the gate:
            lc.append(qec.surface_code.measure_stabilizers, [order[_][0], order[_][1]], order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
        
    if logical_basis == 'X':
        lc.append(qec.surface_code.measure_x, list(range(len(logical_qubits))), observable_include=False)
        lc.append('MOVE_TO_NO_NOISE', lc.qubit_indices, 0)
        global_x = []
        for index in range(len(lc.logical_qubits)):
            global_x += lc.logical_qubits[index].logical_x_operator + [stim.GateTarget(stim.target_combiner())]
        lc.append('MPP', global_x[:-1])
        lc.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], 0)

    elif logical_basis == 'Z':
        lc.append(qec.surface_code.measure_z, list(range(len(logical_qubits))), observable_include=False)
        lc.append('MOVE_TO_NO_NOISE', lc.qubit_indices, 0)

        for index in range(len(logical_qubits) - 1):
            lc.append('MPP', lc.logical_qubits[index].logical_z_operator + [stim.GateTarget(stim.target_combiner())] +
                    lc.logical_qubits[index + 1].logical_z_operator)
            lc.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], lc.num_observables)

    return lc
    
    
    
def random_logical_algorithm(code, num_logicals, depth, distance, n_r, bias_ratio, erasure_ratio, phys_err, output_dir):
    # n_r = num rounds. if < 1, every 1/n_r layers we have 1 QEC round. if >=1, every layer we have n_r QEC rounds.
    def generate_folder_name(num_logicals, depth, distance, n_r, bias_ratio, erasure_ratio, phys_err):
        if bias_ratio >= 1:
            bias_ratio = int(bias_ratio)
        folder_name = f"random_deep_circuits/random_algorithm__n{num_logicals}__depth{depth}/random_algorithm__n{num_logicals}__depth{depth}__distance{distance}__nr{n_r}__p{phys_err}__bias{bias_ratio}__erasure{erasure_ratio}"
        return folder_name

    folder_name = generate_folder_name(num_logicals, depth, distance, n_r, bias_ratio, erasure_ratio, phys_err)
    save_dir = f"{output_dir}/{folder_name}"
    
    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(distance, distance) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(distance, distance) for _ in range(num_logicals)]
        
        
    # Load the NumPy arrays
    loss_probabilities = np.load(os.path.join(save_dir, 'loss_probabilities.npy'))
    potential_lost_qubits = np.load(os.path.join(save_dir, 'potential_lost_qubits.npy'))

    # Load the .stim file
    stim_file_path = os.path.join(save_dir, 'logical_circuit.stim')
    logical_circuit = stim.Circuit()
    with open(stim_file_path, 'r') as f:
        circuit_text = f.read()
    logical_circuit.append_from_stim_program_text(circuit_text)

    lc = GenerateLogicalCircuit(logical_qubits, logical_circuit, loss_probabilities, potential_lost_qubits)
    
    return lc

# def memory_experiment_xzzx(d, cycles, entangling_gate_error_rate, entangling_gate_loss_rate, num_logicals=1, logical_basis='X', biased_pres_gates=False):
#     assert logical_basis in ['X', 'Z']
    
#     print(f"entangling Pauli error rate = {entangling_gate_error_rate}, entangling loss rate = {entangling_gate_loss_rate}")
#     logical_qubits = [qec.codes.xzzx_code.RotatedXZZXCode(d, d) for _ in range(num_logicals)]
#     lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
#                             loss_noise_scale_factor=1, spam_noise_scale_factor=0,
#                             gate_noise_scale_factor=1, idle_noise_scale_factor=0,
#                             entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate)
    
#     if logical_basis == 'X':
#         lc.append(qec.xzzx_code.prepare_plus, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates)
#         for cycle_num in range(cycles-1):
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#             lc.append(qec.xzzx_code.measure_stabilizers, list(range(len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
#         lc.append(qec.xzzx_code.measure_z, list(range(len(logical_qubits))), observable_include=True)

#     elif logical_basis == 'Z':
#         lc.append(qec.xzzx_code.prepare_zero, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates)
#         for cycle_num in range(cycles-1):
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#             lc.append(qec.xzzx_code.measure_stabilizers, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
#         lc.append(qec.xzzx_code.measure_z, list(range(len(logical_qubits))), observable_include=True)

#     return lc


# def memory_experiment_surface(d, code, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, num_logicals=1, logical_basis='X', biased_pres_gates = False):
#     """ This circuit simulated 1 logical qubits, a memory experiment with QEC cycles. We take perfect initialization and measurement and put noise only on the QEC cycles part."""
#     assert logical_basis in ['X', 'Z'] # init and measurement basis for the single qubit logical state
#     print(f"entangling Pauli error rate = {entangling_gate_error_rate}, entangling loss rate = {entangling_gate_loss_rate}")
    
#     if code == 'Rotated_Surface':
#         logical_qubits = [qec.surface_code.RotatedSurfaceCode(d, d) for _ in range(num_logicals)]
#     elif code == 'Surface':
#         logical_qubits = [qec.surface_code.SurfaceCode(d, d) for _ in range(num_logicals)]
        
#     lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
#                             loss_noise_scale_factor=1, spam_noise_scale_factor=0,
#                             gate_noise_scale_factor=1, idle_noise_scale_factor=0,
#                             entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate)
#     lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
#     if logical_basis == 'X':
#         lc.append(qec.surface_code.prepare_plus, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates)
#         lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
#         for round_ix in range(QEC_cycles):
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#             lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#         lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
#         lc.append(qec.surface_code.measure_x, list(range(len(logical_qubits))), observable_include=True)

#     elif logical_basis == 'Z':
#         lc.append(qec.surface_code.prepare_zero, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates)
#         lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
#         for round_ix in range(QEC_cycles):
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#             lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#         lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
#         lc.append(qec.surface_code.measure_z, list(range(len(logical_qubits))), observable_include=True)
#     return lc

# def memory_experiment_surface_theory(d, code, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, num_logicals=1, logical_basis='X', biased_pres_gates = False, loss_detection_method = 'FREE', loss_detection_frequency = 1, atom_array_sim=False):
#     """ This circuit simulated 1 logical qubits, a memory experiment with QEC cycles. We take perfect initialization and measurement and put noise only on the QEC cycles part."""
#     assert logical_basis in ['X', 'Z'] # init and measurement basis for the single qubit logical state
#     assert atom_array_sim == False
#     print(f"entangling Pauli error rate = {entangling_gate_error_rate}, entangling loss rate = {entangling_gate_loss_rate}")
    
#     if code == 'Rotated_Surface':
#         logical_qubits = [qec.surface_code.RotatedSurfaceCode(d, d) for _ in range(num_logicals)]
#     elif code == 'Surface':
#         logical_qubits = [qec.surface_code.SurfaceCode(d, d) for _ in range(num_logicals)]
    

#     lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
#                             loss_noise_scale_factor=1, spam_noise_scale_factor=0,
#                             gate_noise_scale_factor=1, idle_noise_scale_factor=0,
#                             entangling_gate_error_rate=entangling_gate_error_rate, 
#                             entangling_gate_loss_rate=entangling_gate_loss_rate,
#                             atom_array_sim = atom_array_sim)

    
#     lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
#     if logical_basis == 'X':
#         lc.append(qec.surface_code.prepare_plus, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates)
#         lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
        
#         SWAP_round_index = 0; SWAP_round_type = 'none'; SWAP_round = False
#         for round_ix in range(QEC_cycles):
#             if loss_detection_method == 'SWAP' and ((round_ix+1)%loss_detection_frequency == 0): # check if its a SWAP round:
#                 SWAP_round = True
#                 SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
#                 SWAP_round_index += 1
            
#             SWAP_round = False # cancel the SWAP operations
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#             lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type) # append QEC rounds
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#         lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
#         lc.append(qec.surface_code.measure_x, list(range(len(logical_qubits))), observable_include=True)

#     elif logical_basis == 'Z':
#         lc.append(qec.surface_code.prepare_zero, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates)
#         lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
#         for round_ix in range(QEC_cycles):
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#             lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#         lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
#         lc.append(qec.surface_code.measure_z, list(range(len(logical_qubits))), observable_include=True)
    
#     return lc


# def memory_experiment_surface_atom_array(d, code, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, num_logicals=1, logical_basis='X', biased_pres_gates = False, loss_detection_method = 'FREE', loss_detection_frequency = 1, atom_array_sim=False):
#     """ This circuit simulated 1 logical qubits, a memory experiment with QEC cycles. We take perfect initialization and measurement and put noise only on the QEC cycles part."""
#     assert logical_basis in ['X', 'Z'] # init and measurement basis for the single qubit logical state
#     assert atom_array_sim == True
#     print(f"entangling Pauli error rate = {entangling_gate_error_rate}, entangling loss rate = {entangling_gate_loss_rate}")
    
#     if code == 'Rotated_Surface':
#         logical_qubits = [qec.surface_code.RotatedSurfaceCode(d, d) for _ in range(num_logicals)]
#     elif code == 'Surface':
#         logical_qubits = [qec.surface_code.SurfaceCode(d, d) for _ in range(num_logicals)]
    

#     lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
#                                 loss_noise_scale_factor=1, spam_noise_scale_factor=1,
#                                 gate_noise_scale_factor=1, idle_noise_scale_factor=1,
#                                 atom_array_sim = atom_array_sim)

    
#     # lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
#     if logical_basis == 'X':
#         lc.append(qec.surface_code.prepare_plus, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates)
#         # lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
        
#         SWAP_round_index = 0; SWAP_round_type = 'none'; SWAP_round = False
#         for round_ix in range(QEC_cycles):
#             if loss_detection_method == 'SWAP' and ((round_ix+1)%loss_detection_frequency == 0): # check if its a SWAP round:
#                 SWAP_round = True
#                 SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
#                 SWAP_round_index += 1
            
#             SWAP_round = False # cancel the SWAP operations
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#             lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type) # append QEC rounds
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#         # lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
#         lc.append(qec.surface_code.measure_x, list(range(len(logical_qubits))), observable_include=True)

#     elif logical_basis == 'Z':
#         lc.append(qec.surface_code.prepare_zero, list(range(0, len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates)
#         # lc.loss_noise_scale_factor = 1; lc.gate_noise_scale_factor=1
#         for round_ix in range(QEC_cycles):
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#             lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order='fowler', with_cnot=biased_pres_gates, compare_with_previous=True) # append QEC rounds
#             lc.append_from_stim_program_text("""TICK""") # starting a QEC round
#         # lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
#         lc.append(qec.surface_code.measure_z, list(range(len(logical_qubits))), observable_include=True)
    
#     return lc


if __name__ == '__main__':
    pass

