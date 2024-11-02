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

############### Helper functions: ###############

def organize_ordering_array(ordering, QEC_cycles, printing=False):
    ### Allow different orderings for different rounds
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
    
    return ordering


def add_QEC_rounds(lc, logical_qubits, QEC_cycles, loss_detection_method, atom_array_sim, loss_detection_frequency, xzzx, ordering, biased_pres_gates, logical_basis):
    # QEC rounds:
    start_time = time.time()
    SWAP_round_index = 0
    for round_ix in range(QEC_cycles):
        SWAP_round_type = 'none'; SWAP_round = False
        if loss_detection_method == 'SWAP' and ((atom_array_sim and (round_ix+1)%loss_detection_frequency == 0) or (not atom_array_sim and (round_ix+1)%loss_detection_frequency == 0)): # check if its a SWAP round:
            SWAP_round = True
            SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
            SWAP_round_index += 1
        
        # print(f"round_ix = {round_ix}, SWAP_round = {SWAP_round}")
        put_detectors = True # GB's change - we always want to put detectors. Now init_round controls the type of detectors and not the variable put_detectors
        init_round = True if round_ix == 0 else False
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round
        if xzzx: # measure xzzx construction, meaning without the H at the end and beginning of round on data qubits.
            
            # new version of stabilizer checks to fix weight=2 checks:
            lc.append(qec.surface_code.measure_stabilizers_xzzx_weight2_new_ver, list(range(len(logical_qubits))), order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type, compare_with_previous=True, put_detectors = put_detectors, logical_basis=logical_basis, init_round=init_round, automatic_detectors=False) # append QEC rounds
            # old version:
            # lc.append(qec.surface_code.measure_stabilizers_xzzx, list(range(len(logical_qubits))), order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type, compare_with_previous=True, put_detectors = put_detectors, logical_basis=logical_basis, init_round=init_round, automatic_detectors=False) # append QEC rounds
        else:
            lc.append(qec.surface_code.measure_stabilizers, list(range(len(logical_qubits))), order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round = SWAP_round, SWAP_round_type=SWAP_round_type, compare_with_previous=True, put_detectors = put_detectors, logical_basis=logical_basis, init_round=init_round, automatic_detectors=False) # append QEC rounds
        lc.append_from_stim_program_text("""TICK""") # starting a QEC round
    
    # print(f"adding all QEC rounds took: {time.time() - start_time:.6f}s")



############### Circuits: ###############

def memory_experiment_surface_new(dx, dy, code, QEC_cycles, entangling_gate_error_rate, entangling_gate_loss_rate, erasure_ratio, num_logicals=1, 
                                logical_basis='X', biased_pres_gates = False, ordering = 'fowler', loss_detection_method = 'FREE', 
                                loss_detection_frequency = 1, atom_array_sim=False, replace_H_Ry=False, xzzx=False, noise_params={}, printing=False, circuit_index = '0', measure_wrong_basis=False):
    """ This circuit simulated 1 logical qubit, a memory experiment with QEC cycles. We take perfect initialization and measurement and put noise only on the QEC cycles part.
    If measure_wrong_basis = True: we are measuring in the opposite basis to initialization.
    """
    assert logical_basis in ['X', 'Z'] # init and measurement basis for the single qubit logical state
    assert num_logicals == 1
    
    if printing:
        print(f"entangling Pauli error rate = {entangling_gate_error_rate}, entangling loss rate = {entangling_gate_loss_rate}")
    
    
    ordering = organize_ordering_array(ordering=ordering, QEC_cycles=QEC_cycles)


    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(dx, dy) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(dx, dy) for _ in range(num_logicals)]
    
    if atom_array_sim:
        lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                            loss_noise_scale_factor=1, spam_noise_scale_factor=1,
                            gate_noise_scale_factor=1, idle_noise_scale_factor=1,
                            atom_array_sim = atom_array_sim, replace_H_Ry=replace_H_Ry, circuit_index = circuit_index, **noise_params)
        
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
    add_QEC_rounds(lc, logical_qubits, QEC_cycles, loss_detection_method, atom_array_sim, loss_detection_frequency, xzzx, ordering, biased_pres_gates, logical_basis)
    
    
    # logical measurement step:
    start_time = time.time()
    if not atom_array_sim: lc.loss_noise_scale_factor = 0; lc.gate_noise_scale_factor=0
    
    measurement_basis = 'X' if (measure_wrong_basis and logical_basis == 'Z') or (not measure_wrong_basis and logical_basis == 'X') else 'Z' # GB: new. measurement basis can be wrong if we want to
    
    no_ancillas = True if QEC_cycles==0 else False # added by SG
    if measurement_basis == 'X':
        lc.append(qec.surface_code.measure_x, list(range(len(logical_qubits))), observable_include=True, xzzx=xzzx, automatic_detectors=False, no_ancillas = no_ancillas)
    
    elif measurement_basis == 'Z':
        lc.append(qec.surface_code.measure_z, list(range(len(logical_qubits))), observable_include=True, xzzx=xzzx, automatic_detectors=False, no_ancillas = no_ancillas)
    
    # print(f"final measurement round took: {time.time() - start_time:.6f}s")
    # print(lc)

    return lc


def CX_experiment_surface(dx, dy, code, num_CX_per_layer_list, num_layers=3, num_logicals=2, logical_basis='X',
                          biased_pres_gates=False, ordering='fowler', loss_detection_method='FREE',
                          loss_detection_frequency=100, atom_array_sim=True, replace_H_Ry=False, xzzx=False,
                          noise_params={}, printing=False, circuit_index='0'):
    """ This circuit simulated 2 logical qubits, a logical CX experiment with QEC cycles."""

    assert num_logicals == 2
    assert atom_array_sim
    assert xzzx
    assert len(num_CX_per_layer_list) == num_layers

    def construct_detectors_data_qubits_measurement(meas_bases, QEC_cycles, num_CX_in_layer):
        # construct the 5 / 6 body operators between the logicals and with previous rounds (using data qubits).
        # we can use only half of the measure qubits, according to the meas basis of the data qubits.
        # if QEC_cycles > 0: we did QEC before, so we need to compare to previous ancilla qubits measurements.
        # if QEC_cycles == 0: no QEC at all. We only need to check Bell pairs between each set of 4 data qubits in each logical (8 body operator).
        # meas_bases = list of the 2 measurement bases for each logical.
        if QEC_cycles > 0:  # we did some QEC cycles before this transversal measurement

            if num_CX_in_layer % 2 == 0:  # no entanglement in this layer
                # build 5 body operators of data (t) measure qubit (t-1) - each logical separately
                for index in [0, 1]:
                    measure_qubits_set = measure_qubits_x if meas_bases[index] == 'X' else measure_qubits_z
                    for meas_q in measure_qubits_set:
                        # meas_q_type = 'X' if meas_q in measure_qubits_x else 'Z'
                        meas_q_type = meas_bases[index]
                        meas_q_logical = 0 if meas_q in np.concatenate(
                            (lc.logical_qubits[0].measure_qubits_x, lc.logical_qubits[0].measure_qubits_z)) else 1
                        check_ix = measure_qubits_list.index(meas_q)

                        # get the data qubits neighbors:
                        neighbors_data_q = [lc.logical_qubits[meas_q_logical].neighbor_from_index(physical_index=meas_q,
                                                                                                  which=direction) for
                                            direction in [0, 1, 2, 3]]
                        all_relevant_neighbor_qubits = sorted(
                            [neighbor for neighbor in neighbors_data_q if neighbor is not None])
                        data_qubits_list = data_qubits.tolist()
                        check_ixs = [data_qubits_list.index(i) for i in all_relevant_neighbor_qubits]
                        check_targets_data = [stim.target_rec(-(num_of_data_qubits - check_ix)) for check_ix in
                                              check_ixs]
                        # print(f"meas_q: {meas_q}, type: {meas_q_type}, meas_q_logical: {meas_q_logical}")

                        check_targets = check_targets_data + [
                            stim.target_rec(-(num_of_data_qubits + num_of_measure_qubits - check_ix))]

                        # print(f"meas_q: {meas_q}, check_targets: {check_targets}")
                        lc.append('DETECTOR', check_targets)
                        # print("detector added")

            else:  # there is entanglement, build 5/6 body operators of data (t) measure qubit (t-1)
                for index in [0, 1]:
                    measure_qubits_set = measure_qubits_x if meas_bases[index] == 'X' else measure_qubits_z
                    for meas_q in measure_qubits_set:
                        # meas_q_type = 'X' if meas_q in measure_qubits_x else 'Z'
                        meas_q_type = meas_bases[index]
                        meas_q_logical = 0 if meas_q in np.concatenate(
                            (lc.logical_qubits[0].measure_qubits_x, lc.logical_qubits[0].measure_qubits_z)) else 1
                        check_ix = measure_qubits_list.index(meas_q)

                        # get the data qubits neighbors:
                        neighbors_data_q = [lc.logical_qubits[meas_q_logical].neighbor_from_index(physical_index=meas_q,
                                                                                                  which=direction) for
                                            direction in [0, 1, 2, 3]]
                        all_relevant_neighbor_qubits = sorted(
                            [neighbor for neighbor in neighbors_data_q if neighbor is not None])
                        data_qubits_list = data_qubits.tolist()
                        check_ixs = [data_qubits_list.index(i) for i in all_relevant_neighbor_qubits]
                        check_targets_data = [stim.target_rec(-(num_of_data_qubits - check_ix)) for check_ix in
                                              check_ixs]
                        # print(f"meas_q: {meas_q}, type: {meas_q_type}, meas_q_logical: {meas_q_logical}")

                        if (
                                meas_q_type == 'X' and meas_q_logical == 0):  # X type for L0 or Z type for L1, 6 body operator:
                            check_targets = check_targets_data + [
                                stim.target_rec(-(num_of_data_qubits + num_of_measure_qubits - check_ix)),
                                stim.target_rec(-(num_of_data_qubits + int(num_of_measure_qubits / 2) - check_ix))]
                        elif (
                                meas_q_type == 'Z' and meas_q_logical == 1):  # X type for L0 or Z type for L1, 6 body operator:
                            check_targets = check_targets_data + [
                                stim.target_rec(-(num_of_data_qubits + num_of_measure_qubits - check_ix)),
                                stim.target_rec(
                                    -(num_of_data_qubits + int((3 / 2) * num_of_measure_qubits) - check_ix))]
                        else:  # X type for L1 or Z type for L0, 5 body operator:
                            check_targets = check_targets_data + [
                                stim.target_rec(-(num_of_data_qubits + num_of_measure_qubits - check_ix))]

                        # print(f"meas_q: {meas_q}, check_targets: {check_targets}")
                        lc.append('DETECTOR', check_targets)
                        # print("detector added")

        else:  # no QEC in this circuit.
            if num_CX_in_layer % 2 == 0:  # no entanglement in this layer
                # 4 body operators in each logical separately (sets of data qubits). Only on deterministic sets.
                for index in [0, 1]:
                    measure_qubits = lc.logical_qubits[index].measure_qubits_x if meas_bases[index] == 'X' else \
                    lc.logical_qubits[index].measure_qubits_z
                    for meas_q in measure_qubits:
                        neighbors = [
                            lc.logical_qubits[index].neighbor_from_index(physical_index=meas_q, which=direction) for
                            direction in [0, 1, 2, 3]]
                        all_relevant_data_qubits = sorted([neighbor for neighbor in neighbors if neighbor is not None])
                        data_qubits_list = data_qubits.tolist()
                        check_ixs = [data_qubits_list.index(i) for i in all_relevant_data_qubits]
                        check_targets = [stim.target_rec(-(num_of_data_qubits - check_ix)) for check_ix in check_ixs]
                        lc.append('DETECTOR', check_targets)

                # for (data_q0, data_q1) in zip(data_qubits_L0, data_qubits_L1):
                #     data_qubits_pair = [data_q0, data_q1]
                #     check_ixs = [data_qubits_list.index(i) for i in data_qubits_pair]
                #     check_targets = [stim.target_rec(-(num_of_data_qubits - check_ix)) for check_ix in check_ixs]
                #     lc.append('DETECTOR', check_targets)

            else:

                # build 8 body operators between sets of data qubits in each logical (only the relevant set according to measurement basis).
                measure_qubits_0 = lc.logical_qubits[0].measure_qubits_x if meas_bases[0] == 'X' else lc.logical_qubits[
                    0].measure_qubits_z
                measure_qubits_1 = lc.logical_qubits[1].measure_qubits_x if meas_bases[1] == 'X' else lc.logical_qubits[
                    1].measure_qubits_z
                for (meas_q0, meas_q1) in zip(measure_qubits_0, measure_qubits_1):
                    neighbors_L0 = [lc.logical_qubits[0].neighbor_from_index(physical_index=meas_q0, which=direction)
                                    for direction in [0, 1, 2, 3]]
                    neighbors_L1 = [lc.logical_qubits[1].neighbor_from_index(physical_index=meas_q1, which=direction)
                                    for direction in [0, 1, 2, 3]]
                    all_relevant_data_qubits = sorted(
                        [neighbor for neighbor in neighbors_L0 + neighbors_L1 if neighbor is not None])
                    data_qubits_list = data_qubits.tolist()

                    check_ixs = [data_qubits_list.index(i) for i in all_relevant_data_qubits]
                    check_targets = [stim.target_rec(-(num_of_data_qubits - check_ix)) for check_ix in check_ixs]
                    lc.append('DETECTOR', check_targets)

                # OLD: build 2 body operators between pairs of data qubits in each logical. -- deleted, we want the 8 body instead
                # for (data_q0, data_q1) in zip(data_qubits_L0, data_qubits_L1):
                #     data_qubits_pair = [data_q0, data_q1]
                #     check_ixs = [data_qubits_list.index(i) for i in data_qubits_pair]
                #     check_targets = [stim.target_rec(-(num_of_data_qubits - check_ix)) for check_ix in check_ixs]
                #     lc.append('DETECTOR', check_targets)

    
    # ordering = organize_ordering_array(ordering, QEC_cycles)


    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(dx, dy) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(dx, dy) for _ in range(num_logicals)]

    lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                        loss_noise_scale_factor=1, spam_noise_scale_factor=1,
                        gate_noise_scale_factor=1, idle_noise_scale_factor=1,
                        atom_array_sim=atom_array_sim, replace_H_Ry=replace_H_Ry, circuit_index=circuit_index,
                        circuit_type='CX', **noise_params)

    ###  initialization step:
    # initialize qubit 0 in |+> and qubit 1 in |0>:
    if num_layers > 1:  # we have QEC
        lc.append('MOVE_TO_STORAGE',
                  np.concatenate([lc.logical_qubits[i].measure_qubits for i in range(len(lc.logical_qubits))]), 1e-3)
    lc.append('MOVE_TO_ENTANGLING',
              np.concatenate([lc.logical_qubits[i].data_qubits for i in range(len(lc.logical_qubits))]), 1e-3)
    lc.append(qec.surface_code.prepare_plus_no_gates, [0], xzzx=xzzx)
    lc.append(qec.surface_code.prepare_zero_no_gates, [1], xzzx=xzzx)
    # lc.append_from_stim_program_text("""TICK""") # ending init

    ### Variables we are going to use:
    measure_qubits_x = np.sort(
        np.concatenate((lc.logical_qubits[0].measure_qubits_x, lc.logical_qubits[1].measure_qubits_x)))
    measure_qubits_z = np.sort(
        np.concatenate((lc.logical_qubits[0].measure_qubits_z, lc.logical_qubits[1].measure_qubits_z)))
    measure_qubits_L0 = np.sort(
        np.concatenate((lc.logical_qubits[0].measure_qubits_x, lc.logical_qubits[0].measure_qubits_z)))
    measure_qubits_L1 = np.sort(
        np.concatenate((lc.logical_qubits[1].measure_qubits_x, lc.logical_qubits[1].measure_qubits_z)))
    measure_qubits = np.sort(np.concatenate((measure_qubits_z, measure_qubits_x)))
    measure_qubits_list = measure_qubits.tolist()
    num_of_measure_qubits = len(measure_qubits)
    data_qubits_L0 = lc.logical_qubits[0].data_qubits
    data_qubits_L1 = lc.logical_qubits[1].data_qubits
    data_qubits = np.sort(np.concatenate((data_qubits_L0, data_qubits_L1)))
    data_qubits_list = data_qubits.tolist()
    num_of_data_qubits = len(data_qubits)

    ### Layers of CX and QEC:
    QEC_cycles = 0
    for round_ix in range(num_layers):
        # in each layer, we have num_CX_per_layer transversal CX gates, then 1 round of QEC. ordering = N.
        # lc.append_from_stim_program_text("""TICK""") # starting a QEC round

        num_CX_in_layer = num_CX_per_layer_list[round_ix]
        ### Transversal CX gates:
        for cx_ix in range(num_CX_in_layer):
            if cx_ix == 0:
                if xzzx:
                    lc.append(qec.surface_code.global_h_xzzx, [0], move_duration=1e-3,
                              sublattice='odd')  # apply this H only on the first layer
                    lc.append(qec.surface_code.global_h_xzzx, [1], move_duration=1e-3,
                              sublattice='even')  # apply this H only on the first layer
                else:
                    lc.append(qec.surface_code.global_h, [1],
                              move_duration=1e-3)  # apply this H only on the first layer
            # print('num_CX_in_layer', num_CX_in_layer, 10000 / num_CX_in_layer)
            lc.append(qec.surface_code.global_cz, [0, 1], move_duration=10000 / num_CX_in_layer)
            # print('yo', lc[-10:])
            # print(round_ix, cx_ix, lc.no_noise_zone, lc.storage_zone, lc.entangling_zone)

            if cx_ix == num_CX_in_layer // 2:  # new GB - add Y pulse on all qubits:
                lc.append('Y', lc.qubit_indices)

            if (cx_ix == num_CX_in_layer - 1) and (
                    round_ix < num_layers - 1):  # new GB - for last round_ix, we need to cancel out these pulses with the measurement
                if xzzx:
                    lc.append(qec.surface_code.global_h_xzzx, [0], move_duration=1e-3,
                              sublattice='odd')  # apply this H only on the first layer
                    lc.append(qec.surface_code.global_h_xzzx, [1], move_duration=1e-3,
                              sublattice='even')  # apply this H only on the first layer
                else:
                    lc.append(qec.surface_code.global_h, [1],
                              move_duration=1e-3)  # apply this H only on the first layer

        if round_ix < num_layers - 1:
            lc.append('Y', lc.qubit_indices)  # new GB - add Y pulse on all qubits:

        ### Round of QEC:
        if num_layers > 1:  # we do want some QEC..
            if round_ix < num_layers - 1:
                # measure ancilla qubits (without detectors)
                put_detectors = False  # dont put any detectors, we will define them here.
                init_round = None

                if xzzx:  # measure xzzx construction, meaning without the H at the end and beginning of round on data qubits.
                    lc.append(qec.surface_code.measure_stabilizers_xzzx_weight2_new_ver, [0, 1],
                              order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round=False,
                              SWAP_round_type='None', compare_with_previous=False, put_detectors=put_detectors,
                              logical_basis='None', init_round=init_round, automatic_detectors=False,
                              previous_meas_offset=0)  # # new version of stabilizer checks to fix weight=2 checks
                    # lc.append(qec.surface_code.measure_stabilizers_xzzx_weight2_new_ver, [1], order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round = False, SWAP_round_type='None', compare_with_previous=True, put_detectors = put_detectors, logical_basis='None', init_round=init_round, automatic_detectors=False, previous_meas_offset=previous_meas_offset) # # new version of stabilizer checks to fix weight=2 checks
                else:
                    lc.append(qec.surface_code.measure_stabilizers, [0, 1], order=ordering[round_ix],
                              with_cnot=biased_pres_gates, SWAP_round=False, SWAP_round_type='None',
                              compare_with_previous=False, put_detectors=put_detectors, logical_basis='None',
                              init_round=init_round, automatic_detectors=False)  # append QEC rounds
                    # lc.append(qec.surface_code.measure_stabilizers, [1], order=ordering[round_ix], with_cnot=biased_pres_gates, SWAP_round = False, SWAP_round_type='None', compare_with_previous=True, put_detectors = put_detectors, logical_basis='Z', init_round=init_round, automatic_detectors=False) # append QEC rounds
                # print('during ec', round_ix, cx_ix, lc.no_noise_zone, lc.storage_zone, lc.entangling_zone)

            if ('ignoredetectors' in lc.circuit_index):  ### ADDED BY SG ON 22/10/2024
                pass

            else:
                ## constructing detectors for each layer differently:
                if round_ix == 0:  # first layer, we put 2 body operators for the detectors.
                    if num_CX_in_layer % 2 == 0:  # new GB - no entanglement in this layer
                        # construct the 1 body operators in each logical:
                        init_bases = ['X', 'Z']
                        for index in [0, 1]:
                            measure_qubits_set = lc.logical_qubits[index].measure_qubits_x if init_bases[
                                                                                                  index] == 'X' else \
                            lc.logical_qubits[index].measure_qubits_z
                            for meas_q in measure_qubits_set:
                                check_ix = measure_qubits_list.index(meas_q)
                                check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix))]
                                lc.append('DETECTOR', check_targets)
                    else:
                        # construct the 2 body operators between the logicals:
                        for meas_q in np.concatenate(
                                (lc.logical_qubits[0].measure_qubits_x, lc.logical_qubits[0].measure_qubits_z)):
                            # for meas_q in measure_qubits_L0:
                            check_ix = measure_qubits_list.index(meas_q)

                            check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix)),
                                             stim.target_rec(-(int(num_of_measure_qubits / 2) - check_ix))]
                            lc.append('DETECTOR', check_targets)


                elif (round_ix != num_layers - 1):

                    if num_CX_in_layer % 2 == 0:  # new GB - no entanglement in this layer
                        # construct the 2 body operators in each logical (t,t-1):
                        for meas_q in np.concatenate((lc.logical_qubits[0].measure_qubits_x,
                                                    lc.logical_qubits[0].measure_qubits_z,
                                                    lc.logical_qubits[1].measure_qubits_x,
                                                    lc.logical_qubits[1].measure_qubits_z)):
                            meas_q_type = 'X' if meas_q in measure_qubits_x else 'Z'
                            meas_q_logical = 0 if meas_q in np.concatenate(
                                (lc.logical_qubits[0].measure_qubits_x, lc.logical_qubits[0].measure_qubits_z)) else 1
                            check_ix = measure_qubits_list.index(meas_q)

                            # print(f"meas_q: {meas_q}, type: {meas_q_type}, meas_q_logical: {meas_q_logical}")
                            check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix)),
                                             stim.target_rec(-(2 * num_of_measure_qubits - check_ix))]
                            lc.append('DETECTOR', check_targets)
                    else:
                        # construct the 2 / 3 body operators between the logicals and with previous rounds (using measure qubits):
                        for meas_q in np.concatenate((lc.logical_qubits[0].measure_qubits_x,
                                                    lc.logical_qubits[0].measure_qubits_z,
                                                    lc.logical_qubits[1].measure_qubits_x,
                                                    lc.logical_qubits[1].measure_qubits_z)):
                            # for meas_q in measure_qubits:
                            meas_q_type = 'X' if meas_q in measure_qubits_x else 'Z'
                            meas_q_logical = 0 if meas_q in np.concatenate(
                                (lc.logical_qubits[0].measure_qubits_x, lc.logical_qubits[0].measure_qubits_z)) else 1
                            check_ix = measure_qubits_list.index(meas_q)

                            # print(f"meas_q: {meas_q}, type: {meas_q_type}, meas_q_logical: {meas_q_logical}")

                            if (
                                    meas_q_type == 'X' and meas_q_logical == 0):  # X type for L0 or Z type for L1, 3 body operator:
                                # check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix)), stim.target_rec(-(2*num_of_measure_qubits - check_ix)), stim.target_rec(-(int(num_of_measure_qubits/2) - check_ix))]
                                check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix)),
                                                 stim.target_rec(-(2 * num_of_measure_qubits - check_ix)),
                                                 stim.target_rec(-(int((3 / 2) * num_of_measure_qubits) - check_ix))]
                            elif (
                                    meas_q_type == 'Z' and meas_q_logical == 1):  # X type for L0 or Z type for L1, 3 body operator:
                                # check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix)), stim.target_rec(-(2*num_of_measure_qubits - check_ix)), stim.target_rec(-(int((3/2)*num_of_measure_qubits) - check_ix))]
                                check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix)),
                                                 stim.target_rec(-(2 * num_of_measure_qubits - check_ix)),
                                                 stim.target_rec(-(int((5 / 2) * num_of_measure_qubits) - check_ix))]
                            # X type for L1 or Z type for L0, 2 body operator:
                            else:
                                check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix)),
                                                 stim.target_rec(-(2 * num_of_measure_qubits - check_ix))]

                            lc.append('DETECTOR', check_targets)

            # lc.append_from_stim_program_text("""TICK""") # ending a QEC round
            lc.append('MOVE_TO_STORAGE',
                    np.concatenate([lc.logical_qubits[i].measure_qubits for i in range(len(lc.logical_qubits))]),
                    200)
            # print('after ec', round_ix, cx_ix, lc.no_noise_zone, lc.storage_zone, lc.entangling_zone)

            QEC_cycles += 1

    ### logical measurement step:
    # lc.append_from_stim_program_text("""TICK""") # starting a QEC round

    no_ancillas = True if QEC_cycles == 0 else False  # no QEC rounds at all

    if logical_basis == 'XZ':
        # previous_meas_offset = len(lc.logical_qubits[1].measure_qubits_x) + len(lc.logical_qubits[1].measure_qubits_z) # offset for detectors comparison.
        # lc.append(qec.surface_code.measure_x, [0], observable_include=True, xzzx=xzzx, automatic_detectors=False, no_ancillas = no_ancillas, previous_meas_offset=previous_meas_offset) #XI measurement
        # previous_meas_offset = len(lc.logical_qubits[0].data_qubits) # offset for detectors comparison.
        # lc.append(qec.surface_code.measure_z, [1], observable_include=True, xzzx=xzzx, automatic_detectors=False, no_ancillas = no_ancillas, previous_meas_offset=previous_meas_offset) #IZ measurement
        raise NotImplementedError
        lc.append('SQRT_Y', data_qubits)  ## TODO: fix it if we need it
        lc.append('M', data_qubits)

        if ('ignoredetectors' in lc.circuit_index):  ### ADDED BY SG ON 22/10/2024
            construct_detectors_data_qubits_measurement(meas_bases=['X', 'Z'], QEC_cycles=0,
                                                        num_CX_in_layer=num_CX_in_layer)
        else:
            construct_detectors_data_qubits_measurement(meas_bases=['X', 'Z'], QEC_cycles=QEC_cycles,
                                                        num_CX_in_layer=num_CX_in_layer)

        for index in range(num_logicals):
            logical_rec = []
            physical_data_qubit_layout = lc.logical_qubits[index].data_qubits.reshape((lc.logical_qubits[index].dy,
                                                                                    lc.logical_qubits[index].dx))
            if index == 0:
                logical_x = physical_data_qubit_layout[lc.logical_qubits[index].dy // 2, :]
                logical_x_rec = -((num_logicals - index) * len(lc.logical_qubits[index].data_qubits) - np.array(
                    [np.argwhere(i == lc.logical_qubits[index].data_qubits)[0, 0] for i in logical_x]))
                logical_rec.append([stim.target_rec(i) for i in logical_x_rec])
                lc.append('OBSERVABLE_INCLUDE', np.concatenate(logical_rec), lc.num_observables)

            elif index == 1:
                logical_z = physical_data_qubit_layout[:, lc.logical_qubits[index].dx // 2]
                logical_z_rec = -((num_logicals - index) * len(lc.logical_qubits[index].data_qubits) - np.array(
                    [np.argwhere(i == lc.logical_qubits[index].data_qubits)[0, 0] for i in logical_z]))
                logical_rec.append([stim.target_rec(i) for i in logical_z_rec])
                lc.append('OBSERVABLE_INCLUDE', np.concatenate(logical_rec), lc.num_observables)

    if logical_basis == 'XX':

        # new GB - cancel out the H here with the final CX H gates:
        # lc.append('MOVE_TO_ENTANGLING', data_qubits, 0)
        lc.append('SQRT_Y', data_qubits_L0)
        lc.append('M', data_qubits)

        if ('ignoredetectors' in lc.circuit_index):  ### ADDED BY SG ON 22/10/2024
            construct_detectors_data_qubits_measurement(meas_bases=['X', 'X'], QEC_cycles=0,
                                                        num_CX_in_layer=num_CX_in_layer)
        else:
            construct_detectors_data_qubits_measurement(meas_bases=['X', 'X'], QEC_cycles=QEC_cycles,
                                                        num_CX_in_layer=num_CX_in_layer)

        # construct_detectors_data_qubits_measurement(meas_bases = ['X','X'], QEC_cycles=QEC_cycles, num_CX_in_layer=num_CX_in_layer)

        logical_xx_rec = []
        for index in range(num_logicals):
            physical_data_qubit_layout = lc.logical_qubits[index].data_qubits.reshape((lc.logical_qubits[index].dy,
                                                                                       lc.logical_qubits[index].dx))
            logical_x = physical_data_qubit_layout[lc.logical_qubits[index].dy // 2, :]
            logical_x_rec = -((num_logicals - index) * len(lc.logical_qubits[index].data_qubits) - np.array(
                [np.argwhere(i == lc.logical_qubits[index].data_qubits)[0, 0] for i in logical_x]))
            logical_xx_rec.append([stim.target_rec(i) for i in logical_x_rec])

        lc.append('OBSERVABLE_INCLUDE', np.concatenate(logical_xx_rec), lc.num_observables)

    elif logical_basis == 'ZZ':
        # new GB - cancel out the H here with the final CX H gates:
        # lc.append('MOVE_TO_ENTANGLING', data_qubits, 0)
        lc.append('SQRT_Y', data_qubits_L1)
        lc.append('M', data_qubits)

        if ('ignoredetectors' in lc.circuit_index):  ### ADDED BY SG ON 22/10/2024
            construct_detectors_data_qubits_measurement(meas_bases=['Z', 'Z'], QEC_cycles=0,
                                                        num_CX_in_layer=num_CX_in_layer)
        else:
            construct_detectors_data_qubits_measurement(meas_bases=['Z', 'Z'], QEC_cycles=QEC_cycles,
                                                        num_CX_in_layer=num_CX_in_layer)

        # construct_detectors_data_qubits_measurement(meas_bases = ['Z','Z'], QEC_cycles=QEC_cycles, num_CX_in_layer=num_CX_in_layer)

        logical_zz_rec = []
        for index in range(num_logicals):
            physical_data_qubit_layout = lc.logical_qubits[index].data_qubits.reshape((lc.logical_qubits[index].dy,
                                                                                    lc.logical_qubits[index].dx))
            logical_z = physical_data_qubit_layout[:, lc.logical_qubits[index].dx // 2]
            logical_z_rec = -((num_logicals - index) * len(lc.logical_qubits[index].data_qubits) - np.array(
                [np.argwhere(i == lc.logical_qubits[index].data_qubits)[0, 0] for i in logical_z]))
            logical_zz_rec.append([stim.target_rec(i) for i in logical_z_rec])

        lc.append('OBSERVABLE_INCLUDE', np.concatenate(logical_zz_rec), lc.num_observables)
    return lc





def lattice_surgery_experiment_surface(dx, dy, code, num_layers=2, num_logicals=1, logical_basis='XX',
                        biased_pres_gates=False, ordering='fowler', loss_detection_method='None',
                        loss_detection_frequency=100, atom_array_sim=True, replace_H_Ry=False, xzzx=False,
                        noise_params={}, printing=False, circuit_index='0'):
    """ This circuit simulated 2 logical qubits, lattice surgery circuit."""

    assert atom_array_sim
    assert xzzx
    assert num_logicals == 1
    assert logical_basis in ['XX', 'ZZ']

    def prepare_logicals_independently(lc, init_basis):
        lc.append('MOVE_TO_ENTANGLING', data_qubits, move_duration) # Move to entangling
        lc.append('R', data_qubits) # Reset all data qubits to plus
        if lc.replace_H_Ry:
            H_gate_string = 'SQRT_Y_DAG'
        else:
            H_gate_string = 'H'
        if xzzx:
            data_qubit_sub_indices = data_qubits_subA if init_basis == 'X' else data_qubits_subB
            lc.append(H_gate_string, data_qubit_sub_indices)
        else: # regular surface code
            if init_basis == 'X':
                lc.append(H_gate_string, data_qubits)
        
        
    def measure_logicals_independently(lc, meas_basis):
        lc.append('MOVE_TO_ENTANGLING', data_qubits, move_duration) # GB: this doesn't add idling noise because of the if statement inside. maybe we do need to add this noise?
        if lc.replace_H_Ry:
            H_gate_string = 'SQRT_Y_DAG'
        else:
            H_gate_string = 'H'
        if xzzx: # apply H only on sublattice A (data_qubits_even):
            data_qubit_sub_indices = data_qubits_subA if meas_basis == 'X' else data_qubits_subB
            lc.append(H_gate_string, data_qubit_sub_indices)
        else: # regular surface code |+> logical
            if init_basis == 'X':
                lc.append(H_gate_string, data_qubits)
        lc.append('M', data_qubits)


    def define_observable_on_merge_qubits(lc, merge_qubits_x, merge_qubits_z, meas_basis):
        relevant_measure_qubits = lc.logical_qubits[0].measure_qubits_x if meas_basis== 'X' else lc.logical_qubits[0].measure_qubits_z
        relevant_merge_qubits = merge_qubits_x if meas_basis == 'X' else merge_qubits_z
        check_ixs = [sorted(measure_qubits_list).index(i) for i in relevant_merge_qubits]
        check_targets = [stim.target_rec(-(num_of_measure_qubits - check_ix)) for check_ix in check_ixs]
        lc.append('OBSERVABLE_INCLUDE', check_targets, lc.num_observables)

    def define_detectors_on_each_logical_independently(lc, merge_qubits_x, merge_qubits_z, meas_basis):
        relevant_measure_qubits = lc.logical_qubits[0].measure_qubits_x if meas_basis== 'X' else lc.logical_qubits[0].measure_qubits_z
        relevant_merge_qubits = merge_qubits_x if meas_basis == 'X' else merge_qubits_z
        detectors_qubits = [q for q in relevant_measure_qubits if q not in relevant_merge_qubits]
        # detectors_qubits = relevant_measure_qubits - relevant_merge_qubits
        for meas_q in detectors_qubits:
            check_ix = sorted(measure_qubits_list).index(meas_q)
            neighbors = [lc.logical_qubits[0].neighbor_from_index(physical_index=meas_q, which=direction) for
                direction in [0, 1, 2, 3]]
            all_relevant_data_qubits = sorted([neighbor for neighbor in neighbors if neighbor is not None])
            check_ixs = [sorted(data_qubits_list).index(i) for i in all_relevant_data_qubits]
            check_targets_data = [stim.target_rec(-(num_of_data_qubits - check_ix)) for check_ix in check_ixs]
            check_targets_meas_qubits = [stim.target_rec(-(num_of_data_qubits + num_of_measure_qubits - check_ix))]

            lc.append('DETECTOR', check_targets_data + check_targets_meas_qubits)

    def define_z_observables_on_each_logical_independently(lc):
        L1_Z_col = dx-1 
        L2_Z_col = dx+1
        qubits_at_L1_Z_col = [qubit for qubit, pos in zip(logical_qubits[0].qubit_indices, logical_qubits[0].positions) if pos[1] == L1_Z_col]
        qubits_at_L2_Z_col = [qubit for qubit, pos in zip(logical_qubits[0].qubit_indices, logical_qubits[0].positions) if pos[1] == L2_Z_col]
        all_Z1Z2_qubits = qubits_at_L1_Z_col + qubits_at_L2_Z_col
        check_ixs = [data_qubits_list.index(i) for i in all_Z1Z2_qubits]
        check_targets = [stim.target_rec(-(num_of_data_qubits - check_ix)) for check_ix in check_ixs]
        lc.append('OBSERVABLE_INCLUDE', check_targets, lc.num_observables)

    
    # ordering = organize_ordering_array(ordering=ordering, QEC_cycles=num_layers)
    
    if code == 'Rotated_Surface':
        logical_qubits = [qec.surface_code.RotatedSurfaceCode(dx, dy) for _ in range(num_logicals)]
    elif code == 'Surface':
        logical_qubits = [qec.surface_code.SurfaceCode(dx, dy) for _ in range(num_logicals)]

    lc = LogicalCircuit(logical_qubits, initialize_circuit=False,
                        loss_noise_scale_factor=1, spam_noise_scale_factor=1,
                        gate_noise_scale_factor=1, idle_noise_scale_factor=1,
                        atom_array_sim=atom_array_sim, replace_H_Ry=replace_H_Ry, circuit_index=circuit_index,
                        circuit_type='LS', **noise_params)


    # if num_layers > 1:  # we have QEC # Q: do we want this here?
    #     lc.append('MOVE_TO_STORAGE',
    #             np.concatenate([lc.logical_qubits[i].measure_qubits for i in range(len(lc.logical_qubits))]), 1e-3)
    
    # lc.append('MOVE_TO_ENTANGLING',
    #             np.concatenate([lc.logical_qubits[i].data_qubits for i in range(len(lc.logical_qubits))]), 1e-3)
    
    
    ### Variables we are going to use:
    target_dx = dx # find all qubits in the middle column (merge qubits)
    qubits_at_target_dx = [qubit for qubit, pos in zip(logical_qubits[0].qubit_indices, logical_qubits[0].positions) if pos[1] == target_dx]
    merge_qubits_x = [q for q in lc.logical_qubits[0].measure_qubits_x if q in qubits_at_target_dx]
    merge_qubits_z = [q for q in lc.logical_qubits[0].measure_qubits_z if q in qubits_at_target_dx]
    measure_qubits = list(lc.logical_qubits[0].measure_qubits_x) + list(lc.logical_qubits[0].measure_qubits_z)
    measure_qubits_list = measure_qubits
    num_of_measure_qubits = len(measure_qubits)
    data_qubits = lc.logical_qubits[0].data_qubits
    num_of_data_qubits = len(data_qubits)
    data_qubits_list = data_qubits.tolist()
    qubits_at_L0 = [qubit for qubit, pos in zip(logical_qubits[0].qubit_indices, logical_qubits[0].positions) if pos[1] < target_dx]
    qubits_at_L1 = [qubit for qubit, pos in zip(logical_qubits[0].qubit_indices, logical_qubits[0].positions) if pos[1] > target_dx]
    data_qubits_L0 = [qubit for qubit in data_qubits if qubit in qubits_at_L0]
    data_qubits_L1 = [qubit for qubit in data_qubits if qubit in qubits_at_L1]
    data_qubits_L0_subA = [qubit for qubit in data_qubits_L0 if qubit in lc.logical_qubits[0].data_qubits_even]
    data_qubits_L0_subB = [qubit for qubit in data_qubits_L0 if qubit in lc.logical_qubits[0].data_qubits_odd]
    data_qubits_L1_subA = [qubit for qubit in data_qubits_L1 if qubit in lc.logical_qubits[0].data_qubits_even]
    data_qubits_L1_subB = [qubit for qubit in data_qubits_L1 if qubit in lc.logical_qubits[0].data_qubits_odd]
    data_qubits_subA = data_qubits_L0_subA + data_qubits_L1_subA
    data_qubits_subB = data_qubits_L0_subB + data_qubits_L1_subB
    move_duration = 1e-3
    
    
    # we always initialize 1 big surface code in the X basis:
    lc.append(qec.surface_code.prepare_plus_no_gates, [0], xzzx=xzzx)

    # lc.logical_qubits[0].visualize_code()
    

    # in each layer, we have stabilizer checks on each logical and between them. We can use the large code as usual.
    # basis = 'X' if logical_basis == 'XX' else 'Z'
    basis = 'X' # deterministic detectors are X always, because we initalize in |+> all the time
    add_QEC_rounds(lc, logical_qubits, num_layers, loss_detection_method, atom_array_sim, loss_detection_frequency, xzzx, ordering, biased_pres_gates, basis)

    no_ancillas = True if num_layers==0 else False
    if logical_basis == 'XX':
        lc.append(qec.surface_code.measure_x, [0], observable_include=True, xzzx=xzzx, automatic_detectors=False, no_ancillas = no_ancillas)
        return lc
    
    if logical_basis == 'ZZ':
        # circuit 1: define observable as the product of the merge stabilizers:
        Partial_lc = lc.full_copy() # first 2 rounds, no projective measurement
        define_observable_on_merge_qubits(Partial_lc, merge_qubits_x, merge_qubits_z, meas_basis='Z')
        
        # circuit 2: split to 2 unrelated surface code.
        Full_lc =  lc.full_copy() # first 2 rounds, no projective measurement
        Full_lc.append(qec.surface_code.measure_z, [0], observable_include=False, xzzx=xzzx, automatic_detectors=False, no_ancillas = no_ancillas, put_detectors=False)
        define_detectors_on_each_logical_independently(Full_lc, merge_qubits_x, merge_qubits_z, meas_basis='Z')
        define_z_observables_on_each_logical_independently(Full_lc)

        # print(f"Partial_lc = {Partial_lc}")
        # print(f"Full_lc = {Full_lc}")
    
        return Partial_lc, Full_lc
        
    
    
    
    



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
        
        SWAP_round_index = 0
        for round_ix in range(QEC_cycles):
            SWAP_round_type = 'none'; SWAP_round = False
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

