import stim
import numpy as np
from typing import List


class HeraldedCircuit_SWAP_LD:
    def __init__(self, circuit: stim.Circuit, biased_erasure: bool, cycles: int, phys_error:float, erasure_ratio:float, bias_preserving_gates:bool, ancilla_qubits:list, data_qubits:list, distance=None, loss_detection_freq=None, code=None, basis = None, SSR = True, printing=False, **kwargs) -> None:
        self.logical_circuit = circuit
        self.biased_erasure = biased_erasure
        self.distance = distance
        self.code = code
        self.basis = basis
        self.SSR = SSR
        self.cycles = cycles
        self.printing = printing
        self.erasure_ratio = erasure_ratio
        self.bias_preserving_gates = bias_preserving_gates
        self.phys_error = phys_error
        self.loss_detection_freq = loss_detection_freq
        self.ancilla_qubits = ancilla_qubits
        self.data_qubits = data_qubits
        self.all_qubits = ancilla_qubits + data_qubits
        self.lost_ancillas = {}  ###
        self.qec_cycles_complete = False  ###
        self.lost_ancillas_by_qec_round = {}  #!  {qec_round: [lost_ancilla_qubits]}
        self.lost_data_by_ld_round = {}  #!  {ld_round: [lost_data_qubits]}
        self.lost_qubits_by_qec_round = {}  #!  {ld_round: [lost_qubits]}
        self.total_lost_qubits_in_round = {}
        self.lost_data_by_qec_round = {}  #!  {qec_round: [lost_data_qubits]}
        self.QEC_round_types = {}
        self.heralded_loss_qubits = {}   #!  {qec_round: [detectable_loss_qubits_in_this_round]}
        self.total_num_QEC_round = None
        self.qubit_lifecycles_and_losses = {} # qubit: {[R_round, M_round, Lost?], [R_round, M_round, Lost?], ..}
        self.SWAP_circuit = None
        self.gates_ordering_dict = {} # round_ix: {qubit: {gate_order: [neighbor_after_this_gate, error_if_qubit_is_lost] } }
        self.qubits_type_by_qec_round = {} # {qec_round: {index: type}} # TODO: fill this out. for every round the type of the qubit is the type before the SWAP operation.
                
    
    def update_loss_lists(self, instruction, loss_detection_events, lost_qubits_in_round, loss_detector_ix):
        potential_lost_qubits = instruction.targets_copy()
        for q in potential_lost_qubits:
            if loss_detection_events[loss_detector_ix] == True:
                if q.value not in lost_qubits_in_round:
                    lost_qubits_in_round.append(q.value)
            loss_detector_ix += 1
        return loss_detector_ix
        

    def insert_SWAP_operations(self, updated_instruction, SWAP_round_type, SWAP_circuit, updated_qubit_index_mapping):
        qubits = set([q.value for q in updated_instruction.targets_copy()])
        logical_qubits = list(set([self.logical_circuit.qubit_index_to_logical_qubit(q) for q in qubits]))
        for logical_qubit in logical_qubits:
            swap_pairs = logical_qubit.define_swap_pairs('even') if SWAP_round_type == 'even' else logical_qubit.define_swap_pairs('odd')
            # swap_pairs = logical_qubit.swap_pairs_even if SWAP_round_type == 'even' else logical_qubit.swap_pairs_odd
            # print(f"swap pairs = {swap_pairs}")
            flat_list = [qubit for pair in swap_pairs for qubit in pair]
            SWAP_circuit.append('SWAP', flat_list)
            self.add_idling_channel(SWAP_circuit, sorted(list(logical_qubit.data_qubits) + list(logical_qubit.measure_qubits))) # Add SWAP movement errors
            # Update qubit_index_mapping based on the swap_pairs
            for pair in swap_pairs:
                # pair is a set of 2 qubits, for example (8,0).
                # now go over the values in updated_qubit_index_mapping, and for each value (for example 8) replace it with its friend from the pair (for example 0).
                key_for_value1 = [key for key, value in updated_qubit_index_mapping.items() if value == pair[0]][0]
                key_for_value2 = [key for key, value in updated_qubit_index_mapping.items() if value == pair[1]][0]

                # Swap the values associated with those keys
                updated_qubit_index_mapping[key_for_value1], updated_qubit_index_mapping[key_for_value2] = updated_qubit_index_mapping[key_for_value2], updated_qubit_index_mapping[key_for_value1]
            logical_qubit.swap_qubits(swap_pairs) # Update the logical qubit class
            
            
                            
                            
    def transfer_circuit_into_SWAP_circuit(self, input_circuit):
        # This function is called once for all shots, just the alter the original circuit with the SWAP operations.
        round_ix = -1
        inside_qec_round = False
        SWAP_round_index = 0
        SWAP_round_type = None
        SWAP_circuit = stim.Circuit()
        first_QEC_round = True
        insert_H_first = False
        updated_qubit_index_mapping = {i: i for i in self.ancilla_qubits + self.data_qubits}
        self.gates_ordering_dict = {}
        self.qubits_type_by_qec_round = {}
        
        for instruction in input_circuit:
            # Update SWAP relabeling:
            if instruction.name in ['DETECTOR', 'OBSERVABLE_INCLUDE']:
                SWAP_circuit.append(instruction)
                continue
            updated_targets = []
            for q in instruction.targets_copy():
                if isinstance(q, stim.GateTarget): # Update the qubit index if it's a non-negative value
                    updated_targets.append(stim.GateTarget(updated_qubit_index_mapping[q.value]))
                else: # Keep the target as is for other cases
                    updated_targets.append(q)
            
            
            if instruction.name == 'MPP': # different way of updating the instruction:
                new_logical_operator = []
                for target in instruction.targets_copy():
                    if not target.is_combiner:
                        prev_index = target.qubit_value
                        new_index = updated_qubit_index_mapping[prev_index]
                        if target.pauli_type == 'X':
                            new_logical_operator.append(stim.target_x(new_index))
                        elif target.pauli_type == 'Y':
                            new_logical_operator.append(stim.target_y(new_index))
                        elif target.pauli_type == 'Z':
                            new_logical_operator.append(stim.target_z(new_index))
                        else:
                            assert True is False
                    else:
                        new_logical_operator.append(stim.target_combiner())
                SWAP_circuit.append('MPP', new_logical_operator)
                
            else:
                updated_instruction = stim.CircuitInstruction(instruction.name, updated_targets, instruction.gate_args_copy())
                SWAP_circuit.append(updated_instruction)

            # QEC rounds:
            if updated_instruction.name == 'TICK':
                if not inside_qec_round: # beginning of QEC round
                    if first_QEC_round:
                        round_ix += 1 ; first_QEC_round = False
                    CZ_round_ix = 0
                    self.gates_ordering_dict[round_ix] = {}
                    
                    self.qubits_type_by_qec_round[round_ix] = {}
                    for qubit in self.all_qubits:
                        logical_qubit = self.logical_circuit.qubit_index_to_logical_qubit(qubit)
                        qubit_type = 'data' if qubit in logical_qubit.data_qubits else 'ancilla'
                        self.qubits_type_by_qec_round[round_ix][qubit] = qubit_type
                        
                    if (round_ix+1)%self.loss_detection_freq == 0:
                        SWAP_round = True
                        SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
                        SWAP_round_index += 1
                    else:
                        SWAP_round = False
                else: # end of round
                    self.QEC_round_types[round_ix] = SWAP_round_type if SWAP_round else 'regular'
                    round_ix += 1
                    
                inside_qec_round = not inside_qec_round
                continue
            
            if inside_qec_round: # TODO: first do the H and then do the SWAP things
                if updated_instruction.name in ['CZ', 'CX']:
                    gate_type = updated_instruction.name
                    
                    qubits = [q.value for q in updated_instruction.targets_copy()]
                    pairs = [(qubits[i], qubits[i + 1]) for i in range(0, len(qubits), 2)]
                    for (c,t) in pairs:
                        # if we lose c (control), the target will get the following error:
                        if gate_type == 'CZ':
                            noise_type = 'Z'
                        elif gate_type == 'CX':
                            noise_type = 'X'
                        if c not in self.gates_ordering_dict[round_ix].keys():
                            self.gates_ordering_dict[round_ix][c] = {CZ_round_ix: [t,noise_type]}
                        else:
                            self.gates_ordering_dict[round_ix][c][CZ_round_ix] = [t,noise_type]
                            
                        # if we lose t (target), the control will get the following error:
                        if gate_type == 'CZ':
                            noise_type = 'Z'
                        elif gate_type == 'CX':
                            noise_type = 'Z'
                        if t not in self.gates_ordering_dict[round_ix].keys():
                            self.gates_ordering_dict[round_ix][t] = {CZ_round_ix: [c,noise_type]}
                        else:
                            self.gates_ordering_dict[round_ix][t][CZ_round_ix]= [c,noise_type]
                    CZ_round_ix += 1
                    
                if (updated_instruction.name == 'I'): # implement SWAP when needed:
                    if SWAP_round and (CZ_round_ix == 4): # last entangling gates in the round, need to add QI and physical SWAPs
                        # if self.bias_preserving_gates:
                        self.insert_SWAP_operations(updated_instruction, SWAP_round_type, SWAP_circuit, updated_qubit_index_mapping)
                        # else: # first need to insert the next instruction, H:
                            # insert_H_first = True
                            # I_instruction = updated_instruction
                    # CZ_round_ix += 1
                    
                # if insert_H_first and (updated_instruction.name == 'H'):
                #     insert_H_first = False
                #     self.insert_SWAP_operations(I_instruction, SWAP_round_type, SWAP_circuit, updated_qubit_index_mapping)
                            
            else:
                pass
        return SWAP_circuit
    
    
    def get_loss_location(self, loss_detection_events: list, SWAP_circuit=None):
        # Iterate through circuit. Every time we encounter a loss event (flagged by the 'I' gate), record the loss.
        # Also, add the SWAP gates when needed.
        loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
        round_ix = -1
        inside_qec_round = False
        SWAP_round_index = 0
        SWAP_round_type = None
        lost_qubits = [] # qubits that are lost and still undetectable (not measured)
        lost_qubits_in_round = [] # qubit lost in every QEC round. initialized every round.
        self.qubit_lifecycles_and_losses = {i: [] for i in self.ancilla_qubits + self.data_qubits}
        qubit_active_cycle = {i: None for i in self.ancilla_qubits + self.data_qubits}
        first_QEC_round = True
        
        for instruction in SWAP_circuit:
            # Check when each qubit is init and measured:
            if instruction.name in ['R', 'RX']: # Beginning of a cycle for these qubits
                qubits = set([q.value for q in instruction.targets_copy()])
                for q in qubits:
                    self.qubit_lifecycles_and_losses[q].append([round_ix, None, None]) # Begin a new cycle for each qubit
                    qubit_active_cycle[q] = len(self.qubit_lifecycles_and_losses[q]) - 1

            if instruction.name in ['M', 'MX']: # End of a cycle for these qubits
                qubits = set([q.value for q in instruction.targets_copy()])
                lost_qubits.extend(lost_qubits_in_round)
                for q in qubits:
                    if q in lost_qubits:
                        self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][2] = True
                        lost_qubits.remove(q)
                    else:
                        self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][2] = False
                    self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix # Close the active cycle with the measurement round
                    qubit_active_cycle[q] = None

                        
            
            # QEC rounds:
            if instruction.name == 'TICK':
                if not inside_qec_round: # beginning of QEC round
                    if first_QEC_round:
                        round_ix += 1; first_QEC_round = False
                    if self.printing:
                        print(f"Starting QEC Round {round_ix}")
                    lost_qubits_in_round = [] # lost qubits specifically in this QEC round
                    if (round_ix+1)%self.loss_detection_freq == 0:
                        SWAP_round = True
                        SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
                        SWAP_round_index += 1
                    else:
                        SWAP_round = False
                else: # end of round
                    if self.printing:
                        print(f"Finished QEC Round {round_ix}, and lost qubits {lost_qubits_in_round}, thus now we have the following undetectable losses: {lost_qubits}")
                    self.lost_qubits_by_qec_round[round_ix] = lost_qubits_in_round
                    self.QEC_round_types[round_ix] = SWAP_round_type if SWAP_round else 'regular'

                    round_ix += 1
                inside_qec_round = not inside_qec_round
                continue
            
            if inside_qec_round:
                if instruction.name == 'I': # check loss event --> update lost_ancilla_qubits and lost_data_qubits
                    loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, lost_qubits_in_round, loss_detector_ix)
                    
                            
            else:
                pass # we don't need to document losses outside QEC rounds because we assume there are non
                # lost_ancilla_qubits = []
                # lost_data_qubits = []
                # if instruction.name == 'I': # loss event
                #     loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, data_qubits, lost_data_qubits, lost_ancilla_qubits, loss_detector_ix)

        # Handle unmeasured qubits at the end of the circuit
        for q in qubit_active_cycle:
            if qubit_active_cycle[q] is not None:
                self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix



    def heralded_new_circuit(self, loss_detection_events: list):
        """ This function takes the original circuit with places for potential losses and loss detection events, and generates 2 circuits: 1. experimental measurement circuit. 2. Theory decoding circuit. """
        # Initialization for every shot:
        self.lost_ancillas = {}  ###
        self.qec_cycles_complete = False  ###
        self.lost_ancillas_by_qec_round = {}  # {qec_round: [lost_ancilla_qubits]}
        self.lost_data_by_ld_round = {}  #  {ld_round: [lost_data_qubits]}
        self.lost_data_by_qec_round = {}  # {qec_round: [lost_data_qubits]}
        self.lost_qubits_by_qec_round = {}
        self.QEC_round_types = {} # {qec_round: type}
        self.heralded_loss_qubits = {}   #!  {qec_round: [detectable_loss_qubits_in_this_round]}
        self.qubit_lifecycles_and_losses = {} 
        
        # First sweep: get location of lost qubits in the circuit.
        self.get_loss_location(loss_detection_events=loss_detection_events, SWAP_circuit=self.SWAP_circuit) 
        self.total_num_QEC_round = len(self.lost_data_by_qec_round)
        
        
        
        if self.printing :
            print(f"lost_qubits_by_qec_round={self.lost_qubits_by_qec_round}")
            print(f"types of rounds: {self.QEC_round_types}")
            print(f" lifecycles of qubits: {self.qubit_lifecycles_and_losses}\n")
            print(f" self.gates_ordering_dict: {self.gates_ordering_dict}")
            # print(f"\nNew SWAP circuit: \n{SWAP_circuit}")

        # Second sweep: fill in the experimental circuit (heralded_circuit) and decoding circuit (new_lossless_circuit).
        heralded_circuit = stim.Circuit()
        new_lossless_circuit = stim.Circuit()
        loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
        lost_qubits = [] # track which qubits are lost during the round, can be lost in previous round and not detected.
        round_ix = -1
        CZ_round_ix = 0
        first_QEC_round = True
        inside_qec_round = False
        SWAP_round = False
        SWAP_round_index = 0
        qubits_at_risk = set()
        # SWAP_round_type = None
        for instruction in self.SWAP_circuit:
            if instruction.name == 'TICK': # begin of a QEC round:
                if not inside_qec_round: # beginning of round
                    if first_QEC_round:
                        round_ix +=1; first_QEC_round = False
                    self.qec_cycles_complete = True
                    CZ_round_ix = 0
                    # lost_qubits = []
                    # ld_round = int(round_ix / self.loss_detection_freq) # check that its the same
                    # last_QEC_round = True if round_ix == self.total_num_QEC_round-1 else False
                    if (round_ix+1)%self.loss_detection_freq == 0:
                        SWAP_round = True
                        # SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
                        SWAP_round_index += 1
                    else:
                        SWAP_round = False
                    
                    # Check for qubits in the current round that might be lost based on their lifecycle
                    qubits_at_risk = set()
                    for qubit, lifecycle in self.qubit_lifecycles_and_losses.items():
                        for cycle in lifecycle:
                            if cycle[0] <= round_ix <= cycle[1] and cycle[2]:  # Check if the qubit is active and might be lost in this round
                                qubits_at_risk.add(qubit)
                    
                    if self.printing:
                        print(f"a new QEC round number {round_ix}! is it a SWAP round? {SWAP_round} (loss detection freq = {self.loss_detection_freq})")
                
                else: # end of round
                    round_ix += 1
                
                inside_qec_round = not inside_qec_round
                heralded_circuit.append('TICK')
                new_lossless_circuit.append('TICK')
                continue

            if inside_qec_round:
                loss_detector_ix, CZ_round_ix = self.add_instruction(instruction, heralded_circuit, new_lossless_circuit, loss_detection_events, loss_detector_ix,
                                                        lost_qubits=lost_qubits, round_ix=round_ix, CZ_round_ix=CZ_round_ix,
                                                        qubits_at_risk=qubits_at_risk) 
            else:
                loss_detector_ix, CZ_round_ix = self.add_instruction(instruction, heralded_circuit, new_lossless_circuit, loss_detection_events, loss_detector_ix,
                                                        lost_qubits=lost_qubits, round_ix=round_ix, CZ_round_ix=CZ_round_ix,
                                                        qubits_at_risk=qubits_at_risk)
        return heralded_circuit, new_lossless_circuit
    
    def add_instruction(self, instruction, circuit: stim.Circuit, new_lossless_circuit: stim.Circuit, loss_detection_events: list, loss_detector_ix: int,
                        round_ix:int, CZ_round_ix:int, lost_qubits: list, qubits_at_risk=[]):


        # Update lost qubits lists:
        if instruction.name == 'I': # loss event
            loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, lost_qubits, loss_detector_ix) # update lost_qubits, add losses

        elif instruction.name in ['CZ','CX']:
            gate_order_strength_dict_ancilla_loss = {0: 1, 1: 0.75, 2: 0.5, 3: 0.25}
            gate_order_strength_dict_data_loss = {3: 1, 2: 0.75, 1: 0.5, 0: 0.25}
            
            # gate_order_strength =  CZ_order_strength_dict[CZ_round_ix] # first gates have higher probability to cause errors due to loss. # TODO: correct it, for data loss and ancilla loss its different order
            Total_factor = self.phys_error*self.erasure_ratio
            # noise_channel = [0,0,0.5*Total_factor] if instruction.name == 'CZ' else [0.5*Total_factor, 0, 0]
            qubits = [q.value for q in instruction.targets_copy()]
            
            # Lossless circuit: the CZ is here, but we add error model to acount for errors in the decoder:
            new_lossless_circuit.append(instruction) # append the CZ gate to the lossless circuit anyway
            
            pairs = [(qubits[i], qubits[i + 1]) for i in range(0, len(qubits), 2)]
            for (c,t) in pairs:
                ancilla_target = c if (c not in self.data_qubits) else t
                data_target = c if c in self.data_qubits else t

                # Heralded loss circuit: the CZ is not here is we lost one of the qubits:
                if (c in lost_qubits) or (t in lost_qubits): # remove the gate from the heralded circuit:
                    if self.printing :
                        print(f"Removing this gate from the heralded circuit: {instruction.name} {c},{t}, because my lost qubits = {lost_qubits}")
                    pass
                else:
                    circuit.append(instruction.name, [c,t])
            

                # Lossless circuit: Add noise to neighbor data of lost ancilla and to neighbor ancilla of lost data.
                if instruction.name == 'CX':
                    if (self.SSR and c in qubits_at_risk) or (not self.SSR):
                        # logical_qubit = self.logical_circuit.qubit_index_to_logical_qubit(c)
                        # gate_order_strength_dict = gate_order_strength_dict_ancilla_loss if c in logical_qubit.measure_x_qubits + logical_qubit.measure_z_qubits else gate_order_strength_dict_data_loss; gate_order_strength = gate_order_strength_dict[CZ_round_ix]
                        relevant_cycle_length = next((cycle[1] - cycle[0] + 1 for cycle in self.qubit_lifecycles_and_losses[c] if cycle[0] <= round_ix <= cycle[1]), None)
                        # noise_channel = [0.5*Total_factor*gate_order_strength/relevant_cycle_length,0,0]
                        # new_lossless_circuit.append('PAULI_CHANNEL_1', [t], noise_channel)
                        # generate neighbors list by gate order:
                        neighbors_by_order = self.gates_ordering_dict[round_ix][c] # {gate_1: [neigh, noise], gate_2: [neigh, noise],...}
                        qubit_type = self.qubits_type_by_qec_round[round_ix][c]
                        # logical_qubit = self.logical_circuit.qubit_index_to_logical_qubit(c)
                        # qubit_type = 'data' if c in logical_qubit.data_qubits else 'ancilla'
                        self.add_CZ_neighbors_errors(cycle_length = relevant_cycle_length, circuit = new_lossless_circuit, gate_order = CZ_round_ix, lost_qubit=c, lost_qubit_type=qubit_type, neighbors_by_order = neighbors_by_order, round_ix=round_ix) # NEW
                        
                    if (self.SSR and t in qubits_at_risk) or (not self.SSR):
                        
                        # gate_order_strength_dict = gate_order_strength_dict_ancilla_loss if t in self.ancilla_qubits else gate_order_strength_dict_data_loss; gate_order_strength = gate_order_strength_dict[CZ_round_ix]
                        relevant_cycle_length = next((cycle[1] - cycle[0] + 1 for cycle in self.qubit_lifecycles_and_losses[t] if cycle[0] <= round_ix <= cycle[1]), None)
                        # noise_channel = [0,0,0.5*Total_factor*gate_order_strength/relevant_cycle_length]
                        # new_lossless_circuit.append('PAULI_CHANNEL_1', [c], noise_channel)
                        neighbors_by_order = self.gates_ordering_dict[round_ix][t] # {gate_1: [neigh, noise], gate_2: [neigh, noise],...}
                        qubit_type = self.qubits_type_by_qec_round[round_ix][t]
                        # logical_qubit = self.logical_circuit.qubit_index_to_logical_qubit(t)
                        # qubit_type = 'data' if t in logical_qubit.data_qubits else 'ancilla'
                        self.add_CZ_neighbors_errors(cycle_length = relevant_cycle_length, circuit = new_lossless_circuit, gate_order = CZ_round_ix, lost_qubit=t, lost_qubit_type=qubit_type, neighbors_by_order = neighbors_by_order, round_ix=round_ix) # NEW
                        
                elif instruction.name == 'CZ':
                    if (self.SSR and c in qubits_at_risk) or (not self.SSR):
                        # gate_order_strength_dict = gate_order_strength_dict_ancilla_loss if c in self.ancilla_qubits else gate_order_strength_dict_data_loss; gate_order_strength = gate_order_strength_dict[CZ_round_ix]
                        relevant_cycle_length = next((cycle[1] - cycle[0] + 1 for cycle in self.qubit_lifecycles_and_losses[c] if cycle[0] <= round_ix <= cycle[1]), None)
                        # noise_channel = [0, 0, 0.5*Total_factor*gate_order_strength/relevant_cycle_length]
                        # new_lossless_circuit.append('PAULI_CHANNEL_1', [t], noise_channel)
                        neighbors_by_order = self.gates_ordering_dict[round_ix][c] # {gate_1: [neigh, noise], gate_2: [neigh, noise],...}
                        qubit_type = self.qubits_type_by_qec_round[round_ix][c]
                        # logical_qubit = self.logical_circuit.qubit_index_to_logical_qubit(c)
                        # qubit_type = 'data' if c in logical_qubit.data_qubits else 'ancilla'
                        self.add_CZ_neighbors_errors(cycle_length = relevant_cycle_length, circuit = new_lossless_circuit, gate_order = CZ_round_ix, lost_qubit=c, lost_qubit_type=qubit_type, neighbors_by_order = neighbors_by_order, round_ix=round_ix) # NEW
                        
                    if (self.SSR and t in qubits_at_risk) or (not self.SSR):
                        # gate_order_strength_dict = gate_order_strength_dict_ancilla_loss if t in self.ancilla_qubits else gate_order_strength_dict_data_loss; gate_order_strength = gate_order_strength_dict[CZ_round_ix]
                        relevant_cycle_length = next((cycle[1] - cycle[0] + 1 for cycle in self.qubit_lifecycles_and_losses[t] if cycle[0] <= round_ix <= cycle[1]), None)
                        # noise_channel = [0, 0, 0.5*Total_factor*gate_order_strength/relevant_cycle_length]
                        # new_lossless_circuit.append('PAULI_CHANNEL_1', [c], noise_channel)
                        neighbors_by_order = self.gates_ordering_dict[round_ix][t] # {gate_1: [neigh, noise], gate_2: [neigh, noise],...}
                        qubit_type = self.qubits_type_by_qec_round[round_ix][t]
                        # logical_qubit = self.logical_circuit.qubit_index_to_logical_qubit(t)
                        # qubit_type = 'data' if t in logical_qubit.data_qubits else 'ancilla'
                        self.add_CZ_neighbors_errors(cycle_length = relevant_cycle_length, circuit = new_lossless_circuit, gate_order = CZ_round_ix, lost_qubit=t, lost_qubit_type=qubit_type, neighbors_by_order = neighbors_by_order, round_ix=round_ix) # NEW
                        
            CZ_round_ix += 1
            
            
        elif instruction.name in ['SWAP']:
            qubits = [q.value for q in instruction.targets_copy()]

            new_lossless_circuit.append(instruction) # Lossless circuit
            
            pairs = [(qubits[i], qubits[i + 1]) for i in range(0, len(qubits), 2)]
            for (c,t) in pairs:
                # Heralded circuit - remove gate:
                if (c in lost_qubits) or (t in lost_qubits):
                    if self.printing :
                        print(f"Removing this gate from the heralded circuit: {instruction.name} {(c,t)}, because my lost qubits = {lost_qubits}")
                    pass
                else:
                    circuit.append(instruction.name, [c,t])
                    
                # lossless circuit - noise on the paired qubit of a lost qubit:
                if (self.SSR and c in qubits_at_risk) or (not self.SSR):
                    relevant_cycle_length = next((cycle[1] - cycle[0] + 1 for cycle in self.qubit_lifecycles_and_losses[t] if cycle[0] <= round_ix <= cycle[1]), None)
                    self.add_SWAP_neighbors_errors(cycle_length = relevant_cycle_length, circuit = new_lossless_circuit, qubit=t)
                    # logical_qubit = self.logical_circuit.qubit_index_to_logical_qubit(c)
                    # if c in logical_qubit.measure_qubits_x + logical_qubit.measure_qubits_z:
                    #     ancilla_qubit_type = 'X' if c in logical_qubit.measure_qubits_x else 'Z'
                    #     self.add_SWAP_neighbors_errors_ancilla_loss(cycle_length = relevant_cycle_length, circuit = new_lossless_circuit, ancilla_qubit_type = ancilla_qubit_type, data_neigh_index=t) # NEW
                    # else:
                    #     print("need to build this feature")
                        
                if (self.SSR and t in qubits_at_risk) or (not self.SSR):
                    relevant_cycle_length = next((cycle[1] - cycle[0] + 1 for cycle in self.qubit_lifecycles_and_losses[t] if cycle[0] <= round_ix <= cycle[1]), None)
                    self.add_SWAP_neighbors_errors(cycle_length = relevant_cycle_length, circuit = new_lossless_circuit, qubit=c)
                    # logical_qubit = self.logical_circuit.qubit_index_to_logical_qubit(t)
                    # if t in logical_qubit.measure_qubits_x + logical_qubit.measure_qubits_z:
                    #     ancilla_qubit_type = 'X' if t in logical_qubit.measure_qubits_x else 'Z'
                    #     self.add_SWAP_neighbors_errors_ancilla_loss(cycle_length = relevant_cycle_length, circuit = new_lossless_circuit, ancilla_qubit_type = ancilla_qubit_type, data_neigh_index=c) # NEW
                    # else:
                    #     print("need to build this feature")
                
                
        elif instruction.name in ['H', 'R', 'RX']:
        # elif instruction.name not in ['DETECTOR', 'OBSERVABLE_INCLUDE']:
            new_lossless_circuit.append(instruction) # append the gate to the lossless circuit anyway
            qubits = [q.value for q in instruction.targets_copy()]
            for q in qubits: # remove the gate from the heralded circuit:
                if q in lost_qubits:
                    if self.printing :
                        print(f"Removing this gate from the heralded circuit: {instruction.name} {q}, because my lost qubits = {lost_qubits}")
                    pass
                else:
                    circuit.append(instruction.name, [q])

            
        elif instruction.name in ['MRX', 'MR']:
            qbts = instruction.targets_copy()

            if len(lost_qubits) == 0: # no qubit could have been lost in this round anyway.
                circuit.append(instruction)
                
            else:
                for ix, qbt in enumerate(qbts):
                    q = qbt.value
                    
                    if q in lost_qubits: # qubits_at_risk
                        # remove from the lost_qubits list
                        lost_qubits.remove(q)
                        
                        # Lossless circuit - create a supercheck operator:
                        self.add_pauli_channel(new_lossless_circuit, [q])

                        # Heralded circuit - lost ancilla qubits give deterministic |0> measurement:
                        if instruction.name == 'MR':
                            circuit.append('R', [q])
                            circuit.append('MR', [q])
                        elif instruction.name == 'MRX':
                            circuit.append('RX', [q])
                            circuit.append('MRX', [q])
                    else:
                        circuit.append(instruction.name, [q])
                        
            new_lossless_circuit.append(instruction)
            
            
        elif instruction.name in ['M', 'MX']:
            qbts = instruction.targets_copy()

            if len(lost_qubits) == 0:
                circuit.append(instruction)
                
            else:
                for ix, qbt in enumerate(qbts):
                    q = qbt.value
                    
                    if q in lost_qubits:
                        # remove from the lost_qubits list
                        lost_qubits.remove(q)
                        
                        # Lossless circuit - create a supercheck operator:
                        self.add_pauli_channel(new_lossless_circuit, [q])

                        # Heralded circuit - lost ancilla qubits give deterministic |0> measurement:
                        if instruction.name == 'M':
                            circuit.append('R', [q])
                            circuit.append('M', [q])
                        elif instruction.name == 'MX':
                            circuit.append('RX', [q])
                            circuit.append('MX', [q])
                    else:
                        circuit.append(instruction.name, [q])
                        
            new_lossless_circuit.append(instruction)
            
            
        else:
            circuit.append(instruction)
            new_lossless_circuit.append(instruction)
        return loss_detector_ix, CZ_round_ix


    def add_pauli_channel(self, circuit, targets):
        if len(targets) == 1 and not self.biased_erasure:
            circuit.append('PAULI_CHANNEL_1', targets, np.array([0.25, 0.25, 0.25])) # {X,Y,Z,I}
        elif len(targets) == 1 and self.biased_erasure:
            circuit.append('PAULI_CHANNEL_1', targets, np.array([0, 0, 0.5])) # {I,Z}
        elif len(targets) == 2 and not self.biased_erasure:
            circuit.append('PAULI_CHANNEL_2', targets, [1/16 for i in range(15)]) #  {X,Y,Z,I}**2
        elif len(targets) == 2 and self.biased_erasure:
            circuit.append('PAULI_CHANNEL_2', targets, np.array([0,0,0.25,0,0,0,0,0,0,0,0,0.25,0,0,0.25])) #  biased, {Z,I}**2
    
    
    def add_idling_channel(self, circuit, targets, gate_duration=200):
        # gate duration in us
        x_error_rate = 1 - (1-5*1e-6/25)**gate_duration; y_error_rate = 1 - (1-5*1e-6/25)**gate_duration; z_error_rate = 1 - (1-2*1e-5/25)**gate_duration # error probability per gate_duration*1us
        circuit.append('PAULI_CHANNEL_1', targets, np.array([x_error_rate, y_error_rate, z_error_rate])) # {X,Y,Z,I}
        
        
    def add_CZ_neighbors_errors(self, circuit, cycle_length: int, gate_order: int, lost_qubit: int, lost_qubit_type: str, neighbors_by_order: dict, round_ix:int):
        # This function add noise channels on the neighbor qubits of the potential lost qubit. We apply correlated error channels according to the MLE correlated error channels we found in the paper.
        """_summary_

        Args:
            cycle_length (int): _description_
            gate_order (int): the index of the gate in which we potentially lost the qubit (0,1,2,3)
            lost_qubit_type (str): the type of qubit we potentially lost ('ancilla' or 'data')
            neighbors_by_order (dict): a dictionary of all the neighbors {0: [neigh, error type], 1: ...} where error_type is the error to put on the neighbor if we lose this qubit.
        """
        def add_error_channel_mle(neighbors_by_order, error_qubit_indices, probability, circuit):
            updated_error_qubit_indices = [i for i in error_qubit_indices if i in neighbors_by_order]
            if updated_error_qubit_indices != error_qubit_indices:
                stop=1
            
            # apply the correlated noise channels:
            if len(updated_error_qubit_indices) == 0:
                pass
            elif len(updated_error_qubit_indices) == 1: # single qubit channel
                [neigh_qubit, error_type] = neighbors_by_order[updated_error_qubit_indices[0]]
                error_channel = [probability,0,0] if error_type == 'X' else [0,0,probability]
                circuit.append('PAULI_CHANNEL_1', [neigh_qubit],  error_channel)
            elif len(updated_error_qubit_indices) == 2:
                [neigh_qubit0, error_type0] = neighbors_by_order[updated_error_qubit_indices[0]]
                [neigh_qubit1, error_type1] = neighbors_by_order[updated_error_qubit_indices[1]]
                # two_qubit_error_type = f"{error_type0}{error_type1}" # can be XX,XZ,ZX,ZZ
                circuit += stim.Circuit(f'CORRELATED_ERROR({probability}) {error_type0}{neigh_qubit0} {error_type1}{neigh_qubit1}')
            elif len(updated_error_qubit_indices) == 3:
                [neigh_qubit0, error_type0] = neighbors_by_order[updated_error_qubit_indices[0]]
                [neigh_qubit1, error_type1] = neighbors_by_order[updated_error_qubit_indices[1]]
                [neigh_qubit2, error_type2] = neighbors_by_order[updated_error_qubit_indices[2]]
                circuit += stim.Circuit(f'CORRELATED_ERROR({probability}) {error_type0}{neigh_qubit0} {error_type1}{neigh_qubit1} {error_type2}{neigh_qubit2}')
            elif len(updated_error_qubit_indices) == 4:
                [neigh_qubit0, error_type0] = neighbors_by_order[updated_error_qubit_indices[0]]
                [neigh_qubit1, error_type1] = neighbors_by_order[updated_error_qubit_indices[1]]
                [neigh_qubit2, error_type2] = neighbors_by_order[updated_error_qubit_indices[2]]
                [neigh_qubit3, error_type3] = neighbors_by_order[updated_error_qubit_indices[3]]
                circuit += stim.Circuit(f'CORRELATED_ERROR({probability}) {error_type0}{neigh_qubit0} {error_type1}{neigh_qubit1} {error_type2}{neigh_qubit2} {error_type3}{neigh_qubit3}')
        
        if lost_qubit == 3:
            stop = 1
        p = self.phys_error
        last_gate_index = max(neighbors_by_order)
        # Add error channels possible losses within the round or before according to mle logic:
        if gate_order == 0:
            # If we lose it after gate 0:
            probability = p
            error_qubit_indices = [1,2,3] if lost_qubit_type == 'data' else [0]
            add_error_channel_mle(neighbors_by_order, error_qubit_indices, probability, circuit)
        elif gate_order == 1:
            # If we lose it after gate 1:
            probability = (1-p)*p
            error_qubit_indices = [2,3] if lost_qubit_type == 'data' else [0,1]
            add_error_channel_mle(neighbors_by_order, error_qubit_indices, probability, circuit)
        elif gate_order == 2:
            # If we lose it after gate 2:
            probability = p*((1-p)**2)
            error_qubit_indices = [3] if lost_qubit_type == 'data' else [0,1,2] # can be [3]
            add_error_channel_mle(neighbors_by_order, error_qubit_indices, probability, circuit)
        elif gate_order == 3: # no errors on neighbors
            # If we lose it after gate 3:
            probability = p*((1-p)**3)
            error_qubit_indices = [] if lost_qubit_type == 'data' else [0,1,2,3] # can be []
            add_error_channel_mle(neighbors_by_order, error_qubit_indices, probability, circuit)

            
        if gate_order == last_gate_index:
            # Add loss channels for losses in previous QEC rounds:
            loss_in_round_prob = p + (1-p)*p + p*((1-p)**2) + p*((1-p)**3)
            current_cycle_beginning = next((cycle[0] for cycle in self.qubit_lifecycles_and_losses[lost_qubit] if cycle[0] <= round_ix <= cycle[1]), None)
            num_rounds_before = round_ix - current_cycle_beginning
            if num_rounds_before > 0: # there is a possibility that we lost this qubit in a past round
                logical_qubit =  self.logical_circuit.qubit_index_to_logical_qubit(lost_qubit)
                qubit_type = 'data' if lost_qubit in logical_qubit.data_qubits else 'ancilla'
                if qubit_type == 'data': # we need to add noise on neighbors if this qubit is now a data qubit
                    error_qubit_indices = [0,1,2,3]
                    probability = num_rounds_before * loss_in_round_prob
                    add_error_channel_mle(neighbors_by_order, error_qubit_indices, probability, circuit)
        
        
        

    def add_SWAP_neighbors_errors_ancilla_loss(self, cycle_length, circuit, ancilla_qubit_type, data_neigh_index):
        p = self.phys_error
        swap_error_prob = p + (1-p)*p + p*((1-p)**2) + p*((1-p)**3)
        error_channel = [swap_error_prob,0,0] if ancilla_qubit_type == 'Z' else [0,0,swap_error_prob]
        circuit.append('PAULI_CHANNEL_1', [data_neigh_index],  error_channel)

    def add_SWAP_neighbors_errors(self, cycle_length, circuit, qubit):
        p = self.phys_error
        loss_in_round_prob = p + (1-p)*p + p*((1-p)**2) + p*((1-p)**3)
        error_channel = [loss_in_round_prob/3,loss_in_round_prob/3,loss_in_round_prob/3]
        circuit.append('PAULI_CHANNEL_1', [qubit],  error_channel)
        
        
        
    # def add_CZ_neighbors_errors(self, circuit, cycle_length: int, gate_order: int, lost_qubit_type: str, neighbors_by_order: dict):
    #     # This function add noise channels on the neighbor qubits of the potential lost qubit. We apply correlated error channels according to the MLE correlated error channels we found in the paper.
    #     """_summary_

    #     Args:
    #         cycle_length (int): _description_
    #         gate_order (int): the index of the gate in which we potentially lost the qubit (0,1,2,3)
    #         lost_qubit_type (str): the type of qubit we potentially lost ('ancilla' or 'data')
    #         neighbors_by_order (dict): a dictionary of all the neighbors {0: [neigh, error type], 1: ...} where error_type is the error to put on the neighbor if we lose this qubit.
    #     """
    #     p = self.phys_error
    #     if gate_order == 0:
    #         probability = p
    #         error_qubit_indices = [1,2,3] if lost_qubit_type == 'data' else [0]
    #     elif gate_order == 1:
    #         probability = (1-p)*p
    #         error_qubit_indices = [2,3] if lost_qubit_type == 'data' else [0,1]
    #     elif gate_order == 2:
    #         probability = p*((1-p)**2)
    #         error_qubit_indices = [3] if lost_qubit_type == 'data' else [3]
    #     elif gate_order == 3: # no errors on neighbors
    #         probability = p*((1-p)**2)
    #         error_qubit_indices = [] if lost_qubit_type == 'data' else []
        
    #     updated_error_qubit_indices = [i for i in error_qubit_indices if i in neighbors_by_order]
    #     if updated_error_qubit_indices != error_qubit_indices:
    #         stop=1
    #     # apply the correlated noise channels:
    #     if len(updated_error_qubit_indices) == 0:
    #         pass
    #     elif len(updated_error_qubit_indices) == 1: # single qubit channel
    #         [neigh_qubit, error_type] = neighbors_by_order[updated_error_qubit_indices[0]]
    #         error_channel = [probability,0,0] if error_type == 'X' else [0,0,probability]
    #         circuit.append('PAULI_CHANNEL_1', [neigh_qubit],  error_channel)
    #     elif len(updated_error_qubit_indices) == 2:
    #         [neigh_qubit0, error_type0] = neighbors_by_order[updated_error_qubit_indices[0]]
    #         [neigh_qubit1, error_type1] = neighbors_by_order[updated_error_qubit_indices[1]]
    #         # two_qubit_error_type = f"{error_type0}{error_type1}" # can be XX,XZ,ZX,ZZ
    #         # circuit.append('CORRELATED_ERROR', f'{error_type0}{neigh_qubit0}', f'{error_type1}{neigh_qubit1}', probability)
    #         circuit += stim.Circuit(f'CORRELATED_ERROR({probability}) {error_type0}{neigh_qubit0} {error_type1}{neigh_qubit1}')
    #     elif len(updated_error_qubit_indices) == 3:
    #         [neigh_qubit0, error_type0] = neighbors_by_order[updated_error_qubit_indices[0]]
    #         [neigh_qubit1, error_type1] = neighbors_by_order[updated_error_qubit_indices[1]]
    #         [neigh_qubit2, error_type2] = neighbors_by_order[updated_error_qubit_indices[2]]
    #         # circuit.append('CORRELATED_ERROR', f'{error_type0}{neigh_qubit0}', f'{error_type1}{neigh_qubit1}', f'{error_type2}{neigh_qubit2}', probability)
    #         circuit += stim.Circuit(f'CORRELATED_ERROR({probability}) {error_type0}{neigh_qubit0} {error_type1}{neigh_qubit1} {error_type2}{neigh_qubit2}')
        