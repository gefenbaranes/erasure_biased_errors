import stim
import numpy as np
from typing import List
import scipy
from scipy.sparse import lil_matrix
from itertools import product
from scipy.sparse import dok_matrix, csc_matrix, csr_matrix
from scipy.sparse import vstack, csr_matrix
import os
import json
from hashlib import sha256
import pickle
import time
import copy
import itertools

class MLE_Loss_Decoder:
    def __init__(self, Meta_params:dict, bloch_point_params: dict, cycles: int, distance:int, ancilla_qubits:list, data_qubits:list, loss_detection_freq=None, printing=False, output_dir=None, first_comb_weight=0.5, loss_detection_method_str='SWAP', **kwargs) -> None:
        self.Meta_params = Meta_params
        self.bloch_point_params = {'erasure_ratio': '1', 'bias_ratio': '0.5'}
        self.bloch_point_params = bloch_point_params
        self.cycles = cycles
        self.distance = distance
        self.printing = printing
        self.loss_detection_freq = loss_detection_freq
        self.ancilla_qubits = ancilla_qubits
        self.data_qubits = data_qubits
        # self.lost_qubits_by_round_ix = {}  # {ld_round: [lost_qubits]}
        self.QEC_round_types = {}
        self.qubit_lifecycles_and_losses = {} # qubit: {[R_round, M_round, Lost?], [R_round, M_round, Lost?], ..}
        self._circuit = None
        self.gates_ordering_dict = {} # round_ix: {qubit: {gate_order: [neighbor_after_this_gate, error_if_qubit_is_lost] } }
        self.qubits_type_by_qec_round = {} # {qec_round: {index: type}} # TODO: fill this out. for every round the type of the qubit is the type before the SWAP operation.
        self.potential_losses_by_qec_round = {} # round_ix: {gate_before_loss: [lost_qubit, probability_of_this_event]}
        self.rounds_by_ix = {}
        # self.Pauli_DEM = None # detector error model for only Pauli errors
        self.real_losses_by_instruction_ix = {}
        self.loss_decoder_files_dir = f"{output_dir}/loss_circuits/{self.create_loss_file_name(self.Meta_params, self.bloch_point_params)}/d_{distance}__c_{cycles}"
        self.measurement_map = {}
        self.decoder_type = Meta_params['loss_decoder']
        self.loss_detection_method_str = loss_detection_method_str
        self.losses_to_detectors = []
        self.first_comb_weight = first_comb_weight

        
    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        self._circuit = value
        # Update self.extra_num_detectors whenever self.circuit is redefined
        # TODO: remove that, its saved in stim's dem
        observable_instructions_ix = [i for i, instruction in enumerate(self._circuit) if instruction.name == 'OBSERVABLE_INCLUDE']
        self.extra_num_detectors = len(observable_instructions_ix)
        

    def _generate_unique_key(self, losses_by_instruction_ix):
        sorted_losses_by_instruction_ix = {k: sorted(v) for k, v in sorted(losses_by_instruction_ix.items())}

        # Convert the data to a JSON string
        data_str = json.dumps({
            'losses_by_instruction_ix': sorted_losses_by_instruction_ix
            # 'meta_params': self.Meta_params,
            # 'distance': self.distance
        }, sort_keys=True)
        # Use SHA256 to generate a unique hash of the data
        return sha256(data_str.encode()).hexdigest()
    
    def create_loss_file_name(self, Meta_params, bloch_point_params={}):
        return f"{Meta_params['architecture']}__{Meta_params['code']}__{Meta_params['circuit_type']}__{Meta_params['num_logicals']}logicals__{Meta_params['logical_basis']}__{Meta_params['bias_preserving_gates']}__{Meta_params['noise']}__{Meta_params['is_erasure_biased']}__LD_freq_{Meta_params['LD_freq']}__SSR_{Meta_params['SSR']}__LD_method_{Meta_params['LD_method']}__ordering_{Meta_params['ordering']}"



    def initialize_loss_decoder(self, **kargs):
        self.set_up_Pauli_DEM()
        self.rounds_by_ix = self.split_stim_circuit_into_rounds()
        self.generate_measurement_map() # fill out self.measurement_map {measurement_index: (qubit, round_index)}
        # fill out self.qubit_lifecycles_and_losses (without the losses for now):
        
        if self.loss_detection_method_str == 'MBQC':
            self.get_qubits_lifecycles_MBQC()
        elif self.loss_detection_method_str == 'SWAP':
            self.get_qubits_lifecycles_SWAP() 
        elif self.loss_detection_method_str == 'FREE':
            self.get_qubits_lifecycles_FREE()
            
        self.qubit_lifecycles_and_losses_init = copy.deepcopy(self.qubit_lifecycles_and_losses)
        
        if self.printing:
            print(f"Using {self.loss_detection_method_str} method for {self.cycles} cycles and d = {self.distance}, self.qubit_lifecycles_and_losses = {self.qubit_lifecycles_and_losses}")
        
        if len(self.decoder_type) >= 11 and self.decoder_type[:11] ==  'independent': # Independent decoder:
            full_filename_dems = f'{self.loss_decoder_files_dir}/circuit_dems_1_losses.pickle'
            if not os.path.exists(full_filename_dems): # If needed - preprocess this circuit to get all relevant DEMs
                if self.printing:
                    print("Loss circuits need to be generated for these parameters. Starting pre-processing!")
                self.preprocess_circuit(full_filename = full_filename_dems)
            else:
                try:
                    with open(full_filename_dems, 'rb') as file:
                        self.circuit_independent_dems, _ = pickle.load(file) # Load the data from the file
                except EOFError as e:
                    print(f"EOFError: {e}. The file {full_filename_dems} might be corrupted. Regenerating the file.")
                    self.preprocess_circuit(full_filename=full_filename_dems)

        elif self.decoder_type == 'comb':
            self.circuit_comb_dems = {}
            for num_of_losses in [1,2,3,4,5]: # number of losses in the combination # TODO: change back to [1,2,3,4,5,6,7]
                full_filename_dems = f'{self.loss_decoder_files_dir}/circuit_dems_{num_of_losses}_losses.pickle'
                if not os.path.exists(full_filename_dems): # If needed - preprocess this circuit to get all relevant DEMs
                    if self.printing:
                        print(f"Loss circuits need to be generated for these parameters ({num_of_losses} losses). Starting pre-processing!")
                    
                    all_potential_loss_qubits_indices = self.get_all_potential_loss_qubits()
                    
                    # preprocess in batches:
                    self.preprocess_circuit_comb_batches(full_filename = full_filename_dems, num_of_losses=num_of_losses, all_potential_loss_qubits_indices=all_potential_loss_qubits_indices)
                    # self.preprocess_circuit_comb(full_filename = full_filename_dems, num_of_losses=num_of_losses, all_potential_loss_qubits_indices=all_potential_loss_qubits_indices)
                
                else:
                    # with open(full_filename_dems, 'rb') as file:
                    #     circuit_comb_dems, _ = pickle.load(file) # Load the data from the file
                    #     self.circuit_comb_dems.update(circuit_comb_dems) # merge
                    try:
                        with open(full_filename_dems, 'rb') as file:
                            circuit_comb_dems, _ = pickle.load(file) # Load the data from the file
                            self.circuit_comb_dems.update(circuit_comb_dems) # merge
                    except EOFError as e:
                        print(f"EOFError: {e}. The file {full_filename_dems} might be corrupted. Regenerating the file.")
                        
                        all_potential_loss_qubits_indices = self.get_all_potential_loss_qubits()
                        self.preprocess_circuit_comb(full_filename = full_filename_dems, num_of_losses=num_of_losses, all_potential_loss_qubits_indices=all_potential_loss_qubits_indices)


    
    def decode_loss_MLE(self, loss_detection_events):
        """ This function takes the original circuit with places for potential losses and loss detection events, and generates 2 circuits: 1. experimental measurement circuit. 2. Theory decoding circuit. """
        
        if True in loss_detection_events: # there is a loss in this shot:
            # Initialization for every shot:
            self.lost_qubits_by_round_ix = {}
            self.real_losses_by_instruction_ix = {} # {instruction_ix: (lost_qubit), ...}
            
            # First sweep: get location of lost qubits in the circuit --> for a given lost qubit we get a set of potential loss events.
            self.qubit_lifecycles_and_losses = copy.deepcopy(self.qubit_lifecycles_and_losses_init) # init self.qubit_lifecycles_and_losses for this shot
            # self.qubit_lifecycles_and_losses = self.qubit_lifecycles_and_losses_init.copy() # init self.qubit_lifecycles_and_losses for this shot
            self.update_real_losses_by_instruction_ix(loss_detection_events=loss_detection_events)
            self.update_qubit_lifecycles_and_losses()  # update self.qubit_lifecycles_and_losses
            
            # self.get_loss_location_SWAP(loss_detection_events=loss_detection_events) # old code, worked for SWAP only
            
            if self.printing:
                print(f"lost_qubits_by_round_ix={self.lost_qubits_by_round_ix}")
                print(f"types of rounds: {self.QEC_round_types}")
                print(f"lifecycles of qubits: {self.qubit_lifecycles_and_losses}\n")
            

            # Step 1 - generate the circuit that is really running in the experiment, for the given loss pattern (without gates after losing qubits):
            experimental_circuit = self.generate_loss_circuit(losses_by_instruction_ix = self.real_losses_by_instruction_ix, removing_Pauli_errors=False)
        
            # Step 2 - get all possible loss locations (and save in self.potential_losses_by_instruction_index[(lost_q, round_ix)])
            # self.get_all_potential_loss_locations_given_heralded_loss()
            self.get_all_potential_loss_locations_given_heralded_loss_new()

            
            # Step 3 - choose a decoder type:
            if len(self.decoder_type) >= 11 and self.decoder_type[:11] ==  'independent': # Independent decoder:
                final_dem = self.generate_all_DEMs_and_sum_over_independent_events(add_first_combination = True)
            else:
                # All combination decoder:
                self.all_potential_losses_combinations, self.combination_event_probability = self.generate_all_potential_losses_combinations(potential_losses_by_instruction_index = self.potential_losses_by_instruction_index)
                final_dem = self.generate_all_DEMs_and_sum_over_combinations()
                if self.printing:
                    print("Now lets see all loss pattern and which detectors were affected:")
                    for element in self.losses_to_detectors:
                        print(element)
                
            return experimental_circuit, final_dem
                
        else: # no losses in this shot
            experimental_circuit = self.circuit.copy()
            final_dem = experimental_circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)        
            return experimental_circuit, final_dem
    
    
    
    
    
    
    
    
    def generate_dem_loss_mle_experiment(self, measurement_event):
        """ This function takes the original circuit with places for potential losses and loss detection events, and generates 2 circuits: 1. experimental measurement circuit. 2. Theory decoding circuit. """
        
        # updated before for all shots together: self.qubit_lifecycles_and_losses, self.rounds_by_ix, self.measurement_map
        
        shot_had_a_loss = 2 in measurement_event

        if shot_had_a_loss:
            # Step 1 - find out which qubit was lost in which round:
            self.qubit_lifecycles_and_losses = self.qubit_lifecycles_and_losses_init.copy() # init self.qubit_lifecycles_and_losses for this shot
            self.update_lifecycle_from_detections(detection_event=measurement_event) # update self.qubit_lifecycles_and_losses according to measurement_events
            
            if self.printing:
                print(f"lifecycles of qubits: {self.qubit_lifecycles_and_losses}\n")

            # Step 2 - get all possible loss locations (and save in self.potential_losses_by_instruction_index[(lost_q, round_ix)])
            self.get_all_potential_loss_locations_given_heralded_loss_new()

            # Step 3 - choose a decoder type:
            if len(self.decoder_type) >= 11 and self.decoder_type[:11] ==  'independent': # Independent decoder:
                final_dem_hyperedges_matrix = self.generate_all_DEMs_and_sum_over_independent_events(add_first_combination = True, return_hyperedges_matrix=True)
            
            else:
                # All combination decoder:
                self.all_potential_losses_combinations, self.combination_event_probability = self.generate_all_potential_losses_combinations(potential_losses_by_instruction_index = self.potential_losses_by_instruction_index)
                final_dem_hyperedges_matrix = self.generate_all_DEMs_and_sum_over_combinations(return_hyperedges_matrix=True)
                if self.printing:
                    print("Now lets see all loss pattern and which detectors were affected:")
                    for element in self.losses_to_detectors:
                        print(element)
            

        else: # no losses in this shot, bring back regular DEM
            final_dem_hyperedges_matrix = self.Pauli_DEM_matrix # here observables are represented as detectors
            
        
        # Final step - remove observabes from hyperedgesmatrix and create a list of lists of errors that affect observables
        final_dem_hyperedges_matrix, observables_errors_interactions= self.convert_detectors_back_to_observables(final_dem_hyperedges_matrix)
            
        return final_dem_hyperedges_matrix, observables_errors_interactions

    
    def update_lifecycle_from_detections(self, detection_event):
        # Updating self.qubit_lifecycles_and_losses[qubit] and write lifecycle[2] = True when the qubit is lost in this lifecycle
        # self.lost_qubits_by_round_ix = {}
        for i, detection in enumerate(detection_event):
            if detection == 2:  # Qubit lost
                qubit, round_index = self.measurement_map[i]
                # Find the lifecycle phase to update
                for index, lifecycle in enumerate(self.qubit_lifecycles_and_losses[qubit.value]):
                    if lifecycle[0] <= round_index <= lifecycle[1]:
                        lifecycle[2] = True  # Mark as lost
                        # self.qubit_lifecycles_and_losses[qubit][index][2] = True
        
        
        
    def generate_measurement_map(self):
        # Build a mapping from measurement indices to qubits and measurement rounds.
        self.measurement_map = {}
        measurement_index = 0

        # Assume self.rounds_by_ix has the form {round_index: [instruction_list], ...}
        for round_index, instructions in self.rounds_by_ix.items():
            for instruction in instructions:
                if instruction.name in ['M', 'MX']:  # Check if it's a measurement instruction
                    targets = instruction.targets_copy()
                    for qubit in targets:
                        self.measurement_map[measurement_index] = (qubit, round_index)
                        measurement_index += 1
        
        
        
    def split_stim_circuit_into_rounds(self):
        # Takes self.circuit and decompose into cycles
        rounds = {}
        round_ix = -1
        inside_qec_round = False
        first_QEC_round = True
        current_round = []
        
        for instruction in self.circuit:
            if instruction.name == "TICK":
                if not inside_qec_round: # beginning of QEC round
                    if first_QEC_round: # starting first QEC round
                        rounds[round_ix] = current_round
                        round_ix += 1 ; first_QEC_round = False
                    current_round = []
                    current_round.append(instruction)    
                        
                else: # end of round
                    current_round.append(instruction)    
                    rounds[round_ix] = current_round
                    round_ix += 1
                    current_round = []
                    
                inside_qec_round = not inside_qec_round
                continue
            else:
                current_round.append(instruction)    
        
        # add final round (measurement round) to dictionary:
        rounds[round_ix] = current_round
        
        return rounds

    def update_loss_lists(self, instruction, loss_detection_events, lost_qubits_in_round, loss_detector_ix, instruction_ix):
        potential_lost_qubits = instruction.targets_copy()
        for q in potential_lost_qubits:
            if loss_detection_events[loss_detector_ix] == True:
                if q.value not in lost_qubits_in_round:
                    lost_qubits_in_round.append(q.value)
                
                # add loss to real_losses_by_instruction_ix:
                if instruction_ix in self.real_losses_by_instruction_ix:
                    self.real_losses_by_instruction_ix[instruction_ix].append(q.value)
                else:
                    self.real_losses_by_instruction_ix[instruction_ix] = [q.value]
                    
            loss_detector_ix += 1
        return loss_detector_ix
    
    
    def get_loss_location_SWAP(self, loss_detection_events: list):
        # OLD function
        # This function is similar to get_qubits_lifecycles, but it also fill out self.lost_qubits_by_round_ix. Relevant for theory.
        # Iterate through circuit. Every time we encounter a loss event (flagged by the 'I' gate), record the loss.
        loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
        round_ix = -1
        inside_qec_round = False
        SWAP_round_index = 0
        SWAP_round_type = None
        lost_qubits = [] # qubits that are lost and still undetectable (not measured)
        lost_qubits_in_round = [] # qubit lost in every QEC round. initialized every round.
        self.qubit_lifecycles_and_losses = {i: [] for i in self.ancilla_qubits + self.data_qubits}
        self.QEC_round_types = {} # {qec_round: type}
        qubit_active_cycle = {i: None for i in self.ancilla_qubits + self.data_qubits}
        first_QEC_round = True
        
        for instruction_ix, instruction in enumerate(self.circuit):
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
                    if first_QEC_round: # preparation round
                        self.lost_qubits_by_round_ix[round_ix] = lost_qubits_in_round # document losses in preparation round
                        lost_qubits_in_round = [] # lost qubits specifically in this QEC round
                        round_ix += 1; first_QEC_round = False
                    if self.printing:
                        print(f"Starting QEC Round {round_ix}")
                    
                    if (round_ix+1)%self.loss_detection_freq == 0:
                        SWAP_round = True
                        SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
                        SWAP_round_index += 1
                    else:
                        SWAP_round = False
                        
                else: # end of round
                    if self.printing:
                        print(f"Finished QEC Round {round_ix}, and lost qubits {lost_qubits_in_round}, thus now we have the following undetectable losses: {lost_qubits}")
                    self.QEC_round_types[round_ix] = SWAP_round_type if SWAP_round else 'regular'
                    self.lost_qubits_by_round_ix[round_ix] = lost_qubits_in_round
                    lost_qubits_in_round = [] # lost qubits specifically in this QEC round
                    

                    round_ix += 1
                inside_qec_round = not inside_qec_round
                continue
            
            if inside_qec_round:
                if instruction.name == 'I': # check loss event --> update lost_ancilla_qubits and lost_data_qubits
                    loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, lost_qubits_in_round, loss_detector_ix, instruction_ix)
                    
            else:
                if instruction.name == 'I': # loss event
                    loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, lost_qubits_in_round, loss_detector_ix, instruction_ix)
        
        # Handle unmeasured qubits at the end of the circuit (measurement round)
        for q in qubit_active_cycle:
            if qubit_active_cycle[q] is not None:
                self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix
        self.lost_qubits_by_round_ix[round_ix] = lost_qubits_in_round
        
    def update_real_losses_by_instruction_ix(self, loss_detection_events: list):
        # This function is similar to get_qubits_lifecycles, but it also fill out self.lost_qubits_by_round_ix. Relevant for theory.
        # Iterate through circuit. Every time we encounter a loss event (flagged by the 'I' gate), record the loss.
        loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
        round_ix = -1
        inside_qec_round = False
        SWAP_round_index = 0
        SWAP_round_type = None
        lost_qubits = [] # qubits that are lost and still undetectable (not measured)
        lost_qubits_in_round = [] # qubit lost in every QEC round. initialized every round.
        self.QEC_round_types = {} # {qec_round: type}
        # self.qubit_lifecycles_and_losses = {i: [] for i in self.ancilla_qubits + self.data_qubits}
        # qubit_active_cycle = {i: None for i in self.ancilla_qubits + self.data_qubits}
        first_QEC_round = True
        
        for instruction_ix, instruction in enumerate(self.circuit):
            # Check when each qubit is init and measured:
            # if instruction.name in ['R', 'RX']: # Beginning of a cycle for these qubits
                # qubits = set([q.value for q in instruction.targets_copy()])
                # for q in qubits:
                    # self.qubit_lifecycles_and_losses[q].append([round_ix, None, None]) # Begin a new cycle for each qubit
                    # qubit_active_cycle[q] = len(self.qubit_lifecycles_and_losses[q]) - 1

            if instruction.name in ['M', 'MX']: # End of a cycle for these qubits
                qubits = set([q.value for q in instruction.targets_copy()])
                lost_qubits.extend(lost_qubits_in_round)
                for q in qubits:
                    if q in lost_qubits:
                        # self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][2] = True
                        lost_qubits.remove(q)
                    # else:
                        # self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][2] = False
                    # self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix # Close the active cycle with the measurement round
                    # qubit_active_cycle[q] = None

                        
            # QEC rounds:
            if instruction.name == 'TICK':
                
                if not inside_qec_round: # beginning of QEC round
                    if first_QEC_round: # preparation round
                        self.lost_qubits_by_round_ix[round_ix] = lost_qubits_in_round # document losses in preparation round
                        lost_qubits_in_round = [] # lost qubits specifically in this QEC round
                        round_ix += 1; first_QEC_round = False
                    if self.printing:
                        print(f"Starting QEC Round {round_ix}")
                    
                    if (round_ix+1)%self.loss_detection_freq == 0:
                        SWAP_round = True
                        SWAP_round_type = 'even' if SWAP_round_index%2 ==0 else 'odd'
                        SWAP_round_index += 1
                    else:
                        SWAP_round = False
                        
                else: # end of round
                    if self.printing:
                        print(f"Finished QEC Round {round_ix}, and lost qubits {lost_qubits_in_round}, thus now we have the following undetectable losses: {lost_qubits}")
                    self.QEC_round_types[round_ix] = SWAP_round_type if SWAP_round else 'regular'
                    self.lost_qubits_by_round_ix[round_ix] = lost_qubits_in_round
                    lost_qubits_in_round = [] # lost qubits specifically in this QEC round
                    

                    round_ix += 1
                inside_qec_round = not inside_qec_round
                continue
            
            if inside_qec_round:
                if instruction.name == 'I': # check loss event --> update lost_ancilla_qubits and lost_data_qubits
                    loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, lost_qubits_in_round, loss_detector_ix, instruction_ix)
                    
            else:
                if instruction.name == 'I': # loss event
                    loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, lost_qubits_in_round, loss_detector_ix, instruction_ix)
        
        # Handle unmeasured qubits at the end of the circuit (measurement round)
        # for q in qubit_active_cycle:
            # if qubit_active_cycle[q] is not None:
                # self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix
        self.lost_qubits_by_round_ix[round_ix] = lost_qubits_in_round
        
    def update_qubit_lifecycles_and_losses(self):
        # For theory only. Takes self.real_losses_by_instruction_ix = {} # {instruction_ix: (lost_qubit), ...} and fillout self.qubit_lifecycles_and_losses according to the losses
        for instruction_ix in self.real_losses_by_instruction_ix:
            lost_qubits = self.real_losses_by_instruction_ix[instruction_ix]
            round_ix = next((round_ix for round_ix, instructions in sorted(self.rounds_by_ix.items()) if sum(len(self.rounds_by_ix[r]) for r in range(-1, round_ix+1)) > instruction_ix), None)
            for lost_q in lost_qubits:
                relevant_cycle = next((i for i, cycle in enumerate(self.qubit_lifecycles_and_losses[lost_q]) 
                                    if cycle[0] <= round_ix <= cycle[1]), None)
                self.qubit_lifecycles_and_losses[lost_q][relevant_cycle][2] = True
        
        # TODO: maybe fillout all other values with False
        
        
    def generate_loss_circuit(self, losses_by_instruction_ix, removing_Pauli_errors=False):
        
        def fill_loss_qubits_remove_gates_range(lost_q, instruction_ix):
            round_ix = next((round_ix for round_ix, instructions in sorted(self.rounds_by_ix.items()) if sum(len(self.rounds_by_ix[r]) for r in range(-1, round_ix+1)) > instruction_ix), None)
            [reset_round_ix, detection_round_ix] = next(([cycle[0],cycle[1]] for cycle in self.qubit_lifecycles_and_losses[lost_q] if cycle[0] <= round_ix <= cycle[1]), None)
            # detection_round_offset_start = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < reset_round_ix)
            detection_round_offset_end = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < detection_round_ix + 1)
            
            if lost_q in loss_qubits_remove_gates_range: # this qubit was already recorder for loss, and was lost again!
                loss_qubits_remove_gates_range[lost_q].append((instruction_ix, detection_round_offset_end))
            else:
                loss_qubits_remove_gates_range[lost_q] = [(instruction_ix, detection_round_offset_end)]
            return loss_qubits_remove_gates_range
            
        lost_qubits = set(qubit for sublist in losses_by_instruction_ix.values() for qubit in sublist)
        first_loss_instruction_index = min(losses_by_instruction_ix)
        loss_qubits_remove_gates_range = {}
        for instruction_ix in losses_by_instruction_ix:
            lost_qubits_instruction = losses_by_instruction_ix[instruction_ix]
            
            if isinstance(lost_qubits_instruction, list):
                for lost_q in lost_qubits_instruction:
                    loss_qubits_remove_gates_range = fill_loss_qubits_remove_gates_range(lost_q, instruction_ix)
            else:
                lost_q = lost_qubits_instruction
                loss_qubits_remove_gates_range = fill_loss_qubits_remove_gates_range(lost_q, instruction_ix)
                
        first_loss_instruction_index = 0
        circuit_before_ix = self.circuit[:first_loss_instruction_index]
        circuit_after_ix = self.circuit[first_loss_instruction_index:]
        
        heralded_circuit_after_ix = self.generate_circuit_without_lost_qubit(lost_qubits = lost_qubits, circuit = circuit_after_ix, circuit_offset = first_loss_instruction_index,loss_qubits_remove_gates_range=loss_qubits_remove_gates_range, removing_Pauli_errors=removing_Pauli_errors) # after removing the following gates with the lost qubits.
        experimental_circuit = circuit_before_ix + heralded_circuit_after_ix
        
        
        return experimental_circuit
        
        
    def generate_dem_loss_circuit(self, losses_by_instruction_ix, event_probability=1, full_filename=''):
        """ given a dictionary of losses, this function generates the circuit while considering the losses.
        """
        # Generate a unique key based on losses_by_instruction_ix and self.Meta_params
        key = self._generate_unique_key(losses_by_instruction_ix) 

        if os.path.isfile(full_filename): # Load and return the existing dem
            with open(full_filename, 'rb') as file:  # Note the 'rb' mode here
                circuit_independent_dems, _ = pickle.load(file)
                if key in circuit_independent_dems:
                    hyperedges_matrix_dem = circuit_independent_dems[key]
                    return hyperedges_matrix_dem
            
        else: # generate circuit and dem
            loss_circuit = self.generate_loss_circuit(losses_by_instruction_ix, removing_Pauli_errors=True)
            
            # replace final observables with detectors:
            final_loss_circuit = self.observables_to_detectors(loss_circuit)

            # get the dem (with observables on columns):
            dem_heralded_circuit = final_loss_circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=True)
            
            # convert the DEM into a matrix and sum up with previous DEMs:
            hyperedges_matrix_dem = self.convert_dem_into_hyperedges_matrix(dem_heralded_circuit, event_probability=event_probability, observables_converted_to_detectors=True)
            
            if self.printing:
                print(f"The DEM for the loss circuit with losses {losses_by_instruction_ix} is: \n{dem_heralded_circuit}")
                print(hyperedges_matrix_dem)
                print(f"If we would sample this circuit 5 times, we would get:")
                sampler = final_loss_circuit.compile_detector_sampler()
                detection_events, observable_flips = sampler.sample(5, separate_observables=True)
                print(f"detection_events = \n{detection_events}, observable_flips = \n{observable_flips}")

            # for debugging:
            rows_list = []
            for row in range(hyperedges_matrix_dem.shape[0]):
                # Find the indices of non-zero elements in this row.
                row_data = hyperedges_matrix_dem.getrow(row).tocoo()
                non_zero_indices = row_data.col.tolist()
                rows_list.append(non_zero_indices)
            
            self.losses_to_detectors.append([losses_by_instruction_ix, rows_list])
                
                
                
            # Once the circuit is generated, save it along with self.Meta_params
            # with open(full_filename, 'wb') as file:
            #     pickle.dump((hyperedges_matrix_dem, extra_num_detectors, self.Meta_params), file)


            return hyperedges_matrix_dem
    
    def get_all_potential_loss_qubits(self):
        all_potential_loss_qubits_indices = []
        instruction_ix = 0
        for round_ix, round_instructions in self.rounds_by_ix.items():
            for instruction in round_instructions:
                if instruction.name == 'I': # potential loss event
                    targets = instruction.targets_copy()
                    for q in targets:
                        qubit = q.value
                        losses_by_instruction_ix = {instruction_ix: [qubit]}
                        num_potential_losses = 0
                        [reset_round_ix, detection_round_ix] = next(([cycle[0],cycle[1]] for cycle in self.qubit_lifecycles_and_losses[qubit] if cycle[0] <= round_ix <= cycle[1]), None)
                        potential_rounds_for_loss_events = range(reset_round_ix, detection_round_ix+1)
                        for potential_round in potential_rounds_for_loss_events:
                            round_instructions = self.rounds_by_ix[potential_round]
                            round_offset = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < potential_round)
                            losses_indices_in_round = [i + round_offset for i, instruction in enumerate(round_instructions) if instruction.name == 'I' and qubit in [q.value for q in instruction.targets_copy()]]
                            num_potential_losses += len(losses_indices_in_round)
                        event_probability = 1/num_potential_losses
                        loss_event = (instruction_ix, qubit, event_probability)
                        all_potential_loss_qubits_indices.append(loss_event)
                instruction_ix += 1
        return all_potential_loss_qubits_indices
    
    def preprocess_circuit_comb(self, full_filename, num_of_losses=1, all_potential_loss_qubits_indices=[]):
        # Here we look at the lifetimes of each qubit, to get all possible independent loss channels. Each channel corresponds to a qubit lifetime and contains all potential loss places.
        # We would like to generate a DEM for the loss of every single qubit in every location of the circuit and save it.
        # Also, we generate all DEMs for combinations of qubit losses.
        # num_of_losses = how many loss events are they. If 1 --> same as independent function.
        
        start_time = time.time()
        os.makedirs(os.path.dirname(full_filename), exist_ok=True) # Ensure the directory exists
                
        ### Step 1: get all lost events:
        # already implemented before once, we get all_potential_loss_qubits_indices.
                
        
        ### Step 2: get all combinations with num_of_losses losses:
        all_combinations = list(itertools.combinations(all_potential_loss_qubits_indices, num_of_losses))
        total_combinations = len(all_combinations)
        batch_size = max(1, int(total_combinations * 0.05))
    
        ### Step 3: process all combinations:
        circuit_comb_dems = {}
        for combination in all_combinations:
            losses_by_instruction_ix = {}
            combination_event_probability = 1
            for (instruction_ix, qubit, event_probability) in combination:
                combination_event_probability *= event_probability
                if instruction_ix not in losses_by_instruction_ix:
                    losses_by_instruction_ix[instruction_ix] = [qubit]
                else:
                    losses_by_instruction_ix[instruction_ix].append(qubit)
            
            hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = losses_by_instruction_ix, event_probability=combination_event_probability, full_filename=full_filename)
            key = self._generate_unique_key(losses_by_instruction_ix)
            circuit_comb_dems[key] = hyperedges_matrix_dem
                        
        end_time = time.time()
        print(f"building all {len(all_combinations)} loss circuits for {num_of_losses} losses took {end_time - start_time} sec. Saving the result!")
        
        with open(full_filename, 'wb') as file:
            pickle.dump((circuit_comb_dems, self.Meta_params), file)
        
        # combine into the full dictionary with all num_of_losses dems
        self.circuit_comb_dems.update(circuit_comb_dems)
        
        
    def preprocess_circuit_comb_batches(self, full_filename, num_of_losses=1, all_potential_loss_qubits_indices=[]):
        # Here we look at the lifetimes of each qubit, to get all possible independent loss channels. Each channel corresponds to a qubit lifetime and contains all potential loss places.
        # We would like to generate a DEM for the loss of every single qubit in every location of the circuit and save it.
        # Also, we generate all DEMs for combinations of qubit losses.
        # num_of_losses = how many loss events are they. If 1 --> same as independent function.
        
        batches_dir = f'{self.loss_decoder_files_dir}/batches_{num_of_losses}'
        os.makedirs(batches_dir, exist_ok=True)  # Ensure the batch directory exists


        # Step 1: Get all combinations with num_of_losses losses
        all_combinations = list(itertools.combinations(all_potential_loss_qubits_indices, num_of_losses))
        total_combinations = len(all_combinations)
        batch_size = max(1, int(total_combinations * 0.05))
        print(f"For d={self.distance}, num of losses = {num_of_losses}, we got {total_combinations} combinations to process.")

        # Step 2: Process all combinations in batches
        for batch_start in range(0, total_combinations, batch_size):
            batch_end = min(batch_start + batch_size, total_combinations)
            full_filename_batch = f'{batches_dir}/batch_{batch_start}_to_{batch_end}.pickle'
            
            # Check if this batch was already processed
            if os.path.exists(full_filename_batch):
                print(f"Batch {batch_start} to {batch_end} already processed. Skipping.")
                continue # Continue to the next batch

            circuit_comb_dems = {} # Process this batch
            batch_combinations = all_combinations[batch_start:batch_end]
            
            for combination in batch_combinations:
                losses_by_instruction_ix = {}
                combination_event_probability = 1
                for (instruction_ix, qubit, event_probability) in combination:
                    combination_event_probability *= event_probability
                    if instruction_ix not in losses_by_instruction_ix:
                        losses_by_instruction_ix[instruction_ix] = [qubit]
                    else:
                        losses_by_instruction_ix[instruction_ix].append(qubit)

                key = self._generate_unique_key(losses_by_instruction_ix)
                if key not in circuit_comb_dems:
                    hyperedges_matrix_dem = self.generate_dem_loss_circuit(
                        losses_by_instruction_ix=losses_by_instruction_ix,
                        event_probability=combination_event_probability,
                        full_filename=full_filename_batch
                    )
                    circuit_comb_dems[key] = hyperedges_matrix_dem

            # Save the current batch results
            try:
                with open(full_filename_batch, 'wb') as file:
                    pickle.dump((circuit_comb_dems, self.Meta_params), file)
            except Exception as e:
                print(f"Error saving batch {batch_start} to {batch_end}: {e}")
                continue

            # Clear memory after processing and saving each batch
            del circuit_comb_dems, batch_combinations
            print(f"Processed batch {batch_start // batch_size + 1}/{(total_combinations + batch_size - 1) // batch_size + 1}")

        # Step 3: Combine all batch dictionaries into a single dictionary
        print(f"Now we will combine all batches into one! taking batches in folder {batches_dir}")
        combined_circuit_comb_dems = {}

        for batch_file in os.listdir(batches_dir):
            if batch_file.endswith('.pickle'):
                try:
                    with open(os.path.join(batches_dir, batch_file), 'rb') as file:
                        batch_circuit_comb_dems, _ = pickle.load(file)
                        combined_circuit_comb_dems.update(batch_circuit_comb_dems)
                except Exception as e:
                    print(f"Error loading batch file {batch_file}: {e}")
                    continue
                finally:
                    # Clear memory after loading and updating the combined dictionary
                    del batch_circuit_comb_dems

        # Step 4: Save the combined dictionary to full_filename
        try:
            with open(full_filename, 'wb') as file:
                pickle.dump((combined_circuit_comb_dems, self.Meta_params), file)
        except Exception as e:
            print(f"Error saving combined results to {full_filename}: {e}")

        print(f"All batches combined and saved to {full_filename}")

        # Step 5: Load the combined results into self.circuit_comb_dems
        self.circuit_comb_dems.update(combined_circuit_comb_dems)  # merge

        # Clear memory after loading combined results
        del combined_circuit_comb_dems
        
        
        
        
    def preprocess_circuit(self, full_filename):
        # Here we look at the lifetimes of each qubit, to get all possible independent loss channels. Each channel corresponds to a qubit lifetime and contains all potential loss places.
        # We would like to generate a DEM for the loss of every single qubit in every location of the circuit and save it.
        if self.printing:
            print("Preprocessing all loss circuits, one time only and it will be saved for next times!")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_filename), exist_ok=True)

        self.circuit_independent_dems = {}
        instruction_ix = 0
        for round_ix, round_instructions in self.rounds_by_ix.items():
            for instruction in round_instructions:
                if instruction.name == 'I':
                    targets = instruction.targets_copy()
                    for q in targets:
                        qubit = q.value
                        losses_by_instruction_ix = {instruction_ix: [qubit]}
                        num_potential_losses = 0
                        [reset_round_ix, detection_round_ix] = next(([cycle[0],cycle[1]] for cycle in self.qubit_lifecycles_and_losses[qubit] if cycle[0] <= round_ix <= cycle[1]), None)
                        potential_rounds_for_loss_events = range(reset_round_ix, detection_round_ix+1)
                        for potential_round in potential_rounds_for_loss_events:
                            round_instructions = self.rounds_by_ix[potential_round]
                            round_offset = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < potential_round)
                            losses_indices_in_round = [i + round_offset for i, instruction in enumerate(round_instructions) if instruction.name == 'I' and qubit in [q.value for q in instruction.targets_copy()]]
                            num_potential_losses += len(losses_indices_in_round)
                        
                        hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = losses_by_instruction_ix, event_probability = 1/num_potential_losses, full_filename=full_filename)
                        # save into the file:
                        key = self._generate_unique_key(losses_by_instruction_ix)
                        self.circuit_independent_dems[key] = hyperedges_matrix_dem
                        
                instruction_ix += 1
                
        with open(full_filename, 'wb') as file:
            pickle.dump((self.circuit_independent_dems, self.Meta_params), file)

        
        
        
    def get_all_potential_loss_locations_given_heralded_loss(self):
        # Loop over all lost qubits and for each one, mark a potential loss event with a certain probability. old function, compatible only with theory.
        self.potential_losses_by_instruction_index = {} # {(lost_q,loss_round_ix): {loss_instruction_ix: [lost_qubit, probability_of_this_event]}}
        if self.printing:
            print(f"self.lost_qubits_by_round_ix: {self.lost_qubits_by_round_ix}")
        for round_ix in self.lost_qubits_by_round_ix:
            lost_qubits_in_round = self.lost_qubits_by_round_ix[round_ix]
            for lost_q in lost_qubits_in_round:
                if (lost_q, round_ix) not in self.potential_losses_by_instruction_index:
                    self.potential_losses_by_instruction_index[(lost_q, round_ix)] = {}
                
                num_potential_losses = 0
                loss_indices = []
                [reset_round_ix, detection_round_ix] = next(([cycle[0],cycle[1]] for cycle in self.qubit_lifecycles_and_losses[lost_q] if cycle[0] <= round_ix <= cycle[1]), None)
                # relevant_cycle_length = detection_round_ix - reset_round_ix + 1
                potential_rounds_for_loss_events = range(reset_round_ix, detection_round_ix+1)
                for potential_round in potential_rounds_for_loss_events:
                    round_instructions = self.rounds_by_ix[potential_round]
                    round_offset = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < potential_round)
                    losses_indices_in_round = [i + round_offset for i, instruction in enumerate(round_instructions) if instruction.name == 'I' and lost_q in [q.value for q in instruction.targets_copy()]]
                    for potential_loss_index in losses_indices_in_round:
                        self.potential_losses_by_instruction_index[(lost_q, round_ix)][potential_loss_index] = [None, detection_round_ix] # loss index: prob, detection_round_ix
                        num_potential_losses += 1
                        loss_indices.append(potential_loss_index)
                
                if num_potential_losses > 0:
                    loss_probability = 1 / num_potential_losses
                    for potential_loss_index in loss_indices:
                        self.potential_losses_by_instruction_index[(lost_q, round_ix)][potential_loss_index][0] = loss_probability
        if self.printing:
            print(f"potential_losses_by_instruction_index: {self.potential_losses_by_instruction_index}")
    
    def get_all_potential_loss_locations_given_heralded_loss_new(self):
        # Loop over all lost qubits and for each one, mark a potential loss event with a certain probability
        self.potential_losses_by_instruction_index = {} # {(lost_q,loss_round_ix): {loss_instruction_ix: [lost_qubit, probability_of_this_event]}}
        # if self.printing:
        #     print(f"self.lost_qubits_by_round_ix: {self.lost_qubits_by_round_ix}")
        
        for qubit in self.qubit_lifecycles_and_losses:
            qubit = int(qubit)
            qubit_lifecycles = self.qubit_lifecycles_and_losses[qubit]
            for lifecycle in qubit_lifecycles:
                reset_round_ix, detection_round_ix, lost_in_cycle = lifecycle
                if lost_in_cycle: # qubit was lost somewhere in this lifecycles
                    if (qubit, detection_round_ix) not in self.potential_losses_by_instruction_index:
                        self.potential_losses_by_instruction_index[(qubit, detection_round_ix)] = {}
                    num_potential_losses = 0
                    loss_indices = []
                    potential_rounds_for_loss_events = range(reset_round_ix, detection_round_ix+1)
                    for potential_round in potential_rounds_for_loss_events:
                        round_instructions = self.rounds_by_ix[potential_round]
                        round_offset = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < potential_round)
                        losses_indices_in_round = [i + round_offset for i, instruction in enumerate(round_instructions) if instruction.name == 'I' and qubit in [q.value for q in instruction.targets_copy()]]
                        for potential_loss_index in losses_indices_in_round:
                            self.potential_losses_by_instruction_index[(qubit, detection_round_ix)][potential_loss_index] = [None, detection_round_ix] # loss index: prob, detection_round_ix
                            num_potential_losses += 1
                            loss_indices.append(potential_loss_index)
                    
                    if num_potential_losses > 0:
                        loss_probability = 1 / num_potential_losses
                        for potential_loss_index in loss_indices:
                            self.potential_losses_by_instruction_index[(qubit, detection_round_ix)][potential_loss_index][0] = loss_probability
                        
        if self.printing:
            print(f"potential_losses_by_instruction_index: {self.potential_losses_by_instruction_index}")
    
    
    
    def generate_all_potential_losses_combinations(self, potential_losses_by_instruction_index):
        """ 
        Input: a dictionary with loss events, key = (lost_q, round_ix).
        Output: a list of dictionary, each one represent another combination of losses. 
        Each dictionary: key: instruction_ix, value: qubit
        """
        potential_losses = [ [((lost_q, round_ix), potential_loss_index) + tuple(loss_info) for potential_loss_index, loss_info in potential_losses_by_instruction_index[(lost_q, round_ix)].items()]
        for (lost_q, round_ix) in potential_losses_by_instruction_index ]
        
        # # Use itertools.product to get the Cartesian product of all possible losses
        all_combinations = list(product(*potential_losses))
        
        # # Calculate the combination event probability
        combination_event_probability = 1
        for event in potential_losses:
            # Assumes each event list contains at least one loss event and all probabilities are the same for each event
            combination_event_probability *= event[0][2] 
        
        # Convert each combination into the desired dictionary format
        combination_dicts = []
        for combination in all_combinations:
            combination_dict = {}
            for (lost_q, round_ix), instruction_ix, probability, meas_round_ix in combination:
                if instruction_ix not in combination_dict:
                    combination_dict[instruction_ix] = [lost_q]
                else:
                    combination_dict[instruction_ix].append(lost_q)
            
            combination_dicts.append(combination_dict)
        
        return combination_dicts, combination_event_probability

    
    def generate_all_DEMs_and_sum_over_combinations(self, return_hyperedges_matrix=False, use_pre_processed_data=True):
        # Now we generate many DEMs for every potential loss event combination and merge together in 2 steps in order to get 1 final DEM:
        # event_probability is the probability for each loss event combination
        
        DEMs_loss_pauli_events = [self.Pauli_DEM_matrix] # list of all DEMs for every loss event + DEM for Pauli errors.
        num_detectors = self.Pauli_DEM_matrix.shape[1]
        hyperedges_matrix_loss_event = dok_matrix((0, num_detectors), dtype=float) # Initialize as dok_matrix for efficient incremental construction
        
        DEMs_loss_events = []
        for potential_loss_combination in self.all_potential_losses_combinations:
            key = self._generate_unique_key(potential_loss_combination)
            if use_pre_processed_data and key in self.circuit_comb_dems:
                hyperedges_matrix_dem = self.circuit_comb_dems[key]
            else:
                print(f"Combination {potential_loss_combination} not in dictionary. need to generate loss circuit")
                hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = potential_loss_combination, event_probability=self.combination_event_probability)
            
            DEMs_loss_events.append(hyperedges_matrix_dem)
                        
                
        # sum over all loss DEMs:
        hyperedges_matrix_loss_event = self.combine_DEMs_sum(DEMs_list=DEMs_loss_events, num_detectors=num_detectors)
        
        # save the sum of the DEMs for this loss event in DEMs_loss_pauli_events
        # hyperedges_matrix_loss_event = hyperedges_matrix_loss_event.tocsr()
        DEMs_loss_pauli_events.append(hyperedges_matrix_loss_event)
        
        if self.printing:
            print(f"After summing over all DEMs for potential loss combinations, we got the following DEM_loss:")
            print(hyperedges_matrix_loss_event)
                
        # Step 2: sum over loss DEM + Pauli errors DEM, according to the high-order formula:
        final_hyperedges_matrix = self.combine_DEMs_high_order(DEMs_list=DEMs_loss_pauli_events, num_detectors=num_detectors)

        # Step 3: convert the final dem into rows:
        if self.printing:
            print(f"Final DEM matrix after combining all DEMs for losses and Pauli: {final_hyperedges_matrix}")

        if return_hyperedges_matrix:
            return final_hyperedges_matrix
        else:
            # Step 4: bring back the observables and create a stim.DEM object: 
            final_dem = self.from_hyperedges_matrix_into_stim_dem(final_hyperedges_matrix)
            return final_dem
    
    
    def generate_all_DEMs_and_sum_over_independent_events(self, add_first_combination = False, use_pre_processed_data = True, return_hyperedges_matrix = False):
        # if add_first_combination = True, we add the DEM of the first loss combination (only if >1 loss event happened)
        # Now we generate many DEMs for every loss event and merge together in 2 steps in order to get 1 final DEM:
        DEMs_loss_pauli_events = [self.Pauli_DEM_matrix] # list of all DEMs for every loss event + DEM for Pauli errors. TODO: add here the Pauli DEM
        num_detectors = self.Pauli_DEM_matrix.shape[1]                
        
        for (lost_q, detection_round_ix) in self.potential_losses_by_instruction_index:
            DEMs_specific_loss_event = []
            for ix, potential_loss_ix in enumerate(self.potential_losses_by_instruction_index[(lost_q, detection_round_ix)]):
                [event_probability, detection_round_ix2] = self.potential_losses_by_instruction_index[(lost_q, detection_round_ix)][potential_loss_ix]
                losses_by_instruction_ix = {potential_loss_ix: [lost_q]}
                key = self._generate_unique_key(losses_by_instruction_ix)
                if use_pre_processed_data and key in self.circuit_independent_dems:
                    hyperedges_matrix_dem = self.circuit_independent_dems[key]
                else:
                    hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = losses_by_instruction_ix, event_probability=event_probability)
                DEMs_specific_loss_event.append(hyperedges_matrix_dem)
            
            DEM_specific_loss_event = self.combine_DEMs_sum(DEMs_list=DEMs_specific_loss_event, num_detectors=num_detectors)
            DEMs_loss_pauli_events.append(DEM_specific_loss_event)
            
            if self.printing:
                print(f"After summing over all DEMs for potential loss events given the loss of qubit {lost_q}, which was detected in round {detection_round_ix}, we got the following DEM_i:")
                print(DEM_specific_loss_event)
        
        if add_first_combination and len(self.potential_losses_by_instruction_index) > 1:
            first_combination_dict = {}; combination_probability = 1
            for (lost_q, detection_round_ix) in self.potential_losses_by_instruction_index:
                first_potential_loss_ix = min(self.potential_losses_by_instruction_index[(lost_q, detection_round_ix)].keys())
                [event_probability, detection_round_ix2] = self.potential_losses_by_instruction_index[(lost_q, detection_round_ix)][first_potential_loss_ix]
                combination_probability *= event_probability
                if first_potential_loss_ix in first_combination_dict:
                    first_combination_dict[first_potential_loss_ix].append(lost_q)
                else:
                    first_combination_dict[first_potential_loss_ix] = [lost_q]
            # adjust first_comb probability according to the input:
            updated_combination_probability = self.first_comb_weight if self.first_comb_weight > 0 else combination_probability
            hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = first_combination_dict, event_probability=updated_combination_probability)
            DEMs_loss_pauli_events.append(hyperedges_matrix_dem)
            
        # sum over all loss DEMs:
        final_hyperedges_matrix = self.combine_DEMs_high_order(DEMs_list=DEMs_loss_pauli_events, num_detectors=num_detectors)
                
        if self.printing:
            print(f"After summing over all losses DEMS + Pauli DEM (high order equation), we got the final DEM for independent losses decoder:")
            print(final_hyperedges_matrix)
        
        if return_hyperedges_matrix:
            return final_hyperedges_matrix
        else:
            # bring back the observables and create a stim.DEM object: 
            final_dem = self.from_hyperedges_matrix_into_stim_dem(final_hyperedges_matrix)
            return final_dem
            
    
    def convert_hyperedge_matrix_into_binary(self, hyperedges_matrix):
        # Convert to binary matrix and extract row-wise values
        binary_matrix = hyperedges_matrix.copy()
        probs_lists = []

        for i in range(hyperedges_matrix.shape[0]):
            row_data = hyperedges_matrix.getrow(i)
            if row_data.nnz > 0:  # if the row is not entirely zero
                probs_lists.append(row_data.data[0][0])  # Collect the non-zero values before changing them
                binary_matrix.rows[i] = row_data.rows[0]
                binary_matrix.data[i] = np.ones_like(row_data.data[0])
            # else:
            #     probs_lists.append(0)
        
        return binary_matrix, probs_lists

    
    def convert_detectors_back_to_observables(self, dem_hyperedges_matrix):
        """ 
        Input:
        This function takes the hyperedgematrix with rows representing errors and columns representing detectors + observables.
        It returns:
        * updated hyperedge matrix without the columns that represented observables
        * list of lists observables, where for each observable we write the indices of the errors that interact with this observable (the non-zero rows in the relevant row).
        
        Observables columns are in indices: self.observables_indices
        """
        
        # Step 1 - generate observables_errors_interactions lists: (each element in the upper list is an observable)
        observables_errors_interactions = []
        for observable_index in self.observables_indices:
            # Find the rows where the observable is involved
            error_indices = dem_hyperedges_matrix[:, observable_index].nonzero()[0]
            observables_errors_interactions.append(error_indices.tolist())
        
        # Step 2 - remove observables from hyperedge matrix
        mask = np.array([i for i in range(dem_hyperedges_matrix.shape[1]) if i not in self.observables_indices])
        # Apply the mask to get the updated hyperedge matrix without observables columns
        dem_hyperedges_matrix_updated = dem_hyperedges_matrix[:, mask]
    
        return dem_hyperedges_matrix_updated, observables_errors_interactions
        
        
    def from_hyperedges_matrix_into_stim_dem(self, final_hyperedges_matrix):
        # Input hyperedges_matrix includes observables as detectors (in columns self.observables_indices). We need to convert them back into observables.
        
        # Step 4: bring back the observables and create a stim.DEM object: 
        final_dem = stim.DetectorErrorModel()
        
        # Iterate over the rows of the hypergraph_matrix to create the DEM while adjusting the circuit based on the hyperedges_matrix to re-include observables
        for row_index in range(final_hyperedges_matrix.shape[0]):
            row = final_hyperedges_matrix.getrow(row_index)
            non_zero_columns = row.nonzero()[1]
            if self.loss_decoder_files_dir[:9] == '/n/home01': # on the cluster 
                probability = row.data[0] # Assuming all non-zero entries in a row have the same error probability. on the cluster only one [0]
            else:
                # probability = row.data[0]
                probability = row.data[0][0]  # Assuming all non-zero entries in a row have the same error probability. on the cluster only one [0]

            # Construct the error command by specifying detector and observable targets
            error_targets = []
            num_detectors = final_hyperedges_matrix.shape[1] - self.extra_num_detectors  # number of detector columns

            # Append detectors and convert detectors to observables:
            for d in non_zero_columns: # detector or observable index, from 0 up to num detectors + observables

                # new code - without assuming observables are at the end of the circuit:
                if d not in self.observables_indices: # this col is a detector
                    error_targets.append(stim.target_relative_detector_id(d))
                else: # this col is an observable
                    observable_index = self.observables_indices.index(d)  # Finding the index of observable 'd' in the self.observables_indices list
                    error_targets.append(stim.target_logical_observable_id(observable_index))
                    
            # Append error with probability to final_dem
            if self.printing:
                print(f"Error targets = {error_targets}, Probability = {probability}")
            final_dem.append("error", probability, error_targets)
            
        return final_dem
        
        
        
    def generate_circuit_without_lost_qubit(self, lost_qubits, circuit, circuit_offset = 0, loss_qubits_remove_gates_range = {}, removing_Pauli_errors=False):
        # loss_qubits_remove_gates_range is dictionary where for each lost qubit we get a list of ranges of instruction indices in which we should remove the gates of this qubit.
        new_circuit = stim.Circuit()
        for ix, instruction in enumerate(circuit):
            instruction_ix = ix + circuit_offset
            if removing_Pauli_errors and instruction.name in ['PAULI_CHANNEL_1', 'PAULI_CHANNEL_2', 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR']:
                continue # don't put Pauli errors in the experimental loss circuit
            
            targets = [q.value for q in instruction.targets_copy()]
            if set(lost_qubits).intersection(set(targets)):
                if instruction.name in ['CZ', 'CX', 'SWAP']: # pairs of qubits
                    pairs = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
                    for (c,t) in pairs:
                        if (c in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[c])) or (t in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[t])):
                            if self.printing :
                                a=1
                                # print(f"Removing this gate from the heralded circuit: {instruction.name} {(c,t)}, because my lost qubits = {lost_qubits}")
                        else:
                            new_circuit.append(instruction.name, [c,t])

                elif instruction.name in ['H', 'R', 'RX', 'I']:
                    for q in targets:
                        if (q in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[q])):
                            if self.printing :
                                a=1
                                # print(f"Removing this gate from the heralded circuit: {instruction.name}, because my lost qubits = {lost_qubits}")
                        else:
                            new_circuit.append(instruction.name, [q])

                            
                elif instruction.name in ['MRX', 'MR']:
                    for q in targets:
                        if (q in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[q])):
                            # Heralded circuit - lost ancilla qubits give probabilistic 50/50 measurement:
                            if instruction.name == 'MR':
                                new_circuit.append('RX', [q])
                                new_circuit.append('MR', [q])
                            elif instruction.name == 'MRX':
                                new_circuit.append('R', [q])
                                new_circuit.append('MRX', [q])
                        else:
                            new_circuit.append(instruction.name, [q])
                
                elif instruction.name in ['MX', 'M']:
                    for q in targets:
                        if (q in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[q])):
                            # Heralded circuit - lost ancilla qubits give probablistic 50/50 measurement:
                            if instruction.name == 'M':
                                new_circuit.append('RX', [q])
                                new_circuit.append('M', [q])
                            elif instruction.name == 'MX':
                                new_circuit.append('R', [q])
                                new_circuit.append('MX', [q])
                        else:
                            new_circuit.append(instruction.name, [q])
                            
                
                else:
                    new_circuit.append(instruction)
            else:
                new_circuit.append(instruction)
            
        return new_circuit
    
    def add_to_current_DEM(self, current_DEM, new_DEM_to_add):
        # Ensure both matrices are dok_matrix for simple item assignment
        if not isinstance(current_DEM, dok_matrix):
            current_DEM = current_DEM.todok()
        if not isinstance(new_DEM_to_add, dok_matrix):
            new_DEM_to_add = new_DEM_to_add.todok()

        # Directly add new DEM values to the current DEM
        for key, value in new_DEM_to_add.items():
            current_DEM[key] = current_DEM.get(key, 0) + value

        return current_DEM
    
    from scipy.sparse import vstack, csr_matrix

    def combine_DEMs_sum(self, DEMs_list, num_detectors):
        # Stack all DEMs vertically
        all_dems_stacked = vstack(DEMs_list, format='csr')

        # Sum duplicates after vertical stacking
        all_dems_stacked.sum_duplicates()

        # Group by the pattern of non-zero columns
        pattern_to_values = {}
        for i in range(all_dems_stacked.shape[0]):
            # Extract the row slice
            row_data = all_dems_stacked.data[all_dems_stacked.indptr[i]:all_dems_stacked.indptr[i+1]][0]
            row_indices = all_dems_stacked.indices[all_dems_stacked.indptr[i]:all_dems_stacked.indptr[i+1]]
            # Create a hashable representation of the row
            row_pattern = tuple(row_indices)
            # Sum the values of rows that match this pattern
            if row_pattern in pattern_to_values:
                pattern_to_values[row_pattern] += row_data
            else:
                pattern_to_values[row_pattern] = row_data
            
        # Finally, we can build the final matrix
        final_matrix = lil_matrix((0, num_detectors), dtype=float)
        for pattern, value in pattern_to_values.items():
            new_row = np.zeros(num_detectors, dtype=float)
            new_row[list(pattern)] = value
            final_matrix.resize((final_matrix.shape[0] + 1, num_detectors))
            final_matrix[-1, :] = new_row
            
        return final_matrix


    def convert_dem_into_hyperedges_matrix(self, dem, event_probability, observables_converted_to_detectors=False):
        # Output hyperedge matrix have num col = num detectors + num observables
        # If observables_converted_to_detectors = True: DEM where observables are already converted detectors
        # Note that observables_errors_interactions will be non trivial only if observables_converted_to_detectors = False
        
        num_detectors = dem.num_detectors # includes num_observables because 
        num_observables = len(self.observables_indices)
        num_total = num_detectors if observables_converted_to_detectors else num_detectors + num_observables

        num_errors = sum(1 for error in dem if str(error)[:5] == 'error' and error.args_copy()[0] != 0)
        hyperedges_matrix = lil_matrix((num_errors, num_total), dtype=float)

        observables_errors_interactions = [[] for _ in range(num_observables)]

        error_index = 0
        for error in dem:
            if str(error)[:5] == 'error' and error.args_copy()[0] != 0:
                probability = error.args_copy()[0]
                targets = []
                for target in error.targets_copy():
                    if stim.DemTarget.is_relative_detector_id(target):
                        targets.append(target.val)
                    elif stim.DemTarget.is_logical_observable_id(target):
                        observable_index = target.val
                        observables_errors_interactions[observable_index].append(error_index)
                targets = np.asarray(targets)
                hyperedges_matrix[error_index, targets] = probability * event_probability
                error_index += 1
        if observables_converted_to_detectors:
            return hyperedges_matrix
        
        else:
            return hyperedges_matrix, observables_errors_interactions


    def combine_DEMs_high_order(self, DEMs_list, num_detectors):
        # Assuming 'DEMs_list' is a list of lil_matrix objects, each representing a DEM for a loss event and one DEM for Pauli events.
        # We will first convert each sparse matrix row to a hashable tuple of (row_index, column_indices, value)

        # This function will convert the rows of a lil_matrix into a hashable format
        def convert_rows(matrix):
            hashable_rows = {}
            for ridx, (row, data) in enumerate(zip(matrix.rows, matrix.data)):
                if data:  # Skip empty rows
                    value = data[0]  # Assuming all values in a row are the same
                    pattern = tuple(sorted(row))
                    hashable_rows[pattern] = (ridx, value)
            return hashable_rows

        # Now we can convert all matrices and merge rows with the same pattern
        pattern_to_values = {}
        for matrix in DEMs_list:
            for pattern, (ridx, value) in convert_rows(matrix).items():
                if pattern in pattern_to_values:
                    pattern_to_values[pattern].append(value)
                else:
                    pattern_to_values[pattern] = [value]

        # Now we can apply the formula of high-order probability sum to combine the values for each pattern
        final_rows = {}
        for pattern, values in pattern_to_values.items():
            # Apply the formula
            prob_i_terms = [v * np.prod([1 - x for x in values]) / (1-v) for v in values]
            if len(values) > 3:
                # Consider terms where 3 specific events happen
                for i, v1 in enumerate(values):
                    for j, v2 in enumerate(values[i+1:], start=i+1):
                        for k, v3 in enumerate(values[j+1:], start=j+1):
                            prob_i_terms.append(v1 * v2 * v3 * np.prod([1 - x for n, x in enumerate(values) if n not in (i, j, k)]))
            final_rows[pattern] = sum(prob_i_terms)

        # Finally, we can build the final matrix
        final_matrix = lil_matrix((0, num_detectors), dtype=float)
        for pattern, value in final_rows.items():
            new_row = np.zeros(num_detectors, dtype=float)
            new_row[list(pattern)] = value
            final_matrix.resize((final_matrix.shape[0] + 1, num_detectors))
            final_matrix[-1, :] = new_row

        # 'final_matrix' is now the final lil_matrix containing the summed probabilities
        return final_matrix


    def observables_to_detectors(self, circuit: stim.Circuit) -> stim.Circuit:
        result = stim.Circuit()
        self.observables_indices = [] # to keep record on observables indices
        index = 0
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                result.append(stim.CircuitRepeatBlock(
                    repeat_count=instruction.repeat_count,
                    body=self.observables_to_detectors(instruction.body_copy())))
            if instruction.name == 'DETECTOR':
                result.append(instruction)
                index += 1
            elif instruction.name == 'OBSERVABLE_INCLUDE':
                targets = instruction.targets_copy()
                result.append('DETECTOR', targets) # replace with a detector
                self.observables_indices.append(index) # keep track of observable index
                index += 1
            else:
                result.append(instruction)
        return result


    def set_up_Pauli_DEM(self):
        # this function takes the circuit and produces a DEM, considering only Pauli errors on no losses (which are I gates anyway).
        # before generating the DEM, we need to convert the observables into detectors.
        # After generating the DEM, we need to convert it into a sparse matrix.
        
        
        circuit_for_Pauli_dem = self.observables_to_detectors(self.circuit.copy())
                        
        Pauli_DEM = circuit_for_Pauli_dem.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
        
        # Convert the DEM into a matrix:
        hyperedges_matrix_Pauli_DEM = self.convert_dem_into_hyperedges_matrix(Pauli_DEM, event_probability=1, observables_converted_to_detectors=True)

        self.Pauli_DEM = Pauli_DEM
        self.Pauli_DEM_matrix = hyperedges_matrix_Pauli_DEM


        
    def get_qubits_lifecycles_MBQC(self):
        # Iterate through circuit. Record the lifecycles of each qubit, from initialization to measurement.
        # round_ix = -1
        self.qubit_lifecycles_and_losses = {int(i): [] for i in self.ancilla_qubits + self.data_qubits}
        qubit_active_cycle = {i: None for i in self.ancilla_qubits + self.data_qubits}
        
        for round_ix in self.rounds_by_ix:
            for instruction in self.rounds_by_ix[round_ix]:
                # Check when each qubit is init and measured:
                if instruction.name in ['R', 'RX']: # Beginning of a cycle for these qubits
                    qubits = set([q.value for q in instruction.targets_copy()])
                    for q in qubits:
                        self.qubit_lifecycles_and_losses[q].append([round_ix, None, None]) # Begin a new cycle for each qubit
                        qubit_active_cycle[q] = len(self.qubit_lifecycles_and_losses[q]) - 1

                if instruction.name in ['M', 'MX']: # End of a cycle for these qubits
                    qubits = set([q.value for q in instruction.targets_copy()])
                    for q in qubits:
                        self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][2] = None
                        self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix # Close the active cycle with the measurement round
                        qubit_active_cycle[q] = None


        # Handle unmeasured qubits at the end of the circuit
        for q in qubit_active_cycle:
            if qubit_active_cycle[q] is not None:
                self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix



    def get_qubits_lifecycles_FREE(self):
        # Build a dictionary of lifecycles (self.qubit_lifecycles_and_losses) to record the init to loss detection of each qubit. If we lose a qubit during a lifecycle, all loss places during this lifecycle are potential loss locations.
        # Here we assume a FREE loss detection of data qubits (and ancilla qubits are measured every round so they also have free loss detection).
        # Ancilla qubits have lifecycles of 1 round only.
        # Data qubits have lifecycles according to the loss detection period.
        # self.qubit_lifecycles_and_losses = {qubit: {[reset_round, loss_meas_round, was lost during this lifecycle], [], [], ..}}

        self.qubit_lifecycles_and_losses = {int(i): [] for i in self.ancilla_qubits + self.data_qubits}
        data_qubits_active_cycle_index = -1
        
        open_new_lifecycles = True
        for round_ix in self.rounds_by_ix:
            
            # ancilla qubits lifecycles:
            for ancilla_q in self.ancilla_qubits: # open and close lifecycle in every round:
                self.qubit_lifecycles_and_losses[ancilla_q].append([round_ix, round_ix, None]) # begin and close the lifecycle of ancilla qubits every round
            
            # data qubits lifecycles:
            if open_new_lifecycles: # close lifecycles:
                for data_q in self.data_qubits:
                    self.qubit_lifecycles_and_losses[data_q].append([round_ix, None, None])
                    # self.qubit_lifecycles_and_losses[data_q][data_qubits_active_cycle_index][0] = round_ix # Open a cycle
                open_new_lifecycles = False
                data_qubits_active_cycle_index += 1
                
            if (round_ix+1)%self.loss_detection_freq == 0 or round_ix == max(self.rounds_by_ix.keys()): # close lifecycles. data loss detection round:
                for data_q in self.data_qubits:
                    self.qubit_lifecycles_and_losses[data_q][data_qubits_active_cycle_index][1] = round_ix # Close the active cycle with the measurement round
                    open_new_lifecycles = True
                    
        # print(f"self.qubit_lifecycles_and_losses = {self.qubit_lifecycles_and_losses}")
        # print(f"frequency = {self.loss_detection_freq}")
        
    def get_qubits_lifecycles_SWAP(self):
        # Iterate through circuit. Record the lifecycles of each qubit, from initialization to measurement.
        loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
        round_ix = -1
        inside_qec_round = False
        SWAP_round_index = 0
        SWAP_round_type = None
        self.qubit_lifecycles_and_losses = {int(i): [] for i in self.ancilla_qubits + self.data_qubits}
        qubit_active_cycle = {i: None for i in self.ancilla_qubits + self.data_qubits}
        first_QEC_round = True
        
        for instruction_ix, instruction in enumerate(self.circuit):
            # Check when each qubit is init and measured:
            if instruction.name in ['R', 'RX']: # Beginning of a cycle for these qubits
                qubits = set([q.value for q in instruction.targets_copy()])
                for q in qubits:
                    self.qubit_lifecycles_and_losses[q].append([round_ix, None, None]) # Begin a new cycle for each qubit
                    qubit_active_cycle[q] = len(self.qubit_lifecycles_and_losses[q]) - 1

            if instruction.name in ['M', 'MX']: # End of a cycle for these qubits
                qubits = set([q.value for q in instruction.targets_copy()])
                for q in qubits:
                    self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][2] = None
                    self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix # Close the active cycle with the measurement round
                    qubit_active_cycle[q] = None

                        
            # QEC rounds:
            if instruction.name == 'TICK':
                if not inside_qec_round: # beginning of QEC round
                    if first_QEC_round:
                        round_ix += 1; first_QEC_round = False
                    if self.printing:
                        print(f"Starting QEC Round {round_ix}")
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
            

        # Handle unmeasured qubits at the end of the circuit
        for q in qubit_active_cycle:
            if qubit_active_cycle[q] is not None:
                self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix

