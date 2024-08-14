import stim
import numpy as np
from scipy.sparse import lil_matrix
from itertools import product
from scipy.sparse import dok_matrix, csc_matrix, csr_matrix, coo_matrix, vstack
import os
import json
from hashlib import sha256
import pickle
import time
import copy
import itertools
from collections import defaultdict

class MLE_Loss_Decoder:
    def __init__(self, Meta_params:dict, bloch_point_params: dict, cycles: int, dx:int, dy:int, ancilla_qubits:list, data_qubits:list, 
                loss_detection_freq=None, printing=False, output_dir=None, first_comb_weight=0.5, loss_detection_method_str='SWAP', 
                 save_data_during_sim=False, n_r=1, circuit_type = '', use_independent_decoder=True, use_independent_and_first_comb_decoder=True, **kwargs) -> None:
        self.Meta_params = Meta_params
        self.bloch_point_params = {'erasure_ratio': '1', 'bias_ratio': '0.5'}
        self.bloch_point_params = bloch_point_params
        self.cycles = cycles
        self.dx = dx
        self.dy = dy
        self.printing = printing
        self.loss_detection_freq = loss_detection_freq
        self.ancilla_qubits = ancilla_qubits
        self.data_qubits = data_qubits
        self.circuit_type = circuit_type
        self.n_r = n_r
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
        if self.circuit_type == 'random_alg':
            self.loss_decoder_files_dir = f"{output_dir}/loss_circuits/{self.create_loss_file_name(self.Meta_params, self.bloch_point_params)}/dx_{dx}__dy_{dy}__c_{cycles}__nr{n_r}"
        else:
            self.loss_decoder_files_dir = f"{output_dir}/loss_circuits/{self.create_loss_file_name(self.Meta_params, self.bloch_point_params)}/dx_{dx}__dy_{dy}__c_{cycles}"
        
        print(self.loss_decoder_files_dir)
        self.measurement_map = {}
        self.measurement_ix_to_ins_ix = {}
        self.decoder_type = Meta_params['loss_decoder']
        self.loss_detection_method_str = loss_detection_method_str
        self.losses_to_detectors = []
        self.first_comb_weight = first_comb_weight
        self.save_data_during_sim = save_data_during_sim
        self.use_independent_decoder = use_independent_decoder
        self.use_independent_and_first_comb_decoder = use_independent_and_first_comb_decoder
        
    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        self._circuit = value
        # Update self.extra_num_detectors whenever self.circuit is redefined
        # TODO: remove that, its saved in stim's dem
        # observable_instructions_ix = [i for i, instruction in enumerate(self._circuit) if instruction.name == 'OBSERVABLE_INCLUDE']
        # self.extra_num_detectors = len(observable_instructions_ix)
        

    def _generate_unique_key(self, losses_by_instruction_ix):
        sorted_losses_by_instruction_ix = {k: sorted(v) for k, v in sorted(losses_by_instruction_ix.items())}

        # Convert the data to a JSON string
        data_str = json.dumps({
            'losses_by_instruction_ix': sorted_losses_by_instruction_ix
            # 'meta_params': self.Meta_params,
            # 'dx': self.dx,
            # 'dy': self.dy,
        }, sort_keys=True)
        # Use SHA256 to generate a unique hash of the data
        return sha256(data_str.encode()).hexdigest()
    
    def create_loss_file_name(self, Meta_params, bloch_point_params={}):
        if type(Meta_params['ordering']) is list:
            ordering_string = "_".join(Meta_params['ordering'])

            ### Need to use shorter notation for the ordering string otherwise filename too long
            orderings_list = ['fowler','Ztopleft','bad','Zbottomleft','Ztopright','Ntopleft','Nbottomleft','Ntopright']
            replacement_list = ['f', 'Z', 'Z','Zv','Zh','N','Nv','Nh']
            for ordering,replacement in zip(orderings_list,replacement_list):
                ordering_string = ordering_string.replace(ordering,replacement)
            return f"{Meta_params['architecture']}__{Meta_params['code']}__{Meta_params['circuit_type']}__{Meta_params['num_logicals']}log__{Meta_params['logical_basis']}__{int(Meta_params['bias_preserving_gates'] == 'True')}__{Meta_params['noise']}__{int(Meta_params['is_erasure_biased']=='True')}__LD_freq_{Meta_params['LD_freq']}__SSR_{int(Meta_params['SSR']=='True')}__LD_method_{Meta_params['LD_method']}__ordering_{ordering_string}"
        else:
            return f"{Meta_params['architecture']}__{Meta_params['code']}__{Meta_params['circuit_type']}__{Meta_params['num_logicals']}log__{Meta_params['logical_basis']}__{int(Meta_params['bias_preserving_gates'] == 'True')}__{Meta_params['noise']}__{int(Meta_params['is_erasure_biased']=='True')}__LD_freq_{Meta_params['LD_freq']}__SSR_{int(Meta_params['SSR']=='True')}__LD_method_{Meta_params['LD_method']}__ordering_{Meta_params['ordering']}"


    def generate_loss_instruction_indices(self):
        self.circuit.loss_instruction_indices = {}
        
        loss_index = 0
        for idx, instruction in enumerate(self.circuit):
            if instruction.name == 'I':
                self.circuit.loss_instruction_indices[idx] = self.circuit.loss_probabilities[loss_index]
                loss_index += len(instruction.targets_copy())

    #################################################### Decoder initialization functions ####################################################

    # def preprocess_all_SSR_loss_circuits(self, **kargs):
        # we want to look at a circuit, take all options of loss upon measurement, and for each options generate the stim circuit for this option.
        # save all into a dictionary of stim dems
        
        
        # self.set_up_Pauli_DEM()
        # self.rounds_by_ix = self.split_stim_circuit_into_rounds()
        # self.generate_loss_instruction_indices() # setup self.circuit.loss_instruction_indices
        # self.generate_measurement_map() # fill out self.measurement_map {measurement_index: (qubit, round_index)}
        # fill out self.qubit_lifecycles_and_losses (without the losses for now):
        
        # if self.loss_detection_method_str == 'MBQC':
        #     self.get_qubits_lifecycles_MBQC()
        # elif self.loss_detection_method_str == 'SWAP':
        #     self.get_qubits_lifecycles_SWAP() 
        # elif self.loss_detection_method_str == 'FREE':
        #     self.get_qubits_lifecycles_FREE()
            
        # # self.qubit_lifecycles_and_losses_init = copy.deepcopy(self.qubit_lifecycles_and_losses)
        
        # if self.printing:
        #     print(f"Using {self.loss_detection_method_str} method for {self.cycles} cycles and dx = {self.dx}, dy = {self.dy}, self.qubit_lifecycles_and_losses = {self.qubit_lifecycles_and_losses}")
        
        
        # if self.decoder_type ==  'only_ssr': # takes into account SSR information only:
        #     full_filename_dems = f'{self.loss_decoder_files_dir}/dems_SSR_all_options.pickle'
        #     if not os.path.exists(full_filename_dems): # If needed - preprocess this circuit to get all relevant DEMs
        #         if self.printing:
        #             print("Loss circuits need to be generated for these parameters. Starting pre-processing!")
                
        #         start_time = time.time()
        #         self.preprocess_circuit_only_SSR(full_filename = full_filename_dems)
        #         print(f'Preprocessing is done! it took {time.time() - start_time:.2f}s')      
        #     else:
        #         try:
        #             with open(full_filename_dems, 'rb') as file:
        #                 self.dems_stim_only_SSR, _ = pickle.load(file) # Load the data from the file
        #         except EOFError as e:
        #             print(f"EOFError: {e}. The file {full_filename_dems} might be corrupted. Regenerating the file.")
        #             self.preprocess_circuit_only_SSR(full_filename=full_filename_dems)

    def initialize_loss_decoder_for_sampling_only(self):

        self.rounds_by_ix = self.split_stim_circuit_into_rounds()
        self.generate_loss_instruction_indices() # setup self.circuit.loss_instruction_indices
        self.generate_measurement_map() # fill out self.measurement_map {measurement_index: (qubit, round_index)}
        
        # fill out self.qubit_lifecycles_and_losses (without the losses for now):
        if self.loss_detection_method_str == 'None':
            self.get_qubits_lifecycles_None()
        if self.loss_detection_method_str == 'MBQC':
            self.get_qubits_lifecycles_MBQC()
        elif self.loss_detection_method_str == 'SWAP':
            self.get_qubits_lifecycles_SWAP() 
        elif self.loss_detection_method_str == 'FREE':
            self.get_qubits_lifecycles_FREE()
            
        self.qubit_lifecycles_and_losses_init = copy.deepcopy(self.qubit_lifecycles_and_losses)
        
        if self.printing:
            print(f"Using {self.loss_detection_method_str} method for {self.cycles} cycles and dx = {self.dx}, dy = {self.dy}, self.qubit_lifecycles_and_losses = {self.qubit_lifecycles_and_losses}")
        
        
                    
    def initialize_loss_decoder(self, **kargs):

        self.set_up_Pauli_DEM()
        
        if self.use_independent_decoder: # we want to decode with more than superchecks.
            self.rounds_by_ix = self.split_stim_circuit_into_rounds()
            self.generate_loss_instruction_indices() # setup self.circuit.loss_instruction_indices
            self.generate_measurement_map() # fill out self.measurement_map {measurement_index: (qubit, round_index)}
            # fill out self.qubit_lifecycles_and_losses (without the losses for now):
            
            if self.loss_detection_method_str == 'None':
                self.get_qubits_lifecycles_None()
            if self.loss_detection_method_str == 'MBQC':
                self.get_qubits_lifecycles_MBQC()
            elif self.loss_detection_method_str == 'SWAP':
                self.get_qubits_lifecycles_SWAP() 
            elif self.loss_detection_method_str == 'FREE':
                self.get_qubits_lifecycles_FREE()
                
            self.qubit_lifecycles_and_losses_init = copy.deepcopy(self.qubit_lifecycles_and_losses)
            
            if self.printing:
                print(f"Using {self.loss_detection_method_str} method for {self.cycles} cycles and dx = {self.dx}, dy = {self.dy}, self.qubit_lifecycles_and_losses = {self.qubit_lifecycles_and_losses}")
            
        
            if len(self.decoder_type) >= 11 and self.decoder_type[:11] ==  'independent': # Independent decoder:
                full_filename_dems = f'{self.loss_decoder_files_dir}/circuit_dems_1_losses.pickle'
                if not os.path.exists(full_filename_dems): # If needed - preprocess this circuit to get all relevant DEMs
                    if self.printing:
                        print("Loss circuits need to be generated for these parameters. Starting pre-processing!")
                    
                    start_time = time.time()
                    self.preprocess_circuit(full_filename = full_filename_dems)
                    print(f'Preprocessing is done! it took {time.time() - start_time:.2f}s')      
                else:
                    try:
                        with open(full_filename_dems, 'rb') as file:
                            self.circuit_independent_dems, _ = pickle.load(file) # Load the data from the file
                    except EOFError as e:
                        print(f"EOFError: {e}. The file {full_filename_dems} might be corrupted. Regenerating the file.")
                        self.preprocess_circuit(full_filename=full_filename_dems)

            elif self.decoder_type == 'comb':
                self.circuit_comb_dems = {}
                for num_of_losses in [1,2]: # number of losses in the combination # TODO: maybe change back to [1,2,3,4,5,6,7]
                    full_filename_dems = f'{self.loss_decoder_files_dir}/circuit_dems_{num_of_losses}_losses.pickle'
                    
                    # pre-process all combinations before:
                    if not self.save_data_during_sim:
                        if not os.path.exists(full_filename_dems):  # If needed - preprocess this circuit to get all relevant DEMs
                            if self.printing:
                                print(f"Loss circuits need to be generated for these parameters ({num_of_losses} losses). Starting pre-processing!")
                            
                            # Preprocess in batches:
                            all_potential_loss_qubits_indices = self.get_all_potential_loss_qubits()
                            self.preprocess_circuit_comb_batches(full_filename=full_filename_dems, num_of_losses=num_of_losses, all_potential_loss_qubits_indices=all_potential_loss_qubits_indices)
                            # self.preprocess_circuit_comb(full_filename=full_filename_dems, num_of_losses=num_of_losses, all_potential_loss_qubits_indices=all_potential_loss_qubits_indices)
                        else:
                            try:
                                with open(full_filename_dems, 'rb') as file:
                                    circuit_comb_dems, _ = pickle.load(file)  # Load the data from the file
                                    self.circuit_comb_dems.update(circuit_comb_dems)  # Merge
                            except (EOFError, FileNotFoundError) as e:
                                print(f"Error: {e}. The file {full_filename_dems} might be corrupted or missing. Regenerating the file.")
                                all_potential_loss_qubits_indices = self.get_all_potential_loss_qubits()
                                self.preprocess_circuit_comb_batches(full_filename=full_filename_dems, num_of_losses=num_of_losses, all_potential_loss_qubits_indices=all_potential_loss_qubits_indices)
                                # self.preprocess_circuit_comb(full_filename=full_filename_dems, num_of_losses=num_of_losses, all_potential_loss_qubits_indices=all_potential_loss_qubits_indices)
                                    
                    
                    else:  # No pre-processing. Generating DEMs as we go.
                        try:
                            with open(full_filename_dems, 'rb') as file:
                                circuit_comb_dems, _ = pickle.load(file)  # Load the data from the file
                                self.circuit_comb_dems.update(circuit_comb_dems)  # Merge
                        except (EOFError, FileNotFoundError) as e:
                            print(f"Error: {e}. The file {full_filename_dems} might be corrupted or missing. Creating an empty file.")
                            with open(full_filename_dems, 'wb') as file:
                                pickle.dump(({}, self.Meta_params), file)
        
    #################################################### Main decoding functions ####################################################
    
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
            experimental_circuit = self.generate_loss_circuit(losses_by_instruction_ix = self.real_losses_by_instruction_ix, removing_Pauli_errors=False, remove_gates_due_to_loss=True)
        
            # Step 2 - get all possible loss locations (and save in self.potential_losses_by_instruction_index[(lost_q, round_ix)])
            # self.get_all_potential_loss_locations_given_heralded_loss()
            self.get_all_potential_loss_locations_given_heralded_loss_new()

            
            # Step 3 - choose a decoder type:
            if len(self.decoder_type) >= 11 and self.decoder_type[:11] ==  'independent': # Independent decoder:
                final_dem = self.generate_all_DEMs_and_sum_over_independent_events(return_hyperedges_matrix=False)
            else:
                # All combination decoder:
                self.all_potential_losses_combinations, self.combinations_events_probabilities = self.generate_all_potential_losses_combinations(potential_losses_by_instruction_index = self.potential_losses_by_instruction_index)
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
    
    
    
    def generate_dem_loss_mle_experiment(self, measurement_event, return_matrix_with_observables = False):
        """ This function takes the original circuit with places for potential losses and loss detection events, and generates 2 circuits: 
        1. experimental measurement circuit. 2. Theory decoding circuit. 
        if return_matrix_with_observables=True, we return a dem with the observables inside.
        
        """
        
        # updated before for all shots together: self.qubit_lifecycles_and_losses, self.rounds_by_ix, self.measurement_map
        
        shot_had_a_loss = 2 in measurement_event
        if shot_had_a_loss:
            
            # Step 1 - find out which qubit was lost in which round:
            start_time = time.time()
            self.qubit_lifecycles_and_losses = copy.deepcopy(self.qubit_lifecycles_and_losses_init)#.copy() # init self.qubit_lifecycles_and_losses for this shot
            self.update_lifecycle_from_detections(detection_event=measurement_event) # update self.qubit_lifecycles_and_losses according to measurement_events
            if self.printing:
                print(f"lifecycles of qubits: {self.qubit_lifecycles_and_losses}\n")

            # print(f'Updating lifecycles according to SSR information took {time.time() - start_time:.5f}s')      
            
            
            # Step 2 - get all possible loss locations (and save in self.potential_losses_by_instruction_index[(lost_q, round_ix)])
            start_time = time.time()
            self.get_all_potential_loss_locations_given_heralded_loss_new()
            # print(f'Getting all potential loss locations according to SSR information took {time.time() - start_time:.5f}s')      

            # Step 3 - choose a decoder type:
            if len(self.decoder_type) >= 11 and self.decoder_type[:11] ==  'independent': # Independent decoder:
                start_time = time.time()
                final_dem_hyperedges_matrix = self.generate_all_DEMs_and_sum_over_independent_events(return_hyperedges_matrix=True)
                # print(f'Summing over all relevant DEMs to generate the final DEM took {time.time() - start_time:.5f}s')      
            
            else:
                # All combination decoder:
                self.all_potential_losses_combinations, self.combinations_events_probabilities = self.generate_all_potential_losses_combinations(potential_losses_by_instruction_index = self.potential_losses_by_instruction_index)
                final_dem_hyperedges_matrix = self.generate_all_DEMs_and_sum_over_combinations(return_hyperedges_matrix=True)
                if self.printing:
                    print("Now lets see all loss pattern and which detectors were affected:")
                    for element in self.losses_to_detectors:
                        print(element)
            

        else: # no losses in this shot, bring back regular DEM
            final_dem_hyperedges_matrix = self.Pauli_DEM_matrix # here observables are represented as detectors
            
        if not return_matrix_with_observables:
            # Final step - remove observables from hyperedgesmatrix and create a list of lists of errors that affect observables
            # start_time = time.time()
            final_dem_hyperedges_matrix, observables_errors_interactions= self.convert_detectors_back_to_observables(final_dem_hyperedges_matrix)
            # print(f'Convert detectors back to observables and create list observables_errors_interactions took {time.time() - start_time:.5f}s')      
                
            return final_dem_hyperedges_matrix, observables_errors_interactions

        else: 
            return final_dem_hyperedges_matrix, None
        
    
    
    def generate_dem_loss_mle_experiment_only_superchecks(self, measurement_event, return_matrix_with_observables):
        """ This function takes the original circuit with places for potential losses and loss detection events, and generates 2 circuits: 1. experimental measurement circuit. 2. Theory decoding circuit. 
        return_matrix_with_observables = True """
        
        shot_had_a_loss = 2 in measurement_event
        
        if shot_had_a_loss:
            
            # step 1 - generate the circuit with RX before heralded loss measurement:
            # start_time = time.time()
            experimental_circuit, _ = self.add_RX_before_heralded_loss(measurement_event)
            # print(f'Adding RX to the circuit according to loss took {time.time() - start_time:.6f}s.')      
            
            # step 2 - replace final observables with detectors:
            final_loss_circuit = self.observables_to_detectors(experimental_circuit)

            # get the dem (with observables on columns):
            dem_heralded_circuit = final_loss_circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=True)
            
            # convert the DEM into a matrix:
            final_dem_hyperedges_matrix = self.convert_dem_into_hyperedges_matrix(dem_heralded_circuit, observables_converted_to_detectors=True)
            
            
                
        else: # no losses in this shot
            final_dem_hyperedges_matrix = self.Pauli_DEM_matrix
        
        if not return_matrix_with_observables:
            # Final step - remove observabes from hyperedgesmatrix and create a list of lists of errors that affect observables
            start_time = time.time()
            final_dem_hyperedges_matrix, observables_errors_interactions= self.convert_detectors_back_to_observables(final_dem_hyperedges_matrix)
            # print(f'Convert detectors back to observables and create list observables_errors_interactions took {time.time() - start_time:.5f}s')      
            return final_dem_hyperedges_matrix, observables_errors_interactions
        
        else:
            return return_matrix_with_observables, None




    def make_stim_dem_supercheck_given_loss_only_determ_observables(self, measurement_event):
        """ This function takes the original circuit with places for potential losses and loss detection events, and generates 2 circuits: 1. experimental measurement circuit. 2. Theory decoding circuit. """
        
        shot_had_a_loss = 2 in measurement_event
        
        if shot_had_a_loss:
            
            
            # step 1 - generate the circuit with RX before heralded loss measurement:
            # start_time = time.time()
            experimental_circuit, _ = self.add_RX_before_heralded_loss(measurement_event)
            # print(f'Adding RX to the circuit according to loss took {time.time() - start_time:.6f}s.')      
            
            # step 2 - get DEM only for shots with deterministic observables:
            try:
                # start_time = time.time()
                final_dem = experimental_circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=True)      
                # print(f'generating the DEM for the valid shots (no loss right before observable qubits measurements) {time.time() - start_time:.6f}s.')        
                dont_use_shot = False
            except:
                final_dem = 0
                dont_use_shot = True
            # TODO: important: fill-out a dictionary of DEMs given this heralded measurement. if it exists --> just use it. self.stim_dems_given_heralded_loss

            return final_dem, dont_use_shot
                
        else: # no losses in this shot
            experimental_circuit = self.circuit.copy()
            final_dem = experimental_circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)        
            dont_use_shot = False
        
        
        return final_dem, dont_use_shot


    #################################################### Helper functions ####################################################

    def add_RX_before_heralded_loss(self, measurement_event):
        # Identify the locations where instructions need to be inserted
        loss_measurements = [i for i , meas in enumerate(measurement_event) if meas==2] # checking which measurement indices had a hit for loss

        # Convert the circuit string to a list of lines
        circuit_lines = str(self.circuit).strip().split('\n')

        # Prepare a list of (index, new instruction) tuples
        insertions = [
            (self.measurement_ix_to_ins_ix[loss_ins][0], f'RX {self.measurement_ix_to_ins_ix[loss_ins][1]}')
            for loss_ins in loss_measurements
        ]

        # Sort insertions by index to maintain order
        insertions.sort(key=lambda x: x[0])

        # Prepare the result list
        result = []
        insertion_offset = 0

        circuit_length = len(circuit_lines)
        insertions_length = len(insertions)
        total_length = circuit_length + insertions_length - 1  # Last index in the combined length

        for i in range(circuit_length + insertions_length):
            # if i == total_length:
            #     # make observable into detector
            #     instruction = circuit_lines[-1]
            #     obs_targets = ' '.join([str(t) for t in instruction.targets_copy()])
            #     result.append(f'DETECTOR {obs_targets}')

            if insertion_offset < len(insertions) and i == insertions[insertion_offset][0] + insertion_offset:
                result.append(insertions[insertion_offset][1])
                insertion_offset += 1
            else:
                result.append(circuit_lines[i - insertion_offset])

        # Join the lines back into a single string
        updated_circuit_str = '\n'.join(result)
        heralded_circuit = stim.Circuit(updated_circuit_str)
        
        
        obs_targets = 0
        return heralded_circuit, obs_targets



            

    
        
        
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
        
        
    def generate_measurement_ix_to_instruction_ix_map(self):
        # Build a mapping from measurement indices to qubits and measurement rounds. 
        # self.measurement_map[measurement_index] = (instruction_ix, qubit)
        self.measurement_ix_to_ins_ix = {}
        measurement_index = 0
        # Assume self.rounds_by_ix has the form {round_index: [instruction_list], ...}
        for instruction_ix, instruction in enumerate(self.circuit):
            if instruction.name in ['M', 'MX']:  # Check if it's a measurement instruction
                targets = instruction.targets_copy()
                for qubit in targets:
                    self.measurement_ix_to_ins_ix[measurement_index] = (instruction_ix, qubit.value)
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
    
    # def convert_qubit_losses_into_measurement_events(self, loss_detection_events_all_shots: np.array, measurement_events_all_shots: np.array):
    #     num_shots = loss_detection_events_all_shots.shape[0]
    #     num_measurements = measurement_events_all_shots.shape[1]

    #     # Initialize the lost qubits status for all shots
    #     lost_qubits = np.zeros((num_shots, len(self.ancilla_qubits + self.data_qubits)), dtype=bool)

    #     # Keep track of measurement indices for updating measurement events
    #     loss_idx = 0
    #     measurement_idx = 0

    #     # Iterate through the circuit instructions once, for all shots
    #     for instruction in self.circuit:
    #         if instruction.name == 'I':
    #             # This instruction indicates potential loss locations --> update lost_qubits
    #             for target in instruction.targets_copy(): # going over each qubit and checking if it is lost here
    #                 loss_events = loss_detection_events_all_shots[:, loss_idx] # check in all shots together
    #                 lost_qubits[:, target.value] = np.logical_or(lost_qubits[:, target.value], loss_events) # update all shots together
    #                 loss_idx += 1

    #         elif instruction.name in ['R','RX']:
    #             # Re-initialization means qubit is not lost anymore --> update lost_qubits
    #             for target in instruction.targets_copy():
    #                 lost_qubits[:, target.value] = False

    #         elif instruction.name in ['M','MX']:
    #             # This instruction indicates measurements
    #             for target in instruction.targets_copy(): # updating the measurement result of this qubit in all shots together. lost_qubits[:, target.value] tells us in which shot this qubit should we lost now.
    #                 measurement_events_all_shots[:, measurement_idx] = np.where(lost_qubits[:, target.value], 2, measurement_events_all_shots[:, measurement_idx])
    #                 measurement_idx += 1

    #     return measurement_events_all_shots

            
    
    def update_real_losses_by_instruction_ix(self, loss_detection_events: list):
        # only for THEORY
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
        
        
    def generate_experimental_circuit(self, loss_detection_events):
        """ This function takes the original circuit with places for potential losses and loss detection events, and generates the experimental circuit in which some qubits are gone. """
        
        if True in loss_detection_events: # there is a loss in this shot:
            # Initialization for every shot:
            self.lost_qubits_by_round_ix = {}
            self.real_losses_by_instruction_ix = {} # {instruction_ix: (lost_qubit), ...}
            
            # First sweep: get location of lost qubits in the circuit --> for a given lost qubit we get a set of potential loss events.
            self.qubit_lifecycles_and_losses = copy.deepcopy(self.qubit_lifecycles_and_losses_init) # init self.qubit_lifecycles_and_losses for this shot
            self.update_real_losses_by_instruction_ix(loss_detection_events=loss_detection_events)
            self.update_qubit_lifecycles_and_losses()  # update self.qubit_lifecycles_and_losses
            
            # Step 1 - generate the circuit that is really running in the experiment, for the given loss pattern (without gates after losing qubits):
            experimental_circuit = self.generate_loss_circuit(losses_by_instruction_ix = self.real_losses_by_instruction_ix, removing_Pauli_errors=False)
        
        else: # no losses in this shot
            experimental_circuit = self.circuit.copy()

        return experimental_circuit
    
    

    def generate_loss_circuit(self, losses_by_instruction_ix, removing_Pauli_errors=False, remove_gates_due_to_loss=True):
        
        def fill_loss_qubits_remove_gates_range(lost_q, instruction_ix):
            round_ix = next((round_ix for round_ix, instructions in sorted(self.rounds_by_ix.items()) if sum(len(self.rounds_by_ix[r]) for r in range(-1, round_ix+1)) > instruction_ix), None)
            # round_ix = round_lookup.get(instruction_ix)
            [reset_round_ix, detection_round_ix] = next(([cycle[0],cycle[1]] for cycle in self.qubit_lifecycles_and_losses[lost_q] if cycle[0] <= round_ix <= cycle[1]), None)
            # detection_round_offset_start = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < reset_round_ix)
            detection_round_offset_end = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < detection_round_ix + 1)
            
            if lost_q in loss_qubits_remove_gates_range: # this qubit was already recorder for loss, and was lost again!
                loss_qubits_remove_gates_range[lost_q].append((instruction_ix, detection_round_offset_end))
            else:
                loss_qubits_remove_gates_range[lost_q] = [(instruction_ix, detection_round_offset_end)]
            return loss_qubits_remove_gates_range
            
        lost_qubits = set(qubit for sublist in losses_by_instruction_ix.values() for qubit in sublist)
        
        # # Precompute round lookup for instructions
        # round_lookup = {
        #     instruction_ix: next((round_ix for round_ix, instructions in sorted(self.rounds_by_ix.items()) if sum(len(self.rounds_by_ix[r]) for r in range(-1, round_ix + 1)) > instruction_ix), None)
        #     for instruction_ix in losses_by_instruction_ix
        # }
        
        loss_qubits_remove_gates_range = {}
        for instruction_ix, lost_qubits_instruction in losses_by_instruction_ix.items():
            if isinstance(lost_qubits_instruction, list):
                for lost_q in lost_qubits_instruction:
                    loss_qubits_remove_gates_range = fill_loss_qubits_remove_gates_range(lost_q, instruction_ix)
            else:
                lost_q = lost_qubits_instruction
                loss_qubits_remove_gates_range = fill_loss_qubits_remove_gates_range(lost_q, instruction_ix)
        
        first_loss_instruction_index = min(losses_by_instruction_ix.keys(), default=0) # option to make faster: use this, don't give the generate_circuit_without_lost_qubit function the full circuit. put offset.
        first_loss_instruction_index = 0
        circuit_before_ix = self.circuit[:first_loss_instruction_index]
        circuit_after_ix = self.circuit[first_loss_instruction_index:]
        
        heralded_circuit_after_ix = self.generate_circuit_without_lost_qubit(lost_qubits = lost_qubits, circuit = circuit_after_ix, 
                                                                    circuit_offset = first_loss_instruction_index,loss_qubits_remove_gates_range=loss_qubits_remove_gates_range, 
                                                                    removing_Pauli_errors=removing_Pauli_errors, remove_gates_due_to_loss=remove_gates_due_to_loss) # after removing the following gates with the lost qubits.
        experimental_circuit = circuit_before_ix + heralded_circuit_after_ix
        
        
        return experimental_circuit
        
    
    
    
    
    
        
        
    def get_all_potential_loss_qubits(self):
        all_potential_loss_qubits_indices = []
        instruction_ix = 0
        for round_ix, round_instructions in self.rounds_by_ix.items():
            for instruction in round_instructions:
                if instruction.name == 'I': # potential loss event
                    targets = instruction.targets_copy()
                    for q in targets:
                        qubit = q.value
                        # losses_by_instruction_ix = {instruction_ix: [qubit]}
                        # num_potential_losses = 0
                        # [reset_round_ix, detection_round_ix] = next(([cycle[0],cycle[1]] for cycle in self.qubit_lifecycles_and_losses[qubit] if cycle[0] <= round_ix <= cycle[1]), None)
                        # potential_rounds_for_loss_events = range(reset_round_ix, detection_round_ix+1)
                        # for potential_round in potential_rounds_for_loss_events:
                        #     round_instructions = self.rounds_by_ix[potential_round]
                        #     round_offset = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < potential_round)
                        #     losses_indices_in_round = [i + round_offset for i, instruction in enumerate(round_instructions) if instruction.name == 'I' and qubit in [q.value for q in instruction.targets_copy()]]
                            
                            # num_potential_losses += len(losses_indices_in_round)
                            
                            # for potential_loss_index in losses_indices_in_round:
                        event_probability = self.circuit.loss_instruction_indices.get(instruction_ix, 0)
                        # event_probability = 1/num_potential_losses
                        loss_event = (instruction_ix, qubit, event_probability)
                        all_potential_loss_qubits_indices.append(loss_event)
                instruction_ix += 1
        return all_potential_loss_qubits_indices
    
    
    
    # def get_all_potential_loss_qubits(self):
    #     all_potential_loss_qubits_indices = []
    #     instruction_ix = 0
    #     for round_ix, round_instructions in self.rounds_by_ix.items():
    #         for instruction in round_instructions:
    #             if instruction.name == 'I': # potential loss event
    #                 targets = instruction.targets_copy()
    #                 for q in targets:
    #                     qubit = q.value
    #                     # losses_by_instruction_ix = {instruction_ix: [qubit]}
    #                     # num_potential_losses = 0
    #                     [reset_round_ix, detection_round_ix] = next(([cycle[0],cycle[1]] for cycle in self.qubit_lifecycles_and_losses[qubit] if cycle[0] <= round_ix <= cycle[1]), None)
    #                     potential_rounds_for_loss_events = range(reset_round_ix, detection_round_ix+1)
    #                     for potential_round in potential_rounds_for_loss_events:
    #                         round_instructions = self.rounds_by_ix[potential_round]
    #                         round_offset = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < potential_round)
    #                         losses_indices_in_round = [i + round_offset for i, instruction in enumerate(round_instructions) if instruction.name == 'I' and qubit in [q.value for q in instruction.targets_copy()]]
                            
    #                         # num_potential_losses += len(losses_indices_in_round)
                            
    #                         for potential_loss_index in losses_indices_in_round:
    #                             event_probability = self.circuit.loss_instruction_indices.get(potential_loss_index, 0)
    #                             # event_probability = 1/num_potential_losses
    #                             loss_event = (instruction_ix, qubit, event_probability)
    #                             all_potential_loss_qubits_indices.append(loss_event)
    #             instruction_ix += 1
    #     return all_potential_loss_qubits_indices
    
    
        
    # def preprocess_circuit_only_SSR(self, full_filename):
    #     os.makedirs(os.path.dirname(full_filename), exist_ok=True) # Ensure the directory exists
        
    #     self.circuit_supercheck_dems = {}
        
    #     # step 1: find all measurements in the circuit. [instruction_ix: qubit, ...]
        
    #     # step 2: find all combinations of measurements
        
    #     # step 3: for each combination, build a DEM with superchecks according to loss pattern (no Clifford errors propagation, only superchecks)
        
    #     # step 4: save all to the file
        
        
        
    def preprocess_circuit(self, full_filename):
        # GB's improvement: hyperedge matrix dont include event probability anymore
        # Here we look at the lifetimes of each qubit, to get all possible independent loss channels.
        # Each channel corresponds to a qubit lifetime and contains all potential loss places.
        # We would like to generate a DEM for the loss of every single qubit in every location of the circuit and save it.
        if self.printing:
            print("Preprocessing all loss circuits, one time only and it will be saved for next times!")
        
        
        os.makedirs(os.path.dirname(full_filename), exist_ok=True) # Ensure the directory exists

        self.circuit_independent_dems = {}
        instruction_ix = 0
        for round_ix, round_instructions in self.rounds_by_ix.items():
            for instruction in round_instructions:
                if instruction.name == 'I':
                    targets = instruction.targets_copy()
                    for q in targets:
                        qubit = q.value
                        losses_by_instruction_ix = {instruction_ix: [qubit]}
                        # num_potential_losses = 0
                        # [reset_round_ix, detection_round_ix] = next(([cycle[0],cycle[1]] for cycle in self.qubit_lifecycles_and_losses[qubit] if cycle[0] <= round_ix <= cycle[1]), None)
                        # potential_rounds_for_loss_events = range(reset_round_ix, detection_round_ix+1)
                        # for potential_round in potential_rounds_for_loss_events:
                        #     round_instructions = self.rounds_by_ix[potential_round]
                        #     round_offset = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < potential_round)
                        #     losses_indices_in_round = [i + round_offset for i, instruction in enumerate(round_instructions) if instruction.name == 'I' and qubit in [q.value for q in instruction.targets_copy()]]
                        #     num_potential_losses += len(losses_indices_in_round)
                        
                        hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = losses_by_instruction_ix, full_filename=full_filename) # GB: change event prob to 1
                        # save into the file:
                        key = self._generate_unique_key(losses_by_instruction_ix)
                        self.circuit_independent_dems[key] = hyperedges_matrix_dem
                        
                instruction_ix += 1
                
        with open(full_filename, 'wb') as file:
            pickle.dump((self.circuit_independent_dems, self.Meta_params), file)


    def get_all_potential_loss_locations_given_heralded_loss_new(self):
        # Loop over all lost qubits and for each one, mark a potential loss event with a certain probability
        self.potential_losses_by_instruction_index = {} # {(lost_q,loss_round_ix): {loss_instruction_ix: [lost_qubit, probability_of_this_event]}}

        for qubit in self.qubit_lifecycles_and_losses:
            qubit = int(qubit)
            qubit_lifecycles = self.qubit_lifecycles_and_losses[qubit]
            for lifecycle in qubit_lifecycles:
                reset_round_ix, detection_round_ix, lost_in_cycle = lifecycle
                if lost_in_cycle: # qubit was lost somewhere in this lifecycles
                    if (qubit, detection_round_ix) not in self.potential_losses_by_instruction_index:
                        self.potential_losses_by_instruction_index[(qubit, detection_round_ix)] = {}

                    potential_rounds_for_loss_events = range(reset_round_ix, detection_round_ix+1)
                    for potential_round in potential_rounds_for_loss_events:
                        round_instructions = self.rounds_by_ix[potential_round]
                        round_offset = sum(len(self.rounds_by_ix[round_ix]) for round_ix in self.rounds_by_ix if round_ix < potential_round)
                        losses_indices_in_round = [i + round_offset for i, instruction in enumerate(round_instructions) if instruction.name == 'I' and qubit in [q.value for q in instruction.targets_copy()]]
                        
                        for potential_loss_index in losses_indices_in_round:
                            loss_probability = self.circuit.loss_instruction_indices.get(potential_loss_index, 0)
                            self.potential_losses_by_instruction_index[(qubit, detection_round_ix)][potential_loss_index] = [loss_probability, detection_round_ix] # loss index: prob, detection_round_ix         
        if self.printing:
            print(f"potential_losses_by_instruction_index: {self.potential_losses_by_instruction_index}")
    
    
    
    def generate_all_potential_losses_combinations(self, potential_losses_by_instruction_index):
        """ 
        Input: a dictionary with loss events, key = (lost_q, round_ix).
        Output: a list of dictionary, each one represent another combination of losses. 
        Each dictionary: key: instruction_ix, value: qubit
        """
        # normalize the probabilities in each lifecycle to sum up to 1:
        for (lost_q, round_ix) in potential_losses_by_instruction_index:
            lifecycle_events = potential_losses_by_instruction_index[(lost_q, round_ix)]
            total_probability = sum([p for [p,_] in list(lifecycle_events.values())])
            for loss_index in lifecycle_events:
                lifecycle_events[loss_index][0] = lifecycle_events[loss_index][0] / total_probability # update prob 
        
        
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
        combination_events_probabilities = []
        for combination in all_combinations:
            combination_event_probability = 1
            combination_dict = {}
            for (lost_q, round_ix), instruction_ix, probability, meas_round_ix in combination:
                combination_event_probability *= probability
                if instruction_ix not in combination_dict:
                    combination_dict[instruction_ix] = [lost_q]
                else:
                    combination_dict[instruction_ix].append(lost_q)
            
            combination_dicts.append(combination_dict)
            combination_events_probabilities.append(combination_event_probability)
        
        return combination_dicts, combination_events_probabilities

    
    def generate_all_DEMs_and_sum_over_independent_events(self, use_pre_processed_data = True, return_hyperedges_matrix = False, remove_gates_due_to_loss=True):
        # if self.use_independent_and_first_comb_decoder = True, we add the DEM of the first loss combination (only if >1 loss event happened)
        # Now we generate many DEMs for every loss event and merge together in 2 steps in order to get 1 final DEM:
        DEMs_loss_pauli_events = [self.Pauli_DEM_matrix] # list of all DEMs for every loss event + DEM for Pauli errors. TODO: add here the Pauli DEM
        Probs_loss_pauli_events = [1]
        num_detectors = self.Pauli_DEM_matrix.shape[1]                
        
        start_time = time.time()
        for (lost_q, detection_round_ix) in self.potential_losses_by_instruction_index:
            DEMs_specific_loss_event = []
            Probs_specific_loss_event = [] # GB: new
            total_probability = sum(self.potential_losses_by_instruction_index[(lost_q, detection_round_ix)][potential_loss_ix][0] for potential_loss_ix in self.potential_losses_by_instruction_index[(lost_q, detection_round_ix)]) # CHECK
            
            for potential_loss_ix in self.potential_losses_by_instruction_index[(lost_q, detection_round_ix)]:
                event_probability = self.potential_losses_by_instruction_index[(lost_q, detection_round_ix)][potential_loss_ix][0] / total_probability
                losses_by_instruction_ix = {potential_loss_ix: [lost_q]}
                key = self._generate_unique_key(losses_by_instruction_ix)
                
                if use_pre_processed_data and key in self.circuit_independent_dems:
                    hyperedges_matrix_dem = self.circuit_independent_dems[key]
                else:
                    hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = losses_by_instruction_ix, 
                                                                        remove_gates_due_to_loss=remove_gates_due_to_loss) # GB: changed event prob to 1
                DEMs_specific_loss_event.append(hyperedges_matrix_dem) # GB new: matrix without event probability, only the DEM given this event happened.
                Probs_specific_loss_event.append(event_probability) # Gb: new
            
            start_time = time.time()
            DEM_specific_loss_event_lil = self.combine_DEMs_sum(DEMs_list=DEMs_specific_loss_event, num_detectors=num_detectors, Probs_list=Probs_specific_loss_event)
            DEM_specific_loss_event = DEM_specific_loss_event_lil.tocsr()
            # print(f'Time to combine DEMs regularly: {time.time() - start_time:.6f}s.')
            # start_time = time.time()
            # DEM_specific_loss_event = self.combine_DEMs_sum_csr(DEMs_list=DEMs_specific_loss_event, num_detectors=num_detectors, Probs_list=Probs_specific_loss_event)
            # print(f'Time to combine DEMs csr style: {time.time() - start_time:.6f}s.') # regular was faster than csr style
            
            # same_matrices = np.allclose(DEM_specific_loss_event.toarray(), DEM_specific_loss_event_lil.toarray(), atol=1e-8)
            # print(f"are they the same? {same_matrices}")
        
        
            DEMs_loss_pauli_events.append(DEM_specific_loss_event)
            Probs_loss_pauli_events.append(1)
            
            if self.printing:
                print(f"After summing over all DEMs for potential loss events given the loss of qubit {lost_q}, which was detected in round {detection_round_ix}, we got the following DEM_i:")
                print(DEM_specific_loss_event)
        
        # print(f'Time to sum over every lossy lifecycle independently: {time.time() - start_time:.4f}s.')      
        
        # start_time = time.time()
        if self.use_independent_and_first_comb_decoder and len(self.potential_losses_by_instruction_index) > 1:
            
            start_time = time.time()
            first_combination_dict = {}; combination_probability = 1
            for key, loss_dict in self.potential_losses_by_instruction_index.items():
                lost_q, detection_round_ix = key
                first_potential_loss_ix = min(loss_dict.keys())
                event_probability = loss_dict[first_potential_loss_ix][0]
                combination_probability *= event_probability

                if first_potential_loss_ix in first_combination_dict:
                    first_combination_dict[first_potential_loss_ix].append(lost_q)
                else:
                    first_combination_dict[first_potential_loss_ix] = [lost_q]
            # print(f'Time to get the first combination dictionary with the new code: {time.time() - start_time:.7f}s.')      

            # adjust first_comb probability according to the input:
            updated_combination_probability = self.first_comb_weight if self.first_comb_weight > 0 else combination_probability
            start_time2 = time.time()
            hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = first_combination_dict, remove_gates_due_to_loss=remove_gates_due_to_loss) # GB new: event prob=1
            # print(f'Time to generate the combination loss circuit: {time.time() - start_time2:.4f}s.')      
            DEMs_loss_pauli_events.append(hyperedges_matrix_dem)
            Probs_loss_pauli_events.append(updated_combination_probability)
        
        # print(f'Time to generate the loss combination DEM: {time.time() - start_time:.4f}s.')      
        
        # sum over all loss DEMs:
        # start_time = time.time()
        final_hyperedges_matrix = self.combine_DEMs_high_order_csr(DEMs_list=DEMs_loss_pauli_events, num_detectors=num_detectors, Probs_list=Probs_loss_pauli_events) # GB: new Probs_loss_pauli_events. TODO: change this function to also get Probs_loss_pauli_events
        # print(f'New method: Time to sum over all DEMS (independent, combination, Pauli) with high order equation: {time.time() - start_time:.4f}s.')      
        
        # start_time = time.time()
        # final_hyperedges_matrix_old_method = self.combine_DEMs_high_order(DEMs_list=DEMs_loss_pauli_events, num_detectors=num_detectors, Probs_list=Probs_loss_pauli_events)
        # print(f'Old method: Time to sum over all DEMS (independent, combination, Pauli) with high order equation: {time.time() - start_time:.4f}s.')      
        
        # same_matrices = np.allclose(final_hyperedges_matrix.toarray(), final_hyperedges_matrix_old_method.toarray(), atol=1e-8)
        # print(f"are they the same? {same_matrices}")
        
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
        # binary_matrix = hyperedges_matrix.copy()
        # binary_matrix = csr_matrix(hyperedges_matrix.shape, dtype=int)
        binary_matrix = lil_matrix(hyperedges_matrix.shape, dtype=int)
        probs_lists = []

        for i in range(hyperedges_matrix.shape[0]):
            row_data = hyperedges_matrix.getrow(i)
            if row_data.nnz > 0:  # if the row is not entirely zero
                
                # if self.loss_decoder_files_dir[:9] == '/n/home01': # on the cluster 
                #     probability = row_data.data[0] # Assuming all non-zero entries in a row have the same error probability. on the cluster only one [0]
                # else:
                probability = row_data.data[0]
                    
                probs_lists.append(probability)  # Collect the non-zero values before changing them
                # binary_matrix.rows[i] = row_data.rows[0]
                # binary_matrix.data[i] = np.ones_like(row_data.data[0])
                binary_matrix[i, row_data.indices] = 1
            else:
                # print(f"Row is zero. Row {i}")
                probs_lists.append(1e-20)
        
        # return binary_matrix, probs_lists
        return binary_matrix.tocsr(), probs_lists

    


    def convert_multiple_hyperedge_matrices_into_binary(self, hyperedges_matrix_list):
        # Ensure we have matrices to process
        if not hyperedges_matrix_list:
            return [], []

        # Stack all matrices vertically to create one large matrix
        stacked_matrix = vstack(hyperedges_matrix_list)

        # Initialize an array to hold the probabilities
        probs_matrix = np.full(stacked_matrix.shape[0], 1e-20, dtype=float)

        # Extract probabilities
        for i in range(stacked_matrix.shape[0]):
            row_data = stacked_matrix.getrow(i)
            if row_data.nnz > 0:  # If the row is not entirely zero
                # Capture the first non-zero value from the data
                probs_matrix[i] = row_data.data[0]  # Assuming all non-zero entries in a row have the same probability

        # Now convert the entire stacked matrix to a binary matrix
        binary_matrix = stacked_matrix.copy()
        binary_matrix.data[:] = 1  # Set all non-zero elements to 1

        # Calculate row indices for slicing
        row_counts = [m.shape[0] for m in hyperedges_matrix_list]
        split_indices = np.cumsum(row_counts)[:-1]

        # Initialize lists to hold the results
        dems_list = []
        probs_lists = []

        # Slice the binary matrix and probability list manually
        start_idx = 0
        for i, end_idx in enumerate(split_indices.tolist() + [len(probs_matrix)]):
            dems_list.append(binary_matrix[start_idx:end_idx])
            probs_lists.append(probs_matrix[start_idx:end_idx])
            start_idx = end_idx

        return dems_list, probs_lists



    def convert_multiple_hyperedge_matrices_into_binary_new(self, hyperedges_matrix_list):
        # Ensure we have matrices to process
        if not hyperedges_matrix_list:
            return [], []

        # Stack all matrices vertically to create one large matrix
        stacked_matrix = vstack(hyperedges_matrix_list)

        # Extract probabilities in a vectorized way
        probs_matrix = np.array(stacked_matrix.max(axis=1).todense()).flatten()
        probs_matrix[probs_matrix == 0] = 1e-20

        # Now convert the entire stacked matrix to a binary matrix
        binary_matrix = stacked_matrix.copy()
        binary_matrix.data[:] = 1  # Set all non-zero elements to 1

        # Calculate row indices for manual slicing
        row_counts = [m.shape[0] for m in hyperedges_matrix_list]
        split_indices = np.cumsum(row_counts[:-1])

        # Initialize lists to hold the results
        dems_list = []
        probs_lists = []

        # Manually slice the matrices
        start_idx = 0
        for i, count in enumerate(row_counts):
            end_idx = start_idx + count
            dems_list.append(binary_matrix[start_idx:end_idx])
            probs_lists.append(probs_matrix[start_idx:end_idx])
            start_idx = end_idx

        return dems_list, probs_lists


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
        
        
    
        
        
    def generate_circuit_without_lost_qubit(self, lost_qubits, circuit, circuit_offset = 0, loss_qubits_remove_gates_range = {}, removing_Pauli_errors=False, remove_gates_due_to_loss=True):
        # loss_qubits_remove_gates_range is dictionary where for each lost qubit we get a list of ranges of instruction indices in which we should remove the gates of this qubit.
        new_circuit = stim.Circuit()
        for ix, instruction in enumerate(circuit):
            instruction_ix = ix + circuit_offset
            if removing_Pauli_errors and instruction.name in ['PAULI_CHANNEL_1', 'PAULI_CHANNEL_2', 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR']:
                continue # don't put Pauli errors in the experimental loss circuit
            
            targets = [q.value for q in instruction.targets_copy()]
            if set(lost_qubits).intersection(set(targets)):
                
                if instruction.name in ['CZ', 'CX', 'SWAP'] and remove_gates_due_to_loss: # pairs of qubits
                    pairs = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
                    for (c,t) in pairs:
                        if (c in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[c])) or (t in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[t])):
                            if self.printing :
                                a=1
                                # print(f"Removing this gate from the heralded circuit: {instruction.name} {(c,t)}, because my lost qubits = {lost_qubits}")
                        else:
                            new_circuit.append(instruction.name, [c,t])

                elif instruction.name in ['H', 'R', 'RX', 'I'] and remove_gates_due_to_loss:
                    for q in targets:
                        if (q in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[q])):
                            if self.printing :
                                a=1
                                # print(f"Removing this gate from the heralded circuit: {instruction.name}, because my lost qubits = {lost_qubits}")
                        else:
                            new_circuit.append(instruction.name, [q])

                # elif instruction.name in ['PAULI_CHANNEL_1', 'PAULI_CHANNEL_2', 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR']:
                #     for q in targets:
                #         if (q in loss_qubits_remove_gates_range and any(i <= instruction_ix <= j for i, j in loss_qubits_remove_gates_range[q])):
                #             if self.printing :
                #                 a=1
                #                 # print(f"Removing this gate from the heralded circuit: {instruction.name}, because my lost qubits = {lost_qubits}")
                #         else:
                #             new_circuit.append(instruction.name, [q], instruction.gate_args_copy())
                            
                elif instruction.name in ['MRX', 'MR']: # to generate superchecks
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
                
                elif instruction.name in ['MX', 'M']:  # to generate superchecks
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

    #################################################################################### DEMs FUNCTIONS ####################################################################################
    
    def set_up_Pauli_DEM(self):
        # this function takes the circuit and produces a DEM, considering only Pauli errors on no losses (which are I gates anyway).
        # before generating the DEM, we need to convert the observables into detectors.
        # After generating the DEM, we need to convert it into a sparse matrix.
        
        
        circuit_for_Pauli_dem = self.observables_to_detectors(self.circuit.copy())
                        
        Pauli_DEM = circuit_for_Pauli_dem.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
        
        # Convert the DEM into a matrix:
        hyperedges_matrix_Pauli_DEM = self.convert_dem_into_hyperedges_matrix(Pauli_DEM, observables_converted_to_detectors=True)
        hyperedges_matrix_Pauli_DEM = hyperedges_matrix_Pauli_DEM.tocsr()
        self.Pauli_DEM = Pauli_DEM
        self.Pauli_DEM_matrix = hyperedges_matrix_Pauli_DEM


    # def combine_DEMs_high_order_wrong(self, DEMs_list, num_detectors, Probs_list):
    #     # Convert the DEMs to hashable rows
    #     def convert_rows(matrix):
    #         hashable_rows = {}
    #         matrix = matrix.tocsr()  # Ensure matrix is CSR format
    #         for ridx in range(matrix.shape[0]):
    #             row = matrix[ridx].indices
    #             if len(row) > 0:  # Skip empty rows
    #                 value = matrix[ridx].data[0]  # Assuming all values in a row are the same
    #                 pattern = tuple(sorted(row))
    #                 hashable_rows[pattern] = (ridx, value)
    #         return hashable_rows

    #     pattern_to_values = {}

    #     for matrix, prob in zip(DEMs_list, Probs_list):
    #         for pattern, (ridx, value) in convert_rows(matrix).items():
    #             if pattern in pattern_to_values:
    #                 pattern_to_values[pattern].append(value * prob)
    #             else:
    #                 pattern_to_values[pattern] = [value * prob]

    #     final_data = []
    #     final_rows = []
    #     final_cols = []

    #     for pattern, values in pattern_to_values.items():
    #         prod_1_minus_values = np.prod([1 - v for v in values])
    #         prob_i_terms = [(v * prod_1_minus_values / (1 - v)) for v in values]

    #         if len(values) > 3:
    #             for i in range(len(values)):
    #                 for j in range(i + 1, len(values)):
    #                     for k in range(j + 1, len(values)):
    #                         prob_i_terms.append(
    #                             values[i] * values[j] * values[k] * np.prod(
    #                                 [1 - values[n] for n in range(len(values)) if n not in (i, j, k)]
    #                             )
    #                         )

    #         value_sum = sum(prob_i_terms)
    #         for col in pattern:
    #             final_data.append(value_sum)
    #             final_rows.append(len(final_rows))
    #             final_cols.append(col)

    #     # Use COO format for efficient sparse matrix construction and then convert to CSR
    #     final_matrix = coo_matrix((final_data, (final_rows, final_cols)), shape=(len(final_rows), num_detectors)).tocsr()

    #     return final_matrix




    # def combine_DEMs_high_order_optimized(self, DEMs_list, num_detectors, Probs_list):
    #     # Convert the DEMs to hashable rows with sparse matrix operations
    #     def convert_rows_to_hashable(matrix):
    #         hashable_rows = defaultdict(list)
    #         matrix = matrix.tocsr()  # Ensure matrix is in CSR format for efficient row slicing
    #         for row_index in range(matrix.shape[0]):
    #             row_data = matrix.getrow(row_index)
    #             if row_data.nnz > 0:  # Skip empty rows
    #                 pattern = tuple(sorted(row_data.indices))
    #                 hashable_rows[pattern].append(row_data.data[0])
    #         return hashable_rows

    #     pattern_to_values = defaultdict(list)

    #     # Populate the pattern_to_values dictionary
    #     for matrix, prob in zip(DEMs_list, Probs_list):
    #         for pattern, values in convert_rows_to_hashable(matrix).items():
    #             for value in values:
    #                 pattern_to_values[pattern].append(value * prob)

    #     final_data = []
    #     final_rows = []
    #     final_cols = []

    #     # Combine the DEMs using high-order probability calculation
    #     for pattern, values in pattern_to_values.items():
    #         prob_terms = []
    #         prod_1_minus_values = np.prod([1 - v for v in values])

    #         # Handle first-order terms
    #         for v in values:
    #             prob_terms.append(v * prod_1_minus_values / (1 - v))

    #         # Handle higher-order terms
    #         if len(values) > 3:
    #             for i in range(len(values)):
    #                 for j in range(i + 1, len(values)):
    #                     for k in range(j + 1, len(values)):
    #                         prob_terms.append(
    #                             values[i] * values[j] * values[k] * np.prod(
    #                                 [1 - values[n] for n in range(len(values)) if n not in (i, j, k)]
    #                             )
    #                         )

    #         value_sum = sum(prob_terms)
    #         for col in pattern:
    #             final_data.append(value_sum)
    #             final_rows.append(len(final_rows))
    #             final_cols.append(col)

    #     # Use COO format for efficient sparse matrix construction and then convert to CSR
    #     final_matrix = coo_matrix((final_data, (final_rows, final_cols)), shape=(len(final_rows), num_detectors)).tocsr()

    #     return final_matrix


    def combine_DEMs_high_order_csr(self, DEMs_list, num_detectors, Probs_list):
        
        def convert_rows_csr(matrix):
            hashable_rows = {}
            matrix.sort_indices()  # Ensure the indices are sorted for consistent hashing
            for ridx in range(matrix.shape[0]):
                start_idx = matrix.indptr[ridx]
                end_idx = matrix.indptr[ridx + 1]
                if start_idx < end_idx:  # Non-empty row
                    pattern = tuple(matrix.indices[start_idx:end_idx])
                    value = matrix.data[start_idx]  # Assuming all values in a row are the same
                    hashable_rows[pattern] = (ridx, value)
            return hashable_rows

        # Convert all CSR matrices to hashable row format
        start_time = time.time()
        pattern_to_values = {}
        for matrix, prob in zip(DEMs_list, Probs_list):
            # matrix = matrix.tocsr()
            
            for pattern, (ridx, value) in convert_rows_csr(matrix).items():
                if pattern in pattern_to_values:
                    pattern_to_values[pattern].append(value * prob)
                else:
                    pattern_to_values[pattern] = [value * prob]
        
        # print(f'conversion to csr and creating pattern to values took {time.time() - start_time:.6f}s')      
        
        # Apply the formula of high-order probability sum to combine the values for each pattern
        start_time = time.time()
        final_rows = {}
        for pattern, values in pattern_to_values.items():
            prob_i_terms = [v * np.prod([1 - x for x in values]) / (1 - v) for v in values]
            if len(values) > 3:
                # Consider terms where 3 specific events happen
                for i, v1 in enumerate(values):
                    for j, v2 in enumerate(values[i+1:], start=i+1):
                        for k, v3 in enumerate(values[j+1:], start=j+1):
                            prob_i_terms.append(v1 * v2 * v3 * np.prod([1 - x for n, x in enumerate(values) if n not in (i, j, k)]))
            final_rows[pattern] = sum(prob_i_terms)
        # print(f'high order formula took {time.time() - start_time:.6f}s')      
        
        # Finally, we can build the final matrix
        # Extract row indices, column indices, and data in one shot
        # start_time = time.time()
        row_indices = []
        col_indices = []
        data = []

        for i, (pattern, value) in enumerate(final_rows.items()):
            row_indices.extend([i] * len(pattern))  # The same row index for all elements in this row
            col_indices.extend(pattern)             # Column indices corresponding to the pattern
            data.extend([value] * len(pattern))     # Same value for each entry in the pattern
        
        # Create the CSR matrix directly from these arrays
        final_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(final_rows), num_detectors))
        # print(f'building final_matrix_csr took {time.time() - start_time:.6f}s')    
    
    
        # start_time = time.time()
        # final_matrix = lil_matrix((0, num_detectors), dtype=float)
        # for pattern, value in final_rows.items():
        #     new_row = np.zeros(num_detectors, dtype=float)
        #     new_row[list(pattern)] = value
        #     final_matrix.resize((final_matrix.shape[0] + 1, num_detectors))
        #     final_matrix[-1, :] = new_row
        # print(f'building final matrix took {time.time() - start_time:.6f}s')      

        # same_matrices = np.allclose(final_matrix.toarray(), final_matrix_csr.toarray(), atol=1e-8)
        # print(f"are they the same? {same_matrices}")
        
        
        # # Build the final CSR matrix
        # row_indices = []
        # col_indices = []
        # data = []

        # for pattern, value in final_rows.items():
        #     row_indices.append(len(row_indices))
        #     col_indices.extend(pattern)
        #     data.extend([value] * len(pattern))

        # final_matrix_csr = csr_matrix((data, (row_indices, col_indices)), shape=(len(row_indices), num_detectors))

        # 'final_matrix_csr' is now the final CSR matrix containing the summed probabilities
        return final_matrix



    def combine_DEMs_high_order(self, DEMs_list, num_detectors, Probs_list):
        # input: lil matrices. need to convert to csr at the beginning: hyperedges_matrix_dem_csr = hyperedges_matrix_dem_lil.tocsr()
        # Assuming 'DEMs_list' is a list of lil_matrix objects, each representing a DEM for a loss event and one DEM for Pauli events.
        # 'Probs_loss_pauli_events' is a list of probabilities corresponding to each DEM.
        
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
        for matrix, prob in zip(DEMs_list, Probs_list):
            for pattern, (ridx, value) in convert_rows(matrix).items():
                if pattern in pattern_to_values:
                    pattern_to_values[pattern].append(value * prob)
                else:
                    pattern_to_values[pattern] = [value * prob]

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
    
    
    def convert_dem_into_hyperedges_matrix(self, dem, event_probability=1, observables_converted_to_detectors=False):
        # Output hyperedge matrix have num col = num detectors + num observables
        # If observables_converted_to_detectors = True: DEM where observables are already converted detectors
        # Note that observables_errors_interactions will be non trivial only if observables_converted_to_detectors = False
        # generates lil matrix
        
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



    def convert_dem_into_hyperedges_matrix_csr(self, dem, event_probability=1, observables_converted_to_detectors=False):
        # generate csr matrix directly. slower than the lil matrix.
        
        
        num_detectors = dem.num_detectors  # includes num_observables because 
        num_observables = len(self.observables_indices)
        num_total = num_detectors if observables_converted_to_detectors else num_detectors + num_observables

        num_errors = sum(1 for error in dem if str(error)[:5] == 'error' and error.args_copy()[0] != 0)
        hyperedges_matrix = csr_matrix((num_errors, num_total), dtype=float)  # Use CSR matrix directly

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


    def combine_DEMs_sum_csr(self, DEMs_list, num_detectors, Probs_list):
        # Multiply each DEM by its corresponding probability
        weighted_DEMs = [dem.multiply(prob) for dem, prob in zip(DEMs_list, Probs_list)]
        
        # Stack all weighted DEMs vertically into a single CSR matrix
        stacked_matrix = vstack(weighted_DEMs, format='csr')
        
        # Sum all rows with the same pattern by converting to COO and summing the values
        coo_matrix = stacked_matrix.tocoo()
        
        # Get unique rows by summing duplicate patterns
        unique_patterns, unique_indices = np.unique(np.vstack((coo_matrix.row, coo_matrix.col)).T, axis=0, return_inverse=True)
        unique_values = np.bincount(unique_indices, weights=coo_matrix.data)

        # Create the final CSR matrix with the unique patterns and summed values
        row_indices = unique_patterns[:, 0]
        col_indices = unique_patterns[:, 1]
        
        final_matrix = csr_matrix((unique_values, (row_indices, col_indices)), shape=(np.max(row_indices) + 1, num_detectors))

        return final_matrix


    def combine_DEMs_sum(self, DEMs_list, num_detectors, Probs_list):
        pattern_to_value = defaultdict(float)

        for dem, prob in zip(DEMs_list, Probs_list):
            dem = dem.tolil()  # Ensure the DEM is in LIL format for efficient row access
            for i, (row, data) in enumerate(zip(dem.rows, dem.data)):
                if data:  # Skip empty rows
                    value = data[0] * prob  # Multiply by the event probability
                    pattern = tuple(row)
                    pattern_to_value[pattern] += value

        # Build the final matrix
        final_matrix = lil_matrix((0, num_detectors), dtype=float)
        for pattern, value in pattern_to_value.items():
            new_row = np.zeros(num_detectors, dtype=float)
            new_row[list(pattern)] = value
            final_matrix.resize((final_matrix.shape[0] + 1, num_detectors))
            final_matrix[-1, :] = new_row

        return final_matrix
    
    
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
    
    
    
    
    def from_hyperedges_matrix_into_stim_dem(self, final_hyperedges_matrix):
        # Input hyperedges_matrix includes observables as detectors (in columns self.observables_indices). We need to convert them back into observables.
        
        # Step 4: bring back the observables and create a stim.DEM object: 
        final_dem = stim.DetectorErrorModel()
        
        # Iterate over the rows of the hypergraph_matrix to create the DEM while adjusting the circuit based on the hyperedges_matrix to re-include observables
        observables_indices_in_dem = []
        for row_index in range(final_hyperedges_matrix.shape[0]):
            row = final_hyperedges_matrix.getrow(row_index)
            non_zero_columns = row.nonzero()[1]
            # if self.loss_decoder_files_dir[:9] == '/n/home01': # on the cluster 
            #     probability = row.data[0] # Assuming all non-zero entries in a row have the same error probability.
            # else:
            #     # probability = row.data[0]
            probability = row.data[0]  # Assuming all non-zero entries in a row have the same error probability.

            # Construct the error command by specifying detector and observable targets
            error_targets = []
            # num_detectors = final_hyperedges_matrix.shape[1] - self.extra_num_detectors  # number of detector columns

            # Append detectors and convert detectors to observables:
            for d in non_zero_columns: # detector or observable index, from 0 up to num detectors + observables

                # new code - without assuming observables are at the end of the circuit:
                if d not in self.observables_indices: # this col is a detector
                    error_targets.append(stim.target_relative_detector_id(d))
                else: # this col is an observable
                    observable_index = self.observables_indices.index(d)  # Finding the index of observable 'd' in the self.observables_indices list
                    error_targets.append(stim.target_logical_observable_id(observable_index))
                    observables_indices_in_dem.append(d)
                    
            # Append error with probability to final_dem
            if self.printing:
                print(f"Error targets = {error_targets}, Probability = {probability}")
            final_dem.append("error", probability, error_targets)
        
        # new part: append observables to DEM that didn't have any error:
        observables_without_errors = [x for x in self.observables_indices if x not in observables_indices_in_dem]
        for d in observables_without_errors:
            observable_index = self.observables_indices.index(d) 
            error_targets = [stim.target_logical_observable_id(observable_index)]
            final_dem.append("error", 0, error_targets)
            
        return final_dem
        
        
        
    def generate_dem_loss_circuit(self, losses_by_instruction_ix, event_probability = 1, full_filename='', remove_gates_due_to_loss=True):
        """ Generate a DEM for given losses as a lil matrix. new - returns as csr! """
        # Generate a unique key based on losses_by_instruction_ix and self.Meta_params
        key = self._generate_unique_key(losses_by_instruction_ix) 

        if os.path.isfile(full_filename): # Load and return the existing dem
            with open(full_filename, 'rb') as file:  # Note the 'rb' mode here
                circuit_independent_dems, _ = pickle.load(file)
                if key in circuit_independent_dems:
                    hyperedges_matrix_dem = circuit_independent_dems[key]
                    return hyperedges_matrix_dem
            
        else: # generate circuit and dem
            # options to make this part faster:
            # save the loss circuit with the loss combination, with file name according to num of losses
            loss_circuit = self.generate_loss_circuit(losses_by_instruction_ix, removing_Pauli_errors=True, remove_gates_due_to_loss=remove_gates_due_to_loss)
            
            # replace final observables with detectors:
            final_loss_circuit = self.observables_to_detectors(loss_circuit)

            # get the dem (with observables on columns):
            dem_heralded_circuit = final_loss_circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=True)
            
            # convert the DEM into a matrix and sum up with previous DEMs:
            start_time = time.time()
            # hyperedges_matrix_dem_csr = self.convert_dem_into_hyperedges_matrix_csr(dem_heralded_circuit, event_probability=event_probability, observables_converted_to_detectors=True)
            # print(f'csr took {time.time() - start_time:.6f}s')      
            # start_time = time.time()
            hyperedges_matrix_dem = self.convert_dem_into_hyperedges_matrix(dem_heralded_circuit, event_probability=event_probability, observables_converted_to_detectors=True)  # format: lil matrix
            hyperedges_matrix_dem = hyperedges_matrix_dem.tocsr()
            # print(f'lil matrix + conversion to csr took {time.time() - start_time:.6f}s')
            # matrices_are_equal = (hyperedges_matrix_dem != hyperedges_matrix_dem_csr).nnz == 0
            # print(f" matrices_are_equal= {matrices_are_equal}")  # This will print True if they are the same, False otherwise

            if self.printing:
                print(f"The DEM for the loss circuit with losses {losses_by_instruction_ix} is: \n{dem_heralded_circuit}")
                print(hyperedges_matrix_dem)
                print(f"If we would sample this circuit 5 times, we would get:")
                sampler = final_loss_circuit.compile_detector_sampler()
                detection_events, observable_flips = sampler.sample(5, separate_observables=True)
                print(f"detection_events = \n{detection_events}, observable_flips = \n{observable_flips}")


            return hyperedges_matrix_dem
        
    
    #################################################################################### LIFECYCLE FUNCTIONS ####################################################################################:
    
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
        # loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
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


    def get_qubits_lifecycles_None(self):
        # Iterate through circuit. Record the lifecycles of each qubit, from initialization to measurement.
        # loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
        round_ix = -1
        inside_qec_round = False
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
                else: # end of round
                    round_ix += 1
                inside_qec_round = not inside_qec_round
                continue
            

        # Handle unmeasured qubits at the end of the circuit
        for q in qubit_active_cycle:
            if qubit_active_cycle[q] is not None:
                self.qubit_lifecycles_and_losses[q][qubit_active_cycle[q]][1] = round_ix
                
                
    def get_qubits_lifecycles_MBQC(self):
        # Iterate through circuit. Record the lifecycles of each qubit, from initialization to measurement.
        # not only for MBQC, for every method where lifecycles are set according to initialization and measurements
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


####################################################################################: COMBINATION DECODER FUNCTIONS ####################################################################################:


    def generate_all_DEMs_and_sum_over_combinations(self, return_hyperedges_matrix=False, use_pre_processed_data=True):
        # Now we generate many DEMs for every potential loss event combination and merge together in 2 steps in order to get 1 final DEM:
        # event_probability is the probability for each loss event combination
        
        DEMs_loss_pauli_events = [self.Pauli_DEM_matrix] # list of all DEMs for every loss event + DEM for Pauli errors.
        Probs_loss_pauli_list = [1]
        num_detectors = self.Pauli_DEM_matrix.shape[1]
        hyperedges_matrix_loss_event = dok_matrix((0, num_detectors), dtype=float) # Initialize as dok_matrix for efficient incremental construction
        
        DEMs_loss_events = []
        new_circuit_comb_dems = {} # new dems to save now
        Probs_loss_events_list = []
        # TODO: make the comb decoder faster by taking into account only combinations with highest probability
        for (potential_loss_combination, event_comb_prob) in zip(self.all_potential_losses_combinations, self.combinations_events_probabilities):
            key = self._generate_unique_key(potential_loss_combination)
            num_of_losses = sum(len(lst) for lst in potential_loss_combination.values())
            if key in self.circuit_comb_dems:
                hyperedges_matrix_dem = self.circuit_comb_dems[key]
            else:
                if self.printing:
                    print(f"Combination {potential_loss_combination} not in dictionary. need to generate loss circuit")
                hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = potential_loss_combination)
                self.circuit_comb_dems[key] = hyperedges_matrix_dem # new - save the new result into the dictionary with the preprocessed data!
                
                # Update the pre-processed data.
                if num_of_losses in new_circuit_comb_dems:
                    new_circuit_comb_dems[num_of_losses][key] = hyperedges_matrix_dem
                else:
                    new_circuit_comb_dems[num_of_losses] = {key: hyperedges_matrix_dem}
            
            DEMs_loss_events.append(hyperedges_matrix_dem)
            Probs_loss_events_list.append(event_comb_prob)
        
        # Update the relevant file: DEBUG. Change this to save a dict according to num_losses and update together at the end!
        if self.save_data_during_sim:
            for num_of_losses, dem_dict in new_circuit_comb_dems.items():
                full_filename_dems = f'{self.loss_decoder_files_dir}/circuit_dems_{num_of_losses}_losses.pickle'
                if os.path.exists(full_filename_dems):
                    try:
                        with open(full_filename_dems, 'rb') as file:
                            current_circuit_comb_dems, _ = pickle.load(file)  # Get current dictionary.
                    except (FileNotFoundError, EOFError):  # Added EOFError for corrupted files
                        current_circuit_comb_dems = {}
                else:
                    current_circuit_comb_dems = {}

                current_circuit_comb_dems.update(dem_dict)  # Update with new info.
                
                with open(full_filename_dems, 'wb') as file:
                    pickle.dump((current_circuit_comb_dems, self.Meta_params), file)  # Update the file.

                    
                
        # sum over all loss DEMs:
        hyperedges_matrix_loss_event_lil = self.combine_DEMs_sum(DEMs_list=DEMs_loss_events, num_detectors=num_detectors, Probs_list=Probs_loss_events_list)
        hyperedges_matrix_loss_event = hyperedges_matrix_loss_event_lil.tocsr()
        
        
        # save the sum of the DEMs for this loss event in DEMs_loss_pauli_events
        # hyperedges_matrix_loss_event = hyperedges_matrix_loss_event.tocsr()
        DEMs_loss_pauli_events.append(hyperedges_matrix_loss_event)
        Probs_loss_pauli_list.append(1)
        
        if self.printing:
            print(f"After summing over all DEMs for potential loss combinations, we got the following DEM_loss:")
            print(hyperedges_matrix_loss_event)
                
        # Step 2: sum over loss DEM + Pauli errors DEM, according to the high-order formula:
        # final_hyperedges_matrix = self.combine_DEMs_high_order(DEMs_list=DEMs_loss_pauli_events, num_detectors=num_detectors, Probs_list=Probs_loss_pauli_list)
        final_hyperedges_matrix = self.combine_DEMs_high_order_csr(DEMs_list=DEMs_loss_pauli_events, num_detectors=num_detectors, Probs_list=Probs_loss_pauli_list)

        # Step 3: convert the final dem into rows:
        if self.printing:
            print(f"Final DEM matrix after combining all DEMs for losses and Pauli: {final_hyperedges_matrix}")

        if return_hyperedges_matrix:
            return final_hyperedges_matrix
        else:
            # Step 4: bring back the observables and create a stim.DEM object: 
            final_dem = self.from_hyperedges_matrix_into_stim_dem(final_hyperedges_matrix)
            return final_dem




        
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
        batch_size = max(1, int(total_combinations * 0.005))
    
        ### Step 3: process all combinations:
        circuit_comb_dems = {}
        for combination in all_combinations:
            losses_by_instruction_ix = {}
            # combination_event_probability = 1
            for (instruction_ix, qubit, event_probability) in combination:
                # combination_event_probability *= event_probability
                if instruction_ix not in losses_by_instruction_ix:
                    losses_by_instruction_ix[instruction_ix] = [qubit]
                else:
                    losses_by_instruction_ix[instruction_ix].append(qubit)
            
            hyperedges_matrix_dem = self.generate_dem_loss_circuit(losses_by_instruction_ix = losses_by_instruction_ix, full_filename=full_filename) # GB's improvement: event prob is 1 for preprocessing step
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
        batch_size = max(1, int(total_combinations * 0.005))
        print(f"For dx={self.dx}, dy={self.dy}, num of losses = {num_of_losses}, we got {total_combinations} combinations to process.")

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
                # combination_event_probability = 1
                for (instruction_ix, qubit, event_probability) in combination:
                    # combination_event_probability *= event_probability
                    if instruction_ix not in losses_by_instruction_ix:
                        losses_by_instruction_ix[instruction_ix] = [qubit]
                    else:
                        losses_by_instruction_ix[instruction_ix].append(qubit)

                key = self._generate_unique_key(losses_by_instruction_ix)
                if key not in circuit_comb_dems:
                    hyperedges_matrix_dem = self.generate_dem_loss_circuit(
                        losses_by_instruction_ix=losses_by_instruction_ix,
                        full_filename=full_filename_batch
                    ) # GB: event prob = 1 for preprocessing
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
        
        
        
        
        
        
    ####################################################################################: OLD FUNCTIONS  ####################################################################################:
    
    
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
        
        
        
        
        
        
    # def combine_DEMs_sum(self, DEMs_list, num_detectors):
    # old function, don't use Probs_list.
    #     # Stack all DEMs vertically
    #     all_dems_stacked = vstack(DEMs_list, format='csr')

    #     # Sum duplicates after vertical stacking
    #     all_dems_stacked.sum_duplicates()

    #     # Group by the pattern of non-zero columns
    #     pattern_to_values = {}
    #     for i in range(all_dems_stacked.shape[0]):
    #         # Extract the row slice
    #         row_data = all_dems_stacked.data[all_dems_stacked.indptr[i]:all_dems_stacked.indptr[i+1]][0]
    #         row_indices = all_dems_stacked.indices[all_dems_stacked.indptr[i]:all_dems_stacked.indptr[i+1]]
    #         # Create a hashable representation of the row
    #         row_pattern = tuple(row_indices)
    #         # Sum the values of rows that match this pattern
    #         if row_pattern in pattern_to_values:
    #             pattern_to_values[row_pattern] += row_data
    #         else:
    #             pattern_to_values[row_pattern] = row_data
            
    #     # Finally, we can build the final matrix
    #     final_matrix = lil_matrix((0, num_detectors), dtype=float)
    #     for pattern, value in pattern_to_values.items():
    #         new_row = np.zeros(num_detectors, dtype=float)
    #         new_row[list(pattern)] = value
    #         final_matrix.resize((final_matrix.shape[0] + 1, num_detectors))
    #         final_matrix[-1, :] = new_row
            
    #     return final_matrix
