import numpy as np
import matplotlib.pyplot as plt
from BiasedErasure.main_code.noise_channels import *
from BiasedErasure.main_code.utilities import convert_qubit_losses_into_measurement_events
import qec
import progressbar
import time
import sinter
import math
from BiasedErasure.main_code.circuits import *
from distutils.util import strtobool
from BiasedErasure.delayed_erasure_decoders.MLE_Loss_Decoder import MLE_Loss_Decoder
import os.path
import os
import json
from hashlib import sha256
import pickle
from BiasedErasure.main_code.XZZX import XZZX
import itertools
import re


class Simulator:
    def __init__(self, Meta_params, 
                bloch_point_params,
                noise = None,
                atom_array_sim = False,
                phys_err_vec = [],
                loss_detection_method = None,
                cycles = None,
                output_dir = None,
                save_filename = None,
                first_comb_weight=0.5,
                dont_use_loss_decoder=False,
                save_data_during_sim = False,
                n_r = 1,
                use_independent_decoder=True,
                use_independent_and_first_comb_decoder=False
                ) -> None:
        """
        Initialize the Simulator.
        Meta_params defines the architecture ('CBQC', 'MBQC') and the error correction code.
        bloch_point_params include bias_ratio and erasure_ratio, etc.
        """
        self.Meta_params = Meta_params
        self.bloch_point_params = bloch_point_params
        self.architecture = Meta_params['architecture']
        self.code = Meta_params['code'] # code class in qec
        self.num_logicals = int(Meta_params['num_logicals'])
        self.bias_ratio = float(bloch_point_params['bias_ratio'])
        self.erasure_ratio = float(bloch_point_params['erasure_ratio'])
        self.bias_preserving_gates = strtobool(Meta_params['bias_preserving_gates'])
        self.logical_basis = Meta_params['logical_basis']
        self.noise = noise
        self.atom_array_sim = atom_array_sim
        self.phys_err_vec = phys_err_vec
        self.is_erasure_biased = strtobool(Meta_params['is_erasure_biased'])
        self.loss_detection_freq = int(Meta_params['LD_freq'])
        self.SSR = strtobool(Meta_params['SSR'])
        self.heralded_circuit = loss_detection_method
        self.cycles = cycles
        self.cycles_str = Meta_params['cycles']
        self.loss_detection_method_str = Meta_params['LD_method']
        self.ordering_type = Meta_params['ordering'] # relevant for rotated codes
        self.loss_decoder = Meta_params['loss_decoder']
        self.decoder = Meta_params['decoder']
        self.loss_decoder_type = Meta_params['loss_decoder']
        self.circuit_type = Meta_params['circuit_type']
        self.Steane_type = Meta_params['Steane_type']
        self.obs_pos = Meta_params['obs_pos'] if Meta_params['obs_pos'] is not None else 'd/2'
        self.printing = strtobool(Meta_params['printing'])
        self.output_dir = output_dir
        self.save_filename = save_filename
        self.first_comb_weight = first_comb_weight
        self.save_data_during_sim = save_data_during_sim
        self.n_r = Meta_params['n_r'] # num of rounds before a QEC round
        
        ### loss decoder type:
        self.dont_use_loss_decoder = dont_use_loss_decoder # if TRUE, we are not using loss decoder at all. all shots get same DEM.
        self.use_independent_decoder = use_independent_decoder
        self.use_independent_and_first_comb_decoder = use_independent_and_first_comb_decoder
        # TODO: define these params like this instead of variables input here.. (next version of code!)
        # self.use_independent_decoder = True if self.loss_decoder == 'independent' else False
        # self.use_independent_and_first_comb_decoder = True if self.loss_decoder == 'first_comb' else False
        # self.dont_use_loss_decoder = True if self.loss_decoder == 'None' else True


        self.xzzx = True if atom_array_sim else False
        self.replace_H_Ry = True if atom_array_sim else False


        # self.replace_H_Ry = False # debug

        ### Circuit type to use
        self.circuit_index = str(Meta_params['circuit_index']) if 'circuit_index' in Meta_params.keys() else '0'


    def get_job_id(self):
        # Check for environment variables used by different cluster management systems
        for env_var in ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID']:
            job_id = os.environ.get(env_var)
            if job_id is not None:
                return job_id
        return None  # or raise an error if the job ID is critical


    def generate_circuit(self, dx, dy, cycles, phys_err, noise_params={}):
        entangling_gate_error_rate, entangling_gate_loss_rate = self.noise(phys_err, self.bias_ratio)
        if self.circuit_type in ['memory', 'memory_wrong']:
            measure_wrong_basis = True if self.circuit_type == 'memory_wrong' else False
            if self.architecture == 'CBQC':
                return memory_experiment_surface_new(dx=dx, dy=dy, code=self.code, QEC_cycles=cycles, entangling_gate_error_rate=entangling_gate_error_rate, 
                                                entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio = self.erasure_ratio,
                                                num_logicals=self.num_logicals, logical_basis=self.logical_basis, 
                                                biased_pres_gates = self.bias_preserving_gates, ordering = self.ordering_type,
                                                loss_detection_method = self.loss_detection_method_str, 
                                                loss_detection_frequency = self.loss_detection_freq, atom_array_sim=self.atom_array_sim, 
                                                replace_H_Ry=self.replace_H_Ry, xzzx=self.xzzx, noise_params=noise_params, circuit_index = self.circuit_index, measure_wrong_basis = measure_wrong_basis)
                
                # return memory_experiment_surface(dx=dx, dy=dy, code=self.code, QEC_cycles=cycles-1, entangling_gate_error_rate=entangling_gate_error_rate, 
                #                                 entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio = self.erasure_ratio,
                #                                 num_logicals=self.num_logicals, logical_basis=self.logical_basis, 
                #                                 biased_pres_gates = self.bias_preserving_gates, ordering = self.ordering_type,
                #                                 loss_detection_method = self.loss_detection_method_str, 
                #                                 loss_detection_frequency = self.loss_detection_freq, atom_array_sim=self.atom_array_sim)
            elif self.architecture == 'MBQC':
                return memory_experiment_MBQC(dx=dx, dy=dy, QEC_cycles=cycles, entangling_gate_error_rate=entangling_gate_error_rate, 
                                                entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio = self.erasure_ratio,
                                                logical_basis=self.logical_basis, 
                                                biased_pres_gates = self.bias_preserving_gates, atom_array_sim=self.atom_array_sim)
                
            elif self.circuit_type == 'random_alg':
                assert dx==dy
                return random_logical_algorithm(code=self.code, num_logicals = self.num_logicals, depth=self.cycles+1, distance=dx, n_r = self.n_r, bias_ratio = self.bias_ratio, erasure_ratio = self.erasure_ratio, phys_err = phys_err, output_dir = self.output_dir)
        
        
        elif self.circuit_type[:10] == 'logical_CX':
            num_CX_per_layer_list = self.Meta_params["num_CX_per_layer_list"]
            num_layers = len(num_CX_per_layer_list)
            self.Meta_params['circuit_type'] = f'logical_CX__Nlayers{num_layers}__NCX{"_".join(map(str, num_CX_per_layer_list))}' 
            print(self.Meta_params['circuit_type'])
            return CX_experiment_surface(dx=dx, dy=dy, code=self.code, num_CX_per_layer_list=num_CX_per_layer_list, num_layers=num_layers, 
                                                num_logicals=self.num_logicals, logical_basis=self.logical_basis, 
                                                biased_pres_gates = self.bias_preserving_gates, ordering = self.ordering_type,
                                                loss_detection_method = self.loss_detection_method_str, 
                                                loss_detection_frequency = self.loss_detection_freq, atom_array_sim=self.atom_array_sim, 
                                                replace_H_Ry=self.replace_H_Ry, xzzx=self.xzzx, noise_params=noise_params, printing=self.printing, circuit_index = self.circuit_index )
        
        elif self.circuit_type == 'lattice_surgery':
            self.num_logicals = 1
            return lattice_surgery_experiment_surface(dx=dx, dy=dy, code=self.code, num_layers=self.cycles,
                                                logical_basis=self.logical_basis, 
                                                biased_pres_gates = self.bias_preserving_gates, ordering = self.ordering_type,
                                                loss_detection_method = self.loss_detection_method_str, 
                                                loss_detection_frequency = self.loss_detection_freq, atom_array_sim=self.atom_array_sim, 
                                                replace_H_Ry=self.replace_H_Ry, xzzx=self.xzzx, noise_params=noise_params, printing=self.printing, circuit_index = self.circuit_index )
        
            
                
        elif self.circuit_type in ['GHZ_all_o1', 'GHZ_save_o1','GHZ_all_o2', 'GHZ_save_o2']:
            order_1 = []
            order_2 = []
            for i in range(1, self.num_logicals):
                order_1.append((0, i))
                order_2.append((i - 1, i))
            chosen_order = order_1 if self.circuit_type.endswith("1") else order_2
            if self.circuit_type in ['GHZ_all_o1', 'GHZ_all_o2']:
                return GHZ_experiment_Surface(dx=dx, dy=dy, order=chosen_order, num_logicals=self.num_logicals, code=self.code, QEC_cycles=cycles, 
                                            entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate, 
                                            erasure_ratio = self.erasure_ratio,
                                            logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates, 
                                            loss_detection_on_all_qubits=True, atom_array_sim=self.atom_array_sim)
            elif self.circuit_type in ['GHZ_save_o1', 'GHZ_save_o2']:
                return GHZ_experiment_Surface(dx=dx, dy=dy, order=chosen_order, num_logicals=self.num_logicals, code=self.code, QEC_cycles=cycles, 
                                            entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate, 
                                            erasure_ratio = self.erasure_ratio,
                                            logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates, 
                                            loss_detection_on_all_qubits=False, atom_array_sim=self.atom_array_sim)
        elif self.circuit_type == 'Steane_QEC':
            obs_pos = int(eval(self.obs_pos.replace('d', str(min(dx, dy)))))
            return Steane_QEC_circuit(dx=dx, dy=dy, code=self.code, Steane_type=self.Steane_type, QEC_cycles=cycles-1,
                                        entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate,
                                        erasure_ratio=self.erasure_ratio,
                                        logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates,
                                        loss_detection_on_all_qubits=True, atom_array_sim=self.atom_array_sim, obs_pos=obs_pos)
                
        else:
            return None
    
    
    def get_detection_events_from_measurements(self, measurement_events, circuit):
        
        measurement_events_no_loss = measurement_events.copy() 
        measurement_events_no_loss[measurement_events_no_loss == 2] = 0 #change all values in detection_events from 2 to 0
        measurement_events_no_loss = measurement_events_no_loss.astype(np.bool_)
        detection_events, observable_flips = circuit.compile_m2d_converter().convert(measurements=measurement_events_no_loss, separate_observables=True)
        detection_events_int = detection_events.astype(np.int32)
        
        return detection_events_int, observable_flips
    
    
    
    def get_detections_from_experimental_data(self, dx: int, dy: int, measurement_events: np.ndarray, noise_params:dict):
        # Step 1 - generate the experimental circuit in our simulation:
        LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None, noise_params=noise_params)

        # Step 2 - get observables flips from lossless circuit
        detection_events, observable_flips = self.get_detection_events_from_measurements(measurement_events, LogicalCircuit)

        return detection_events, observable_flips
    
    
    
    def initialize_loss_decoder_class(self, circuit, dx: int, dy: int):
        ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in circuit.logical_qubits[i].measure_qubits]
        data_qubits = [qubit for i in range(self.num_logicals) for qubit in circuit.logical_qubits[i].data_qubits]
        
        MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params, 
                                                dx = dx, dy = dy, loss_detection_method_str=self.loss_detection_method_str,
                                                ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                cycles=self.cycles, printing=False, loss_detection_freq = self.loss_detection_freq,
                                                output_dir = self.output_dir, decoder_type=self.loss_decoder_type,
                                                save_data_during_sim=self.save_data_during_sim, n_r=self.n_r, circuit_type=self.circuit_type,
                                                use_independent_decoder=self.use_independent_decoder, first_comb_weight=self.first_comb_weight,
                                                use_independent_and_first_comb_decoder=self.use_independent_and_first_comb_decoder)
        
        MLE_Loss_Decoder_class.circuit = circuit
        
        # if self.loss_detection_method_str == 'SWAP':
        #     loss_detection_class = self.heralded_circuit(circuit=circuit, biased_erasure=self.is_erasure_biased, bias_preserving_gates=self.bias_preserving_gates,
        #                                                             basis = self.logical_basis, erasure_ratio = self.erasure_ratio, 
        #                                                             phys_error = None, ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
        #                                                             SSR=self.SSR, cycles=self.cycles, printing=False, loss_detection_freq = self.loss_detection_freq)
        #     SWAP_circuit = loss_detection_class.transfer_circuit_into_SWAP_circuit(circuit, movement_errors=True)
        #     if self.printing:
        #         print(f"Logical circuit after implementing SWAP is: \n {SWAP_circuit}\n")
        #     loss_detection_class.SWAP_circuit = SWAP_circuit
        #     MLE_Loss_Decoder_class.circuit = SWAP_circuit
        
        # elif self.loss_detection_method_str in ['FREE', 'MBQC', 'None']:
        #     MLE_Loss_Decoder_class.circuit = circuit
        
        # if self.printing:
        #     print(f"Logical circuit that will be used: \n{MLE_Loss_Decoder_class.circuit}")
        #     print(f"len potential_lost_qubits: {len(MLE_Loss_Decoder_class.circuit.potential_lost_qubits)}")
            # print(f"potential_lost_qubits: {list(MLE_Loss_Decoder_class.circuit.potential_lost_qubits)}")
            # print(f"loss_probabilities: {MLE_Loss_Decoder_class.circuit.loss_probabilities}")
            
        return MLE_Loss_Decoder_class
        
    
    def generate_logical_circuit(self, dx: int, dy: int, noise_params={}):
        LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None, noise_params=noise_params)
        return LogicalCircuit
        
    
    def sampling_with_loss(self, num_shots: int, dx: int, dy: int, noise_params={}, circuit = None):
        """This function samples measurements and detection events including loss.
        """
        
        if circuit is None:
            if self.circuit_type == 'lattice_surgery' and self.logical_basis == 'ZZ':
                PartialCircuit, LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None, noise_params=noise_params)
            else:
                LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None, noise_params=noise_params)
        else:
            LogicalCircuit = circuit
        ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].measure_qubits]
        data_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].data_qubits]
        
        
        MLE_Loss_Decoder_class = self.initialize_loss_decoder_class(LogicalCircuit, dx, dy)

        loss_detection_events_all_shots = np.random.rand(num_shots, len(LogicalCircuit.potential_lost_qubits)) < LogicalCircuit.loss_probabilities # sample losses according to the circuit

        measurement_events_all_shots = []
        detection_events_all_shots = []
        observable_flips_all_shots = []

        if np.any(loss_detection_events_all_shots): # we have loss in this simulation:
            MLE_Loss_Decoder_class.initialize_loss_decoder_for_sampling_only()
            for loss_shot in range(num_shots):
                # print(loss_shot, end = " ")
                loss_detection_events = loss_detection_events_all_shots[loss_shot]
                
                experimental_circuit = MLE_Loss_Decoder_class.generate_experimental_circuit(loss_detection_events=loss_detection_events)
                
                measurement_sampler = experimental_circuit.compile_sampler()
                measurement_events = measurement_sampler.sample(shots=1)
                
                detection_events, observable_flip = LogicalCircuit.compile_m2d_converter().convert(measurements=measurement_events, separate_observables=True)
                
                detection_events_all_shots.extend(detection_events)
                measurement_events_all_shots.extend(measurement_events)
                observable_flips_all_shots.extend(observable_flip[0])
                
                # if self.printing:
                #     print(f"sampling for the following loss pattern: {np.where(loss_detection_events)[0]}")
                #     print(f"{MLE_Loss_Decoder_class.real_losses_by_instruction_ix}")
                    # final_loss_circuit = MLE_Loss_Decoder_class.observables_to_detectors(experimental_circuit)
                    # final_dem = final_loss_circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=True)        
                    # print(f"dem for the lossy circuit = {final_dem}")
                    
            measurement_events_all_shots = np.array(measurement_events_all_shots).astype(int)
            measurement_events_all_shots = convert_qubit_losses_into_measurement_events(LogicalCircuit, ancilla_qubits, data_qubits, loss_detection_events_all_shots, measurement_events_all_shots) # mark 2 if we lost the qubit before its measurement
            detection_events_all_shots = np.array(detection_events_all_shots).astype(int)    
            observable_flips_all_shots = np.array(observable_flips_all_shots).astype(int)
            observable_flips_all_shots = observable_flips_all_shots.reshape(-1)
            
        
        else: # no loss! no need to loop over all shots
            experimental_circuit = MLE_Loss_Decoder_class.circuit

            measurement_sampler = experimental_circuit.compile_sampler()
            measurement_events_all_shots = measurement_sampler.sample(shots=num_shots)
            
            detection_events_all_shots, observable_flips_all_shots = experimental_circuit.compile_m2d_converter().convert(measurements=measurement_events_all_shots, separate_observables=True)
            measurement_events_all_shots = measurement_events_all_shots.astype(int)
            observable_flips_all_shots = observable_flips_all_shots.astype(int)
                
        return measurement_events_all_shots, detection_events_all_shots, observable_flips_all_shots, LogicalCircuit

    
    
    def generate_dems_loss_decoders(self, measurement_events, num_shots, type, MLE_Loss_Decoder_class):
        
        dems_list = []
        probs_lists = []
        hyperedges_matrix_list = []
        observables_errors_interactions_lists = []
        
        if self.printing:
            print("Shot:", end = " ")
        loss_start_time = time.time()
        for shot in range(num_shots):
            if shot % 100 == 0:
                print(shot, end = " ")
            measurement_event = measurement_events[shot] # change it to measurements
            
            return_matrix_with_observables = False if self.decoder == 'MLE' else True
            if type == 'independent': # Delayed erasure decoder, counting lifecycles and Clifford propagation. can also be comb decoder here!
                final_dem_hyperedges_matrix, observables_errors_interactions = MLE_Loss_Decoder_class.generate_dem_loss_mle_experiment(measurement_event, return_matrix_with_observables=return_matrix_with_observables) # final_dem_hyperedges_matrix doesn't contain observables, only detectors               
            elif type == 'ssr': # Loss decoder with only superchecks according to SSR
                final_dem_hyperedges_matrix, observables_errors_interactions  = MLE_Loss_Decoder_class.generate_dem_loss_mle_experiment_only_superchecks(measurement_event, return_matrix_with_observables)

            observables_errors_interactions_lists.append(observables_errors_interactions)
            if self.decoder == "MLE":
                hyperedges_matrix_list.append(final_dem_hyperedges_matrix)
            else:
                raise NotImplementedError # gives bad results
                final_dem = MLE_Loss_Decoder_class.from_hyperedges_matrix_into_stim_dem(final_dem_hyperedges_matrix) # convert into stim format. 
                #TODO: bug - fix it! here final_dem_hyperedges_matrix doesn't contain observables so the DEM will not contain them.
                dems_list.append(final_dem)
                
        if self.decoder == "MLE":
            dems_list, probs_lists = MLE_Loss_Decoder_class.convert_multiple_hyperedge_matrices_into_binary_new(hyperedges_matrix_list)

        return dems_list, probs_lists, observables_errors_interactions_lists
    
    
    def normalize_detection_events(self, detection_events, detection_events_signs):
        # add normalization step of detection events: - debug - dont normalize the detectors because we have the correct circuit!
        if type(detection_events_signs) != type(None):
            print('Using detection events signs!')
            detection_events_int = detection_events.astype(np.int32)
            detection_events_flipped = np.where(detection_events_signs == -1,  1 - detection_events_int, detection_events_int) # change ~detection_events_int to 1 - detection_events_int
            detection_events = detection_events_flipped.astype(np.bool_)
        return detection_events
    
    def decode_circuit(self, LogicalCircuit, num_shots: int, dx: int, dy: int, measurement_events: np.ndarray, detection_events_signs: np.ndarray, 
                                        noise_params={}, logical_gap = False):
        
        MLE_Loss_Decoder_class = self.initialize_loss_decoder_class(LogicalCircuit, dx, dy)

        if 2 in measurement_events and (not self.dont_use_loss_decoder):
            
            start_time = time.time()
            MLE_Loss_Decoder_class.initialize_loss_decoder() # this part can be improved to be a bit faster
            if self.printing:
                print(f'Decoder initialized, it took {time.time() - start_time:.2f}s for everything')      
            
            type = 'independent' if self.use_independent_decoder else 'ssr'
            dems_list, probs_lists, observables_errors_interactions_lists = self.generate_dems_loss_decoders(measurement_events, num_shots, type, MLE_Loss_Decoder_class)
            
            detection_events, observable_flips = self.get_detection_events_from_measurements(measurement_events, LogicalCircuit)
            detection_events = self.normalize_detection_events(detection_events, detection_events_signs) ### ADDED BACK IN 2024/08/20 BY SG ###

            if self.printing:
                print(f"Loss decoder is done! Now starting to decode with {self.decoder}")
            
            if logical_gap:
                assert self.decoder == 'MLE'
                if self.decoder == "MLE":
                    start_time = time.time()
                    # print(f"observables_errors_interactions_lists.shape = {len(observables_errors_interactions_lists)} and shape of first element: {len(observables_errors_interactions_lists[0])}")
                    predictions, log_probabilities = qec.correlated_decoders.mle_loss.logical_gap_gurobi_with_dem_loss_fast(dems_list=dems_list,
                                                                                                    probs_lists=probs_lists,
                                                                                                    detector_shots=detection_events,
                                                                                                    observables_lists=observables_errors_interactions_lists)
                    print(f'MLE decoder took {time.time() - start_time:.6f}s.')
                
                if self.printing:
                    num_errors = np.sum(np.logical_xor(observable_flips, predictions))
                    print(f"for dx = {dx}, dy = {dy}, {self.cycles} cycles, {num_shots} shots, we had {num_errors} errors (logical error = {(num_errors/num_shots):.1e})")
                return predictions, log_probabilities, observable_flips, dems_list
                
            else:
                if self.decoder == "MLE":
                    start_time = time.time()
                    predictions = qec.correlated_decoders.mle_loss.decode_gurobi_with_dem_loss_fast(dems_list=dems_list, probs_lists = probs_lists, detector_shots = detection_events, observables_lists=observables_errors_interactions_lists)   
                    if self.printing:
                        print(f'MLE decoder took {time.time() - start_time:.6f}s.')

                else:
                    raise NotImplementedError # gives bad results
                    start_time = time.time()
                    predictions = []
                    for (d, detection_event) in enumerate(detection_events):
                        detector_error_model = dems_list[d]
                        matching = pymatching.Matching.from_detector_error_model(detector_error_model)
                        # prediction = matching.decode_batch(detection_event)
                        prediction = matching.decode(detection_event) # GB change Oct22: uncomment this
                        # predictions.append(prediction[0][0])
                        predictions.append(prediction[0])
                    if self.printing:
                        print(f'MWPM decoder took {time.time() - start_time:.6f}s.')
            
                num_errors = np.sum(np.logical_xor(observable_flips, predictions))
                if self.printing:
                    print(f"for dx = {dx}, dy = {dy}, {self.cycles} cycles, {num_shots} shots, we had {num_errors} errors (logical error = {(num_errors/num_shots):.1e})")
                return predictions, observable_flips, dems_list
        

        
        else: # no loss in the experiment or not using the delayed-erasure decoder --> regular decoding, without delayed erasure decoder
            detection_events, observable_flips = self.get_detection_events_from_measurements(measurement_events, LogicalCircuit)
            detection_events = self.normalize_detection_events(detection_events, detection_events_signs) ### ADDED BACK IN 2024/08/20 BY SG ###
            
            if logical_gap:
                assert self.decoder == 'MLE'
                raise NotImplementedError
            
            else:
                if self.decoder == "MLE":
                    if self.circuit_type == 'memory_wrong' or (self.circuit_type == 'lattice_surgery' and self.logical_basis == 'ZZ'): # we need to first convert observables to detectors, get dem, and reconvert back.
                        MLE_Loss_Decoder_class.set_up_Pauli_DEM()
                        detector_error_model = MLE_Loss_Decoder_class.from_hyperedges_matrix_into_stim_dem(MLE_Loss_Decoder_class.Pauli_DEM_matrix)
                    else:
                        detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
                        # detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=True) # debug
                        print(detector_error_model) # debug
                    predictions = qec.correlated_decoders.mle.decode_gurobi_with_dem(dem=detector_error_model, detector_shots = detection_events)
                else:
                    detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True) 
                    predictions = sinter.predict_observables(
                        dem=detector_error_model,
                        dets=detection_events,
                        decoder='pymatching',
                    )

                if self.printing:
                    num_errors = np.sum(np.logical_xor(observable_flips, predictions))
                    print(f"for dx = {dx}, dy = {dy}, {self.cycles} cycles, {num_shots} shots, we had {num_errors} errors (logical error = {(num_errors/num_shots):.1e})")
                return predictions, observable_flips, detector_error_model


    def count_logical_errors_lattice_surgery(self, num_shots: int, dx: int, dy: int, measurement_events: np.ndarray, detection_events_signs: np.ndarray, 
                                        noise_params, logical_gap):
        
        PartialCircuit, FullCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None, noise_params=noise_params)
        
        if self.printing:
            print(f"circuit used for lattice surgery simulations are: \n{LogicalCircuit}")
            print(f"PartialCircuit: \n{PartialCircuit}")
            print(f"FullCircuit: \n{FullCircuit}")
        
        
        
        PartialMeasurements = measurement_events[:, :PartialCircuit.num_measurements]
        PartialDetection_events_signs = detection_events_signs[:PartialCircuit.num_detectors] if detection_events_signs is not None else None
        
        if logical_gap:
            self.Meta_params['circuit_type'] = 'lattice_surgery_full'
            Full_predictions, Full_log_probabilities, Full_observable_flips, Full_dems_list = self.decode_circuit(LogicalCircuit=FullCircuit, num_shots = num_shots, dx = dx, dy = dy,
                                                        measurement_events = measurement_events, detection_events_signs=detection_events_signs,
                                                        noise_params=noise_params)
            self.Meta_params['circuit_type'] = 'lattice_surgery_partial'

            Partial_predictions, Partial_log_probabilities, Partial_observable_flips, Partial_dems_list = self.decode_circuit(LogicalCircuit=PartialCircuit, num_shots = num_shots, dx = dx, dy = dy,
                                                        measurement_events = PartialMeasurements, detection_events_signs=PartialDetection_events_signs,
                                                        noise_params=noise_params)
            self.Meta_params['circuit_type'] = 'lattice_surgery'

            return Partial_predictions, Partial_log_probabilities, Partial_observable_flips, Partial_dems_list, Full_predictions, Full_log_probabilities, Full_observable_flips, Full_dems_list
            
        else:
            self.Meta_params['circuit_type'] = 'lattice_surgery_full'
            Full_predictions, Full_observable_flips, Full_dems_list = self.decode_circuit(LogicalCircuit=FullCircuit, num_shots = num_shots, dx = dx, dy = dy,
                                                        measurement_events = measurement_events, detection_events_signs=detection_events_signs,
                                                        noise_params=noise_params)
            self.Meta_params['circuit_type'] = 'lattice_surgery_partial'
            Partial_predictions, Partial_observable_flips, Partial_dems_list = self.decode_circuit(LogicalCircuit=PartialCircuit, num_shots = num_shots, dx = dx, dy = dy,
                                                        measurement_events = PartialMeasurements, detection_events_signs=PartialDetection_events_signs,
                                                        noise_params=noise_params)
            self.Meta_params['circuit_type'] = 'lattice_surgery'

            

            return Partial_predictions, Partial_observable_flips, Partial_dems_list, Full_predictions, Full_observable_flips, Full_dems_list
            
    def count_logical_errors_experiment(self, num_shots: int, dx: int, dy: int, measurement_events: np.ndarray, detection_events_signs: np.ndarray, 
                                        noise_params, logical_gap):
        """This function decodes the loss information using mle. 
        Given heralded losses upon measurements, there are multiple potential loss events (with some probability) in the circuit.
        We use the MLE approximate decoding - for each potential loss event we create a DEM and connect all to generate the final DEM. We use MLE to decode given the ginal DEM and the experimental measurements.
        Input: Meta_params, dx, dy, num shots, experimental data: detector shots.
        Output: final DEM, corrections, num errors.
        
        Meta_params = {'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': 'X', 'bias_preserving_gates': 'False', 
                'noise': 'atom_array', 'is_erasure_biased': 'False', 'LD_freq': '1', 'LD_method': 'SWAP', 'SSR': 'True', 'cycles': '2', 'ordering': 'bad', 'decoder': 'MLE',
                'circuit_type': 'memory', 'printing': 'False', 'num_logicals': '1'}
        ordering: bad or fowler (good)
        decoder: MLE or MWPM
        
        """
        # Step 1 - generate the experimental circuit in our simulation:
        if self.circuit_type == 'lattice_surgery' and self.logical_basis == 'ZZ':
            return self.count_logical_errors_lattice_surgery(num_shots = num_shots, dx = dx, dy = dy,
                                                        measurement_events = measurement_events, detection_events_signs=detection_events_signs,
                                                        noise_params=noise_params, logical_gap=logical_gap)
        else:
            LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None, noise_params=noise_params)
                
            # LogicalCircuit.logical_qubits[0].visualize_code()
        
            if self.printing:
                print(f"Circuit used for simulation is: \n{LogicalCircuit}")
            
            return self.decode_circuit(LogicalCircuit=LogicalCircuit, num_shots = num_shots, dx = dx, dy = dy,
                                                            measurement_events = measurement_events, detection_events_signs=detection_events_signs,
                                                            noise_params=noise_params, logical_gap=logical_gap)





    def make_dem_SSR_experiment(self, num_shots: int, dx: int, dy: int, measurement_events: np.ndarray, detection_events_signs: np.ndarray):
        """This function decodes the loss information using mle. 
        Given heralded losses upon measurements, there are multiple potential loss events (with some probability) in the circuit.
        We use the MLE approximate decoding - for each potential loss event we create a DEM and connect all to generate the final DEM. We use MLE to decode given the ginal DEM and the experimental measurements.
        Input: Meta_params, dx, dy, num shots, experimental data: detector shots.
        Output: final DEM, corrections, num errors.
        
        Meta_params = {'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': 'X', 'bias_preserving_gates': 'False', 
                'noise': 'atom_array', 'is_erasure_biased': 'False', 'LD_freq': '1', 'LD_method': 'SWAP', 'SSR': 'True', 'cycles': '2', 'ordering': 'bad', 'decoder': 'MLE',
                'circuit_type': 'memory', 'printing': 'False', 'num_logicals': '1'}
        ordering: bad or fowler (good)
        decoder: MLE or MWPM
        """
        
        # Step 1 - generate the experimental circuit in our simulation:
        LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None)
        
        ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].measure_qubits]
        data_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].data_qubits]
        


        MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params, 
                                                dx = dx, dy = dy, loss_detection_method_str=self.loss_detection_method_str,
                                                ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                cycles=self.cycles, printing=False, loss_detection_freq = self.loss_detection_freq,
                                                output_dir = self.output_dir, decoder_type=self.loss_decoder_type,
                                                save_data_during_sim=self.save_data_during_sim, n_r=self.n_r, circuit_type=self.circuit_type)
        

        MLE_Loss_Decoder_class.circuit = LogicalCircuit
        MLE_Loss_Decoder_class.generate_measurement_ix_to_instruction_ix_map() # mapping between measurement events to instruction ix
        
        # if self.printing:
        #     print(f"Logical circuit that will be used: \n{MLE_Loss_Decoder_class.circuit}")
            
        if 2 in measurement_events:
            loss_sampling = 1 # how many times we will use the same loss sampling result
            num_loss_shots = math.ceil(num_shots / loss_sampling)
            num_shots = num_loss_shots * loss_sampling
            

            # Loss decoding - creating DEMs:
            dems_list = []
            valid_measurement_events = []
            # probs_lists = []
            # observables_errors_interactions_lists = []
            print("Shot:", end = " ")
            for shot in range(num_loss_shots):
                print(shot, end = " ")
                measurement_event = measurement_events[shot] # change it to measurements
                
                # start_time = time.time()
                stim_dem_supercheck, dont_use_shot = MLE_Loss_Decoder_class.make_stim_dem_supercheck_given_loss_only_determ_observables(measurement_event)
                if not dont_use_shot:
                    dems_list.append(stim_dem_supercheck)
                    valid_measurement_events.append(measurement_event)

            print(f"\n left {len(valid_measurement_events)} valid shots out of {len(measurement_events)} shots!")
            valid_measurement_events = np.array(valid_measurement_events)
            
            detection_events, observable_flips = self.get_detection_events_from_measurements(valid_measurement_events, LogicalCircuit)
            detection_events = self.normalize_detection_events(detection_events, detection_events_signs)

            return dems_list, detection_events, observable_flips
        
        
        else: # regular decoding, without delayed erasure decoder
            detection_events, observable_flips = self.get_detection_events_from_measurements(measurement_events, LogicalCircuit)
            detection_events = self.normalize_detection_events(detection_events, detection_events_signs)
            return dems_list, detection_events, observable_flips
        
        
    def get_detection_events(self, dx: int, dy: int, measurement_events: np.ndarray, noise_params, circuit=None):
        """This function generates the circuit, takes the measurement events and generates the detection events.
        """
        if circuit is None:
            if self.circuit_type == 'lattice_surgery' and self.logical_basis == 'ZZ':
                PartialCircuit, LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None, noise_params=noise_params)
            else:
                LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None, noise_params=noise_params)
        else:
            LogicalCircuit = circuit
            
        # LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None)
        # MLE_Loss_Decoder_class = self.initialize_loss_decoder_class(LogicalCircuit, dx, dy)
        # MLE_Loss_Decoder_class.generate_measurement_ix_to_instruction_ix_map() # mapping between measurement events to instruction ix
        detection_events, observable_flips = self.get_detection_events_from_measurements(measurement_events, LogicalCircuit)

        return detection_events, observable_flips
    
    
        
    # def comb_preprocessing(self, dx: int, dy: int, num_of_losses: int, batch_index: int):
    #     # Generate the logical circuit
    #     LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None)
        
    #     # Identify qubits
    #     ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].measure_qubits]
    #     data_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].data_qubits]

    #     # Initialize the MLE Loss Decoder
    #     MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params, 
    #                                             dx=dx, dy=dy, loss_detection_method_str=self.loss_detection_method_str,
    #                                             ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
    #                                             cycles=self.cycles, printing=self.printing, loss_detection_freq=self.loss_detection_freq,
    #                                             output_dir=self.output_dir, decoder_type=self.loss_decoder_type,
    #                                             save_data_during_sim=self.save_data_during_sim, n_r=self.n_r, circuit_type=self.circuit_type)

    #     MLE_Loss_Decoder_class.circuit = LogicalCircuit
    #     MLE_Loss_Decoder_class.initialize_loss_decoder_for_sampling_only() 
    #     all_potential_loss_qubits_indices = MLE_Loss_Decoder_class.get_all_potential_loss_qubits()
    #     loss_decoder_files_dir = f"{self.output_dir}/loss_circuits/{MLE_Loss_Decoder_class.create_loss_file_name(self.Meta_params, self.bloch_point_params)}/dx_{dx}__dy_{dy}__c_{self.cycles}"
    #     full_filename_dems = f'{loss_decoder_files_dir}/circuit_dems_{num_of_losses}_losses.pickle'
        
    #     # Step 1: Get all combinations with num_of_losses losses
    #     all_combinations = list(itertools.combinations(all_potential_loss_qubits_indices, num_of_losses))
    #     total_combinations = len(all_combinations)
    #     batch_size = max(1, int(total_combinations * 0.001))
    #     num_batches = (total_combinations + batch_size - 1) // batch_size

    #     print(f"Num total combinations: {total_combinations}")
    #     print(f"Batch size: {batch_size}")
    #     print(f"Number of batches: {num_batches}")
    #     print(f"For dx={dx}, dy={dy}, num of losses = {num_of_losses}, we got {total_combinations} combinations to process.")

    #     # Determine the range for the specific batch
    #     batch_start = batch_index * batch_size
    #     batch_end = min(batch_start + batch_size, total_combinations)

    #     # Get the combinations for this specific batch
    #     batch_combinations = all_combinations[batch_start:batch_end]

    #     # Call the function to analyze this specific batch
    #     MLE_Loss_Decoder_class.preprocess_circuit_comb_specific_batch(batches_dir=f'{loss_decoder_files_dir}/batches_{num_of_losses}',
    #                                                                 batch_combinations=batch_combinations,
    #                                                                 batch_start=batch_start,
    #                                                                 batch_end=batch_end)



    # def count_logical_errors_preselection(self, num_shots: int, dx=None, dy=None, phys_error=0, debugging=False, cycles=None):
    #     decoder_type = 'independent'

    #     entangling_gate_error_rate, entangling_gate_loss_rate = self.noise(phys_error, self.bias_ratio)
    #     def steane(dx, dy, ancilla1_for_preselection=False, ancilla2_for_preselection=False):
    #         return Steane_QEC_circuit(dx=dx, dy=dy, code=self.code, Steane_type=self.Steane_type, QEC_cycles=cycles, entangling_gate_error_rate=entangling_gate_error_rate,
    #                                 entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio=self.erasure_ratio, logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates,
    #                                 loss_detection_on_all_qubits=True, atom_array_sim=self.atom_array_sim, ancilla1_for_preselection=ancilla1_for_preselection, ancilla2_for_preselection=ancilla2_for_preselection)

    #     lc = steane(dx, dy)
        
    #     ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in lc.logical_qubits[i].measure_qubits]
    #     data_qubits = [qubit for i in range(self.num_logicals) for qubit in lc.logical_qubits[i].data_qubits]
        
    #     loss_sampling = 1 if self.erasure_ratio > 0 else  num_shots # how many times we will use the same loss sampling result
    #     num_loss_shots = math.ceil(num_shots / loss_sampling)
    #     num_shots = num_loss_shots * loss_sampling
    #     loss_detection_events_all_shots = np.random.rand(num_loss_shots, len(lc.potential_lost_qubits)) < lc.loss_probabilities

        
    #     # get SWAP circuit for each circuit ('regular','1','2'):
    #     loss_detection_class = self.heralded_circuit(circuit=lc, biased_erasure=self.is_erasure_biased, bias_preserving_gates=self.bias_preserving_gates,
    #                                                                 basis = self.logical_basis, erasure_ratio = self.erasure_ratio, 
    #                                                                 phys_error = phys_error, ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
    #                                                                 SSR=self.SSR, cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq)
    #     if self.loss_detection_method_str == 'SWAP':
    #         SWAP_circuit_regular = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc)
    #     else:
    #         SWAP_circuit_regular = lc
            
    #     lc_for_preselection1 = steane(dx, dy, ancilla1_for_preselection=True)
    #     loss_detection_class.logical_circuit = lc_for_preselection1
    #     if self.loss_detection_method_str == 'SWAP':
    #         SWAP_circuit_preselection1 = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc_for_preselection1)
    #     else:
    #         SWAP_circuit_preselection1 = lc_for_preselection1

    #     lc_for_preselection2 = steane(dx, dy, ancilla2_for_preselection=True)
    #     loss_detection_class.logical_circuit = lc_for_preselection2
    #     if self.loss_detection_method_str == 'SWAP':
    #         SWAP_circuit_preselection2 = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc_for_preselection2)
    #     else:
    #         SWAP_circuit_preselection2 = lc_for_preselection2

    #     # get DEMs for each circuit:
    #     MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params, 
    #                                             dx = dx, dy = dy, loss_detection_method_str=self.loss_detection_method_str,
    #                                             ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
    #                                             cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq, 
    #                                             first_comb_weight=self.first_comb_weight,
    #                                             output_dir = self.output_dir, decoder_type = decoder_type)
    #     #MLE_Loss_Decoder_class.initialize_loss_decoder()
    #     SWAP_circuits = {'regular': SWAP_circuit_regular, '1': SWAP_circuit_preselection1, '2': SWAP_circuit_preselection2}
    #     #DEMs = {'regular': [], '1': [], '2': []}

    #     corrections = []
    #     observables = []
    #     probabilities1 = np.zeros((0, 2))
    #     probabilities2 = np.zeros((0, 2))
    #     for circuit_name in ['regular','1','2']:
    #         print(circuit_name)
            
    #         if self.erasure_ratio > 0:
    #             MLE_Loss_Decoder_class.circuit = SWAP_circuits[circuit_name]
    #             MLE_Loss_Decoder_class.initialize_loss_decoder()
    #             for shot in range(num_loss_shots):
    #                 loss_detection_events = loss_detection_events_all_shots[shot]
    #                 experimental_circuit, dem = MLE_Loss_Decoder_class.decode_loss_MLE(loss_detection_events)
    #                 if circuit_name == 'regular':
    #                     # Sample MEASUREMENTS from experimental_circuit
    #                     sampler = experimental_circuit.compile_detector_sampler()
    #                     detection_events, observable_flips = sampler.sample(loss_sampling, separate_observables=True)
    #                     observables.extend(observable_flips)
    #                     corrections.extend(qec.correlated_decoders.mle.decode_gurobi_with_dem(dem, detection_events))
    #                 elif circuit_name == '1':
    #                     # _, prob1 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
    #                     _, prob1 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events)
    #                     probabilities1 = np.concatenate((probabilities1, prob1), axis=0)
    #                 elif circuit_name == '2':
    #                     # _, prob2 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
    #                     _, prob2 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events)
    #                     probabilities2 = np.concatenate((probabilities2, prob2), axis=0)
    #                 else:
    #                     assert True is False
                        
    #         else: # loss ratio = 0, execute all shot together
    #             dem = SWAP_circuits[circuit_name].detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
    #             if circuit_name == 'regular':
    #                 sampler = SWAP_circuits[circuit_name].compile_detector_sampler()
    #                 detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    #                 observables.extend(observable_flips)
    #                 corrections.extend(qec.correlated_decoders.mle.decode_gurobi_with_dem(dem, detection_events))
    #             elif circuit_name == '1':
    #                 _, prob1 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
    #                 probabilities1 = np.concatenate((probabilities1, prob1), axis=0)
    #             elif circuit_name == '2':
    #                 _, prob2 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
    #                 probabilities2 = np.concatenate((probabilities2, prob2), axis=0)
    #             else:
    #                 assert True is False
            
    #     return corrections, observables, probabilities1, probabilities2




    # def count_logical_errors(self, LogicalCircuit, num_shots: int, dx=None, dy=None, phys_error=0, debugging=False, cycles=None):
    #     """This function decodes the loss information using mle. 
    #     Given heralded losses upon measurements, there are multiple potential loss events (with some probability) in the circuit.
    #     There are 2 options:
    #     1. MLE approximate decoding - for each potential loss event we create a DEM and connect all to generate the final DEM. We use MLE to decode given the ginal DEM and the experimental measurements.
    #     2. Accurate MLE - for each potential loss event i we decode with Gorubi to get P_i, and take argmax(P_i) to decode.
    #     """
    #     # decoder_type = 'independent'
        
    #     if self.printing:
    #         print(f"Starting the decoding!")
    #         start_time = time.time()

    #     if self.architecture == "MBQC":
    #         xzzx_instance = XZZX()
    #         data_qubits, ancilla_qubits = xzzx_instance.get_data_ancilla_indices(dx=dx, dy=dy, cycles=cycles, architecture="MBQC")
    #     else:
    #         ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].measure_qubits]
    #         data_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].data_qubits]
        
        
    #     sampler = LogicalCircuit.compile_detector_sampler()
    #     loss_sampling = 1 if self.erasure_ratio > 0 else  num_shots # how many times we will use the same loss sampling result
    #     num_loss_shots = math.ceil(num_shots / loss_sampling)
    #     num_shots = num_loss_shots * loss_sampling
    #     loss_detection_events_all_shots = np.random.rand(num_loss_shots, len(LogicalCircuit.potential_lost_qubits)) < LogicalCircuit.loss_probabilities
        
        
    #     # LogicalCircuit.logical_qubits[1].visualize_code()
    #     # print(LogicalCircuit.loss_probabilities)
    #     loss_detection_class = self.heralded_circuit(circuit=LogicalCircuit, biased_erasure=self.is_erasure_biased, bias_preserving_gates=self.bias_preserving_gates,
    #                                                                 basis = self.logical_basis, erasure_ratio = self.erasure_ratio, 
    #                                                                 phys_error = phys_error, ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
    #                                                                 SSR=self.SSR, cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq)
        
        
    #     MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params,
    #                                             dx = dx, dy = dy, loss_detection_method_str=self.loss_detection_method_str,
    #                                             ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
    #                                             cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq, 
    #                                             first_comb_weight=self.first_comb_weight,
    #                                             output_dir = self.output_dir, decoder_type = self.loss_decoder_type,
    #                                             save_data_during_sim=self.save_data_during_sim, n_r=self.n_r, circuit_type=self.circuit_type)
        
    #     if self.loss_detection_method_str == 'SWAP':
    #         SWAP_circuit = loss_detection_class.transfer_circuit_into_SWAP_circuit(LogicalCircuit)
    #         if self.printing:
    #             print(f"Logical circuit after implementing SWAP is: \n {SWAP_circuit}\n")
    #         # loss_detection_class.SWAP_circuit = SWAP_circuit
    #         MLE_Loss_Decoder_class.circuit = SWAP_circuit
        
    #     elif self.loss_detection_method_str in ['FREE', 'MBQC', 'None']:
    #         MLE_Loss_Decoder_class.circuit = LogicalCircuit
            
        
        
    #     if self.printing:
    #         end_time = time.time()
    #         print(f"Initialization of loss decoders (and building SWAP circuit if needed) took {end_time - start_time} sec.")
    #         start_time = time.time()
                
    #     if self.erasure_ratio > 0:
            
    #         MLE_Loss_Decoder_class.initialize_loss_decoder()
            
    #         if self.printing:
    #             end_time = time.time()
    #             print(f"Building the Pauli DEM and loading all independent loss DEMs took {end_time - start_time} sec. Starting the loss decoding!")
    #             start_time = time.time()
            
    #         num_errors = 0
    #         dems_list = []
    #         detection_events_list = []
    #         observable_flips_list = []
    #         start_time_all_shots = time.time()
    #         for shot in range(num_loss_shots):
    #             loss_detection_events = loss_detection_events_all_shots[shot]
                
    #             experimental_circuit, detector_error_model = MLE_Loss_Decoder_class.decode_loss_MLE(loss_detection_events)

    #             if self.printing:
    #                 # print(f"\n Loss detection events: {loss_detection_events}")
    #                 # print(f"\n Potential lost qubits: {LogicalCircuit.potential_lost_qubits} \n with loss probabilities: {LogicalCircuit.loss_probabilities}")
    #                 print("\n Experimental circuit (for measurements):")
    #                 print(experimental_circuit)
    #                 print("\n MLE DEM:")
    #                 print(detector_error_model)
                
    #             # Sample MEASUREMENTS from experimental_circuit
    #             sampler = experimental_circuit.compile_detector_sampler()
    #             detection_events, observable_flips = sampler.sample(loss_sampling, separate_observables=True)
                
    #             detection_events_list.append(detection_events)
    #             observable_flips_list.append(observable_flips)
                
    #             # Extract decoder configuration data from the circuit.
    #             if self.dont_use_loss_decoder:
    #                 no_loss_decoder_dem = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
    #                 dems_list.append(no_loss_decoder_dem)
    #             else:
    #                 dems_list.append(detector_error_model)
                
    #         if self.printing:
    #             end_time_all_shots = time.time()
    #             print(f"Building the MLE loss DEM and experimental circuits for all shots took {end_time_all_shots - start_time_all_shots} sec")
    #             start_time = time.time()
            
    #         if self.decoder == "MLE":
                
    #             detector_shots = np.array(detection_events_list)
    #             predictions = qec.correlated_decoders.mle_loss.decode_gurobi_with_dem_loss_theory(dems_list=dems_list, detector_shots = detector_shots)   
    #             if self.printing:
    #                 end_time = time.time()
    #                 print(f"Gurobi correlated mle decoding for all shots took {end_time - start_time} sec")
    #         else:
    #             predictions = []
    #             for (d, detection_events) in enumerate(detection_events_list):
    #                 detector_error_model = dems_list[d]
    #                 matching = pymatching.Matching.from_detector_error_model(detector_error_model)
    #                 prediction = matching.decode_batch(detection_events)
    #                 predictions.append(prediction[0][0])
    #             predictions = np.array(predictions)
    #             if self.printing:
    #                 end_time = time.time()
    #                 print(f"MWPM decoding for all shots took {end_time - start_time} sec")
                
    #         observable_flips = np.array(observable_flips_list)
    #         predictions_bool = predictions.astype(bool).squeeze()
    #         observable_flips_squeezed = observable_flips.squeeze()

    #         num_errors = np.sum(np.logical_xor(observable_flips_squeezed, predictions_bool))

    #         if self.circuit_type == 'random_alg':
    #             corrected_observables_correlated = np.logical_xor(observable_flips_squeezed, predictions_bool)
    #             num_errors = np.sum(1 - np.alltrue(1-corrected_observables_correlated, axis=1))

                                                    
    #     else: # no loss errors at all
    #         sampler = MLE_Loss_Decoder_class.circuit.compile_detector_sampler()
    #         detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
            

    #         if self.decoder == "MLE":
    #             detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
    #             prediction = qec.correlated_decoders.mle.decode_gurobi_with_dem(dem=detector_error_model, detector_shots = detection_events)
    #         else:
    #             detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True) 
    #             prediction = sinter.predict_observables(
    #                 dem=detector_error_model,
    #                 dets=detection_events,
    #                 decoder='pymatching',
    #             )

    #         num_errors = np.sum(np.logical_xor(observable_flips, prediction))
            
    #         if self.circuit_type == 'random_alg':
    #             corrected_observables_correlated = np.logical_xor(observable_flips, prediction)
    #             num_errors = np.sum(1 - np.alltrue(1-corrected_observables_correlated, axis=1))
                
    #     # logical_error = num_errors / num_shots
    #     return num_errors, num_shots
    



    # def get_logical_gap_experiment(self, num_shots: int, dx: int, dy: int, measurement_events: np.ndarray,
    #                                     detection_events_signs: np.ndarray,
    #                                     use_loss_decoding=True, use_independent_decoder=True,
    #                                     use_independent_and_first_comb_decoder=True,
    #                                     noise_params={}):
    #     assert self.decoder == 'MLE'
    #     """This function decodes the loss information using mle. 
    #     Given heralded losses upon measurements, there are multiple potential loss events (with some probability) in the circuit.
    #     We use the MLE approximate decoding - for each potential loss event we create a DEM and connect all to generate the final DEM. We use MLE to decode given the ginal DEM and the experimental measurements.
    #     Input: Meta_params, dx, dy, num shots, experimental data: detector shots.
    #     Output: final DEM, corrections, num errors.

    #     Meta_params = {'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': 'X', 'bias_preserving_gates': 'False', 
    #             'noise': 'atom_array', 'is_erasure_biased': 'False', 'LD_freq': '1', 'LD_method': 'SWAP', 'SSR': 'True', 'cycles': '2', 'ordering': 'bad', 'decoder': 'MLE',
    #             'circuit_type': 'memory', 'printing': 'False', 'num_logicals': '1'}
    #     ordering: bad or fowler (good)
    #     decoder: MLE or MWPM
    #     if simulate_data = True, simulating the data from the circuit, not using experimental real data.

    #     """

    #     # Step 1 - generate the experimental circuit in our simulation:
    #     start_time = time.time()
    #     LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None,
    #                                         noise_params=noise_params)  # real experimental circuit with the added pulses

    #     print(f"generating the Logical circuit took: {time.time() - start_time:.6f}s")
    #     # LogicalCircuit_no_pulses = self.generate_circuit(dx=dx, dy=dy, cycles=self.cycles, phys_err=None) # vanilla circuit, no pulses, regular surface code

    #     ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in
    #                     LogicalCircuit.logical_qubits[i].measure_qubits]
    #     data_qubits = [qubit for i in range(self.num_logicals) for qubit in
    #                     LogicalCircuit.logical_qubits[i].data_qubits]

    #     # LogicalCircuit.logical_qubits[0].visualize_code()

    #     MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params,
    #                                             bloch_point_params=self.bloch_point_params,
    #                                             dx=dx, dy=dy,
    #                                             loss_detection_method_str=self.loss_detection_method_str,
    #                                             ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
    #                                             cycles=self.cycles, printing=self.printing,
    #                                             loss_detection_freq=self.loss_detection_freq,
    #                                             output_dir=self.output_dir, decoder_type=self.loss_decoder_type,
    #                                             save_data_during_sim=self.save_data_during_sim, n_r=self.n_r,
    #                                             circuit_type=self.circuit_type,
    #                                             use_independent_decoder=use_independent_decoder,
    #                                             first_comb_weight=self.first_comb_weight,
    #                                             use_independent_and_first_comb_decoder=use_independent_and_first_comb_decoder)

    #     if self.loss_detection_method_str == 'SWAP':
    #         loss_detection_class = self.heralded_circuit(circuit=LogicalCircuit, biased_erasure=self.is_erasure_biased,
    #                                                     bias_preserving_gates=self.bias_preserving_gates,
    #                                                     basis=self.logical_basis, erasure_ratio=self.erasure_ratio,
    #                                                     phys_error=None, ancilla_qubits=ancilla_qubits,
    #                                                     data_qubits=data_qubits,
    #                                                     SSR=self.SSR, cycles=self.cycles, printing=self.printing,
    #                                                     loss_detection_freq=self.loss_detection_freq)
    #         SWAP_circuit = loss_detection_class.transfer_circuit_into_SWAP_circuit(LogicalCircuit)
    #         if self.printing:
    #             print(f"Logical circuit after implementing SWAP is: \n {SWAP_circuit}\n")
    #         loss_detection_class.SWAP_circuit = SWAP_circuit
    #         MLE_Loss_Decoder_class.circuit = SWAP_circuit

    #     elif self.loss_detection_method_str in ['FREE', 'MBQC', 'None']:
    #         MLE_Loss_Decoder_class.circuit = LogicalCircuit

    #     if self.printing:
    #         print(f"Logical circuit that will be used: \n{MLE_Loss_Decoder_class.circuit}")
    #         # print(f"Vanilla circuit without any pulses: \n{LogicalCircuit_no_pulses}")

    #     if simulate_data:  # SIMULATING OUR DATA!
    #         measurement_events, _, _, _ = self.sampling_with_loss(num_shots = num_shots, dx = dx, dy = dy, noise_params=noise_params, circuit=LogicalCircuit)
    #         # loss_detection_events_all_shots = np.random.rand(num_shots,
    #         #                                                 len(LogicalCircuit.potential_lost_qubits)) < LogicalCircuit.loss_probabilities  # sample losses according to the circuit
    #         # sampler = MLE_Loss_Decoder_class.circuit.compile_sampler()
    #         # measurement_events_all_shots = sampler.sample(shots=num_shots)  # sample without losses
    #         # np.save(f"{self.output_dir}/measurement_events_no_loss_pulses.npy", measurement_events_all_shots)
    #         # measurement_events_all_shots = measurement_events_all_shots.astype(int)
    #         # measurement_events = convert_qubit_losses_into_measurement_events(LogicalCircuit, ancilla_qubits,
    #         #                                                                 data_qubits,
    #         #                                                                 loss_detection_events_all_shots,
    #         #                                                                 measurement_events_all_shots)  # mark 2 if we lost the qubit before its measurement

    #     if 2 in measurement_events and use_loss_decoding:

    #         start_time = time.time()
    #         MLE_Loss_Decoder_class.initialize_loss_decoder()  # this part can be improved to be a bit faster
    #         print(f'Decoder initialized, it took {time.time() - start_time:.2f}s for everything')

    #         if use_independent_decoder:  # Delayed erasure decoder, counting lifecycles and Clifford propagation. can also be comb decoder here!

    #             # Loss decoding - creating DEMs:
    #             dems_list = []
    #             probs_lists = []
    #             hyperedges_matrix_list = []
    #             observables_errors_interactions_lists = []
    #             print("Shot:", end=" ")
    #             loss_start_time = time.time()
    #             for shot in range(num_shots):
    #                 if shot % 100 == 0:
    #                     print(shot, end=" ")
    #                 measurement_event = measurement_events[shot]  # change it to measurements

    #                 start_time = time.time()
    #                 return_matrix_with_observables = False if self.decoder == 'MLE' else True
    #                 final_dem_hyperedges_matrix, observables_errors_interactions = MLE_Loss_Decoder_class.generate_dem_loss_mle_experiment(
    #                     measurement_event,
    #                     return_matrix_with_observables=return_matrix_with_observables)  # final_dem_hyperedges_matrix doesn't contain observables, only detectors
    #                 # print(f'Total loss decoder time per shot is {time.time() - start_time:.4f}s.')

                
    #                 observables_errors_interactions_lists.append(observables_errors_interactions)
    #                 if self.decoder == "MLE":
    #                     start_time = time.time()
    #                     # final_dem_hyperedges_matrix_01, probs_list = MLE_Loss_Decoder_class.convert_hyperedge_matrix_into_binary(hyperedges_matrix = final_dem_hyperedges_matrix)
    #                     # print(f'convert hyperedge matrix into binary per shot took {time.time() - start_time:.2f}s.')
    #                     # dems_list.append(final_dem_hyperedges_matrix_01)
    #                     # probs_lists.append(probs_list)
    #                     hyperedges_matrix_list.append(final_dem_hyperedges_matrix)
    #                 else:
    #                     final_dem = MLE_Loss_Decoder_class.from_hyperedges_matrix_into_stim_dem(
    #                         final_dem_hyperedges_matrix)  # convert into stim format. #TODO: bug - fix it! here final_dem_hyperedges_matrix doesn't contain observables so the DEM will not contain them.
    #                     dems_list.append(final_dem)

    #             print(f'\nTotal loss decoder time for all shots {time.time() - loss_start_time:.4f} sec.')
    #             start_time = time.time()
    #             dems_list, probs_lists = MLE_Loss_Decoder_class.convert_multiple_hyperedge_matrices_into_binary_new(
    #                 hyperedges_matrix_list)
    #             print(f'new method: convert ALL hyperedge matrix into binary took {time.time() - start_time:.6f} sec.')


    #         else:  # Loss decoder with only superchecks according to SSR

    #             MLE_Loss_Decoder_class.generate_measurement_ix_to_instruction_ix_map()  # mapping between measurement events to instruction ix

    #             # Loss decoding - creating DEMs:
    #             dems_list = []
    #             probs_lists = []
    #             observables_errors_interactions_lists = []

    #             print("Shot:", end=" ")
    #             for shot in range(num_shots):
    #                 print(shot, end=" ")
    #                 measurement_event = measurement_events[shot]  # change it to measurements

    #                 start_time = time.time()
    #                 return_matrix_with_observables = False if self.decoder == 'MLE' else True
    #                 final_dem_hyperedges_matrix, observables_errors_interactions = MLE_Loss_Decoder_class.generate_dem_loss_mle_experiment_only_superchecks(
    #                     measurement_event, return_matrix_with_observables=return_matrix_with_observables)
    #                 # print(f'Total loss decoder time per shot is {time.time() - start_time:.4f}s.')

    #                 observables_errors_interactions_lists.append(observables_errors_interactions)
    #                 if self.decoder == "MLE":
    #                     start_time = time.time()
    #                     final_dem_hyperedges_matrix, probs_list = MLE_Loss_Decoder_class.convert_hyperedge_matrix_into_binary(
    #                         hyperedges_matrix=final_dem_hyperedges_matrix)
    #                     # print(f'MLE gurobi decoding time per shot took {time.time() - start_time:.2f}s.')
    #                     dems_list.append(final_dem_hyperedges_matrix)
    #                     probs_lists.append(probs_list)
    #                 else:
    #                     final_dem = MLE_Loss_Decoder_class.from_hyperedges_matrix_into_stim_dem(
    #                         final_dem_hyperedges_matrix)  # convert into stim format. #TODO: bug - fix it! here final_dem_hyperedges_matrix doesn't contain observables so the DEM will not contain them.
    #                     dems_list.append(final_dem)

    #         measurement_events_no_loss = measurement_events.copy()
    #         measurement_events_no_loss[
    #             measurement_events_no_loss == 2] = 0  # change all values in detection_events from 2 to 0
    #         measurement_events_no_loss = measurement_events_no_loss.astype(np.bool_)
    #         detection_events, observable_flips = MLE_Loss_Decoder_class.circuit.compile_m2d_converter().convert(measurements=measurement_events_no_loss, separate_observables=True)

    #         ### ADDED BACK IN 2024/08/20 BY SG ###
    #         # add normalization step of detection events
    #         if (not simulate_data) and type(detection_events_signs) != type(None):
    #             print('Using detection events signs!')
    #             detection_events_int = detection_events.astype(np.int32)
    #             detection_events_flipped = np.where(detection_events_signs == -1,  1 - detection_events_int, detection_events_int)
    #             detection_events = detection_events_flipped.astype(np.bool_)

    #         print(f"Loss decoder is done! Now starting to decode with {self.decoder}")
    #         # Creating the predictions using the DEM:
    #         if self.decoder == "MLE":
    #             start_time = time.time()
    #             predictions, log_probabilities = qec.correlated_decoders.mle_loss.logical_gap_gurobi_with_dem_loss_fast(dems_list=dems_list,
    #                                                                                             probs_lists=probs_lists,
    #                                                                                             detector_shots=detection_events,
    #                                                                                             observables_lists=observables_errors_interactions_lists)
    #             print(f'MLE decoder took {time.time() - start_time:.6f}s.')

    #             # start_time = time.time()
    #             # new_predictions = qec.correlated_decoders.mle_loss.decode_gurobi_with_dem_loss_batch(dems_list=dems_list, probs_lists = probs_lists, detector_shots = detection_events, observables_lists=observables_errors_interactions_lists)
    #             # print(f'MLE decoder took {time.time() - start_time:.6f}s.')

    #             # are_close = np.allclose(predictions, new_predictions, atol=1e-8)
    #             # print(f"Are the predictions close within tolerance? {are_close}")

    #         num_errors = np.sum(np.logical_xor(observable_flips, predictions))
    #         if self.printing:
    #             print(
    #                 f"for dx = {dx}, dy = {dy}, {self.cycles} cycles, {num_shots} shots, we had {num_errors} errors (logical error = {(num_errors / num_shots):.1e})")
    #         return predictions, log_probabilities, observable_flips, dems_list



    #     else:  # no loss in the experiment --> regular decoding, without delayed erasure decoder
    #         raise NotImplementedError
    #         measurement_events_no_loss = measurement_events.copy()
    #         measurement_events_no_loss[
    #             measurement_events_no_loss == 2] = 0  # change all values in detection_events from 2 to 0
    #         measurement_events_no_loss = measurement_events_no_loss.astype(np.bool_)

    #         # print(f"Lets debug!!!!")
    #         # print(f"measurement_events = {measurement_events}")
    #         # print(f"measurement_events_no_loss = {measurement_events_no_loss}")
    #         # print(MLE_Loss_Decoder_class.circuit)
    #         detection_events, observable_flips = MLE_Loss_Decoder_class.circuit.compile_m2d_converter().convert(
    #             measurements=measurement_events_no_loss, separate_observables=True)

    #         # print(detection_events)
    #         # print(f"thats it!")

    #         # DEBUGGING:
    #         # detection_events, observable_flips = LogicalCircuit_no_pulses.compile_m2d_converter().convert(measurements=measurement_events_no_loss, separate_observables=True)

    #         if self.printing:
    #             detection_events_int = detection_events.astype(np.int32)
    #             detection_events_signs_theory = np.where(detection_events_int == 0, -1.0, 1.0)

    #             print(f"detection_events from the pulses circuit = {detection_events_signs_theory[0]}")
    #             print(f"detection_events signs from the experiment = {detection_events_signs}")
    #             are_equal = np.array_equal(detection_events_signs_theory, detection_events_signs)
    #             print("Arrays are equal:", are_equal)
    #             # np.save(f"{self.output_dir}/measurement_events_no_loss.npy", measurement_events_no_loss)
    #             # for shot in detection_events_int:
    #             # print(detection_events_int[shot])
    #             # print(f"0 in detection_events_int : {0 in detection_events_int}")
    #             # print(f"1 in detection_events_int : {1 in detection_events_int}")

    #             print(f"measurement_events_no_loss = {measurement_events_no_loss}")

    #             # print("again:")
    #             # for shot in measurement_events_no_loss:
    #             #     print(measurement_events_no_loss[shot])

    #         ### ADDED BACK IN 2024/08/20 BY SG ###
    #         # add normalization step of detection events
    #         if (not simulate_data) and type(detection_events_signs) != type(None):
    #             print('Using detection events signs!')
    #             detection_events_int = detection_events.astype(np.int32)
    #             detection_events_flipped = np.where(detection_events_signs == -1,  1 - detection_events_int, detection_events_int)
    #             detection_events = detection_events_flipped.astype(np.bool_)

    #         if self.decoder == "MLE":
    #             detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=False,
    #                                                                                     approximate_disjoint_errors=True,
    #                                                                                     ignore_decomposition_failures=True,
    #                                                                                     allow_gauge_detectors=False)
    #             predictions = qec.correlated_decoders.mle.decode_gurobi_with_dem(dem=detector_error_model,
    #                                                                                 detector_shots=detection_events)


    #         num_errors = np.sum(np.logical_xor(observable_flips, predictions))
    #         if self.printing:
    #             print(
    #                 f"for dx = {dx}, dy = {dy}, {self.cycles} cycles, {num_shots} shots, we had {num_errors} errors (logical error = {(num_errors / num_shots):.1e})")
    #         # predictions_bool = predictions.astype(bool).squeeze()
    #         return predictions, observable_flips, detector_error_model


# def simulate(self, distances, num_shots):
#         """
#         Simulate the circuits for a set of distances and physical error rates.
#         Save results to the specified output file.
#         """
#         f = open(f'{self.output_dir}/{self.save_filename}.txt', "a")
        
#         for (dx, dy) in distances:
#             cycles = min(dx, dy) if self.cycles == None else self.cycles
                        
#             for phys_err in self.phys_err_vec:
#                 start_time = time.time()
                
#                 LogicalCircuit = self.generate_circuit(dx=dx, dy=dy, cycles=cycles, phys_err=phys_err)
#                 if self.printing:
#                     print(f"\nCircuit after noise:\n{LogicalCircuit} \n")
#                     print(f"potential lost qubits: {LogicalCircuit.potential_lost_qubits} \n with loss probabilities: {LogicalCircuit.loss_probabilities}")
                
                
#                 if self.circuit_type == 'Steane_QEC':
#                     ValueError (self.num_logicals == 3)
#                     corrections, observables, probabilities1, probabilities2 = self.count_logical_errors_preselection(num_shots=num_shots, dx=dx, dy=dy, phys_error=phys_err, cycles=cycles)
                    
#                     job_id = self.get_job_id()
#                     full_filename = f'{self.output_dir}/{self.save_filename}/dx{dx}__dy{dy}__c{cycles}__p{phys_err}__Steane_QEC_results.pickle'

#                     # Create folder if it doesn't exist.
#                     folder_path = f'{self.output_dir}/{self.save_filename}'
#                     if not os.path.exists(folder_path):
#                         # If the folder does not exist, create it
#                         os.makedirs(folder_path)
                    
#                     # Data structure to append (now as a list)
#                     data_to_append = [job_id, corrections, observables, probabilities1, probabilities2]
                    
#                     # Open the file in append mode, binary
#                     with open(full_filename, 'ab') as file:
#                         pickle.dump(data_to_append, file)
                        
#                     f.write(f'{dx} {dy} {phys_err} {cycles} {job_id} {num_shots} {time.time()-start_time}\n')
                    
                    
#                 else:
#                     num_errors_sampled, num_shots = self.count_logical_errors(LogicalCircuit=LogicalCircuit, num_shots=num_shots, dx=dx, dy=dy, phys_error=phys_err, cycles=cycles)
#                     print(f"for dx = {dx}, dy = {dy}, {cycles} cycles, physical error rate = {phys_err}, {num_shots} shots, we had {num_errors_sampled} errors (logical error = {(num_errors_sampled/num_shots):.1e})")
#                     f.write(f'{dx} {dy} {phys_err} {num_errors_sampled} {num_shots} {time.time()-start_time}\n')
                    
#         f.close()
