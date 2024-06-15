import numpy as np
import matplotlib.pyplot as plt
from BiasedErasure.main_code.noise_channels import *
import qec
import progressbar
import time
import sinter
import math
import pickle
from BiasedErasure.main_code.circuits import *
from distutils.util import strtobool
from BiasedErasure.delayed_erasure_decoders.MLE_Loss_Decoder import MLE_Loss_Decoder
import os.path
import os
import json
from hashlib import sha256
import pickle
from BiasedErasure.main_code.XZZX import XZZX

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
                dont_use_loss_decoder=False
                ) -> None:
        """
        Architecture is one of 'CBQC', 'MBQC'
        Code is one of: 'Rotated_Surface', 'Surface'
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
        self.printing = strtobool(Meta_params['printing'])
        self.output_dir = output_dir
        self.save_filename = save_filename
        self.first_comb_weight = first_comb_weight
        self.dont_use_loss_decoder = dont_use_loss_decoder # if TRUE, we are not using loss decoder at all. all shots get same DEM.

    def get_job_id(self):
        # Check for environment variables used by different cluster management systems
        for env_var in ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID']:
            job_id = os.environ.get(env_var)
            if job_id is not None:
                return job_id
        return None  # or raise an error if the job ID is critical

    def generate_unique_key(self):
        # Convert the data to a JSON string
        data_str = json.dumps({
            'meta_params': self.Meta_params,
            'distance': self.distance
        }, sort_keys=True)
        # Use SHA256 to generate a unique hash of the data
        return sha256(data_str.encode()).hexdigest()
    

    def generate_circuit(self, d, cycles, phys_err):
        entangling_gate_error_rate, entangling_gate_loss_rate = self.noise(phys_err, self.bias_ratio)
        if self.circuit_type == 'memory':
            if self.architecture == 'CBQC':
                return memory_experiment_surface_new(d=d, code=self.code, QEC_cycles=cycles, entangling_gate_error_rate=entangling_gate_error_rate, 
                                                entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio = self.erasure_ratio,
                                                num_logicals=self.num_logicals, logical_basis=self.logical_basis, 
                                                biased_pres_gates = self.bias_preserving_gates, ordering = self.ordering_type,
                                                loss_detection_method = self.loss_detection_method_str, 
                                                loss_detection_frequency = self.loss_detection_freq, atom_array_sim=self.atom_array_sim)
                
                # return memory_experiment_surface(d=d, code=self.code, QEC_cycles=cycles-1, entangling_gate_error_rate=entangling_gate_error_rate, 
                #                                 entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio = self.erasure_ratio,
                #                                 num_logicals=self.num_logicals, logical_basis=self.logical_basis, 
                #                                 biased_pres_gates = self.bias_preserving_gates, ordering = self.ordering_type,
                #                                 loss_detection_method = self.loss_detection_method_str, 
                #                                 loss_detection_frequency = self.loss_detection_freq, atom_array_sim=self.atom_array_sim)
            elif self.architecture == 'MBQC':
                return memory_experiment_MBQC(d=d, QEC_cycles=cycles, entangling_gate_error_rate=entangling_gate_error_rate, 
                                                entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio = self.erasure_ratio,
                                                logical_basis=self.logical_basis, 
                                                biased_pres_gates = self.bias_preserving_gates, atom_array_sim=self.atom_array_sim)
                
                
        elif self.circuit_type in ['GHZ_all_o1', 'GHZ_save_o1','GHZ_all_o2', 'GHZ_save_o2']:
            order_1 = []
            order_2 = []
            for i in range(1, self.num_logicals):
                order_1.append((0, i))
                order_2.append((i - 1, i))
            chosen_order = order_1 if self.circuit_type.endswith("1") else order_2
            if self.circuit_type in ['GHZ_all_o1', 'GHZ_all_o2']:
                return GHZ_experiment_Surface(d=d, order=chosen_order, num_logicals=self.num_logicals, code=self.code, QEC_cycles=cycles, 
                                            entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate, 
                                            erasure_ratio = self.erasure_ratio,
                                            logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates, 
                                            loss_detection_on_all_qubits=True, atom_array_sim=self.atom_array_sim)
            elif self.circuit_type in ['GHZ_save_o1', 'GHZ_save_o2']:
                return GHZ_experiment_Surface(d=d, order=chosen_order, num_logicals=self.num_logicals, code=self.code, QEC_cycles=cycles, 
                                            entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate, 
                                            erasure_ratio = self.erasure_ratio,
                                            logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates, 
                                            loss_detection_on_all_qubits=False, atom_array_sim=self.atom_array_sim)
        elif self.circuit_type == 'Steane_QEC':
            return Steane_QEC_circuit(d=d, code=self.code, Steane_type=self.Steane_type, QEC_cycles=cycles-1,
                                        entangling_gate_error_rate=entangling_gate_error_rate, entangling_gate_loss_rate=entangling_gate_loss_rate,
                                        erasure_ratio=self.erasure_ratio,
                                        logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates,
                                        loss_detection_on_all_qubits=True, atom_array_sim=self.atom_array_sim)
                

        else:
            return None
    
    
    def simulate(self, distances, num_shots):
        f = open(f'{self.output_dir}/{self.save_filename}.txt', "a")
        
        for d in distances:
            cycles = d if self.cycles == None else self.cycles
                        
            for phys_err in self.phys_err_vec:
                start_time = time.time()
                
                LogicalCircuit = self.generate_circuit(d=d, cycles=cycles, phys_err=phys_err)
                if self.printing:
                    print(f"\nCircuit after noise:\n{LogicalCircuit} \n")
                    print(f"potential lost qubits: {LogicalCircuit.potential_lost_qubits} \n with loss probabilities: {LogicalCircuit.loss_probabilities}")
                
                
                if self.circuit_type == 'Steane_QEC':
                    ValueError (self.num_logicals == 3)
                    corrections, observables, probabilities1, probabilities2 = self.count_logical_errors_preselection(num_shots=num_shots, distance=d, phys_error=phys_err, cycles=cycles)
                    
                    job_id = self.get_job_id()
                    full_filename = f'{self.output_dir}/{self.save_filename}/d{d}__c{cycles}__p{phys_err}__Steane_QEC_results.pickle'

                    # Create folder if it doesn't exist.
                    folder_path = f'{self.output_dir}/{self.save_filename}'
                    if not os.path.exists(folder_path):
                        # If the folder does not exist, create it
                        os.makedirs(folder_path)
                    
                    # Data structure to append (now as a list)
                    data_to_append = [job_id, corrections, observables, probabilities1, probabilities2]
                    
                    # Open the file in append mode, binary
                    with open(full_filename, 'ab') as file:
                        pickle.dump(data_to_append, file)
                        
                    f.write(f'{d} {phys_err} {cycles} {job_id} {num_shots} {time.time()-start_time}\n')
                    
                    
                else:
                    num_errors_sampled, num_shots = self.count_logical_errors(LogicalCircuit=LogicalCircuit, num_shots=num_shots, distance=d, phys_error=phys_err, cycles=cycles)
                    print(f"for d = {d}, {cycles} cycles, physical error rate = {phys_err}, {num_shots} shots, we had {num_errors_sampled} errors (logical error = {(num_errors_sampled/num_shots):.1e})")
                    f.write(f'{d} {phys_err} {num_errors_sampled} {num_shots} {time.time()-start_time}\n')
                    
        f.close()

    
    def count_logical_errors(self, LogicalCircuit, num_shots: int, distance=None, phys_error=0, debugging=False, cycles=None):
        """This function decodes the loss information using mle. 
        Given heralded losses upon measurements, there are multiple potential loss events (with some probability) in the circuit.
        There are 2 options:
        1. MLE approximate decoding - for each potential loss event we create a DEM and connect all to generate the final DEM. We use MLE to decode given the ginal DEM and the experimental measurements.
        2. Accurate MLE - for each potential loss event i we decode with Gorubi to get P_i, and take argmax(P_i) to decode.
        """
        # decoder_type = 'independent'
        
        if self.printing:
            print(f"Starting the decoding!")
            start_time = time.time()

        if self.architecture == "MBQC":
            xzzx_instance = XZZX()
            data_qubits, ancilla_qubits = xzzx_instance.get_data_ancilla_indices(dx=distance, dy=distance, cycles=cycles, architecture="MBQC")
        else:
            ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].measure_qubits]
            data_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].data_qubits]
        
        
        sampler = LogicalCircuit.compile_detector_sampler()
        loss_sampling = 1 if self.erasure_ratio > 0 else  num_shots # how many times we will use the same loss sampling result
        num_loss_shots = math.ceil(num_shots / loss_sampling)
        num_shots = num_loss_shots * loss_sampling
        loss_detection_events_all_shots = np.random.rand(num_loss_shots, len(LogicalCircuit.potential_lost_qubits)) < LogicalCircuit.loss_probabilities
        # LogicalCircuit.logical_qubits[0].visualize_code()
        # print(LogicalCircuit.loss_probabilities)
        loss_detection_class = self.heralded_circuit(circuit=LogicalCircuit, biased_erasure=self.is_erasure_biased, bias_preserving_gates=self.bias_preserving_gates,
                                                                    basis = self.logical_basis, distance=distance, erasure_ratio = self.erasure_ratio, 
                                                                    phys_error = phys_error, ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                                    SSR=self.SSR, cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq)
        
        
        MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params, distance = distance, loss_detection_method_str=self.loss_detection_method_str,
                                                ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq, 
                                                first_comb_weight=self.first_comb_weight,
                                                output_dir = self.output_dir, decoder_type = self.loss_decoder_type)
        
        if self.loss_detection_method_str == 'SWAP':
            SWAP_circuit = loss_detection_class.transfer_circuit_into_SWAP_circuit(LogicalCircuit)
            if self.printing:
                print(f"Logical circuit after implementing SWAP is: \n {SWAP_circuit}\n")
            # loss_detection_class.SWAP_circuit = SWAP_circuit
            MLE_Loss_Decoder_class.circuit = SWAP_circuit
        
        elif self.loss_detection_method_str in ['FREE', 'MBQC']:
            MLE_Loss_Decoder_class.circuit = LogicalCircuit
            
        
        
        if self.printing:
            end_time = time.time()
            print(f"Initialization of loss decoders (and building SWAP circuit if needed) took {end_time - start_time} sec.")
            start_time = time.time()
                
        if self.erasure_ratio > 0:
            
            MLE_Loss_Decoder_class.initialize_loss_decoder()
            
            if self.printing:
                end_time = time.time()
                print(f"Building the Pauli DEM and loading all independent loss DEMs took {end_time - start_time} sec. Starting the loss decoding!")
                start_time = time.time()
            
            num_errors = 0
            dems_list = []
            detection_events_list = []
            observable_flips_list = []
            start_time_all_shots = time.time()
            for shot in range(num_loss_shots):
                loss_detection_events = loss_detection_events_all_shots[shot]
                
                experimental_circuit, detector_error_model = MLE_Loss_Decoder_class.decode_loss_MLE(loss_detection_events)

                if self.printing:
                    # print(f"\n Loss detection events: {loss_detection_events}")
                    # print(f"\n Potential lost qubits: {LogicalCircuit.potential_lost_qubits} \n with loss probabilities: {LogicalCircuit.loss_probabilities}")
                    print("\n Experimental circuit (for measurements):")
                    print(experimental_circuit)
                    print("\n MLE DEM:")
                    print(detector_error_model)
                
                # Sample MEASUREMENTS from experimental_circuit
                sampler = experimental_circuit.compile_detector_sampler()
                detection_events, observable_flips = sampler.sample(loss_sampling, separate_observables=True)
                
                detection_events_list.append(detection_events)
                observable_flips_list.append(observable_flips)
                
                # Extract decoder configuration data from the circuit.
                if self.dont_use_loss_decoder:
                    no_loss_decoder_dem = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
                    dems_list.append(no_loss_decoder_dem)
                else:
                    dems_list.append(detector_error_model)
                
            if self.printing:
                end_time_all_shots = time.time()
                print(f"Building the MLE loss DEM and experimental circuits for all shots took {end_time_all_shots - start_time_all_shots} sec")
                start_time = time.time()
            
            if self.decoder == "MLE":
                
                detector_shots = np.array(detection_events_list)
                predictions = qec.correlated_decoders.mle_loss.decode_gurobi_with_dem_loss(dems_list=dems_list, detector_shots = detector_shots)   
                if self.printing:
                    end_time = time.time()
                    print(f"Gurobi correlated mle decoding for all shots took {end_time - start_time} sec")
            else:
                predictions = []
                for (d, detection_events) in enumerate(detection_events_list):
                    detector_error_model = dems_list[d]
                    matching = pymatching.Matching.from_detector_error_model(detector_error_model)
                    prediction = matching.decode_batch(detection_events)
                    predictions.append(prediction[0][0])
                predictions = np.array(predictions)
                if self.printing:
                    end_time = time.time()
                    print(f"MWPM decoding for all shots took {end_time - start_time} sec")
                
            observable_flips = np.array(observable_flips_list)
            predictions_bool = predictions.astype(bool).squeeze()
            observable_flips_squeezed = observable_flips.squeeze()

            num_errors = np.sum(np.logical_xor(observable_flips_squeezed, predictions_bool))


                                                    
        else: # no loss errors at all
            sampler = MLE_Loss_Decoder_class.circuit.compile_detector_sampler()
            detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
            

            if self.decoder == "MLE":
                detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
                prediction = qec.correlated_decoders.mle.decode_gurobi_with_dem(dem=detector_error_model, detector_shots = detection_events)
            else:
                detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True) 
                prediction = sinter.predict_observables(
                    dem=detector_error_model,
                    dets=detection_events,
                    decoder='pymatching',
                )

            num_errors = np.sum(np.logical_xor(observable_flips, prediction))
        
        # logical_error = num_errors / num_shots
        return num_errors, num_shots
    
    
    
    def count_logical_errors_experiment(self, num_shots: int, distance: int, measurement_events: np.ndarray, detection_events_signs: np.ndarray, use_loss_decoding=True):
        """This function decodes the loss information using mle. 
        Given heralded losses upon measurements, there are multiple potential loss events (with some probability) in the circuit.
        We use the MLE approximate decoding - for each potential loss event we create a DEM and connect all to generate the final DEM. We use MLE to decode given the ginal DEM and the experimental measurements.
        Input: Meta_params, distance, num shots, experimental data: detector shots.
        Output: final DEM, corrections, num errors.
        
        Meta_params = {'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': 'X', 'bias_preserving_gates': 'False', 
                'noise': 'atom_array', 'is_erasure_biased': 'False', 'LD_freq': '1', 'LD_method': 'SWAP', 'SSR': 'True', 'cycles': '2', 'ordering': 'bad', 'decoder': 'MLE',
                'circuit_type': 'memory', 'printing': 'False', 'num_logicals': '1'}
        ordering: bad or fowler (good)
        decoder: MLE or MWPM
        """
        
        # Step 1 - generate the experimental circuit in our simulation:
        LogicalCircuit = self.generate_circuit(d=distance, cycles=self.cycles, phys_err=None)
        
        ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].measure_qubits]
        data_qubits = [qubit for i in range(self.num_logicals) for qubit in LogicalCircuit.logical_qubits[i].data_qubits]
                
        
        # LogicalCircuit.logical_qubits[0].visualize_code()
        loss_detection_class = self.heralded_circuit(circuit=LogicalCircuit, biased_erasure=self.is_erasure_biased, bias_preserving_gates=self.bias_preserving_gates,
                                                                    basis = self.logical_basis, distance=distance, erasure_ratio = self.erasure_ratio, 
                                                                    phys_error = None, ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                                    SSR=self.SSR, cycles=self.cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq)
        
        
        MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params, distance = distance, loss_detection_method_str=self.loss_detection_method_str,
                                                ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                cycles=self.cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq,
                                                output_dir = self.output_dir, decoder_type=self.loss_decoder_type)
        
        if self.loss_detection_method_str == 'SWAP':
            SWAP_circuit = loss_detection_class.transfer_circuit_into_SWAP_circuit(LogicalCircuit)
            if self.printing:
                print(f"Logical circuit after implementing SWAP is: \n {SWAP_circuit}\n")
            loss_detection_class.SWAP_circuit = SWAP_circuit
            MLE_Loss_Decoder_class.circuit = SWAP_circuit
        
        elif self.loss_detection_method_str == 'FREE':
            MLE_Loss_Decoder_class.circuit = LogicalCircuit
        
        if self.printing:
            print(f"Logical circuit that will be used: \n{MLE_Loss_Decoder_class.circuit}")
            
        if 2 in measurement_events and use_loss_decoding:
            loss_sampling = 1 # how many times we will use the same loss sampling result
            num_loss_shots = math.ceil(num_shots / loss_sampling)
            num_shots = num_loss_shots * loss_sampling
        
            MLE_Loss_Decoder_class.initialize_loss_decoder() # this part can be improved to be a bit faster
            
            # Loss decoding - creating DEMs:
            dems_list = []
            probs_lists = []
            observables_errors_interactions_lists = []
            for shot in range(num_loss_shots):
                measurement_event = measurement_events[shot] # change it to measurements
                final_dem_hyperedges_matrix, observables_errors_interactions = MLE_Loss_Decoder_class.generate_dem_loss_mle_experiment(measurement_event) # final_dem_hyperedges_matrix doesn't contain observables, only detectors               
                observables_errors_interactions_lists.append(observables_errors_interactions)
                
                if self.decoder == "MLE":
                    final_dem_hyperedges_matrix, probs_list = MLE_Loss_Decoder_class.convert_hyperedge_matrix_into_binary(hyperedges_matrix = final_dem_hyperedges_matrix)
                    dems_list.append(final_dem_hyperedges_matrix)
                    probs_lists.append(probs_list)
                else:
                    final_dem = MLE_Loss_Decoder_class.from_hyperedges_matrix_into_stim_dem(final_dem_hyperedges_matrix) # convert into stim format. #TODO: bug - fix it! here final_dem_hyperedges_matrix doesn't contain observables so the DEM will not contain them.
                    dems_list.append(final_dem)
                    
                if self.printing:
                    print("\n MLE DEM hyperedges matrix:")
                    print(final_dem_hyperedges_matrix)

            measurement_events[measurement_events == 2] = 0 #change all values in detection_events from 2 to 0
            measurement_events = measurement_events.astype(np.bool_)
            detection_events, observable_flips = MLE_Loss_Decoder_class.circuit.compile_m2d_converter().convert(measurements=measurement_events, separate_observables=True)
            
            # add normalization step of detection events:
            detection_events = detection_events * detection_events_signs
            
            
            # # # FOR DEBUGGING ONLY! Sample MEASUREMENTS from experimental_circuit - DELETE!! START
            # sampler = MLE_Loss_Decoder_class.circuit.compile_detector_sampler()
            # detection_events, observable_flips = sampler.sample(num_loss_shots, separate_observables=True)
            # # # FOR DEBUGGING ONLY! Sample MEASUREMENTS from experimental_circuit - DELETE!! END
            
            # Creating the predictions using the DEM:
            if self.decoder == "MLE":
                
                predictions = qec.correlated_decoders.mle_loss.decode_gurobi_with_dem_loss(dems_list=dems_list, probs_lists = probs_lists, detector_shots = detection_events, observables_lists=observables_errors_interactions_lists)   
                # save an example for Maddie:
                # full_filename = f"{self.output_dir}/example_for_maddie.pickle"
                # with open(full_filename, 'wb') as file:
                #     pickle.dump((dems_list, probs_lists, detection_events, observables_errors_interactions_lists), file)
            
            
            else:
                predictions = []
                for (d, detection_event) in enumerate(detection_events):
                    detector_error_model = dems_list[d]
                    matching = pymatching.Matching.from_detector_error_model(detector_error_model)
                    prediction = matching.decode_batch(detection_event)
                    predictions.append(prediction[0][0])
            
            num_errors = np.sum(np.logical_xor(observable_flips, predictions))
            if self.printing:
                print(f"for d = {distance}, {self.cycles} cycles, {num_shots} shots, we had {num_errors} errors (logical error = {(num_errors/num_shots):.1e})")
            # predictions_bool = predictions.astype(bool).squeeze()
            return predictions, observable_flips, dems_list
        
        
        
        
        else: # regular decoding, without delayed erasure decoder
            measurement_events[measurement_events == 2] = 0 #change all values in detection_events from 2 to 0
            measurement_events = measurement_events.astype(np.bool_)
            detection_events, observable_flips = MLE_Loss_Decoder_class.circuit.compile_m2d_converter().convert(measurements=measurement_events, separate_observables=True)
            
            # add normalization step of detection events:
            detection_events_int = detection_events.astype(np.int32)
            detection_events_flipped = np.where(detection_events_signs == -1, ~detection_events_int, detection_events_int)
            detection_events = detection_events_flipped.astype(np.bool_)

            
            
            if self.decoder == "MLE":
                detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
                predictions = qec.correlated_decoders.mle.decode_gurobi_with_dem(dem=detector_error_model, detector_shots = detection_events)
            else:
                detector_error_model = MLE_Loss_Decoder_class.circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True) 
                predictions = sinter.predict_observables(
                    dem=detector_error_model,
                    dets=detection_events,
                    decoder='pymatching',
                )

            num_errors = np.sum(np.logical_xor(observable_flips, predictions))
            if self.printing:
                print(f"for d = {distance}, {self.cycles} cycles, {num_shots} shots, we had {num_errors} errors (logical error = {(num_errors/num_shots):.1e})")
            # predictions_bool = predictions.astype(bool).squeeze()
            return predictions, observable_flips, detector_error_model
        
    
    
    def count_logical_errors_preselection_old(self, num_shots: int, distance=None, phys_error=0, debugging=False, cycles=None):
        decoder_type = 'independent'
        d = distance

        entangling_gate_error_rate, entangling_gate_loss_rate = self.noise(phys_error, self.bias_ratio, self.erasure_ratio)
        def steane(d, ancilla1_for_preselection=False, ancilla2_for_preselection=False):
            return Steane_QEC_circuit(d=d, code=self.code, Steane_type=self.Steane_type, QEC_cycles=cycles, entangling_gate_error_rate=entangling_gate_error_rate,
                                    entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio=self.erasure_ratio, logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates,
                                    loss_detection_on_all_qubits=True, atom_array_sim=self.atom_array_sim, ancilla1_for_preselection=ancilla1_for_preselection, ancilla2_for_preselection=ancilla2_for_preselection)

        lc = steane(d)
        
        ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in lc.logical_qubits[i].measure_qubits]
        data_qubits = [qubit for i in range(self.num_logicals) for qubit in lc.logical_qubits[i].data_qubits]
        
        loss_sampling = 1 if self.erasure_ratio > 0 else  num_shots # how many times we will use the same loss sampling result
        num_loss_shots = math.ceil(num_shots / loss_sampling)
        num_shots = num_loss_shots * loss_sampling
        loss_detection_events_all_shots = np.random.rand(num_loss_shots, len(lc.potential_lost_qubits)) < lc.loss_probabilities

        
        # get SWAP circuit for each circuit ('regular','1','2'):
        loss_detection_class = self.heralded_circuit(circuit=lc, biased_erasure=self.is_erasure_biased, bias_preserving_gates=self.bias_preserving_gates,
                                                                    basis = self.logical_basis, distance=distance, erasure_ratio = self.erasure_ratio, 
                                                                    phys_error = phys_error, ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                                    SSR=self.SSR, cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq)
        SWAP_circuit_regular = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc)

            
        lc_for_preselection1 = steane(d, ancilla1_for_preselection=True)
        loss_detection_class.logical_circuit = lc_for_preselection1
        SWAP_circuit_preselection1 = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc_for_preselection1)

        lc_for_preselection2 = steane(d, ancilla2_for_preselection=True)
        loss_detection_class.logical_circuit = lc_for_preselection2
        SWAP_circuit_preselection2 = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc_for_preselection2)


        # get DEMs for each circuit:
        MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params, distance = distance, loss_detection_method_str=self.loss_detection_method_str,
                                                ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq, 
                                                first_comb_weight=self.first_comb_weight,
                                                output_dir = self.output_dir, decoder_type = decoder_type)
        #MLE_Loss_Decoder_class.initialize_loss_decoder()
        SWAP_circuits = {'regular': SWAP_circuit_regular, '1': SWAP_circuit_preselection1, '2': SWAP_circuit_preselection2}
        #DEMs = {'regular': [], '1': [], '2': []}

        corrections = []
        observables = []
        probabilities1 = np.zeros((0, 2))
        probabilities2 = np.zeros((0, 2))
        for circuit_name in ['regular','1','2']:
            print(circuit_name)
            MLE_Loss_Decoder_class.circuit = SWAP_circuits[circuit_name]
            MLE_Loss_Decoder_class.initialize_loss_decoder()
            for shot in range(num_loss_shots):
                loss_detection_events = loss_detection_events_all_shots[shot]
                experimental_circuit, dem = MLE_Loss_Decoder_class.decode_loss_MLE(loss_detection_events)
                if circuit_name == 'regular':
                    # Sample MEASUREMENTS from experimental_circuit
                    sampler = experimental_circuit.compile_detector_sampler()
                    detection_events, observable_flips = sampler.sample(loss_sampling, separate_observables=True)
                    observables.extend(observable_flips)
                    corrections.extend(qec.correlated_decoders.mle.decode_gurobi_with_dem(dem, detection_events))
                elif circuit_name == '1':
                    _, prob1 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
                    probabilities1 = np.concatenate((probabilities1, prob1), axis=0)
                elif circuit_name == '2':
                    _, prob2 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
                    probabilities2 = np.concatenate((probabilities2, prob2), axis=0)
                else:
                    assert True is False

            
        return corrections, observables, probabilities1, probabilities2


    def count_logical_errors_preselection(self, num_shots: int, distance=None, phys_error=0, debugging=False, cycles=None):
        decoder_type = 'independent'
        d = distance

        entangling_gate_error_rate, entangling_gate_loss_rate = self.noise(phys_error, self.bias_ratio)
        def steane(d, ancilla1_for_preselection=False, ancilla2_for_preselection=False):
            return Steane_QEC_circuit(d=d, code=self.code, Steane_type=self.Steane_type, QEC_cycles=cycles, entangling_gate_error_rate=entangling_gate_error_rate,
                                    entangling_gate_loss_rate=entangling_gate_loss_rate, erasure_ratio=self.erasure_ratio, logical_basis=self.logical_basis, biased_pres_gates = self.bias_preserving_gates,
                                    loss_detection_on_all_qubits=True, atom_array_sim=self.atom_array_sim, ancilla1_for_preselection=ancilla1_for_preselection, ancilla2_for_preselection=ancilla2_for_preselection)

        lc = steane(d)
        
        ancilla_qubits = [qubit for i in range(self.num_logicals) for qubit in lc.logical_qubits[i].measure_qubits]
        data_qubits = [qubit for i in range(self.num_logicals) for qubit in lc.logical_qubits[i].data_qubits]
        
        loss_sampling = 1 if self.erasure_ratio > 0 else  num_shots # how many times we will use the same loss sampling result
        num_loss_shots = math.ceil(num_shots / loss_sampling)
        num_shots = num_loss_shots * loss_sampling
        loss_detection_events_all_shots = np.random.rand(num_loss_shots, len(lc.potential_lost_qubits)) < lc.loss_probabilities

        
        # get SWAP circuit for each circuit ('regular','1','2'):
        loss_detection_class = self.heralded_circuit(circuit=lc, biased_erasure=self.is_erasure_biased, bias_preserving_gates=self.bias_preserving_gates,
                                                                    basis = self.logical_basis, distance=distance, erasure_ratio = self.erasure_ratio, 
                                                                    phys_error = phys_error, ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                                    SSR=self.SSR, cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq)
        if self.loss_detection_method_str == 'SWAP':
            SWAP_circuit_regular = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc)
        else:
            SWAP_circuit_regular = lc
            
        lc_for_preselection1 = steane(d, ancilla1_for_preselection=True)
        loss_detection_class.logical_circuit = lc_for_preselection1
        if self.loss_detection_method_str == 'SWAP':
            SWAP_circuit_preselection1 = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc_for_preselection1)
        else:
            SWAP_circuit_preselection1 = lc_for_preselection1

        lc_for_preselection2 = steane(d, ancilla2_for_preselection=True)
        loss_detection_class.logical_circuit = lc_for_preselection2
        if self.loss_detection_method_str == 'SWAP':
            SWAP_circuit_preselection2 = loss_detection_class.transfer_circuit_into_SWAP_circuit(lc_for_preselection2)
        else:
            SWAP_circuit_preselection2 = lc_for_preselection2

        # get DEMs for each circuit:
        MLE_Loss_Decoder_class = MLE_Loss_Decoder(Meta_params=self.Meta_params, bloch_point_params=self.bloch_point_params, distance = distance, loss_detection_method_str=self.loss_detection_method_str,
                                                ancilla_qubits=ancilla_qubits, data_qubits=data_qubits,
                                                cycles=cycles, printing=self.printing, loss_detection_freq = self.loss_detection_freq, 
                                                first_comb_weight=self.first_comb_weight,
                                                output_dir = self.output_dir, decoder_type = decoder_type)
        #MLE_Loss_Decoder_class.initialize_loss_decoder()
        SWAP_circuits = {'regular': SWAP_circuit_regular, '1': SWAP_circuit_preselection1, '2': SWAP_circuit_preselection2}
        #DEMs = {'regular': [], '1': [], '2': []}

        corrections = []
        observables = []
        probabilities1 = np.zeros((0, 2))
        probabilities2 = np.zeros((0, 2))
        for circuit_name in ['regular','1','2']:
            print(circuit_name)
            
            if self.erasure_ratio > 0:
                MLE_Loss_Decoder_class.circuit = SWAP_circuits[circuit_name]
                MLE_Loss_Decoder_class.initialize_loss_decoder()
                for shot in range(num_loss_shots):
                    loss_detection_events = loss_detection_events_all_shots[shot]
                    experimental_circuit, dem = MLE_Loss_Decoder_class.decode_loss_MLE(loss_detection_events)
                    if circuit_name == 'regular':
                        # Sample MEASUREMENTS from experimental_circuit
                        sampler = experimental_circuit.compile_detector_sampler()
                        detection_events, observable_flips = sampler.sample(loss_sampling, separate_observables=True)
                        observables.extend(observable_flips)
                        corrections.extend(qec.correlated_decoders.mle.decode_gurobi_with_dem(dem, detection_events))
                    elif circuit_name == '1':
                        # _, prob1 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
                        _, prob1 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events)
                        probabilities1 = np.concatenate((probabilities1, prob1), axis=0)
                    elif circuit_name == '2':
                        # _, prob2 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
                        _, prob2 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events)
                        probabilities2 = np.concatenate((probabilities2, prob2), axis=0)
                    else:
                        assert True is False
                        
            else: # loss ratio = 0, execute all shot together
                dem = SWAP_circuits[circuit_name].detector_error_model(decompose_errors=False, approximate_disjoint_errors=True, ignore_decomposition_failures=True, allow_gauge_detectors=False)
                if circuit_name == 'regular':
                    sampler = SWAP_circuits[circuit_name].compile_detector_sampler()
                    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
                    observables.extend(observable_flips)
                    corrections.extend(qec.correlated_decoders.mle.decode_gurobi_with_dem(dem, detection_events))
                elif circuit_name == '1':
                    _, prob1 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
                    probabilities1 = np.concatenate((probabilities1, prob1), axis=0)
                elif circuit_name == '2':
                    _, prob2 = qec.correlated_decoders.mle_sliding_scale.sliding_scale(dem, detection_events[:, :dem.num_detectors])
                    probabilities2 = np.concatenate((probabilities2, prob2), axis=0)
                else:
                    assert True is False
            
        return corrections, observables, probabilities1, probabilities2
