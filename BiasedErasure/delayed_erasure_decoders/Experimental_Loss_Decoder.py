import numpy as np
import qec
import time
from BiasedErasure.main_code.Simulator import *
from BiasedErasure.delayed_erasure_decoders.HeraldedCircuit_SWAP_LD import HeraldedCircuit_SWAP_LD
from BiasedErasure.main_code.noise_channels import atom_array

def Loss_MLE_Decoder_Experiment(Meta_params, distance: int, output_dir: str, measurement_events: np.ndarray, detection_events_signs: np.ndarray, use_loss_decoding=True):
        """This function decodes the loss information using mle. 
        Given heralded losses upon measurements, there are multiple potential loss events (with some probability) in the circuit.
        We use the MLE approximate decoding - for each potential loss event we create a DEM and connect all to generate the final DEM. We use MLE to decode given the ginal DEM and the experimental measurements.
        Input: Meta_params, distance, num shots, experimental data: detector shots.
        Output: final DEM, corrections, num errors.

        Meta_params = {'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': 'X', 'bias_preserving_gates': 'False', 
        'noise': 'atom_array', 'is_erasure_biased': 'False', 'LD_freq': '1', 'LD_method': 'SWAP', 'SSR': 'True', 'cycles': '2', 'ordering': 'bad', 'decoder': 'MLE',
        'circuit_type': 'memory', 'Steane_type': 'regular', 'printing': 'False', 'num_logicals': '1', 'loss_decoder': 'independent'}
        ordering: bad or fowler (good)
        decoder: MLE or MWPM
        use_loss_decoding: True or False. Do we want to use the delayed-erasure decoder?
        """
        num_shots = measurement_events.shape[0]
        
        
        # Step 0 - generate the Simulator class:
        bloch_point_params = {'erasure_ratio': '1', 'bias_ratio': '0.5'}
        # file_name = create_file_name(Meta_params, bloch_point_params = bloch_point_params)
        cycles = int(Meta_params['cycles'])
        simulator = Simulator(Meta_params=Meta_params, atom_array_sim=True, 
                                bloch_point_params=bloch_point_params, noise=atom_array , phys_err_vec=None, loss_detection_method=HeraldedCircuit_SWAP_LD, 
                                cycles = cycles, output_dir=output_dir, save_filename=None)
        
        # Step 1 - decode:
        predictions_bool, dems_list = simulator.count_logical_errors_experiment(num_shots = num_shots, distance = distance, measurement_events = measurement_events, detection_events_signs=detection_events_signs, use_loss_decoding=use_loss_decoding)
        
        
        return predictions_bool, dems_list