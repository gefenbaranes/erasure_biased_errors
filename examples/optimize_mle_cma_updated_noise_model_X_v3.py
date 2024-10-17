import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
import torch
from BiasedErasure.delayed_erasure_decoders.Experimental_Loss_Decoder import *
from BiasedErasure.main_code.Simulator import *
from BiasedErasure.main_code.noise_channels import atom_array
from BiasedErasure.delayed_erasure_decoders.HeraldedCircuit_SWAP_LD import HeraldedCircuit_SWAP_LD
import cma
from tqdm.auto import tqdm
import os
import pickle
import sys

def optimize_theory_fidelity(measurement_events, Meta_params, output_dir, decoder_basis, job_id):
    # Maximum values we want to explore
    baseline  =  {'idle_loss_rate': 2.676856486112905e-07,
                  'idle_error_rate': np.array([6.52979756e-09, 2.64915613e-08, 2.16597810e-07]),
                  'entangling_zone_error_rate': np.array([4.00355696e-04, 1.63884225e-06, 4.28796254e-03]),
                  'entangling_gate_error_rate': [2.362013083202686e-05, 0.00016270794884179973, 0.0017266631741779512, 2.362013083202686e-05, 0, 0, 0, 0.00016270794884179973, 0, 0, 0, 0.0017266631741779512, 0, 0, 0.0029008132760479427],
                  'entangling_gate_loss_rate': 0.0011091290921765565,
                  'single_qubit_error_rate': np.array([7.10795094e-06, 9.87198084e-04, 3.03794916e-06]),
                  'reset_error_rate': 6.701124116506681e-05, 'measurement_error_rate': 0.0011473113938730995,
                  'reset_loss_rate': 0.0009129238932495187, 'measurement_loss_rate': 0.053940630039510155,
                  'ancilla_idle_loss_rate': 1.7891971787580816e-07,
                  'ancilla_idle_error_rate': np.array([1.37831997e-07, 4.65401621e-08, 2.98365770e-06]),
                  'ancilla_reset_error_rate': 0.016331623957935835, 'ancilla_measurement_error_rate': 0.004982596935810498,
                  'ancilla_reset_loss_rate': 0.00020060212988431113, 'ancilla_measurement_loss_rate': 0.00037245089055685864}

    # Flatten the scaled dictionary into a numpy array for further processing
    baseline = np.array([baseline['idle_loss_rate'], *baseline['idle_error_rate'], *baseline['entangling_zone_error_rate'],
                    *(np.take(baseline['entangling_gate_error_rate'], [0, 1, 2, 14])),
                    baseline['entangling_gate_loss_rate'],
                    *baseline['single_qubit_error_rate'], baseline['reset_error_rate'], baseline['measurement_error_rate'],
                    baseline['reset_loss_rate'], baseline['measurement_loss_rate'],
                    baseline['ancilla_idle_loss_rate'], *baseline['ancilla_idle_error_rate'],
                    baseline['ancilla_reset_error_rate'], baseline['ancilla_measurement_error_rate'],
                    baseline['ancilla_reset_loss_rate'], baseline['ancilla_measurement_loss_rate']]) * 10

    # Set a unique random seed based on the job_id to ensure different initial points for each job
    np.random.seed(job_id)
    
    # Generate a unique initial point for each job based on the random seed
    initial_point = np.random.uniform(low=-np.log(10), high=np.log(10), size=len(baseline))

    def objective(noise_params):
        # Scale the baseline by the noise parameters
        noise_params = baseline / (1 + np.exp(-np.array(noise_params)))
        print(noise_params)
        noise_params = dict(
            idle_loss_rate=noise_params[0],
            idle_error_rate=noise_params[1:4],
            entangling_zone_error_rate=noise_params[4:7],
            entangling_gate_error_rate=[noise_params[7], noise_params[8], noise_params[9], noise_params[7],
                                        0, 0, 0, noise_params[8], 0, 0, 0, noise_params[9], 0, 0, noise_params[10]],
            entangling_gate_loss_rate=noise_params[11],
            single_qubit_error_rate=noise_params[12:15],
            reset_error_rate=noise_params[15],
            measurement_error_rate=noise_params[16],
            reset_loss_rate=noise_params[17],
            measurement_loss_rate=noise_params[18],
            ancilla_idle_loss_rate=noise_params[19],
            ancilla_idle_error_rate=noise_params[20:23],
            ancilla_reset_error_rate=noise_params[23],
            ancilla_measurement_error_rate=noise_params[24],
            ancilla_reset_loss_rate=noise_params[25],
            ancilla_measurement_loss_rate=noise_params[26],
            gate_noise=LogicalCircuit.ancilla_data_differentiated_gate_noise,
            idle_noise=LogicalCircuit.ancilla_data_differentiated_idle_noise
        )

        # Loss decoder parameters
        use_loss_decoding = True
        use_independent_decoder = True
        use_independent_and_first_comb_decoder = False
        simulate_data = False
        detection_events_signs = None
        predictions, observable_flips, _ = Loss_MLE_Decoder_Experiment(
            Meta_params, distance, distance, output_dir, measurement_events, detection_events_signs,
            use_loss_decoding, use_independent_decoder, use_independent_and_first_comb_decoder,
            simulate_data=simulate_data, noise_params=noise_params)

        loss = np.mean(np.logical_xor(predictions, observable_flips)).squeeze()

        # Save intermediate results
        with open(f'intermediate_results_cma_update_noise_model_{decoder_basis}_maddie.pkl', 'ab') as f:
            pickle.dump({'params': noise_params, 'loss': loss}, f)
        print(f"intermediate fidelity = {1 - loss}")

        return loss

    # Perform optimization using CMA-ES with the unique initial point for each job
    xopt, es = cma.purecma.fmin(
        lambda x: objective(x), initial_point, 0.5,
        maxfevals=1000, verb_disp=1, verb_log=1, verb_save=1)

    # Save final results
    with open(f'final_results_gradient_{decoder_basis}_maddie.pkl', 'wb') as f:
        pickle.dump({'params': xopt, 'result': es}, f)

    return xopt, es


# Set parameters
decoder_basis = 'X'
job_id = int(sys.argv[1])  # Job ID should be passed as a command-line argument to ensure unique seed
distance = 5
num_rounds = 5

# Load measurement events
measurement_events = np.load(f'measurement_events_{decoder_basis}_NZZrNr_2024_10_06.npy')
output_dir = '.'

# Split data into training and testing, first half is training
num_of_shots = len(measurement_events) // 2
train_measurement_events = measurement_events[::2]
gate_ordering = ['N', 'Z', 'Zr', 'Nr']

# Set Meta_params and output directory
Meta_params = {
    'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': decoder_basis,
    'bias_preserving_gates': 'False', 'noise': 'atom_array', 'is_erasure_biased': 'False',
    'LD_freq': '1000', 'LD_method': 'None', 'SSR': 'True', 'cycles': str(num_rounds - 1),
    'ordering': gate_ordering, 'decoder': 'MLE', 'circuit_type': 'memory',
    'Steane_type': 'regular', 'printing': 'False', 'num_logicals': '1',
    'loss_decoder': 'independent', 'obs_pos': 'd-1', 'n_r': '0'
}

# Run optimization with job_id to ensure unique seed
result, loss = optimize_theory_fidelity(train_measurement_events, Meta_params, output_dir, decoder_basis, job_id)

# Access and print the final optimized parameters and loss
print(f"xopt = {result}")
final_loss = loss
print(f"Final loss = {final_loss}")
