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
    baseline = {'idle_loss_rate': 4.98160074030583e-08,
                'idle_error_rate': np.array([7.49653788e-09, 3.72326111e-08, 2.67099631e-07]),
                'entangling_zone_error_rate': np.array([2.04672004e-04, 6.23289467e-06, 3.09778205e-03]),
                'entangling_gate_error_rate': [2.0755591589852423e-05, 0.00016984775281617153, 0.0011538858133391043, 2.0755591589852423e-05, 0, 0, 0, 0.00016984775281617153, 0, 0, 0, 0.0011538858133391043, 0, 0, 0.003032903402680309],
                'entangling_gate_loss_rate': 0.0008586229473731896, 'single_qubit_error_rate': np.array([8.23970193e-06, 9.82248672e-04, 1.09953701e-05]),
                'reset_error_rate': 6.892153363873398e-05, 'measurement_error_rate': 0.0015851403653281318, 'reset_loss_rate': 0.0008551699800160818,
                'measurement_loss_rate': 0.030531805469354502, 'ancilla_idle_loss_rate': 1.6532106659092695e-07,
                'ancilla_idle_error_rate': np.array([1.40346453e-07, 4.40995969e-08, 3.57759765e-06]),
                'ancilla_reset_error_rate': 0.023270161141473917, 'ancilla_measurement_error_rate': 0.002600670896675209,
                'ancilla_reset_loss_rate': 0.00013390498123550768, 'ancilla_measurement_loss_rate': 0.0014111332311036796}

    # Flatten the scaled dictionary into a numpy array for further processing
    baseline = np.array([baseline['idle_loss_rate'], *baseline['idle_error_rate'], *baseline['entangling_zone_error_rate'],
                    *(np.take(baseline['entangling_gate_error_rate'], [0, 1, 2, 14])),
                    baseline['entangling_gate_loss_rate'],
                    *baseline['single_qubit_error_rate'], baseline['reset_error_rate'], baseline['measurement_error_rate'],
                    baseline['reset_loss_rate'], baseline['measurement_loss_rate'],
                    baseline['ancilla_idle_loss_rate'], *baseline['ancilla_idle_error_rate'],
                    baseline['ancilla_reset_error_rate'], baseline['ancilla_measurement_error_rate'],
                    baseline['ancilla_reset_loss_rate'], baseline['ancilla_measurement_loss_rate']]) * 3

    # Set a unique random seed based on the job_id to ensure different initial points for each job
    np.random.seed(job_id)
    
    # Generate a unique initial point for each job based on the random seed
    initial_point = np.random.uniform(low=-np.log(10), high=np.log(10), size=len(baseline))

    def objective(noise_params):
        # Scale the baseline by the noise parameters
        noise_params = baseline / (1 + np.exp(-np.array(noise_params)))
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
        print(noise_params)
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

    return xopt, es


# Set parameters
decoder_basis = 'Z'
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
