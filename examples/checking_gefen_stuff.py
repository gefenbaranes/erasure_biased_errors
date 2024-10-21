from BiasedErasure.delayed_erasure_decoders.Experimental_Loss_Decoder import *
import numpy as np
# num_rounds = 5
num_layers = 3
num_cxs_per_round = 3
distance = 3
decoder_basis = 'XX'
gate_ordering = ['N', 'Z']
noise_params = {'idle_loss_rate': 2.793300220405646e-07, 'idle_error_rate': np.array([6.60547942e-09, 3.38336163e-08, 2.67533789e-07]),
                'entangling_zone_error_rate': np.array([3.66476387e-04, 6.14732819e-06, 2.35857048e-03]),
                'entangling_gate_error_rate': [2.2260729018707513e-05, 0.00017139584089578063, 0.0012948317242757047, 2.2260729018707513e-05, 0, 0, 0, 0.00017139584089578063, 0, 0, 0, 0.0012948317242757047, 0, 0, 0.002621736717313752],
                'entangling_gate_loss_rate': 0.00039272255674060926, 'single_qubit_error_rate': np.array([1.53681034e-05, 9.93583065e-04, 1.94650113e-05]),
                'reset_error_rate': 5.89409983290463e-05, 'measurement_error_rate': 0.0006138700821647161, 'reset_loss_rate': 0.0007531131027610011, 'measurement_loss_rate': 0.07131074481520218, 'ancilla_idle_loss_rate': 1.6989311035347498e-07,
                'ancilla_idle_error_rate': np.array([1.46727589e-07, 4.60893305e-08, 2.30298714e-06]), 'ancilla_reset_error_rate': 0.024549181355318986, 'ancilla_measurement_error_rate': 0.0012815874700447462, 'ancilla_reset_loss_rate': 0.00019528486460263086, 'ancilla_measurement_loss_rate': 0.00047357577582906143,
                'gate_noise': LogicalCircuit.ancilla_data_differentiated_gate_noise, 'idle_noise': LogicalCircuit.ancilla_data_differentiated_idle_noise}


Meta_params = {'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': decoder_basis,
            'bias_preserving_gates': 'False',
            'noise': 'atom_array', 'is_erasure_biased': 'False', 'LD_freq': '1000', 'LD_method': 'None',
            'SSR': 'True', 'cycles': str(num_layers - 1),
            'ordering': gate_ordering,
            'decoder': 'MLE',
            'circuit_type': f'logical_CX_NL{num_layers}_NCX{num_cxs_per_round}', 'Steane_type': 'None', 'printing': 'True', 'num_logicals': '2',
            'loss_decoder': 'independent',
            'obs_pos': 'd-1', 'n_r': '0'}



# Load the experimental measurements

simulate_data = True

if simulate_data:
    detection_events_signs = None
    exp_measurements = None
    num_shots = 150

else:
    # Load the theory circuit
    _, _, _, circuit = get_simulated_measurement_events(Meta_params, distance, distance, 1, noise_params)
    # Use the theory circuit to get the detection events and observable flips corresponding to the exp data
    detection_events, observable_flips = circuit.compile_m2d_converter().convert(measurements=exp_measurements.astype(bool), separate_observables=True)
    # Find detection event signs
    detection_events_signs = -np.sign(2*np.nanmean(detection_events.astype(int), axis=0)-1).astype(int)
    exp_measurements = np.load(f'/Users/gefenbaranes/Documents/2024_10_15_measurement_events_1CNOT_XX.npy')
    exp_measurements = np.concatenate([exp_measurements[:, 0, :distance**2-1],
                                    exp_measurements[:, 1, :distance**2-1],
                                    exp_measurements[:, 0, distance**2-1:2*(distance**2-1)],
                                    exp_measurements[:, 1, distance**2-1:2*(distance**2-1)],
                                    exp_measurements[:, 0, 2*(distance**2-1):],
                                    exp_measurements[:, 1, 2*(distance**2-1):]], axis=1)


# Now let's decode!
use_loss_decoding = True  # if False: use same DEM every shot, without utilizing SSR.
use_independent_decoder = True  # if False: in every lifecycle, we just apply supercheck at the end. If True: we count the full lifecycle with different potential loss locations and corresponding Clifford propagations.
use_independent_and_first_comb_decoder = False  # This is relevant only if use_independent_decoder=True. If False: use only independent lifecycles. If True: adds a single combination of lifecycles to the decoder.
output_dir = '.'
# DO IT
predictions, observable_flips, dems_list = Loss_MLE_Decoder_Experiment(Meta_params, distance, distance, output_dir,
                                                                    exp_measurements,
                                                                    detection_events_signs, use_loss_decoding,
                                                                    use_independent_decoder,
                                                                    use_independent_and_first_comb_decoder,
                                                                    simulate_data=simulate_data, logical_gaps=False,
                                                                    noise_params=noise_params, num_shots=num_shots)
logical_probability = np.mean(np.logical_xor(observable_flips, predictions))
print('/n logical error', logical_probability)


# error bar: (np.sqrt(P*(1-P)/num_shots))