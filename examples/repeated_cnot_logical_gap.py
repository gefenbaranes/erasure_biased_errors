from BiasedErasure.delayed_erasure_decoders.Experimental_Loss_Decoder import *
import numpy as np

num_rounds = 3
distance = 5
decoder_basis = 'XX'
gate_ordering = ['N', 'Z']
num_cxs_per_round = 3
noise_params = {'idle_loss_rate': 2.1462892652881424e-07, 'idle_error_rate': np.array([5.31106535e-09, 2.59649716e-08, 2.70017446e-07]), 'entangling_zone_error_rate': np.array([3.22871520e-04, 5.55115000e-06, 1.28240286e-03]), 'entangling_gate_error_rate': [1.8729598643991336e-05, 0.00016597465639499589, 0.0013401575256883555, 1.8729598643991336e-05, 0, 0, 0, 0.00016597465639499589, 0, 0, 0, 0.0013401575256883555, 0, 0, 0.0026654438378731237], 'entangling_gate_loss_rate': 0.0012268907363777474, 'single_qubit_error_rate': np.array([9.01549152e-06, 8.45064836e-04, 1.91825416e-05]), 'reset_error_rate': 0.00013112864576086654, 'measurement_error_rate': 0.003220085408683493, 'reset_loss_rate': 0.0007849977760100565, 'measurement_loss_rate': 0.06657247422436202, 'ancilla_idle_loss_rate': 1.7048289168299613e-07, 'ancilla_idle_error_rate': np.array([1.30011070e-07, 3.79578658e-08, 3.73757626e-06]), 'ancilla_reset_error_rate': 0.02267054400731952, 'ancilla_measurement_error_rate': 0.011477399332064406, 'ancilla_reset_loss_rate': 0.00014151808789913066, 'ancilla_measurement_loss_rate': 0.0004062050339110557,
                'gate_noise':LogicalCircuit.ancilla_data_differentiated_gate_noise,
            'idle_noise':LogicalCircuit.ancilla_data_differentiated_idle_noise}

Meta_params = {'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': decoder_basis,
            'bias_preserving_gates': 'False',
            'noise': 'atom_array', 'is_erasure_biased': 'False', 'LD_freq': '1000', 'LD_method': 'None',
            'SSR': 'True', 'cycles': str(num_rounds - 1),
            'ordering': gate_ordering,
            'decoder': 'MLE',
            'circuit_type': f'logical_CX_NL{num_rounds}_NCX{num_cxs_per_round}',
               'Steane_type': 'None', 'printing': 'False', 'num_logicals': '2',
            'loss_decoder': 'independent',
            'obs_pos': 'd-1', 'n_r': '0'}


# Load the experimental measurements
exp_measurements = np.load('2024_10_15_measurement_events_1CNOT_XX.npy')#[:100, :]
exp_measurements = np.concatenate([exp_measurements[:, 0, :distance**2-1],
                                   exp_measurements[:, 1, :distance**2-1],
                                   exp_measurements[:, 0, distance**2-1:2*(distance**2-1)],
                                   exp_measurements[:, 1, distance**2-1:2*(distance**2-1)],
                                   exp_measurements[:, 0, 2*(distance**2-1):],
                                   exp_measurements[:, 1, 2*(distance**2-1):]], axis=1)

# Load the theory circuit
simulated_measurements, simulated_detection_events, simulated_observable_flips, circuit = get_simulated_measurement_events(Meta_params, distance, distance, len(exp_measurements), noise_params)
# Use the theory circuit to get the detection events and observable flips corresponding to the exp data
print(np.mean(simulated_detection_events))
detection_events, observable_flips = circuit.compile_m2d_converter().convert(measurements=exp_measurements.astype(bool), separate_observables=True)
# Find detection event signs
detection_events_signs = -np.sign(2*np.nanmean(detection_events.astype(int), axis=0)-1).astype(int)
detection_events = detection_events.astype(int)
detection_events[detection_events > .5] = 1-detection_events[detection_events > .5]
print(np.mean(detection_events))

simulated_measurements_loaded = np.load('measurement_events_CX_converted.npy')
plt.hist(np.sum(simulated_measurements_loaded == 2, axis=1).flatten(), bins=np.arange(16), color='purple', alpha=0.5, label='theory other')
plt.hist(np.sum(simulated_measurements == 2, axis=1).flatten(), bins=np.arange(16), color='blue', alpha=0.5, label='theory')
plt.hist(np.sum(exp_measurements == 2, axis=1).flatten(), bins=np.arange(16), color='red', alpha=0.5, label='exp')
plt.ylabel('number of events')
plt.xlabel('number of losses')
plt.show()

# Now let's decode!
use_loss_decoding = True  # if False: use same DEM every shot, without utilizing SSR.
use_independent_decoder = True  # if False: in every lifecycle, we just apply supercheck at the end. If True: we count the full lifecycle with different potential loss locations and corresponding Clifford propagations.
use_independent_and_first_comb_decoder = False  # This is relevant only if use_independent_decoder=True. If False: use only independent lifecycles. If True: adds a single combination of lifecycles to the decoder.
output_dir = '.'
simulate_data = False
# DO IT
"""predictions_exp, log_probabilities_exp, observable_flips_exp, dems_list = Loss_MLE_Decoder_Experiment(Meta_params, distance, distance, output_dir,
                                                                    exp_measurements, #[:100, :],
                                                                    detection_events_signs, use_loss_decoding,
                                                                    use_independent_decoder,
                                                                    use_independent_and_first_comb_decoder,
                                                                    simulate_data=simulate_data, logical_gaps=True,
                                                                    noise_params=noise_params)

corrected_exp = np.logical_xor(observable_flips_exp.flatten(), predictions_exp[:, 0].flatten())
logical_probability_exp = np.mean(corrected_exp)
np.save('corrected_exp', corrected_exp)
np.save('log_probabilities_exp', log_probabilities_exp)
np.save('observable_flips_exp', observable_flips_exp)
print('infidelity exp', 1-logical_probability_exp)"""

# DO IT
"""predictions_theory, log_probabilities_theory, observable_flips_theory, dems_list = Loss_MLE_Decoder_Experiment(Meta_params, distance, distance, output_dir,
                                                                    simulated_measurements,
                                                                    None, use_loss_decoding,
                                                                    use_independent_decoder,
                                                                    use_independent_and_first_comb_decoder,
                                                                    simulate_data=False, logical_gaps=True,
                                                                    noise_params=noise_params)

corrected_theory = np.logical_xor(observable_flips_theory.flatten(), predictions_theory[:, 0].flatten())
logical_probability_theory = np.mean(corrected_theory)

np.save('predictions_theory', predictions_theory)
np.save('log_probabilities_theory', log_probabilities_theory)
np.save('observable_flips_theory', observable_flips_theory)
np.save('corrected_theory', corrected_theory)
print('infidelity theory', 1-logical_probability_theory)"""



predictions_theory, observable_flips_theory, dems_list = Loss_MLE_Decoder_Experiment(Meta_params, distance, distance, output_dir,
                                                                    exp_measurements, detection_events_signs,
                                                                    use_loss_decoding,
                                                                    use_independent_decoder,
                                                                    use_independent_and_first_comb_decoder,
                                                                    simulate_data=False, logical_gaps=False,
                                                                    noise_params=noise_params)

corrected_theory = np.logical_xor(observable_flips_theory.flatten(), predictions_theory.flatten())
logical_probability_theory = np.mean(corrected_theory)

print('infidelity theory', 1-logical_probability_theory)

