from BiasedErasure.delayed_erasure_decoders.Experimental_Loss_Decoder import *
import numpy as np

num_rounds = 3
distance = 5
decoder_basis = 'XX'
gate_ordering = ['N', 'Z']
noise_params = {}
Meta_params = {'architecture': 'CBQC', 'code': 'Rotated_Surface', 'logical_basis': decoder_basis,
            'bias_preserving_gates': 'False',
            'noise': 'atom_array', 'is_erasure_biased': 'False', 'LD_freq': '1000', 'LD_method': 'None',
            'SSR': 'True', 'cycles': str(num_rounds - 1),
            'ordering': gate_ordering,
            'decoder': 'MLE',
            'circuit_type': 'logical_CX_3', 'Steane_type': 'None', 'printing': 'True', 'num_logicals': '2',
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
exp_measurements, detection_events, observable_flips, circuit = get_simulated_measurement_events(Meta_params, distance, distance, 1, noise_params)
# Use the theory circuit to get the detection events and observable flips corresponding to the exp data
detection_events, observable_flips = circuit.compile_m2d_converter().convert(measurements=exp_measurements.astype(bool), separate_observables=True)
# Find detection event signs
detection_events_signs = -np.sign(2*np.nanmean(detection_events.astype(int), axis=0)-1).astype(int)

# Now let's decode!
use_loss_decoding = True  # if False: use same DEM every shot, without utilizing SSR.
use_independent_decoder = True  # if False: in every lifecycle, we just apply supercheck at the end. If True: we count the full lifecycle with different potential loss locations and corresponding Clifford propagations.
use_independent_and_first_comb_decoder = False  # This is relevant only if use_independent_decoder=True. If False: use only independent lifecycles. If True: adds a single combination of lifecycles to the decoder.
output_dir = '.'
simulate_data = False
# DO IT
predictions, observable_flips, dems_list = Loss_MLE_Decoder_Experiment(Meta_params, distance, distance, output_dir,
                                                                    exp_measurements,
                                                                    detection_events_signs, use_loss_decoding,
                                                                    use_independent_decoder,
                                                                    use_independent_and_first_comb_decoder,
                                                                    simulate_data=simulate_data, logical_gaps=False,
                                                                    noise_params=noise_params)


logical_probability = np.mean(np.logical_xor(observable_flips, predictions))

print('infidelity', 1-logical_probability)
