from BiasedErasure.delayed_erasure_decoders.Experimental_Loss_Decoder import *
import numpy as np
import pickle

# Set parameters
decoder_basis = 'X'
distance = 5
num_rounds = 5

"""# dump information to that file
for _ in range(100):
    with open('intermediate_results_cma_update_noise_model_X_v3.pkl', 'rb') as pickle_file:
        data = pickle_file.readline()
        print(data)
#print(type(params))
for item in data.items():
    print(item)"""

best_loss = 1e9
best_params = None
with (open('intermediate_results_cma_update_noise_model_X_v3.pkl', "rb")) as openfile:
    while True:
        try:
            data = pickle.load(openfile)
            params = data['params']
            loss = data['loss']
            if loss < best_loss:
                best_loss = loss
                best_params = params
        except EOFError:
            break

print(best_loss, best_params)

with (open('intermediate_results_sa_update_noise_model_X_v3.pkl', "rb")) as openfile:
    while True:
        try:
            data = pickle.load(openfile)
            params = data['params']
            loss = data['loss']
            if loss < best_loss:
                best_loss = loss
                best_params = params
        except EOFError:
            break

print(best_loss, best_params)

# Load measurement events
measurement_events = np.load(f'measurement_events_{decoder_basis}_NZZrNr_2024_10_06.npy')#[:1000, :]
train_measurement_events = measurement_events[::2]
test_measurement_events = (measurement_events[1:])[::2]
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


# Now let's decode!
use_loss_decoding = True  # if False: use same DEM every shot, without utilizing SSR.
use_independent_decoder = True  # if False: in every lifecycle, we just apply supercheck at the end. If True: we count the full lifecycle with different potential loss locations and corresponding Clifford propagations.
use_independent_and_first_comb_decoder = False  # This is relevant only if use_independent_decoder=True. If False: use only independent lifecycles. If True: adds a single combination of lifecycles to the decoder.
output_dir = '.'
simulate_data = False
# DO IT
predictions, log_probabilities, observable_flips, dems_list = Loss_MLE_Decoder_Experiment(Meta_params, distance, distance, output_dir,
                                                                    measurement_events,
                                                                    None, use_loss_decoding,
                                                                    use_independent_decoder,
                                                                    use_independent_and_first_comb_decoder,
                                                                    simulate_data=simulate_data, logical_gaps=True,
                                                                    noise_params=best_params)
logical_gaps = np.e**log_probabilities[:, 0] / (np.e**log_probabilities[:, 0] + np.e**log_probabilities[:, 1])

logical_probability = np.mean(np.logical_xor(observable_flips.flatten(), predictions[:, 0].flatten()))
print('infidelity', 1-logical_probability)
np.save('mle_logical_gaps_memory_X_hq', logical_gaps)
np.save('mle_predictions_memory_X_hq', predictions[:, 0])
acceptance_fractions = np.linspace(.05, 1, 20)
order = np.argsort(-logical_gaps.flatten())
# Sort by logical gap
corrected_observables = np.logical_xor(predictions[:, 0], observable_flips.flatten()).flatten()
corrected_observables = corrected_observables[order].flatten()

plt.plot(acceptance_fractions, [1-np.mean(corrected_observables[:int(len(corrected_observables)*acceptance_fraction)]) for acceptance_fraction in acceptance_fractions], marker='o', color='blue', markeredgecolor='navy')
plt.xlabel('Acceptance fraction')
plt.ylabel('Probability of logical $+1$')

plt.show()

