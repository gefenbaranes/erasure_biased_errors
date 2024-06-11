import numpy as np
import qec
import pymatching, stim


def ghz(d, eta, order, num_qubits=3, measure='x'):
    assert measure in ['x', 'z']
    p = 0.001
    logical_qubits = [qec.surface_code.RotatedSurfaceCode(d, d) for _ in range(num_qubits)]
    lc = qec.LogicalCircuit(logical_qubits, initialize_circuit=False,
                            loss_noise_scale_factor=0.0, spam_noise_scale_factor=1,
                            gate_noise_scale_factor=1, idle_noise_scale_factor=0,
                            measurement_error_rate=1e-7,
                            reset_error_rate=0,
                            entangling_error_rate=(eta * p / (1+eta) / 2, eta * p / (1+eta) / 2, eta * p / (1+eta)),
                            single_qubit_error_rate=(0, 0, 0))

    # First prepare all logical qubits in |0>
    lc.append(qec.surface_code.prepare_zero, list(range(0, len(logical_qubits))))

    # Rotate noiselessly
    lc.append(qec.surface_code.rotate_code, 0)

    # Hadamard all qubits
    lc.append(qec.surface_code.global_h, list(range(len(logical_qubits))), move_duration=0)

    for _ in range(len(order)):
        lc.append(qec.surface_code.global_cz, [order[_][0], order[_][1]], move_duration=0)
        lc.append(qec.surface_code.global_h, order[_][1], move_duration=0)

    if measure == 'x':
        lc.append(qec.surface_code.measure_x, list(range(len(logical_qubits))), observable_include=False)
        lc.append('MOVE_TO_NO_NOISE', lc.qubit_indices, 0)
        global_x = []
        for index in range(len(lc.logical_qubits)):
            global_x += lc.logical_qubits[index].logical_x_operator + [stim.GateTarget(stim.target_combiner())]
        lc.append('MPP', global_x[:-1])
        lc.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], 0)

    elif measure == 'z':
        lc.append(qec.surface_code.measure_z, list(range(len(logical_qubits))), observable_include=False)
        lc.append('MOVE_TO_NO_NOISE', lc.qubit_indices, 0)

        for index in range(len(logical_qubits) - 1):
            lc.append('MPP', lc.logical_qubits[index].logical_z_operator + [stim.GateTarget(stim.target_combiner())] +
                      lc.logical_qubits[index + 1].logical_z_operator)
            lc.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], lc.num_observables)

    return lc



#biases = 10**np.linspace(0, 2., 10)
d = 7

z_fidelities_1 = []
x_fidelities_1 = []
z_fidelities_2 = []
x_fidelities_2 = []

for n in np.arange(2, 8, 1):
    order_1 = []
    order_2 = []
    for i in range(1, n):
        order_1.append((0, i))
        order_2.append((i - 1, i))

    orders = [order_1, order_2]
    print('GHZ state of {} logical qubits with d = {}'.format(n, d))
    for (o, order) in enumerate(orders):
        for bias in [1/2]:
            for measure in ['x', 'z']:
                num_shots = 10000
                lc = ghz(d, bias, order, num_qubits=n, measure=measure)
                dem = lc.without_loss().detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)

                # Sample the circuit.
                shots = lc.compile_sampler().sample(num_shots)
                detector_shots, observable_shots = lc.without_loss().compile_m2d_converter().convert(measurements=shots,
                                                                                                     separate_observables=True)

                prediction = qec.correlated_decoders.mle.decode_gurobi_with_dem(dem, detector_shots)
                corrected_observables = 1 - np.logical_xor(observable_shots, prediction)
                if measure == 'x' and o == 0:
                    x_fidelities_1.append(np.mean(corrected_observables))
                elif measure == 'x' and o == 1:
                    x_fidelities_2.append(np.mean(corrected_observables))
                elif measure == 'z' and o == 0:
                    z_fidelities_1.append(np.mean(np.alltrue(corrected_observables == np.ones(n-1), axis=1)))
                elif measure == 'z' and o == 1:
                    z_fidelities_2.append(np.mean(np.alltrue(corrected_observables == np.ones(n-1), axis=1)))


    print(x_fidelities_1)
    print(x_fidelities_2)
    print(z_fidelities_1)
    print(z_fidelities_2)
