import copy

import numpy as np
import stim

import qec
from qec import LogicalCode, tools
from typing import List, Union, Optional, Callable, Iterable, Sized
import random


class LogicalCircuit(stim.Circuit):
    def __init__(self, logical_qubits: List[LogicalCode],
                gate_noise: Optional[
                    Callable[[Union[stim.CircuitInstruction, stim.CircuitRepeatBlock]], stim.Circuit]] = None,
                idle_noise: Optional[
                    Callable[[Union[stim.CircuitInstruction, stim.CircuitRepeatBlock]], stim.Circuit]] = None,
                idle_noise_scale_factor: float = 1.0,
                gate_noise_scale_factor: float = 1.0,
                loss_noise_scale_factor: float = 1.0,
                spam_noise_scale_factor: float = 1.0,
                idle_loss_rate: float = 1e-7,
                idle_error_rate: tuple = (5e-6/25, 5e-6/25, 2e-5/25),
                entangling_zone_error_rate: tuple = (.002/4, .002/4, .005/4),
                entangling_gate_error_rate: tuple = (.002 / 4, .002 / 4, .0025 / 4, .002 / 4, 0, 0, 0, .002 / 4, 0, 0, 0, .0025 / 4, 0, 0, .005 / 4), 
                erasure_ratio: float = 0.0,
                entangling_gate_loss_rate: float = .005/4,
                single_qubit_loss_rate: float = 0.0,
                single_qubit_error_rate: tuple = (1e-4, 1e-4, 1e-4),
                reset_error_rate: float = 0.003, measurement_error_rate: float = 0.004,
                reset_loss_rate: float = 0,
                initialize_circuit: bool = True,
                atom_array_sim: bool = False):
        """
        Generates a LogicalCircuit, which is a subclass of stim.Circuit() but with additional attributes that are
        atom-array specific.
        :param logical_qubits: A list of qec.LogicalCode objects corresponding to the logical qubits in the circuit.
        :param gate_noise: A function that takes in a stim CircuitInstruction, and decorates it with noise (defaults
        to self.standard_gate_noise)
        :param idle_noise: A function that takes in a stim CircuitInstruction, and decorates it with noise (defaults
        to self.standard_idle_noise)
        :param idle_loss_rate: The rate at which qubits are lost during atom movements, in loss per qubit per us)
        :param idle_error_rate: The rate at which qubits accrue errors during movements, in error probability per qubit
        per us)
        :param entangling_gate_error_rate: The rate at which qubits accrue errors during entangling operations, in error
        probability per qubit per us)
        :param entangling_gate_loss_rate: The rate at which qubits are lost during an entangling operation
        :param single_qubit_loss_rate: The rate at which qubits are lost during a single-qubit gate operation
        :param single_qubit_error_rate:
        :param reset_error_rate: The probability that a qubit is initialized in |1> instead of |0>
        :param measurement_error_rate: The probability that an atom is flipped before it is measured in the Z basis
        :param reset_loss_rate: The probability that an atom is lost when it is reset (initialized in |0>)
        :param initialize_circuit: True if qubit coordinate initialization is done at the beginning of the circuit
        """
        super().__init__()
        self.logical_qubits = list(logical_qubits)
        self.num_logical_qubits = len(self.logical_qubits)
        self.logical_qubit_indices = np.arange(self.num_logical_qubits)
        LogicalCircuit.gate_noise = gate_noise
        LogicalCircuit.idle_noise = idle_noise
        
        self.atom_array_sims = atom_array_sim # if True: we work with zones. False: no zones, only errors after operations on the targets
        
        if gate_noise is None:
            if self.atom_array_sims:
                self.gate_noise = self.standard_gate_noise
            else:
                self.gate_noise = self.biased_erasure_entangling_noise
        if idle_noise is None:
            if self.atom_array_sims:
                self.idle_noise = self.standard_idle_noise
            else:
                self.idle_noise = self.no_idle_noise

        # Rename physical indices to account for multiple logical qubits
        for (_, logical_qubit) in enumerate(self.logical_qubits):
            if _ == 0:
                logical_qubit.qubit_indices = np.arange(logical_qubit.qubit_number)
            else:
                logical_qubit.qubit_indices = np.arange(self.logical_qubits[_ - 1].qubit_indices[-1] + 1,
                                                        self.logical_qubits[_ - 1].qubit_indices[-1] +
                                                        logical_qubit.qubit_number + 1)

        if len(self.logical_qubits) > 0:
            self.qubit_indices = np.concatenate([logical_qubit.qubit_indices
                                                for logical_qubit in self.logical_qubits])
        else:
            self.qubit_indices = np.array([])

        # Zones - relevant for atom array sims:
        self.entangling_zone = set()
        self.storage_zone = set()
        self.lost_qubits = set()
        self.no_noise_zone = set(self.qubit_indices)
        
        self.idle_noise_scale_factor = idle_noise_scale_factor
        self.gate_noise_scale_factor = gate_noise_scale_factor
        self.loss_noise_scale_factor = loss_noise_scale_factor
        self.spam_noise_scale_factor = spam_noise_scale_factor
        self.idle_loss_rate = idle_loss_rate
        self.idle_error_rate = idle_error_rate
        self.entangling_zone_error_rate = entangling_zone_error_rate
        self.entangling_gate_error_rate = entangling_gate_error_rate
        self.entangling_gate_loss_rate = entangling_gate_loss_rate
        self.erasure_ratio = erasure_ratio
        self.reset_error_rate = reset_error_rate
        self.reset_loss_rate = reset_loss_rate
        self.single_qubit_error_rate = single_qubit_error_rate
        self.single_qubit_loss_rate = single_qubit_loss_rate
        self.measurement_error_rate = measurement_error_rate
        self._without_loss = stim.Circuit()

        # Handling loss:
        self.potential_lost_qubits = np.array([], dtype=np.int32)
        self.loss_probabilities = np.array([], dtype=np.float32)
        
        
        
        if initialize_circuit:
            self.initialize_circuit()

    def initialize_circuit(self, circuit: Optional[stim.Circuit] = None):
        if circuit is None:
            for logical_qubit in self.logical_qubits:
                super(LogicalCircuit, self).__iadd__(logical_qubit.qubit_coordinates())
        else:
            for logical_qubit in self.logical_qubits:
                circuit += logical_qubit.qubit_coordinates()
            return circuit

    def qubit_index_to_logical_qubit_index(self, index: int):
        for (_, lq) in enumerate(self.logical_qubits):
            if index in lq.qubit_indices:
                return _
        raise Exception('No logical qubit found with corresponding physical index')

    def biased_erasure_entangling_noise(self, operation: stim.CircuitInstruction, add_to_potential_loss: bool = True) -> stim.Circuit:
        """
        GB: biased_erasure noise channel - noise after 2q gates.
        """
        noisy_circuit = stim.Circuit()
        noisy_circuit.append(operation)
        if qec.is_entangling(operation.name):
            targets = [t.value for t in operation.targets_copy()]
            
            # put errors only on data qubits - debugging:
            # data_qubits = []
            # for lq in self.logical_qubits:
            #     data_qubits.extend(lq.data_qubits)
            # targets = [q for q in targets if q in data_qubits]
            
            # put errors only on ancilla qubits - debugging:
            # ancilla_qubits = []
            # for lq in self.logical_qubits:
            #     ancilla_qubits.extend(lq.measure_qubits)
            # targets = [q for q in targets if q in ancilla_qubits]
            
            
            # 2nd approach - put I channel all the time, to not make mistakes with the SWAP
            # Randomly sample which qubit gets loss and which qubit gets Pauli.
            err_types = random.choices(['loss','pauli'], weights=[self.erasure_ratio,(1-self.erasure_ratio)], k=int(len(targets)))

            # Pauli noise channel:
            pauli_targets = [target for target, err_type in zip(targets, err_types) if err_type == 'pauli'] # take all targets with 'pauli' errors:
            pauli_error = np.array(self.entangling_gate_error_rate) * self.gate_noise_scale_factor
            if sum(pauli_error) > 0 and len(pauli_targets)> 0:
                if len(pauli_error) == 3:
                    noisy_circuit.append('PAULI_CHANNEL_1', pauli_targets, pauli_error)
                elif len(pauli_error) == 8:
                    noisy_circuit.append('PAULI_CHANNEL_2', pauli_targets, pauli_error)
            
            # 1q loss channel on each qubit that did the gate: (TODO: change this to put I channel on all qubits "targets" but adjust the self.loss_probabilities to be loss_prob multiply by 1 if 'loss' on this qubit or 0 is 'pauli' on this qubit
            loss_targets = [target for target, err_type in zip(targets, err_types) if err_type == 'loss'] # take all other targets:
            loss_prob = self.entangling_gate_loss_rate * self.loss_noise_scale_factor
            if add_to_potential_loss and loss_prob>0:
                # Apply I channel to all targets to simulate identity operation
                noisy_circuit.append('I', targets)

                # Update the potential lost qubits list and loss probabilities
                for target, err_type in zip(targets, err_types):
                    if err_type == 'loss':
                        loss_prob = self.entangling_gate_loss_rate * self.loss_noise_scale_factor
                        self.potential_lost_qubits = np.append(self.potential_lost_qubits, target)
                        self.loss_probabilities = np.append(self.loss_probabilities, loss_prob)
                    else:
                        # If it's not a loss, append zero to maintain the correct indices
                        self.potential_lost_qubits = np.append(self.potential_lost_qubits, target)
                        self.loss_probabilities = np.append(self.loss_probabilities, 0)

            
        return noisy_circuit

    def standard_gate_noise(self, operation: stim.CircuitInstruction, add_to_potential_loss:bool = False) -> stim.Circuit:
        assert self.atom_array_sims
        
        noisy_circuit = stim.Circuit()

        # Add measurement noise before operation
        if qec.is_measurement(operation.name):
            measurement_noise = [t.value for t in operation.targets_copy() if t.value not in self.no_noise_zone]
            if len(measurement_noise) != 0:
                noisy_circuit.append('X_ERROR', measurement_noise,
                                     [self.measurement_error_rate * self.spam_noise_scale_factor])

        noisy_circuit.append(operation)

        # Add other errors after operation
        if qec.is_entangling(operation.name):
            # Add noise to all qubits in the entangling zone
            targets = [t.value for t in operation.targets_copy() if t.value not in self.no_noise_zone]
            # qubits that undergo an entangling gate, get a 2q channel:
            noisy_circuit.append('PAULI_CHANNEL_2', targets,
                                np.array(self.entangling_gate_error_rate) * self.gate_noise_scale_factor)
            # qubits that don't undergo an entangling gate but are in the entangling zone, get a 1q channel:
            residual_entangling_zone_qubits = self.entangling_zone - set(targets)
            noisy_circuit.append('PAULI_CHANNEL_1', residual_entangling_zone_qubits,
                                np.array(self.entangling_zone_error_rate) * self.gate_noise_scale_factor)
            # qubits in the entangling zone also get loss:
            qubits = np.setdiff1d(list(self.entangling_zone), list(self.no_noise_zone)) # entangling zone minus no noise zone
            loss_prob = self.entangling_gate_loss_rate * self.loss_noise_scale_factor
            if loss_prob > 0 and add_to_potential_loss:
                noisy_circuit.append('I', qubits)
                self.potential_lost_qubits = np.append(self.potential_lost_qubits, qubits)
                self.loss_probabilities = np.append(self.loss_probabilities, np.full(len(qubits), loss_prob))     


        if qec.is_reset(operation.name):
            # noisy_circuit.append('X_ERROR', list(self.entangling_zone), [self.reset_error_rate * self.spam_noise_scale_factor])
            noisy_circuit.append('X_ERROR', list(operation.targets_copy()), [self.reset_error_rate * self.spam_noise_scale_factor]) # GB's new change - only initialized qubits get the error
            qubits = np.setdiff1d(list(self.qubit_indices), list(self.no_noise_zone))
            loss_prob = self.reset_loss_rate * self.loss_noise_scale_factor
            if loss_prob > 0 and add_to_potential_loss:
                noisy_circuit.append('I', qubits)
                self.potential_lost_qubits = np.append(self.potential_lost_qubits, qubits)
                self.loss_probabilities = np.append(self.loss_probabilities, np.full(len(qubits), loss_prob))


        if qec.is_single_qubit(operation.name):
            # Add single qubit noise only on the qubits we do the operation on
            noisy_circuit.append('PAULI_CHANNEL_1', [t.value for t in operation.targets_copy() if t.value
                                                    not in self.no_noise_zone],
                                 np.array(self.single_qubit_error_rate) * self.gate_noise_scale_factor)
            qubits = np.setdiff1d(list(self.qubit_indices), list(self.no_noise_zone))
            loss_prob = self.single_qubit_loss_rate * self.loss_noise_scale_factor
            if loss_prob > 0 and add_to_potential_loss:
                noisy_circuit.append('I', qubits) # loss probability: self.entangling_gate_loss_rate * self.loss_noise_scale_factor
                self.potential_lost_qubits = np.append(self.potential_lost_qubits, qubits)
                self.loss_probabilities = np.append(self.loss_probabilities, np.full(len(qubits), loss_prob))
            
        return noisy_circuit
    
    
    def standard_idle_noise(self, duration: Union[float, int], add_to_potential_loss:bool) -> stim.Circuit:
        noisy_circuit = stim.Circuit()
        if duration != 0:
            # noisy_circuit.append('TICK', ())
            noisy_circuit.append('PAULI_CHANNEL_1', self.entangling_zone | self.storage_zone, (1 -
                                 (1 - np.array(self.idle_error_rate)) ** duration) * self.idle_noise_scale_factor)
            # Add per qubit loss noise
            loss_prob = (1 - (1 - self.idle_loss_rate) ** duration) * self.loss_noise_scale_factor
            qubits = np.setdiff1d(list(self.qubit_indices), list(self.no_noise_zone))
            if loss_prob > 0 and add_to_potential_loss:
                noisy_circuit.append('I', qubits) # loss probability: self.entangling_gate_loss_rate * self.loss_noise_scale_factor
                self.potential_lost_qubits = np.append(self.potential_lost_qubits, qubits)
                self.loss_probabilities = np.append(self.loss_probabilities, np.full(len(qubits), loss_prob))

        return noisy_circuit


    def add_idle_noise(self, duration: Union[float, int], add_to_potential_loss: bool) -> stim.Circuit:
        return self.idle_noise(duration, add_to_potential_loss)
    
    def no_idle_noise(self, duration: Union[float, int]) -> stim.Circuit:
        return stim.Circuit()
    

    def add_noise(self, circuit: stim.Circuit, add_to_potential_loss: bool) -> stim.Circuit:
        noisy_circuit = stim.Circuit()
        for operation in circuit:
            if isinstance(operation, stim.CircuitRepeatBlock):
                noisy_circuit += self.add_noise(operation.body_copy(), add_to_potential_loss)
            else:
                assert isinstance(operation, stim.CircuitInstruction)
                noisy_circuit += self.gate_noise(operation, add_to_potential_loss)
        return noisy_circuit

    def _generate_detector(self, pauli: stim.PauliString, noiseless_circuit=None):
        # Look at the last time that the measure qubit was measured
        if noiseless_circuit is None:
            noiseless_circuit = self.without_noise()
        #print('New')
        #print(pauli)
        n_measurements = len(noiseless_circuit[-1].targets_copy())
        c_t = stim.Circuit()
        c_t = self.initialize_circuit(c_t)
        # Generate Pauli measurement corresponding to current stabilizer
        recs = []
        for i in range(2, len(noiseless_circuit)):
            if qec.is_measurement(noiseless_circuit[-i].name):
                # Find the measured stabilizers
                # measured_qubits = [target.value for target in noiseless_circuit[-i].targets_copy()]
                if noiseless_circuit[-i].name == 'MPP':
                    measured_qubits = np.array([target.value for target in noiseless_circuit[-i].targets_copy()])[::2]
                else:
                    measured_qubits = [target.value for target in noiseless_circuit[-i].targets_copy()]
                    
                # Find the propagated stabilizer
                t = stim.Tableau.from_circuit(c_t, ignore_measurement=True, ignore_noise=True, ignore_reset=True)
                propagated_stabilizer = np.array(t(stim.PauliString(pauli)))
                new_pauli = propagated_stabilizer.copy()
                #print(stim.PauliString(new_pauli))
                measured_stabilizers = []
                propagated_stabilizers = []
                # Figure out what stabilizers we propagated to
                for lq in self.logical_qubits:
                    for mq in lq.measure_qubits:
                        dq = [d for d in lq.measure_to_data[mq] if d is not None]
                        if mq in lq.measure_qubits_x:
                            if np.alltrue(np.isin(propagated_stabilizer[dq], [1, 2])):
                                propagated_stabilizers.append(mq)
                                if mq in measured_qubits:
                                    measured_stabilizers.append(mq)
                                    for d in dq:
                                        if new_pauli[d] == 1:
                                            new_pauli[d] = 0
                                        elif new_pauli[d] == 2:
                                            new_pauli[d] = 3

                        elif mq in lq.measure_qubits_z:
                            if np.alltrue(np.isin(propagated_stabilizer[dq], [3, 2])):
                                propagated_stabilizers.append(mq)
                                if mq in measured_qubits:
                                    measured_stabilizers.append(mq)
                                    for d in dq:
                                        if new_pauli[d] == 3:
                                            new_pauli[d] = 0
                                        elif new_pauli[d] == 2:
                                            new_pauli[d] = 1
                        new_pauli[mq] = 0

                # Now we need to find the corresponding measurement record
                # Because we've measured these stabilizers, we want to multiply them into the tableau to cancel
                # them out.
                # n_measurements += len(noiseless_circuit[-i].targets_copy())
                if noiseless_circuit[-i].name == 'MPP':
                    n_measurements += 1
                else:
                    n_measurements += len(noiseless_circuit[-i].targets_copy())
                #print(stim.PauliString(new_pauli))
                if len(measured_stabilizers) > 0:
                    for (_, s) in enumerate(measured_stabilizers):
                        if s in measured_qubits:
                            recs.append(-(n_measurements +
                                        (self.without_noise().num_measurements - noiseless_circuit.num_measurements)
                                        - np.argwhere(measured_qubits == s)[0, 0]))
                    if np.all(new_pauli == np.zeros_like(propagated_stabilizer)):
                        return recs
                    #elif np.all(new_pauli == pauli) and np.sum(pauli!=0):
                    #    break
                    else:
                        recs = recs + self._generate_detector(stim.PauliString(new_pauli),
                                                            noiseless_circuit=noiseless_circuit[:-i + 1])
                        break
            c_t_current = stim.Circuit()
            c_t_current.append(noiseless_circuit[-i])
            # Try to add the inverse instruction. If it's not possible, then it's a measurement, reset, or detector.
            # Then we just want to add the circuit element
            try:
                c_t += (c_t_current.inverse())
            except:
                c_t += c_t_current

        return recs

    def append(self, name: object, targets: object = (), arg: object = None, **kwargs) -> None:
        if callable(name):
            name(self, targets, **kwargs)

        # Handle noise from movement
        elif name in ['MOVE', 'MOVE_TO_STORAGE', 'MOVE_TO_ENTANGLING', 'MOVE_TO_NO_NOISE']:
            if not self.atom_array_sims:
                pass
            else:
                assert isinstance(targets, int) or isinstance(targets, Iterable)
                assert isinstance(arg, int) or isinstance(arg, float) or arg is None
                if not(arg is None):
                    if name == 'MOVE_TO_STORAGE':
                        self.entangling_zone = self.entangling_zone - set(targets)
                        self.no_noise_zone = self.no_noise_zone - set(targets)
                        self.storage_zone = self.storage_zone | set(targets)

                        # Check if adding noise from the movement is actually necessary 
                        if not set(targets).issubset(self.storage_zone): #GB: isn't this always False? because we just more the targets into the storage_zone
                            super().__iadd__(self.add_idle_noise(arg, add_to_potential_loss = True))
                            idle_loss_rate, self.idle_loss_rate = self.idle_loss_rate, 0.
                            self._without_loss += self.add_idle_noise(arg, add_to_potential_loss = False)
                            self.idle_loss_rate = idle_loss_rate

                    elif name == 'MOVE_TO_ENTANGLING':
                        self.entangling_zone = self.entangling_zone | set(list(targets))
                        self.no_noise_zone = self.no_noise_zone - set(list(targets))
                        self.storage_zone = self.storage_zone - set(list(targets))

                        # Check if adding noise from the movement is actually necessary 
                        if not set(list(targets)).issubset(self.entangling_zone): #GB: isn't this always False? because we just more the targets into the entangling_zone
                            super().__iadd__(self.add_idle_noise(arg, add_to_potential_loss = True))
                            idle_loss_rate, self.idle_loss_rate = self.idle_loss_rate, 0.
                            self._without_loss += self.add_idle_noise(arg, add_to_potential_loss = False)
                            self.idle_loss_rate = idle_loss_rate

                    elif name == 'MOVE_TO_NO_NOISE':
                        self.no_noise_zone = self.no_noise_zone | set(list(targets))
                        self.entangling_zone = self.entangling_zone - set(list(targets))
                        self.storage_zone = self.storage_zone - set(list(targets))

                        # Check if adding noise from the movement is actually necessary
                        if not set(list(targets)).issubset(self.entangling_zone): #GB: why do we check this here?
                            super().__iadd__(self.add_idle_noise(arg, add_to_potential_loss = True))
                            idle_loss_rate, self.idle_loss_rate = self.idle_loss_rate, 0.
                            self._without_loss += self.add_idle_noise(arg, add_to_potential_loss = False)
                            self.idle_loss_rate = idle_loss_rate

                    else:
                        super().__iadd__(self.add_idle_noise(arg, add_to_potential_loss = True))
                        idle_loss_rate, self.idle_loss_rate = self.idle_loss_rate, 0.
                        self._without_loss += self.add_idle_noise(arg, add_to_potential_loss = False)
                        self.idle_loss_rate = idle_loss_rate
            
        # Add detectors for stabilizer measurements
        elif name in ['MEASURE_STABILIZERS']:
            # Assert that the previous circuit instruction was to measure the stabilizers
            if not qec.is_measurement((self.without_noise())[-1].name):
                raise Exception('MEASURE_STABILIZERS command must follow a measurement of stabilizers.')

            measured_stabilizers = targets
            measured_qubits = [target.value for target in (self.without_noise())[-1].targets_copy()]
            assert self.without_noise()[-1].name in ['M', 'MZ', 'MRZ', 'MR']

            # Default to False if compare_with_previous not specified
            if not ('compare_with_previous' in kwargs):
                kwargs['compare_with_previous'] = False
            circuit = stim.Circuit()
            if not kwargs['compare_with_previous']:
                circuit.append('DETECTOR', [stim.target_rec(-i) for i in range(1, len(measured_qubits) + 1)])
            else:
                if isinstance(measured_stabilizers, int):
                    measured_stabilizers = [measured_stabilizers]
                assert isinstance(measured_stabilizers, Iterable)
                # We need to go back through our circuit and match with the previous stabilizer measurement
                # First determine if we've measured stabilizers or done a round of perfect error correction
                all_recs = []
                for stabilizer in measured_stabilizers:
                    # Find the logical qubit and corresponding data qubits for this measure qubit
                    logical_qubit = self.qubit_index_to_logical_qubit(stabilizer)
                    data_qubits = [d for d in logical_qubit.measure_to_data[stabilizer] if d is not None]

                    if stabilizer in measured_qubits and np.all([d in measured_qubits for d in data_qubits]):
                        # If we did a round of perfect error correction and measured stabilizers simultaneously, we
                        # pick this branch
                        recs = []
                        # Add the measure qubit measurement record
                        matches = np.argwhere(measured_qubits == stabilizer)
                        if len(matches) > 0:
                            recs.append(-(len(measured_qubits) - matches[0, 0]))
                        # Find the correspond data qubit measurement records
                        for data_qubit in data_qubits:
                            matches = np.argwhere(measured_qubits == data_qubit)
                            recs.append(-(len(measured_qubits) - matches[0, 0]))

                    elif stabilizer in measured_qubits:
                        # If we just measured a measure qubit, we pick this branch
                        recs = []
                        matches = np.argwhere(measured_qubits == stabilizer)
                        if len(matches) > 0:
                            recs.append(-(len(measured_qubits)-matches[0,0]))
                        # If the ancilla qubit was measured we pick this branch
                        pauli = np.zeros(self.num_qubits, dtype=int)
                        # Assume we've measured in the Z-basis
                        pauli[stabilizer] = 3
                        recs = recs + self._generate_detector(stim.PauliString(pauli))

                    else:
                        # If only the data qubits were measured we pick this branch
                        # Search for these qubits in
                        if not all([d in measured_qubits for d in data_qubits]):
                            raise Exception('The data qubits corresponding to the stabilizers were not measured.')

                        recs = []
                        for (m, qubit) in enumerate(measured_qubits):
                            if qubit in data_qubits:
                                recs.append(-(len(measured_qubits) - m))
                        pauli = np.zeros(self.num_qubits, dtype=int)
                        logical_qubit = self.qubit_index_to_logical_qubit(stabilizer)
                        data_qubit_indices = [d for d in logical_qubit.measure_to_data[stabilizer] if d is not None]
                        measurement_operation = self.without_noise()[-1].name
                        #print(stabilizer, data_qubit_indices)
                        if measurement_operation in ['M', 'MR', 'MZ', 'MRZ']:
                            # Measure in the Z basis
                            pauli[data_qubit_indices] = 3
                        elif measurement_operation in ['MX', 'MRX']:
                            # Measure in the X basis
                            pauli[data_qubit_indices] = 1
                        elif measurement_operation in ['MY', 'MRY']:
                            # Measure in the X basis
                            pauli[data_qubit_indices] = 2
                        else:
                            # We measured a Pauli product. We need to deal with this. But another day.
                            raise NotImplementedError('Automatic detectors not implemented for MPP measurements.')
                        recs = recs + self._generate_detector(stim.PauliString(pauli))
                    all_recs.append([stim.target_rec(r) for r in recs])

                for recs in all_recs:
                    self.append('DETECTOR', recs)

            super().__iadd__(circuit)
            self._without_loss += circuit

        # Handle other circuit noise
        else:
            assert isinstance(targets, Sized)

            circuit = stim.Circuit()
            circuit.append(name, targets, arg)
            entangling_gate_loss_rate, reset_loss_rate, single_qubit_loss_rate = self.entangling_gate_loss_rate, self.reset_loss_rate, self.single_qubit_loss_rate
            self.entangling_gate_loss_rate, self.reset_loss_rate, self.single_qubit_loss_rate = 0., 0., 0.
            self._without_loss += self.add_noise(circuit, add_to_potential_loss = False)
            self.entangling_gate_loss_rate, self.reset_loss_rate, self.single_qubit_loss_rate = entangling_gate_loss_rate, reset_loss_rate, single_qubit_loss_rate

            # Add only circuit instructions on non-lost qubits # NOT RELEVANT ANYMORE, if we want to make it faster we can just append the instruction to all targets
            circuit = stim.Circuit()
            if tools.is_entangling(str(name)):
                # Remove both qubits in the gate if a single qubit in the entangling gate is lost
                not_lost_targets = []
                for _ in range(len(targets) // 2):
                    # We always want to do swaps, because it corresponds to when we physically rotate the code!
                    if (not (targets[2 * _] in self.lost_qubits) and not (targets[2 * _ + 1] in self.lost_qubits)) \
                            or str(name) == 'SWAP':
                        not_lost_targets.append(targets[2 * _])
                        not_lost_targets.append(targets[2 * _ + 1])
                circuit.append(name, not_lost_targets, arg)
                super().__iadd__(self.add_noise(circuit, add_to_potential_loss=True))
            elif tools.is_single_qubit(str(name)):
                circuit.append(name, targets, arg)
                super().__iadd__(self.add_noise(circuit, add_to_potential_loss=True))
            elif tools.is_measurement(str(name)):
                # Reset lost qubits in |1>
                reset_circuit = stim.Circuit()
                lost_targets = [t for t in targets if t in self.lost_qubits]
                if len(lost_targets) > 0:
                    reset_circuit.append('R', lost_targets)
                    reset_circuit.append('X_ERROR', lost_targets, 1)
                    super().__iadd__(reset_circuit, add_to_potential_loss=True)

                self.lost_qubits = self.lost_qubits - set(lost_targets)
                circuit.append(name, targets, arg)
                super().__iadd__(self.add_noise(circuit, add_to_potential_loss=True))
            
            elif tools.is_reset(str(name)): # GB's addition
                circuit.append(name, targets, arg)
                super().__iadd__(self.add_noise(circuit, add_to_potential_loss=True))
            
            else:
                circuit.append(name, targets, arg)
                super().__iadd__(self.add_noise(circuit, add_to_potential_loss=True))

    def qubit_index_to_logical_qubit(self, index: int):
        for lq in self.logical_qubits:
            if index in lq.qubit_indices:
                return lq
        raise Exception('No logical qubit found with corresponding physical index')

    def without_loss(self) -> stim.Circuit:
        # print("This feature is not implemented anymore.")
        # return None
        return self._without_loss

    def without_noise(self) -> stim.Circuit:
        # print("This feature is not implemented anymore.")
        # return None
        return self.without_loss().without_noise()

    def copy(self, memodict={}):
        return copy.deepcopy(self, memo=memodict)
