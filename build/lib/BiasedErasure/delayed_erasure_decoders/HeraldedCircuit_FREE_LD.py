import stim
import numpy as np
from typing import List


class HeraldedCircuit_FREE_LD:
    def __init__(self, circuit: stim.Circuit, biased_erasure: bool, cycles: int, phys_error:float, erasure_ratio:float, ancilla_qubits:list, data_qubits:list, distance=None, loss_detection_freq=None, code=None, basis = None, SSR = True, printing=False, **kwargs) -> None:
        self.circuit = circuit
        self.biased_erasure = biased_erasure
        self.distance = distance
        self.code = code
        self.basis = basis
        self.SSR = SSR
        self.cycles = cycles
        self.printing = printing
        self.erasure_ratio = erasure_ratio
        self.phys_error = phys_error
        self.loss_detection_freq = loss_detection_freq
        self.ancilla_qubits = ancilla_qubits
        self.data_qubits = data_qubits
        self.lost_ancillas = {}  ###
        self.qec_cycles_complete = False  ###
        self.lost_ancillas_by_qec_round = {}  #!  {qec_round: [lost_ancilla_qubits]}
        self.lost_data_by_ld_round = {}  #!  {ld_round: [lost_data_qubits]}
        self.total_num_QEC_round = None
        
    def sample(self):
        pass

    
    def update_loss_lists(self, instruction, loss_detection_events, lost_data_qubits, lost_ancilla_qubits, loss_detector_ix):
        potential_lost_qubits = instruction.targets_copy()
        for q in potential_lost_qubits:
            if loss_detection_events[loss_detector_ix] == True:
                if q.value in self.data_qubits:
                    if q.value not in lost_data_qubits:
                        lost_data_qubits.append(q.value)
                        if self.printing:
                            print(f"Detected the loss of data qubit = {q.value}. Lost data qubits = {lost_data_qubits}")
                else: # q is an ancilla --> replace with a fresh noisy qubit
                    if q.value not in lost_ancilla_qubits:
                        lost_ancilla_qubits.append(q.value)
                        if self.printing:
                            print(f"Detected the loss of ancilla qubit = {q.value}. Lost ancilla qubits = {lost_ancilla_qubits}")        
            loss_detector_ix += 1
        return loss_detector_ix
        
        
    def get_loss_location(self, loss_detection_events: list):
        # Iterate through circuit. Every time we encounter a loss event (flagged by the 'I' gate), record the loss.
        loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
        round_ix = 0
        inside_qec_round = False
        for instruction in self.circuit:
            if instruction.name == 'TICK': # begin of a QEC round:
                if not inside_qec_round: # beginning of round
                    if self.printing:
                        print(f"Starting QEC Round {round_ix}")
                    lost_ancilla_qubits = []
                    lost_data_qubits = []
                    qec_round = round_ix
                    ld_round = int(round_ix / self.loss_detection_freq)
                else: # end of round
                    if self.printing:
                        print(f"Finished QEC Round {round_ix}")
                    self.lost_ancillas_by_qec_round[qec_round] = lost_ancilla_qubits
                    if ld_round in self.lost_data_by_ld_round:
                        self.lost_data_by_ld_round[ld_round].extend(lost_data_qubits)
                    else:
                        self.lost_data_by_ld_round[ld_round] = lost_data_qubits
                    round_ix += 1
                
                inside_qec_round = not inside_qec_round
                continue
            
            if inside_qec_round:
                if instruction.name == 'I': # loss event
                    loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, lost_data_qubits, lost_ancilla_qubits, loss_detector_ix)
                            
            else:
                pass # we don't need to document losses outside QEC rounds because we assume there are non
                # lost_ancilla_qubits = []
                # lost_data_qubits = []
                # if instruction.name == 'I': # loss event
                #     loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, data_qubits, lost_data_qubits, lost_ancilla_qubits, loss_detector_ix)

    def heralded_new_circuit(self, loss_detection_events: list):
        """ This function takes the original circuit with places for potential losses and loss detection events, and generates 2 circuits: 1. experimental measurement circuit. 2. Theory decoding circuit. """
        # Initialization for every shot:
        self.lost_ancillas = {}  ###
        self.qec_cycles_complete = False  ###
        self.lost_ancillas_by_qec_round = {}  #!  {qec_round: [lost_ancilla_qubits]}
        self.lost_data_by_ld_round = {}  #!  {ld_round: [lost_data_qubits]}
        
        
        # First sweep: get location of lost qubits in the circuit.
        self.get_loss_location(loss_detection_events=loss_detection_events) 
        self.total_num_QEC_round = len(self.lost_ancillas_by_qec_round)
        
        if self.printing :
            print(f"lost_ancillas_by_qec_round={self.lost_ancillas_by_qec_round}")
            print(f"lost_data_by_ld_round={self.lost_data_by_ld_round}")

        # Second sweep: fill in the experimental circuit (heralded_circuit) and decoding circuit (new_lossless_circuit).
        heralded_circuit = stim.Circuit()
        new_lossless_circuit = stim.Circuit()
        loss_detector_ix = 0  # tracks the index of detectors in the circuit as we iterate.
        lost_data_qubits = []  # data_qubit_index: number_of_rounds_before_replace. if>0 --> delete entangling gates. if==0: reinitialize the qubit + depolarizing det error
        lost_ancilla_qubits = []
        round_ix = 0
        inside_qec_round = False
        LD_round = False
        for instruction in self.circuit:
            if instruction.name == 'TICK': # begin of a QEC round:
                if not inside_qec_round: # beginning of round
                    self.qec_cycles_complete = True
                    ld_round = int(round_ix / self.loss_detection_freq) # check that its the same
                    last_QEC_round = True if round_ix == self.total_num_QEC_round-1 else False
                    if (round_ix+1)%self.loss_detection_freq == 0:
                        LD_round = True
                    else:
                        LD_round = False
                    if self.printing:
                        print(f"a new QEC round number {round_ix}! is it a LD round? {LD_round} (loss detection freq = {self.loss_detection_freq})")
                else: # end of round
                    
                    # Replace lost ancilla qubits after every round.
                    if self.printing :
                        print(f"End of a QEC round! replaced ancilla qubits {lost_ancilla_qubits}")
                    lost_ancilla_qubits = [] # every QEC round we replace ancilla qubits with fresh ones
                    
                    
                    # If loss detection round (or last QEC round) --> replace data qubits with fresh noisy qubits
                    if LD_round or last_QEC_round:
                        if self.printing :
                            if LD_round:
                                print(f"We just finished a LD round! lets replace the following qubits with fresh ones: {lost_data_qubits}")
                            elif last_QEC_round:
                                print(f"We just finished the last round of QEC! now we have full detection on the loss of: {lost_data_qubits}")
                        for q in lost_data_qubits:
                            if LD_round:
                                # Heralded circuit - experiment - replacing the qubit with a fresh qubit: initialization + noise:
                                logical_qubit = self.circuit.qubit_index_to_logical_qubit(q)
                                prep = logical_qubit.get_qubit_init_basis(physical_index=q, logical_basis = self.basis)
                                heralded_circuit.append('R', [q])
                                if prep == 'RX':
                                    heralded_circuit.append('H', [q])

                            # Lossless circuit - decoding - p=1/2 channel to set w=0 (giving the decoder the loss information):
                            self.add_pauli_channel(new_lossless_circuit, [q])
                        
                        lost_data_qubits = []

                    round_ix += 1
                
                inside_qec_round = not inside_qec_round
                heralded_circuit.append('TICK')
                new_lossless_circuit.append('TICK')
                continue

            if inside_qec_round:
                loss_detector_ix = self.add_instruction(instruction, heralded_circuit, new_lossless_circuit, loss_detection_events, loss_detector_ix,
                                                        lost_data_qubits=lost_data_qubits, lost_ancilla_qubits=lost_ancilla_qubits, 
                                                        last_QEC_round=last_QEC_round, round_ix=round_ix) 
            else:
                loss_detector_ix = self.add_instruction(instruction, heralded_circuit, new_lossless_circuit, loss_detection_events, loss_detector_ix,
                                                        lost_data_qubits=lost_data_qubits, lost_ancilla_qubits=lost_ancilla_qubits,
                                                        last_QEC_round=False)
        return heralded_circuit, new_lossless_circuit
    
    def add_instruction(self, instruction, circuit: stim.Circuit, new_lossless_circuit: stim.Circuit, loss_detection_events: list, loss_detector_ix: int,
                        lost_data_qubits: list, lost_ancilla_qubits:list, last_QEC_round: bool = None, round_ix: int = None):


        # Update lost qubits lists:
        if instruction.name == 'I': # loss event
            loss_detector_ix = self.update_loss_lists(instruction, loss_detection_events, lost_data_qubits, lost_ancilla_qubits, loss_detector_ix)

        elif instruction.name in ['CZ','CX']:
            qubits = [q.value for q in instruction.targets_copy()]
            
            # Lossless circuit: the CZ is here, but we add error model to acount for errors in the decoder:
            new_lossless_circuit.append(instruction) # append the CZ gate to the lossless circuit anyway
            
            pairs = [(qubits[i], qubits[i + 1]) for i in range(0, len(qubits), 2)]
            for (c,t) in pairs:
                ancilla_target = c if (c not in self.data_qubits) else t
                data_target = c if c in self.data_qubits else t

                # Lossless circuit: Add noise to neighbor data of lost ancilla and to neighbor ancilla of lost data.
                if round_ix is not None:  #!
                    qec_round = round_ix  #!
                    ld_round = int(round_ix / self.loss_detection_freq)  #!
                    if instruction.name == 'CZ':
                        if (not self.SSR) or ((self.SSR) and (ancilla_target in self.lost_ancillas_by_qec_round[qec_round])):
                            strength = 1/4 if self.SSR else self.phys_error*(self.erasure_ratio)  # TODO: fix the 1/4 to be 1/degree_of_ancilla
                            new_lossless_circuit.append('PAULI_CHANNEL_1', [data_target], strength*np.array([0, 0, 0.5])) # {I,Z}
                            # new_lossless_circuit.append('PAULI_CHANNEL_1', [data_target], strength*np.array([0.25, 0.25, 0.25])) # {I,Z}
                        if data_target in self.lost_data_by_ld_round[ld_round]:
                            r = self.loss_detection_freq if self.cycles%self.loss_detection_freq == 0 else self.cycles%self.loss_detection_freq
                            strength = 1/(4*r) if last_QEC_round else 1/(4*self.loss_detection_freq)  # TODO: fix the 1/4 to be 1/degree_of_data. 
                            new_lossless_circuit.append('PAULI_CHANNEL_1', [ancilla_target], strength*np.array([0, 0, 0.5])) # {I,Z}  # TODO: check if the basis is right.
                            # new_lossless_circuit.append('PAULI_CHANNEL_1', [ancilla_target], strength*np.array([0.25, 0.25, 0.25])) # {I,Z}
                    elif instruction.name == 'CX':
                        if (not self.SSR) or ((self.SSR) and (ancilla_target in self.lost_ancillas_by_qec_round[qec_round])):
                            strength = 1/4 if self.SSR else self.phys_error*(self.erasure_ratio)  # TODO: fix the 1/4 to be 1/degree_of_ancilla
                            noise_channel = strength*np.array([0.5, 0, 0]) if ancilla_target == c else strength*np.array([0, 0, 0.5])
                            # new_lossless_circuit.append('PAULI_CHANNEL_1', [data_target], strength*np.array([0.5, 0, 0])) # {I,X} # TODO: causing an error!
                            new_lossless_circuit.append('PAULI_CHANNEL_1', [data_target], noise_channel)
                        if data_target in self.lost_data_by_ld_round[ld_round]:
                            r = self.loss_detection_freq if self.cycles%self.loss_detection_freq == 0 else self.cycles%self.loss_detection_freq
                            strength = 1/(4*r) if last_QEC_round else 1/(4*self.loss_detection_freq)  # TODO: fix the 1/4 to be 1/degree_of_data
                            noise_channel = strength*np.array([0.5, 0, 0]) if data_target == c else strength*np.array([0, 0, 0.5])
                            # new_lossless_circuit.append('PAULI_CHANNEL_1', [ancilla_target], strength*np.array([0.5, 0, 0])) # {I,X}  # TODO: causing an error!
                            new_lossless_circuit.append('PAULI_CHANNEL_1', [ancilla_target], noise_channel)
                
                # Heralded loss circuit: the CZ is not here is we lost one of the qubits:
                if (c in lost_data_qubits + lost_ancilla_qubits) or (t in lost_data_qubits + lost_ancilla_qubits): # remove the gate from the heralded circuit:
                    if self.printing :
                        print(f"Removing this gate from the heralded circuit: {instruction.name} {c},{t}, because my lost data qubits = {lost_data_qubits} and ancilla qubits = {lost_ancilla_qubits}")
                    pass
                else:
                    circuit.append(instruction.name, [c,t])

        elif instruction.name in ['H']:
            new_lossless_circuit.append(instruction) # append the gate to the lossless circuit anyway
            qubits = [q.value for q in instruction.targets_copy()]
            for q in qubits: # remove the gate from the heralded circuit:
                if q in lost_data_qubits + lost_ancilla_qubits:
                    if self.printing :
                        print(f"Removing this gate from the heralded circuit: {instruction.name} {q}, because my lost data qubits = {lost_data_qubits} and ancilla qubits = {lost_ancilla_qubits}")
                    pass
                else:
                    circuit.append(instruction.name, [q])

            
        elif instruction.name in ['MRX', 'MR']:
            qbts = instruction.targets_copy()
            num_measurements = len(qbts)

            # Update measurement_ixs of lost ancilla qubits.
            # for lost_ancilla_id in self.lost_ancillas:
            #     self.lost_ancillas[lost_ancilla_id] -= num_measurements

            for ix, qbt in enumerate(qbts):
                q = qbt.value

                # SSR --> ancilla loss is heralded --> create supercheck operators: ###NEW
                if q in lost_ancilla_qubits:
                    self.add_pauli_channel(new_lossless_circuit, [q])
                        
                # if q in lost_ancilla_qubits and q not in self.lost_ancillas:
                #     # store the last valid measurement ix
                #     self.lost_ancillas[q] = -(num_measurements - ix) - num_ancillas  ### ASSUME: measurement order of ancilla qubits is always the same.

                new_lossless_circuit.append(instruction.name, [q])
                
                # Heralded circuit - lost ancilla qubits give deterministic |0> measurement:
                # if SSR is on this measurement will not even be used.
                if q in lost_ancilla_qubits:
                    if instruction.name == 'MR':
                        circuit.append('R', [q])
                        circuit.append('MR', [q])
                    elif instruction.name == 'MRX':
                        circuit.append('RX', [q])
                        circuit.append('MRX', [q])
                else:
                    circuit.append(instruction.name, [q])

        elif instruction.name in ['M', 'MX']:
            qbts = instruction.targets_copy()
            num_measurements = len(qbts)
                        
            # Heralded circuit - lost qubits give deterministic |0> measurement:
            if len(lost_ancilla_qubits + lost_data_qubits) == 0:
                circuit.append(instruction)
                
            else:
                for ix, qbt in enumerate(qbts):
                    q = qbt.value
                    
                    # SSR --> ancilla loss is heralded --> create supercheck operators: ###NEW
                    if q in lost_ancilla_qubits:
                        self.add_pauli_channel(new_lossless_circuit, [q])
                    
                    if q in lost_ancilla_qubits + lost_data_qubits:
                        if instruction.name == 'M':
                            circuit.append('R', [q])
                            circuit.append('M', [q])
                        elif instruction.name == 'MX':
                            circuit.append('RX', [q])
                            circuit.append('MX', [q])
                    else:
                        circuit.append(instruction.name, [q])    
            
            new_lossless_circuit.append(instruction)
                
            # Update measurement_ixs of lost ancilla qubits.
            # for lost_ancilla_id in self.lost_ancillas:
            #     self.lost_ancillas[lost_ancilla_id] -= num_measurements
        else:
            circuit.append(instruction)
            new_lossless_circuit.append(instruction)
        return loss_detector_ix


    def add_pauli_channel(self, circuit, targets):
        if len(targets) == 1 and not self.biased_erasure:
            circuit.append('PAULI_CHANNEL_1', targets, np.array([0.25, 0.25, 0.25])) # {X,Y,Z,I}
        elif len(targets) == 1 and self.biased_erasure:
            circuit.append('PAULI_CHANNEL_1', targets, np.array([0, 0, 0.5])) # {I,Z}
        elif len(targets) == 2 and not self.biased_erasure:
            circuit.append('PAULI_CHANNEL_2', targets, [1/16 for i in range(15)]) #  {X,Y,Z,I}**2
        elif len(targets) == 2 and self.biased_erasure:
            circuit.append('PAULI_CHANNEL_2', targets, np.array([0,0,0.25,0,0,0,0,0,0,0,0,0.25,0,0,0.25])) #  biased, {Z,I}**2
