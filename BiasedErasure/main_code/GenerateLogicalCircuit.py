import stim
import numpy as np


class GenerateLogicalCircuit(stim.Circuit):
    def __init__(self, logical_qubits, circuit, loss_probabilities, potential_lost_qubits):
        # super().__init__(circuit if circuit else [])
        super().__init__()
        for instruction in circuit:
            self.append(instruction)
        self.logical_qubits = list(logical_qubits)
        self.loss_probabilities = loss_probabilities
        self.potential_lost_qubits = potential_lost_qubits
        self.num_logical_qubits = len(self.logical_qubits)
        self.logical_qubit_indices = np.arange(self.num_logical_qubits)
        
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
            
    
    def qubit_index_to_logical_qubit_index(self, index: int):
        for (_, lq) in enumerate(self.logical_qubits):
            if index in lq.qubit_indices:
                return _
        raise Exception('No logical qubit found with corresponding physical index')
    
    
    def qubit_index_to_logical_qubit(self, index: int):
        for lq in self.logical_qubits:
            if index in lq.qubit_indices:
                return lq
        raise Exception('No logical qubit found with corresponding physical index')
