import stim

class LogicalCircuitMBQC(stim.Circuit):
    def __init__(self, circuit, loss_probabilities, potential_lost_qubits):
        super().__init__()
        for instruction in circuit:
            self.append(instruction)
        self.loss_probabilities = loss_probabilities
        self.potential_lost_qubits = potential_lost_qubits