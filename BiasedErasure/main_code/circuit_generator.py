from BiasedErasure.main_code.config import Config



class CircuitGenerator:
    def __init__(self, config: Config, noise):
        """
        Initialize the CircuitGenerator with a configuration object and a noise object.
        
        Args:
            config (Config): Configuration object containing experiment parameters.
            noise (Noise): Noise model to apply in the circuits.
        """
        self.config = config
        self.noise = noise

    def generate_circuit(self, phys_err, xzzx=False):
        """
        Generate a circuit based on the configuration and physical error rate.
        
        Args:
            phys_err (float): Physical error rate for the circuit.        
        Returns:
            Circuit: The generated logical circuit.
        """
        # Apply noise and get error rates
        entangling_gate_error_rate, entangling_gate_loss_rate = self.noise.apply_noise(
            noise_type=self.config.noise,
            phys_err=phys_err,
            bias_ratio=self.config.bias_ratio
        )

        # Generate different types of experiments based on config.circuit_type
        if self.config.circuit_type in ['memory', 'memory_wrong']:
            measure_wrong_basis = self.config.circuit_type == 'memory_wrong'
            if self.config.architecture == 'CBQC':
                return memory_experiment_surface_new(
                    self.config.dx, self.config.dy, self.config.code, self.config.cycles,
                    entangling_gate_error_rate, entangling_gate_loss_rate, self.config.erasure_ratio,
                    self.config.num_logicals, self.config.logical_basis, self.config.bias_preserving_gates,
                    self.config.ordering_type, self.config.loss_detection_method_str, self.config.LD_freq,
                    self.config.atom_array_sim, self.config.replace_H_Ry, self.config.xzzx, self.config.noise_params,
                    self.config.circuit_index, measure_wrong_basis
                )
            elif self.config.architecture == 'MBQC':
                return memory_experiment_MBQC(
                    self.config.dx, self.config.dy, self.config.cycles, entangling_gate_error_rate,
                    entangling_gate_loss_rate, self.config.erasure_ratio, self.config.logical_basis,
                    self.config.bias_preserving_gates, self.config.atom_array_sim
                )
            elif self.config.circuit_type == 'random_alg':
                return random_logical_algorithm(
                    self.config.code, self.config.num_logicals, self.config.cycles + 1, self.config.distance,
                    self.config.n_r, self.config.bias_ratio, self.config.erasure_ratio, phys_err, self.config.output_dir
                )

        elif self.config.circuit_type[:10] == 'logical_CX':
            num_CX_per_layer_list = self.config.Meta_params["num_CX_per_layer_list"]
            num_layers = len(num_CX_per_layer_list)
            self.config.Meta_params['circuit_type'] = f'logical_CX__Nlayers{num_layers}__NCX{"_".join(map(str, num_CX_per_layer_list))}'
            return CX_experiment_surface(
                self.config.dx, self.config.dy, self.config.code, num_CX_per_layer_list, num_layers,
                self.config.num_logicals, self.config.logical_basis, self.config.bias_preserving_gates,
                self.config.ordering_type, self.config.loss_detection_method_str, self.config.LD_freq,
                self.config.atom_array_sim, self.config.replace_H_Ry, self.config.xzzx, self.config.noise_params,
                self.config.printing, self.config.circuit_index
            )

        elif self.config.circuit_type in ['GHZ_all_o1', 'GHZ_save_o1', 'GHZ_all_o2', 'GHZ_save_o2']:
            order_1 = [(0, i) for i in range(1, self.config.num_logicals)]
            order_2 = [(i - 1, i) for i in range(1, self.config.num_logicals)]
            chosen_order = order_1 if self.config.circuit_type.endswith("1") else order_2
            loss_detection_on_all_qubits = 'all' in self.config.circuit_type
            return GHZ_experiment_Surface(
                self.config.dx, self.config.dy, chosen_order, self.config.num_logicals, self.config.code,
                self.config.cycles, entangling_gate_error_rate, entangling_gate_loss_rate, self.config.erasure_ratio,
                self.config.logical_basis, self.config.bias_preserving_gates, loss_detection_on_all_qubits,
                self.config.atom_array_sim
            )

        elif self.config.circuit_type == 'Steane_QEC':
            return Steane_QEC_circuit(
                self.config.dx, self.config.dy, self.config.code, self.config.Steane_type, self.config.cycles - 1,
                entangling_gate_error_rate, entangling_gate_loss_rate, self.config.erasure_ratio,
                self.config.logical_basis, self.config.bias_preserving_gates, True, self.config.atom_array_sim,
                self.config.obs_pos
            )

        else:
            return None
