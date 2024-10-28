from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from distutils.util import strtobool

@dataclass
class Config:
    dx: int
    dy: int
    cycles: int
    output_dir: str
    noise_params: dict
    architecture: str = 'CBQC' # CBQC, MBQC
    code: str = 'Rotated_Surface' # Rotated_Surface, Surface
    logical_basis: str = 'X' # X, Z
    bias_preserving_gates: bool = False # False, True
    is_erasure_biased: bool = False # False, True
    LD_freq: int = 1 # any int >= 1
    LD_method: str = 'SWAP' # None, SWAP, FREE, erasure
    SSR: bool = True # False, True
    ordering: List[str] = field(default_factory=lambda: ['N', 'Z', 'Zr', 'Nr']) # length should be num cycles - 1
    decoder: str = 'MLE' # MLE, MWPM
    circuit_type: str = 'memory' # memory, memory_wrong, CX, ..
    Steane_type: str = 'regular' # regular, SWAP
    printing: bool = True # False, True
    num_logicals: int = 1 # any int >= 1
    loss_decoder: str = 'independent' # independent, first_comb, comb, None
    obs_pos: str = 'd-1' # relevant for Steane QEC.
    n_r: float = 0.0 # relevant for logical algs.
    circuit_index: str = '0' 
    bloch_point_params: Dict[str, Any] = field(default_factory=dict)
    noise: str = 'atom_array' 
    replace_H_Ry: bool = True # Whether to replace H with Ry gate
    xzzx: bool = True # Whether to use xzzx error correction (optional).
    use_loss_decoding: bool = True
    use_independent_decoder: bool = True
    use_independent_and_first_comb_decoder: bool = False
    first_comb_weight: float = 0.0
    erasure_ratio: float = 0.0
    atom_array_sim: bool = False
    save_data_during_sim: bool = False
    Meta_params: Optional[Dict[str, Any]] = field(default_factory=dict)



    def __post_init__(self):
        """Populate Config attributes from Meta_params with type conversion."""
        def convert_and_set(attribute, default_type, default_value=None):
            """Helper function to set attribute with type conversion."""
            if attribute in self.Meta_params:
                value = self.Meta_params[attribute]
                setattr(self, attribute, default_type(value))
            elif default_value is not None:
                print(f"setting {attribute} to its default value: {default_value}")
                setattr(self, attribute, default_value)

        # Convert each parameter as needed
        convert_and_set('architecture', str)
        convert_and_set('code', str)
        convert_and_set('num_logicals', int, self.num_logicals)
        convert_and_set('bias_preserving_gates', bool, self.bias_preserving_gates)
        convert_and_set('logical_basis', str, self.logical_basis)
        convert_and_set('is_erasure_biased', bool, self.is_erasure_biased)
        convert_and_set('LD_freq', int, self.LD_freq)
        convert_and_set('SSR', bool, self.SSR)
        convert_and_set('printing', bool, self.printing)
        convert_and_set('cycles', int, self.cycles)
        convert_and_set('loss_decoder', str, self.loss_decoder)
        convert_and_set('decoder', str, self.decoder)
        convert_and_set('circuit_type', str, self.circuit_type)
        convert_and_set('Steane_type', str, self.Steane_type)
        convert_and_set('obs_pos', str, self.obs_pos)
        convert_and_set('n_r', int, self.n_r)
        convert_and_set('circuit_index', str, self.circuit_index)
        convert_and_set('noise', str, self.noise)
        
        # other params to set:
        use_loss_decoding = False if self.Meta_params['loss_decoder'] == 'None' else True
        use_independent_decoder = True if self.Meta_params['loss_decoder'] == 'independent'
        use_independent_and_first_comb_decoder = True if self.Meta_params['loss_decoder'] == 'first_comb' else False
        
        # for Steane QEC:
        elif self.circuit_type == 'Steane_QEC':
            self.obs_pos = int(eval(self.obs_pos.replace('d', str(min(self.dx, self.dy)))))
        
        
        # special params for atom array experiment:
        if self.noise == 'atom_array':
            self.replace_H_Ry = True
            self.xzzx = True
        else:
            self.replace_H_Ry = False
            self.xzzx = False