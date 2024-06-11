from BiasedErasure.main_code.simulate import *
import numpy as np
from distutils.util import strtobool



if __name__ == '__main__':
    """_summary_
    Examples of an input in the .sh file:
    
    "python loop_num_cycles.py --architecture=MBQC --code=XZZX --logical_basis=X --bias_preserving_gates=False \
    --noise=atom_array --with_erasures=True --is_erasure_biased=False --output_dir=/Users/gefenbaranes/Documents/results \
    --num_shots=1000 --distances=3,5 --SSR=True --LD_method=MBQC"
    
    "python loop_num_cycles.py --architecture=CBQC --code=XZZX --logical_basis=X --bias_preserving_gates=False \
    --noise=atom_array --with_erasures=True --is_erasure_biased=False --output_dir=/Users/gefenbaranes/Documents/results \
    --num_shots=1000 --distances=3,5 --SSR=True --LD_method=SWAP"
    
    """
    
    Meta_params_keys = ['architecture', 'code', 'logical_basis', 'bias_preserving_gates', 
                'noise', 'is_erasure_biased', 'LD_freq', 'LD_method', 'SSR', 'cycles', 'ordering', 'decoder', 'circuit_type',
                'printing', 'num_logicals', 'Steane_type', 'loss_decoder']
    
    Meta_params = {key: None for key in Meta_params_keys}
    
    for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
        if k in Meta_params:
            Meta_params[k] = v
        elif k =='output_dir':
            output_dir = v
        elif k == 'distances':
            distances = [int(x) for x in v.split(',')]
        elif k == 'num_shots':
            num_shots = int(v)
    
    erasure_ratio = 1.0; bias_ratio = 0.5 ; num_p_phys = 1; phys_err_vec = np.round(np.linspace(0.5e-2, 4e-2, num_p_phys),8) # not relevant, we use the error model of atom array
    bloch_point_params = {'erasure_ratio': erasure_ratio, 'bias_ratio': bias_ratio}
            
    num_cycles_vec = list(np.arange(2,22,2))
    LD_freq_vec = [1]  
    
    for LD_freq in LD_freq_vec:
        Meta_params['LD_freq'] = str(LD_freq)
        for num_cycles in num_cycles_vec:
            Meta_params['cycles'] = str(num_cycles)
            print(f"num_cycles = {num_cycles}, LD_freq = {LD_freq}")
            print(f"Meta params dictionary = {Meta_params}")
            Bloch_sphere_point_simulate(Meta_params=Meta_params, bloch_point_params=bloch_point_params, output_dir=output_dir,
                                        phys_err_vec=phys_err_vec, distances=distances, num_shots=num_shots)

            # file_name = create_tile_name(Meta_params, bloch_point_params)
            # threshold = calculate_logical_prob(folder=output_dir, filename=file_name, title=file_name, Erasure_ratio=erasure_ratio, 
            #                                 Bias=bias_ratio, plotting = False, distances=distances, num_shots=num_shots)
            # threshold[i,j] = threshold
        
    
