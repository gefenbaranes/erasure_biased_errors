from BiasedErasure.main_code.simulate import *
import numpy as np
from distutils.util import strtobool



if __name__ == '__main__':
    """_summary_
    This code is here to collect data on the bloch sphere. Run it many times to get enough data (at least 1000 jobs).
    An example of an input in the .sh file:
    "python build_bloch_sphere.py --architecture=CBQC --code=XZZX --logical_basis=X --bias_preserving_gates=True 
    --noise=biased_erasure_noise --with_erasures=True --is_erasure_biased=False 
    --output_dir=/Users/gefenbaranes/Documents/results --num_shots=5 --distances=9,11,13 --num_p_phys=15 --loss_detection_freq=2 --SRR=False --cycles=d"
    """
    
    Meta_params_keys = ['architecture', 'code', 'logical_basis', 'bias_preserving_gates', 
                'noise', 'is_erasure_biased', 'LD_freq', 'LD_method', 'SSR', 'cycles', 'ordering', 'decoder', 'circuit_type',
                'printing', 'num_logicals', 'Steane_type', 'loss_decoder']
    Meta_params = {key: None for key in Meta_params_keys}
    
    distances=[5,7,9]; num_shots=30 # default values, user will change them
    # we take a small number because we loop here over many phi and theta. One needs to run this function over many jobs to collect enough data.
    
    for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
        if k == 'num_p_phys':
            Meta_params[k] = int(v)
        elif k in Meta_params:
            Meta_params[k] = v
        elif k =='output_dir':
            output_dir = v
        elif k == 'distances':
            distances= [int(x) for x in v.split(',')]
        elif k == 'num_shots':
            num_shots = int(v)
    print(f"Meta params dictionary = {Meta_params}")
    
    # theta_vec = np.linspace(0, np.pi/2, 7)
    # phi_vec = np.linspace(0, np.pi/2, 7)
    # thresholds_mat = np.ndarray((len(theta_vec),len(phi_vec)), dtype = 'float')
    # erasure_ratio_vec = np.concatenate([np.linspace(0,1,11),[0.98]]); erasure_ratio_vec.sort()
    # bias_ratio_vec = np.linspace(0,1,11)
    # erasure_ratio_vec = [1.0, 0.75, 0.5, 0.25, 0.0]
    # bias_ratio_vec = [0.5, 10, 50, 100]
    erasure_ratio_vec = [1.0]
    bias_ratio_vec = [0.5]
    
    for i in range(len(erasure_ratio_vec)):
        for j in range(len(bias_ratio_vec)):
            erasure_ratio = erasure_ratio_vec[i]
            bias_ratio = bias_ratio_vec[j]
            # erasure_ratio = (np.cos(phi))**2
            # bias_ratio = (np.cos(theta))**2
            bloch_point_params = {'erasure_ratio': erasure_ratio, 'bias_ratio': bias_ratio}
            print(f"bloch point params dictionary = {bloch_point_params}")
            # Estimating the threshold to know which range of phys_error we should run:
            num_p_phys = Meta_params['num_p_phys']
            # cond_0 = strtobool(Meta_params["with_erasures"])
            cond_1 = strtobool(Meta_params['is_erasure_biased'])
            cond_2 = strtobool(Meta_params["bias_preserving_gates"])
            phys_err_vec = np.round(np.linspace(3.5e-2, 5.5e-2, num_p_phys),8)
                    
            Bloch_sphere_point_simulate(Meta_params=Meta_params, bloch_point_params=bloch_point_params, output_dir=output_dir,
                                        phys_err_vec=phys_err_vec, distances=distances, num_shots=num_shots)
            # file_name = create_tile_name(Meta_params, bloch_point_params)
            # threshold = calculate_logical_prob(folder=output_dir, filename=file_name, title=file_name, Erasure_ratio=erasure_ratio, 
            #                                 Bias=bias_ratio, plotting = False, distances=distances, num_shots=num_shots)
            # threshold[i,j] = threshold
        
    
