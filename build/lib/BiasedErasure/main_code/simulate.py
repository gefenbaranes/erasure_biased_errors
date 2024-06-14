import stim
import numpy as np
import os
import qec
import pymatching, stim
from BiasedErasure.main_code.LogicalCircuit import LogicalCircuit 
import random
import matplotlib.pyplot as plt
import progressbar
from BiasedErasure.delayed_erasure_decoders.HeraldedCircuit_FREE_LD import HeraldedCircuit_FREE_LD
from BiasedErasure.delayed_erasure_decoders.HeraldedCircuit_SWAP_LD import HeraldedCircuit_SWAP_LD
from BiasedErasure.main_code.Simulator import *
import time
import sys
from collections import defaultdict
from distutils.util import strtobool
from BiasedErasure.main_code.noise_channels import biased_erasure_noise, biased_erasure_noise_correlated, no_noise, atom_array


def create_file_name(Meta_params, bloch_point_params):
    # if Meta_params['circuit_type'] in ['Steane_Regular', 'Steane_SWAP']:  # TODO: add all the extra handles for Steane.
    #     return f"{Meta_params['architecture']}__{Meta_params['code']}__{Meta_params['circuit_type']}__{Meta_params['num_logicals']}logicals__{Meta_params['logical_basis']}__{Meta_params['bias_preserving_gates']}__{Meta_params['noise']}__{Meta_params['is_erasure_biased']}__{bloch_point_params['erasure_ratio']}__{bloch_point_params['bias_ratio']}__LD_freq_{Meta_params['LD_freq']}__SSR_{Meta_params['SSR']}__LD_method_{Meta_params['LD_method']}__num_cycles_{Meta_params['cycles']}__ordering_{Meta_params['ordering']}__decoder_{Meta_params['decoder']}"
    # elif Meta_params['circuit_type'] == 'memory' and post_selection_prob > 0:
    #     return f"{Meta_params['architecture']}__{Meta_params['code']}__{Meta_params['circuit_type']}__{Meta_params['num_logicals']}logicals__{Meta_params['logical_basis']}__{Meta_params['bias_preserving_gates']}__{Meta_params['noise']}__{Meta_params['is_erasure_biased']}__{bloch_point_params['erasure_ratio']}__{bloch_point_params['bias_ratio']}__LD_freq_{Meta_params['LD_freq']}__SSR_{Meta_params['SSR']}__LD_method_{Meta_params['LD_method']}__num_cycles_{Meta_params['cycles']}__ordering_{Meta_params['ordering']}__decoder_{Meta_params['decoder']}"
    # else:
    return f"{Meta_params['architecture']}__{Meta_params['code']}__{Meta_params['circuit_type']}__{Meta_params['num_logicals']}logicals__{Meta_params['logical_basis']}__{Meta_params['bias_preserving_gates']}__{Meta_params['noise']}__{Meta_params['is_erasure_biased']}__{bloch_point_params['erasure_ratio']}__{bloch_point_params['bias_ratio']}__LD_freq_{Meta_params['LD_freq']}__SSR_{Meta_params['SSR']}__LD_method_{Meta_params['LD_method']}__num_cycles_{Meta_params['cycles']}__ordering_{Meta_params['ordering']}__loss_decoder_{Meta_params['loss_decoder']}__decoder_{Meta_params['decoder']}"



def Bloch_sphere_point_simulate(Meta_params={}, bloch_point_params={}, output_dir='', phys_err_vec = [], distances=[5,7,9], num_shots=1000, first_comb_weight=0.5, dont_use_loss_decoder=False):
    """_summary_
    Given a set of params (meta_params for a type of a bloch sphere, and bloch_point_params for the point within), it is calling simulator
    to collect the data and save it with the file name that have all the important information.
    Args:
        Meta_params (dict, optional): Specify the bloch sphere type. Defaults to {}.
        bloch_point_params (dict, optional): Specify the R and Bias location within the sphere. Defaults to {}.
        output_dir (str, optional): Folder to save the data. Defaults to ''.
        phys_err_vec (list, optional): physical error vector. Defaults to [].
        distances (list, optional): distances. Defaults to [5,7,9].
        num_shots (int, optional): number of shots. Defaults to 1000.
    """
    file_name = create_file_name(Meta_params, bloch_point_params)
    
    noise_dict = {'biased_erasure_noise': biased_erasure_noise, 'no_noise': no_noise, 'biased_erasure_noise_correlated': biased_erasure_noise_correlated, 'atom_array': atom_array}
    noise = noise_dict[Meta_params['noise']]
    atom_array_sim = True if Meta_params['noise'] == 'atom_array' else False
    loss_detection_method_dict = {'FREE': HeraldedCircuit_FREE_LD, 'SWAP': HeraldedCircuit_SWAP_LD, 'MBQC': HeraldedCircuit_FREE_LD}
    loss_detection_method = loss_detection_method_dict[Meta_params['LD_method']]
    
    if Meta_params['code'] not in ['Rotated_XZZX', 'Rotated_Surface']: 
        Meta_params['ordering'] = None # not relevant
        
    cycles_str = Meta_params['cycles']; cycles = None if (cycles_str == 'd' or cycles_str is None) else int(cycles_str)
    
    
    simulator = Simulator(Meta_params=Meta_params, 
                        noise=noise, atom_array_sim=atom_array_sim,
                        bloch_point_params=bloch_point_params, 
                        phys_err_vec=phys_err_vec,
                        loss_detection_method=loss_detection_method, cycles = cycles,
                        output_dir=output_dir, save_filename=file_name, first_comb_weight=first_comb_weight, dont_use_loss_decoder=dont_use_loss_decoder)

    simulator.simulate(distances=distances, num_shots=num_shots)



if __name__ == '__main__':
    pass