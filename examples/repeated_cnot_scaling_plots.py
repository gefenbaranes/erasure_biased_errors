import matplotlib.pyplot as plt
from BiasedErasure.delayed_erasure_decoders.Experimental_Loss_Decoder import *
import numpy as np


fidelities_5_3 = 1-np.array([0.967, 0.9393333333333334, 0.904, 0.8543333333333334, 0.8143333333333334, 0.782, 0.74, 0.6946666666666667])
fidelities_3_3 = 1-np.array([0.9393333333333334, 0.912, 0.8766666666666667, 0.8486666666666667, 0.8156666666666667, 0.7576666666666667, 0.742, 0.7093333333333334])

num_rounds = 3
distance = 5
decoder_basis = 'XX'
gate_ordering = ['N', 'Z']
num_cxs_per_round = 3
num_cxs_per_rounds = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17])

plt.plot(num_cxs_per_rounds[:len(fidelities_5_3)], .5*(1-(1-fidelities_5_3/.5)**(1/(3*num_cxs_per_rounds[:len(fidelities_5_3)]))), 'o-', color='blue', label='d=5, num_cxs_per_round=3')
plt.plot(num_cxs_per_rounds[:len(fidelities_3_3)], .5*(1-(1-fidelities_3_3/.5)**(1/(3*num_cxs_per_rounds[:len(fidelities_3_3)]))), 'o-', color='purple', label='d=3, num_cxs_per_round=3')
plt.xlabel('CNOTs per round')
plt.ylabel('Logical error rate per CNOT')
plt.xticks(num_cxs_per_rounds)
plt.legend()
plt.show()