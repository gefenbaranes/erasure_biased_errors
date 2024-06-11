import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.optimize import curve_fit
import copy

def draw_graph2(g, save=False, filename=None, from_edges=False, size = (2,2)):
    if from_edges:
        g_ = nx.Graph()
        n_nodes = max([m for e in g for m in e])
        g_.add_nodes_from(list(range(n_nodes)))
        g_.add_edges_from(g)
        g = g_
    colour_map = []
    for n in range(len(g.nodes)):
        colour_map.append('gold')
    pos = nx.kamada_kawai_layout(g)
    plt.figure(figsize=size)
    nx.draw(g, pos=pos, node_color=colour_map, with_labels=True)

    edge_labels = nx.get_edge_attributes(g, "weight")
    print([e for e in edge_labels])
    #print(pos)
    nx.draw_networkx_edge_labels(g, pos, edge_labels)
    if save:
        plt.savefig(filename)
    plt.show()

def draw_graph(G):

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 1]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 1]

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")

    for key, value in edge_labels.items():
        edge_labels[key] = round(value,2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return edge_labels



def model_func(x, A, B, C, threshold, alpha, beta):
    p_physical = x[0]
    n = x[1]
    x_val = (p_physical - threshold) * alpha * n**beta
    p_logical = A + B * x_val + C * x_val**2
    return p_logical

def fit_model(p_physical, p_logical, n):
    def fit_func(x, A, B, C, threshold, alpha, beta):
        return model_func(x, A, B, C, threshold, alpha, beta) - p_logical

    popt, _ = curve_fit(fit_func, (p_physical, n), p_logical, method='lm')

    A_fit, B_fit, C_fit, threshold_fit, alpha_fit, beta_fit = popt
    return alpha_fit, beta_fit, threshold_fit, A_fit, B_fit, C_fit



# Functions for fitting the threshold
def CriticalExponentFit(xdata_tuple, pc, nu, A, B, C):
    p, d = xdata_tuple
    x = (p - pc)*d**(1/nu)
    pl = A + B*x + C*x**2
    return pl

def EmpericalFit(xdata_tuple, pc, A):
    p, d = xdata_tuple
    pl = A*(p/pc)**(d/2)
    return pl

def FitDistance(p, A, d):
    pl = A*p**(d/2)
    return pl

def DistanceEst(sweep_p_list, sweep_pl_total_list, if_plot=False):
    num_p = len(sweep_p_list)
    num_code = len(sweep_pl_total_list)
    sweep_d_list = []
    for sweep_pl_list in sweep_pl_total_list:
        initial_guess = (0.01, 3)
        popt, pcov = curve_fit(FitDistance, np.array(sweep_p_list), np.array(sweep_pl_list) + 1e-10, p0=initial_guess)
        sweep_d_list.append(popt[1])

    return sweep_d_list

 
       
def ThresholdEst_extrapolation(sweep_p_list, sweep_pl_total_list, if_plot=False):
    num_p = len(sweep_p_list)
    num_code = len(sweep_pl_total_list)
    sweep_d_list = DistanceEst(sweep_p_list, sweep_pl_total_list, if_plot=False)
    
    fit_d_list = copy.deepcopy(sweep_d_list)
    sweep_p_list = list(sweep_p_list)*num_code
    sweep_d1_list = []
    for sweep_d in sweep_d_list:
        sweep_d1_list += [sweep_d]*num_p
    sweep_d_list = sweep_d1_list
    sweep_pl_list = list(np.reshape(np.array(sweep_pl_total_list) + 1e-10, [num_p*num_code, ]))
    
    fit_X = np.vstack([np.reshape(np.array(sweep_p_list), [1, num_p*num_code]), 
                       np.reshape(np.array(sweep_d_list), [1, num_p*num_code])])
    fit_Z = np.reshape(np.array(sweep_pl_total_list), [num_p*num_code, ])
    initial_guess = (0.04, 0.1)
    popt, pcov = curve_fit(EmpericalFit, fit_X, fit_Z, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    
    # plot
    p_c, A = popt[0], popt[1]
    fit_p_list = list(set(sweep_p_list))
    fit_pl_list = np.reshape(np.array(sweep_pl_list), [len(fit_d_list), len(fit_p_list)])
    if if_plot:
        fitted_pl_list = []
        for sweep_d in fit_d_list:
            fitted_pl_list.append([EmpericalFit((sweep_p, sweep_d), p_c, A) for sweep_p in fit_p_list])
        
        plt.figure()
        for i in range(len(fit_d_list)):
            plt.plot(fit_p_list, fitted_pl_list[i], '-', c = 'C%i'%i)
            plt.plot(sweep_p_list[:num_p], sweep_pl_list[i*num_p:(i + 1)*num_p], 'D', c = 'C%i'%i)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('p')
        plt.ylabel('WER')
    
    print('p_c:', popt[0])
    
    return popt[0]

def DistanceEst(sweep_p_list, sweep_pl_total_list, if_plot=False):
    num_p = len(sweep_p_list)
    num_code = len(sweep_pl_total_list)
    sweep_d_list = []
    for sweep_pl_list in sweep_pl_total_list:
        initial_guess = (0.01, 3)
        popt, pcov = curve_fit(FitDistance, np.array(sweep_p_list), np.array(sweep_pl_list) + 1e-10, p0=initial_guess)
        sweep_d_list.append(popt[1])

    return sweep_d_list

def FitDistance(p, A, d):
    pl = A*p**(d/2)
    return pl

 

if __name__ == '__main__':
    # Example usage
    p_physical = np.array([1.0, 2.0, 3.0, 4.0])
    p_logical = np.array([5.0, 8.0, 15.0, 26.0])
    n = 10  # Your choice of n

    alpha, beta, threshold, A, B, C = fit_model(p_physical, p_logical, n)
    print("alpha:", alpha)
    print("beta:", beta)
    print("threshold:", threshold)
    print("A:", A)
    print("B:", B)
    print("C:", C)
