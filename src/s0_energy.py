#@title SDE Setup and Euler-Maruyama Numerical Approximation

import os
import torch
import numpy as np
import scipy as sp
import matplotlib
import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

# Making the plot outputs portable and reproducible
# matplotlib.rcParams['pdf.fonttype'] = 42
# plt.rcParams['text.antialiased'] = True
# plt.rcParams['lines.antialiased'] = True
# sns.set_theme()

# Allows user interaction
# import ipywidgets as widgets
# from IPython.display import display


def theta_fn(s, beta_max=20, beta_min=0.1):
    """
    Estimates theta in discrete time.
    Args:
        s (float or np.ndarray): A value or array of values in the interval [0, 1] for which to calculate theta, s= T- t
        beta_max (float, optional): Maximum beta value. Default is 20.
        beta_min (float, optional): Minimum beta value. Default is 0.1.
    Returns:
        float or np.ndarray: The calculated theta value(s).

    Raises:
        AssertionError: If any value in `s` is not in the interval [0, 1].
    """
    if isinstance(s,(np.ndarray)):
        assert 0<= s.all() <=1, "t \in [0,1]"
    else:
        assert 0<= s <=1, "t \in [0,1]"

    factor = -0.25*s**2*(beta_max-beta_min)-0.5*s*beta_min
    return np.exp(factor)	

def get_time_step(thetas_n, theta_c):
    """ 
    Given a tensor of theta_n values and a specified theta_c value, this function returns the index 
    in the tensor that is closest to the given theta_c value. This index corresponds to the time step
    associated with the closest theta_n value.

    Parameters
    ----------
    thetas_n : torch.Tensor
        A tensor containing thetas_n values. Each thetas_n is the variance level at a particular time step.

    theta_c : torch.Tensor
        A tensor representing a specific variance level. This function will find the time step 
        corresponding to the closest theta_n in 'thetas_n'.
        
    Returns
    -------
    int
        The index of 'thetas_n' that is closest to 'theta_c'. This index represents the time step corresponding 
        to the closest theta_n.
        
    Raises
    ------
    AssertionError
        If 'theta_c' is not in the range [0,1].
    """
    assert 0 <= theta_c.all() <= 1.0, "theta_c should be in [0,1]"

    diff = np.absolute(thetas_n - theta_c)
    return torch.tensor(diff.argmin()).item()



#@title Potential function of continous diffusion model

def U(x, t, beta_min=0.1, beta_max=20, T=1.0, N=1000): 
    """
    This function represents the potential function of a continous generative diffusion model.
    
    Input
    ----------
    x : np.ndarray or float
        The value(s) at which the potential function will be evaluated.
    t : np.ndarray or float
        The time(s) at which the potential function will be evaluated.
    beta_min : float, optional
        The minimum value of the inverse temperature beta, by default 0.1
    beta_max : float, optional
        The maximum value of the inverse temperature beta, by default 20
    T : float, optional
        The total time for the SDE to run, by default 1.0
    N : int, optional
        The total number of time steps for the SDE, by default 1000

    Returns
    -------
    np.ndarray or float
        The value(s) of the potential function at the given x and t.
    """
    beta_t = beta_min + (T-t) * (beta_max - beta_min) 
    print(beta_t)
    theta = theta_fn(T-t, beta_max=beta_max, beta_min=beta_min)
    exp1 = np.exp(-0.5 * (x - theta)**2 / (1 - theta**2))
    exp2 = np.exp(-0.5 * (x + theta)**2 / (1 - theta**2))

    log_term = np.log(exp1 + exp2) 
    print(beta_t, theta)
    return  beta_t * (-0.25 * x**2  + 2*np.sqrt(2*np.pi*(1-theta**2))-log_term)

if __name__ == '__main__':
    N = 1000        # Total number of steps
    # beta_min = 0.1  # Minimum value of beta
    # beta_max = 20   # Maximum value of beta
    eps = 1e-3      # A small value for numerical stability
    T = 1           # Total time
    # ts = np.linspace(0, T-eps, N)  # Time partitions

    t = np.linspace(0, T, N)
    axis_x = np.linspace(0,1000, 1000)[::-1]
    potential = U(0, t)
    print(potential)

    plt.plot(axis_x, potential)
    plt.gca().invert_xaxis()  # x軸を反転
    plt.savefig("/workspace/images/s_0/potential.png")

# #@title SDE parameters 
# N = 1000        # Total number of steps
# # beta_min = 0.1  # Minimum value of beta
# # beta_max = 20   # Maximum value of beta
# eps = 1e-3      # A small value for numerical stability
# T = 1           # Total time
# # ts = np.linspace(0, T-eps, N)  # Time partitions

# t = np.linspace(0, T, N)
# axis_x = np.linspace(0,1000, 1000)[::-1]
# potential = U(0, t)
# print(potential)

# plt.plot(axis_x, potential)
# plt.gca().invert_xaxis()  # x軸を反転
# plt.savefig("/workspace/images/s_0/potential.png")
