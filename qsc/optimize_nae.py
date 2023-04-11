"""
This module optimises parameters of a NAE
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
from scipy import optimize
from .util import mu0

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def opt_fun_stel(x_iter, stel, x_param_label, fun_opt, info = {'Nfeval':0}, res_history = [],verbose =0, extras =[], x_orig = [], thresh = 1.5):
    info['Nfeval'] += 1
    x_iter_all = stel.get_dofs()
    if len(x_orig):
        for ind, label in enumerate(x_param_label):
            if np.any(x_iter[ind]>thresh) or np.any(x_iter[ind]<1/thresh):
                res = 1000
                res_history.append(res)
                return res
            x_iter_all[stel.names.index(label)] = x_iter[ind]*x_orig[stel.names.index(label)]
    else:
        for ind, label in enumerate(x_param_label):
            x_iter_all[stel.names.index(label)] = x_iter[ind]
    stel.set_dofs(x_iter_all)
    res = fun_opt(stel, extras)
    if verbose:
        print(f"{info['Nfeval']} -", x_iter)
        print(f"\N{GREEK CAPITAL LETTER DELTA}B20 = {stel.B20_variation:.4f},",
            f"1/rc = {1/stel.r_singularity:.2f},",
            f"1/L\N{GREEK CAPITAL LETTER DELTA}B = {np.max(stel.inv_L_grad_B):.2f},",
            f"Residual = {res:.4f}")
    res_history.append(res)
    return res

def fun(stel, extras):
    # QS quality as a simple first case
    res = stel.B20_variation
    return res

def optimise_params(stel, x_param_label, fun_opt = fun, verbose = 0, maxiter = 200, maxfev  = 200, method = 'Nelder-Mead', scale = 0, extras = [], thresh = 1.5):
    # Example of x_parameter_label: (['etabar', 'rc(1)'])

    x0_all = stel.get_dofs()
    x_parameter_label_checked = []
    x0 = []
    if scale:
        for ind, label in enumerate(x_param_label):
            if label in stel.names:
                x0.append(1.0)
                x_parameter_label_checked.append(label)
            else:
                print('The label ',label, 'does not exist, and will be ignored.')
    else:
        for ind, label in enumerate(x_param_label):
            if label in stel.names:
                x0.append(x0_all[stel.names.index(label)])
                x_parameter_label_checked.append(label)
            else:
                print('The label ',label, 'does not exist, and will be ignored.')

    res_history = []
    res_history.append(fun_opt(stel, extras))
    
    if scale:
        opt = optimize.minimize(opt_fun_stel, x0, args=(stel, x_parameter_label_checked, fun_opt, {'Nfeval':0}, res_history, verbose, extras, x0_all, thresh), method=method, tol=1e-3, options={'maxiter': maxiter, 'maxfev': maxfev})
    else:
        opt = optimize.minimize(opt_fun_stel, x0, args=(stel, x_parameter_label_checked, fun_opt, {'Nfeval':0}, res_history, verbose, extras), method=method, tol=1e-3, options={'maxiter': maxiter, 'maxfev': maxfev})

    x = stel.get_dofs()
    if scale:
        for ind, label in enumerate(x_parameter_label_checked):
            x[stel.names.index(label)] = opt.x[ind]*x0_all[stel.names.index(label)]
    else:
        for ind, label in enumerate(x_parameter_label_checked):
            x[stel.names.index(label)] = opt.x[ind]
    stel.set_dofs(x)
    
    if verbose:
        print(opt.x)
        plt.plot(res_history)
        plt.ylabel('Objective function')
        plt.show()
    
    return opt.message
