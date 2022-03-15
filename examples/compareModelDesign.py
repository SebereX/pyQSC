#!/usr/bin/env python3
import glob
import numpy as np
from qsc import Qsc
import scipy.optimize as optimize 
import os
import matplotlib.pyplot as plt

# Plot iota profiles
# path = "C:\\Users\\erodrigu\\Documents\\MATLAB\\Stellerator\\NAE\\NAE_from_design\\RichardStellarators\\QA Nfp 3 scans\\"
def findOptModel(rc, zs, B0, nfp, order):
    stel = Qsc(rc=rc, zs=zs, B0 = B0, nfp=nfp, order=order)
    def iota_eta(x):
        stel.etabar = x
        stel.order = "r1"
        stel.calculate() # It seems unnecessary having to recompute all these quantities just because some etabar was introduced in the calculate axis routine. Used to not use it but not properly updated
        val = -np.abs(stel.iotaN)
        return val

    opt = optimize.minimize(iota_eta, x0 = min(stel.curvature))

    stel.etabar = opt.x[0]
    stel.calculate()

    def B20dev(x):
        stel.B2c = x
        stel.order = "r3"
        stel.calculate_r2()
        val = stel.B20_variation
        return val

    bnd = ((-30,30))
    opt = optimize.minimize_scalar(B20dev, bounds =bnd) # , method='bounded'
    stel.B2c = opt.x
    stel.order = "r2"
    stel.calculate_r2()

    return stel

path = "C:\\Users\\erodrigu\\Documents\\MATLAB\\Stellerator\\NAE\\NAE_from_design\\RichardStellarators\\Aspect ratio scan\\"
nc_file = glob.glob(path+'wout*')
booz_file = glob.glob(path+'boozmn*')
for i in range(np.size(nc_file)):
    stel = Qsc.from_boozxform(vmec_file=nc_file[i], booz_xform_file=booz_file[i], \
                                order='r3', N_phi=200, max_s_for_fit=0.1, N_axis=10)
    stel_opt = findOptModel(stel.rc, stel.zs, stel.B0, stel.nfp, 'r2')
    # print(stel.etabar/stel_opt.etabar, stel.iota/stel_opt.iota, stel.B2c/stel_opt.B2c, stel.B20_variation/stel_opt.B20_variation, stel_opt.r_singularity/stel.r_singularity)
    # stel.calculate_shear()
    # stel_opt.calculate_shear()
    plt.subplot(2, 3, i+1)
    # plt.plot(stel.psi_vmec[1:], np.abs(stel.iota_vmec[1:]))
    # plt.plot(stel.psi_vmec, np.abs(stel.iota+stel.psi_vmec*2/stel.B0*stel.iota2))
    # plt.plot(stel.psi_vmec, np.abs(stel_opt.iota+stel.psi_vmec*2/stel_opt.B0*stel_opt.iota2))
    # plt.title(nc_file[i].replace(path,''))
    ax = plt.gca()
    r = np.sqrt(stel.psi_vmec[-1]*2/stel.B0)/2
    print(r)
    try:
        stel_opt.plot_boundary(r=r,show=False, existing_axis = ax, plot_3d=False, legend=False)
        stel.plot_boundary(r=r, show=False, existing_axis = ax,plot_3d=False, vmec = True, legend=False)
        r = np.sqrt(stel.psi_vmec[-1]*2/stel.B0)
        stel_opt.plot_boundary(r=r,show=False, existing_axis = ax, plot_3d=False, legend=False)
        stel.plot_boundary(r=r, show=False, existing_axis = ax,plot_3d=False, vmec = True, legend=False)
    except:
        continue
plt.tight_layout()
plt.show()

