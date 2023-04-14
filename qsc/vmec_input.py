"""
This module contains a function to read VMEC files.
"""
import logging
import os
import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def read_vmec(cls, vmec_file, path = os.path.dirname(__file__), N_axis = []):
    """
    Read VMEC file to extract the shape of the magnetic axis (to construct NAE) and flux surfaces
    out of the equilibrium file.

    Args:
        file_name: name of the VMEC file
        path: path to the VMEC file.
        N_axis: number of harmonics to keep to describe the magnetic axis. If N_axis = [], then
            all harmonics available are kept
    """
    if path:
        vmec_file = os.path.join(path, vmec_file)
    f = netcdf.netcdf_file(vmec_file,'r',mmap=False)
    nfp = f.variables['nfp'][()]
    rc = f.variables['raxis_cc'][()]
    zs = f.variables['zaxis_cs'][()]
    if N_axis>len(rc):
        N_axis = []
    if N_axis:
        if N_axis<=len(rc):
            rc = rc[0:N_axis]
            zs = zs[0:N_axis]
        else:
            N_axis = []
    psi = f.variables['phi'][()]/2/np.pi
    psi_booz_vmec = np.abs(psi)
    am = f.variables['am'][()] # Pressure profile polynomial
    bsubumnc = f.variables['bsubumnc'][()]   
    bsubvmnc = f.variables['bsubvmnc'][()] 
    gmnc = f.variables['gmnc'][()]
    rmnc = f.variables['rmnc'][()]
    zmns = -f.variables['zmns'][()] 
    gmnc = f.variables['gmnc'][()]
    xm_vmec = f.variables['xm'][()]
    xn_vmec = f.variables['xn'][()]
    iota_vmec = f.variables['iotas'][()] 
    try:
        rs = -f.variables['raxis_cs'][()]
        zc = f.variables['zaxis_cc'][()]
        if N_axis:
            rs = rs[0:N_axis]
            zc = zc[0:N_axis]
        logger.info('Non stellarator symmetric configuration')
    except:
        rs=[]
        zc=[]
        logger.info('Stellarator symmetric configuration')
    # Vp_vmec = f.variables['vp'][()] 
    # plt.plot(Vp_vmec)
    f.close()

    # Save VMEC attributes to class
    cls.s_n = rc*(1+nfp**2*np.arange(0,np.size(rc),1)**2)/rc[0]
    cls.rmnc_vmec = rmnc
    cls.zmns_vmec = zmns
    cls.gmnc_vmec = gmnc
    cls.xm_vmec = xm_vmec
    cls.xn_vmec = xn_vmec
    cls.psi_vmec = psi_booz_vmec
    cls.iota_vmec = iota_vmec
    
    return psi_booz_vmec, rc, rs, zc, zs, am, bsubumnc, bsubvmnc

# def read_boozxform(file_name, path = os.path.dirname(__file__), helicity=0):
#     """
#     Read BOOZXFORM file to extract the eta parameter associated to the NAE model

#     Args:
#         file_name: name of the BOOZXFORM file
#         path: path to the BOOZXFORM file.
#     """
#     file_abs = os.path.join(path, file_name)
#     f = netcdf.netcdf_file(file_abs, mode='r', mmap=False)
#     bmnc_b = f.variables['bmnc_b'][()]
#     ixm_b = f.variables['ixm_b'][()]
#     ixn_b = f.variables['ixn_b'][()]
#     jlist = f.variables['jlist'][()]
#     f.close()
#     for i in range(np.size(jlist)):
#         if ixm_b[i] == 1 and ixn_b[i]-ixm_b[i]*helicity==0:
#             b_cos = bmnc_b[:,i]
#         elif ixm_b[i] == 2 and ixn_b[i]-ixm_b[i]*helicity==0:
#             b_cos2 = bmnc_b[:,i]
#         elif ixm_b[i] == 0 and ixn_b[i] == 0:
#             b_0 = bmnc_b[:,i]
#     return b_cos2, b_cos, b_0