"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import logging
import numpy as np
from scipy.io import netcdf
#from numba import jit

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qsc():
    """
    This is the main class for representing the quasisymmetric
    stellarator construction.
    """
    
    # Import methods that are defined in separate files:
    from .init_axis import init_axis, convert_to_spline
    from .calculate_r1 import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity, r1_diagnostics
    from .grad_B_tensor import calculate_grad_B_tensor, calculate_grad_grad_B_tensor, \
        Bfield_cylindrical, Bfield_cartesian, grad_B_tensor_cartesian, \
        grad_grad_B_tensor_cylindrical, grad_grad_B_tensor_cartesian
    from .calculate_r2 import calculate_r2
    from .calculate_r3 import calculate_r3, calculate_shear
    from .mercier import mercier
    from .r_singularity import calculate_r_singularity
    from .plot import plot, plot_boundary, get_boundary, get_boundary_vmec, B_fieldline, B_contour, plot_axis, flux_tube
    from .Frenet_to_cylindrical import Frenet_to_cylindrical, to_RZ
    from .vmec_input import read_vmec
    from .qs_optim_config import choose_eta, choose_B22c, choose_Z_axis
    from .optimize_nae import optimise_params
    from .to_vmec import to_vmec
    from .vmec_input import read_vmec
    from .util import B_mag
    from .configurations import from_paper, configurations
    
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1.,
                 I2=0., sG=1, spsi=1, nphi=61, B2s=0., B2c=0., p2=0., order="r1"):
        """
        Create a quasisymmetric stellarator.
        """
        # First, force {rc, zs, rs, zc} to have the same length, for
        # simplicity.
        nfourier = np.max([len(rc), len(zs), len(rs), len(zc)])
        self.nfourier = nfourier
        self.rc = np.zeros(nfourier)
        self.zs = np.zeros(nfourier)
        self.rs = np.zeros(nfourier)
        self.zc = np.zeros(nfourier)
        self.rc[:len(rc)] = rc
        self.zs[:len(zs)] = zs
        self.rs[:len(rs)] = rs
        self.zc[:len(zc)] = zc

        # Force nphi to be odd:
        if np.mod(nphi, 2) == 0:
            nphi += 1

        if sG != 1 and sG != -1:
            raise ValueError('sG must be +1 or -1')
        
        if spsi != 1 and spsi != -1:
            raise ValueError('spsi must be +1 or -1')

        self.nfp = nfp
        self.etabar = etabar
        self.sigma0 = sigma0
        self.B0 = B0
        self.I2 = I2
        self.sG = sG
        self.spsi = spsi
        self.nphi = nphi
        self.B2s = B2s
        self.B2c = B2c
        self.p2 = p2
        self.order = order
        self.min_R0_threshold = 0.3
        self._set_names()

        self.calculate()

    def change_nfourier(self, nfourier_new):
        """
        Resize the arrays of Fourier amplitudes. You can either increase
        or decrease nfourier.
        """
        rc_old = self.rc
        rs_old = self.rs
        zc_old = self.zc
        zs_old = self.zs
        index = np.min((self.nfourier, nfourier_new))
        self.rc = np.zeros(nfourier_new)
        self.rs = np.zeros(nfourier_new)
        self.zc = np.zeros(nfourier_new)
        self.zs = np.zeros(nfourier_new)
        self.rc[:index] = rc_old[:index]
        self.rs[:index] = rs_old[:index]
        self.zc[:index] = zc_old[:index]
        self.zs[:index] = zs_old[:index]
        nfourier_old = self.nfourier
        self.nfourier = nfourier_new
        self._set_names()
        # No need to recalculate if we increased the Fourier
        # resolution, only if we decreased it.
        if nfourier_new < nfourier_old:
            self.calculate()

    def calculate(self):
        """
        Driver for the main calculations.
        """
        self.init_axis()
        if self.order != 'r0':
            self.solve_sigma_equation()
            self.r1_diagnostics()
            if self.order != 'r1':
                self.calculate_r2()
                if self.order == 'r3':
                    self.calculate_r3()
    
    def get_dofs(self):
        """
        Return a 1D numpy vector of all possible optimizable
        degrees-of-freedom, for simsopt.
        """
        return np.concatenate((self.rc, self.zs, self.rs, self.zc,
                               np.array([self.etabar, self.sigma0, self.B2s, self.B2c, self.p2, self.I2, self.B0])))

    def set_dofs(self, x):
        """
        For interaction with simsopt, set the optimizable degrees of
        freedom from a 1D numpy vector.
        """
        assert len(x) == self.nfourier * 4 + 7
        self.rc = x[self.nfourier * 0 : self.nfourier * 1]
        self.zs = x[self.nfourier * 1 : self.nfourier * 2]
        self.rs = x[self.nfourier * 2 : self.nfourier * 3]
        self.zc = x[self.nfourier * 3 : self.nfourier * 4]
        self.etabar = x[self.nfourier * 4 + 0]
        self.sigma0 = x[self.nfourier * 4 + 1]
        self.B2s = x[self.nfourier * 4 + 2]
        self.B2c = x[self.nfourier * 4 + 3]
        self.p2 = x[self.nfourier * 4 + 4]
        self.I2 = x[self.nfourier * 4 + 5]
        self.B0 = x[self.nfourier * 4 + 6]
        self.calculate()
        logger.info('set_dofs called with x={}. Now iota={}, elongation={}'.format(x, self.iota, self.max_elongation))
        
    def _set_names(self):
        """
        For simsopt, sets the list of names for each degree of freedom.
        """
        names = []
        names += ['rc({})'.format(j) for j in range(self.nfourier)]
        names += ['zs({})'.format(j) for j in range(self.nfourier)]
        names += ['rs({})'.format(j) for j in range(self.nfourier)]
        names += ['zc({})'.format(j) for j in range(self.nfourier)]
        names += ['etabar', 'sigma0', 'B2s', 'B2c', 'p2', 'I2', 'B0']
        self.names = names

    @classmethod
    def from_cxx(cls, filename):
        """
        Load a configuration from a ``qsc_out.<extension>.nc`` output file
        that was generated by the C++ version of QSC. Almost all the
        data will be taken from the output file, over-writing any
        calculations done in python when the new Qsc object is
        created.
        """
        def to_string(nc_str):
            """ Convert a string from the netcdf binary format to a python string. """
            temp = [c.decode('UTF-8') for c in nc_str]
            return (''.join(temp)).strip()
        
        f = netcdf.netcdf_file(filename, mmap=False)
        nfp = f.variables['nfp'][()]
        nphi = f.variables['nphi'][()]
        rc = f.variables['R0c'][()]
        rs = f.variables['R0s'][()]
        zc = f.variables['Z0c'][()]
        zs = f.variables['Z0s'][()]
        I2 = f.variables['I2'][()]
        B0 = f.variables['B0'][()]
        spsi = f.variables['spsi'][()]
        sG = f.variables['sG'][()]
        etabar = f.variables['eta_bar'][()]
        sigma0 = f.variables['sigma0'][()]
        order_r_option = to_string(f.variables['order_r_option'][()])
        if order_r_option == 'r2.1':
            order_r_option = 'r3'
        if order_r_option == 'r1':
            p2 = 0.0
            B2c = 0.0
            B2s = 0.0
        else:
            p2 = f.variables['p2'][()]
            B2c = f.variables['B2c'][()]
            B2s = f.variables['B2s'][()]

        q = cls(nfp=nfp, nphi=nphi, rc=rc, rs=rs, zc=zc, zs=zs,
                B0=B0, sG=sG, spsi=spsi,
                etabar=etabar, sigma0=sigma0, I2=I2, p2=p2, B2c=B2c, B2s=B2s, order=order_r_option)
        
        def read(name, cxx_name=None):
            if cxx_name is None: cxx_name = name
            setattr(q, name, f.variables[cxx_name][()])

        [read(v) for v in ['R0', 'Z0', 'R0p', 'Z0p', 'R0pp', 'Z0pp', 'R0ppp', 'Z0ppp',
                           'sigma', 'curvature', 'torsion', 'X1c', 'Y1c', 'Y1s', 'elongation']]
        if order_r_option != 'r1':
            [read(v) for v in ['X20', 'X2c', 'X2s', 'Y20', 'Y2c', 'Y2s', 'Z20', 'Z2c', 'Z2s', 'B20']]
            if order_r_option != 'r2':
                [read(v) for v in ['X3c1', 'Y3c1', 'Y3s1']]
                    
        f.close()
        return q
    
    @classmethod
    def from_boozxform(cls, vmec_file, booz_xform_file, order='r2', min_s_for_fit = 0.05, max_s_for_fit = 0.4, N_phi = [], N_axis = [],
                        rc=[], rs=[], zc=[], zs=[], sigma0=0, I2=0, p2=0, order_fit = 0):
        """
        Load a configuration from a VMEC and a BOOZ_XFORM output files
        """
        # Read properties of BOOZ_XFORM output file
        f = netcdf.netcdf_file(booz_xform_file,'r',mmap=False)
        bmnc = f.variables['bmnc_b'][()]
        ixm = f.variables['ixm_b'][()]
        ixn = f.variables['ixn_b'][()]
        jlist = f.variables['jlist'][()]
        nfp = f.variables['nfp_b'][()]
        f.close()

        # Read axis-shape from VMEC output file
        psi_booz_vmec, rc, rs, zc, zs, am, bsubumnc, bsubvmnc = Qsc.read_vmec(cls, vmec_file, N_axis = N_axis)
        psi_edge = psi_booz_vmec[-1]
        # Calculate nNormal
        stel = Qsc(rc=rc, rs=rs, zc=zc, zs=zs, nfp=nfp)
        helicity = stel.iota - stel.iotaN

        # Prepare coordinates for fit
        psi_booz = psi_booz_vmec[jlist-1]
        sqrt_psi_booz = np.sqrt(psi_booz)
        mask = (psi_booz/psi_edge < max_s_for_fit) & (psi_booz/psi_edge > min_s_for_fit)
        mask_vmec = (psi_booz_vmec/psi_edge < max_s_for_fit) & (psi_booz_vmec/psi_edge > min_s_for_fit)
        # s_fine = np.linspace(0,1,400)
        # sqrts_fine = s_fine
        if N_phi:
            phi = np.linspace(0,2*np.pi / nfp, N_phi)
            B0_phi  = np.zeros(N_phi)
            B1s_phi = np.zeros(N_phi)
            B1c_phi = np.zeros(N_phi)
            B20_phi = np.zeros(N_phi)
            B2s_phi = np.zeros(N_phi)
            B2c_phi = np.zeros(N_phi)
            chck_phi = 1
        else:
            chck_phi = 0
            N_phi = 200
        ### PARAMETER FIT ####
        # Perform fit of parameters for NAE
        for jmn in range(len(ixm)):
            m = ixm[jmn]
            n = ixn[jmn]
            if m>2:
                continue
            if m==0:
                # For m=0, fit a polynomial in s (not sqrt(s)) that does not need to go through the origin.
                if n==0:
                    b_0 = bmnc[mask,jmn]
                    poly_ord = np.arange(3+order_fit) # [2, 1, 0]
                    z = np.polynomial.polynomial.polyfit(psi_booz[mask], b_0, poly_ord)
                    B0 = z[0]
                    B20 = z[1]/2*B0
                if chck_phi==1:
                    poly_ord = np.arange(3+order_fit)
                    z = np.polynomial.polynomial.polyfit(psi_booz[mask], bmnc[mask,jmn], poly_ord)
                    B0_phi += z[0] * np.cos(n*phi)
                    B20_phi += z[1] * np.cos(n*phi)/2*b_0[0]
            if m==1:
                if ixn[jmn]-ixm[jmn]*helicity==0:
                    poly_ord = np.arange(1,4+2*order_fit,2) # [1, 3]
                    b_cos = bmnc[mask,jmn]
                    z = np.polynomial.polynomial.polyfit(sqrt_psi_booz[mask], b_cos, poly_ord)
                    etabar = z[1]/np.sqrt(2*B0)
                    B31cp = z[3]
                if chck_phi==1:
                    poly_ord = np.arange(1,4+2*order_fit,2)
                    z = np.polynomial.polynomial.polyfit(sqrt_psi_booz[mask],bmnc[mask,jmn], poly_ord)
                    B1c_phi += z[1] * np.cos((n-helicity)*phi)*np.sqrt(B0/2)
                    B1s_phi += z[1] * np.sin((n-helicity)*phi)*np.sqrt(B0/2)
            if m==2:
                # For m=2, fit a polynomial in s (not sqrt(s)) that does need to go through the origin.
                if ixn[jmn]-ixm[jmn]*helicity==0:
                    poly_ord = np.arange(1,3+order_fit) # [1, 2]
                    z = np.polynomial.polynomial.polyfit(psi_booz[mask], bmnc[mask,jmn], poly_ord)
                    B2c = z[1]/2*B0
                if chck_phi==1:
                    poly_ord = np.arange(1,3+order_fit)
                    z = np.polynomial.polynomial.polyfit(psi_booz[mask], bmnc[mask,jmn], poly_ord)
                    B2c_phi += z[1] * np.cos((n-2*helicity)*phi)/2*B0
                    B2s_phi += z[1] * np.sin((n-2*helicity)*phi)/2*B0

        # Compute B31c: note that we want the 1/B**2 B31c component and not that of B (the shear
        # expression was obtained using the Jacobian form of |B|). Very sensitive, does not appear
        # to be too reliable (perhaps need a larger s_max for the fit). Often better to take B31c=0 for shear
        B20c = 4*B0**4*(0.75*B0*etabar**2-B20)
        B22c = 4*B0**4*(0.75*B0*etabar**2-B2c)
        eta = etabar*np.sqrt(2/B0)
        B31c = -2/B0**2*(B31cp/B0+1.5*eta*(B20c*B0**2+B22c/2*B0**2)-15*eta**3/8)
        # print(B31c)

        # Read I2 from VMEC
        if I2==0:
            for jmn in range(len(ixm)):
                m = ixm[jmn]
                n = ixn[jmn]
                if m==0 and n==0:
                    poly_ord = np.arange(3+order_fit)   # [0, 1, 2]
                    G_psi = bsubvmnc[mask_vmec,jmn]
                    I_psi = bsubumnc[mask_vmec,jmn]
                    z = np.polynomial.polynomial.polyfit(psi_booz_vmec[mask_vmec], I_psi, poly_ord)
                    I2 = z[1]*B0/2
                    if np.abs(I2)<1e-10:
                        I2=0
        # Read p2 from VMEC
        if p2==0:
            r  = np.sqrt(2*psi_edge/B0)
            p2 = am[1]/r/r

        ### CONSTRUCT NAE MODEL (using same cls as VMEC read) ####
        if order=='r1':
            q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=etabar,nphi=N_phi,nfp=nfp,B0=B0,sigma0=sigma0, I2=I2)
        else:
            q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=etabar,nphi=N_phi,nfp=nfp,B0=B0,sigma0=sigma0, I2=I2, B2c=B2c, order=order)
        if chck_phi==1:
            q.B0_boozxform_array=B0_phi
            q.B1c_boozxform_array=B1c_phi
            q.B1s_boozxform_array=B1s_phi
            q.B20_boozxform_array=B20_phi
            q.B2c_boozxform_array=B2c_phi
            q.B2s_boozxform_array=B2s_phi
        q.B31c = B31c
        return q
    
    def min_R0_penalty(self):
        """
        This function can be used in optimization to penalize situations
        in which min(R0) < min_R0_constraint.
        """
        return np.max((0, self.min_R0_threshold - self.min_R0)) ** 2
        
