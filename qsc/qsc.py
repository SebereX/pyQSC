"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import logging
import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt
from sympy import im

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
    from .Frenet_to_cylindrical import Frenet_to_cylindrical
    from .vmec_input import read_vmec
    from .qs_optim_config import choose_eta, choose_B22c, choose_Z_axis
    from .to_vmec import to_vmec
    from .util import B_mag
    
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
    def from_paper(cls, name, **kwargs):
        """
        Get one of the configurations that has been used in our papers.
        Available values for ``name`` are
        ``"r1 section 5.1"``,
        ``"r1 section 5.2"``,
        ``"r1 section 5.3"``,
        ``"r2 section 5.1"``,
        ``"r2 section 5.2"``,
        ``"r2 section 5.3"``,
        ``"r2 section 5.4"``, and
        ``"r2 section 5.5"``.
        These last 5 configurations can also be obtained by specifying an integer 1-5 for ``name``.
        The configurations that begin with ``"r1"`` refer to sections in 
        Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).
        The configurations that begin with ``"r2"`` refer to sections in 
        Landreman and Sengupta, Journal of Plasma Physics 85, 815850601 (2019).

        You can specify any other arguments of the ``Qsc`` constructor
        in ``kwargs``. You can also use ``kwargs`` to override any of
        the properties of the configurations from the papers. For
        instance, you can modify the value of ``etabar`` in the first
        example using

        .. code-block::

          q = qsc.Qsc.from_paper('r1 section 5.1', etabar=1.1)
        """

        def add_default_args(kwargs_old, **kwargs_new):
            """
            Take any key-value arguments in ``kwargs_new`` and treat them as
            defaults, adding them to the dict ``kwargs_old`` only if
            they are not specified there.
            """
            for key in kwargs_new:
                if key not in kwargs_old:
                    kwargs_old[key] = kwargs_new[key]

                    
        if name == "r1 section 5.1":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.1 """
            add_default_args(kwargs, rc=[1, 0.045], zs=[0, -0.045], nfp=3, etabar=-0.9)
                
        elif name == "r1 section 5.2":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.2 """
            add_default_args(kwargs, rc=[1, 0.265], zs=[0, -0.21], nfp=4, etabar=-2.25)
                
        elif name == "r1 section 5.3":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.3 """
            add_default_args(kwargs, rc=[1, 0.042], zs=[0, -0.042], zc=[0, -0.025], nfp=3, etabar=-1.1, sigma0=-0.6)
                
        elif name == "r2 section 5.1" or name == '5.1' or name == 1:
            """ The configuration from Landreman & Sengupta (2019), section 5.1 """
            add_default_args(kwargs, rc=[1, 0.155, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, etabar=0.64, order='r3', B2c=-0.00322)
            
        elif name == "r2 section 5.2" or name == '5.2' or name == 2:
            """ The configuration from Landreman & Sengupta (2019), section 5.2 """
            add_default_args(kwargs, rc=[1, 0.173, 0.0168, 0.00101], zs=[0, 0.159, 0.0165, 0.000985], nfp=2, etabar=0.632, order='r3', B2c=-0.158)
                             
        elif name == "r2 section 5.3" or name == '5.3' or name == 3:
            """ The configuration from Landreman & Sengupta (2019), section 5.3 """
            add_default_args(kwargs, rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r3', B2c=-0.7, p2=-600000.)
                             
        elif name == "r2 section 5.4" or name == '5.4' or name == 4:
            """ The configuration from Landreman & Sengupta (2019), section 5.4 """
            add_default_args(kwargs, rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
                       zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05], nfp=4, etabar=1.569, order='r3', B2c=0.1348)
                             
        elif name == "r2 section 5.5" or name == '5.5' or name == 5:
            """ The configuration from Landreman & Sengupta (2019), section 5.5 """
            add_default_args(kwargs, rc=[1, 0.3], zs=[0, 0.3], nfp=5, etabar=2.5, sigma0=0.3, I2=1.6, order='r3', B2c=1., B2s=3., p2=-0.5e7)

        elif name == "LandremanPaul2021QA" or name == "precise QA":
            """
            A fit of the near-axis model to the quasi-axisymmetric
            configuration in Landreman & Paul, arXiv:2108.03711 (2021).

            The fit was performed to the boozmn data using the script
            20200621-01-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=2,
                             rc=[1.0038581971135636, 0.18400998741139907, 0.021723381370503204, 0.0025968236014410812, 0.00030601568477064874, 3.5540509760304384e-05, 4.102693907398271e-06, 5.154300428457222e-07, 4.8802742243232844e-08, 7.3011320375259876e-09],
                             zs=[0.0, -0.1581148860568176, -0.02060702320552523, -0.002558840496952667, -0.0003061368667524159, -3.600111450532304e-05, -4.174376962124085e-06, -4.557462755956434e-07, -8.173481495049928e-08, -3.732477282851326e-09],
                             B0=1.006541121335688,
                             etabar=-0.6783912804454629,
                             B2c=0.26859318908803137,
                             nphi=99,
                             order='r3')

        elif name == "precise QA+well":
            """
            A fit of the near-axis model to the precise quasi-axisymmetric
            configuration from SIMSOPT with magnetic well.

            The fit was performed to the boozmn data using the script
            20200621-01-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=2,
                             rc=[1.0145598919163676, 0.2106377247598754, 0.025469267136340394, 0.0026773601516136727, 0.00021104172568911153, 7.891887175655046e-06, -8.216044358250985e-07, -2.379942694112007e-07, -2.5495108673798585e-08, 1.1679227114962395e-08, 8.961288962248274e-09],
                             zs=[0.0, -0.14607192982551795, -0.021340448470388084, -0.002558983303282255, -0.0002355043952788449, -1.2752278964149462e-05, 3.673356209179739e-07, 9.261098628194352e-08, -7.976283362938471e-09, -4.4204430633540756e-08, -1.6019372369445714e-08],
                             B0=1.0117071561808106,
                             etabar=-0.5064143402495729,
                             B2c=-0.2749140163639202,
                             nphi=99,
                             order='r3')
            
        elif name == "LandremanPaul2021QH" or name == "precise QH":
            """
            A fit of the near-axis model to the quasi-helically symmetric
            configuration in Landreman & Paul, arXiv:2108.03711 (2021).

            The fit was performed to the boozmn data using the script
            20211001-02-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=4,
                             rc=[1.0033608429348413, 0.19993025252481125, 0.03142704185268144, 0.004672593645851904, 0.0005589954792333977, 3.298415996551805e-05, -7.337736061708705e-06, -2.8829857667619663e-06, -4.51059545517434e-07],
                             zs=[0.0, 0.1788824025525348, 0.028597666614604524, 0.004302393796260442, 0.0005283708386982674, 3.5146899855826326e-05, -5.907671188908183e-06, -2.3945326611145963e-06, -6.87509350019021e-07],
                             B0=1.003244143729638,
                             etabar=-1.5002839921360023,
                             B2c=0.37896407142157423,
                             nphi=99,
                             order='r3')

        elif name == "precise QH+well":
            """
            A fit of the near-axis model to the precise quasi-helically symmetric
            configuration from SIMSOPT with magnetic well.

            The fit was performed to the boozmn data using the script
            20211001-02-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=4,
                             rc=[1.000474932581454, 0.16345392520298313, 0.02176330066615466, 0.0023779201451133163, 0.00014141976024376502, -1.0595894482659743e-05, -2.9989267970578764e-06, 3.464574408947338e-08],
                             zs=[0.0, 0.12501739099323073, 0.019051257169780858, 0.0023674771227236587, 0.0001865909743321566, -2.2659053455802824e-06, -2.368335337174369e-06, -1.8521248561490157e-08],
                             B0=0.999440074325872,
                             etabar=-1.2115187546668142,
                             B2c=0.6916862277166693,
                             nphi=99,
                             order='r3')
        elif name == "2022 QA":
            """ The QA nfp=2 configuration from section 5.1 of Landreman, arXiv:2209.11849 (2022) """
            add_default_args(
                kwargs,
                nfp=2,
                rc=[1, -0.199449520320017, 0.0239839546865877, -0.00267077266433249, 
                    0.000263369906075079, -2.28252686940861e-05, 1.77481423558342e-06, 
                    -1.11886947533483e-07],
                zs=[0, 0.153090987614971, -0.0220380957634702, 0.00273207449905532, 
                    -0.000289902600946716, 2.60032185367434e-05, -1.93900596618347e-06, 
                    1.07177057081779e-07],
                etabar=-0.546960261227405,
                B2c=-0.226693190121799,
                order='r3',
            )

        elif name == "2022 QH nfp2":
            """ The QH nfp=2 configuration from section 5.2 of Landreman, arXiv:2209.11849 (2022) """
            add_default_args(
                kwargs,
                nfp=2,
                rc=[1,0.6995680628446487, 0.23502036382115418, 0.061503864369157564,
                    0.010419139882799225, -5.311696004759487e-08,
                    -0.0007331779959884904, -0.0002900010988343009,
                    -6.617198558802484e-05, -9.241481219213564e-06,
                    -6.284956172067802e-07],
                zs=[0, 0.6214598287182819, 0.23371749756309024, 0.06541788070010997,
                    0.011645099864023048, -7.5568378122204045e-06, -0.0008931603464766644,
                    -0.00036651597175926245, -8.685195584634676e-05, -1.2617030747711465e-05,
                    -8.945854983981342e-07],
                etabar=-0.7598964639478568,
                B2c=-0.09169914960557547,
                order='r3',
            )

        elif name == "2022 QH nfp3 vacuum":
            """ The QA nfp=3 vacuum configuration from section 5.3 of Landreman, arXiv:2209.11849 (2022) """
            add_default_args(
                kwargs,
                nfp=3,
                rc=[1, 0.44342438066028106, 0.1309928804381408, 0.036826101497868344,
                    0.009472569725432308, 0.0021825892707486904, 0.00043801313411164704,
                    7.270090423024292e-05, 8.847711104877492e-06, 4.863820022333069e-07,
                    -6.48572267338807e-08, -1.309512199798216e-08],
                zs=[0, 0.40118483156012347, 0.1245296767597972, 0.0359252575240197,
                    0.009413071841635272, 0.002202227882755186, 0.00044793727963748345,
                    7.513385401132283e-05, 9.092986418282475e-06, 3.993637202113794e-07,
                    -1.1523282290069935e-07, -2.3010157353892155e-08],
                etabar=1.253110036546191,
                B2c=0.1426420204102797,
                order='r3',
            )

        elif name == "2022 QH nfp3 beta":
            """ The QA nfp=3 configuration with beta > 0 from section 5.3 of Landreman, arXiv:2209.11849 (2022) """
            add_default_args(
                kwargs,
                nfp=3,
                rc=[1, 0.35202226158037475, 0.07950774007599863, 0.01491931003455014,
                    0.0019035177304995063, 2.974489668543068e-05, -5.7768875975485955e-05,
                    -1.4029165878029966e-05, -3.636566770484427e-07, 7.14616952513107e-07,
                    2.1991450219049712e-07, 2.602997321736813e-08],
                zs=[0, 0.2933368717265116, 0.07312772496167881, 0.014677291769133093,
                    0.002032497421621057, 6.751908932231852e-05, -5.485713404214329e-05,
                    -1.5321940269647778e-05, -8.529635395421784e-07, 6.820412266134571e-07,
                    2.4768295839385676e-07, 3.428344210929051e-08],
                etabar=1.1722273002245573,
                B2c=0.04095882972842455,
                p2=-2000000.0,
                order='r3',
            )

        elif name == "2022 QH nfp4 long axis":
            """ The QA nfp=4 vacuum configuration with long magnetic axis from section 5.4 of Landreman, arXiv:2209.11849 (2022) """
            add_default_args(
                kwargs,
                nfp=4,
                rc=[1, 0.364526157493978, 0.121118955282543, 0.0442501956803706,
                    0.0167330353750722, 0.00642771673694224, 0.00248894981601539,
                    0.000968018513866967, 0.00037746342020491, 0.000147428591967393,
                    5.76454424945328e-05, 2.25531946406253e-05, 8.82168816589983e-06,
                    3.44372305427649e-06, 1.33656083525841e-06, 5.11778503697701e-07,
                    1.90503825097649e-07, 6.7112150192589e-08, 2.13264982947534e-08,
                    5.58167149815304e-09, 9.68106325338102e-10, 2.05080481895135e-11,
                    -3.01196911646619e-11, -1.24160079888634e-12],
                zs=[0, 0.348452925506333, 0.118208201403097, 0.0435300780241838,
                    0.0165374758530089, 0.00637325522789471, 0.00247384835222063,
                    0.000963898894229973, 0.000376342323515169, 0.000147097870171498,
                    5.75176913109258e-05, 2.24818089823666e-05, 8.77306129195814e-06,
                    3.4101895486024e-06, 1.31510010810783e-06, 4.99667861692231e-07,
                    1.84913811184955e-07, 6.54248703443968e-08, 2.14617350652655e-08,
                    6.18264134083988e-09, 1.39965848798875e-09, 1.80608129195069e-10,
                    -9.21974743938235e-12, -4.43329488732321e-12],
                etabar=1.44414649103253,
                B2c=0.398731902154125,
                order='r3',
            )

        elif name == "2022 QH nfp4 well":
            """ The QA nfp=4 vacuum configuration with magnetic well from section 5.4 of Landreman, arXiv:2209.11849 (2022) """
            add_default_args(
                kwargs,
                nfp=4,
                rc=[1, 0.146058715426138, 0.015392595644143, 0.000635861018673727, -0.000182708084736453,
                    -4.19903763762926e-05, -9.67542241621129e-07, 1.28914480335549e-06,
                    2.62211480710386e-07, 2.1800341254237e-10, -8.42417594606744e-09,
                    -1.15051050955081e-09, 1.12882841570496e-11, -1.02361915094644e-12,
                    1.06847707789217e-12, 2.6540060912065e-12, -6.41390701236282e-14,
                    -2.53410665876572e-13, 4.88649173594043e-14],
                zs=[0, 0.0944567013421266, 0.0124314733883564, 0.000870479159752327,
                    -9.20779986777308e-05, -3.53758456786842e-05, -2.95739225483239e-06,
                    6.84445244254255e-07, 2.22277038153942e-07, 1.43648174908159e-08,
                    -3.79406884758791e-09, 1.16393592133432e-10, 5.40884472454975e-10,
                    9.78511899449798e-11, -1.28030361642495e-11, -1.56552634173487e-12,
                    1.4116353201481e-12, 5.65879693563372e-14, 2.08417037461475e-13],
                etabar=1.10047627852273,
                B2c=0.369119914393565,
                order='r3',
            )

        elif name == "2022 QH nfp4 Mercier":
            """ The QA nfp=4 finite-beta configuration with Mercier stability from section 5.4 of Landreman, arXiv:2209.11849 (2022) """
            add_default_args(
                kwargs,
                nfp=4,
                rc=[1, 0.2129392535673467, 0.015441107613823592, -0.002203653896196156,
                    -0.0008518709398790752, -0.00010150954708573669, 3.014820573209385e-06,
                    2.5369987051312065e-06, 2.4360891263250144e-07, -8.05847285028083e-09],
                zs=[0, 0.1939647576102168, 0.01497505771701485, -0.0020519752655822378,
                    -0.0008327023600275835, -0.00010346032332812117, 2.349757901177408e-06,
                    2.5571742536309522e-06, 2.6606204989469905e-07, -1.1550694494402663e-08],
                etabar=2.477004356118308,
                B2c=-1.4201479609742513,
                p2=-1018591.6352336337,
                order='r3',
            )

        elif name == "2022 QH nfp7":
            """ The QA nfp=7 vacuum configuration from section 5.5 of Landreman, arXiv:2209.11849 (2022) """
            add_default_args(
                kwargs,
                nfp=7,
                rc=[1, 0.258042492428667, 0.0737980849415119, 0.0245235419692282,
                    0.0087489425800056, 0.00325710762056782, 0.00124798398776684,
                    0.00048824282997784, 0.000194039834892702, 7.80531836428887e-05,
                    3.16850344178095e-05, 1.29427362598943e-05, 5.30054625293292e-06,
                    2.16386280302807e-06, 8.71522369664416e-07, 3.39805766815203e-07,
                    1.2389287673191e-07, 3.9648686370117e-08, 9.84425205161371e-09,
                    1.39986408835728e-09],
                zs=[0, 0.259299681361503, 0.0744954847634618, 0.0247675058258839,
                    0.00883445611523715, 0.00328774095300022, 0.00125909923475415,
                    0.000492254404917503, 0.000195431061105875, 7.84791099248868e-05,
                    3.17656436049729e-05, 1.29116163099803e-05, 5.24481359745833e-06,
                    2.11430136671029e-06, 8.36820327603084e-07, 3.19761744590898e-07,
                    1.14811054736367e-07, 3.69561282325895e-08, 9.69489045054884e-09,
                    1.59629224344766e-09],
                etabar=2.63694707090359,
                B2c=1.89378085087321,
                order='r3',
            )
            
        else:
            raise ValueError('Unrecognized configuration name')

        return cls(**kwargs)

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
        
    def min_R0_penalty(self):
        """
        This function can be used in optimization to penalize situations
        in which min(R0) < min_R0_constraint.
        """
        return np.max((0, self.min_R0_threshold - self.min_R0)) ** 2
        
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