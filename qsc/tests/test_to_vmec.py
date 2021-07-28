#!/usr/bin/env python3

import unittest
import os
from scipy.io import netcdf
import numpy as np
import logging
from qsc.qsc import Qsc
from mpi4py import MPI
import vmec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_to_fortran(name, filename):
    """
    Compare output from pyQSC to the fortran code, for one
    of the example configurations from the papers.
    """
    # Add the directory of this file to the specified filename:
    abs_filename = os.path.join(os.path.dirname(__file__), filename)
    f = netcdf.netcdf_file(abs_filename, 'r')
    nphi = f.variables['N_phi'][()]
    r = f.variables['r'][()]
    mpol = f.variables['mpol'][()]
    ntor = f.variables['ntor'][()]

    logger.info('Creating pyQSC configuration')
    py = Qsc.from_paper(name, nphi=nphi)
    logger.info('Outputing to VMEC')
    py.to_vmec(str("input."+name).replace(" ",""), r=r,params={'mpol': mpol, 'ntor': ntor})

    logger.info('Comparing to fortran file ' + abs_filename)
    def compare_field(fortran_name, py_field, rtol=1e-9, atol=1e-9):
        fortran_field = f.variables[fortran_name][()]
        logger.info('max difference in {}: {}'.format(fortran_name, np.max(np.abs(fortran_field - py_field))))
        np.testing.assert_allclose(fortran_field, py_field, rtol=rtol, atol=atol)

    compare_field('RBC', py.RBC)
    compare_field('RBS', py.RBS)
    compare_field('ZBC', py.ZBC)
    compare_field('ZBS', py.ZBS)
    f.close()

def compare_to_vmec(name, r=0.01, nphi=101):
    """
    Check that VMEC can run the input file outputed by pyQSC
    and check that the resulting VMEC output file has
    the expected parameters
    """
    # Add the directory of this file to the specified filename:
    inputFile="input."+str(name).replace(" ","")
    abs_filename = os.path.join(os.path.dirname(__file__), inputFile)
    # Run pyQsc and create a VMEC input file
    logger.info('Creating pyQSC configuration')
    py = Qsc.from_paper(name, nphi=nphi)
    logger.info('Outputing to VMEC')
    py.to_vmec(inputFile,r)
    # Run VMEC
    fcomm = MPI.COMM_WORLD.py2f()
    logger.info("Calling runvmec. comm={}".format(fcomm))
    vmec.runvmec(np.array([15,0,0,0,0], dtype=np.int32), inputFile, True, fcomm, '')
    # Open VMEC output file
    woutFile="wout_"+str(name).replace(" ","")+".nc"
    f = netcdf.netcdf_file(woutFile, 'r')
    # Compare the results
    print('pyQSC iota on axis =',py.iota)
    print('VMEC iota on axis =',-f.variables['iotaf'][()][0])
    print('pyQSC field on axis =',py.B0)
    print('VMEC bmnc[1][0] =',f.variables['bmnc'][()][1][0])
    assert np.isclose(py.iota,-f.variables['iotaf'][()][0],rtol=1e-2)
    assert np.isclose(py.B0,f.variables['bmnc'][()][1][0],rtol=1e-2)
    vmec.cleanup(True)
    f.close()

class ToVmecTests(unittest.TestCase):

    def test_boundary(self):
        """
        Compare the RBC/RBS/ZBC/ZBS values to those generated by the fortran version,
        for the 3 O(r^1) examples in LandremanSenguptaPlunk. When the second order
        is successfully added to to_vmec, these tests might go to test_qsc.py
        """
        compare_to_fortran("r1 section 5.1", "quasisymmetry_out.LandremanSenguptaPlunk_section5.1_order_r1_finite_r_nonlinear.reference.nc")
        compare_to_fortran("r1 section 5.2", "quasisymmetry_out.LandremanSenguptaPlunk_section5.2_order_r1_finite_r_nonlinear.reference.nc")
        compare_to_fortran("r1 section 5.3", "quasisymmetry_out.LandremanSenguptaPlunk_section5.3_order_r1_finite_r_nonlinear.reference.nc")

    def test_vmec(self):
        """
        Verify that vmec can actually read the generated input files
        and that vmec's Bfield and iota on axis match the predicted values.
        """
        compare_to_vmec("r1 section 5.1")
        compare_to_vmec("r1 section 5.2")
        compare_to_vmec("r1 section 5.3")

    def test_3(self):
        """
        Transforming with to_Fourier and then un-transforming should give the identity,
        for both even and odd ntheta and phi, and for lasym True or False.
        """

if __name__ == "__main__":
    unittest.main()