from qsc import Qsc
import numpy as np
import sys
from scipy import optimize
import matplotlib.pyplot as plt

nfp=4
Na = int(sys.argv[1])
Nb = int(sys.argv[4])
aVal = np.linspace(float(sys.argv[2]),float(sys.argv[3]),Na,endpoint=False) #Endpoint not included to avoid repetition when splitting the whole space into quadrants
bVal = np.linspace(float(sys.argv[5]),float(sys.argv[6]),Nb,endpoint=False) #Drawback is that we will be missing the outer edge, but at least we do not repeat work unnecessarily
dirFolder = sys.argv[7]
np.savetxt(dirFolder + 'a.out', aVal)   
np.savetxt(dirFolder + 'b.out', bVal)   

zRela = np.zeros((Na,Nb))
zRelb = np.zeros((Na,Nb))
etabar = np.zeros((Na,Nb))
iota = np.zeros((Na,Nb))
hel = np.zeros((Na,Nb))
B22c = np.zeros((Na,Nb))
shear = np.zeros((Na,Nb))
delB20 = np.zeros((Na,Nb))
avB20  = np.zeros((Na,Nb))
resB20 = np.zeros((Na,Nb))

for i in range(Na):
    for j in range(Nb):
        try:
            rcOr=[1, aVal[i], bVal[j]]
            zsOr=[0.0, aVal[i], bVal[j]]
            # print(rcOr)
            B0=1
            # stel = Qsc(rc=rcOr, zs=zsOr, B0 = B0, nfp=nfp, order='r1')
            def findZOpt(z, info):
                zs = zsOr.copy()
                zs[1] = z[0]*rcOr[1]
                zs[2] = z[1]*rcOr[2]
                stel = Qsc(rc=rcOr, zs=zs, B0 = B0, nfp=nfp, order='r1')
                stel.calculate()
                def iota_eta(x):
                    stel.etabar = x
                    stel.order = "r1"
                    stel.calculate() # It seems unnecessary having to recompute all these quantities just because some etabar was introduced in the calculate axis routine. Used to not use it but not properly updated
                    val = -np.abs(stel.iotaN)
                    return val

                opt = optimize.minimize(iota_eta, x0 = min(stel.curvature))

                stel.etabar = opt.x
                stel.calculate()

                def B20dev(x):
                    stel.B2c = x
                    stel.order = "r3"
                    stel.calculate_r2()
                    val = stel.B20_variation
                    return val

                bnd = ((-30,30))
                opt = optimize.minimize_scalar(B20dev, bounds =bnd)
                stel.B2c = opt.x
                stel.order = "r2"
                stel.calculate_r2()

                # # display information
                # if info['Nfeval']%1 == 0:
                #     print('{0:4d}   {1: 3.9f}   {2: 3.9f}  {3: 3.9f}'.format(info['Nfeval'], z[0], z[1], stel.B20_variation))
                # info['Nfeval'] += 1
                
                return stel.B20_variation

            x0In = [0.8, 0.8]
            # print('{0:4s}   {1:9s}   {2:9s}   {3:9s}'.format('Iter', ' Zs1', ' Zs2', 'f(Zs)'))
            opt = optimize.minimize(findZOpt, x0 = (x0In[0], x0In[1]), args=({'Nfeval':0}), method = 'Nelder-Mead', options={'xatol':0.003,'fatol': 0.01, 'maxiter': 70, 'maxfev': 70})
            # Nelder-Mead L-BFGS-B
            # bounds = ((0.3,1.7),(0.3,1.7)), 

            rc=[1, aVal[i], bVal[j]]
            zs=[0.0, aVal[i]*opt.x[0],bVal[j]*opt.x[1]]
            zRela[i,j] = opt.x[0]
            zRelb[i,j] = opt.x[1]
            B0=1
            stel = Qsc(rc=rc, zs=zs, B0 = B0, nfp=nfp, order='r1')

            def iota_eta(x):
                stel.etabar = x
                stel.order = "r1"
                stel.calculate() # It seems unnecessary having to recompute all these quantities just because some etabar was introduced in the calculate axis routine. Used to not use it but not properly updated
                val = -np.abs(stel.iotaN)
                return val

            opt = optimize.minimize(iota_eta, x0 = min(stel.curvature))

            stel.etabar = opt.x
            etabar[i,j] = stel.etabar
            stel.calculate()
            iota[i,j] = stel.iota
            hel[i,j] = stel.iota - stel.iotaN

            def B20dev(x):
                stel.B2c = x
                stel.order = "r3"
                stel.calculate_r2()
                val = stel.B20_variation
                return val

            bnd = ((-30,30))
            opt = optimize.minimize_scalar(B20dev, bounds =bnd)
            stel.B2c = opt.x
            stel.order = "r2"
            stel.calculate_r2()

            stel.calculate_shear()

            B22c[i,j] = stel.B2c
            shear[i,j] = stel.iota2
            delB20[i,j] = stel.B20_variation
            avB20[i,j] = stel.B20_mean
            resB20[i,j] = stel.B20_residual
        except:
            zRela[i,j] = 'NaN'
            zRelb[i,j] = 'NaN'
            B22c[i,j] = 'NaN'
            shear[i,j] = 'NaN'
            delB20[i,j] = 'NaN'
            avB20[i,j] = 'NaN'
            resB20[i,j] = 'NaN'
            etabar[i,j] = 'NaN'
            iota[i,j] = 'NaN'
            hel[i,j] = 'NaN'
            
        print(j, dirFolder)   
    print(i, dirFolder)     

np.savetxt(dirFolder + 'zRela.out', zRela)   
np.savetxt(dirFolder + 'zRelb.out', zRelb)  
np.savetxt(dirFolder + 'etabar.out', etabar)   
np.savetxt(dirFolder + 'iota.out', iota)   
np.savetxt(dirFolder + 'hel.out', hel)   
np.savetxt(dirFolder + 'B22c.out', B22c)   
np.savetxt(dirFolder + 'shear.out', shear)   
np.savetxt(dirFolder + 'delB20.out', delB20)   
np.savetxt(dirFolder + 'avB20.out', avB20)   
np.savetxt(dirFolder + 'resB20.out', resB20)   
