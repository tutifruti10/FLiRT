import LgrGp
from LgrGp import LGR
from Options import Option
import numpy as np
import matplotlib.pyplot as plt
import h5py

f=h5py.File('MSSMEW.hdf5')

list(f.keys())
g = f.get('MSSMEW')


l1 = g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_3b_24invfb__combined_LogLike')
M2 = g.get('#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::M2')
M1 = g.get('#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::M1')
MU = g.get('#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::mu')
TANB = g.get('#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::TanBeta')

l1l=np.asarray(l1)
bins=np.vstack((M1,M2))

Xulim=2000
Xblim=-2000
Xr=Xulim-Xblim

Yulim=1000
Yblim=0
Yr=Yulim-Yblim

Nbins=[500,500]
l=np.zeros(Nbins)
for i in range(Nbins[0]):
	for j in range(Nbins[1]):
		X1=np.where(np.logical_and(bins[0]>Xblim+i*Xr/Nbins[0],bins[0]<Xblim+(i+1)*Xr/Nbins[0]))[0]
		X2=np.where(np.logical_and(bins[1]>Yblim+j*Yr/Nbins[1],bins[1]<Yblim+(j+1)*Yr/Nbins[1]))[0]
		r=np.intersect1d(X1,X2)
		if r.size==0:
			l[i,j]=np.nan
		else:
			l[i,j]=l1l[r].max()

plt.imshow(l,interpolation='nearest')
plt.show()