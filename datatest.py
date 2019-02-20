import h5py 
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import LgrGp
from LgrGp import LGR
import LM
from LM import LocalModel
import Options
from Options import Option

from collections import Counter


f=h5py.File('MSSMEW.hdf5') 



list(f.keys())

group2 = f.get('MSSMEW/MSSM/"#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::M1')
g = f.get('MSSMEW')

#combined liklehood from each study 
l1 = g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_3b_24invfb__combined_LogLike')
l2 = g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_3b_36invfb__combined_LogLike')
l3 = g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_3b_discoverySR_24invfb__combined_LogLike')
l4= g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_3b_discoverySR_36invfb__combined_LogLike')
l5= g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_4LEP_36invfb__combined_LogLike')
l6 = g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_MultiLEP_2Lep0Jets_36invfb__combined_LogLike')
l7 = g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_MultiLEP_2LepPlusJets_36invfb__combined_LogLike')
l8 = g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_MultiLEP_3Lep_36invfb__combined_LogLike')
l9= g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_RJ3L_2Lep2Jets_36invfb__combined_LogLike')
l10= g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::ATLAS_13TeV_RJ3L_3Lep_36invfb__combined_LogLike')
l11= g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::CMS_13TeV_1LEPbb_36invfb__combined_LogLike')
l12= g.get('#LHC_LogLike_per_SR @ColliderBit::get_LHC_LogLike_per_SR::CMS_13TeV_2LEPsoft_36invfb__combined_LogLike')

#MSSMEW parameters:
M2 = g.get('#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::M2')
M1 = g.get('#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::M1')
MU = g.get('#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::mu')
TANB = g.get('#MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::TanBeta')

X=np.vstack((M1,M2,MU,TANB)).T

mus=Counter(MU).most_common(1)

m1=[]
m2=[]
tanb=[]
l=[]

for i in range(len(MU)):
	if MU[i]==mus[list(mus)[0]]:
		m1.append(M1[i])
		m2.append(M2[i])
		tanb.append(TANB[i])
		l.append(l1[i])

X=np.vstack((m1,m2,tanb))
t=Counter(tanb).most_common(1)
tan=t[list(t)[0]]
opt=Option()
model=LGR(opt,3)
model.initialize_lm(X,l)

step1=0.5
step2=0.5
Xpred=np.mgrid([-2000:2000:step1,0:1000:step2]).reshape(2,int(4000/step1)*int(1000/step2))
Xtanb=np.zeros(4000*1000/(step1*step2))
Xtanb[:]=tan
Xn=np.vstack((Xpred,Xtanb))

ypred,sigma=model.predict(Xn)