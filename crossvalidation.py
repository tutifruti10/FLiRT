import numpy as np
from Lgr3 import LGR3
from Options import Option
import h5py
import sys


#usage: python crossvalidation.py <input file path> <n-fold of cross validation> <inverse(bool)>
in_file_path=sys.argv[1]
n_cross_val=int(sys.argv[2])
inverse=sys.argv[3]

if inverse == 'True':
    inverse = True
elif inverse == 'False':
    inverse = False
else:
    raise ValueError

f=h5py.File(in_file_path)
g=f.get('MSSMEW')

l=g.get('LHC_Combined_LogLike_allSR')
unc=g.get('LHC_Combined_LogLike_allSR_uncertainty')
M1 = g.get('M1')
M2=g.get('M2')
Mu=g.get('mu')
TanB=g.get('TanBeta')

X=np.vstack((M1,M2,Mu,TanB)).T
y=np.atleast_2d(l).T
un=np.atleast_2d(unc).T

fracerr=[]

for i in range(0,n_cross_val):
    if inverse:
        Xtrain=X[i::n_cross_val]
        print(Xtrain.shape)
        ytrain=y[i::n_cross_val]
        unctrain=un[i::n_cross_val]
        delrang = np.arange(i,X.shape[0],n_cross_val)
        Xpred=np.delete(X, delrang, axis=0)
        ytrue=np.delete(y, delrang, axis=0)
        
    else:
        Xpred=X[i::n_cross_val]
        ytrue=y[i::n_cross_val]
        delrang = np.arange(i,X.shape[0],n_cross_val)
        Xtrain=np.delete(X,delrang,axis=0)
        print(Xtrain.shape)
        ytrain=np.delete(y,delrang,axis=0)
        print(ytrain.shape)
        unctrain=np.delete(un,delrang,axis=0)
        print(unctrain.shape)
        
    opt=Option()
    model=LGR3(opt,4,100)
    
    model.initialize_lm(Xtrain,ytrain,unctrain)
    ypred,sig=model.predict4(Xpred)
    ypred=ypred[:,np.newaxis]
    sig=sig[:,np.newaxis]
    
    err=100*(np.abs(ypred - ytrue)/ypred)
    avgerr=np.mean(err)
    fracerr.append(avgerr)
    prec= in_file_path[:-5] + '_crossvaln' + str(n_cross_val) + '_run' + str(i) 
    
    thresh=(1.96*sig)/np.abs(ypred)
    kept=thresh < 0.3
    So
    centres=[]
    for m in range(0,model.M):
        centres.append(model.lmodels[m].center)
        
    
    np.save(prec + '_centres',centres)
    np.save(prec + '_Xpred',Xpred)
    np.save(prec + '_Xtrain',Xtrain)
    np.save(prec + '_ytrain',ytrain)
    np.save(prec + '_ytrue',ytrue)
    np.save(prec + '_ypred',ypred)
    np.save(prec + '_err',err)
    np.save(prec + '_sigma',sig)
    np.save(prec + '_kept',kept)
    


        

    
    




