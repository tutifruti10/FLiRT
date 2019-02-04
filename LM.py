import numpy as np
from numpy import dot
from scipy.linalg import cholesky, inv, solve, cho_solve
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class LocalModel(object):
    def __init__(self, opt,D):
        self.D=D
        self.opt = opt
        self.set_initial_state()
        self.trained=False

    def set_initial_state(self):
        self.center = np.array(self.D) #figure out why this is the initial state... might be wrong
        self.lm=None #added, to save the local gp trained maybe.. who knows
        self.X=None
        self.Y=None
        self.num_data = 0
        self.eta = self.opt.init_eta
        self.kernel=self.opt.kern
        self.gp= GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.opt.n_restarts_optimizer)
        return
    
    def init_lm(self, X=None, y=None):
        if X.ndim==1:
            self.center = X
        else:
            self.center = np.mean(X,axis=0)
        gp=self.gp
        if (X is not None) and (y is not None):
            self.X=X
            self.Y=y
            dist = X - self.center
            self.gp=gp.fit(dist, y)
            self.kernel=gp.kernel_
            self.trained=True
            self.overlap=np.zeros(y.shape)
        return
    
    
    def get_ww(self, X):
        return self.kernel(self.center,X)
        
    def predict_(self, X):
        dist=X-self.center
        y_pred, sigma = self.gp.predict(dist, return_std=True)
        return y_pred, sigma
        
    def update_(self,x_n,y_new):
        if self.trained is True:
            
            dim=len(self.gp.X_train_)
            
            self.X=np.concatenate((self.X,x_n),axis=0) #save all of the data used for training
            self.Y=np.concatenate((self.Y,y_new),axis=0)
            self.center = np.mean(self.X,axis=0)
            x_new=x_n-self.center
            k_new=self.kernel(x_new,x_new)
            kk_new=self.kernel(self.gp.X_train_,x_new)
            self.gp.X_train_=self.X-self.center
            self.gp.y_train_=self.Y
            
            l=solve(self.gp.L_,kk_new) 
            l_star=np.sqrt(k_new-dot(l.T,l))
            a=np.concatenate((self.gp.L_,np.zeros(dim).reshape(dim,1)),axis=1) 
            b=np.concatenate((l.T,l_star.reshape(1,1)),axis=1)
            L_new= np.concatenate((a,b),axis=0)
            alpha_new=cho_solve((L_new, True), self.gp.y_train_)
 
            self.gp.L_=L_new        #update the relavant  variables in the GP
            self.gp.alpha_=alpha_new
            self.gp._K_inv=None
        else: 
            raise ValueError('GP not trained; please use the train function to train it')
    
    
       
