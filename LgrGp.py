import numpy as np
import operator
from LM import LocalModel
import scipy
from scipy.spatial import KDTree
from joblib import Parallel,delayed

class LGR(object):

    def __init__(self, opt,dim):
        self.D = dim
        self.M = 0  # number of local models
        self.opt = opt
        self.beta=opt.beta #noise of data
        self.lmodels = []
        self.lmasks=[]
        self.cut=[]
        self.PointCheck=[]
            
    def add_local_model(self, X=None ,Y=None): 
        if(self.M + 1 < self.opt.max_num_lm):
            self.lmodels.append(LocalModel(self.opt,self.D))
            self.lmasks.append(False)
            self.lmodels[self.M].init_lm(X, Y) 
            self.M = self.M + 1
        else:
            print("maximum number of local models reached")

        return 0       
    def init_train(self,X,Y,lm):
        w=np.zeros(X.shape[0])
        for i in range(self.opt.n_train):
            x=[]
            y=[]
            w=lm.get_ww(X)
            print(self.M,lm.kernel,lm.center)
            where=w > self.opt.activ_thresh
            indices=[i for i, x in enumerate(where.T) if x]
            for index in indices:
                x.append(X[index])
                y.append(Y[index])
            if x==[] or y==[]:
                print("no points, model deleted")
                lm=None
                return lm
            x=np.asarray(x)
            y=np.asarray(y)
            lm.init_lm(x, y)
        return lm
                                
                                
    def initialize_lm(self, X,Y):
        n_data = X.shape[0]
        for n in range(0, n_data):
            xn = X[n, :]
            w = np.zeros(self.M)
            for m in range(0, self.M):
                lm = self.lmodels[m]
                w[m] = lm.get_ww(xn[np.newaxis, :])
            if w.size==0:
                max_act=0
            else:
                max_act = w.max()
            if max_act < self.opt.activ_thresh:
                self.add_local_model(xn)
                print('center added',xn)
                lm=self.lmodels[(self.M-1)]
                lm = self.init_train(X,Y,lm)
                if lm==None:
                    del self.lmodels[(self.M-1)]
                    del self.lmasks[(self.M-1)]
                    self.M-=1            
        for i in range(self.M-1,-1,-1):
            if self.prune_overlaps(i,True):
                del self.lmodels[i]
                self.M-=1
        
        #self.prune_empty()
        #for i in range(self.M-1,-1,-1):
        #    if not self.lmodels[i].trained:
        #        del self.lmodels[i]
        #        self.M-=1
        #self.Check_TrainData(X,Y)
        
    def check_overlap2(self,lm1,lm2):
        if not isinstance(lm1,LocalModel) or not isinstance(lm2,LocalModel):
            raise TypeError("Incorrect argument type; please pass two local models.")
        x=lm1.X
        y=lm2.X
        nrows,ncols=x.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols*[x.dtype]}
        C,indx,indy=np.intersect1d(x.view(dtype),y.view(dtype),return_indices=True)
        return indy
        
    def prune_overlaps(self,ind,delete=False):
        """
        Give one local model index as an argument to produce a count of its unique points, deleting the model if it has none.
        """
        lm = self.lmodels[ind]
        mask=np.zeros(lm.X.shape[0],dtype=bool)
        for l in range(self.M):        
            if self.lmodels[l]==lm:
                continue
            r=self.check_overlap2(self.lmodels[l],lm)
            mask[r]=True
        unique=np.sum(mask)
        if unique==lm.X.shape[0]:
            print("Model has no unique points; deleting")
            return True
        else:
            print("Model has ", lm.X.shape[0] - unique, " unique points")
            lm.unique=lm.X.shape[0] - unique
            return False
    def prune_empty(self):
        for j in reversed(range(self.M)):
            if self.lmodels[j].trained==False:
                del self.lmodels[j]
                self.M-=1
                print('Untrained model',j, 'del')
        return
    def Check_TrainData(self,X,y):
        X_train=[]
        for i in range(self.M):
            lm=self.lmodels[i]
            xx=lm.X[:,0]
            xt=xx.tolist()
            X_train.append(xt)
        bb=np.concatenate(X_train, axis=0)
        b=bb.tolist()
        a=X[:,0].tolist()
        dif=list(set(a) - set(b))   
        if dif!=[]:
            intd,indices,sth=np.intersect1d(X[:,0], dif, return_indices=bool)
            for index in indices:
                self.update2(np.atleast_2d(X[index,:]),np.atleast_2d(y[index]))
                print('Updating Lgr with missed points')
        else:
            print('All training data included')
            return
    
    def predict2(self, x): 
        yp = 0.0
        sig=0.0
        var1=0.0
        var2=0.0
        wtot=0.0
        for m in range(0, self.M):
            lm = self.lmodels[m]
            w =  self.lmodels[m].get_wpred(x) 
            y_pred, sigma=  self.lmodels[m].predict_(x)
            var1+=(w.reshape(y_pred.shape)*sigma.reshape(y_pred.shape)**2)+w.reshape(y_pred.shape)*y_pred**2
            var2+=w.reshape(y_pred.shape)*y_pred
            wtot+=np.array(w)
            yp += w.reshape(y_pred.shape) * y_pred
        return yp/wtot.reshape(yp.shape), np.sqrt(var1/wtot.reshape(yp.shape)-(var2/wtot.reshape(yp.shape))**2)
    def predict3(self, x): 
        yp = 0.0
        sig=0.0
        var1=0.0
        var2=0.0
        wtot=0.0
        for m in range(0, self.M):
            lm = self.lmodels[m]
            w =  self.lmodels[m].get_wk2(x) 
            y_pred, sigma=  self.lmodels[m].predict_(x)
            var1+=(w.reshape(y_pred.shape)*sigma.reshape(y_pred.shape)**2)+w.reshape(y_pred.shape)*y_pred**2
            var2+=w.reshape(y_pred.shape)*y_pred
            wtot+=np.array(w)
            yp += w.reshape(y_pred.shape) * y_pred
        return yp/wtot.reshape(yp.shape), np.sqrt(var1/wtot.reshape(yp.shape)-(var2/wtot.reshape(yp.shape))**2)
    def predict4(self, x): 
        yp = 0.0
        sig=0.0
        var1=0.0
        var2=0.0
        wtot=0.0
        for m in range(0, self.M):
            lm = self.lmodels[m]
            w =  self.lmodels[m].get_wpk2(x) 
            y_pred, sigma=  self.lmodels[m].predict_(x)
            var1+=(w.reshape(y_pred.shape)*sigma.reshape(y_pred.shape)**2)+w.reshape(y_pred.shape)*y_pred**2
            var2+=w.reshape(y_pred.shape)*y_pred
            wtot+=np.array(w)
            yp += w.reshape(y_pred.shape) * y_pred
        return yp/wtot.reshape(yp.shape), np.sqrt(var1/wtot.reshape(yp.shape)-(var2/wtot.reshape(yp.shape))**2)
    
    def predict(self, x): 
        yp = 0.0
        sig=0.0
        var1=0.0
        var2=0.0
        wtot=0.0
        for m in range(0, self.M):
            lm = self.lmodels[m]
            w =  self.lmodels[m].get_ww(x) 
            y_pred, sigma=  self.lmodels[m].predict_(x)
            var1+=(w.reshape(y_pred.shape)*sigma.reshape(y_pred.shape)**2)+w.reshape(y_pred.shape)*y_pred**2
            var2+=w.reshape(y_pred.shape)*y_pred
            wtot+=np.array(w)
            yp += w.reshape(y_pred.shape) * y_pred
        return yp/wtot.reshape(yp.shape), np.sqrt(var1/wtot.reshape(yp.shape)-(var2/wtot.reshape(yp.shape))**2)
        
    def update(self,x_new,y_new):
        n_data = x_new.shape[0]
        for n in range(0, n_data):
            xn = x_new[n, :]
            w = np.zeros(self.M)
            for m in range(0, self.M):
                lm = self.lmodels[m]
                w[m] = lm.get_ww(xn[np.newaxis, :])
                index, max_act = max(enumerate(w), key=operator.itemgetter(1))
            print(index,max_act,w)
            if max_act < self.opt.activ_thresh:
                self.add_local_model(xn)
            else: 
                self.lmodels[index].update_(x_new,y_new)
        return 
    def update2(self,x_new,y_new):
        n_data = x_new.shape[0]
        for n in range(0, n_data):
            xn = x_new[n, :]
            w = np.zeros(self.M)
            for m in range(0, self.M):
                lm = self.lmodels[m]
                w[m] = lm.get_ww(xn[np.newaxis, :])
                index, max_act = max(enumerate(w), key=operator.itemgetter(1))
            print('LM updated',index,'w_max',max_act,'data point updated',xn)
            if max_act < self.opt.activ_thresh:
                self.add_local_model(xn,y_new[n])
            else:
                where=w > self.opt.activ_thresh
                indices=[i for i, x in enumerate(where) if x]
                for index in indices:
                    self.lmodels[index].update_(x_new,y_new.reshape(1,))
        return 
