from sklearn.gaussian_process.kernels import RBF, Matern,ConstantKernel as C
class Option(object):
    '''parameter settings for LGR'''

    def __init__(self,const=1.0,length=0.3,n_train=3,max_overlap=3):
        self.const=const
        self.length=length
        self.max_iter = 100
        self.activ_thresh = 0.00005
        self.init_eta = 0.0001
        self.max_num_lm = 1000
        self.beta = 1e-6
        self.do_pruning = True
        self.n_restarts_optimizer=9
        self.n_train=n_train
        self.min_train=2
        self.kern=C(self.const, (0.001, 1000)) * Matern(self.length, (1e-2, 2))
        self.max_overlap=max_overlap