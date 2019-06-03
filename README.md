# LGR-Local Model 
Status:
- able to build a global model consisting of multiple local Gaussian processes, given an input training data set
- predictions can be made globally with two different weighting schemes
- can incrementally update training data

Dependencies: numpy, scipy, sci-kit learn, joblib (does some local multiprocessing, with some minor modification can be foregone)

Key modules: 
 - Lgr.py: contains overall model structure, including methods for global model building, incremental updating and global prediction.
           There are functions for the different methods of prediction, including the centre and the closest training point weighting              schemes. We prefer the closest training point method for accuracy but it still has computation time issues. Both are included            to allow for comparisons. 
           
 - LM.py: contains local model structure, for building of each local Gaussian process. Includes standard GP training procedures from               sci-kit learn, methods for calculating the weights in both weighting schemes. For maximally weighting near model centres or             the closest training point use get_wwP and get_wpk2 respectively (details about how this actually works in the report). 
 
 - Options.py: small class for structuring key parameters for the global model. 
 
 
For a set of training data with training inputs Xtrain and training outputs Ytrain, and prediction inputs Xpred, to train a model & make basic predictions:



from Lgr import LGR3
from Options import Option

opt=Options(constant, lengthscale, n_train, max_overlap) i.e. specify kernel parameter initial guesses constant and lengthscale (uses Matern kernel)

model=LGR3(opt, dim, n) - opt is the desired options, dim is the dimensionality of the training input space, n is the number of points to take as an initial training cluster

model.initialize_lm(Xtrain,Ytrain, noise) - can include noise on training data, defaults to None
ypred,sigma = model.predict(Xpred) to predict using centre weighting
ypred,sigma = model.predict4(Xpred) to predict using closest point weighting

ypred is predicted values at the prediction points Xpred, sigma is the GP variances at those points

