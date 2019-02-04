import Options
from Options import Options

import LM
from LM import LocalModel

import LgrGp
from LgrGp import LGR

import importlib
import numpy as np
from matplotlib import pyplot as plt



opt=Options()
def f(x):
    return np.sin(x)

def g(x):
    return 1/x

c=g(10)-f(10)


N=10
X = np.atleast_2d([np.random.uniform(-30,30,N)]).T
y=f(X)



model=LGR(opt,1)
model.initialize_lm(X,y)
x=np.atleast_2d([np.linspace(-30,30,1000)]).T
print(x.shape)
y_pred,sigma=model.predict(x)

centres=np.zeros(model.M)
for i in range(model.M):
	centres[i]=model.lmodels[i].center
ycentres=np.zeros(model.M)

sigma=sigma.reshape(1000,1)
y_pred=y_pred.reshape(1000,1) 

plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.plot(centres,ycentres,'g.',markersize=10,label=u'Model centres')
plt.fill(np.concatenate([x, x[::-1]]),
          np.concatenate([y_pred - 1.9600 * sigma, 
	(y_pred + 1.9600 * sigma)[::-1]]),
          alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
(-10, 20)
plt.legend(loc='upper left') 
plt.show()