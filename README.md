# LGR-Local Model 
Status:
- able to initialize the a local model with x,y training data
- possibility to make predictions
- can increment by calling lm.update_(x_new,y_new) where x_new and y_new consist of one data point

Example of initalizig a local model and then incrementing it: 
opt=Options()
def f(x):
    return x*np.sin(x)

X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel()
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
x_new = np.atleast_2d([9.5.]).T
y_new = f(x_new).ravel()

model=LGR(opt,1)
model.initialize_lm(X)
model.train(X,y)
y_pred,sigma=model.predict(x)
sigma=sigma.reshape(1000,1)
y_pred=y_pred.reshape(1000,1) 


model.update(x_new,y_new)
y_pred2,sigma2=model.predict(x)
model.update(x_new,y_new)
sigma2=sigma2.reshape(1000,1)
y_pred2=y_pred2.reshape(1000,1) 

PLOTTING:
plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x_new, y_new, 'y.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
          np.concatenate([y_pred - 1.9600 * sigma,
                         (y_pred + 1.9600 * sigma)[::-1]]),
          alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.plot(x, y_pred2, 'g-', label=u'PredictionInc')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred2 - 1.9600 * sigma2,
 (y_pred2 + 1.9600 * sigma2)[::-1]]),
 alpha=.5, fc='g', ec='None', label='95% confidence interval')
 plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
(-10, 20)
plt.legend(loc='upper left') 
plt.show()

