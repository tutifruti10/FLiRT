# LGR-Local Model 
Status:
- able to initialize the a local model with x,y training data
- possibility to make predictions
- can increment by calling lm.update_(x_new,y_new) where x_new and y_new consist of one data point

Example of initalizig a local model and then incrementing it: 
Options=opt(1)
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
def f(x):
    return x*np.sin(x)
y = f(X).ravel()
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
x_new = np.atleast_2d([9.]).T
y_new = f(x_new).ravel()

lm=LocalModel(opt,1)
lm.init_lm(center=0.0,X,y)
y_pred1,sigma1=lm.predict_(x)
lm.update_(x_new,y_new)
y_pred2,sigma2=lm.predict_(x)
