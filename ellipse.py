#ELLIPSE PLOTTING CODE

fig,ax =plt.subplots()    
ax.scatter(X[0::2,0],X[0::2,1],s=3,c=y_pred2)
plt.xlabel('M1')
plt.ylabel('M2')
ax.scatter(centers[:,0],centers[:,1],s=4,c='r')
for i in model.lmodels:
    ax.add_artist(matplotlib.patches.Ellipse((i.center[0],i.center[1]),i.kernel.k2.length_scale[0],i.kernel.k2.length_scale[1],edgecolor='r',fill=False))
    
plt.show()

