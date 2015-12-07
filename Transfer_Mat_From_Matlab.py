import scipy.io as sio

def txmat(name,type):
    # name is a string ending in a .mat ... e.g. 'xtrain.mat'
    # type is the name the variable is called in matlab
    # possible types are xtrain,ytrain,xtest,ytest,ybintrain,ybintest,xtruncLength (this is for the testing set),xtltrain, and xtltest
    x = sio.loadmat(name)
    x = x[type]
    return x
