import torch.autograd as autograd
import torch
from config import *
import numpy as np
import matplotlib.pyplot as plt


def Variable(data, *args, **kwargs):
    if USE_CUDA:
        data = data.cuda()
    return autograd.Variable(data,*args, **kwargs)

def unit_prefix(x, n=1):
    for i in range(n): x = x.unsqueeze(0)
    return x

def align(x, y, start_dim=0):
    xd, yd = x.dim(), y.dim()
    if xd > yd: y = unit_prefix(y, xd - yd)
    elif yd > xd: x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if   ys[td]==1: ys[td] = xs[td]
        elif xs[td]==1: xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)

def matmul(X,Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i],Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)



def xrecons_grid(X,B,A):
	"""
	plots canvas for single time step
	X is x_recons, (batch_size x img_size)
	assumes features = BxA images
	batch is assumed to be a square number
	"""
	padsize=1
	padval=.5
	ph=B+2*padsize
	pw=A+2*padsize
	batch_size=X.shape[0]
	N=int(np.sqrt(batch_size))
	X=X.reshape((N,N,B,A))
	img=np.ones((N*ph,N*pw))*padval
	for i in range(N):
		for j in range(N):
			startr=i*ph+padsize
			endr=startr+B
			startc=j*pw+padsize
			endc=startc+A
			img[startr:endr,startc:endc]=X[i,j,:,:]
	return img

def save_image(x,count=0):
    for t in range(T):
        img = xrecons_grid(x[t],B,A)
        plt.matshow(img, cmap=plt.cm.gray)
        imgname = 'image/count_%d_%s_%d.png' % (count,'test', t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
        plt.savefig(imgname)
        print(imgname)