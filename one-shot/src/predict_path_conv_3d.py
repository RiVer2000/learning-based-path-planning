
import keras
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import time
import math

input_path = '../data_3/maze_10x10x10_rnd/'
# input_path = '../data_3/maze_15x15x15_rnd/'
# input_path = '../data_3/maze_20x20x20_rnd/'

model_file = '../results_3/model_10_conv_3d.hf5'
# model_file = '../results_3/model_15_conv_3d.hf5'
# model_file = '../results_3/model_20_conv_3d.hf5'

ind_test = np.array(range(95000,97000)) #the range of samples for testing

#######################################################################################################################
def reconstruct_path(y_pred,env,s,g):

	sx=s[0]+1
	sy=s[1]+1
	sz=s[2]+1
	gx=g[0]+1
	gy=g[1]+1
	gz=g[2]+1
	#find predicted path
	y_pred[env==1]=-1
	pred_map=-1*np.ones((n+2,n+2,n+2))
	pred_map[1:n+1,1:n+1,1:n+1]=y_pred
	pred_map_2=-1*np.ones((n+2,n+2,n+2))
	pred_map_2[1:n+1,1:n+1,1:n+1]=y_pred

	pred_map[sx,sy,sz]=0
	pred_map_2[gx,gy,gz]=0

	T=pow(n,2)

	px_pred=np.zeros(T,dtype='int')
	py_pred=np.zeros(T,dtype='int')
	pz_pred=np.zeros(T,dtype='int')
	px_pred[0]=sx
	py_pred[0]=sy
	pz_pred[0]=sz

	px_pred_2=np.zeros(T,dtype='int')
	py_pred_2=np.zeros(T,dtype='int')
	pz_pred_2=np.zeros(T,dtype='int')
	px_pred_2[0]=gx
	py_pred_2[0]=gy
	pz_pred_2[0]=gz

	path_found=False
	got_stuck=False
	got_stuck_2=False

	for k in range(T):
		if not got_stuck:
			if math.sqrt(pow(px_pred[k]-gx,2)+pow(py_pred[k]-gy,2)+pow(pz_pred[k]-gz,2)) <= math.sqrt(3):
				px_pred[k+1]=gx
				py_pred[k+1]=gy
				pz_pred[k+1]=gz
				px_pred=px_pred[0:k+2]
				py_pred=py_pred[0:k+2]
				pz_pred=pz_pred[0:k+2]
				path_found=True
				# print('forward path')
				break
			
			tmp=pred_map[px_pred[k]-1:px_pred[k]+2,py_pred[k]-1:py_pred[k]+2,pz_pred[k]-1:pz_pred[k]+2]		
			ii, jj, kk = np.unravel_index(tmp.argmax(), tmp.shape)
			px_pred[k+1]=px_pred[k]+ii-1
			py_pred[k+1]=py_pred[k]+jj-1
			pz_pred[k+1]=pz_pred[k]+kk-1
			pred_map[px_pred[k+1],py_pred[k+1],pz_pred[k+1]]=0          
			
			h=np.array(np.where((px_pred_2==px_pred[k+1])&(py_pred_2==py_pred[k+1])&(pz_pred_2==pz_pred[k+1]))).squeeze()
			if h.size==1:
				px_pred_2=px_pred_2[0:h]
				py_pred_2=py_pred_2[0:h]
				pz_pred_2=pz_pred_2[0:h]
				px_pred=px_pred[0:k+2]
				py_pred=py_pred[0:k+2]
				pz_pred=pz_pred[0:k+2]
				px_pred=np.append(px_pred,np.flip(px_pred_2))
				py_pred=np.append(py_pred,np.flip(py_pred_2))
				pz_pred=np.append(pz_pred,np.flip(pz_pred_2))
				path_found=True
				# print('forward cross')
				break
			
			if (px_pred[k+1]==px_pred[k]) and (py_pred[k+1]==py_pred[k]) and (pz_pred[k+1]==pz_pred[k]):
				px_pred=px_pred[0:px_pred.size-1]
				py_pred=py_pred[0:py_pred.size-1]
				pz_pred=pz_pred[0:pz_pred.size-1]
				got_stuck=True

		if not got_stuck_2:
			if math.sqrt(pow(px_pred_2[k]-sx,2)+pow(py_pred_2[k]-sy,2)+pow(pz_pred_2[k]-sz,2)) <= math.sqrt(3):
				px_pred_2[k+1]=sx
				py_pred_2[k+1]=sy
				pz_pred_2[k+1]=sz
				px_pred_2=px_pred_2[0:k+2]
				py_pred_2=py_pred_2[0:k+2]
				pz_pred_2=pz_pred_2[0:k+2]
				px_pred = np.flip(px_pred_2)
				py_pred = np.flip(py_pred_2)
				pz_pred = np.flip(pz_pred_2)
				path_found=True
				# print('backward path')
				break
			
			tmp=pred_map_2[px_pred_2[k]-1:px_pred_2[k]+2,py_pred_2[k]-1:py_pred_2[k]+2,pz_pred_2[k]-1:pz_pred_2[k]+2]
			ii, jj, kk = np.unravel_index(tmp.argmax(), tmp.shape)
			px_pred_2[k+1]=px_pred_2[k]+ii-1
			py_pred_2[k+1]=py_pred_2[k]+jj-1
			pz_pred_2[k+1]=pz_pred_2[k]+kk-1
			pred_map_2[px_pred_2[k+1],py_pred_2[k+1],pz_pred_2[k+1]]=0
			
			h=np.array(np.where((px_pred==px_pred_2[k+1])&(py_pred==py_pred_2[k+1])&(pz_pred==pz_pred_2[k+1]))).squeeze()
			if h.size==1:
				px_pred=px_pred[0:h]
				py_pred=py_pred[0:h]
				pz_pred=pz_pred[0:h]
				px_pred_2=px_pred_2[0:k+2]
				py_pred_2=py_pred_2[0:k+2]
				pz_pred_2=pz_pred_2[0:k+2]
				px_pred=np.append(px_pred,np.flip(px_pred_2))
				py_pred=np.append(py_pred,np.flip(py_pred_2))
				pz_pred=np.append(pz_pred,np.flip(pz_pred_2))
				path_found=True
				# print('backward cross')
				break
			
			if (px_pred_2[k+1]==px_pred_2[k]) and (py_pred_2[k+1]==py_pred_2[k]) and (pz_pred_2[k+1]==pz_pred_2[k]):
				px_pred_2=px_pred_2[0:px_pred_2.size-1]
				py_pred_2=py_pred_2[0:py_pred_2.size-1]
				pz_pred_2=pz_pred_2[0:pz_pred_2.size-1]
				got_stuck_2=True      

		if got_stuck and got_stuck_2:
			break

	###remove loops
	ind=np.array([])
	for k in range(px_pred.size):
		h=np.array( np.where( (px_pred==px_pred[k])&(py_pred==py_pred[k])&(pz_pred==pz_pred[k]) ) ).squeeze()
		if h.size>=2:
			ind=np.append(ind, np.array(range(h[0]+1,h[1]+1)))
	px_pred=np.delete(px_pred,ind)
	py_pred=np.delete(py_pred,ind)
	pz_pred=np.delete(pz_pred,ind)

	###remove zig-zags
	while 1:
		k=0
		ind=np.array([])
		for i in range(px_pred.size-2):
			if math.sqrt(pow(px_pred[k]-px_pred[k+2],2)+pow(py_pred[k]-py_pred[k+2],2)+pow(pz_pred[k]-pz_pred[k+2],2)) <= math.sqrt(3):
				ind=np.append(ind, k+1)
				k=k+2
			else:
				k=k+1

			if k>=px_pred.size-2:
				break
		
		if ind.size:
			px_pred=np.delete(px_pred,ind)
			py_pred=np.delete(py_pred,ind)
			pz_pred=np.delete(pz_pred,ind)
		else:
			break
	return px_pred, py_pred, pz_pred
#######################################################################################################################


print('Load data ...')
env = np.loadtxt(input_path + 'inputs.dat')
path = np.loadtxt(input_path + 'astar_path.dat', dtype='str', delimiter='\n')

print('Transform data ...')
env=env[ind_test,:]
path=path[ind_test]

m = env.shape[0]
n = int(round(pow(env.shape[1],1/2)))

env = env.reshape(m,n,n)
x=np.zeros((m,n,n,n))
s_map=np.zeros((m,n,n,n))
g_map=np.zeros((m,n,n,n))
y=np.zeros((m,n,n,n))

print('Generate start-, goal- and path-maps ...')
for k in range(m):
	for i in range(n):
		for j in range(n):
			h=int(env[k,i,j])
			if h>0:
				x[k,0:h,i,j] = 1
	p = np.fromstring(path[k], dtype=int, sep=' ')
	l = int(p.shape[0]/3)
	p = p.reshape(3, l)
	s_map[k,p[2,0]-1,p[1,0]-1,p[0,0]-1] = 1
	g_map[k,p[2,l-1]-1,p[1,l-1]-1,p[0,l-1]-1] = 1
	y[k,p[2,:]-1,p[1,:]-1,p[0,:]-1] = 1
del env, path
	
x_test=np.zeros((m,n,n,n,3))
x_test[:,:,:,:,0]=x
x_test[:,:,:,:,1]=s_map
x_test[:,:,:,:,2]=g_map
y_test=y
del x, s_map, g_map, y


print('Load model ...')
model=load_model(model_file)
model.summary()
#do prediction ones to initialize GPU
y_pred=model.predict(np.zeros((1,n,n,n,3)))

print('Predict path ...')
for i in range(m):
	print(i)
	
	### Predict path
	a = time.process_time()
	y_pred=model.predict(x_test[i,:,:,:,:].reshape(1,n,n,n,3)).squeeze()
	pred_t = time.process_time() - a
	print('Prediction time: ', pred_t)

	### Reconstruct path from the prediction
	env=x_test[i,:,:,:,0].squeeze()
	s=np.array(np.nonzero(x_test[i,:,:,:,1].squeeze()==1))
	g=np.array(np.nonzero(x_test[i,:,:,:,2].squeeze()==1))
	
	a = time.process_time()
	px,py,pz=reconstruct_path(y_pred,env,s,g)
	rec_t = time.process_time() - a
	print('Reconstruction time: ', rec_t)		

	### Plot path
	env=np.sum(env,0) #transform environment from 3D to 2D 
	t=np.array(np.nonzero(y_test[i,:,:,:].squeeze()==1)) #ground-truth path
	t_pred=np.max(y_pred,0) #predicted path
	
	fig=plt.matshow(env.T,cmap='binary')
	plt.plot(t[1],t[2],'kx',markersize=15)	
	for j in range(n):
		for k in range(n):
			if t_pred[j,k]>0.001:
				plt.plot(j,k,'b.',markersize=15*t_pred[j,k]+1)
	plt.plot(py-1,pz-1,'b')
	plt.plot(s[1],s[2],'g.',markersize=15)
	plt.plot(g[1],g[2],'r.',markersize=15)
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)	
	plt.show()		
	# plt.savefig('../results_3/images/env%05d.png' % (i+1))
	plt.close()
	
	
	
	