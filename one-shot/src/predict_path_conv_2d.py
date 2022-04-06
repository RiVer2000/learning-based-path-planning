
import keras
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import time
import math

input_path = '../data_2/maze_10x10_rnd/'
# input_path = '../data_2/maze_15x15_rnd/'
# input_path = '../data_2/maze_20x20_rnd/'
# input_path = '../data_2/maze_30x30_rnd/'

model_file = '../results_2/model_10_conv.hf5'
# model_file = '../results_2/model_15_conv.hf5'
# model_file = '../results_2/model_20_conv.hf5'
# model_file = '../results_2/model_30_conv.hf5'

ind_test = np.array(range(28000,30000)) #the range of samples for testing (the last 2000 samples are used)

#######################################################################################################################
def reconstruct_path(y_pred,env,s,g):

	sx=s[0]+1
	sy=s[1]+1
	gx=g[0]+1
	gy=g[1]+1
	#find predicted path
	y_pred[env==1]=-1
	pred_map=-1*np.ones((n+2,n+2))
	pred_map[1:n+1,1:n+1]=y_pred
	pred_map_2=-1*np.ones((n+2,n+2))
	pred_map_2[1:n+1,1:n+1]=y_pred

	pred_map[sx,sy]=0
	pred_map_2[gx,gy]=0

	T=pow(n,2)

	px_pred=np.zeros(T,dtype='int')
	py_pred=np.zeros(T,dtype='int')
	px_pred[0]=sx
	py_pred[0]=sy

	px_pred_2=np.zeros(T,dtype='int')
	py_pred_2=np.zeros(T,dtype='int')
	px_pred_2[0]=gx
	py_pred_2[0]=gy

	path_found=False
	got_stuck=False
	got_stuck_2=False

	for k in range(T):
		if not got_stuck:
			if math.sqrt(pow(px_pred[k]-gx,2)+pow(py_pred[k]-gy,2)) <= math.sqrt(2):
				px_pred[k+1]=gx
				py_pred[k+1]=gy
				px_pred=px_pred[0:k+2]
				py_pred=py_pred[0:k+2]
				path_found=True
				# print('forward path')
				break
			
			tmp=pred_map[px_pred[k]-1:px_pred[k]+2,py_pred[k]-1:py_pred[k]+2]		
			ii, jj = np.unravel_index(tmp.argmax(), tmp.shape)
			px_pred[k+1]=px_pred[k]+ii-1
			py_pred[k+1]=py_pred[k]+jj-1
			pred_map[px_pred[k+1],py_pred[k+1]]=0          
			
			h=np.array(np.where((px_pred_2==px_pred[k+1])&(py_pred_2==py_pred[k+1]))).squeeze()
			if h.size==1:
				px_pred_2=px_pred_2[0:h]
				py_pred_2=py_pred_2[0:h]
				px_pred=px_pred[0:k+2]
				py_pred=py_pred[0:k+2]
				px_pred=np.append(px_pred,np.flip(px_pred_2))
				py_pred=np.append(py_pred,np.flip(py_pred_2))
				path_found=True
				# print('forward cross')
				break
			
			if (px_pred[k+1]==px_pred[k]) and (py_pred[k+1]==py_pred[k]):
				px_pred=px_pred[0:px_pred.size-1]
				py_pred=py_pred[0:py_pred.size-1]
				got_stuck=True

		if not got_stuck_2:
			if math.sqrt(pow(px_pred_2[k]-sx,2)+pow(py_pred_2[k]-sy,2)) <= math.sqrt(2):
				px_pred_2[k+1]=sx
				py_pred_2[k+1]=sy
				px_pred_2=px_pred_2[0:k+2]
				py_pred_2=py_pred_2[0:k+2]
				px_pred = np.flip(px_pred_2)
				py_pred = np.flip(py_pred_2)
				path_found=True
				# print('backward path')
				break
			
			tmp=pred_map_2[px_pred_2[k]-1:px_pred_2[k]+2,py_pred_2[k]-1:py_pred_2[k]+2]
			ii, jj = np.unravel_index(tmp.argmax(), tmp.shape)
			px_pred_2[k+1]=px_pred_2[k]+ii-1
			py_pred_2[k+1]=py_pred_2[k]+jj-1
			pred_map_2[px_pred_2[k+1],py_pred_2[k+1]]=0
			
			h=np.array(np.where((px_pred==px_pred_2[k+1])&(py_pred==py_pred_2[k+1]))).squeeze()
			if h.size==1:
				px_pred=px_pred[0:h]
				py_pred=py_pred[0:h]
				px_pred_2=px_pred_2[0:k+2]
				py_pred_2=py_pred_2[0:k+2]
				px_pred=np.append(px_pred,np.flip(px_pred_2))
				py_pred=np.append(py_pred,np.flip(py_pred_2))
				path_found=True
				# print('backward cross')
				break
			
			if (px_pred_2[k+1]==px_pred_2[k]) and (py_pred_2[k+1]==py_pred_2[k]):
				px_pred_2=px_pred_2[0:px_pred_2.size-1]
				py_pred_2=py_pred_2[0:py_pred_2.size-1]
				got_stuck_2=True      

		if got_stuck and got_stuck_2:
			break

	###remove loops
	ind=np.array([])
	for k in range(px_pred.size):
		h=np.array( np.where( (px_pred==px_pred[k])&(py_pred==py_pred[k]) ) ).squeeze()
		if h.size>=2:
			ind=np.append(ind, np.array(range(h[0]+1,h[1]+1)))
	px_pred=np.delete(px_pred,ind)
	py_pred=np.delete(py_pred,ind)

	###remove zig-zags
	while 1:
		k=0
		ind=np.array([])
		for i in range(px_pred.size-2):
			if math.sqrt(pow(px_pred[k]-px_pred[k+2],2)+pow(py_pred[k]-py_pred[k+2],2)) <= math.sqrt(2):
				ind=np.append(ind, k+1)
				k=k+2
			else:
				k=k+1

			if k>=px_pred.size-2:
				break
		
		if ind.size:
			px_pred=np.delete(px_pred,ind)
			py_pred=np.delete(py_pred,ind)
		else:
			break
	return px_pred, py_pred
#######################################################################################################################


print('Load data ...')
x = np.loadtxt(input_path + 'inputs.dat')
s_map = np.loadtxt(input_path + 's_maps.dat')
g_map = np.loadtxt(input_path + 'g_maps.dat')
y = np.loadtxt(input_path + 'outputs.dat')

print('Transform data ...')
x=x[ind_test,:]
s_map=s_map[ind_test,:]
g_map=g_map[ind_test,:]
y=y[ind_test,:]

m = x.shape[0]
n = int(np.sqrt(x.shape[1]))

x = x.reshape(m,n,n)
s_map = s_map.reshape(m,n,n)
g_map = g_map.reshape(m,n,n)

x_test=np.zeros((m,n,n,3))
x_test[:,:,:,0]=x
x_test[:,:,:,1]=s_map
x_test[:,:,:,2]=g_map
del x, s_map, g_map

y_test = y.reshape(m,n,n)
del y

print('Load model ...')
model=load_model(model_file)
model.summary()
#do prediction ones to initialize GPU
model.predict(np.zeros((1,n,n,3)))

print('Predict path ...')
for i in range(m):
	print(i)
	
	### Predict path
	a = time.process_time()
	y_pred=model.predict(x_test[i,:,:,:].reshape(1,n,n,3)).squeeze()
	pred_t = time.process_time() - a
	print('Prediction time: ', pred_t)
	
	### Reconstruct path from the prediction
	env=x_test[i,:,:,0].squeeze()
	s=np.array(np.nonzero(x_test[i,:,:,1].squeeze()==1))
	g=np.array(np.nonzero(x_test[i,:,:,2].squeeze()==1))
	
	a = time.process_time()
	px,py=reconstruct_path(y_pred,env,s,g)
	rec_t = time.process_time() - a
	print('Reconstruction time: ', rec_t)
		
	### Plot path 
	t=np.array(np.nonzero(y_test[i,:,:].squeeze()==1)) #ground-truth path
	t_pred=np.array(np.nonzero(y_pred>0.001)) #predicted path
	
	fig = plt.matshow(env.T,cmap='binary')
	plt.plot(t[0],t[1],'kx',markersize=15)
	for j in range(t_pred.shape[1]):
		plt.plot(t_pred[0,j],t_pred[1,j],'b.',markersize=15*y_pred[t_pred[0,j],t_pred[1,j]]+1)
	plt.plot(px-1,py-1,'b')
	plt.plot(s[0],s[1],'g.',markersize=15)
	plt.plot(g[0],g[1],'r.',markersize=15)
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.show()
	# plt.savefig('../results_2/images/env%05d.png' % (i+1))	
	plt.close()
	
	




