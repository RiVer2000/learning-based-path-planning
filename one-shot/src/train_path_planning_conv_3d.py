
import keras
from keras.layers import Input, Conv3D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

input_path = '../data_3/maze_10x10x10_rnd/'
# input_path = '../data_3/maze_15x15x15_rnd/'
# input_path = '../data_3/maze_20x20x20_rnd/'

print('Load data ...')
env = np.loadtxt(input_path + 'inputs.dat')
path = np.loadtxt(input_path + 'astar_path.dat', dtype='str', delimiter='\n')

m = 97000 #there are 100.000 samples in the data set but we will use only 97.000
env = env[0:m,:]
path = path[0:m]
n = int(round(pow(env.shape[1],1/2)))

### Data split
n_train = 95000 #number of training samples; first n_train samples in the data set are used for training 
n_test = m - n_train #the last n_test samples are used for testing

print('Transform data ...')
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
	
x4d=np.zeros((m,n,n,n,3))
x4d[:,:,:,:,0]=x
x4d[:,:,:,:,1]=s_map
x4d[:,:,:,:,2]=g_map
del x, s_map, g_map

x_train = x4d[0:n_train,:,:,:,:]
y_train = y[0:n_train,:,:,:]
x_test = x4d[n_train:m,:,:,:,:]
y_test = y[n_train:m,:,:,:]
del x4d, y


x = Input(shape=(None, None, None, 3))

net = Conv3D(filters=1024, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(x)
net = BatchNormalization()(net)

net = Conv3D(filters=512, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
net = BatchNormalization()(net)

net = Conv3D(filters=256, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
net = BatchNormalization()(net)

net = Conv3D(filters=128, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
net = BatchNormalization()(net)
for i in range(16):
	net = Conv3D(filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
	net = BatchNormalization()(net)
	
net = Conv3D(filters=1, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", kernel_initializer='orthogonal', activation='sigmoid')(net)
net = BatchNormalization()(net)	
net = Dropout(0.10)(net)

model = Model(inputs=x,outputs=net)
model.summary()

early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
save_weights = ModelCheckpoint(filepath='weights_3d.hf5', monitor='val_acc',verbose=1, save_best_only=True)

print('Train network ...')
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(x_train.reshape(n_train,n,n,n,3), y_train.reshape(n_train,n,n,n,1), batch_size=16, validation_split=5/95, epochs=1000, verbose=1, callbacks=[early_stop, save_weights])

print('Save trained model ...')
model.load_weights('weights_3d.hf5')
model.save("model_3d.hf5")

print('Test network ...')
model=load_model("model_3d.hf5")
score = model.evaluate(x_test.reshape(n_test,n,n,n,3), y_test.reshape(n_test,n,n,n,1), verbose=1)
print('test_acc:', score[1])


