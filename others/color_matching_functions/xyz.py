from matplotlib import pyplot as plt
import numpy as np
import os

f = r'/home/devici/github/color_interferometry/color_matching_functions'
os.chdir(f)

data_xyz = np.loadtxt('xyz.txt')

l = np.shape(data_xyz)[0]
m = np.shape(data_xyz)[1]

wave = data_xyz[:,0]

# print(data_xyz[0,1:4])
# input()

data_rgb = np.zeros((l,m))

data_rgb[:,0] = wave

conv = np.matrix([[0.49, 0.31, 0.20],[0.17697, 0.81240, 0.01063],[0.00, 0.01, 0.99]])
conv_T = conv**-1

print(np.linalg.det(conv))
print(np.linalg.det(conv_T))
input()

# print(conv)
# print(conv_T)
# input()

for i in np.arange(0,l,1):
    data_rgb[i,1] = conv_T[0,0]*data_xyz[i,1] + conv_T[0,1]*data_xyz[i,2] + conv_T[0,2]*data_xyz[i,3]
    data_rgb[i,2] = conv_T[1,0]*data_xyz[i,1] + conv_T[1,1]*data_xyz[i,2] + conv_T[1,2]*data_xyz[i,3]
    data_rgb[i,3] = conv_T[2,0]*data_xyz[i,1] + conv_T[2,1]*data_xyz[i,2] + conv_T[2,2]*data_xyz[i,3]

plt.figure(0)

plt.plot(data_xyz[:,0],data_xyz[:,1])
plt.scatter(data_xyz[:,0],data_xyz[:,1])

plt.plot(data_xyz[:,0],data_xyz[:,2])
plt.scatter(data_xyz[:,0],data_xyz[:,2])

plt.plot(data_xyz[:,0],data_xyz[:,3])
plt.scatter(data_xyz[:,0],data_xyz[:,3])

plt.figure(1)

plt.plot(data_rgb[:,0],data_rgb[:,1],c='red')
plt.scatter(data_rgb[:,0],data_rgb[:,1],c='red')

plt.plot(data_rgb[:,0],data_rgb[:,2],c='green')
plt.scatter(data_rgb[:,0],data_rgb[:,2],c='green')

plt.plot(data_rgb[:,0],data_rgb[:,3],c='blue')
plt.scatter(data_rgb[:,0],data_rgb[:,3],c='blue')

plt.show()
