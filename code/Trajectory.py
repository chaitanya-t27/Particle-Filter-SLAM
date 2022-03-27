import math
import numpy as np
from matplotlib import pyplot as plt
from math import cos, sin, radians, pi
import os, cv2


encoder_file = open("encoder.csv")
encoder_array = np.loadtxt(encoder_file, delimiter=",")

fog_file = open("fog.csv")
fog_array = np.loadtxt(fog_file,delimiter=",")


# Forming an array of time stamps in the first column against the cummulative yaw angles in the second column:
yaw_array = np.delete(fog_array,[1,2],1)
yaw_array[:,1] = np.cumsum(yaw_array[:,1])


# Matching timestamps of the yaw angles and the timestamps of the encoder to obtain synchronous data: 
total_angles = []
jump=0
for k in range(0,len(encoder_array)):
    ct = 0
    for i in range(25):
        ct = i
        if yaw_array[jump+k+i,0]>=encoder_array[k,0]:
            break
    jump += ct
    total_angles.append(yaw_array[jump+k][1])
total_angles = np.array(total_angles)

# Forming a distance array of travel of the car at each point corresponding to the timestamp
dist_array = np.zeros((len(encoder_array),1))

enc_diff_array = np.zeros((len(encoder_array),2))

enc_diff_array[1:len(encoder_array),0] = np.diff(encoder_array[:,1])
enc_diff_array[1:len(encoder_array),1] = np.diff(encoder_array[:,2])


dist_array = np.zeros((len(encoder_array),1))

dist_array = ((enc_diff_array[:,0]*np.pi*0.623479/4096) + (enc_diff_array[:,1]*np.pi*0.622806/4096))/2


coord_array = np.zeros((len(encoder_array),3))

for i in range(1,len(coord_array)):
    coord_array[i,0] = coord_array[i-1,0] + (dist_array[i]*np.cos(total_angles[i])) 
    coord_array[i,1] = coord_array[i-1,1] + (dist_array[i]*np.sin(total_angles[i]))

coord_array[:,0] = coord_array[:,0] - 0.335
coord_array[:,1] = coord_array[:,1] - 0.035
coord_array[:,2] = coord_array[:,2] + 0.78
plt.plot(coord_array[:,0],coord_array[:,1])
plt.show()
np.savetxt('vehiCord.csv',coord_array,delimiter=',')
np.savetxt('vehiangle.csv',total_angles,delimiter=',')