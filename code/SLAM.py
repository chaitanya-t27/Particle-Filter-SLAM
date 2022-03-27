import numpy as np
from matplotlib import pyplot as plt
from math import cos, sin, radians, pi
import os, cv2
from pr2_utils import bresenham2D
from tqdm import tqdm
from usable_functions import predict, lidar_wep, map_correlation_update, map_update, resampling

fog_file = open("fog.csv")
fog_array = np.loadtxt(fog_file,delimiter=",")

# Forming an array of time stamps in the first column against the cummulative yaw angles in the second column:
yaw_array = np.delete(fog_array,[1,2],1)

encoder_file = open("encoder.csv")
encoder_array = np.loadtxt(encoder_file, delimiter=",")

lidar_file = open("lidar.csv")
lidar_array = np.loadtxt(lidar_file,delimiter=",")

angle_file = open("matchedvehiangle.csv")
cum_angle_array = np.loadtxt(angle_file,delimiter=",")

dist_file = open("matcheddist.csv")
mat_dist_array = np.loadtxt(dist_file,delimiter=",")

# init MAP
MAP = {}
MAP['res']   = 1 #meters
MAP['xmin']  = -100  #meters
MAP['ymin']  = -1200
MAP['xmax']  =  1300    
MAP['ymax']  =  200
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float16) #DATA TYPE: char or int8


num_par = 5
weights = np.ones(num_par)/num_par
u = np.zeros((3,(num_par)))

x_plot = []
y_plot = []
for i in tqdm(range(740)):
    if i==0:
        new_par_loc,new_noise_angle = predict(i)
        traj_plot = new_par_loc[1,:]
        lid_wep = lidar_wep(i,new_par_loc,new_noise_angle)
        MAP,xmapsp,ymapsp = map_update(MAP,new_par_loc[1,:],lid_wep)
        x_plot.append(xmapsp)
        y_plot.append(ymapsp)
    if i >= 1:
        a,b = predict(i,new_par_loc,new_noise_angle)
        lid_wep = lidar_wep(i,a,b)
        weights = map_correlation_update(MAP,lid_wep,weights)
        particle_max_position = np.argmax(weights)
        particle_max_state = a[particle_max_position,:]
        lidar_rays = lid_wep[particle_max_position]
        MAP,xmapsp,ymapsp = map_update(MAP,particle_max_state,lidar_rays)
        x_plot.append(xmapsp)
        y_plot.append(ymapsp)
        new_par_loc,weights = resampling(particle_max_state)
        
x_plot = np.array(x_plot)
y_plot = np.array(y_plot)
plt.imshow(MAP['map'], cmap="gray" )
plt.title("MAP")
plt.pause(25)


