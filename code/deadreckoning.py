import numpy as np
from matplotlib import pyplot as plt
from math import cos, sin, radians, pi
import os, cv2
from pr2_utils import bresenham2D
from tqdm import tqdm


encoder_file = open("encoder.csv")
encoder_array = np.loadtxt(encoder_file, delimiter=",")

angle_file = open("vehiangle.csv")
vehiangle = np.loadtxt(angle_file,delimiter=",")

lidar_file = open("lidar.csv")
lidar_array = np.loadtxt(lidar_file,delimiter=",")

coord_file = open("vehiCord.csv")
vehi_coord_array = np.loadtxt(coord_file,delimiter=",")

# Body2Lidar
RPY_lidar = [142.759, 0.0584636, 89.9254]
R_lidar = [0.00130201, 0.796097, 0.605167, 0.999999, -0.000419027, -0.00160026, -0.00102038, 0.605169, -0.796097]
T_lidar = [0.8349, -0.0126869, 1.76416]

transform_lidar2body = np.array([[R_lidar[0], R_lidar[1], R_lidar[2], T_lidar[0]],
                                 [R_lidar[3], R_lidar[4], R_lidar[5], T_lidar[1]],
                                 [R_lidar[6], R_lidar[7], R_lidar[8], T_lidar[2]],
                                 [0, 0, 0, 1]])


valididx = []
k = 0

for i in range(len(lidar_array)):
    value = lidar_array[i,0]
    array = np.asarray(encoder_array[k:k+10,0])
    idx = (np.abs(array - value)).argmin()
    valididx.append(idx+k)
    k += idx

valididx = np.array(valididx)

vehiangle = vehiangle[valididx]

vehi_coord_array = vehi_coord_array[valididx,:]


lid_wep = []
for i in range(len(lidar_array)):    
    lid_rays = lidar_array[i,1:287]
    lid_ang = np.linspace(-5, 185, 286) / 180 * np.pi
    indValid = np.logical_and((lid_rays < 60),(lid_rays> 2))
    lid_rays = lid_rays[indValid]
    lid_ang = lid_ang[indValid]

    lid_ep = np.zeros((len(lid_ang),4))
    lid_ep[:,0] = lid_rays * np.cos(lid_ang)
    lid_ep[:,1] = lid_rays * np.sin(lid_ang)
    lid_ep[:,3] = 1

    lid_ep = np.transpose(lid_ep)

    lid_veh = np.zeros((4,len(lid_rays)))
    lid_veh = np.matmul(transform_lidar2body,lid_ep)

    transform_body2world = np.array([[np.cos(vehiangle[i]),-np.sin(vehiangle[i]),0,vehi_coord_array[i,0]],
                                 [np.sin(vehiangle[i]),np.cos(vehiangle[i]),0,vehi_coord_array[i,1]],
                                 [0,0,1,vehi_coord_array[i,2]],                                                   
                                 [0, 0, 0, 1]])
    
    lid_world_ep = np.zeros((4,len(lid_rays)))
    lid_world_ep = np.matmul(transform_body2world,lid_veh)

    lid_world_ep = np.delete(lid_world_ep,3,0)
    lid_world_ep = np.transpose(lid_world_ep)
    lid_wep.append(lid_world_ep)


# init MAP
MAP = {}
MAP['res']   = 1 #meters
MAP['xmin']  = -100  #meters
MAP['ymin']  = -1200
MAP['xmax']  =  1300    
MAP['ymax']  =  200
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int32) #DATA TYPE: char or int8

# Finding start point in MAP
x_mapsp = np.ceil((vehi_coord_array[:,0]-MAP['xmin'])/MAP['res']).astype(np.int16)-1
y_mapsp = np.ceil((vehi_coord_array[:,1]-MAP['ymin'])/MAP['res']).astype(np.int16)-1

#Finding end point in MAP
x_mapep = []
y_mapep = []

for i in range(len(lidar_array)):
    x_mapep.append(np.ceil((lid_wep[i][:,0]-MAP['xmin'])/MAP['res']).astype(np.int16)-1)
    y_mapep.append(np.ceil((lid_wep[i][:,1]-MAP['ymin'])/MAP['res']).astype(np.int16)-1)

for i in tqdm(range(len(lidar_array))):
    for j in range(len(x_mapep[i])):
        bresenham_xy = bresenham2D(x_mapsp[i],y_mapsp[i],x_mapep[i][j],y_mapep[i][j])
        bresenham_x = bresenham_xy[0,:].astype(np.int16)
        bresenham_y = bresenham_xy[1,:].astype(np.int16)
        MAP['map'][-bresenham_y,bresenham_x] -= int(np.log(4))
        MAP['map'][-y_mapep[i][j],x_mapep[i][j]] += int(np.log(8))

       # if ((x_mapep[i][j]>1) and (x_mapep[i][j] < MAP['sizex']) and (y_mapep[i][j]>1) and (y_mapep[i][j] < MAP['sizey'])):
         #   MAP['map'][x_mapep[i][j], y_mapep[i][j]] += 2*np.log(4)
    MAP['map'] = np.clip(MAP['map'], -100, 100)

plt.imshow(MAP['map'], cmap="gray" )
plt.plot(x_mapsp,y_mapsp)
plt.title("Dead Reckoning")
plt.pause(25)