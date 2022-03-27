import numpy as np
from matplotlib import pyplot as plt
from math import cos, sin, radians, pi
import os, cv2
from pr2_utils import bresenham2D,mapCorrelation
from tqdm import tqdm

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


num_par = 5
weights = np.ones(num_par)/num_par
u = np.zeros((3,(num_par)))

# Body2Lidar
RPY_lidar = [142.759, 0.0584636, 89.9254]
R_lidar = [0.00130201, 0.796097, 0.605167, 0.999999, -0.000419027, -0.00160026, -0.00102038, 0.605169, -0.796097]
T_lidar = [0.8349, -0.0126869, 1.76416]

transform_lidar2body = np.array([[R_lidar[0], R_lidar[1], R_lidar[2], T_lidar[0]],
                                 [R_lidar[3], R_lidar[4], R_lidar[5], T_lidar[1]],
                                 [R_lidar[6], R_lidar[7], R_lidar[8], T_lidar[2]],
                                 [0, 0, 0, 1]])

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

def predict(i,par_loc=0,noisy_angle=0):
    if i == 0:
        ang_noise = np.random.normal(0,np.abs(cum_angle_array[i])/100,num_par).reshape(num_par,1)
    else:
        ang_noise = np.random.normal(0,np.abs(cum_angle_array[i]-cum_angle_array[i-1])/100,num_par).reshape(num_par,1)

    if i == 0:
        noisy_angle = ang_noise + cum_angle_array[i]
    else:
        noisy_angle = ang_noise + cum_angle_array[i]
        # noisy_angle = resam_angle + ang_noise + (cum_angle_array[i] - cum_angle_array[i-1])

    dist_noise = np.random.normal(0,np.abs(mat_dist_array[i])/100,num_par).reshape(num_par,1)

    noisy_dist = dist_noise + mat_dist_array[i]
    
    if i == 0:
        par_loc = np.zeros((num_par,3))
        
    else:
        par_loc[:,0] += ((noisy_dist)*np.cos(noisy_angle)).flatten()
        par_loc[:,1] += ((noisy_dist)*np.sin(noisy_angle)).flatten()
    
        # par_loc = np.zeros((num_par,3))
        # par_loc[:,0] = resam_par_loc[:,0] + (noisy_dist)*np.cos(noisy_angle)
        # par_loc[:,1] = resam_par_loc[:,1] + (noisy_dist)*np.sin(noisy_angle)
    
    # par_loc[:,0] -= 0.335
    # par_loc[:,1] -= 0.035
    # par_loc[:,2] += 0.78

    return(par_loc,noisy_angle)

def lidar_wep (i,par_loc,noisy_angle):
    if i == 0:
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

        transform_body2world = np.array([[np.cos(cum_angle_array[i]),-np.sin(cum_angle_array[i]),0,par_loc[i,0]],
                                        [np.sin(cum_angle_array[i]),np.cos(cum_angle_array[i]),0,par_loc[i,1]],
                                        [0,0,1,par_loc[i,2]],                                                   
                                        [0, 0, 0, 1]])

        lid_world_ep = np.zeros((4,len(lid_rays)))
        lid_world_ep = np.matmul(transform_body2world,lid_veh)

        lid_world_ep = np.delete(lid_world_ep,3,0)
        lid_wep = np.transpose(lid_world_ep)

    else:
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

        lid_wep = []
        for k in range(num_par):
            transform_body2world = np.array([[np.cos(noisy_angle[k]),-np.sin(noisy_angle[k]),0,par_loc[k,0]],
                                    [np.sin(noisy_angle[k]),np.cos(noisy_angle[k]),0,par_loc[k,1]],
                                    [0,0,1,par_loc[k,2]],                                                   
                                    [0, 0, 0, 1]])
        
            lid_world_ep = np.zeros((4,len(lid_rays)))
            lid_world_ep = np.matmul(transform_body2world,lid_veh)

            lid_world_ep = np.delete(lid_world_ep,3,0)
            lid_world_ep = np.transpose(lid_world_ep)
            lid_wep.append(lid_world_ep)
    return(lid_wep)

def map_correlation_update(MAP,lid_wep,weights):
    # grid cells representing walls with 1
    # map_wall = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.5).astype(np.int)

    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x index of each pixel on log-odds map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y index of each pixel on log-odds map

    # 9x9 grid around particle
    x_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])  # x deviation
    y_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])  # y deviation

    correlation = np.zeros(num_par)
    for k in range(num_par):
        ex_w = lid_wep[k][:,0].reshape(1,len(lid_wep[k]))
        ey_w = lid_wep[k][:,1].reshape(1,len(lid_wep[k]))
        Y = np.stack((ex_w, ey_w))
        c = mapCorrelation(MAP['map'], x_im, y_im, Y, x_range, y_range)

        # find largest correlation
        correlation[k] = np.max(c)
    
    d = np.max(correlation)
    beta = np.exp(correlation - d)
    p_h = beta / beta.sum()
    weights*= p_h / np.sum(weights * p_h)
    
    return weights

def map_update(MAP,particle_max_state,lidar_rays):
    x_mapsp = np.ceil((particle_max_state[0]-MAP['xmin'])/MAP['res']).astype(np.int16)-1
    y_mapsp = np.ceil((particle_max_state[1]-MAP['ymin'])/MAP['res']).astype(np.int16)-1

    x_mapep = (np.ceil((lidar_rays[:,0]-MAP['xmin'])/MAP['res']).astype(np.int16)-1)
    y_mapep = (np.ceil((lidar_rays[:,1]-MAP['ymin'])/MAP['res']).astype(np.int16)-1)

    
    for j in range(len(x_mapep)):
        bresenham_xy = bresenham2D(x_mapsp,y_mapsp,x_mapep[j],y_mapep[j])
        bresenham_x = bresenham_xy[0,:].astype(np.int16)
        bresenham_y = bresenham_xy[1,:].astype(np.int16)
        MAP['map'][bresenham_x,bresenham_y] -= np.log(4)
        MAP['map'][x_mapep[j],y_mapep[j]] += np.log(8)

    # if ((x_mapep[i][j]>1) and (x_mapep[i][j] < MAP['sizex']) and (y_mapep[i][j]>1) and (y_mapep[i][j] < MAP['sizey'])):
        #   MAP['map'][x_mapep[i][j], y_mapep[i][j]] += 2*np.log(4)
    MAP['map'] = np.clip(MAP['map'], -100, 100)

    return(MAP,x_mapsp,y_mapsp)

def resampling(particle_max_state):
    particle_state_new = np.zeros((num_par,3))
    particle_weight_new = np.tile(1 / num_par, num_par).reshape(num_par,1)

    x_new = np.random.uniform(0,particle_max_state[0]/100,5)
    y_new = np.random.uniform(0,particle_max_state[1]/100,5)

    particle_state_new = np.zeros((num_par,3))
    particle_state_new[:,0] = x_new + particle_max_state[0]
    particle_state_new[:,1] = y_new + particle_max_state[1]
    particle_state_new[:,2] = np.zeros((num_par)) + particle_max_state[2]
    
    particle_weight_new = np.ones(num_par)/num_par

    return particle_state_new, particle_weight_new
