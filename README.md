# Particle_Filter_SLAM

Implemented simultaneous localization and mapping using odometry, 2-D LiDAR scans, stereo camera measurements to localize the robot, build a 2-D occupancy grid map of the environment and add RGB texture to the 2D map

How to run the code:

1. Copy all csv files to source folder.
2. Add lidar, encoder and FOG csv files generated from sensor to source folder.
3. Run pr2_utils and usable_functions which are .py function files from the code folder.
4. Then run deadreckoning.py to plot deadreckoning map and trajectory.
5. Then run Trajectory.py to plot trajectory with one particle.
6. Then run SLAM.py to plot map with 5 particles.
