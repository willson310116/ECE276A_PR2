# ECE-276A-Project2
## Particle Filter SLAM
This is the project 2 of the course UCSD ECE276A: Sensing & Estimation in Robotics.

1. Particle-filter SLAM: Implement a particle filter with a differential- drive motion model and scan-grid correlation observation model for simultaneous localization and occupancy-grid mapping. 
2. Texture map: Using the RGBD images and the estimated robot trajectory to produce a 2D color map of the floor surface.

## Usage:
### Install package:
    pip3 install -r requirement.txt
### Run code:
    Follow the steps in main.ipynb


### Source code description:
- **code/main.ipynb**: Main function.
- **code/load_data.py**: Functions for loading data and sync data.
- **code/map_utils.py**: Class for occupancy map.
- **code/motion_utils.py**: Functions motion model and odometry.
- **code/particle.py**: Class and function for particles.
- **code/transformation.py**: Functions for data transformation and others.

    
