import numpy as np
from transformation import *
from pr2_utils import *


class OccupancyMap:
    def __init__(self, 
                 xlim=(-30,30),
                 ylim=(-30,30),
                 res=0.05):
        self.res = res
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.xsize = int(np.ceil((self.xmax - self.xmin) / self.res))
        self.ysize = int(np.ceil((self.ymax - self.ymin) / self.res))
        self.grid = np.zeros((self.xsize, self.ysize))
    
    def mapping(self, data, id, pose):
        # free_odd = np.log(9)/4
        # occu_odd = np.log(9)
        free_odd = np.log(4)
        occu_odd = np.log(4)

        # from range to points
        laser_point = laserScanToPoints(data["lidar_ranges"][id], (data["lidar_range_min"], data["lidar_range_max"]))
        # from laser frame to world frame
        laser_point_w = laserToWorldCoordinate(laser_point, pose)
        # filter out too close or far points
        laser_point_w = filtering(laser_point_w, pose)

        xi, yi = (laser_point_w[:, 0]/self.res).astype(int), (laser_point_w[:, 1]/self.res).astype(int)
        for (a, b) in zip(xi, yi):
            line = bresenham2D(int(pose['x'] / self.res), int(pose['y'] / self.res), a, b).astype(np.int16)
            x = a + self.grid.shape[0] // 2  # offset to center
            y = b + self.grid.shape[1] // 2  # offset to center
            self.grid[x, y] += occu_odd
            self.grid[line[0] + self.grid.shape[0] // 2, line[1] + self.grid.shape[1] // 2] -= free_odd
        # clip
        self.grid[self.grid >= 100]  = 100
        self.grid[self.grid <= -100] = -100

    def plot(self, robot_pose, trajectory, file_name):
        print("Plot...")
        fig = plt.figure(figsize=(12,6))

        ax1 = fig.add_subplot(121)
        plt.plot(robot_pose.T[0], robot_pose.T[1], label="Lidar Odom")
        plt.scatter((trajectory[1:].T[0] - self.grid.shape[0] // 2) * self.res,
                    (trajectory[1:].T[1] - self.grid.shape[1] // 2) * self.res,
                    label="Particle Filter Odom", s=2, c='r')
        plt.legend(loc='upper left')
        
        ax2 = fig.add_subplot(122)
        plt.imshow(self.grid, cmap='gray', vmin=-100, vmax=100, origin='lower')
        plt.scatter(trajectory[1:].T[1], trajectory[1:].T[0], s=1, c='r')
        plt.title("Occupancy grid (log-odds)")
        plt.savefig(file_name)
        plt.show()