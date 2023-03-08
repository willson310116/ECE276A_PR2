import numpy as np

def laserScanToPoints(laser_scan, range_lim):
    lidar_points = []
    rmin, rmax = range_lim
    for i in range(1081):
        r = laser_scan[i]
        if rmin < r < rmax:
            rad = np.deg2rad(-135 + (i / 1080) * 270)
            lidar_points.append([r * np.cos(rad), r * np.sin(rad), 0])
    return np.array(lidar_points)

def filtering(scan, pose, min_dist=0.1, max_dist=25):
    post_scan = np.empty([1,3])

    for i in range(len(scan)):
        x = scan[i][0] - pose['x']
        y = scan[i][1] - pose['y']
        distance = np.sqrt(x**2 + y**2)
        if (distance > min_dist and distance < max_dist):
            post_scan = np.vstack((post_scan, scan[i]))
        
    post_scan = post_scan[1:]
    return post_scan

def laserToWorldCoordinate(laser_point, pose):
    # create points transformation matrix
    T_laser = np.vstack([laser_point.T, np.ones((laser_point.shape[0], 1)).T])

    # use transform tree
    T = Transform(**pose)

    # transform laser points to world frame
    laser_point_w = T.chain('wTb', 'bTl') @ T_laser

    # get the x-y-z coordinates
    laser_point_w = laser_point_w[:3,:].T
    return laser_point_w

def yaw(rad):
    return np.array([[np.cos(rad), -np.sin(rad), 0],
                     [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])


def pitch(rad):
    return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0],
                     [-np.sin(rad), 0, np.cos(rad)]])


def roll(rad):
    return np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)],
                     [0, np.sin(rad), np.cos(rad)]])


class Transform:
    def __init__(self,
                 x=0,
                 y=0,
                 theta=0,):
        self.x = x
        self.y = y
        self.theta = theta
    
    @property
    def bodyToLidar(self):
        return np.array([[1, 0, 0, 0.150915],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0.51435],
                         [0, 0, 0, 1]])
    
    @property
    def worldTobody(self):
        R = yaw(self.theta)
        P = np.array([[self.x, self.y, 0.127]]).T
        return np.vstack([np.hstack([R, P]), np.array([0, 0, 0, 1])])
    

    @property
    def bTl(self):
        return self.bodyToLidar

    @property
    def wTb(self):
        return self.worldTobody

    def chain(self, *transforms):
        ret = np.eye(4)
        for t in transforms:
            ret = ret @ getattr(self, t)
        return ret