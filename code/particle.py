import numpy as np
from transformation import laserScanToPoints, filtering, laserToWorldCoordinate
from pr2_utils import mapCorrelation

class Particle:
    def __init__(self, num=128):
        self.num = num
        self.weight = np.ones(num) / num
        self.states = np.zeros((num, 3)) + \
            np.random.randn(num, 3) * np.array([0.1, 0.1, 0.1 * np.pi/180])

    def resampling(self):
        N = self.num
        beta = 0
        chose_idx = []
        index = int(np.random.choice(np.arange(N), 1, p=[1/N]*N))  # choose an index uniformly

        for _ in range(N):
            beta = beta + np.random.uniform(low=0, high=2*np.max(self.weight), size=1)
            while(self.weight[index] < beta):
                beta  = beta - self.weight[index]
                index = (index+1) % N
            chose_idx.append(index)
        
        self.states = self.states[chose_idx]
        self.weight.fill(1/self.num)


def softmax(x):
    x = np.exp(x-np.max(x))
    return x / x.sum()

def measurement_model_update(Map, P, lidar_data, i):
    # calculate map correlation for each particle
    l = 2
    corrs = []
    res = Map.res
    particles = P.states

    grid_tmp = np.zeros_like(Map.grid)  # for calculate correlation
    grid_tmp[Map.grid > 0] = 1          # occupied
    grid_tmp[Map.grid < 0] = 0          # free

    # X, Y  = polar2cart(data['scan'], angles)  # polar coord -> cartesian coord
    laser_point = laserScanToPoints(lidar_data["lidar_ranges"][i], (lidar_data["lidar_range_min"], lidar_data["lidar_range_max"]))
    x_im, y_im = np.arange(Map.xmin, Map.xmax + res, res), np.arange(Map.ymin, Map.ymax + res, res)
    x_range, y_range = np.arange(-res * l, res * l + res, res), np.arange(-res * l, res * l + res, res)

    for i in range(len(particles)):
        particle_state = {'x':particles[i][0], 'y':particles[i][1], 'theta':particles[i][2]}
        scan_w = laserToWorldCoordinate(laser_point, particle_state)
        scan_w = filtering(scan_w, particle_state)
        x, y = scan_w[:,0], scan_w[:,1]
        corr = mapCorrelation(grid_tmp, x_im, y_im, np.vstack((x,y)), particles[i][0] + x_range, particles[i][1] + y_range)
        corrs.append(np.max(corr))

    # get the particle with largest weight
    corrs = np.array(corrs)
    P.weight = softmax(P.weight * corrs)
    best_idx = np.where(P.weight==np.max(P.weight))[0][0]
    best_particle = particles[best_idx]
    return best_particle

