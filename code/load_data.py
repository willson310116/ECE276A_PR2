import os
import numpy as np
from collections import defaultdict

def t_align(desire_time, ref_time):
    idx = np.argmin(abs(desire_time - ref_time))
    return idx

def alignDataWithTime(data):
  data_aligned = defaultdict(dict)
  data_aligned['time_stamps'] = data['lidar']['lidar_stamps']
  data_aligned['lidar']['data'] = data['lidar']['lidar_ranges']
  data_aligned['imu']['linear_acceleration'] = data['imu']['imu_linear_acceleration'][0]
  data_aligned['imu']['angular_velocity'] = data['imu']['imu_angular_velocity'][0]
  data_aligned['encoder']['counts'] = data['encoder']['encoder_counts'][0]

  data_aligned['imu']['linear_acceleration'] = np.expand_dims(data_aligned['imu']['linear_acceleration'], axis=0)
  data_aligned['imu']['angular_velocity'] = np.expand_dims(data_aligned['imu']['angular_velocity'], axis=0)
  data_aligned['encoder']['counts'] = np.expand_dims(data_aligned['encoder']['counts'], axis=0)
  
  ref_time = data['imu']['imu_stamps']
  ref_time_l = data['lidar']['lidar_stamps']
  for i in range(len(data['lidar']['lidar_stamps'])):
    lidar_time = data['lidar']['lidar_stamps'][i]
    idx = t_align(lidar_time, ref_time)
    data_aligned['imu']['angular_velocity'] = np.concatenate((data_aligned['imu']['angular_velocity'],
                                                            np.expand_dims(data['imu']['imu_angular_velocity'][idx], axis=0)))
    data_aligned['imu']['linear_acceleration'] = np.concatenate((data_aligned['imu']['linear_acceleration'],
                                                                np.expand_dims(data['imu']['imu_linear_acceleration'][idx], axis=0)))
    if i < len(data['encoder']['encoder_stamps']):
      encoder_time = data['encoder']['encoder_stamps'][i]
      idx_e = t_align(encoder_time, ref_time_l)
      if idx_e >= len(data['encoder']['encoder_stamps']) - 1:
        idx_e = len(data['encoder']['encoder_stamps']) - 1
      data_aligned["encoder"]['counts'] = np.concatenate((data_aligned["encoder"]['counts'],
                                                          np.expand_dims(data['encoder']['encoder_counts'][idx_e], axis=0)))
    else:
      data_aligned["encoder"]['counts'] = np.concatenate((data_aligned["encoder"]['counts'],
                                                          np.expand_dims(data['encoder']['encoder_counts'][-1], axis=0)))
  
  data_aligned['imu']['linear_acceleration'] = data_aligned['imu']['linear_acceleration'][1:,]
  data_aligned['imu']['angular_velocity'] = data_aligned['imu']['angular_velocity'][1:,]
  data_aligned['encoder']['counts'] = data_aligned['encoder']['counts'][1:,]

  return data_aligned

def getData(data_path, encoder_file_name, lidar_file_name, imu_file_name):
  data = dict()
  data["encoder"] = getEncoderData(os.path.join(data_path, encoder_file_name))
  data["lidar"] = getLidarData(os.path.join(data_path, lidar_file_name))
  data["imu"] = getImuData(os.path.join(data_path, imu_file_name))
  return data

def getEncoderData(file_path):
  data_parsed = dict()
  with np.load(file_path) as data:
    data_parsed["encoder_counts"] = data["counts"].T # 4 x n encoder counts, (4, 4956)
    data_parsed["encoder_stamps"] = data["time_stamps"] # encoder time stamps, (4956,)
    # exchange dim to (vec, num)
  return data_parsed

def getLidarData(file_path):
  data_parsed = dict()
  with np.load(file_path) as data:
    data_parsed["lidar_angle_min"] = data["angle_min"].item() # start angle of the scan [rad], ()
    data_parsed["lidar_angle_max"] = data["angle_max"].item() # end angle of the scan [rad], ()
    data_parsed["lidar_angle_increment"] = data["angle_increment"][0][0] # angular distance between measurements [rad], (1, 1)
    data_parsed["lidar_range_min"] = data["range_min"].item() # minimum range value [m], ()
    data_parsed["lidar_range_max"] = data["range_max"].item() # maximum range value [m], ()
    data_parsed["lidar_ranges"] = data["ranges"].T       # range data [m] (Note: values < range_min or > range_max should be discarded), (1081, 4962)
    data_parsed["lidar_stamps"] = data["time_stamps"]  # acquisition times of the lidar scans, (4962,)
    # exchange dim to (vec, num)
  return data_parsed

def getImuData(file_path):
  data_parsed = dict()
  with np.load(file_path) as data:
    data_parsed["imu_angular_velocity"] = data["angular_velocity"].T # angular velocity in rad/sec, (3, 12187)
    data_parsed["imu_linear_acceleration"] = data["linear_acceleration"].T # Accelerations in gs (gravity acceleration scaling), (3, 12187)
    data_parsed["imu_stamps"] = data["time_stamps"]  # acquisition times of the imu measurements, (12187,)
  return data_parsed

def getKinectData(file_path):
  data_parsed = dict()
  with np.load(file_path) as data:
    data_parsed["disp_stamps"] = data["disparity_time_stamps"] # acquisition times of the disparity images, (2407,)
    data_parsed["rgb_stamps"] = data["rgb_time_stamps"] # acquisition times of the rgb images, (2289,)
    # exchange dim to (vec, num)
  return data_parsed

  
# if __name__ == '__main__':
#   dataset = 20
  
#   with np.load("../data/Encoders%d.npz"%dataset) as data:
#     encoder_counts = data["counts"] # 4 x n encoder counts, (4, 4956)
#     encoder_stamps = data["time_stamps"] # encoder time stamps, (4956,)

#   with np.load("../data/Hokuyo%d.npz"%dataset) as data:
#     lidar_angle_min = data["angle_min"] # start angle of the scan [rad], ()
#     lidar_angle_max = data["angle_max"] # end angle of the scan [rad], ()
#     lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad], (1, 1)
#     lidar_range_min = data["range_min"] # minimum range value [m], ()
#     lidar_range_max = data["range_max"] # maximum range value [m], ()
#     lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded), (1081, 4962)
#     lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans, (4962,)
#     # lidar_angle = lidar_angle_min + lidar_angle_increment * (idx - 1)
    
#   with np.load("../data/Imu%d.npz"%dataset) as data:
#     imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec, (3, 12187)
#     imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling), (3, 12187)
#     imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements, (12187,)
  
#   with np.load("../data/Kinect%d.npz"%dataset) as data:
#     disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images, (2407,)
#     rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images, (2289,)

