import numpy as np

def getOdometry(data_aligned):
    robot_pose = np.zeros((len(data_aligned['lidar']['data']), 3))
    curr_x = 0
    curr_y = 0
    curr_angle = 0
    av_list = data_aligned['imu']['angular_velocity'][:,2]
    encoder_motion = getMotionFromEncoder(data_aligned['encoder']['counts'])

    for i, t in enumerate(data_aligned['time_stamps']):
        if i == 0:
            continue
        
        tau = (data_aligned['time_stamps'][i] - data_aligned['time_stamps'][i-1])
        omega = av_list[i]
        vt = (encoder_motion[i][0] + encoder_motion[i][1]) / 2

        curr_x += tau * vt * np.cos(curr_angle)
        curr_y += tau * vt * np.sin(curr_angle)
        curr_angle += tau * omega
        robot_pose[i][0] = curr_x
        robot_pose[i][1] = curr_y
        robot_pose[i][2] = curr_angle
    return robot_pose

def getMotionFromEncoder(encoder_data):
    encoder_motion = np.empty((1,2))
    for (fr, fl, rr, rl) in encoder_data:
        travel_r = (fr + rr) / 2 * 0.0022 * 40
        travel_l = (fl + rl) / 2 * 0.0022 * 40
        encoder_motion = np.concatenate((encoder_motion, np.array([[travel_l, travel_r]])))
    encoder_motion = encoder_motion[1:,:]
    return encoder_motion
    
def getRelativeMotion(robot_pose, idx, step_size):
    if(idx >= step_size): 
        delta_pose = robot_pose[idx] - robot_pose[idx-step_size]
    else:
        # first iteration delta can only get from previous pose
        delta_pose = robot_pose[idx] - robot_pose[idx-1]
        
    return delta_pose

def motionModelPrediction(particles, motion, var_scale):
    motion_noise = np.random.randn(particles.shape[0], 3) * var_scale
    particles = particles + motion + motion_noise
    particles[:,2] = particles[:,2] % (2 * np.pi)
    return particles