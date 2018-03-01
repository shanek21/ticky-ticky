""" Detect an object in the camera and plot its position """

import numpy as np
import matplotlib.pyplot as plt


file_path = '../data/'
file_name = 'double_pendula_in_phase'
file_ext = '.txt'
loaded_data = np.loadtxt(file_path + file_name + file_ext)
crop_start = 0
crop_stop = loaded_data.shape[1] - 0
loaded_data = loaded_data[:, crop_start:crop_stop]
time_l = loaded_data[0]
theta_l = loaded_data[1]
time_r = loaded_data[2]
theta_r = loaded_data[3]

plt.rcParams.update({'font.size': 20})
plt.plot(time_l, theta_l)
plt.plot(time_r, theta_r)
plt.title(file_name.replace('_', ' ').title())
plt.xlabel('Time (s)')
plt.ylabel('Angle')
plt.legend(['Left Pendulum', 'Right Pendulum'])
plt.show()
