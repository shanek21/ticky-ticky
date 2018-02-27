""" Detect an object in the camera and plot its position """

import numpy as np
import matplotlib.pyplot as plt


file_path = '../data/'
file_name = '2018-02-26-20-00-05.txt'
loaded_data = np.loadtxt(file_path + file_name)
time_l = loaded_data[0]
theta_l = loaded_data[1]
time_r = loaded_data[2]
theta_r = loaded_data[3]

plt.plot(time_l, theta_l)
plt.plot(time_r, theta_r)
plt.title('Angle vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle')
plt.legend(['Left Pendulum', 'Right Pendulum'])
plt.show()
