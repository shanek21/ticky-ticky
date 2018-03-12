""" Detect an object in the camera and plot its position """

import numpy as np
import matplotlib.pyplot as plt


def plot_saved_data(file_path, file_name, file_ext):
    loaded_data = np.loadtxt(file_path + file_name + file_ext, delimiter=',')
    crop_start = 0
    crop_stop = loaded_data.shape[1] - 0
    loaded_data = loaded_data[:, crop_start:crop_stop]

    cart_data = np.loadtxt(file_path + file_name + '_cart' + file_ext, delimiter=',')

    shape = loaded_data.shape
    if shape[0] == 4: # two pendula
        time_l = loaded_data[0]
        theta_l = loaded_data[1]
        time_r = loaded_data[2]
        theta_r = loaded_data[3]
        plt.plot(time_l, theta_l)
        plt.plot(time_r, theta_r)
        legend = ['Left Pendulum', 'Right Pendulum']
    elif shape[0] == 2: # one pendulum
        time = loaded_data[0]
        theta = loaded_data[1]
        plt.plot(time, theta)
        legend = ['Pendulum']
    else:
        print 'Data has unexpected shape: ' + str(shape)

    time_cart = cart_data[0]
    cart_pos = (cart_data[1] - np.average(cart_data[1])) / (640 * 2)
    plt.plot(time_cart, cart_pos)

    legend.append('Cart')
    plt.legend(legend)
    plt.rcParams.update({'font.size': 20})
    plt.title(file_name.replace('_', ' ').title())
    plt.xlabel('Time (s)')
    plt.ylabel('Angle')
    plt.show()

if __name__ == '__main__':
    file_path = '../data/'
    file_name = '2018-03-11-14-30-04'
    file_ext = '.csv'
    plot_saved_data(file_path, file_name, file_ext)
