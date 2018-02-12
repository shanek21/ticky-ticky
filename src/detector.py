""" Detect an object in the camera and plot its position """

import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Detector(object):
    """ Detect an object based on color """

    def __init__(self, plot=True):

        # Whether or not to do a live plot of the data
        self.plot = plot

        # Grab video capture
        self.cap = cv.VideoCapture(0)
        self.cap_height = self.cap.get(4) # height of video stream in pixels

        # Create windows
        cv.namedWindow('bgr_window') # window for unprocessed image
        cv.namedWindow('mask_window') # window for binary image
        cv.namedWindow('sliders_window') # window for parameter sliders

        # Initialize CV images
        self.bgr_img = None # the latest image from the camera
        self.hsv_img = None
        self.mask_img = None
        self.blurred_mask_img = None

        # HSV filter sliders
        self.hsv_lb = np.array([0, 155, 120]) # hsv lower bound
        self.hsv_ub = np.array([21, 249, 220]) # hsv upper bound
        cv.createTrackbar('H lb', 'sliders_window', self.hsv_lb[0], 255, self.set_h_lb)
        cv.createTrackbar('H ub', 'sliders_window', self.hsv_ub[0], 255, self.set_h_ub)
        cv.createTrackbar('S lb', 'sliders_window', self.hsv_lb[1], 255, self.set_s_lb)
        cv.createTrackbar('S ub', 'sliders_window', self.hsv_ub[1], 255, self.set_s_ub)
        cv.createTrackbar('V lb', 'sliders_window', self.hsv_lb[2], 255, self.set_v_lb)
        cv.createTrackbar('V ub', 'sliders_window', self.hsv_ub[2], 255, self.set_v_ub)

        # Enable interactive plotting mode
        plt.ion()

        # Initialize plot variables
        self.start_time = time.time()
        self.t = np.array([]) # time
        self.y = np.array([]) # height


    def set_h_lb(self, val):
        """ Slider callback to set hue lower bound """

        self.hsv_lb[0] = val


    def set_h_ub(self, val):
        """ Slider callback to set hue upper bound """

        self.hsv_ub[0] = val


    def set_s_lb(self, val):
        """ Slider callback to set saturation lower bound """

        self.hsv_lb[1] = val


    def set_s_ub(self, val):
        """ Slider callback to set saturation upper bound """

        self.hsv_ub[1] = val


    def set_v_lb(self, val):
        """ Slider callback to set value lower bound """

        self.hsv_lb[2] = val


    def set_v_ub(self, val):
        """ Slider callback to set value upper bound """

        self.hsv_ub[2] = val


    def hsv_filt(self):
        """ Hsv filter the video cap """

        while 1:
            # Setup cv windows
            cv.namedWindow('bgr_window') # window for bgr video stream
            cv.namedWindow('sliders_window') # window for parameter sliders
            cv.namedWindow('mask_window') # window for binary mask image

            # Read in a single image from the video stream
            _, self.bgr_img = self.cap.read()

            # Convert bgr to hsv
            self.hsv_img = cv.cvtColor(self.bgr_img, cv.COLOR_BGR2HSV)

            # Threshold the hsv_img to get only blue colors
            self.mask_img = cv.inRange(self.hsv_img, self.hsv_lb, self.hsv_ub)

            # Erode away small particles in the mask image
            self.mask_img = cv.erode(self.mask_img, None, iterations=3)

            # Blur the masked image to improve contour detection
            self.blurred_mask_img = cv.GaussianBlur(self.mask_img, (11, 11), 0)

            # Detect contours
            contours = cv.findContours(self.blurred_mask_img.copy(), cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_SIMPLE)[-2]

            # Draw bounding circle around largest contour
            if len(contours) > 0:
                largest_contour = max(contours, key=cv.contourArea)
                ((x, y), radius) = cv.minEnclosingCircle(largest_contour)

                # Draw circles on image to represent the ball
                if radius > 10:
                    cv.circle(self.bgr_img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv.circle(self.bgr_img, (int(x), int(y)), 5, (0, 0, 255), -1)
                    self.t = np.append(self.t, time.time() - self.start_time)
                    self.y = np.append(self.y, self.cap_height - y)
                    if self.plot:
                        plt.scatter(self.t[-1], self.y[-1])
                        plt.pause(0.05)
                else:
                    print "Radius too small!"

            # Show windows
            cv.imshow('bgr_window', self.bgr_img)
            # cv.imshow('mask_window', self.mask_img)

            # Delay and look for user input to exit loop
            k = cv.waitKey(5) & 0xFF
            if k == 27: # exit if user presses the Escape key
                # Save data to text file
                now = datetime.now()
                timestamp = '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
                np.savetxt('../data/' + timestamp + '.txt', np.array([self.t, self.y]))

                # Plot data
                plt.ioff() # turn off interactive plotting
                plt.title('Height vs Time')
                plt.xlabel('Time (s)')
                plt.ylabel('Height (pixels)')
                plt.plot(self.t, self.y)
                plt.show()
                break

        # Close opencv windows
        cv.destroyAllWindows()


if __name__ == '__main__':
    detector = Detector(plot=True)
    detector.hsv_filt()
