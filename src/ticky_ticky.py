""" Detect an object in the camera and plot its position """

import time
import math
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
        self.cap_width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH) # width of video stream in pixels
        self.cap_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) # height of video stream in pixels

        # Create windows
        cv.namedWindow('bgr_window') # window for unprocessed image
        cv.moveWindow('bgr_window', 0, 0)
        cv.namedWindow('mask_window') # window for binary image
        cv.moveWindow('mask_window', 700, 0)

        # Initialize CV images
        self.bgr_img = None # the latest image from the camera
        self.hsv_img = None
        self.mask_img = None
        self.mask_img = None

        # HSV filter sliders
        hsv_parameters = np.loadtxt('hsv_parameters.txt', dtype=np.int_)
        self.hsv_lb = hsv_parameters[0]
        self.hsv_ub = hsv_parameters[1]
        cv.createTrackbar('H lb', 'mask_window', self.hsv_lb[0], 255, self.set_h_lb)
        cv.createTrackbar('H ub', 'mask_window', self.hsv_ub[0], 255, self.set_h_ub)
        cv.createTrackbar('S lb', 'mask_window', self.hsv_lb[1], 255, self.set_s_lb)
        cv.createTrackbar('S ub', 'mask_window', self.hsv_ub[1], 255, self.set_s_ub)
        cv.createTrackbar('V lb', 'mask_window', self.hsv_lb[2], 255, self.set_v_lb)
        cv.createTrackbar('V ub', 'mask_window', self.hsv_ub[2], 255, self.set_v_ub)

        # Enable interactive plotting mode
        plt.ion()

        # Initialize plot variables
        self.start_time = time.time()
        self.t = np.array([]) # time in seconds
        self.x = np.array([]) # x position in pixels
        self.y = np.array([]) # y position in pixels


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
            # cv.namedWindow('sliders_window') # window for parameter sliders
            cv.namedWindow('mask_window') # window for binary mask image

            # Read in a single image from the video stream
            _, self.bgr_img = self.cap.read()

            # Draw lines to visually split bgr_img in four equal quadrants
            line_color = (255, 255, 0)
            cv.line(self.bgr_img, (int(self.cap_width / 2), 0),
                    (int(self.cap_width / 2), int(self.cap_height)), line_color)
            cv.line(self.bgr_img, (0, int(self.cap_height / 2)),
                    (int(self.cap_width), int(self.cap_height / 2)), line_color)

            # Convert bgr to hsv
            self.hsv_img = cv.cvtColor(self.bgr_img, cv.COLOR_BGR2HSV)

            # Threshold the hsv_img to get only blue colors
            self.mask_img = cv.inRange(self.hsv_img, self.hsv_lb, self.hsv_ub)

            # Erode away small particles in the mask image
            self.mask_img = cv.erode(self.mask_img, None, iterations=1)

            # Blur the masked image to improve contour detection
            self.mask_img = cv.GaussianBlur(self.mask_img, (11, 11), 0)

            # Detect contours
            contours = cv.findContours(self.mask_img.copy(), cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_SIMPLE)[-2]

            # Draw bounding circle around largest contour
            tl = [] # largest contour in top-left quadrant
            tla = 0 # area of largest contour in top-left quadrant
            tr = [] # largest contour in top-right quadrant
            tra = 0 # area of largest contour in top-right quadrant
            bl = [] # largest contour in bottom-left quadrant
            bla = 0 # area of largest contour in bottom-left quadrant
            br = [] # largest contour in bottom-right quadrant
            bra = 0 # area of largest contour in bottom-right quadrant
            for c in contours:
                p = c[0][0] # the first point in the contour
                if p[0] < self.cap_width / 2: # left half of bgr_img
                    if p[1] < self.cap_height / 2: # top-left quadrant of bgr_img
                        if cv.contourArea(c) > tla:
                            tl = c
                            tla = cv.contourArea(c)
                    else: # bottom-left quadrant of bgr_img
                        if cv.contourArea(c) > bla:
                            bl = c
                            bla = cv.contourArea(c)
                else: # right half of bgr_img
                    if p[1] < self.cap_height / 2: # top-right quadrant of bgr_img
                        if cv.contourArea(c) > tra:
                            tr = c
                            tra = cv.contourArea(c)
                    else: # bottom-right quadrant of bgr_img
                        if cv.contourArea(c) > bra:
                            br = c
                            bra = cv.contourArea(c)

            largest_contours = [ [tl, 'Top Left'], [tr, 'Top Right'], [bl, 'Bottom Left'], [br, 'Bottom Right'] ]
            for c in largest_contours:
                if len(c[0]) == 0: # if c is actually a contour
                    continue

                # Draw circle on image to show detected object
                ((x, y), radius) = cv.minEnclosingCircle(c[0])
                cv.circle(self.bgr_img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv.putText(self.bgr_img, c[1], (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                if self.plot:
                    self.t = np.append(self.t, time.time() - self.start_time)
                    self.x = np.append(self.x, x)
                    self.y = np.append(self.y, self.cap_height - y)
                    plt.scatter(self.t[-1], self.x[-1])
                    plt.pause(0.005)

            # Show windows
            cv.imshow('bgr_window', self.bgr_img)
            cv.imshow('mask_window', self.mask_img)

            # Delay and look for user input to exit loop
            k = cv.waitKey(5) & 0xFF
            if k == 27: # exit if user presses the 'Escape' key
                if self.plot:
                    # Save data to text file
                    now = datetime.now()
                    timestamp = '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
                    np.savetxt('../data/' + timestamp + '.txt', np.array([self.t, self.x]))

                    # Plot data
                    plt.ioff() # turn off interactive plotting
                    plt.title('Position vs Time')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Position (pixels)')
                    plt.plot(self.t, self.x)
                    plt.show()
                break
            elif k == 115: # save hsv parameters if user presses 's' key
                np.savetxt('hsv_parameters.txt', np.array([self.hsv_lb, self.hsv_ub]), fmt='%03d')
                print 'Saved HSV parameters.'

        # Close opencv windows
        cv.destroyAllWindows()


if __name__ == '__main__':
    import sys


    plot = True
    if len(sys.argv) > 1 and sys.argv[1] == 'false':
        plot = False
        print 'Plot set to False.'
    detector = Detector(plot=plot)
    detector.hsv_filt()
