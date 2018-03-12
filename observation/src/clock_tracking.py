""" Detect an object in the camera and plot its position """

import time
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Detector(object):
    """ Detect an object based on color """

    def __init__(self, num_pendula=2, cart=True, plot=True):

        # The number of pendula to track
        self.num_pendula = num_pendula

        # Whether or not to track a cart
        self.cart = cart

        # Whether or not to do a live plot of the data
        self.plot = plot

        # Grab video capture
        self.pendula_cap = cv.VideoCapture(0)
        if self.cart: self.cart_cap = cv.VideoCapture(1)
        self.cap_width = self.pendula_cap.get(cv.CAP_PROP_FRAME_WIDTH) # width of video stream in pixels
        self.cap_height = self.pendula_cap.get(cv.CAP_PROP_FRAME_HEIGHT) # height of video stream in pixels

        # Create windows
        cv.namedWindow('pendula_window')
        cv.moveWindow('pendula_window', 0, 0)
        if self.cart:
            cv.namedWindow('cart_window')
            cv.moveWindow('cart_window', 0, 700)


        # Initialize images
        self.pendula_bgr_img = None
        self.pendula_hsv_img = None
        self.pendula_mask_img = None
        if self.cart:
            self.cart_bgr_img = None
            self.cart_hsv_img = None
            self.cart_mask_img = None

        # Pendula HSV filter sliders
        pendula_hsv_parameters = np.loadtxt('pendula_hsv_parameters.csv', delimiter=',', dtype=np.int_)
        self.pendula_hsv_lb, self.pendula_hsv_ub = pendula_hsv_parameters
        cv.createTrackbar('H lb', 'pendula_window', self.pendula_hsv_lb[0], 255, self.set_pendula_h_lb)
        cv.createTrackbar('H ub', 'pendula_window', self.pendula_hsv_ub[0], 255, self.set_pendula_h_ub)
        cv.createTrackbar('S lb', 'pendula_window', self.pendula_hsv_lb[1], 255, self.set_pendula_s_lb)
        cv.createTrackbar('S ub', 'pendula_window', self.pendula_hsv_ub[1], 255, self.set_pendula_s_ub)
        cv.createTrackbar('V lb', 'pendula_window', self.pendula_hsv_lb[2], 255, self.set_pendula_v_lb)
        cv.createTrackbar('V ub', 'pendula_window', self.pendula_hsv_ub[2], 255, self.set_pendula_v_ub)

        # Cart HSV filter sliders
        if self.cart:
            cart_hsv_parameters = np.loadtxt('cart_hsv_parameters.csv', delimiter=',', dtype=np.int_)
            self.cart_hsv_lb, self.cart_hsv_ub = cart_hsv_parameters
            cv.createTrackbar('H lb', 'cart_window', self.cart_hsv_lb[0], 255, self.set_cart_h_lb)
            cv.createTrackbar('H ub', 'cart_window', self.cart_hsv_ub[0], 255, self.set_cart_h_ub)
            cv.createTrackbar('S lb', 'cart_window', self.cart_hsv_lb[1], 255, self.set_cart_s_lb)
            cv.createTrackbar('S ub', 'cart_window', self.cart_hsv_ub[1], 255, self.set_cart_s_ub)
            cv.createTrackbar('V lb', 'cart_window', self.cart_hsv_lb[2], 255, self.set_cart_v_lb)
            cv.createTrackbar('V ub', 'cart_window', self.cart_hsv_ub[2], 255, self.set_cart_v_ub)

        # Enable interactive plotting mode
        plt.ion()

        # Initialize plot variables
        self.start_time = time.time()

        if self.num_pendula == 2:
            # We need different times for different pendulums, because we might only detect one of the
            #   two pendulums during a time step, in which case we will need to append a new time to
            #   the time array for that pendulum, but not for the other pendulum.
            self.time_l = np.array([]) # time in seconds for left pendulum
            self.time_r = np.array([]) # time in seconds for right pendulum
            self.theta_l = np.array([]) # angle of left pendulum (from vertical)
            self.theta_r = np.array([]) # angle of right pendulum (from vertical)
        else: # num_pendula == 1
            self.time = np.array([]) # time in seconds for pendulum
            self.theta = np.array([]) # angle of pendulum (from vertical)

        if self.cart:
            self.time_cart = np.array([]) # time in seconds for cart
            self.cart_pos = np.array([]) # position of cart


    # Pendula HSV slider callbacks
    def set_pendula_h_lb(self, val): self.pendula_hsv_lb[0] = val
    def set_pendula_h_ub(self, val): self.pendula_hsv_ub[0] = val
    def set_pendula_s_lb(self, val): self.pendula_hsv_lb[1] = val
    def set_pendula_s_ub(self, val): self.pendula_hsv_ub[1] = val
    def set_pendula_v_lb(self, val): self.pendula_hsv_lb[2] = val
    def set_pendula_v_ub(self, val): self.pendula_hsv_ub[2] = val


    # Cart HSV slider callbacks
    def set_cart_h_lb(self, val): self.cart_hsv_lb[0] = val
    def set_cart_h_ub(self, val): self.cart_hsv_ub[0] = val
    def set_cart_s_lb(self, val): self.cart_hsv_lb[1] = val
    def set_cart_s_ub(self, val): self.cart_hsv_ub[1] = val
    def set_cart_v_lb(self, val): self.cart_hsv_lb[2] = val
    def set_cart_v_ub(self, val): self.cart_hsv_ub[2] = val


    def circle_around_contour(self, contour, label):
        """ Find a minimum enclosing circle around a given contour, draw the circle on
        self.pendula_bgr_img, and return the circle's position """

        ((x, y), r) = cv.minEnclosingCircle(contour)
        cv.circle(self.pendula_bgr_img, (int(x), int(y)), int(r), (0, 255, 255), 2)
        cv.putText(self.pendula_bgr_img, label, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        return (x, y)


    def hsv_filt(self):
        """ Hsv filter the video pendula_cap """

        while 1:
            # Read in a single image from the video stream
            _, self.pendula_bgr_img = self.pendula_cap.read()
            if cart:
                _, self.cart_bgr_img = self.cart_cap.read()

            curr_time = time.time() - self.start_time

            # Draw lines to visually split pendula_bgr_img in four equal quadrants
            line_color = (255, 255, 0)
            # Horizontal bisecting line
            cv.line(self.pendula_bgr_img, (0, int(self.cap_height / 2)),
                    (int(self.cap_width), int(self.cap_height / 2)), line_color)
            if self.num_pendula == 2:
                # Vertical bisecting line
                cv.line(self.pendula_bgr_img, (int(self.cap_width / 2), 0),
                        (int(self.cap_width / 2), int(self.cap_height)), line_color)

            # Convert bgr to hsv
            self.pendula_hsv_img = cv.cvtColor(self.pendula_bgr_img, cv.COLOR_BGR2HSV)
            # Threshold the pendula_hsv_img
            self.pendula_mask_img = cv.inRange(self.pendula_hsv_img, self.pendula_hsv_lb, self.pendula_hsv_ub)
            # Erode away small particles in the mask image
            self.pendula_mask_img = cv.erode(self.pendula_mask_img, None, iterations=1)
            # Blur the masked image to improve contour detection
            self.pendula_mask_img = cv.GaussianBlur(self.pendula_mask_img, (11, 11), 0)
            # Detect contours
            pendula_contours = cv.findContours(self.pendula_mask_img.copy(), cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_SIMPLE)[-2]

            if self.num_pendula == 2:
                # Draw bounding circle around largest contour in each quadrant
                tl = [] # largest contour in top-left quadrant
                tla = 0 # area of largest contour in top-left quadrant
                tr = [] # largest contour in top-right quadrant
                tra = 0 # area of largest contour in top-right quadrant
                bl = [] # largest contour in bottom-left quadrant
                bla = 0 # area of largest contour in bottom-left quadrant
                br = [] # largest contour in bottom-right quadrant
                bra = 0 # area of largest contour in bottom-right quadrant
                for c in pendula_contours:
                    p = c[0][0] # the first point in the contour
                    if p[0] < self.cap_width / 2: # left half of pendula_bgr_img
                        if p[1] < self.cap_height / 2: # top-left quadrant of pendula_bgr_img
                            if cv.contourArea(c) > tla:
                                tl = c
                                tla = cv.contourArea(c)
                        else: # bottom-left quadrant of pendula_bgr_img
                            if cv.contourArea(c) > bla:
                                bl = c
                                bla = cv.contourArea(c)
                    else: # right half of pendula_bgr_img
                        if p[1] < self.cap_height / 2: # top-right quadrant of pendula_bgr_img
                            if cv.contourArea(c) > tra:
                                tr = c
                                tra = cv.contourArea(c)
                        else: # bottom-right quadrant of pendula_bgr_img
                            if cv.contourArea(c) > bra:
                                br = c
                                bra = cv.contourArea(c)

                largest_contours = [ [tl, 'Top Left'], [tr, 'Top Right'], [bl, 'Bottom Left'], [br, 'Bottom Right'] ]

                found_l_pivot = False
                found_l_pendulum = False
                found_r_pivot = False
                found_r_pendulum = False
                if len(tl) > 0:
                    found_l_pivot = True
                    (x_tl, y_tl) = self.circle_around_contour(tl, 'Left Pivot')
                if len(tr) > 0:
                    found_r_pivot = True
                    (x_tr, y_tr) = self.circle_around_contour(tr, 'Right Pivot')
                if len(bl) > 0:
                    found_l_pendulum = True
                    (x_bl, y_bl) = self.circle_around_contour(bl, 'Left Pendulum')
                if len(br) > 0:
                    found_r_pendulum = True
                    (x_br, y_br) = self.circle_around_contour(br, 'Right Pendulum')

                if self.plot and found_l_pivot and found_l_pendulum:
                    opposite = x_bl - x_tl
                    adjacent = y_bl - y_tl
                    hypotenuse = math.sqrt(opposite ** 2 + adjacent ** 2)
                    try:
                        self.theta_l = np.append(self.theta_l, math.asin(opposite / adjacent))
                        self.time_l = np.append(self.time_l, curr_time)
                    except ValueError: print 'Illegal maths: asin(' + str(opposite) + '/' + str(adjacent) + ')'

                if self.plot and found_r_pivot and found_r_pendulum:
                    opposite = x_br - x_tr
                    adjacent = y_br - y_tr
                    hypotenuse = math.sqrt(opposite ** 2 + adjacent ** 2)
                    try:
                        self.theta_r = np.append(self.theta_r, math.asin(opposite / adjacent))
                        self.time_r = np.append(self.time_r, curr_time)
                    except ValueError: print 'Illegal maths: asin(' + str(opposite) + '/' + str(adjacent) + ')'
            else: # self.num_pendula == 1
                t = [] # largest contour in top half
                ta = 0 # area of largest contour in top half
                b = [] # largest contour in bottom half
                ba = 0 # area of largest contour in bottom half
                for c in pendula_contours:
                    p = c[0][0] # the first point in the contour
                    if p[1] < self.cap_height / 2: # top half of pendula_bgr_img
                        if cv.contourArea(c) > ta:
                            t = c
                            ta = cv.contourArea(c)
                    else: # bottom half of pendula_bgr_img
                        if cv.contourArea(c) > ba:
                            b = c
                            ba = cv.contourArea(c)

                largest_contours = [ [t, 'Top'], [b, 'Bottom'] ]

                found_pivot = False
                found_pendulum = False
                if len(t) > 0:
                    found_pivot = True
                    (x_t, y_t) = self.circle_around_contour(t, 'Pivot')
                if len(b) > 0:
                    found_pendulum = True
                    (x_b, y_b) = self.circle_around_contour(b, 'Pendulum')

                if self.plot and found_pivot and found_pendulum:
                    opposite = x_b - x_t
                    adjacent = y_b - y_t
                    hypotenuse = math.sqrt(opposite ** 2 + adjacent ** 2)
                    try:
                        self.theta = np.append(self.theta, math.asin(opposite / adjacent))
                        self.time = np.append(self.time, curr_time)
                    except ValueError: print 'Illegal maths: asin(' + str(opposite) + '/' + str(adjacent) + ')'

            if self.cart:
                # Convert bgr to hsv
                self.cart_hsv_img = cv.cvtColor(self.cart_bgr_img, cv.COLOR_BGR2HSV)
                # Threshold the cart_hsv_img
                self.cart_mask_img = cv.inRange(self.cart_hsv_img, self.cart_hsv_lb, self.cart_hsv_ub)
                # Erode away small particles in the mask image
                self.cart_mask_img = cv.erode(self.cart_mask_img, None, iterations=1)
                # Blur the masked image to improve contour detection
                self.cart_mask_img = cv.GaussianBlur(self.cart_mask_img, (11, 11), 0)
                # Detect contours
                cart_contours = cv.findContours(self.cart_mask_img.copy(), cv.RETR_EXTERNAL,
                        cv.CHAIN_APPROX_SIMPLE)[-2]
                if cart_contours:
                    largest_contour = max(cart_contours, key=cv.contourArea)
                    x, y, w, h = cv.boundingRect(largest_contour)
                    cv.rectangle(self.cart_bgr_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    self.time_cart = np.append(self.time_cart, curr_time)
                    self.cart_pos = np.append(self.cart_pos, w)

            # Show windows
            cv.imshow('pendula_window', np.hstack((self.pendula_bgr_img, cv.cvtColor(self.pendula_mask_img, cv.COLOR_GRAY2RGB))))
            if self.cart:
                cv.imshow('cart_window', np.hstack((self.cart_bgr_img, cv.cvtColor(self.cart_mask_img, cv.COLOR_GRAY2RGB))))

            # Delay and look for user input to exit loop
            k = cv.waitKey(5) & 0xFF
            if k == 27: # exit if user presses the 'Escape' key
                if self.plot:
                    # Turn off interactive plotting
                    plt.ioff()

                    # Save data to text file
                    now = datetime.now()
                    timestamp = '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
                    if self.num_pendula == 2:
                        # Add buffer to collected data so that they are the same size and can be plotted against each other
                        num_data_points = max(self.time_l.size, self.time_r.size)
                        self.time_l = np.pad(self.time_l, (0, num_data_points - self.time_l.size), 'edge')
                        self.theta_l = np.pad(self.theta_l, (0, num_data_points - self.theta_l.size), 'edge')
                        self.time_r = np.pad(self.time_r, (0, num_data_points - self.time_r.size), 'edge')
                        self.theta_r = np.pad(self.theta_r, (0, num_data_points - self.theta_r.size), 'edge')

                        # Save collected data
                        np.savetxt('../data/' + timestamp + '.csv', np.array([self.time_l, self.theta_l, self.time_r, self.theta_r]), delimiter=',')

                        # Plot data
                        plt.plot(self.time_l, self.theta_l)
                        plt.plot(self.time_r, self.theta_r)
                        plt.legend(['Left Pendulum', 'Right Pendulum'])
                    else: # self.num_pendula == 1
                        # Save collected data
                        np.savetxt('../data/' + timestamp + '.csv', np.array([self.time, self.theta]), delimiter=',')

                        # Plot data
                        plt.plot(self.time, self.theta)

                    if self.cart:
                        np.savetxt('../data/' + timestamp + '_cart.csv', np.array([self.time_cart, self.cart_pos]), delimiter=',')


                    # Label graph
                    plt.title('Angle vs Time')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Pendulum Angle')
                    plt.show()
                break
            elif k == 115: # save hsv parameters if user presses 's' key
                np.savetxt('pendula_hsv_parameters.csv', np.array([self.pendula_hsv_lb, self.pendula_hsv_ub]), fmt='%03d', delimiter=',')
                print 'Saved pendula HSV parameters.'
                if self.cart:
                    np.savetxt('cart_hsv_parameters.csv', np.array([self.cart_hsv_lb, self.cart_hsv_ub]), fmt='%03d', delimiter=',')
                    print 'Saved cart HSV parameters.'

        # Close opencv windows
        cv.destroyAllWindows()


if __name__ == '__main__':
    import sys


    plot = True
    if len(sys.argv) != 4:
        print 'ERROR: you must define the number of pendula you want to track (1 or 2) and ' + \
                'whether or not you want to track a cart (true or false) and whether or not ' + \
                'plot and save the data (true or false)'
        sys.exit()
    elif sys.argv[1] not in ['1', '2']:
        print 'ERROR: the first argument must be the number of pendula you want to track (1 or 2)'
        sys.exit()
    elif sys.argv[2].lower() not in ['true', 'false']:
        print 'ERROR: the second argument must be whether or not you want to track a cart ' + \
                '(true or false)'
        sys.exit()
    elif sys.argv[3].lower() not in ['true', 'false']:
        print 'ERROR: the second argument must be whether or not you want to plot and save ' + \
                'the data (true or false)'
        sys.exit()
    else:
        num_pendula = int(sys.argv[1])
        print 'Number of pendula set to ' + str(num_pendula)
        if sys.argv[2].lower() == 'true': cart = True
        else: cart = False
        print 'Cart tracking set to ' + str(cart)
        if sys.argv[3].lower() == 'true': plot = True
        else: plot = False
        print 'Plotting set to ' + str(plot)

    detector = Detector(num_pendula, cart, plot)
    detector.hsv_filt()
