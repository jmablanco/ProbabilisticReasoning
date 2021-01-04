""" Kalman filtering """

import numpy as np
from filterpy.kalman import KalmanFilter as kalmanfilter

class KalmanFilter:
    def __init__(self, variance, width):
        """
        Constructs a kalman filter
        variance: variance of gaussian GPS noise
        width: width of uniform GPS noise
        """
        self.kf = kalmanfilter(dim_x=4, dim_z=2)
        self.variance = variance
        self.width = width
        # BEGIN_YOUR_CODE ######################################################
        raise NotImplementedError

        # END_YOUR_CODE ########################################################

    def predict_and_update(self, measurement, which="gaussian"):
        """
        Returns the state after predicting and updating
        measurement: GPS measurement
        which: gaussian or uniform
        """
        # BEGIN_YOUR_CODE ######################################################
        raise NotImplementedError
        
        # END_YOUR_CODE ########################################################
        return self.kf.x
