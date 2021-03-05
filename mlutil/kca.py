# adapted from lopezdeprado's impl.
import numpy as np
from pykalman import KalmanFilter

class KCA:
    """ Kinetic Component Analysis
    """
    def __init__(self):
        self._fitted = False

    def fit(self, t, z, q):
        """
        Inputs:
            t: Iterable with time indices
            z: Iterable with measurements
            q: Scalar that multiplies the seed states covariance
       """
 
        # Set up matrices A,H and a seed for Q
        h = (t[-1]-t[0])/t.shape[0]
        A = np.array([[1,h,.5*h**2],
                    [0,1,h],
                    [0,0,1]])
        Q = q*np.eye(A.shape[0])
        # Apply the filter    
        self._kf = KalmanFilter(transition_matrices=A, transition_covariance=Q)
        # EM estimates
        self._kf=kf.em(z)
        
        self._fitted = True


    def predict(self, z, fwd=1):
        """
        Inputs: 
            same with fit.
        Output:
            x[0]: smoothed state means of position velocity and acceleration
            x[1]: smoothed state covar of position velocity and acceleration
        """
        if not self._fitted:
            raise Exception('The KCA model is not fit yet.')
        # Smooth
        x_mean, x_covar = self._kf.smooth(z)
        # Forecast
        for fwd_ in range(fwd):
            x_mean_,x_covar_ = self._kf.filter_update(filtered_state_mean=x_mean[-1], \
                filtered_state_covariance=x_covar[-1])
            x_mean = np.append(x_mean,x_mean_.reshape(1,-1),axis=0)
            x_covar_ = np.expand_dims(x_covar_,axis=0)
            x_covar = np.append(x_covar,x_covar_,axis=0)
        # Standize series
        x_std =(x_covar[:,0,0]**.5).reshape(-1,1)
        for i in range(1,x_covar.shape[1]):
            x_std_ = x_covar[:,i,i]**.5
            x_std = np.append(x_std,x_std_.reshape(-1,1),axis=1)
        return x_mean, x_std, x_covar


    def fit_predict(self, t, z, q, fwd):
        """
        Inputs: 
            same with fit.
        Output:
            same with predict.
        """
        # Fit the KCA
        self.fit(t, z, q)
        # predict
        return self.predict(z, fwd)



