# adapted from lopezdeprado's impl.
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as pp
import statsmodels.nonparametric.smoothers_lowess as sml

class KCA:
    """ Kinetic Component Analysis
    """
    def __init__(self, fwd):
        self._fwd = fwd

    def fit(self, t, z):

    def transform(self):


def fitKCA(t,z,q,fwd=0):    
    '''
    Inputs:
        t: Iterable with time indices
        z: Iterable with measurements
        q: Scalar that multiplies the seed states covariance
        fwd: number of steps to forecast (optional, default=0)
    Output:
        x[0]: smoothed state means of position velocity and acceleration
        x[1]: smoothed state covar of position velocity and acceleration
    Dependencies: numpy, pykalman
    '''
    #1) Set up matrices A,H and a seed for Q
    h=(t[-1]-t[0])/t.shape[0]
    A=np.array([[1,h,.5*h**2],
                [0,1,h],
                [0,0,1]])
    Q=q*np.eye(A.shape[0])
    #2) Apply the filter    
    kf=KalmanFilter(transition_matrices=A,transition_covariance=Q)
    #3) EM estimates
    kf=kf.em(z)
    #4) Smooth
    x_mean,x_covar=kf.smooth(z)
    #5) Forecast
    for fwd_ in range(fwd):
        x_mean_,x_covar_=kf.filter_update(filtered_state_mean=x_mean[-1], \
            filtered_state_covariance=x_covar[-1])
        x_mean=np.append(x_mean,x_mean_.reshape(1,-1),axis=0)
        x_covar_=np.expand_dims(x_covar_,axis=0)
        x_covar=np.append(x_covar,x_covar_,axis=0)
    #6) Std series
    x_std=(x_covar[:,0,0]**.5).reshape(-1,1)
    for i in range(1,x_covar.shape[1]):
        x_std_=x_covar[:,i,i]**.5
        x_std=np.append(x_std,x_std_.reshape(-1,1),axis=1)
    return x_mean,x_std,x_covar


def selectFFT(series,minAlpha=None):
    # Implements a forward algorithm for selecting FFT frequencies
    #1) Initialize variables
    series_=series
    fftRes=np.fft.fft(series_,axis=0)
    fftRes={i:j[0] for i,j in zip(range(fftRes.shape[0]),fftRes)}
    fftOpt=np.zeros(series_.shape,dtype=complex)
    lags,crit=int(12*(series_.shape[0]/100.)**.25),None
    #2) Search forward
    while True:
        key,critOld=None,crit
        for key_ in fftRes.keys():
            fftOpt[key_,0]=fftRes[key_]
            series__=np.fft.ifft(fftOpt,axis=0)
            series__=np.real(series__)
            crit_=sm3.acorr_ljungbox(series_-series__,lags=lags) # test for the max # lags
            crit_=crit_[0][-1],crit_[1][-1]
            if crit==None or crit_[0]<crit[0]:crit,key=crit_,key_
            fftOpt[key_,0]=0
        if key!=None:
            fftOpt[key,0]=fftRes[key]
            del fftRes[key]
        else:break
        if minAlpha!=None:
            if crit[1]>minAlpha:break
            if critOld!=None and crit[0]/critOld[0]>1-minAlpha:break
    series_=np.fft.ifft(fftOpt,axis=0)
    series_=np.real(series_)
    out={'series':series_,'fft':fftOpt,'res':fftRes,'crit':crit}
    return out


def getPeriodic(periods,nobs,scale,seed=0):
    t=np.linspace(0,np.pi*periods/2.,nobs)
    rnd=np.random.RandomState(seed)
    signal=np.sin(t)
    z=signal+scale*rnd.randn(nobs)
    return t,signal,z


def vsFFT():
    #1) Set parameters
    nobs,periods=300,10
    #2) Get Periodic noisy measurements
    t,signal,z=getPeriodic(periods,nobs,scale=.5)
    #3) Fit KCA
    x_point,x_bands=kca.fitKCA(t,z,q=.001)[:2]
    #4) Plot KCA's point estimates
    color=['b','g','r']
    pp.plot(t,z,marker='x',linestyle='',label='measurements')
    pp.plot(t,x_point[:,0],marker='o',linestyle='-',label='position', \
        color=color[0])
    pp.plot(t,x_point[:,1],marker='o',linestyle='-',label='velocity', \
        color=color[1])
    pp.plot(t,x_point[:,2],marker='o',linestyle='-',label='acceleration', \
        color=color[2])
    pp.legend(loc='lower left',prop={'size':8})
    pp.savefig(mainPath+'Data/test/Figure1.png')
    #5) Plot KCA's confidence intervals (2 std)
    for i in range(x_bands.shape[1]):
        pp.plot(t,x_point[:,i]-2*x_bands[:,i],linestyle='-',color=color[i])
        pp.plot(t,x_point[:,i]+2*x_bands[:,i],linestyle='-',color=color[i])
    pp.legend(loc='lower left',prop={'size':8})
    pp.savefig(mainPath+'Data/test/Figure2.png')
    pp.clf();pp.close() # reset pylab
    #6) Plot comparison with FFT
    fft=selectFFT(z.reshape(-1,1),minAlpha=.05)
    pp.plot(t,signal,marker='x',linestyle='',label='Signal')
    pp.plot(t,x_point[:,0],marker='o',linestyle='-',label='KCA position')
    pp.plot(t,fft['series'],marker='o',linestyle='-',label='FFT position')
    pp.legend(loc='lower left',prop={'size':8})
    pp.savefig(mainPath+'Data/test/Figure3.png')
    return


# Kinetic Component Analysis of a periodic function
mainPath='../../'   
#---------------------------------------------------------
def vsLOWESS():
    # by MLdP on 02/24/2014 <lopezdeprado@lbl.gov>
    # Kinetic Component Analysis of a periodic function
    #1) Set parameters
    nobs,periods,frac=300,10,[.5,.25,.1]
    #2) Get Periodic noisy measurements
    t,signal,z=getPeriodic(periods,nobs,scale=.5)
    #3) Fit KCA
    x_point,x_bands=kca.fitKCA(t,z,q=.001)[:2]
    #4) Plot comparison with LOWESS
    pp.plot(t,z,marker='o',linestyle='',label='measurements')
    pp.plot(t,signal,marker='x',linestyle='',label='Signal')
    pp.plot(t,x_point[:,0],marker='o',linestyle='-',label='KCA position')
    for frac_ in frac:
        lowess=sml.lowess(z.flatten(),range(z.shape[0]),frac=frac_)[:,1].reshape(-1,1)
        pp.plot(t,lowess,marker='o',linestyle='-',label='LOWESS('+str(frac_)+')')
    pp.legend(loc='lower left',prop={'size':8})
    pp.savefig(mainPath+'Data/test/Figure4.png')
    return


