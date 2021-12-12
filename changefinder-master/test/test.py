# -*- coding: utf-8 -*-
import changefinder
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import csv
from numpy.core.fromnumeric import partition
import scipy
from colorsys import hls_to_rgb
from PIL import Image
import matplotlib.ticker as ticker
def transfer_space(x, y):
    '''
    Input: coordinate of x, and y
    Output: Intensity and Angle
    '''
    vec = [x, y]
    sign = 1
    if(y < 0):
        sign = -1
    intensity = sign*np.sqrt(x * x + y * y)
    e_vec = np.array(vec) / intensity
    e_x = np.array([1, 0])
    angle = np.arccos(np.dot(e_vec, e_x))
    if y < 0:  # below the x axis
        angle = 2 * np.pi - angle
    norm_angle = angle / (np.pi * 2)  # 0-1
    # print(norm_angle * 360)
    return intensity, norm_angle


def convert_color_vector(na):
    '''
    Input: Norm Angle
    Output: RGB value of the angle for plotting
    '''
    rgb = hls_to_rgb(na, 0.7, 0.7)
    # print(rgb)
    return rgb


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # if x.ndim != 1:
    #    raise ValueError, "smooth only accepts 1 dimension arrays."

    # if x.size < window_len:
    #    raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

class TestChangeFinder():
    def setup(self):
        self._term = 30
        self._smooth = 7
        self._order = 1
        self._arima_order = (1, 0, 0)
        csv_file_path = str(pathlib.Path(__file__).parent.resolve()) + '/' + str(10) + '.csv'
        val = []
        aro = []
        with open(csv_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    val.append(float(row[10]))
                    aro.append(float(row[11]))
                    line_count += 1
        windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
        val_copy = val
        aro_copy = aro
        val = val - np.mean(val)
        aro = aro - np.mean(aro)
        sval = smooth(val, 12, windows[0])
        saro = smooth(aro, 12, windows[0])
        intensity = []
        color = []
        for i in range(sval.shape[0]):
            inte, ang = transfer_space(sval[i], saro[i])
            intensity.append(inte)
            color.append(convert_color_vector(ang))
        self._data = np.concatenate([val_copy[0:500], val_copy[500: 2000], val_copy[2000:50000]])
        
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(50,10)) 
        f.subplots_adjust(hspace=0.4) 
        ax1.scatter(np.arange(sval.shape[0]),intensity, c=color, s=150)
        ax1.set_title("data point") 
        cf = changefinder. ChangeFinder()
        score = [cf.update(p) for p in self._data] 
        ax2.plot(score) 
        ax2.set_title("second") 
         
        # plt.scatter(np.arange(sval.shape[0]),intensity, c=color, s=150)
        # plt.subplots_adjust(hspace=0.4)
        # plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + '/sample.png')
        # plt.figure(figsize=(50,10))  
        # cf = changefinder.ChangeFinder()
        # scores = [cf.update(p) for p in self._data]
        # plt.plot(scores)
        plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + '/output.png')

    def test_changefinder(self):
        cf = changefinder.ChangeFinder(r=0.1, order=self._order, smooth=self._smooth)
        for i in self._data:
            cf.update(i)

    def test_changefinderarima(self):
        cf = changefinder.ChangeFinderARIMA(term=self._term, smooth=self._smooth, order=self._arima_order)
        for i in self._data:
            cf.update(i)
            
TestChangeFinder.setup(TestChangeFinder)