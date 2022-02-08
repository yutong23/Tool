import matplotlib.pyplot as plt
import csv
import matplotlib.ticker as ticker
import numpy as np
import scipy
import colorsys
from colorsys import hls_to_rgb
import pathlib
from PIL import Image
from scipy import log
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score


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

def func(x, a, b):
    #    y = a * log(x) + b
    y = x/(a*x+b)
    return y
if __name__ == '__main__':

    # csv_file_path = 'subj1/jun_self.csv'
    paths = []
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/1.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/2.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/3.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/5.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/6.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/7.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/8.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/9.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/10.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/11.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/12.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/13.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/14.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/16.csv')
    paths.append(str(pathlib.Path(__file__).parent.resolve()) + '/17.csv')
    
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    curr = []
    m = []
    for path in paths:
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    curr.append(float(row[11]))
                    line_count += 1
            curr = curr - np.mean(curr)
            curr = smooth(curr, 11, windows[0])
            m.append(sum(curr)/len(curr))
            curr = []
            print(len(m))
            
    pos = [0]*6000
    value = []
    index = 0
    for path in paths:
        ele = m[index]
        index = index + 1
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    value.append(float(row[11]))
                    line_count += 1
            value = value - np.mean(value)
            value = smooth(value, 11, windows[0])
            for i in range(len(value)):
                if value[i] > ele:
                    pos[i] = pos[i] + 1
            value = []
    
    pos = [x/15 for x in pos]
    pos = pos[0:5400]
    fig, ax = plt.subplots(figsize=(50,10))
    plt.xlabel('time',fontsize=18)
    plt.ylabel('arousal-high engagament',fontsize=18)
    
    parameter = np.polyfit(range(5400), pos, 42)
    p = np.poly1d(parameter)
    
    ax.scatter(range(5400), pos, s=4, color='b', label='Data points')
    plt.plot(range(5400), p(range(5400)), color='black')
    
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.xticks(rotation=25)
    plt.legend(loc='upper right',fontsize=20)
    

    plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + '/arousal.png')
    