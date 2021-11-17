#loading pkgs
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np

#qualified participants
participant_ids=[i for i in range(1,18)]
participant_ids.remove(4)
participant_ids.remove(15)

#collective data processing
#since participant `4` dont have data and `15` quit the study, here we need to remove them
#Combine all previous data into a big dataframe
def pre_collective(participant_ids,keyword):
    co_data=pd.DataFrame()
    for id_ in participant_ids:
        temp_data=pd.read_csv("collective_data/P"+str(id_)+".csv",usecols=["frames_ids","valence","arousal"],nrows=5000)
        co_data=pd.concat([co_data, temp_data], axis=1)
    up_line=np.mean(co_data[keyword], axis=1)+np.std(co_data[keyword], axis=1)
    down_line=np.mean(co_data[keyword], axis=1)-np.std(co_data[keyword], axis=1)
    df_1=pd.DataFrame(up_line,columns=[keyword+"_upline"])
    df_2=pd.DataFrame(down_line,columns=[keyword+"_downline"])
    df_up_down=pd.concat([df_1, df_2], axis=1)
    return(df_up_down)

pre_collective(participant_ids,"valence")

#individual visualization
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy
import numpy as np
import ruptures as rpt
import colorsys
from colorsys import hls_to_rgb
import pathlib
from PIL import Image



def plot_change_points(ts,ts_change_loc, color, intensity, penalty, participant_id,keyword):


    fig, ax = plt.subplots(figsize = (50,10))
    ax.set_title("Participant_" +str(participant_id)+"_Penalty_"+str(penalty)+"_"+keyword,fontsize=40,color='black')
#     print(color.shape())
#     print(intensity.shape())
    plt.scatter(np.arange(ts.shape[0]),intensity, c=color, s=150)
    #plt.plot(intensity)
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    plt.xlim([0,ts.shape[0]])
    plt.draw()

    # use the list index can help us find the block value
#     print(ts_change_loc)
    
    my_label1="postive to negative"
    my_label2="negative to positive"
    my_label3="other changes"
    for change_id in range(len(ts_change_loc)):
        x=ts_change_loc[change_id]
        
        if x!=0 and x!=len(ts):
            if change_id==0:
                pre_cluster=sum(ts[0:x])
            else:
                pre_x=ts_change_loc[change_id-1]
                pre_cluster=sum(ts[pre_x:x])
            
            if change_id==len(ts_change_loc):
                after_cluster=sum(ts[x:])
            else:
                after_x=ts_change_loc[change_id+1]
                after_cluster=sum(ts[x:after_x])  
            
            if pre_cluster > 0 and after_cluster < 0 :
                plt.axvline(ts_change_loc[change_id],lw=5, color="red", label = my_label1 )
                my_label1= "_nolegend_"
            elif pre_cluster < 0 and after_cluster > 0 :
                plt.axvline(ts_change_loc[change_id],lw=5, color="green", label = my_label2)
                my_label2= "_nolegend_"
#             else:
#                 plt.axvline(ts_change_loc[change_id],lw=5, color="blue", label = my_label3)
#                 my_label3= "_nolegend_"
    plt.legend(loc='upper right',fontsize=20)
    ax.set_xticks(ts_change_loc)
    ax.set_xticklabels(ts_change_loc, rotation=45, fontsize=20)
#     pre = 0
#     pre_block = 0
#     count_block = 1
#     color_line = 'grey'
#     pre_x = ts_change_loc[0]
#     legend1 = True
#     legend2 = True
#     for x in ts_change_loc:
#         count = 0
#         for t in ts[pre: x]:
#             count += t
#         block = count/(x - pre)
#         # print(block)
#         if pre_block > 0 and block < 0:
# #             print("postive to negative")
#             color_line = 'red'
# #             print(count_block)
#         if pre_block < 0 and block > 0:
# #             print("negative to postive")
#             color_line = 'green'
# #             print(count_block)
#         pre_block = block
#         pre = x
#         count_block = count_block + 1
#         if color_line == 'red'  and legend1:
#             plt.axvline(pre_x,lw=5, color=color_line, label = "postive to negative")
#             legend1 = False
#         if color_line == 'green' and legend2:
#             plt.axvline(pre_x,lw=5, color=color_line, label = "negative to postive")
#             legend2 = False
#         else:
#             plt.axvline(pre_x,lw=5, ls="--",color=color_line)
#         pre_x = x
#         color_line = 'grey'
    
#     ax.set_xticks(ts_change_loc)
#     ax.set_xticklabels(ts_change_loc, rotation=45, fontsize=15)
#     plt.legend(loc='upper right',markerscale=2.,numpoints=2,scatterpoints=1,fontsize=20)
#     plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + './output1.png')
#     im = Image.open("./output1.png")
    # im1 = im.crop((621, 118, 4502, 894))
    # im1 = im1.save(str(pathlib.Path(__file__).parent.resolve()) + "./output1.png")
def transfer_space(x, y):
    '''
    Input: coordinate of x, and y
    Output: Intensity and Angle
    '''
    sign = 1
    if(y < 0):
        sign = -1
    intensity = sign*np.sqrt(x * x + y * y)
    vec = [x, y]
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


def smooth(x, window_len=12, window='hanning'):
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
    return (y[int(window_len/2-1):-int(window_len/2)])
    
# csv_file_path = 'subj1/jun_self.csv'

def individual(participant_ids,id_,penalty,keyword):
    csv_file_path = "/Users/yilinxia/Jupyter/Mirror/collective_data/P"+str(id_)+".csv"
    val = []
    aro = []
    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
    #             print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                val.append(float(row[10]))
                aro.append(float(row[11]))
                line_count += 1
    #     print(f'Processed {line_count} lines.')

    #plt.figure(figsize=(50,10))
    #plt.plot(val)

    #plt.figure(figsize=(50,10))
    #plt.plot(aro)


    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    #plt.figure()
    #plt.hist(val, bins=100)

    #plt.figure()
    #plt.hist(aro, bins=100)

    # print(np.mean(val), np.mean(aro))
    val = val - np.mean(val)   # to make the points much closer
    aro = aro - np.mean(aro)
    sval = smooth(val, 12, windows[0])  # to ensure the input and output have the same numbe of points
    saro = smooth(aro, 12, windows[0])
    intensity = []
    color = []
    for i in range(sval.shape[0]):
        inte, ang = transfer_space(sval[i], saro[i])
        intensity.append(inte)
        #print(convert_color_vector(ang))
        color.append(convert_color_vector(ang))

    if keyword=="Valence":
        # Detect the change points for valence
        algo1 = rpt.Pelt(model="rbf").fit(sval)
        change_location1 = algo1.predict(pen=penalty)
        # Plot the change points:
        plot_change_points(sval,change_location1, color, intensity, penalty, id_,keyword)
    else:
        # Detect the change points for arounsal
        algo1 = rpt.Pelt(model="rbf").fit(saro)
        change_location1 = algo1.predict(pen=penalty)
        # Plot the change points:
        plot_change_points(saro,change_location1, color, intensity, penalty, id_,keyword)
penalty=25
id_=1
individual(participant_ids,id_,penalty,"Arousal")
pre_collective(participant_ids,"valence")
