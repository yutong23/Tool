#loading pkgs
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import csv
import scipy
import numpy as np
import ruptures as rpt
from colorsys import hls_to_rgb
import pathlib
from PIL import Image



def plot_change_points(ts,ts_change_loc, color, intensity, penalty, participant_id,keyword,ax):
    
    positive_negative_id = []
    negative_positive_id = []
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
                # plt.axvline(ts_change_loc[change_id],lw=5, color="red", label = my_label1 )
                positive_negative_id.append(change_id)
                my_label1= "_nolegend_"
            elif pre_cluster < 0 and after_cluster > 0 :
                # plt.axvline(ts_change_loc[change_id],lw=5, color="green", label = my_label2)
                negative_positive_id.append(change_id)
                my_label2= "_nolegend_"
            # else: plt.axvline(ts_change_loc[change_id],lw=5, color="grey", label = my_label2)
    
    positive_negative = []
    negative_positive = []
    # for i in positive_negative_id:
    #     for ii in range(ts_change_loc[i - 1], ts_change_loc[i + 1]):
    #         positive_negative.append(ii)
    # for i in negative_positive_id:
    #     for ii in range(ts_change_loc[i - 1], ts_change_loc[i + 1]):
    #         negative_positive.append(ii)

    for i in positive_negative_id:
        positive_negative.append(ts_change_loc[i])
    for i in negative_positive_id:
        negative_positive.append(ts_change_loc[i])
    
    ax.bar(negative_positive, 0.7, 1, color = '#FF0000',bottom=4)
    ax.bar(positive_negative, 0.7, 1, color = '#005DFF',bottom=4)
    xx = []
    for i in range(5400):
        xx.append(i)
    ax.fill_between(xx, 0, 0.3, color = '#2f3659')
    ax.fill_between(xx, 0.3, 1, color = '#1a1e33')
    ax.fill_between(xx, 1, 4, color = '#2f3659')
    ax.fill_between(xx, 4, 4.7, color = '#1a1e33')
    ax.fill_between(xx, 4.7, 5, color = '#2f3659')
    plt.ylim((0, 5))
    plt.axis('off')   
    plt.xticks([]) 
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


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




def individual(participant_ids,id_,penalty,keyword, ax, csv_file_path):
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
   
    val = val - np.mean(val)   # to make the points much closer
    aro = aro - np.mean(aro)
    # sval = smooth(val, 12, windows[0])  # to ensure the input and output have the same numbe of points
    # saro = smooth(aro, 12, windows[0])
    intensity = []
    color = []
    for i in range(len(val)):
        inte, ang = transfer_space(val[i], aro[i])
        intensity.append(inte)
        #print(convert_color_vector(ang))
        color.append(convert_color_vector(ang))
    
    if keyword=="Valence":
        # Detect the change points for valence
        algo1 = rpt.Pelt(model="rbf").fit(np.array(val))
        change_location1 = algo1.predict(pen=penalty)
        # Plot the change points:
        plot_change_points(val,change_location1, color, intensity, penalty, id_,keyword, ax)
    else:
        # Detect the change points for arounsal
        algo1 = rpt.Pelt(model="rbf").fit(np.array(aro))
        change_location1 = algo1.predict(pen=penalty)
        # Plot the change points:
        plot_change_points(aro,change_location1, color, intensity, penalty, id_,keyword, ax)

def ind(penalty, id_, participant_ids, ax, path):
    individual(participant_ids,id_,penalty,"Valence", ax, path)
