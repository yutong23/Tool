import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy
import numpy as np
import colorsys
from colorsys import hls_to_rgb
import pathlib
from PIL import Image

def start(num, ax):

    # csv_file_path = 'subj1/jun_self.csv'
    csv_file_path = str(pathlib.Path(__file__).parent.resolve()) + './'+str(num)+'.csv'
    val = []
    aro = []
    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                val.append(float(row[10]))
                aro.append(float(row[11]))
                line_count += 1
        print(f'Processed {line_count} lines.')

    store_postive = []
    store_negative = []
    ii = 0
    aro_copy = list(map(abs,aro))
    length = len(aro)
    while ii < 5:
        index1 = aro_copy.index(max(aro_copy))
        aro_copy[index1] = 0
        if index1 - 9 > 0:
            if aro[index1] > 0:
                for i in range(index1 - 9, index1):
                    store_postive.append(i)
                    aro_copy[i] = 0
            else:
                for i in range(index1 - 9, index1):
                    store_negative.append(i)
                    aro_copy[i] = 0
        else:
            if aro[index1] > 0:
                for i in range(0, index1):
                    store_postive.append(i)
                    aro_copy[i] = 0
            else:
                for i in range(0, index1):
                    store_negative.append(i)
                    aro_copy[i] = 0
        if index1 + 9 < length:
            if aro[index1] > 0:
                for i in range(index1, index1 + 9):
                    store_postive.append(i)
                    aro_copy[i] = 0
            else:
                for i in range(index1, index1 + 9):
                    store_negative.append(i)
                    aro_copy[i] = 0
        else:
            if aro[index1] > 0:
                for i in range(index1, length):
                    store_postive.append(i)
                    aro_copy[i] = 0
            else:
                for i in range(index1, length):
                    store_negative.append(i)
                    aro_copy[i] = 0
        ii = ii + 1

    
    
    # print(store_negative)
    # fig, ax = plt.subplots(figsize=(50,10))

    # ax.bar(store_negative, 1)
    # ax.grid()
    # ax.margins(0) # remove default margins (matplotlib verision 2+)

    # ax.axhspan(0, 0.01, facecolor='green', alpha=0.5)
    # ax.axhspan(0.01, 0.02, facecolor='yellow', alpha=0.5)
    # fig, ax = plt.subplots(figsize=(50,5))
    # ax.grid()
    # ax.margins(0) # remove default margins (matplotlib verision 2+)
    # ax.axhspan(0, 0.5, alpha=0.5, color='#1a1e33')
    # ax.axhspan(0.5, 1, facecolor='#4d5370', alpha=0.5)
    ax.bar(store_negative, 0.7, 1, color = '#8538C8',bottom=4)
    # ax.bar(store_postive, 0.7, 1, color = '#B8812E',bottom=4)
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
    # plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + './x_arousal' + '.png')

    # plt.figure(figsize=(50,10))
    # plt.bar(data_arousal, 1)
    # plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + './x_arousal' + '.png')
    # im = Image.open('./x_arousal' + '.png')
    # im1 = im.crop((621, 118, 4502, 894))
    # im1 = im1.save(str(pathlib.Path(__file__).parent.resolve()) + './x_arousal' + '.png')
    
