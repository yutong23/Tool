import change_detect_others
import change_detect_you
import arousal_you
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy
import numpy as np
import colorsys
from colorsys import hls_to_rgb
import pathlib
from PIL import Image
fig, ax = plt.subplots(figsize=(65,5))
ax.grid()
ax.margins(0) # remove default margins (matplotlib verision 2+)
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//1.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//2.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//3.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//4.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//5.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//6.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//7.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//8.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//9.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//10.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//11.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//12.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//13.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//14.csv")
change_detect_others.ind(25,1,1,ax,"C://Users//10736//Desktop//test//15.csv")
change_detect_you.ind(25,1,1,ax,"C://Users//10736//Desktop//test//1.csv")


plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + './x_left1and2' + '.png')
