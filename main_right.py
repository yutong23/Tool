import arousal_others
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
arousal_others.start(1, ax)
arousal_others.start(2, ax)
arousal_others.start(3, ax)
arousal_others.start(4, ax)
arousal_others.start(5, ax)
arousal_others.start(6, ax)
arousal_others.start(7, ax)
arousal_others.start(8, ax)
arousal_others.start(9, ax)
arousal_others.start(10, ax)
arousal_others.start(11, ax)
arousal_others.start(12, ax)
arousal_others.start(13, ax)
arousal_others.start(14, ax)
arousal_others.start(15, ax)
arousal_you.start(2,ax)


plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + './x_right1and2' + '.png')
