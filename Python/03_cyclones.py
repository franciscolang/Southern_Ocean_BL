import numpy as np
import scipy.io as sio
from datetime import datetime, timedelta
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator
import matplotlib.mlab as mlab
import scipy as sp
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as si
from scipy.interpolate import interp1d

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'


#*****************************************************************************\
# Readinf CSV
df_macyotc_final= pd.read_csv(path_data + 'df_macyotc_final.csv', sep='\t')
df_mac_final= pd.read_csv(path_data + 'df_mac_final.csv', sep='\t')
df_yotc_all= pd.read_csv(path_data + 'df_yotc_all.csv', sep='\t')
df_yotc= pd.read_csv(path_data + 'df_yotc.csv', sep='\t')
#*****************************************************************************\
yot_clas=np.array(df_yotc['Clas'])
mac_clas=np.array(df_mac_final['Clas'])
macyotc_clas=np.array(df_macyotc_final['Clas'])
ny, bin_edgesy =np.histogram(yot_clas, bins=[1, 2, 3, 4,5],normed=1)
nm, bin_edgesm =np.histogram(mac_clas, bins=[1, 2, 3, 4,5],normed=1)
nmy, bin_edgesmy =np.histogram(macyotc_clas, bins=[1, 2, 3, 4,5],normed=1)

#*****************************************************************************\

NI=[nm[0],  ny[0],  nmy[0]]
SI=[nm[1],  ny[1],  nmy[1]]
DL=[nm[2],  ny[2],  nmy[2]]
BL=[nm[3],  ny[3],  nmy[3]]

#*****************************************************************************\

raw_data = {'tipo': ['MAC', 'YOTC', 'MACyotc'],
        'NI': NI,
        'SI': SI,
        'DL': DL,
        'BL': BL}

df = pd.DataFrame(raw_data, columns = ['tipo', 'NI', 'SI', 'DL', 'BL'])

# Create the general blog and the "subplots" i.e. the bars
f, ax1 = plt.subplots(1, figsize=(10,6))

# Set the bar width
bar_width = 0.5

# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df['NI']))]

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l]

# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the pre_score data
        df['NI'],
        # set the width
        width=bar_width,
        # with the label pre score
        label='NI',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#BFEFFF')

# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the mid_score data
        df['SI'],
        # set the width
        width=bar_width,
        # with pre_score on the bottom
        bottom=df['NI'],
        # with the label mid score
        label='SI',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#60AFFE')

# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the post_score data
        df['DL'],
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[i+j for i,j in zip(df['NI'],df['SI'])],
        # with the label post score
        label='DL',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#0276FD')

# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the post_score data
        df['BL'],
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[i+j+k for i,j,k in zip(df['NI'],df['SI'],df['DL'])],
        # with the label post score
        label='BL',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#26466D')

# set the x ticks with names
plt.xticks(tick_pos, df['tipo'])

# Set the label and legends
ax1.set_ylabel("Absolute Percentage Occurrence")
#ax1.set_xlabel("Soundings")
ax1.set_title('Boundary Layer Categories')
#plt.legend(loc='upper left')
#ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 1.05),
#          ncol=1, fancybox=True, shadow=True)
#ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)

# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
#plt.show()

#*****************************************************************************\

H = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])  # added some commas and array creation code

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()
