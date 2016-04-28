import numpy as np
import scipy.io as sio
import os
from pylab import plot,show, grid
import math
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
#*****************************************************************************\
#Default Info
latMac=-54.50;
lonMac=158.95;

#*****************************************************************************\
#Reading CSV
#*****************************************************************************\
df_front= pd.read_csv(path_data + 'df_front.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')
#*****************************************************************************\
#df_mac= pd.read_csv(path_data + 'df_mac_final.csv', sep='\t', parse_dates=['Date'])
df_mac= pd.read_csv(path_data + 'df_yotc_all.csv', sep='\t', parse_dates=['Date'])
df_mac= df_mac.set_index('Date')
#*****************************************************************************\
#Merge datraframe mac with
df_macfro=pd.concat([df_mac, df_front],axis=1)

#*****************************************************************************\
#Clasification by type
#*****************************************************************************\
df_BL1 = df_macfro[df_macfro['Clas']==4]
df_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

df_DL1 = df_macfro[df_macfro['Clas']==3]
df_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

df_SI1 = df_macfro[df_macfro['Clas']==2]
df_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

df_NI1 = df_macfro[df_macfro['Clas']==1]
df_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

df_1i= df_macfro[np.isfinite(df_macfro['1ra Inv'])]
df_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

df_2i= df_macfro[np.isfinite(df_macfro['2da Inv'])]
df_2inv = df_2i[np.isfinite(df_2i['Dist Front'])]
#*****************************************************************************\
#*****************************************************************************\
#Graphs
#*****************************************************************************\
#*****************************************************************************\
colorh='#2929a3'
colorb='blue'
figsize3=(7.5, 5)
fsize0=12
fsize1=14
fsize2=16
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts/'
#*****************************************************************************\
#Histogram by type MAC
#*****************************************************************************\
row = 2
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2, ax3, ax4= axes.flat
x1=-15
x2=15
y1=0
y2=0.2
dx=1
bx=np.arange(x1,x2+1,3)
by=np.arange(y1,y2+0.02,0.02)
bins = np.arange(x1, x2+1, dx)

df1=15
df2=30
width = 1

#*****************************************************************************\
x=np.array(df_NI['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax1.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color='#c6c9d0', label='before')
ax1.bar(pos,n[df1:df2], width,alpha=0.5, color='#67832F', label='after')

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_title('No inversion',fontsize = 14 ,fontweight='bold')
ax1.grid()


#*****************************************************************************\
x=np.array(df_SI['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax2.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color='#c6c9d0', label='before')
ax2.bar(pos,n[df1:df2], width,alpha=0.5, color='#67832F', label='after')

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_xlim([x1,x2])
ax2.set_ylim([y1,y2])
ax2.legend(loc='upper left')
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_title('Single inversion',fontsize = 14 ,fontweight='bold')
ax2.grid()



#*****************************************************************************\

x=np.array(df_DL['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax3.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color='#c6c9d0', label='before')
ax3.bar(pos,n[df1:df2], width,alpha=0.5, color='#67832F', label='after')

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_xlim([x1,x2])
ax3.set_ylim([y1,y2])
ax3.legend(loc='upper left')
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_title('Decoupled layer',fontsize = 14 ,fontweight='bold')
ax3.grid()


#*****************************************************************************\
x=np.array(df_BL['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax4.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color='#c6c9d0', label='before')
ax4.bar(pos,n[df1:df2], width,alpha=0.5, color='#67832F', label='after')

ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_yticks(by)
ax4.set_xlim([x1,x2])
ax4.set_ylim([y1,y2])
ax4.legend(loc='upper left')
ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
ax4.set_title('Buffer layer',fontsize = 14 ,fontweight='bold')
ax4.grid()

fig.savefig(path_data + 'hist.eps', format='eps', dpi=1200)
plt.show()
