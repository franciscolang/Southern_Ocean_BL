import numpy as np
import scipy.io as sio
import pandas as pd
from datetime import datetime
import os
import csv
from matplotlib.ticker import ScalarFormatter, MultipleLocator
import matplotlib.mlab as mlab
import scipy as sp
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as si
from scipy.interpolate import interp1d
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/cyclones/'
latMac=-54.50;
lonMac=158.95;

#*****************************************************************************\
# Readinf CSV
#*****************************************************************************\
df_maccyc= pd.read_csv(path_data + 'df_maccyc.csv', sep='\t', parse_dates=['Date'])
df_yotccyc= pd.read_csv(path_data + 'df_yotccyc.csv', sep='\t', parse_dates=['Date'])
df_mycyc= pd.read_csv(path_data + 'df_mycyc.csv', sep='\t', parse_dates=['Date'])

df_maccyc= df_maccyc.set_index('Date')
df_yotccyc= df_yotccyc.set_index('Date')
df_mycyc= df_mycyc.set_index('Date')

#*****************************************************************************\
#MAC
#*****************************************************************************\
#Extraer Category y coincidir con Lows
df_BL1 = df_maccyc[df_maccyc['Clas']==4]
df_BL = df_BL1[np.isfinite(df_BL1['dy'])]

df_DL1 = df_maccyc[df_maccyc['Clas']==3]
df_DL = df_DL1[np.isfinite(df_DL1['dy'])]

df_SI1 = df_maccyc[df_maccyc['Clas']==2]
df_SI = df_SI1[np.isfinite(df_SI1['dy'])]

df_NI1 = df_maccyc[df_maccyc['Clas']==1]
df_NI = df_NI1[np.isfinite(df_NI1['dy'])]

df_1i= df_maccyc[np.isfinite(df_maccyc['1ra Inv'])]
df_1inv = df_1i[np.isfinite(df_1i['dy'])]

df_2i= df_maccyc[np.isfinite(df_maccyc['2da Inv'])]
df_2inv = df_2i[np.isfinite(df_2i['dy'])]
#*****************************************************************************\
#YOTC
#*****************************************************************************\
#Extraer Category y coincidir con Lows
df_BL1y = df_yotccyc[df_yotccyc['Clas']==4]
df_BLy = df_BL1y[np.isfinite(df_BL1y['dy'])]

df_DL1y = df_yotccyc[df_yotccyc['Clas']==3]
df_DLy = df_DL1y[np.isfinite(df_DL1y['dy'])]

df_SI1y = df_yotccyc[df_yotccyc['Clas']==2]
df_SIy = df_SI1y[np.isfinite(df_SI1y['dy'])]

df_NI1y = df_yotccyc[df_yotccyc['Clas']==1]
df_NIy = df_NI1y[np.isfinite(df_NI1y['dy'])]

df_1iy= df_yotccyc[np.isfinite(df_yotccyc['1ra Inv'])]
df_1invy = df_1iy[np.isfinite(df_1iy['dy'])]

df_2iy= df_yotccyc[np.isfinite(df_yotccyc['2da Inv'])]
df_2invy = df_2iy[np.isfinite(df_2iy['dy'])]

#*****************************************************************************\
#MAC levels YOTC
#*****************************************************************************\
#Extraer Category y coincidir con Lows
df_BL1my = df_mycyc[df_mycyc['Clas']==4]
df_BLmy = df_BL1my[np.isfinite(df_BL1my['dy'])]

df_DL1my = df_mycyc[df_mycyc['Clas']==3]
df_DLmy = df_DL1my[np.isfinite(df_DL1my['dy'])]

df_SI1my = df_mycyc[df_mycyc['Clas']==2]
df_SImy = df_SI1my[np.isfinite(df_SI1my['dy'])]

df_NI1my = df_mycyc[df_mycyc['Clas']==1]
df_NImy = df_NI1my[np.isfinite(df_NI1my['dy'])]

df_1imy= df_mycyc[np.isfinite(df_mycyc['1ra Inv'])]
df_1invmy = df_1imy[np.isfinite(df_1imy['dy'])]

df_2imy= df_mycyc[np.isfinite(df_mycyc['2da Inv'])]
df_2invmy = df_2imy[np.isfinite(df_2imy['dy'])]


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                       Creating mapa Height
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Main Inversion
#*****************************************************************************\
Lat1=np.array(df_1inv['dy'])
Lon1=np.array(df_1inv['dx'])
Data1 = np.array(df_1inv['1ra Inv'])

Lat2=np.array(df_1invy['dy'])
Lon2=np.array(df_1invy['dx'])
Data2 = np.array(df_1invy['1ra Inv'])

Lat3=np.array(df_1invmy['dy'])
Lon3=np.array(df_1invmy['dx'])
Data3 = np.array(df_1invmy['1ra Inv'])

#*****************************************************************************\


zi, yi, xi = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data1, normed=False)
counts, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
zi = zi / counts
zi = np.ma.masked_invalid(zi)

zi1, yi1, xi1 = np.histogram2d(Lat2, Lon2, bins=(30,30), weights=Data2, normed=False)
counts1, _, _ = np.histogram2d(Lat2, Lon2, bins=(30,30))
zi1 = zi1 / counts1
zi1 = np.ma.masked_invalid(zi1)

zi2, yi2, xi2 = np.histogram2d(Lat3, Lon3, bins=(30,30), weights=Data3, normed=False)
counts2, _, _ = np.histogram2d(Lat3, Lon3, bins=(30,30))
zi2 = zi2 / counts2
zi2 = np.ma.masked_invalid(zi2)
#*****************************************************************************\
row = 1
column = 3
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(17,5))
ax0, ax1, ax2 = axes.flat
cmap='Blues'
vmax=2400
limu=vmax+300
v = np.arange(0, limu, 300) #Crea barra colores
#*****************************************************************************\

#img1= ax0.pcolormesh(xi, yi, zi, cmap=cmap,vmin=0, vmax=vmax)

img1=ax0.imshow(zi, interpolation='bicubic', cmap=cmap)

    # div = make_axes_locatable(ax0)
# ax0.set_title('MAC', size=18)
# ax0.set_yticks([0], minor=True)
# ax0.set_xticks([0], minor=True)
# ax0.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
# ax0.grid(b=True, which='major', color='grey', linestyle='--')
# ax0.set_ylabel('Distance from low (deg)', size=18)
# ax0.set_xlabel('Distance from low (deg)', size=18)
# ax0.margins(0.05)
#*****************************************************************************\
img2= ax1.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax1)
ax1.set_title('YOTC', size=18)
ax1.set_yticks([0], minor=True)
ax1.set_xticks([0], minor=True)
ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax1.grid(b=True, which='major', color='grey', linestyle='--')
ax1.set_xlabel('Distance from low (deg)', size=18)
ax1.margins(0.05)
#*****************************************************************************\


img3= ax2.pcolor(xi2, yi2, zi2, cmap=cmap,vmin=0, vmax=vmax)
ax2.set_title('MAC$_{AVE}$', size=18)
div = make_axes_locatable(ax2)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img3, cax=cax, format="%.0f",ticks=v)
cbar.set_label(' height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax2.set_yticks([0], minor=True)
ax2.set_xticks([0], minor=True)
ax2.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax2.grid(b=True, which='major', color='grey', linestyle='--')
ax2.set_xlabel('Distance from low (deg)', size=18)
ax2.margins(0.05)
plt.tight_layout()

plt.show()


