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

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'

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
#Percentages
#*****************************************************************************\

Tmac=len(df_BL)+len(df_DL)+len(df_SI)+len(df_NI)
Tyotc=len(df_BLy)+len(df_DLy)+len(df_SIy)+len(df_NIy)
Tmy=len(df_BLmy)+len(df_DLmy)+len(df_SImy)+len(df_NImy)

perc_BL=len(df_BL)/float(Tmac)
perc_DL=len(df_DL)/float(Tmac)
perc_SI=len(df_SI)/float(Tmac)
perc_NI=len(df_NI)/float(Tmac)

perc_BLy=len(df_BLy)/float(Tyotc)
perc_DLy=len(df_DLy)/float(Tyotc)
perc_SIy=len(df_SIy)/float(Tyotc)
perc_NIy=len(df_NIy)/float(Tyotc)

perc_BLmy=len(df_BLmy)/float(Tmy)
perc_DLmy=len(df_DLmy)/float(Tmy)
perc_SImy=len(df_SImy)/float(Tmy)
perc_NImy=len(df_NImy)/float(Tmy)

Tmac1=len(df_BL1)+len(df_DL1)+len(df_SI1)+len(df_NI1)
Tyotc1=len(df_BL1y)+len(df_DL1y)+len(df_SI1y)+len(df_NI1y)

perc_maccyc=Tmac/float(Tmac1)
perc_yotccyc=Tyotc/float(Tyotc1)

print perc_maccyc, perc_yotccyc

#*****************************************************************************\
#*****************************************************************************\

Lat1=np.array(df_BLmy['dy'])
Lon1=np.array(df_BLmy['dx'])
Data1=np.ones(len(df_BLmy))

Lat2=np.array(df_DLmy['dy'])
Lon2=np.array(df_DLmy['dx'])
Data2=np.ones(len(df_DLmy))

Lat3=np.array(df_SImy['dy'])
Lon3=np.array(df_SImy['dx'])
Data3=np.ones(len(df_SImy))

Lat4=np.array(df_NImy['dy'])
Lon4=np.array(df_NImy['dx'])
Data4=np.ones(len(df_NImy))

zi=np.empty([30,30,4])
xi=np.empty([31,4])
yi=np.empty([31,4])


# zi, yi, xi = np.histogram2d(Lat2, Lon2, bins=(30,30), weights=Data2, normed=True)
zi[:,:,0], yi[:,0], xi[:,0] = np.histogram2d(Lat4, Lon4, bins=(30,30), weights=Data4, normed=True)

zi[:,:,1], yi[:,1], xi[:,1] = np.histogram2d(Lat3, Lon3, bins=(30,30), weights=Data3, normed=True)

zi[:,:,2], yi[:,2], xi[:,2] = np.histogram2d(Lat2, Lon2, bins=(30,30), weights=Data2, normed=True)

zi[:,:,3], yi[:,3], xi[:,3] = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data1, normed=True)

titulo=['No Inv.','Single Inv.','Decoupled L.','Buffer L.']

#*****************************************************************************\
#Creating mapa Frequency Subplot
#*****************************************************************************\

from mpl_toolkits.axes_grid1 import make_axes_locatable

# row = 1
# column = 2
# fig, ax = plt.subplots(row, column, facecolor='w', figsize=(18,4))

# #fig.suptitle('Title of fig', fontsize=24)

# for i, ax in enumerate(ax.flat, start=1):
#   print i
#   #ax.set_title('Subplot {}'.format(i))
#   #ax.set_title(titulo[i-1])

#   # plots random data using the 'jet' colormap
#   #img = ax.contourf(np.random.rand(10,10),20,cmap=plt.cm.jet)
#   img = ax.pcolor(xi[:,i-1], yi[:,i-1], zi[:,:,i-1], cmap='BuGn')
#   # creates a new axis, cax, located 0.05 inches to the right of ax, whose width is 15% of ax
#   # cax is used to plot a colorbar for each subplot
#   div = make_axes_locatable(ax)
#   cax = div.append_axes("right", size="6%", pad=0.05)
#   #cbar = plt.colorbar(img, cax=cax, ticks=np.arange(0,.01,.001), format="%.2g")
#   cbar = plt.colorbar(img, cax=cax, format="%.3f")
#   #cbar.set_label('Colorbar {}'.format(i), size=6)
#   cbar.ax.set_title('             freq.', size=10)


#   ax.set_yticks([0], minor=True)
#   ax.set_xticks([0], minor=True)
#   ax.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
#   ax.grid(b=True, which='major', color='grey', linestyle='--')
#   ax.set_ylabel('Distance from low (deg)')
#   ax.set_xlabel('Distance from low (deg)')
#   ax.margins(0.05)

#   # removes x and y ticks
#   #ax.xaxis.set_visible(False)
#   #ax.yaxis.set_visible(False)

# # prevents each subplot's axes from overlapping
# plt.tight_layout()

# # moves subplots down slightly to make room for the figure title
# plt.subplots_adjust(top=0.9)
# plt.show()

#savefig('./clas_anual_cyc', ext='png', close=False, verbose=True)
#*****************************************************************************\
#Individual Maps Clasifications
#*****************************************************************************\
# k=3
# fig, ax = plt.subplots(facecolor='w', figsize=(6,5.5))
# img = ax.pcolor(xi[:,k], yi[:,k], zi[:,:,k], cmap='BuGn')
# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.3f")
# cbar.ax.set_title('             freq.', size=12)
# cbar.ax.tick_params(labelsize=14)

# ax.set_xlim([-15,15])
# ax.set_ylim([-15,15])
# ax.set_yticks([0], minor=True)
# ax.set_xticks([0], minor=True)
# ax.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
# ax.grid(b=True, which='major', color='grey', linestyle='--')
# ax.set_ylabel('Distance from low (deg)', size=18)
# ax.set_xlabel('Distance from low (deg)', size=18)
# ax.margins(0.05)
# plt.tight_layout()

# # moves subplots down slightly to make room for the figure title
# plt.subplots_adjust(top=0.9)
# plt.show()

#*****************************************************************************\
#*****************************************************************************\
#Creating mapa Height
#*****************************************************************************\
#*****************************************************************************\
#Extrar variables

# Lat1=np.array(df_1inv['dy'])
# Lon1=np.array(df_1inv['dx'])
# Data1 = np.array(df_1inv['1ra Inv'])

# Lat2=np.array(df_1invy['dy'])
# Lon2=np.array(df_1invy['dx'])
# Data2 = np.array(df_1invy['1ra Inv'])

# Lat3=np.array(df_1invmy['dy'])
# Lon3=np.array(df_1invmy['dx'])
# Data3 = np.array(df_1invmy['1ra Inv'])



# Lat1=np.array(df_2inv['dy'])
# Lon1=np.array(df_2inv['dx'])
# Data1 = np.array(df_2inv['2da Inv'])

# Lat2=np.array(df_2invy['dy'])
# Lon2=np.array(df_2invy['dx'])
# Data2 = np.array(df_2invy['2da Inv'])

# Lat3=np.array(df_2invmy['dy'])
# Lon3=np.array(df_2invmy['dx'])
# Data3 = np.array(df_2invmy['2da Inv'])


# zi, yi, xi = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data1, normed=False)
# counts, _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))
# zi = zi / counts
# zi = np.ma.masked_invalid(zi)


# zi1, yi1, xi1 = np.histogram2d(Lat2, Lon2, bins=(30,30), weights=Data2, normed=False)
# counts1, _, _ = np.histogram2d(Lat2, Lon2, bins=(30,30))
# zi1 = zi1 / counts1
# zi1 = np.ma.masked_invalid(zi1)


# zi2, yi2, xi2 = np.histogram2d(Lat3, Lon3, bins=(30,30), weights=Data3, normed=False)
# counts2, _, _ = np.histogram2d(Lat3, Lon3, bins=(30,30))
# zi2 = zi2 / counts2
# zi2 = np.ma.masked_invalid(zi2)


# row = 1
# column = 3
# fig, axes = plt.subplots(row, column, facecolor='w', figsize=(17,5))
# ax0, ax1, ax2 = axes.flat
# cmap='Blues'
# vmax=2400
# limu=vmax+300


# img1= ax0.pcolor(xi, yi, zi, cmap=cmap,vmin=0, vmax=vmax)
# div = make_axes_locatable(ax0)
# # cax = div.append_axes("right", size="6%", pad=0.05)
# # cbar = plt.colorbar(img1, cax=cax, format="%.0f")
# # cbar.ax.set_title('     height (mts.)', size=12)
# # cbar.ax.tick_params(labelsize=14)
# ax0.set_title('MAC', size=18)
# ax0.set_yticks([0], minor=True)
# ax0.set_xticks([0], minor=True)
# ax0.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
# ax0.grid(b=True, which='major', color='grey', linestyle='--')
# ax0.set_ylabel('Distance from low (deg)', size=18)
# ax0.set_xlabel('Distance from low (deg)', size=18)
# ax0.margins(0.05)

# img2= ax1.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
# div = make_axes_locatable(ax1)
# ax1.set_title('YOTC', size=18)
# # cax = div.append_axes("right", size="6%", pad=0.05)
# # cbar = plt.colorbar(img2, cax=cax, format="%.0f")
# # cbar.ax.set_title('     height (mts.)', size=12)
# # cbar.ax.tick_params(labelsize=14)
# ax1.set_yticks([0], minor=True)
# ax1.set_xticks([0], minor=True)
# ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
# ax1.grid(b=True, which='major', color='grey', linestyle='--')
# #ax1.set_ylabel('Distance from low (deg)', size=18)
# ax1.set_xlabel('Distance from low (deg)', size=18)
# ax1.margins(0.05)

# v = np.arange(0, limu, 300)

# img3= ax2.pcolor(xi2, yi2, zi2, cmap=cmap,vmin=0, vmax=vmax)
# ax2.set_title('MAC at YOTC levels', size=18)
# div = make_axes_locatable(ax2)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img3, cax=cax, format="%.0f",ticks=v)
# cbar.set_label(' height (mts.)', size=12)
# #cbar.ax.set_title('                height (mts.)', size=12)
# cbar.ax.tick_params(labelsize=14)
# ax2.set_yticks([0], minor=True)
# ax2.set_xticks([0], minor=True)
# ax2.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
# ax2.grid(b=True, which='major', color='grey', linestyle='--')
# #ax1.set_ylabel('Distance from low (deg)', size=18)
# ax2.set_xlabel('Distance from low (deg)', size=18)
# ax2.margins(0.05)
# plt.tight_layout()
# plt.show()

# zi=np.empty([30,30,2])
# xi=np.empty([31,2])
# yi=np.empty([31,2])
# counts=np.empty([30,30,2])


# zi[:,:,0], yi[:,0], xi[:,0] = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data1, normed=False)
# counts[:,:,0], _, _ = np.histogram2d(Lat1, Lon1, bins=(30,30))

# zi[:,:,0] = zi[:,:,0] / counts[:,:,0]

# zi[:,:,1], yi[:,1], xi[:,1] = np.histogram2d(Lat2, Lon2, bins=(30,30), weights=Data2, normed=False)
# counts[:,:,1], _, _ = np.histogram2d(Lat2, Lon2, bins=(30,30))

# zi[:,:,1] = zi[:,:,1] / counts[:,:,1]


# zi = np.ma.masked_invalid(zi)
#*****************************************************************************\
#*****************************************************************************\
# MAC Quadrants
#*****************************************************************************\
#*****************************************************************************\
dfBL_Q1=df_BL[(df_BL['dx']>0) & (df_BL['dy']>0)]
dfBL_Q2=df_BL[(df_BL['dx']<0) & (df_BL['dy']>0)]
dfBL_Q3=df_BL[(df_BL['dx']<0) & (df_BL['dy']<0)]
dfBL_Q4=df_BL[(df_BL['dx']>0) & (df_BL['dy']<0)]

dfDL_Q1=df_DL[(df_DL['dx']>0) & (df_DL['dy']>0)]
dfDL_Q2=df_DL[(df_DL['dx']<0) & (df_DL['dy']>0)]
dfDL_Q3=df_DL[(df_DL['dx']<0) & (df_DL['dy']<0)]
dfDL_Q4=df_DL[(df_DL['dx']>0) & (df_DL['dy']<0)]

dfSI_Q1=df_SI[(df_SI['dx']>0) & (df_SI['dy']>0)]
dfSI_Q2=df_SI[(df_SI['dx']<0) & (df_SI['dy']>0)]
dfSI_Q3=df_SI[(df_SI['dx']<0) & (df_SI['dy']<0)]
dfSI_Q4=df_SI[(df_SI['dx']>0) & (df_SI['dy']<0)]

dfNI_Q1=df_NI[(df_NI['dx']>0) & (df_NI['dy']>0)]
dfNI_Q2=df_NI[(df_NI['dx']<0) & (df_NI['dy']>0)]
dfNI_Q3=df_NI[(df_NI['dx']<0) & (df_NI['dy']<0)]
dfNI_Q4=df_NI[(df_NI['dx']>0) & (df_NI['dy']<0)]

Q1=[len(dfNI_Q1)/float(len(df_NI)),len(dfSI_Q1)/float(len(df_SI)),len(dfDL_Q1)/float(len(df_DL)),len(dfBL_Q1)/float(len(df_BL))]

Q2=[len(dfNI_Q2)/float(len(df_NI)),len(dfSI_Q2)/float(len(df_SI)),len(dfDL_Q2)/float(len(df_DL)),len(dfBL_Q2)/float(len(df_BL))]

Q3=[len(dfNI_Q3)/float(len(df_NI)),len(dfSI_Q3)/float(len(df_SI)),len(dfDL_Q3)/float(len(df_DL)),len(dfBL_Q3)/float(len(df_BL))]

Q4=[len(dfNI_Q4)/float(len(df_NI)),len(dfSI_Q4)/float(len(df_SI)),len(dfDL_Q4)/float(len(df_DL)),len(dfBL_Q4)/float(len(df_BL))]

raw_data = {'tipo': ['No Inv.', 'Single Inv.', 'Decoupled L.', 'Buffer L.'],
        'Q1': Q1,
        'Q2': Q2,
        'Q3': Q3,
        'Q4': Q4}

df = pd.DataFrame(raw_data, columns = ['tipo', 'Q1', 'Q2', 'Q3','Q4'])

df.to_csv(path_data + 'dfpie_mac.csv', sep='\t', encoding='utf-8')


#*****************************************************************************\
# MAC Quadrants
#*****************************************************************************\
dfBLy_Q1=df_BLy[(df_BLy['dx']>0) & (df_BLy['dy']>0)]
dfBLy_Q2=df_BLy[(df_BLy['dx']<0) & (df_BLy['dy']>0)]
dfBLy_Q3=df_BLy[(df_BLy['dx']<0) & (df_BLy['dy']<0)]
dfBLy_Q4=df_BLy[(df_BLy['dx']>0) & (df_BLy['dy']<0)]

dfDLy_Q1=df_DLy[(df_DLy['dx']>0) & (df_DLy['dy']>0)]
dfDLy_Q2=df_DLy[(df_DLy['dx']<0) & (df_DLy['dy']>0)]
dfDLy_Q3=df_DLy[(df_DLy['dx']<0) & (df_DLy['dy']<0)]
dfDLy_Q4=df_DLy[(df_DLy['dx']>0) & (df_DLy['dy']<0)]

dfSIy_Q1=df_SIy[(df_SIy['dx']>0) & (df_SIy['dy']>0)]
dfSIy_Q2=df_SIy[(df_SIy['dx']<0) & (df_SIy['dy']>0)]
dfSIy_Q3=df_SIy[(df_SIy['dx']<0) & (df_SIy['dy']<0)]
dfSIy_Q4=df_SIy[(df_SIy['dx']>0) & (df_SIy['dy']<0)]

dfNIy_Q1=df_NIy[(df_NIy['dx']>0) & (df_NIy['dy']>0)]
dfNIy_Q2=df_NIy[(df_NIy['dx']<0) & (df_NIy['dy']>0)]
dfNIy_Q3=df_NIy[(df_NIy['dx']<0) & (df_NIy['dy']<0)]
dfNIy_Q4=df_NIy[(df_NIy['dx']>0) & (df_NIy['dy']<0)]

Q1y=[len(dfNIy_Q1)/float(len(df_NIy)),len(dfSIy_Q1)/float(len(df_SIy)),len(dfDLy_Q1)/float(len(df_DLy)),len(dfBLy_Q1)/float(len(df_BLy))]

Q2y=[len(dfNIy_Q2)/float(len(df_NIy)),len(dfSIy_Q2)/float(len(df_SIy)),len(dfDLy_Q2)/float(len(df_DLy)),len(dfBLy_Q2)/float(len(df_BLy))]

Q3y=[len(dfNIy_Q3)/float(len(df_NIy)),len(dfSIy_Q3)/float(len(df_SIy)),len(dfDLy_Q3)/float(len(df_DLy)),len(dfBLy_Q3)/float(len(df_BLy))]

Q4y=[len(dfNIy_Q4)/float(len(df_NIy)),len(dfSIy_Q4)/float(len(df_SIy)),len(dfDLy_Q4)/float(len(df_DLy)),len(dfBLy_Q4)/float(len(df_BLy))]



raw_datay = {'tipo': ['No Inv.', 'Single Inv.', 'Decoupled L.', 'Buffer L.'],
        'Q1': Q1y,
        'Q2': Q2y,
        'Q3': Q3y,
        'Q4': Q4y}

dfQ_yotc = pd.DataFrame(raw_datay, columns = ['tipo', 'Q1', 'Q2', 'Q3','Q4'])

dfQ_yotc.to_csv(path_data + 'dfpie_yotc.csv', sep='\t', encoding='utf-8')

#*****************************************************************************\
# MAC levels YOTC Quadrants
#*****************************************************************************\

dfBLmy_Q1=df_BLmy[(df_BLmy['dx']>0) & (df_BLmy['dy']>0)]
dfBLmy_Q2=df_BLmy[(df_BLmy['dx']<0) & (df_BLmy['dy']>0)]
dfBLmy_Q3=df_BLmy[(df_BLmy['dx']<0) & (df_BLmy['dy']<0)]
dfBLmy_Q4=df_BLmy[(df_BLmy['dx']>0) & (df_BLmy['dy']<0)]

dfDLmy_Q1=df_DLmy[(df_DLmy['dx']>0) & (df_DLmy['dy']>0)]
dfDLmy_Q2=df_DLmy[(df_DLmy['dx']<0) & (df_DLmy['dy']>0)]
dfDLmy_Q3=df_DLmy[(df_DLmy['dx']<0) & (df_DLmy['dy']<0)]
dfDLmy_Q4=df_DLmy[(df_DLmy['dx']>0) & (df_DLmy['dy']<0)]

dfSImy_Q1=df_SImy[(df_SImy['dx']>0) & (df_SImy['dy']>0)]
dfSImy_Q2=df_SImy[(df_SImy['dx']<0) & (df_SImy['dy']>0)]
dfSImy_Q3=df_SImy[(df_SImy['dx']<0) & (df_SImy['dy']<0)]
dfSImy_Q4=df_SImy[(df_SImy['dx']>0) & (df_SImy['dy']<0)]

dfNImy_Q1=df_NImy[(df_NImy['dx']>0) & (df_NImy['dy']>0)]
dfNImy_Q2=df_NImy[(df_NImy['dx']<0) & (df_NImy['dy']>0)]
dfNImy_Q3=df_NImy[(df_NImy['dx']<0) & (df_NImy['dy']<0)]
dfNImy_Q4=df_NImy[(df_NImy['dx']>0) & (df_NImy['dy']<0)]

Q1my=[len(dfNImy_Q1)/float(len(df_NImy)),len(dfSImy_Q1)/float(len(df_SImy)),len(dfDLmy_Q1)/float(len(df_DLmy)),len(dfBLmy_Q1)/float(len(df_BLmy))]

Q2my=[len(dfNImy_Q2)/float(len(df_NImy)),len(dfSImy_Q2)/float(len(df_SImy)),len(dfDLmy_Q2)/float(len(df_DLmy)),len(dfBLmy_Q2)/float(len(df_BLmy))]

Q3my=[len(dfNImy_Q3)/float(len(df_NImy)),len(dfSImy_Q3)/float(len(df_SImy)),len(dfDLmy_Q3)/float(len(df_DLmy)),len(dfBLmy_Q3)/float(len(df_BLmy))]

Q4my=[len(dfNImy_Q4)/float(len(df_NImy)),len(dfSImy_Q4)/float(len(df_SImy)),len(dfDLmy_Q4)/float(len(df_DLmy)),len(dfBLmy_Q4)/float(len(df_BLmy))]


raw_datamy = {'tipo': ['No Inv.', 'Single Inv.', 'Decoupled L.', 'Buffer L.'],
        'Q1': Q1my,
        'Q2': Q2my,
        'Q3': Q3my,
        'Q4': Q4my}

dfQ_macyotc = pd.DataFrame(raw_datamy, columns = ['tipo', 'Q1', 'Q2', 'Q3','Q4'])

dfQ_macyotc.to_csv(path_data + 'dfpie_macyotc.csv', sep='\t', encoding='utf-8')

# # Create the general blog and the "subplots" i.e. the bars
# f, ax1 = plt.subplots(1, figsize=(10,6))
# # Set the bar width
# bar_width = 0.7
# # positions of the left bar-boundaries
# bar_l = [i+1 for i in range(len(df['Q1']))]
# # positions of the x-axis ticks (center of the bars as bar labels)
# tick_pos = [i+(bar_width/2) for i in bar_l]

# ax1.barh(bar_l,
#         df['Q1'],
#         label='Q1',
#         color='#7C83AF')

# ax1.barh(bar_l,
#         df['Q2'],
#         left=df['Q1'],
#         label='Q2',
#         color='#525B92')

# ax1.barh(bar_l,
#         df['Q3'],
#         left=[i+j for i,j in zip(df['Q1'],df['Q2'])],
#         label='Q3',
#         color='#182157')

# ax1.barh(bar_l,
#         df['Q4'],
#         left=[i+j+k for i,j,k in zip(df['Q1'],df['Q2'],df['Q3'])],
#         label='Q4',
#         color='#080F3A')


# plt.yticks(tick_pos, df['tipo'])
# ax1.set_xlabel("Absolute Percentage Occurrence")
# ax1.set_title('Boundary Layer Categories')
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=4)

# plt.ylim([min(tick_pos)-bar_width+0.1, max(tick_pos)+bar_width])
# plt.show()



# # Create the general blog and the "subplots" i.e. the bars
# f, ax1 = plt.subplots(1, figsize=(10,6))
# # Set the bar width
# bar_width = 0.7
# # positions of the left bar-boundaries
# bar_l = [i+1 for i in range(len(df['Q1']))]
# # positions of the x-axis ticks (center of the bars as bar labels)
# tick_pos = [i+(bar_width/2) for i in bar_l]

# ax1.bar(bar_l,
#         df['Q1'],
#         label='NE',
#         color='#7C83AF')

# ax1.bar(bar_l,
#         df['Q2'],
#         bottom=df['Q1'],
#         label='NW',
#         color='#525B92')

# ax1.bar(bar_l,
#         df['Q3'],
#         bottom=[i+j for i,j in zip(df['Q1'],df['Q2'])],
#         label='SW',
#         color='#182157')

# ax1.bar(bar_l,
#         df['Q4'],
#         bottom=[i+j+k for i,j,k in zip(df['Q1'],df['Q2'],df['Q3'])],
#         label='SE',
#         color='#080F3A')


# plt.xticks(tick_pos, df['tipo'])
# ax1.set_ylabel("Absolute Percentage Occurrence")
# #ax1.set_title('Boundary Layer Categories')
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=4)

# plt.xlim([min(tick_pos)-bar_width+0.1, max(tick_pos)+bar_width])
# plt.show()

#*****************************************************************************\
#Pie Chart
#*****************************************************************************\
# outd={}
# outd[0] = {'names':df['tipo'], 'values':df['Q1']}
# outd[1] = {'names':df['tipo'], 'values':df['Q2']}
# outd[2] = {'names':df['tipo'], 'values':df['Q3']}
# outd[3] = {'names':df['tipo'], 'values':df['Q4']}

# colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
# explode = (0, 0, 0, 0)  # explode a slice if required

# # This first pie chart fill the plot, it's the lowest level
# plt.pie(outd[3]['values'], labels=outd[3]['names'], labeldistance=0.9)
# ax = plt.gca()
# # For each successive plot, change the max radius so that they overlay
# for i in np.arange(2,-1,-1):
#     ax.pie(outd[i]['values'], labels=outd[i]['names'],
#            radius=np.float(i+1)/4.0, labeldistance=((2*(i+1)-1)/8.0)/((i+1)/4.0))
# ax.set_aspect('equal')
# plt.axis('equal')
# plt.show()


#*****************************************************************************\
#*****************************************************************************\
# Height Seasonal
#*****************************************************************************\
#*****************************************************************************

# df1inv_DJF=df_1inv[(df_1inv.index.month==12) | (df_1inv.index.month==1) | (df_1inv.index.month==2)]
# df1inv_MAM=df_1inv[(df_1inv.index.month==3) | (df_1inv.index.month==4) | (df_1inv.index.month==5)]
# df1inv_JJA=df_1inv[(df_1inv.index.month==6) | (df_1inv.index.month==7) | (df_1inv.index.month==8)]
# df1inv_SON=df_1inv[(df_1inv.index.month==9) | (df_1inv.index.month==10) | (df_1inv.index.month==11)]

# df2inv_DJF=df_2inv[(df_2inv.index.month==12) | (df_2inv.index.month==1) | (df_2inv.index.month==2)]
# df2inv_MAM=df_2inv[(df_2inv.index.month==3) | (df_2inv.index.month==4) | (df_2inv.index.month==5)]
# df2inv_JJA=df_2inv[(df_2inv.index.month==6) | (df_2inv.index.month==7) | (df_2inv.index.month==8)]
# df2inv_SON=df_2inv[(df_2inv.index.month==9) | (df_2inv.index.month==10) | (df_2inv.index.month==11)]

# df1invy_DJF=df_1invy[(df_1invy.index.month==12) | (df_1invy.index.month==1) | (df_1invy.index.month==2)]
# df1invy_MAM=df_1invy[(df_1invy.index.month==3) | (df_1invy.index.month==4) | (df_1invy.index.month==5)]
# df1invy_JJA=df_1invy[(df_1invy.index.month==6) | (df_1invy.index.month==7) | (df_1invy.index.month==8)]
# df1invy_SON=df_1invy[(df_1invy.index.month==9) | (df_1invy.index.month==10) | (df_1invy.index.month==11)]

# df2invy_DJF=df_2invy[(df_2invy.index.month==12) | (df_2invy.index.month==1) | (df_2invy.index.month==2)]
# df2invy_MAM=df_2invy[(df_2invy.index.month==3) | (df_2invy.index.month==4) | (df_2invy.index.month==5)]
# df2invy_JJA=df_2invy[(df_2invy.index.month==6) | (df_2invy.index.month==7) | (df_2invy.index.month==8)]
# df2invy_SON=df_2invy[(df_2invy.index.month==9) | (df_2invy.index.month==10) | (df_2invy.index.month==11)]

# df1invmy_DJF=df_1invmy[(df_1invmy.index.month==12) | (df_1invmy.index.month==1) | (df_1invmy.index.month==2)]
# df1invmy_MAM=df_1invmy[(df_1invmy.index.month==3) | (df_1invmy.index.month==4) | (df_1invmy.index.month==5)]
# df1invmy_JJA=df_1invmy[(df_1invmy.index.month==6) | (df_1invmy.index.month==7) | (df_1invmy.index.month==8)]
# df1invmy_SON=df_1invmy[(df_1invmy.index.month==9) | (df_1invmy.index.month==10) | (df_1invmy.index.month==11)]

# df2invmy_DJF=df_2invmy[(df_2invmy.index.month==12) | (df_2invmy.index.month==1) | (df_2invmy.index.month==2)]
# df2invmy_MAM=df_2invmy[(df_2invmy.index.month==3) | (df_2invmy.index.month==4) | (df_2invmy.index.month==5)]
# df2invmy_JJA=df_2invmy[(df_2invmy.index.month==6) | (df_2invmy.index.month==7) | (df_2invmy.index.month==8)]
# df2invmy_SON=df_2invmy[(df_2invmy.index.month==9) | (df_2invmy.index.month==10) | (df_2invmy.index.month==11)]


# mean_1inv_DJF=np.nanmean(df1inv_DJF['1ra Inv'])
# mean_1inv_MAM=np.nanmean(df1inv_MAM['1ra Inv'])
# mean_1inv_JJA=np.nanmean(df1inv_JJA['1ra Inv'])
# mean_1inv_SON=np.nanmean(df1inv_SON['1ra Inv'])

# mean_2inv_DJF=np.nanmean(df2inv_DJF['2da Inv'])
# mean_2inv_MAM=np.nanmean(df2inv_MAM['2da Inv'])
# mean_2inv_JJA=np.nanmean(df2inv_JJA['2da Inv'])
# mean_2inv_SON=np.nanmean(df2inv_SON['2da Inv'])

# mean_1invy_DJF=np.nanmean(df1invy_DJF['1ra Inv'])
# mean_1invy_MAM=np.nanmean(df1invy_MAM['1ra Inv'])
# mean_1invy_JJA=np.nanmean(df1invy_JJA['1ra Inv'])
# mean_1invy_SON=np.nanmean(df1invy_SON['1ra Inv'])

# mean_2invy_DJF=np.nanmean(df2invy_DJF['2da Inv'])
# mean_2invy_MAM=np.nanmean(df2invy_MAM['2da Inv'])
# mean_2invy_JJA=np.nanmean(df2invy_JJA['2da Inv'])
# mean_2invy_SON=np.nanmean(df2invy_SON['2da Inv'])

# mean_1invmy_DJF=np.nanmean(df1invmy_DJF['1ra Inv'])
# mean_1invmy_MAM=np.nanmean(df1invmy_MAM['1ra Inv'])
# mean_1invmy_JJA=np.nanmean(df1invmy_JJA['1ra Inv'])
# mean_1invmy_SON=np.nanmean(df1invmy_SON['1ra Inv'])

# mean_2invmy_DJF=np.nanmean(df2invmy_DJF['2da Inv'])
# mean_2invmy_MAM=np.nanmean(df2invmy_MAM['2da Inv'])
# mean_2invmy_JJA=np.nanmean(df2invmy_JJA['2da Inv'])
# mean_2invmy_SON=np.nanmean(df2invmy_SON['2da Inv'])

# raw_data1 = {'sounding': ['MAC', 'YOTC', 'MAC at YOTC levels'],
#         'DJF': [mean_1inv_DJF,mean_1invy_DJF,mean_1invmy_DJF],
#         'MAM': [mean_1inv_MAM,mean_1invy_MAM,mean_1invmy_MAM],
#         'JJA': [mean_1inv_JJA,mean_1invy_JJA,mean_1invmy_JJA],
#         'SON': [mean_1inv_SON,mean_1invy_SON,mean_1invmy_SON]}

# df1inv = pd.DataFrame(raw_data1, columns = ['sounding', 'DJF', 'MAM', 'JJA','SON'])

# raw_data2 = {'sounding': ['MAC', 'YOTC', 'MAC at YOTC levels'],
#         'DJF': [mean_2inv_DJF,mean_2invy_DJF,mean_2invmy_DJF],
#         'MAM': [mean_2inv_MAM,mean_2invy_MAM,mean_2invmy_MAM],
#         'JJA': [mean_2inv_JJA,mean_2invy_JJA,mean_2invmy_JJA],
#         'SON': [mean_2inv_SON,mean_2invy_SON,mean_2invmy_SON]}

# df2inv = pd.DataFrame(raw_data2, columns = ['sounding', 'DJF', 'MAM', 'JJA','SON'])


# # Setting Plot
# fig, ax = plt.subplots(figsize=(8, 5))

# width = 0.2
# pos = list(range(len(df1inv['DJF'])))
# #*****************************************************************************\
# plt.bar(pos,
#         #using df['pre_score'] data,
#         df1inv['DJF'],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='#c6c9d0',
#         #color='#7C83AF',
#         # with label the first value in first_name
#         label='DJF')


# plt.bar([p + width for p in pos],
#         #using df['mid_score'] data,
#         df1inv['MAM'],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='#67832F',
#         # with label the second value in first_name
#         label='MAM')

# plt.bar([p + width*2 for p in pos],
#         #using df['post_score'] data,
#         df1inv['JJA'],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='tomato',
#         #color='#182157',
#         # with label the third value in first_name
#         label='JJA')

# plt.bar([p + width*3 for p in pos],
#         #using df['post_score'] data,
#         df1inv['SON'],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='blue',
#         #color='#182157',
#         # with label the third value in first_name
#         label='SON')


# # Set the position of the x ticks
# ax.set_xticks([p + 2 * width for p in pos])
# # Set the labels for the x ticks
# ax.set_xticklabels(df1inv['sounding'],fontsize = 13)
# # Setting the x-axis and y-axis limits
# plt.xlim(min(pos)-width, max(pos)+width*5)
# ax.set_ylabel('Height (mts.)',fontsize = 13)
# #Legend
# plt.legend(['DJF', 'MAM', 'JJA','SON'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
# plt.grid()
# plt.ylim(0,1600)
# plt.show()



# # Setting Plot
# fig, ax = plt.subplots(figsize=(8, 5))

# width = 0.2
# pos = list(range(len(df1inv['DJF'])))
# #*****************************************************************************\
# plt.bar(pos,
#         #using df['pre_score'] data,
#         df2inv['DJF'],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='#c6c9d0',
#         #color='#7C83AF',
#         # with label the first value in first_name
#         label='DJF')


# plt.bar([p + width for p in pos],
#         #using df['mid_score'] data,
#         df2inv['MAM'],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='#67832F',
#         # with label the second value in first_name
#         label='MAM')

# plt.bar([p + width*2 for p in pos],
#         #using df['post_score'] data,
#         df2inv['JJA'],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='tomato',
#         #color='#182157',
#         # with label the third value in first_name
#         label='JJA')

# plt.bar([p + width*3 for p in pos],
#         #using df['post_score'] data,
#         df2inv['SON'],
#         # of width
#         width,
#         # with alpha 0.5
#         alpha=0.5,
#         # with color
#         color='blue',
#         #color='#182157',
#         # with label the third value in first_name
#         label='SON')


# # Set the position of the x ticks
# ax.set_xticks([p + 2 * width for p in pos])
# # Set the labels for the x ticks
# ax.set_xticklabels(df2inv['sounding'],fontsize = 13)
# # Setting the x-axis and y-axis limits
# plt.xlim(min(pos)-width, max(pos)+width*5)
# plt.ylim(0,1600)
# ax.set_ylabel('Height (mts.)',fontsize = 13)
# #Legend
# plt.legend(['DJF', 'MAM', 'JJA','SON'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
# plt.grid()
# plt.show()


#*****************************************************************************\
#*****************************************************************************\
# Height Quadrant
#*****************************************************************************\
#*****************************************************************************

df1inv_Q1=df_1inv[(df_1inv['dx']>0) & (df_1inv['dy']>0)]
df1inv_Q2=df_1inv[(df_1inv['dx']<0) & (df_1inv['dy']>0)]
df1inv_Q3=df_1inv[(df_1inv['dx']<0) & (df_1inv['dy']<0)]
df1inv_Q4=df_1inv[(df_1inv['dx']>0) & (df_1inv['dy']<0)]

df2inv_Q1=df_2inv[(df_2inv['dx']>0) & (df_2inv['dy']>0)]
df2inv_Q2=df_2inv[(df_2inv['dx']<0) & (df_2inv['dy']>0)]
df2inv_Q3=df_2inv[(df_2inv['dx']<0) & (df_2inv['dy']<0)]
df2inv_Q4=df_2inv[(df_2inv['dx']>0) & (df_2inv['dy']<0)]

df1invmy_Q1=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']>0)]
df1invmy_Q2=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']>0)]
df1invmy_Q3=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']<0)]
df1invmy_Q4=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']<0)]

df2invmy_Q1=df_2invmy[(df_2invmy['dx']>0) & (df_2invmy['dy']>0)]
df2invmy_Q2=df_2invmy[(df_2invmy['dx']<0) & (df_2invmy['dy']>0)]
df2invmy_Q3=df_2invmy[(df_2invmy['dx']<0) & (df_2invmy['dy']<0)]
df2invmy_Q4=df_2invmy[(df_2invmy['dx']>0) & (df_2invmy['dy']<0)]


df1invy_Q1=df_1invy[(df_1invy['dx']>0) & (df_1invy['dy']>0)]
df1invy_Q2=df_1invy[(df_1invy['dx']<0) & (df_1invy['dy']>0)]
df1invy_Q3=df_1invy[(df_1invy['dx']<0) & (df_1invy['dy']<0)]
df1invy_Q4=df_1invy[(df_1invy['dx']>0) & (df_1invy['dy']<0)]

df2invy_Q1=df_2invy[(df_2invy['dx']>0) & (df_2invy['dy']>0)]
df2invy_Q2=df_2invy[(df_2invy['dx']<0) & (df_2invy['dy']>0)]
df2invy_Q3=df_2invy[(df_2invy['dx']<0) & (df_2invy['dy']<0)]
df2invy_Q4=df_2invy[(df_2invy['dx']>0) & (df_2invy['dy']<0)]

Q1_1inv=np.nanmean(df1inv_Q1['1ra Inv'])
Q2_1inv=np.nanmean(df1inv_Q2['1ra Inv'])
Q3_1inv=np.nanmean(df1inv_Q3['1ra Inv'])
Q4_1inv=np.nanmean(df1inv_Q4['1ra Inv'])

Q1_1invy=np.nanmean(df1invy_Q1['1ra Inv'])
Q2_1invy=np.nanmean(df1invy_Q2['1ra Inv'])
Q3_1invy=np.nanmean(df1invy_Q3['1ra Inv'])
Q4_1invy=np.nanmean(df1invy_Q4['1ra Inv'])

Q1_1invmy=np.nanmean(df1invmy_Q1['1ra Inv'])
Q2_1invmy=np.nanmean(df1invmy_Q2['1ra Inv'])
Q3_1invmy=np.nanmean(df1invmy_Q3['1ra Inv'])
Q4_1invmy=np.nanmean(df1invmy_Q4['1ra Inv'])


raw_data1 = {'sounding': ['MAC', 'YOTC', 'MAC at YOTC levels'],
        'Q1': [Q1_1inv, Q1_1invy, Q1_1invmy],
        'Q2': [Q2_1inv, Q2_1invy, Q2_1invmy],
        'Q3': [Q3_1inv, Q3_1invy, Q3_1invmy],
        'Q4': [Q4_1inv, Q4_1invy, Q4_1invmy]}

df1invQ = pd.DataFrame(raw_data1, columns = ['sounding', 'Q1', 'Q2', 'Q3','Q4'])

# Setting Plot
fig, ax = plt.subplots(figsize=(8, 5))

width = 0.2
pos = list(range(len(df1invQ['Q1'])))
#*****************************************************************************\
plt.bar(pos,
        #using df['pre_score'] data,
        df1invQ['Q1'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#c6c9d0',
        #color='#7C83AF',
        # with label the first value in first_name
        label='Q1')


plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        df1invQ['Q2'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#67832F',
        # with label the second value in first_name
        label='Q2')

plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        df1invQ['Q3'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='tomato',
        #color='#182157',
        # with label the third value in first_name
        label='Q3')

plt.bar([p + width*3 for p in pos],
        #using df['post_score'] data,
        df1invQ['Q4'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='blue',
        #color='#182157',
        # with label the third value in first_name
        label='Q4')


# Set the position of the x ticks
ax.set_xticks([p + 2 * width for p in pos])
# Set the labels for the x ticks
ax.set_xticklabels(df1invQ['sounding'],fontsize = 13)
# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*5)
ax.set_ylabel('Height (mts.)',fontsize = 13)
#Legend
plt.legend(['NE', 'NW', 'SW','SE'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
plt.grid()
plt.ylim(0,1600)
plt.show()








Q1_2inv=np.nanmean(df2inv_Q1['2da Inv'])
Q2_2inv=np.nanmean(df2inv_Q2['2da Inv'])
Q3_2inv=np.nanmean(df2inv_Q3['2da Inv'])
Q4_2inv=np.nanmean(df2inv_Q4['2da Inv'])

Q1_2invy=np.nanmean(df2invy_Q1['2da Inv'])
Q2_2invy=np.nanmean(df2invy_Q2['2da Inv'])
Q3_2invy=np.nanmean(df2invy_Q3['2da Inv'])
Q4_2invy=np.nanmean(df2invy_Q4['2da Inv'])

Q1_2invmy=np.nanmean(df2invmy_Q1['2da Inv'])
Q2_2invmy=np.nanmean(df2invmy_Q2['2da Inv'])
Q3_2invmy=np.nanmean(df2invmy_Q3['2da Inv'])
Q4_2invmy=np.nanmean(df2invmy_Q4['2da Inv'])


raw_data2 = {'sounding': ['MAC', 'YOTC', 'MAC at YOTC levels'],
        'Q1': [Q1_2inv, Q1_2invy, Q1_2invmy],
        'Q2': [Q2_2inv, Q2_2invy, Q2_2invmy],
        'Q3': [Q3_2inv, Q3_2invy, Q3_2invmy],
        'Q4': [Q4_2inv, Q4_2invy, Q4_2invmy]}

df2invQ = pd.DataFrame(raw_data2, columns = ['sounding', 'Q1', 'Q2', 'Q3','Q4'])

# Setting Plot
fig, ax = plt.subplots(figsize=(8, 5))

width = 0.2
pos = list(range(len(df2invQ['Q1'])))
#*****************************************************************************\
plt.bar(pos,
        #using df['pre_score'] data,
        df2invQ['Q1'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#c6c9d0',
        #color='#7C83AF',
        # with label the first value in first_name
        label='Q1')


plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        df2invQ['Q2'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#67832F',
        # with label the second value in first_name
        label='Q2')

plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        df2invQ['Q3'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='tomato',
        #color='#182157',
        # with label the third value in first_name
        label='Q3')

plt.bar([p + width*3 for p in pos],
        #using df['post_score'] data,
        df2invQ['Q4'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='blue',
        #color='#182157',
        # with label the third value in first_name
        label='Q4')


# Set the position of the x ticks
ax.set_xticks([p + 2 * width for p in pos])
# Set the labels for the x ticks
ax.set_xticklabels(df2invQ['sounding'],fontsize = 13)
# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*5)
ax.set_ylabel('Height (mts.)',fontsize = 13)
#Legend
plt.legend(['NE', 'NW', 'SW','SE'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
plt.grid()
plt.ylim(0,1600)
plt.show()
