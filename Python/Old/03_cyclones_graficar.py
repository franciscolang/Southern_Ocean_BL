import numpy as np
import scipy.io as sio
import pandas as pd
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator
import matplotlib.mlab as mlab
import scipy as sp
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as si
from scipy.interpolate import interp1d
from glob import glob
import matplotlib as mpl

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'

latMac=-54.50;
lonMac=158.95;

#*****************************************************************************\
# Readinf CSV

df_macyotc_final= pd.read_csv(path_data + 'df_macyotc_final.csv', sep='\t', parse_dates=['Date'])
df_mac_final= pd.read_csv(path_data + 'df_mac_final.csv', sep='\t', parse_dates=['Date'])
df_yotc_all= pd.read_csv(path_data + 'df_yotc_all.csv', sep='\t', parse_dates=['Date'])
df_yotc= pd.read_csv(path_data + 'df_yotc.csv', sep='\t', parse_dates=['Date'])
#*****************************************************************************\
# yot_clas=np.array(df_yotc['Clas'])
# mac_clas=np.array(df_mac_final['Clas'])
# macyotc_clas=np.array(df_macyotc_final['Clas'])
# ny, bin_edgesy =np.histogram(yot_clas, bins=[1, 2, 3, 4,5],normed=1)
# nm, bin_edgesm =np.histogram(mac_clas, bins=[1, 2, 3, 4,5],normed=1)
# nmy, bin_edgesmy =np.histogram(macyotc_clas, bins=[1, 2, 3, 4,5],normed=1)

# #*****************************************************************************\

# NI=[nm[0],  ny[0],  nmy[0]]
# SI=[nm[1],  ny[1],  nmy[1]]
# DL=[nm[2],  ny[2],  nmy[2]]
# BL=[nm[3],  ny[3],  nmy[3]]

# #*****************************************************************************\

# raw_data = {'tipo': ['MAC', 'YOTC', 'MACyotc'],
#         'NI': NI,
#         'SI': SI,
#         'DL': DL,
#         'BL': BL}

# df = pd.DataFrame(raw_data, columns = ['tipo', 'NI', 'SI', 'DL', 'BL'])

# # Create the general blog and the "subplots" i.e. the bars
# f, ax1 = plt.subplots(1, figsize=(10,6))

# # Set the bar width
# bar_width = 0.7

# # positions of the left bar-boundaries
# bar_l = [i+1 for i in range(len(df['NI']))]

# # positions of the x-axis ticks (center of the bars as bar labels)
# tick_pos = [i+(bar_width/2) for i in bar_l]

# # Create a bar plot, in position bar_1
# ax1.barh(bar_l,
#         # using the pre_score data
#         df['NI'],
#         # set the width
#         #width=bar_width,
#         # with the label pre score
#         label='NI',
#         # with alpha 0.5
#         #alpha=0.5,
#         # with color
#         #color='#BFEFFF')
#         color='#7C83AF')

# # Create a bar plot, in position bar_1
# ax1.barh(bar_l,
#         # using the mid_score data
#         df['SI'],
#         # set the width
#         #width=bar_width,
#         # with pre_score on the bottom
#         #bottom=df['NI'],
#         left=df['NI'],
#         # with the label mid score
#         label='SI',
#         # with alpha 0.5
#         #alpha=0.5,
#         # with color
#         #color='#60AFFE')
#         color='#525B92')

# # Create a bar plot, in position bar_1
# ax1.barh(bar_l,
#         # using the post_score data
#         df['DL'],
#         # set the width
#         #width=bar_width,
#         # with pre_score and mid_score on the bottom
#         #bottom=[i+j for i,j in zip(df['NI'],df['SI'])],
#         left=[i+j for i,j in zip(df['NI'],df['SI'])],
#         # with the label post score
#         label='DL',
#         # with alpha 0.5
#         #alpha=0.5,
#         # with color
#         #color='#0276FD')
#         color='#182157')

# # Create a bar plot, in position bar_1
# ax1.barh(bar_l,
#         # using the post_score data
#         df['BL'],
#         # set the width
#         #width=bar_width,
#         # with pre_score and mid_score on the bottom
#         #bottom=[i+j+k for i,j,k in zip(df['NI'],df['SI'],df['DL'])],
#         left=[i+j+k for i,j,k in zip(df['NI'],df['SI'],df['DL'])],
#         # with the label post score
#         label='BL',
#         # with alpha 0.5
#         #alpha=0.5,
#         # with color
#         #color='#26466D')
#         color='#080F3A')

# # set the x ticks with names
# plt.yticks(tick_pos, df['tipo'])

# # Set the label and legends
# ax1.set_xlabel("Absolute Percentage Occurrence")
# #ax1.set_xlabel("Soundings")
# ax1.set_title('Boundary Layer Categories')
# #plt.legend(loc='upper left')
# #ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 1.05),
# #          ncol=1, fancybox=True, shadow=True)
# #ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# # Put a legend below current axis
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=4)

# # Set a buffer around the edge
# plt.ylim([min(tick_pos)-bar_width+0.1, max(tick_pos)+bar_width])
# plt.show()
#*****************************************************************************\
#*****************************************************************************\
#                           CYCLONES
#*****************************************************************************\
#*****************************************************************************\
#Reading
path_mcms=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MCMS/'
#Reading Multiples Files
fnames = glob(path_mcms+'*.txt')
arrays = [np.loadtxt(f, delimiter=' ') for f in fnames]
array_cyc = np.concatenate(arrays, axis=0)
print len(array_cyc)
#np.savetxt('result.txt', final_array, fmt='%.2f')
#*****************************************************************************\
#Filtrar
#*****************************************************************************\
#Latitud
CoLat=array_cyc[:,5]
lat = 90.0 - (CoLat*0.01);
indlan = np.nonzero(lat<-30.)
lat2=lat[indlan]

cycf1=array_cyc[indlan]
print len(cycf1)

#Lontitud
CoLon=cycf1[:,6]
lon=CoLon*0.01;
indlon = np.nonzero(lon<180)
lon2=lon[indlon]

cyc_close=cycf1[indlon]
print len(cyc_close)

#Hora
hora=cyc_close[:,3]
indh1 = np.nonzero(hora!=6)
cyc_h1=cyc_close[indh1]
hora1=cyc_h1[:,3]
indh2 = np.nonzero(hora1!=18)
cyc_filt=cyc_h1[indh2]
print len(cyc_filt)


#*****************************************************************************\
#Latitud
lati = 90.0 - (cyc_filt[:,5]*0.01);
#Longitud
longi=cyc_filt[:,6]*0.01;
#*****************************************************************************\
#Distance from MAC
dy=(-lati+latMac);
dx=(-longi+lonMac);

#15 degrees from MAC
lim=15;

distance=np.sqrt(dy**2+dx**2);
ind_dist = np.nonzero(distance<lim)

latitud=lati[ind_dist]
longitud=longi[ind_dist]
cyc_mac=cyc_filt[ind_dist]
dist_low=distance[ind_dist]
print len(cyc_mac)
del dy,dx
#*****************************************************************************\
#Creating datetime variables
array=cyc_mac[:,0:4] #Array fechas

yy=array[:,0].astype(int)
mm=array[:,1].astype(int)
dd=array[:,2].astype(int)
hh=array[:,3].astype(int)

ndates,_=array.shape

mydates=np.array([])
for n in range(0,ndates):
    mydates=np.append(mydates, datetime(yy[n], mm[n], dd[n],hh[n],0,0))

#*****************************************************************************\
#Dataframe Pandas

dm={'USI':cyc_mac[:,-1],
'lat':latitud,
'lon':longitud,
'dist':dist_low}

df_cyc1 = pd.DataFrame(data=dm,index=mydates)

#*****************************************************************************\
#Closest Cyclon
#df_cyc1.index.name = 'Date'
#df=df_cyc1.groupby('Date')['dist'].min().reset_index()
# df=df.ix[dc.groupby(dc.index)['dist'].min().reset_index()]
# print df

#df=df_cyc1.groupby(df_cyc1.index)['dist'].min().reset_index()

df2 = df_cyc1.groupby(df_cyc1.index)
target = df2['dist'].min().values
idx=[np.where(df_cyc1['dist']==v)[0][0] for v in target]
df_cyc2 = df_cyc1.iloc[idx].copy()

# Eliminate Duplicate Soundings
df_cyc3=df_cyc2.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

#Date index del periodo 2006-2010
date_index_all = pd.date_range('2006-01-01 00:00', periods=3652, freq='12H')
df_cyc4=df_cyc3.reindex(date_index_all)
#*****************************************************************************\
#Calcular distancia from low
df_cyc4['dy']=(-df_cyc3['lat']+latMac)
df_cyc4['dx']=(-df_cyc3['lon']+lonMac)

df_cyc4.index.name = 'Fecha'
df_mac= df_mac_final.set_index('Date')

#Unir datraframe mac con cyclones
df_maccyc=pd.concat([df_mac, df_cyc4],axis=1)

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

#*****************************************************************************\

Lat1=np.array(df_BL['dy'])
Lon1=np.array(df_BL['dx'])
Data1=np.ones(len(df_BL))

Lat2=np.array(df_DL['dy'])
Lon2=np.array(df_DL['dx'])
Data2=np.ones(len(df_DL))

zi=np.empty([30,30,2])
xi=np.empty([31,2])
yi=np.empty([31,2])


#zi, yi, xi = np.histogram2d(Lat2, Lon2, bins=(30,30), weights=Data2, normed=True)
zi[:,:,0], yi[:,0], xi[:,0] = np.histogram2d(Lat1, Lon1, bins=(30,30), weights=Data1, normed=True)

zi[:,:,1], yi[:,1], xi[:,1] = np.histogram2d(Lat2, Lon2, bins=(30,30), weights=Data2, normed=True)

#*****************************************************************************\
#Creating mapa Frequency
#*****************************************************************************\

from mpl_toolkits.axes_grid1 import make_axes_locatable



row = 1
column = 2
fig, ax = plt.subplots(row, column, facecolor='w', figsize=(11,5))

#fig.suptitle('Title of fig', fontsize=24)

for i, ax in enumerate(ax.flat, start=1):
  print i
  ax.set_title('Subplot {}'.format(i))

  # plots random data using the 'jet' colormap
  #img = ax.contourf(np.random.rand(10,10),20,cmap=plt.cm.jet)
  img = ax.pcolor(xi[:,i-1], yi[:,i-1], zi[:,:,i-1], cmap='BuGn')
  # creates a new axis, cax, located 0.05 inches to the right of ax, whose width is 15% of ax
  # cax is used to plot a colorbar for each subplot
  div = make_axes_locatable(ax)
  cax = div.append_axes("right", size="6%", pad=0.05)
  #cbar = plt.colorbar(img, cax=cax, ticks=np.arange(0,10,.1), format="%.2g")
  cbar = plt.colorbar(img, cax=cax, format="%.2g")
  #cbar.set_label('Colorbar {}'.format(i), size=10)

  ax.set_yticks([0], minor=True)
  ax.set_xticks([0], minor=True)
  ax.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
  ax.grid(b=True, which='major', color='grey', linestyle='--')
  ax.set_ylabel('Distance from low (deg)')
  ax.set_xlabel('Distance from low (deg)')
  ax.margins(0.05)

  # removes x and y ticks
  #ax.xaxis.set_visible(False)
  #ax.yaxis.set_visible(False)

# prevents each subplot's axes from overlapping
plt.tight_layout()

# moves subplots down slightly to make room for the figure title
plt.subplots_adjust(top=0.9)
plt.show()

















#*****************************************************************************\
#Creating mapa Frequency
#*****************************************************************************\
#Extrar variables


# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
# ax1, ax2, ax3, ax4 = axes.ravel()


# Lat=np.array(df_BL['dy'])
# Lon=np.array(df_BL['dx'])
# Data=np.ones(len(df_BL))

# zi, yi, xi = np.histogram2d(Lat, Lon, bins=(30,30), weights=Data, normed=True)
# counts, _, _ = np.histogram2d(Lat, Lon, bins=(30,30))


# fig1=ax1.pcolor(xi, yi, zi, cmap='BuGn')
# #fig1=ax.pcolormesh(xi, yi, zi, cmap='BuGn',edgecolors='black',linewidth=0.01,rasterized=True)
# ax1.set_yticks([0], minor=True)
# ax1.set_xticks([0], minor=True)
# ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
# ax1.grid(b=True, which='major', color='grey', linestyle='--')
# ax1.set_ylabel('Distance from low (deg)')
# ax1.set_xlabel('Distance from low (deg)')
# cbar=plt.colorbar(fig1)
# #cb.set_label('Frequency')
# ax1.margins(0.05)



# #del xi,yi,xi,Data, Lat, Lon
# #..............................................................................

# Lat=np.array(df_DL['dy'])
# Lon=np.array(df_DL['dx'])
# Data=np.ones(len(df_DL))

# zi, yi, xi = np.histogram2d(Lat, Lon, bins=(30,30), weights=Data, normed=True)
# counts, _, _ = np.histogram2d(Lat, Lon, bins=(30,30))

# figu=ax2.pcolor(xi, yi, zi, cmap='BuGn')
# #fig1=ax.pcolormesh(xi, yi, zi, cmap='BuGn',edgecolors='black',linewidth=0.01,rasterized=True)
# ax2.set_yticks([0], minor=True)
# ax2.set_xticks([0], minor=True)
# ax2.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
# ax2.grid(b=True, which='major', color='grey', linestyle='--')
# ax2.set_ylabel('Distance from low (deg)')
# ax2.set_xlabel('Distance from low (deg)')
# # cb=plt.colorbar(figu)
# # cb.set_label('Frequency')
# ax2.margins(0.05)


# plt.show()








#*****************************************************************************\
#Creating mapa Height
#*****************************************************************************\
# #Extrar variables
# Lat=np.array(df_BL['dy'])
# Lon=np.array(df_BL['dx'])
# Data = np.array(df_BL['Depth'])

# zi, yi, xi = np.histogram2d(Lat, Lon, bins=(30,30), weights=Data, normed=False)
# counts, _, _ = np.histogram2d(Lat, Lon, bins=(30,30))

# zi = zi / counts
# zi = np.ma.masked_invalid(zi)

# fig, ax = plt.subplots()

# fig1=ax.pcolor(xi, yi, zi, cmap='BuGn')
# #fig1=ax.pcolormesh(xi, yi, zi, cmap='BuGn',edgecolors='black',linewidth=0.01,rasterized=True)
# ax.set_yticks([0], minor=True)
# ax.set_xticks([0], minor=True)
# ax.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
# ax.grid(b=True, which='major', color='grey', linestyle='--')
# ax.set_ylabel('Distance from low (deg)')
# ax.set_xlabel('Distance from low (deg)')
# cb=fig.colorbar(fig1)
# cb.set_label('height (mts.)')
# ax.margins(0.05)
# plt.show()

