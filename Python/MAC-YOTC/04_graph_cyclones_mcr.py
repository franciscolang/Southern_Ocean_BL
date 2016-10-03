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
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/MCR/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/cyclones/MCR/'
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

df_1i= df_maccyc[np.isfinite(df_maccyc['1ra Inv'])]
df_1inv = df_1i[np.isfinite(df_1i['dy'])]


#*****************************************************************************\
#YOTC
#*****************************************************************************\

df_1iy= df_yotccyc[np.isfinite(df_yotccyc['1ra Inv'])]
df_1invy = df_1iy[np.isfinite(df_1iy['dy'])]



#*****************************************************************************\
#MAC levels YOTC
#*****************************************************************************\
df_1imy= df_mycyc[np.isfinite(df_mycyc['1ra Inv'])]
df_1invmy = df_1imy[np.isfinite(df_1imy['dy'])]

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
s_bin=12


zi, yi, xi = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
counts, _, _ = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin))
zi = zi / counts
zi = np.ma.masked_invalid(zi)


zi1, yi1, xi1 = np.histogram2d(Lat2, Lon2, bins=(s_bin,s_bin), weights=Data2, normed=False)
counts1, _, _ = np.histogram2d(Lat2, Lon2, bins=(s_bin,s_bin))
zi1 = zi1 / counts1
zi1 = np.ma.masked_invalid(zi1)


zi2, yi2, xi2 = np.histogram2d(Lat3, Lon3, bins=(s_bin,s_bin), weights=Data3, normed=False)
counts2, _, _ = np.histogram2d(Lat3, Lon3, bins=(s_bin,s_bin))
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

#*****************************************************************************\

img1= ax0.pcolor(xi, yi, zi, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax0)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img1, cax=cax, format="%.0f")
# cbar.ax.set_title('     height (mts.)', size=12)
# cbar.ax.tick_params(labelsize=14)
ax0.set_title('MAC', size=18)
ax0.set_yticks([0], minor=True)
ax0.set_xticks([0], minor=True)
ax0.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax0.grid(b=True, which='major', color='grey', linestyle='--')
ax0.set_ylabel('Distance from low (deg)', size=18)
#ax0.set_xlabel('Distance from low (deg)', size=18)
ax0.margins(0.05)

img2= ax1.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax1)
ax1.set_title('YOTC', size=18)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img2, cax=cax, format="%.0f")
# cbar.ax.set_title('     height (mts.)', size=12)
# cbar.ax.tick_params(labelsize=14)
ax1.set_yticks([0], minor=True)
ax1.set_xticks([0], minor=True)
ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax1.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
ax1.set_xlabel('Distance from low (deg)', size=18)
ax1.margins(0.05)

v = np.arange(0, limu, 300)

img3= ax2.pcolor(xi2, yi2, zi2, cmap=cmap,vmin=0, vmax=vmax)
ax2.set_title('MAC$_{AVE}$', size=18)
div = make_axes_locatable(ax2)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img3, cax=cax, format="%.0f",ticks=v)
cbar.set_label(' height (mts.)', size=12)
#cbar.ax.set_title('                height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax2.set_yticks([0], minor=True)
ax2.set_xticks([0], minor=True)
ax2.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax2.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
#ax2.set_xlabel('Distance from low (deg)', size=18)
ax2.margins(0.05)
plt.tight_layout()

fig.savefig(path_data_save + 'height_1inv_soundings.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                       Creating mapa Strenght
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Main Inversion
#*****************************************************************************\
#*****************************************************************************\
#MAC
#*****************************************************************************\

df_1i= df_maccyc[np.isfinite(df_maccyc['Strg 1inv'])]
df_1str = df_1i[np.isfinite(df_1i['dy'])]


#*****************************************************************************\
#YOTC
#*****************************************************************************\

df_1iy= df_yotccyc[np.isfinite(df_yotccyc['Strg 1inv'])]
df_1stry = df_1iy[np.isfinite(df_1iy['dy'])]



#*****************************************************************************\
#MAC levels YOTC
#*****************************************************************************\
df_1imy= df_mycyc[np.isfinite(df_mycyc['Strg 1inv'])]
df_1strmy = df_1imy[np.isfinite(df_1imy['dy'])]

#*****************************************************************************\

Lat1=np.array(df_1str['dy'])
Lon1=np.array(df_1str['dx'])
Data1 = np.array(df_1str['Strg 1inv'])

Lat2=np.array(df_1stry['dy'])
Lon2=np.array(df_1stry['dx'])
Data2 = np.array(df_1invy['Strg 1inv'])

Lat3=np.array(df_1strmy['dy'])
Lon3=np.array(df_1strmy['dx'])
Data3 = np.array(df_1strmy['Strg 1inv'])

#*****************************************************************************\

zi, yi, xi = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
counts, _, _ = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin))
zi = zi / counts
zi = np.ma.masked_invalid(zi)


zi1, yi1, xi1 = np.histogram2d(Lat2, Lon2, bins=(s_bin,s_bin), weights=Data2, normed=False)
counts1, _, _ = np.histogram2d(Lat2, Lon2, bins=(s_bin,s_bin))
zi1 = zi1 / counts1
zi1 = np.ma.masked_invalid(zi1)


zi2, yi2, xi2 = np.histogram2d(Lat3, Lon3, bins=(s_bin,s_bin), weights=Data3, normed=False)
counts2, _, _ = np.histogram2d(Lat3, Lon3, bins=(s_bin,s_bin))
zi2 = zi2 / counts2
zi2 = np.ma.masked_invalid(zi2)
#*****************************************************************************\
row = 1
column = 3
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(17,5))
ax0, ax1, ax2 = axes.flat
cmap='Reds'
vmax=0.08
limu=vmax+0.02

#*****************************************************************************\

img1= ax0.pcolormesh(xi, yi, zi, cmap=cmap,vmin=0, vmax=vmax)
#img1= ax0.imshow(zi, cmap=cmap,vmin=0, vmax=vmax, interpolation = 'bicubic')


div = make_axes_locatable(ax0)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img1, cax=cax, format="%.0f")
# cbar.ax.set_title('     height (mts.)', size=12)
# cbar.ax.tick_params(labelsize=14)
ax0.set_title('MAC', size=18)
ax0.set_yticks([0], minor=True)
ax0.set_xticks([0], minor=True)
ax0.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax0.grid(b=True, which='major', color='grey', linestyle='--')
ax0.set_ylabel('Distance from low (deg)', size=18)
#ax0.set_xlabel('Distance from low (deg)', size=18)
ax0.margins(0.05)

img2= ax1.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax1)
ax1.set_title('YOTC', size=18)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img2, cax=cax, format="%.0f")
# cbar.ax.set_title('     height (mts.)', size=12)
# cbar.ax.tick_params(labelsize=14)
ax1.set_yticks([0], minor=True)
ax1.set_xticks([0], minor=True)
ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax1.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
ax1.set_xlabel('Distance from low (deg)', size=18)
ax1.margins(0.05)

v = np.arange(0, limu, 0.01)

img3= ax2.pcolor(xi2, yi2, zi2, cmap=cmap,vmin=0, vmax=vmax)
ax2.set_title('MAC$_{AVE}$', size=18)
div = make_axes_locatable(ax2)
cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img3, cax=cax, format="%.3f",ticks=v)
cbar.set_label('strength (K m$^{-1}$)', size=12)
#cbar.ax.set_title('                height (mts.)', size=12)
cbar.ax.tick_params(labelsize=14)
ax2.set_yticks([0], minor=True)
ax2.set_xticks([0], minor=True)
ax2.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax2.grid(b=True, which='major', color='grey', linestyle='--')
#ax1.set_ylabel('Distance from low (deg)', size=18)
#ax2.set_xlabel('Distance from low (deg)', size=18)
ax2.margins(0.05)
plt.tight_layout()

fig.savefig(path_data_save + 'strenght_1inv_soundings.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                       Height Seasonal
#*****************************************************************************\
#*****************************************************************************
#*****************************************************************************\
#*****************************************************************************\
df1inv_DJF=df_1inv[(df_1inv.index.month==12) | (df_1inv.index.month==1) | (df_1inv.index.month==2)]
df1inv_MAM=df_1inv[(df_1inv.index.month==3) | (df_1inv.index.month==4) | (df_1inv.index.month==5)]
df1inv_JJA=df_1inv[(df_1inv.index.month==6) | (df_1inv.index.month==7) | (df_1inv.index.month==8)]
df1inv_SON=df_1inv[(df_1inv.index.month==9) | (df_1inv.index.month==10) | (df_1inv.index.month==11)]

df1invy_DJF=df_1invy[(df_1invy.index.month==12) | (df_1invy.index.month==1) | (df_1invy.index.month==2)]
df1invy_MAM=df_1invy[(df_1invy.index.month==3) | (df_1invy.index.month==4) | (df_1invy.index.month==5)]
df1invy_JJA=df_1invy[(df_1invy.index.month==6) | (df_1invy.index.month==7) | (df_1invy.index.month==8)]
df1invy_SON=df_1invy[(df_1invy.index.month==9) | (df_1invy.index.month==10) | (df_1invy.index.month==11)]


df1invmy_DJF=df_1invmy[(df_1invmy.index.month==12) | (df_1invmy.index.month==1) | (df_1invmy.index.month==2)]
df1invmy_MAM=df_1invmy[(df_1invmy.index.month==3) | (df_1invmy.index.month==4) | (df_1invmy.index.month==5)]
df1invmy_JJA=df_1invmy[(df_1invmy.index.month==6) | (df_1invmy.index.month==7) | (df_1invmy.index.month==8)]
df1invmy_SON=df_1invmy[(df_1invmy.index.month==9) | (df_1invmy.index.month==10) | (df_1invmy.index.month==11)]



mean_1inv_DJF=np.nanmean(df1inv_DJF['1ra Inv'])
mean_1inv_MAM=np.nanmean(df1inv_MAM['1ra Inv'])
mean_1inv_JJA=np.nanmean(df1inv_JJA['1ra Inv'])
mean_1inv_SON=np.nanmean(df1inv_SON['1ra Inv'])

mean_1invy_DJF=np.nanmean(df1invy_DJF['1ra Inv'])
mean_1invy_MAM=np.nanmean(df1invy_MAM['1ra Inv'])
mean_1invy_JJA=np.nanmean(df1invy_JJA['1ra Inv'])
mean_1invy_SON=np.nanmean(df1invy_SON['1ra Inv'])

mean_1invmy_DJF=np.nanmean(df1invmy_DJF['1ra Inv'])
mean_1invmy_MAM=np.nanmean(df1invmy_MAM['1ra Inv'])
mean_1invmy_JJA=np.nanmean(df1invmy_JJA['1ra Inv'])
mean_1invmy_SON=np.nanmean(df1invmy_SON['1ra Inv'])
#*****************************************************************************\
std_1inv_DJF=np.nanstd(df1inv_DJF['1ra Inv'])
std_1inv_MAM=np.nanstd(df1inv_MAM['1ra Inv'])
std_1inv_JJA=np.nanstd(df1inv_JJA['1ra Inv'])
std_1inv_SON=np.nanstd(df1inv_SON['1ra Inv'])

std_1invy_DJF=np.nanstd(df1invy_DJF['1ra Inv'])
std_1invy_MAM=np.nanstd(df1invy_MAM['1ra Inv'])
std_1invy_JJA=np.nanstd(df1invy_JJA['1ra Inv'])
std_1invy_SON=np.nanstd(df1invy_SON['1ra Inv'])

std_1invmy_DJF=np.nanstd(df1invmy_DJF['1ra Inv'])
std_1invmy_MAM=np.nanstd(df1invmy_MAM['1ra Inv'])
std_1invmy_JJA=np.nanstd(df1invmy_JJA['1ra Inv'])
std_1invmy_SON=np.nanstd(df1invmy_SON['1ra Inv'])

#*****************************************************************************\

DJF_ST=[std_1inv_DJF, std_1invy_DJF, std_1invmy_DJF]
MAM_ST=[std_1inv_MAM, std_1invy_MAM, std_1invmy_MAM]
JJA_ST=[std_1inv_JJA, std_1invy_JJA, std_1invmy_JJA]
SON_ST=[std_1inv_SON, std_1invy_SON, std_1invmy_SON]

raw_data1 = {'sounding': ['MAC', 'YOTC', 'MAC$_{AVE}$'],
        'DJF': [mean_1inv_DJF,mean_1invy_DJF,mean_1invmy_DJF],
        'MAM': [mean_1inv_MAM,mean_1invy_MAM,mean_1invmy_MAM],
        'JJA': [mean_1inv_JJA,mean_1invy_JJA,mean_1invmy_JJA],
        'SON': [mean_1inv_SON,mean_1invy_SON,mean_1invmy_SON]}

df1inv = pd.DataFrame(raw_data1, columns = ['sounding', 'DJF', 'MAM', 'JJA','SON'])



# Setting Plot
fig, ax = plt.subplots(figsize=(8, 5))

width = 0.2
pos = list(range(len(df1inv['DJF'])))
error_config = {'ecolor': '0.4'}

#*****************************************************************************\
# First Inversion (Main)
#*****************************************************************************\
plt.bar(pos,
        #using df['pre_score'] data,
        df1inv['DJF'],
        # of width
        width, yerr=[(0,0,0), DJF_ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#c6c9d0',
        #color='#7C83AF',
        # with label the first value in first_name
        label='DJF')


plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        df1inv['MAM'],
        # of width
        width, yerr=[(0,0,0), MAM_ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#67832F',
        # with label the second value in first_name
        label='MAM')

plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        df1inv['JJA'],
        # of width
        width, yerr=[(0,0,0), JJA_ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='tomato',
        #color='#182157',
        # with label the third value in first_name
        label='JJA')

plt.bar([p + width*3 for p in pos],
        #using df['post_score'] data,
        df1inv['SON'],
        # of width
        width, yerr=[(0,0,0), SON_ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='blue',
        #color='#182157',
        # with label the third value in first_name
        label='SON')


# Set the position of the x ticks
ax.set_xticks([p + 2 * width for p in pos])
# Set the labels for the x ticks
ax.set_xticklabels(df1inv['sounding'],fontsize = 13)
# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*5)
ax.set_ylabel('height (mts.)',fontsize = 13)
#Legend
plt.legend(['DJF', 'MAM', 'JJA','SON'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
plt.grid()
ax.set_yticks(np.arange(0,3900,600))
plt.ylim(0,3600)
fig.savefig(path_data_save + 'height_1inv_season.eps', format='eps', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Height Quadrant
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
df1inv_Q1=df_1inv[(df_1inv['dx']>0) & (df_1inv['dy']>0)]
df1inv_Q2=df_1inv[(df_1inv['dx']<0) & (df_1inv['dy']>0)]
df1inv_Q3=df_1inv[(df_1inv['dx']<0) & (df_1inv['dy']<0)]
df1inv_Q4=df_1inv[(df_1inv['dx']>0) & (df_1inv['dy']<0)]


df1invmy_Q1=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']>0)]
df1invmy_Q2=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']>0)]
df1invmy_Q3=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']<0)]
df1invmy_Q4=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']<0)]



df1invy_Q1=df_1invy[(df_1invy['dx']>0) & (df_1invy['dy']>0)]
df1invy_Q2=df_1invy[(df_1invy['dx']<0) & (df_1invy['dy']>0)]
df1invy_Q3=df_1invy[(df_1invy['dx']<0) & (df_1invy['dy']<0)]
df1invy_Q4=df_1invy[(df_1invy['dx']>0) & (df_1invy['dy']<0)]


#*****************************************************************************
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
#*****************************************************************************
Q1_1inv_std=np.nanstd(df1inv_Q1['1ra Inv'])
Q2_1inv_std=np.nanstd(df1inv_Q2['1ra Inv'])
Q3_1inv_std=np.nanstd(df1inv_Q3['1ra Inv'])
Q4_1inv_std=np.nanstd(df1inv_Q4['1ra Inv'])

Q1_1invy_std=np.nanstd(df1invy_Q1['1ra Inv'])
Q2_1invy_std=np.nanstd(df1invy_Q2['1ra Inv'])
Q3_1invy_std=np.nanstd(df1invy_Q3['1ra Inv'])
Q4_1invy_std=np.nanstd(df1invy_Q4['1ra Inv'])

Q1_1invmy_std=np.nanstd(df1invmy_Q1['1ra Inv'])
Q2_1invmy_std=np.nanstd(df1invmy_Q2['1ra Inv'])
Q3_1invmy_std=np.nanstd(df1invmy_Q3['1ra Inv'])
Q4_1invmy_std=np.nanstd(df1invmy_Q4['1ra Inv'])
#*****************************************************************************
Q1ST=[Q1_1inv_std, Q1_1invy_std, Q1_1invmy_std]
Q2ST=[Q2_1inv_std, Q2_1invy_std, Q2_1invmy_std]
Q3ST=[Q3_1inv_std, Q3_1invy_std, Q3_1invmy_std]
Q4ST=[Q4_1inv_std, Q4_1invy_std, Q4_1invmy_std]

raw_data1 = {'sounding': ['MAC', 'YOTC', 'MAC$_{AVE}$'],
        'Q1': [Q1_1inv, Q1_1invy, Q1_1invmy],
        'Q2': [Q2_1inv, Q2_1invy, Q2_1invmy],
        'Q3': [Q3_1inv, Q3_1invy, Q3_1invmy],
        'Q4': [Q4_1inv, Q4_1invy, Q4_1invmy]}

df1invQ = pd.DataFrame(raw_data1, columns = ['sounding', 'Q1', 'Q2', 'Q3','Q4'])

# Setting Plot
fig, ax = plt.subplots(figsize=(8, 5))

width = 0.2
pos = list(range(len(df1invQ['Q1'])))
error_config = {'ecolor': '0.4'}
#*****************************************************************************\
plt.bar(pos,
        #using df['pre_score'] data,
        df1invQ['Q1'],
        # of width
        width, yerr=[(0,0,0), Q1ST], error_kw=error_config,
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
        width, yerr=[(0,0,0), Q2ST], error_kw=error_config,
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
        width, yerr=[(0,0,0), Q3ST], error_kw=error_config,
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
        width, yerr=[(0,0,0), Q4ST], error_kw=error_config,
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
ax.set_ylabel('height (mts.)',fontsize = 13)
#Legend
plt.legend(['NE', 'NW', 'SW','SE'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
plt.grid()
ax.set_yticks(np.arange(0,3900,600))
plt.ylim(0,3600)
fig.savefig(path_data_save + 'height_1inv_quadrant.eps', format='eps', dpi=1200)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                      Strength Seasonal
#*****************************************************************************\
#*****************************************************************************
#*****************************************************************************\
#*****************************************************************************\
df1str_DJF=df_1str[(df_1str.index.month==12) | (df_1str.index.month==1) | (df_1str.index.month==2)]
df1str_MAM=df_1str[(df_1str.index.month==3) | (df_1str.index.month==4) | (df_1str.index.month==5)]
df1str_JJA=df_1str[(df_1str.index.month==6) | (df_1str.index.month==7) | (df_1str.index.month==8)]
df1str_SON=df_1str[(df_1str.index.month==9) | (df_1str.index.month==10) | (df_1str.index.month==11)]

df1stry_DJF=df_1stry[(df_1stry.index.month==12) | (df_1stry.index.month==1) | (df_1stry.index.month==2)]
df1stry_MAM=df_1stry[(df_1stry.index.month==3) | (df_1stry.index.month==4) | (df_1stry.index.month==5)]
df1stry_JJA=df_1stry[(df_1stry.index.month==6) | (df_1stry.index.month==7) | (df_1stry.index.month==8)]
df1stry_SON=df_1stry[(df_1stry.index.month==9) | (df_1stry.index.month==10) | (df_1stry.index.month==11)]


df1strmy_DJF=df_1strmy[(df_1strmy.index.month==12) | (df_1strmy.index.month==1) | (df_1strmy.index.month==2)]
df1strmy_MAM=df_1strmy[(df_1strmy.index.month==3) | (df_1strmy.index.month==4) | (df_1strmy.index.month==5)]
df1strmy_JJA=df_1strmy[(df_1strmy.index.month==6) | (df_1strmy.index.month==7) | (df_1strmy.index.month==8)]
df1strmy_SON=df_1strmy[(df_1strmy.index.month==9) | (df_1strmy.index.month==10) | (df_1strmy.index.month==11)]



mean_1str_DJF=np.nanmean(df1str_DJF['Strg 1inv'])
mean_1str_MAM=np.nanmean(df1str_MAM['Strg 1inv'])
mean_1str_JJA=np.nanmean(df1str_JJA['Strg 1inv'])
mean_1str_SON=np.nanmean(df1str_SON['Strg 1inv'])

mean_1stry_DJF=np.nanmean(df1stry_DJF['Strg 1inv'])
mean_1stry_MAM=np.nanmean(df1stry_MAM['Strg 1inv'])
mean_1stry_JJA=np.nanmean(df1stry_JJA['Strg 1inv'])
mean_1stry_SON=np.nanmean(df1stry_SON['Strg 1inv'])

mean_1strmy_DJF=np.nanmean(df1strmy_DJF['Strg 1inv'])
mean_1strmy_MAM=np.nanmean(df1strmy_MAM['Strg 1inv'])
mean_1strmy_JJA=np.nanmean(df1strmy_JJA['Strg 1inv'])
mean_1strmy_SON=np.nanmean(df1strmy_SON['Strg 1inv'])
#*****************************************************************************\
std_1str_DJF=np.nanstd(df1str_DJF['Strg 1inv'])
std_1str_MAM=np.nanstd(df1str_MAM['Strg 1inv'])
std_1str_JJA=np.nanstd(df1str_JJA['Strg 1inv'])
std_1str_SON=np.nanstd(df1str_SON['Strg 1inv'])

std_1stry_DJF=np.nanstd(df1stry_DJF['Strg 1inv'])
std_1stry_MAM=np.nanstd(df1stry_MAM['Strg 1inv'])
std_1stry_JJA=np.nanstd(df1stry_JJA['Strg 1inv'])
std_1stry_SON=np.nanstd(df1stry_SON['Strg 1inv'])

std_1strmy_DJF=np.nanstd(df1strmy_DJF['Strg 1inv'])
std_1strmy_MAM=np.nanstd(df1strmy_MAM['Strg 1inv'])
std_1strmy_JJA=np.nanstd(df1strmy_JJA['Strg 1inv'])
std_1strmy_SON=np.nanstd(df1strmy_SON['Strg 1inv'])

#*****************************************************************************\

DJF_ST=[std_1str_DJF, std_1stry_DJF, std_1strmy_DJF]
MAM_ST=[std_1str_MAM, std_1stry_MAM, std_1strmy_MAM]
JJA_ST=[std_1str_JJA, std_1stry_JJA, std_1strmy_JJA]
SON_ST=[std_1str_SON, std_1stry_SON, std_1strmy_SON]
MAM_ST=DJF_ST

raw_data1 = {'sounding': ['MAC', 'YOTC', 'MAC$_{AVE}$'],
        'DJF': [mean_1str_DJF,mean_1stry_DJF,mean_1strmy_DJF],
        'MAM': [mean_1str_MAM,mean_1stry_MAM,mean_1strmy_MAM],
        'JJA': [mean_1str_JJA,mean_1stry_JJA,mean_1strmy_JJA],
        'SON': [mean_1str_SON,mean_1stry_SON,mean_1strmy_SON]}

df1str = pd.DataFrame(raw_data1, columns = ['sounding', 'DJF', 'MAM', 'JJA','SON'])



# Setting Plot
fig, ax = plt.subplots(figsize=(8, 5))

width = 0.2
pos = list(range(len(df1str['DJF'])))
error_config = {'ecolor': '0.4'}

#*****************************************************************************\
# First Inversion (Main)
#*****************************************************************************\
plt.bar(pos,
        #using df['pre_score'] data,
        df1str['DJF'],
        # of width
        width, yerr=[(0,0,0), DJF_ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#c6c9d0',
        #color='#7C83AF',
        # with label the first value in first_name
        label='DJF')


plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        df1str['MAM'],
        # of width
        width, yerr=[(0,0,0), MAM_ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#67832F',
        # with label the second value in first_name
        label='MAM')

plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        df1str['JJA'],
        # of width
        width, yerr=[(0,0,0), JJA_ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='tomato',
        #color='#182157',
        # with label the third value in first_name
        label='JJA')

plt.bar([p + width*3 for p in pos],
        #using df['post_score'] data,
        df1str['SON'],
        # of width
        width, yerr=[(0,0,0), SON_ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='blue',
        #color='#182157',
        # with label the third value in first_name
        label='SON')


# Set the position of the x ticks
ax.set_xticks([p + 2 * width for p in pos])
# Set the labels for the x ticks
ax.set_xticklabels(df1str['sounding'],fontsize = 13)
# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*5)
ax.set_ylabel('strength (K m$^{-1}$)',fontsize = 13)
#Legend
plt.legend(['DJF', 'MAM', 'JJA','SON'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
plt.grid()
ax.set_yticks(np.arange(0,0.09,0.01))
plt.ylim(0,0.08)
fig.savefig(path_data_save + 'strg_1inv_season.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Strength Quadrant
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
df1str_Q1=df_1str[(df_1str['dx']>0) & (df_1str['dy']>0)]
df1str_Q2=df_1str[(df_1str['dx']<0) & (df_1str['dy']>0)]
df1str_Q3=df_1str[(df_1str['dx']<0) & (df_1str['dy']<0)]
df1str_Q4=df_1str[(df_1str['dx']>0) & (df_1str['dy']<0)]


df1strmy_Q1=df_1strmy[(df_1strmy['dx']>0) & (df_1strmy['dy']>0)]
df1strmy_Q2=df_1strmy[(df_1strmy['dx']<0) & (df_1strmy['dy']>0)]
df1strmy_Q3=df_1strmy[(df_1strmy['dx']<0) & (df_1strmy['dy']<0)]
df1strmy_Q4=df_1strmy[(df_1strmy['dx']>0) & (df_1strmy['dy']<0)]



df1stry_Q1=df_1stry[(df_1stry['dx']>0) & (df_1stry['dy']>0)]
df1stry_Q2=df_1stry[(df_1stry['dx']<0) & (df_1stry['dy']>0)]
df1stry_Q3=df_1stry[(df_1stry['dx']<0) & (df_1stry['dy']<0)]
df1stry_Q4=df_1stry[(df_1stry['dx']>0) & (df_1stry['dy']<0)]


#*****************************************************************************
Q1_1str=np.nanmean(df1str_Q1['Strg 1inv'])
Q2_1str=np.nanmean(df1str_Q2['Strg 1inv'])
Q3_1str=np.nanmean(df1str_Q3['Strg 1inv'])
Q4_1str=np.nanmean(df1str_Q4['Strg 1inv'])

Q1_1stry=np.nanmean(df1stry_Q1['Strg 1inv'])
Q2_1stry=np.nanmean(df1stry_Q2['Strg 1inv'])
Q3_1stry=np.nanmean(df1stry_Q3['Strg 1inv'])
Q4_1stry=np.nanmean(df1stry_Q4['Strg 1inv'])

Q1_1strmy=np.nanmean(df1strmy_Q1['Strg 1inv'])
Q2_1strmy=np.nanmean(df1strmy_Q2['Strg 1inv'])
Q3_1strmy=np.nanmean(df1strmy_Q3['Strg 1inv'])
Q4_1strmy=np.nanmean(df1strmy_Q4['Strg 1inv'])
#*****************************************************************************
Q1_1str_std=np.nanstd(df1str_Q1['Strg 1inv'])
Q2_1str_std=np.nanstd(df1str_Q2['Strg 1inv'])
Q3_1str_std=np.nanstd(df1str_Q3['Strg 1inv'])
Q4_1str_std=np.nanstd(df1str_Q4['Strg 1inv'])

Q1_1stry_std=np.nanstd(df1stry_Q1['Strg 1inv'])
Q2_1stry_std=np.nanstd(df1stry_Q2['Strg 1inv'])
Q3_1stry_std=np.nanstd(df1stry_Q3['Strg 1inv'])
Q4_1stry_std=np.nanstd(df1stry_Q4['Strg 1inv'])

Q1_1strmy_std=np.nanstd(df1strmy_Q1['Strg 1inv'])
Q2_1strmy_std=np.nanstd(df1strmy_Q2['Strg 1inv'])
Q3_1strmy_std=np.nanstd(df1strmy_Q3['Strg 1inv'])
Q4_1strmy_std=np.nanstd(df1strmy_Q4['Strg 1inv'])
#*****************************************************************************
Q1ST=[Q1_1str_std, Q1_1stry_std, Q1_1strmy_std]
Q2ST=[Q2_1str_std, Q2_1stry_std, Q2_1strmy_std]
Q3ST=[Q3_1str_std, Q3_1stry_std, Q3_1strmy_std]
Q4ST=[Q4_1str_std, Q4_1stry_std, Q4_1strmy_std]

Q1ST=Q3ST

raw_data1 = {'sounding': ['MAC', 'YOTC', 'MAC$_{AVE}$'],
        'Q1': [Q1_1str, Q1_1stry, Q1_1strmy],
        'Q2': [Q2_1str, Q2_1stry, Q2_1strmy],
        'Q3': [Q3_1str, Q3_1stry, Q3_1strmy],
        'Q4': [Q4_1str, Q4_1stry, Q4_1strmy]}

df1strQ = pd.DataFrame(raw_data1, columns = ['sounding', 'Q1', 'Q2', 'Q3','Q4'])

# Setting Plot
fig, ax = plt.subplots(figsize=(8, 5))

width = 0.2
pos = list(range(len(df1strQ['Q1'])))
error_config = {'ecolor': '0.4'}
#*****************************************************************************\
plt.bar(pos,
        #using df['pre_score'] data,
        df1strQ['Q1'],
        # of width
        width, yerr=[(0,0,0), Q1ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#c6c9d0',
        #color='#7C83AF',
        # with label the first value in first_name
        label='Q1')


plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        df1strQ['Q2'],
        # of width
        width, yerr=[(0,0,0), Q2ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#67832F',
        # with label the second value in first_name
        label='Q2')

plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        df1strQ['Q3'],
        # of width
        width, yerr=[(0,0,0), Q3ST], error_kw=error_config,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='tomato',
        #color='#182157',
        # with label the third value in first_name
        label='Q3')

plt.bar([p + width*3 for p in pos],
        #using df['post_score'] data,
        df1strQ['Q4'],
        # of width
        width, yerr=[(0,0,0), Q4ST], error_kw=error_config,
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
ax.set_xticklabels(df1strQ['sounding'],fontsize = 13)
# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*5)
ax.set_ylabel('strength (K m$^{-1}$)',fontsize = 13)
#Legend
plt.legend(['NE', 'NW', 'SW','SE'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
plt.grid()
ax.set_yticks(np.arange(0,0.09,0.01))
plt.ylim(0,0.08)
fig.savefig(path_data_save + 'strength_1str_quadrant.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Relative Freq Height Quadrant
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
df1inv_Q1=df_1inv[(df_1inv['dx']>0) & (df_1inv['dy']>0)]
df1inv_Q2=df_1inv[(df_1inv['dx']<0) & (df_1inv['dy']>0)]
df1inv_Q3=df_1inv[(df_1inv['dx']<0) & (df_1inv['dy']<0)]
df1inv_Q4=df_1inv[(df_1inv['dx']>0) & (df_1inv['dy']<0)]


df1invmy_Q1=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']>0)]
df1invmy_Q2=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']>0)]
df1invmy_Q3=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']<0)]
df1invmy_Q4=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']<0)]



df1invy_Q1=df_1invy[(df_1invy['dx']>0) & (df_1invy['dy']>0)]
df1invy_Q2=df_1invy[(df_1invy['dx']<0) & (df_1invy['dy']>0)]
df1invy_Q3=df_1invy[(df_1invy['dx']<0) & (df_1invy['dy']<0)]
df1invy_Q4=df_1invy[(df_1invy['dx']>0) & (df_1invy['dy']<0)]


#*****************************************************************************
Q1_1inv=len(df1inv_Q1)/float(len(df_1inv))
Q2_1inv=len(df1inv_Q2)/float(len(df_1inv))
Q3_1inv=len(df1inv_Q3)/float(len(df_1inv))
Q4_1inv=len(df1inv_Q4)/float(len(df_1inv))

Q1_1invy=len(df1invy_Q1)/float(len(df_1invy))
Q2_1invy=len(df1invy_Q2)/float(len(df_1invy))
Q3_1invy=len(df1invy_Q3)/float(len(df_1invy))
Q4_1invy=len(df1invy_Q4)/float(len(df_1invy))



Q1_1invmy=len(df1invmy_Q1)/float(len(df_1invmy))
Q2_1invmy=len(df1invmy_Q2)/float(len(df_1invmy))
Q3_1invmy=len(df1invmy_Q3)/float(len(df_1invmy))
Q4_1invmy=len(df1invmy_Q4)/float(len(df_1invmy))
#*****************************************************************************

raw_data1 = {'sounding': ['MAC', 'YOTC', 'MAC$_{AVE}$'],
        'Q1': [Q1_1inv, Q1_1invy, Q1_1invmy],
        'Q2': [Q2_1inv, Q2_1invy, Q2_1invmy],
        'Q3': [Q3_1inv, Q3_1invy, Q3_1invmy],
        'Q4': [Q4_1inv, Q4_1invy, Q4_1invmy]}

df1invQ = pd.DataFrame(raw_data1, columns = ['sounding', 'Q1', 'Q2', 'Q3','Q4'])

# Setting Plot
fig, ax = plt.subplots(figsize=(8, 5))

width = 0.2
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
ax.set_ylabel('height (mts.)',fontsize = 13)
#Legend
plt.legend(['NE', 'NW', 'SW','SE'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
plt.grid()
ax.set_yticks(np.arange(0,1.1,0.2))
plt.ylim(0,1)
#fig.savefig(path_data_save + 'height_1inv_quadrant.eps', format='eps', dpi=1200)
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Relative Freq Height Quadrant
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
df1inv_Q1=df_1inv[(df_1inv['dx']>0) & (df_1inv['dy']>0)]
df1inv_Q2=df_1inv[(df_1inv['dx']<0) & (df_1inv['dy']>0)]
df1inv_Q3=df_1inv[(df_1inv['dx']<0) & (df_1inv['dy']<0)]
df1inv_Q4=df_1inv[(df_1inv['dx']>0) & (df_1inv['dy']<0)]


df1invmy_Q1=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']>0)]
df1invmy_Q2=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']>0)]
df1invmy_Q3=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']<0)]
df1invmy_Q4=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']<0)]



df1invy_Q1=df_1invy[(df_1invy['dx']>0) & (df_1invy['dy']>0)]
df1invy_Q2=df_1invy[(df_1invy['dx']<0) & (df_1invy['dy']>0)]
df1invy_Q3=df_1invy[(df_1invy['dx']<0) & (df_1invy['dy']<0)]
df1invy_Q4=df_1invy[(df_1invy['dx']>0) & (df_1invy['dy']<0)]


#*****************************************************************************
Q1_1inv=len(df1inv_Q1)/float(len(df_1inv))
Q2_1inv=len(df1inv_Q2)/float(len(df_1inv))
Q3_1inv=len(df1inv_Q3)/float(len(df_1inv))
Q4_1inv=len(df1inv_Q4)/float(len(df_1inv))

Q1_1invy=len(df1invy_Q1)/float(len(df_1invy))
Q2_1invy=len(df1invy_Q2)/float(len(df_1invy))
Q3_1invy=len(df1invy_Q3)/float(len(df_1invy))
Q4_1invy=len(df1invy_Q4)/float(len(df_1invy))

Q1_1invmy=len(df1invmy_Q1)/float(len(df_1invmy))
Q2_1invmy=len(df1invmy_Q2)/float(len(df_1invmy))
Q3_1invmy=len(df1invmy_Q3)/float(len(df_1invmy))
Q4_1invmy=len(df1invmy_Q4)/float(len(df_1invmy))
#*****************************************************************************

raw_data1 = {'sounding': ['MAC', 'YOTC', 'MAC$_{AVE}$'],
        'Q1': [Q1_1inv, Q1_1invy, Q1_1invmy],
        'Q2': [Q2_1inv, Q2_1invy, Q2_1invmy],
        'Q3': [Q3_1inv, Q3_1invy, Q3_1invmy],
        'Q4': [Q4_1inv, Q4_1invy, Q4_1invmy]}

df1invQ = pd.DataFrame(raw_data1, columns = ['sounding', 'Q1', 'Q2', 'Q3','Q4'])

# Setting Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Set the bar width
bar_width = 0.75

# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df1invQ['Q1']))]

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l]
#*****************************************************************************\
plt.bar(bar_l,
        #using df['pre_score'] data,
        df1invQ['Q1'],
        # of width
        width=bar_width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#c6c9d0',
        #color='#7C83AF',
        # with label the first value in first_name
        label='Q1')


plt.bar(bar_l,
        #using df['mid_score'] data,
        df1invQ['Q2'],
        # of width
        width=bar_width,
        bottom=df1invQ['Q1'],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#67832F',
        # with label the second value in first_name
        label='Q2')

plt.bar(bar_l,
        #using df['post_score'] data,
        df1invQ['Q3'],
        # of width
        width=bar_width,
        bottom=[i+j for i,j in zip(df1invQ['Q1'],df1invQ['Q2'])],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='tomato',
        #color='#182157',
        # with label the third value in first_name
        label='Q3')

plt.bar(bar_l,
        #using df['post_score'] data,
        df1invQ['Q4'],
        # of width
        width=bar_width,
        bottom=[i+j+z for i,j,z in zip(df1invQ['Q1'],df1invQ['Q2'],df1invQ['Q3'])],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='blue',
        #color='#182157',
        # with label the third value in first_name
        label='Q4')


plt.xticks(tick_pos, df1invQ['sounding'])


ax.set_ylabel('frequency by quadrant',fontsize = 13)
#Legend
plt.legend(['NE', 'NW', 'SW','SE'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize = 13)
plt.grid()
ax.set_yticks(np.arange(0,1.1,0.2))
plt.ylim(0,1)
# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
fig.savefig(path_data_save + 'freq_height_1inv_quadrant.eps', format='eps', dpi=1200)
plt.show()
