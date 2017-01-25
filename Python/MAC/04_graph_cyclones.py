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
import datetime
import scipy.stats

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/MCR/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/cyclones/'
latMac=-54.50;
lonMac=158.95;

#*****************************************************************************\
# Readinf CSV
#*****************************************************************************\
df_mycyc= pd.read_csv(path_data + 'df_mycyc_5k.csv', sep='\t', parse_dates=['Date'])

df_mycyc= df_mycyc.set_index('Date')

#*****************************************************************************\
#MAC levels YOTC
#*****************************************************************************\
df_1imy= df_mycyc[np.isfinite(df_mycyc['1ra Inv'])]
df_1invmy = df_1imy[np.isfinite(df_1imy['dy'])]


#*****************************************************************************\
#Percentage
#*****************************************************************************\
df_mycyc1= df_mycyc[np.isfinite(df_mycyc['Clas'])]
df_my_1 = df_mycyc1[np.isfinite(df_mycyc1['dy'])]

n_my=len(df_my_1)/float(len(df_mycyc1))*100

print n_my


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                       Creating mapa Height
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Main Inversion
#*****************************************************************************\

Lat1=np.array(df_1invmy['dy'])
Lon1=np.array(df_1invmy['dx'])
Data1 = np.array(df_1invmy['1ra Inv'])

#*****************************************************************************\
s_bin=12

zi1, yi1, xi1 = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
counts1, _, _ = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin))
zi1 = zi1 / counts1
zi1 = np.ma.masked_invalid(zi1)
#*****************************************************************************\
cmap='Blues'
vmax=2400
limu=vmax+300
#*****************************************************************************\

fig=plt.figure(figsize=(8, 6))
ax1=fig.add_subplot(111)


v = np.arange(0, limu, 300)
img1= ax1.pcolor(xi1, yi1, zi1, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax1)
ax1.set_title('MAC', size=16)


cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img1, cax=cax, format="%.0f",ticks=v)
cbar.set_label(' height (m)', size=12)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img2, cax=cax, format="%.0f")
# cbar.ax.set_title('     height (mts.)', size=12)
# cbar.ax.tick_params(labelsize=14)
ax1.set_yticks([0], minor=True)
ax1.set_xticks([0], minor=True)
ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax1.grid(b=True, which='major', color='grey', linestyle='--')
ax1.set_ylabel('Distance from low (deg)', size=14)
ax1.set_xlabel('Distance from low (deg)', size=14)
ax1.margins(0.05)

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
#MAC levels YOTC
#*****************************************************************************\
df_1imy= df_mycyc[np.isfinite(df_mycyc['Strg 1inv'])]
df_1strmy = df_1imy[np.isfinite(df_1imy['dy'])]

#*****************************************************************************\
del Lat1, Lon1, Data1
Lat1=np.array(df_1strmy['dy'])
Lon1=np.array(df_1strmy['dx'])
Data1 = np.array(df_1strmy['Strg 1inv'])

#*****************************************************************************\

zi, yi, xi = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin), weights=Data1, normed=False)
counts, _, _ = np.histogram2d(Lat1, Lon1, bins=(s_bin,s_bin))
zi = zi / counts
zi = np.ma.masked_invalid(zi)

#*****************************************************************************\
cmap='Reds'
vmax=0.08
limu=vmax+0.02

#*****************************************************************************\
fig=plt.figure(figsize=(8, 6))
ax1=fig.add_subplot(111)

v = np.arange(0, limu, 0.01)

img1= ax1.pcolor(xi, yi, zi, cmap=cmap,vmin=0, vmax=vmax)
div = make_axes_locatable(ax1)
ax1.set_title('MAC', size=16)

cax = div.append_axes("right", size="6%", pad=0.05)
cbar = plt.colorbar(img1, cax=cax, format="%.2f",ticks=v)
cbar.set_label('strength (K m$^{-1}$)', size=12)

ax1.set_yticks([0], minor=True)
ax1.set_xticks([0], minor=True)
ax1.grid(b=True, which='minor', color='k', linestyle='-',linewidth=1)
ax1.grid(b=True, which='major', color='grey', linestyle='--')
ax1.set_ylabel('Distance from low (deg)', size=14)
ax1.set_xlabel('Distance from low (deg)', size=14)
ax1.margins(0.05)


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
df_1invmy['distance']=np.sqrt(df_1invmy['dx']**2+df_1invmy['dy']**2)



df1invmy_DJF=df_1invmy[(df_1invmy.index.month==12) | (df_1invmy.index.month==1) | (df_1invmy.index.month==2)]
df1invmy_MAM=df_1invmy[(df_1invmy.index.month==3) | (df_1invmy.index.month==4) | (df_1invmy.index.month==5)]
df1invmy_JJA=df_1invmy[(df_1invmy.index.month==6) | (df_1invmy.index.month==7) | (df_1invmy.index.month==8)]
df1invmy_SON=df_1invmy[(df_1invmy.index.month==9) | (df_1invmy.index.month==10) | (df_1invmy.index.month==11)]


#*****************************************************************************\
# Box Plot
#*****************************************************************************\

data_to_plot = [df1invmy_DJF['1ra Inv'], df1invmy_MAM['1ra Inv'], df1invmy_JJA['1ra Inv'], df1invmy_SON['1ra Inv']]

fig= plt.figure(3, figsize=(9, 6))
ax = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot, whis=[5, 95], patch_artist=True)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_ylabel('Height (m)',fontsize = 16)
ax.set_xticklabels(['DJF', 'MAM', 'JJA','SON'],fontsize = 16)
ax.set_ylim(0,5000)
ax.set_yticks(np.arange(0,5500,500))
ax.set_yticklabels(np.arange(0,5500,500), size=16)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

for cap in bp['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')

fig.savefig(path_data_save + 'boxplot_hght1_season.eps', format='eps', dpi=1200)






#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Height by Season and Distance
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************




#DJF
df1invmy_DJF_5=df1invmy_DJF[(df1invmy_DJF['distance']>0) & (df1invmy_DJF['distance']<=5)]
df1invmy_DJF_10=df1invmy_DJF[(df1invmy_DJF['distance']>5) & (df1invmy_DJF['distance']<=10)]
df1invmy_DJF_15=df1invmy_DJF[(df1invmy_DJF['distance']>10) & (df1invmy_DJF['distance']<=15)]
#MAM
df1invmy_MAM_5=df1invmy_MAM[(df1invmy_MAM['distance']>0) & (df1invmy_MAM['distance']<=5)]
df1invmy_MAM_10=df1invmy_MAM[(df1invmy_MAM['distance']>5) & (df1invmy_MAM['distance']<=10)]
df1invmy_MAM_15=df1invmy_MAM[(df1invmy_MAM['distance']>10) & (df1invmy_MAM['distance']<=15)]

#JJA
df1invmy_JJA_5=df1invmy_JJA[(df1invmy_JJA['distance']>0) & (df1invmy_JJA['distance']<=5)]
df1invmy_JJA_10=df1invmy_JJA[(df1invmy_JJA['distance']>5) & (df1invmy_JJA['distance']<=10)]
df1invmy_JJA_15=df1invmy_JJA[(df1invmy_JJA['distance']>10) & (df1invmy_JJA['distance']<=15)]

#SON
df1invmy_SON_5=df1invmy_SON[(df1invmy_SON['distance']>0) & (df1invmy_SON['distance']<=5)]
df1invmy_SON_10=df1invmy_SON[(df1invmy_SON['distance']>5) & (df1invmy_SON['distance']<=10)]
df1invmy_SON_15=df1invmy_SON[(df1invmy_SON['distance']>10) & (df1invmy_SON['distance']<=15)]

#*****************************************************************************\
mean_DJF_5=[np.mean(df1invmy_DJF_5['1ra Inv']),np.mean(df1invmy_MAM_5['1ra Inv']), np.mean(df1invmy_JJA_5['1ra Inv']), np.mean(df1invmy_SON_5['1ra Inv'])]

mean_DJF_10=[np.mean(df1invmy_DJF_10['1ra Inv']),np.mean(df1invmy_MAM_10['1ra Inv']), np.mean(df1invmy_JJA_10['1ra Inv']), np.mean(df1invmy_SON_10['1ra Inv'])]

mean_DJF_15=[np.mean(df1invmy_DJF_15['1ra Inv']),np.mean(df1invmy_MAM_15['1ra Inv']), np.mean(df1invmy_JJA_15['1ra Inv']), np.mean(df1invmy_SON_15['1ra Inv'])]
#*****************************************************************************\
std_DJF_5=[np.std(df1invmy_DJF_5['1ra Inv']),np.std(df1invmy_MAM_5['1ra Inv']), np.std(df1invmy_JJA_5['1ra Inv']), np.std(df1invmy_SON_5['1ra Inv'])]

std_DJF_10=[np.std(df1invmy_DJF_10['1ra Inv']),np.std(df1invmy_MAM_10['1ra Inv']), np.std(df1invmy_JJA_10['1ra Inv']), np.std(df1invmy_SON_10['1ra Inv'])]

std_DJF_15=[np.std(df1invmy_DJF_15['1ra Inv']),np.std(df1invmy_MAM_15['1ra Inv']), np.std(df1invmy_JJA_15['1ra Inv']), np.std(df1invmy_SON_15['1ra Inv'])]


#*****************************************************************************\
data_to_plot_5= [df1invmy_DJF_5['1ra Inv'], df1invmy_MAM_5['1ra Inv'], df1invmy_JJA_5['1ra Inv'], df1invmy_SON_5['1ra Inv']]

data_to_plot_10= [df1invmy_DJF_10['1ra Inv'], df1invmy_MAM_10['1ra Inv'], df1invmy_JJA_10['1ra Inv'], df1invmy_SON_10['1ra Inv']]

data_to_plot_15= [df1invmy_DJF_15['1ra Inv'], df1invmy_MAM_15['1ra Inv'], df1invmy_JJA_15['1ra Inv'], df1invmy_SON_15['1ra Inv']]


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(15, 6))

bp1 = axes[0].boxplot(data_to_plot_5, whis=[5, 95], patch_artist=True)
bp2 = axes[1].boxplot(data_to_plot_10, whis=[5, 95], patch_artist=True)
bp3 = axes[2].boxplot(data_to_plot_15, whis=[5, 95], patch_artist=True)

for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(data_to_plot_5))], )
    ax.set_xticklabels(['DJF', 'MAM', 'JJA','SON'],fontsize = 16)
    ax.set_ylim(0,5000)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.get_yaxis().set_visible(False)
    # ax.get_yaxis().set_ticks([])
    # ax.yaxis.grid(True)
    ax.set_yticks(np.arange(0,5500,500))
    ax.set_yticklabels(np.arange(0,5500,500), size=16)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)


axes[1].yaxis.set_ticklabels([])
axes[2].yaxis.set_ticklabels([])
axes[0].set_ylabel('Height (m)',fontsize = 16)
axes[0].get_yaxis().set_visible(True)
axes[0].set_xlabel('5 deg',fontsize = 16)
axes[1].set_xlabel('10 deg',fontsize = 16)
axes[2].set_xlabel('15 deg',fontsize = 16)

plt.subplots_adjust(wspace=0, hspace=0)


#B1
for cap in bp1['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp1['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')
#B2
for cap in bp2['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp2['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')
#B3
for cap in bp3['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp3['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')


fig.savefig(path_data_save + 'boxplot_hght1_season_deg.eps', format='eps', dpi=1200)





















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
df1invmy_Q1=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']>0)]
df1invmy_Q2=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']>0)]
df1invmy_Q3=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']<0)]
df1invmy_Q4=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']<0)]


print len(df1invmy_Q1),


#*****************************************************************************\
# Box Plot
#*****************************************************************************\

data_to_plot = [df1invmy_Q1['1ra Inv'], df1invmy_Q2['1ra Inv'], df1invmy_Q3['1ra Inv'], df1invmy_Q4['1ra Inv']]

fig= plt.figure(5, figsize=(9, 6))
ax = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot, whis=[5, 95], patch_artist=True)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_ylabel('Height (m)',fontsize = 16)
ax.set_xticklabels(['NE', 'NW', 'SW','SE'],fontsize = 16)
ax.set_ylim(0,5000)
ax.set_yticks(np.arange(0,5500,500))
ax.set_yticklabels(np.arange(0,5500,500), size=16)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

for cap in bp['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')

fig.savefig(path_data_save + 'boxplot_hght1_quadrant.eps', format='eps', dpi=1200)





#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Height by Quadrant and Distance
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************


df_1invmy['distance']=np.sqrt(df_1invmy['dx']**2+df_1invmy['dy']**2)

df1invmy_Q1=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']>0)]
df1invmy_Q2=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']>0)]
df1invmy_Q3=df_1invmy[(df_1invmy['dx']<0) & (df_1invmy['dy']<0)]
df1invmy_Q4=df_1invmy[(df_1invmy['dx']>0) & (df_1invmy['dy']<0)]



#Q1
df1invmy_Q1_5=df1invmy_Q1[(df1invmy_Q1['distance']>0) & (df1invmy_Q1['distance']<=5)]
df1invmy_Q1_10=df1invmy_Q1[(df1invmy_Q1['distance']>5) & (df1invmy_Q1['distance']<=10)]
df1invmy_Q1_15=df1invmy_Q1[(df1invmy_Q1['distance']>10) & (df1invmy_Q1['distance']<=15)]
#Q2
df1invmy_Q2_5=df1invmy_Q2[(df1invmy_Q2['distance']>0) & (df1invmy_Q2['distance']<=5)]
df1invmy_Q2_10=df1invmy_Q2[(df1invmy_Q2['distance']>5) & (df1invmy_Q2['distance']<=10)]
df1invmy_Q2_15=df1invmy_Q2[(df1invmy_Q2['distance']>10) & (df1invmy_Q2['distance']<=15)]

#Q3
df1invmy_Q3_5=df1invmy_Q3[(df1invmy_Q3['distance']>0) & (df1invmy_Q3['distance']<=5)]
df1invmy_Q3_10=df1invmy_Q3[(df1invmy_Q3['distance']>5) & (df1invmy_Q3['distance']<=10)]
df1invmy_Q3_15=df1invmy_Q3[(df1invmy_Q3['distance']>10) & (df1invmy_Q3['distance']<=15)]

#Q4
df1invmy_Q4_5=df1invmy_Q4[(df1invmy_Q4['distance']>0) & (df1invmy_Q4['distance']<=5)]
df1invmy_Q4_10=df1invmy_Q4[(df1invmy_Q4['distance']>5) & (df1invmy_Q4['distance']<=10)]
df1invmy_Q4_15=df1invmy_Q4[(df1invmy_Q4['distance']>10) & (df1invmy_Q4['distance']<=15)]

#*****************************************************************************\
mean_Q1_5=[np.mean(df1invmy_Q1_5['1ra Inv']),np.mean(df1invmy_Q2_5['1ra Inv']), np.mean(df1invmy_Q3_5['1ra Inv']), np.mean(df1invmy_Q4_5['1ra Inv'])]

mean_Q1_10=[np.mean(df1invmy_Q1_10['1ra Inv']),np.mean(df1invmy_Q2_10['1ra Inv']), np.mean(df1invmy_Q3_10['1ra Inv']), np.mean(df1invmy_Q4_10['1ra Inv'])]

mean_Q1_15=[np.mean(df1invmy_Q1_15['1ra Inv']),np.mean(df1invmy_Q2_15['1ra Inv']), np.mean(df1invmy_Q3_15['1ra Inv']), np.mean(df1invmy_Q4_15['1ra Inv'])]
#*****************************************************************************\
std_Q1_5=[np.std(df1invmy_Q1_5['1ra Inv']),np.std(df1invmy_Q2_5['1ra Inv']), np.std(df1invmy_Q3_5['1ra Inv']), np.std(df1invmy_Q4_5['1ra Inv'])]

std_Q1_10=[np.std(df1invmy_Q1_10['1ra Inv']),np.std(df1invmy_Q2_10['1ra Inv']), np.std(df1invmy_Q3_10['1ra Inv']), np.std(df1invmy_Q4_10['1ra Inv'])]

std_Q1_15=[np.std(df1invmy_Q1_15['1ra Inv']),np.std(df1invmy_Q2_15['1ra Inv']), np.std(df1invmy_Q3_15['1ra Inv']), np.std(df1invmy_Q4_15['1ra Inv'])]


#*****************************************************************************\
data_to_plot_5= [df1invmy_Q1_5['1ra Inv'], df1invmy_Q2_5['1ra Inv'], df1invmy_Q3_5['1ra Inv'], df1invmy_Q4_5['1ra Inv']]

data_to_plot_10= [df1invmy_Q1_10['1ra Inv'], df1invmy_Q2_10['1ra Inv'], df1invmy_Q3_10['1ra Inv'], df1invmy_Q4_10['1ra Inv']]

data_to_plot_15= [df1invmy_Q1_15['1ra Inv'], df1invmy_Q2_15['1ra Inv'], df1invmy_Q3_15['1ra Inv'], df1invmy_Q4_15['1ra Inv']]


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(15, 6))

bp1 = axes[0].boxplot(data_to_plot_5, whis=[5, 95], patch_artist=True)
bp2 = axes[1].boxplot(data_to_plot_10, whis=[5, 95], patch_artist=True)
bp3 = axes[2].boxplot(data_to_plot_15, whis=[5, 95], patch_artist=True)

for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(data_to_plot_5))], )
    ax.set_xticklabels(['NE', 'NW', 'SW','SE'],fontsize = 13)
    ax.set_ylim(0,5000)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.get_yaxis().set_visible(False)
    # ax.get_yaxis().set_ticks([])
    # ax.yaxis.grid(True)
    ax.set_yticks(np.arange(0,5500,500))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)


axes[1].yaxis.set_ticklabels([])
axes[2].yaxis.set_ticklabels([])
axes[0].set_ylabel('height (m)',fontsize = 13)
axes[0].get_yaxis().set_visible(True)
axes[0].set_xlabel('5 deg',fontsize = 13)
axes[1].set_xlabel('10 deg',fontsize = 13)
axes[2].set_xlabel('15 deg',fontsize = 13)

plt.subplots_adjust(wspace=0, hspace=0)


#B1
for cap in bp1['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp1['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')
#B2
for cap in bp2['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp2['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')
#B3
for cap in bp3['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp3['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')


fig.savefig(path_data_save + 'boxplot_hght1_quadrant_deg.eps', format='eps', dpi=1200)














#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                      Strength Seasonal
#*****************************************************************************\
#*****************************************************************************
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
df_1strmy['distance']=np.sqrt(df_1strmy['dx']**2+df_1strmy['dy']**2)

df1strmy_DJF=df_1strmy[(df_1strmy.index.month==12) | (df_1strmy.index.month==1) | (df_1strmy.index.month==2)]
df1strmy_MAM=df_1strmy[(df_1strmy.index.month==3) | (df_1strmy.index.month==4) | (df_1strmy.index.month==5)]
df1strmy_JJA=df_1strmy[(df_1strmy.index.month==6) | (df_1strmy.index.month==7) | (df_1strmy.index.month==8)]
df1strmy_SON=df_1strmy[(df_1strmy.index.month==9) | (df_1strmy.index.month==10) | (df_1strmy.index.month==11)]



#*****************************************************************************\
# Box Plot
#*****************************************************************************\

data_to_plot = [df1strmy_DJF['Strg 1inv'], df1strmy_MAM['Strg 1inv'], df1strmy_JJA['Strg 1inv'], df1strmy_SON['Strg 1inv']]

fig= plt.figure(7, figsize=(9, 6))
ax = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot, whis=[5, 95], patch_artist=True)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_ylabel('Strength (K m$^{-1}$)',fontsize = 16)
ax.set_xticklabels(['DJF', 'MAM', 'JJA','SON'],fontsize = 16)
ax.set_ylim(0,0.05)
ax.set_yticks(np.arange(0,0.06,0.01))
ax.set_yticklabels(np.arange(0,0.06,0.01), size=16)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

for cap in bp['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')

fig.savefig(path_data_save + 'boxplot_strg1_season.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Strength by Season and Distance
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************

#DJF
df1strmy_DJF_5=df1strmy_DJF[(df1strmy_DJF['distance']>0) & (df1strmy_DJF['distance']<=5)]
df1strmy_DJF_10=df1strmy_DJF[(df1strmy_DJF['distance']>5) & (df1strmy_DJF['distance']<=10)]
df1strmy_DJF_15=df1strmy_DJF[(df1strmy_DJF['distance']>10) & (df1strmy_DJF['distance']<=15)]
#MAM
df1strmy_MAM_5=df1strmy_MAM[(df1strmy_MAM['distance']>0) & (df1strmy_MAM['distance']<=5)]
df1strmy_MAM_10=df1strmy_MAM[(df1strmy_MAM['distance']>5) & (df1strmy_MAM['distance']<=10)]
df1strmy_MAM_15=df1strmy_MAM[(df1strmy_MAM['distance']>10) & (df1strmy_MAM['distance']<=15)]

#JJA
df1strmy_JJA_5=df1strmy_JJA[(df1strmy_JJA['distance']>0) & (df1strmy_JJA['distance']<=5)]
df1strmy_JJA_10=df1strmy_JJA[(df1strmy_JJA['distance']>5) & (df1strmy_JJA['distance']<=10)]
df1strmy_JJA_15=df1strmy_JJA[(df1strmy_JJA['distance']>10) & (df1strmy_JJA['distance']<=15)]

#SON
df1strmy_SON_5=df1strmy_SON[(df1strmy_SON['distance']>0) & (df1strmy_SON['distance']<=5)]
df1strmy_SON_10=df1strmy_SON[(df1strmy_SON['distance']>5) & (df1strmy_SON['distance']<=10)]
df1strmy_SON_15=df1strmy_SON[(df1strmy_SON['distance']>10) & (df1strmy_SON['distance']<=15)]


#*****************************************************************************\


data_to_plot_5= [df1strmy_DJF_5['Strg 1inv'], df1strmy_MAM_5['Strg 1inv'], df1strmy_JJA_5['Strg 1inv'], df1strmy_SON_5['Strg 1inv']]

data_to_plot_10= [df1strmy_DJF_10['Strg 1inv'], df1strmy_MAM_10['Strg 1inv'], df1strmy_JJA_10['Strg 1inv'], df1strmy_SON_10['Strg 1inv']]

data_to_plot_15= [df1strmy_DJF_15['Strg 1inv'], df1strmy_MAM_15['Strg 1inv'], df1strmy_JJA_15['Strg 1inv'], df1strmy_SON_15['Strg 1inv']]



fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(15, 6))

bp1 = axes[0].boxplot(data_to_plot_5, whis=[5, 95], patch_artist=True)
bp2 = axes[1].boxplot(data_to_plot_10, whis=[5, 95], patch_artist=True)
bp3 = axes[2].boxplot(data_to_plot_15, whis=[5, 95], patch_artist=True)

for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(data_to_plot_5))], )
    ax.set_xticklabels(['DJF', 'MAM', 'JJA','SON'],fontsize = 13)
    ax.set_ylim(0,0.05)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_yticks(np.arange(0,0.06,0.01))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)



axes[1].yaxis.set_ticklabels([])
axes[2].yaxis.set_ticklabels([])
axes[0].set_ylabel('strength (K m$^{-1}$)',fontsize = 13)
axes[0].get_yaxis().set_visible(True)
axes[0].set_xlabel('5 deg',fontsize = 13)
axes[1].set_xlabel('10 deg',fontsize = 13)
axes[2].set_xlabel('15 deg',fontsize = 13)

plt.subplots_adjust(wspace=0, hspace=0)


#B1
for cap in bp1['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp1['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')
#B2
for cap in bp2['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp2['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')
#B3
for cap in bp3['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp3['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')


fig.savefig(path_data_save + 'boxplot_strg1_season_deg.eps', format='eps', dpi=1200)





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

df1strmy_Q1=df_1strmy[(df_1strmy['dx']>0) & (df_1strmy['dy']>0)]
df1strmy_Q2=df_1strmy[(df_1strmy['dx']<0) & (df_1strmy['dy']>0)]
df1strmy_Q3=df_1strmy[(df_1strmy['dx']<0) & (df_1strmy['dy']<0)]
df1strmy_Q4=df_1strmy[(df_1strmy['dx']>0) & (df_1strmy['dy']<0)]




#*****************************************************************************\
# Box Plot
#*****************************************************************************\

data_to_plot = [df1invmy_Q1['Strg 1inv'], df1invmy_Q2['Strg 1inv'], df1invmy_Q3['Strg 1inv'], df1invmy_Q4['Strg 1inv']]

fig= plt.figure(9, figsize=(9, 6))
ax = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot, whis=[5, 95], patch_artist=True)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_ylabel('Strength (K m$^{-1}$)',fontsize = 16)
ax.set_xticklabels(['NE', 'NW', 'SW','SE'],fontsize = 16)
ax.set_ylim(0,0.05)
ax.set_yticks(np.arange(0,0.06,0.01))
ax.set_yticklabels(np.arange(0,0.06,0.01), size=16)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

for cap in bp['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')

fig.savefig(path_data_save + 'boxplot_strg1_quadrant.eps', format='eps', dpi=1200)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Strength by Quadrant and Distance
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************


df_1strmy['distance']=np.sqrt(df_1strmy['dx']**2+df_1strmy['dy']**2)

df1strmy_Q1=df_1strmy[(df_1strmy['dx']>0) & (df_1strmy['dy']>0)]
df1strmy_Q2=df_1strmy[(df_1strmy['dx']<0) & (df_1strmy['dy']>0)]
df1strmy_Q3=df_1strmy[(df_1strmy['dx']<0) & (df_1strmy['dy']<0)]
df1strmy_Q4=df_1strmy[(df_1strmy['dx']>0) & (df_1strmy['dy']<0)]



#Q1
df1strmy_Q1_5=df1strmy_Q1[(df1strmy_Q1['distance']>0) & (df1strmy_Q1['distance']<=5)]
df1strmy_Q1_10=df1strmy_Q1[(df1strmy_Q1['distance']>5) & (df1strmy_Q1['distance']<=10)]
df1strmy_Q1_15=df1strmy_Q1[(df1strmy_Q1['distance']>10) & (df1strmy_Q1['distance']<=15)]
#Q2
df1strmy_Q2_5=df1strmy_Q2[(df1strmy_Q2['distance']>0) & (df1strmy_Q2['distance']<=5)]
df1strmy_Q2_10=df1strmy_Q2[(df1strmy_Q2['distance']>5) & (df1strmy_Q2['distance']<=10)]
df1strmy_Q2_15=df1strmy_Q2[(df1strmy_Q2['distance']>10) & (df1strmy_Q2['distance']<=15)]

#Q3
df1strmy_Q3_5=df1strmy_Q3[(df1strmy_Q3['distance']>0) & (df1strmy_Q3['distance']<=5)]
df1strmy_Q3_10=df1strmy_Q3[(df1strmy_Q3['distance']>5) & (df1strmy_Q3['distance']<=10)]
df1strmy_Q3_15=df1strmy_Q3[(df1strmy_Q3['distance']>10) & (df1strmy_Q3['distance']<=15)]

#Q4
df1strmy_Q4_5=df1strmy_Q4[(df1strmy_Q4['distance']>0) & (df1strmy_Q4['distance']<=5)]
df1strmy_Q4_10=df1strmy_Q4[(df1strmy_Q4['distance']>5) & (df1strmy_Q4['distance']<=10)]
df1strmy_Q4_15=df1strmy_Q4[(df1strmy_Q4['distance']>10) & (df1strmy_Q4['distance']<=15)]

#*****************************************************************************\
mean_Q1_5=[np.mean(df1strmy_Q1_5['Strg 1inv']),np.mean(df1strmy_Q2_5['Strg 1inv']), np.mean(df1strmy_Q3_5['Strg 1inv']), np.mean(df1strmy_Q4_5['Strg 1inv'])]

mean_Q1_10=[np.mean(df1strmy_Q1_10['Strg 1inv']),np.mean(df1strmy_Q2_10['Strg 1inv']), np.mean(df1strmy_Q3_10['Strg 1inv']), np.mean(df1strmy_Q4_10['Strg 1inv'])]

mean_Q1_15=[np.mean(df1strmy_Q1_15['Strg 1inv']),np.mean(df1strmy_Q2_15['Strg 1inv']), np.mean(df1strmy_Q3_15['Strg 1inv']), np.mean(df1strmy_Q4_15['Strg 1inv'])]
#*****************************************************************************\
std_Q1_5=[np.std(df1strmy_Q1_5['Strg 1inv']),np.std(df1strmy_Q2_5['Strg 1inv']), np.std(df1strmy_Q3_5['Strg 1inv']), np.std(df1strmy_Q4_5['Strg 1inv'])]

std_Q1_10=[np.std(df1strmy_Q1_10['Strg 1inv']),np.std(df1strmy_Q2_10['Strg 1inv']), np.std(df1strmy_Q3_10['Strg 1inv']), np.std(df1strmy_Q4_10['Strg 1inv'])]

std_Q1_15=[np.std(df1strmy_Q1_15['Strg 1inv']),np.std(df1strmy_Q2_15['Strg 1inv']), np.std(df1strmy_Q3_15['Strg 1inv']), np.std(df1strmy_Q4_15['Strg 1inv'])]


#*****************************************************************************\
data_to_plot_5= [df1strmy_Q1_5['Strg 1inv'], df1strmy_Q2_5['Strg 1inv'], df1strmy_Q3_5['Strg 1inv'], df1strmy_Q4_5['Strg 1inv']]

data_to_plot_10= [df1strmy_Q1_10['Strg 1inv'], df1strmy_Q2_10['Strg 1inv'], df1strmy_Q3_10['Strg 1inv'], df1strmy_Q4_10['Strg 1inv']]

data_to_plot_15= [df1strmy_Q1_15['Strg 1inv'], df1strmy_Q2_15['Strg 1inv'], df1strmy_Q3_15['Strg 1inv'], df1strmy_Q4_15['Strg 1inv']]


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(15, 6))

bp1 = axes[0].boxplot(data_to_plot_5, whis=[5, 95], patch_artist=True)
bp2 = axes[1].boxplot(data_to_plot_10, whis=[5, 95], patch_artist=True)
bp3 = axes[2].boxplot(data_to_plot_15, whis=[5, 95], patch_artist=True)

for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(data_to_plot_5))], )
    ax.set_xticklabels(['NE', 'NW', 'SW','SE'],fontsize = 13)
    ax.set_ylim(0,0.05)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_yticks(np.arange(0,0.06,0.01))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)



axes[1].yaxis.set_ticklabels([])
axes[2].yaxis.set_ticklabels([])
axes[0].set_ylabel('strength (K m$^{-1}$)',fontsize = 13)
axes[0].get_yaxis().set_visible(True)
axes[0].set_xlabel('5 deg',fontsize = 13)
axes[1].set_xlabel('10 deg',fontsize = 13)
axes[2].set_xlabel('15 deg',fontsize = 13)

plt.subplots_adjust(wspace=0, hspace=0)


#B1
for cap in bp1['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp1['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')
#B2
for cap in bp2['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp2['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')
#B3
for cap in bp3['caps']:
    cap.set(color='b', linewidth=1)
# for flier in bp3['fliers']:
#     flier.set(marker='o',markerfacecolor='none', markeredgecolor='b')


fig.savefig(path_data_save + 'boxplot_strg1_quadrant_deg.eps', format='eps', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           No Inversion
#*****************************************************************************\
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************

df_NI_1= df_mycyc[df_mycyc['Clas']==1]
df_NI = df_NI_1[np.isfinite(df_NI_1['dy'])]

# df_NI=df_my_1[df_my_1['Clas']==1]

