import numpy as np
import scipy.io as sio
import os
from pylab import plot,show, grid
import math
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from numpy.random import random
from numpy import arange

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
df_yotc= pd.read_csv(path_data + 'df_yotc_all925.csv', sep='\t', parse_dates=['Date'])
df_my= pd.read_csv(path_data + 'df_macyotc_final925.csv', sep='\t', parse_dates=['Date'])

df_yotc= df_yotc.set_index('Date')
df_my= df_my.set_index('Date')
#*****************************************************************************\
#Merge datraframe mac with
df_yotcfro=pd.concat([df_yotc, df_front],axis=1)
df_myfro=pd.concat([df_my, df_front],axis=1)

#*****************************************************************************\
#Percentages
#*****************************************************************************\
#Total Number Sounding at YOTC of 1460
n_yotcsound=len(df_yotcfro[np.isfinite(df_yotcfro['Clas'])])
#Total Number Sounding at MAC of 3652
n_mysound=len(df_myfro[np.isfinite(df_myfro['Clas'])])

#Sounding YOTC
df_yotcsound=df_yotcfro[np.isfinite(df_yotcfro['Clas'])]
#Sounding MAC-YOTC
df_mysound=df_myfro[np.isfinite(df_myfro['Clas'])]

#Total Number Sounding at YOTC with front
n_yotcfronts=len(df_yotcsound[np.isfinite(df_yotcsound['Dist Front'])])
#Total Number Sounding at MAC-YOTC with front
n_myfronts=len(df_mysound[np.isfinite(df_mysound['Dist Front'])])

#Percentages fronts regarding soundings
per_yotcfro=n_yotcfronts/float(n_yotcsound)*100
per_myfro=n_myfronts/float(n_mysound)*100
#*****************************************************************************\
#Clasification by type
#*****************************************************************************\
# df_BL1 = df_yotcfro[df_yotcfro['Clas']==4]
# df_yotc_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

df_DL1 = df_yotcfro[(df_yotcfro['Clas']==3) | (df_yotcfro['Clas']==4)]
df_yotc_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

df_SI1 = df_yotcfro[df_yotcfro['Clas']==2]
df_yotc_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

df_NI1 = df_yotcfro[df_yotcfro['Clas']==1]
df_yotc_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

del df_DL1, df_NI1, df_SI1
#*****************************************************************************\

#df_BL1 = df_myfro[df_myfro['Clas']==4]
#df_my_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

df_DL1 = df_myfro[(df_myfro['Clas']==3) | (df_myfro['Clas']==4)]
df_my_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

df_SI1 = df_myfro[df_myfro['Clas']==2]
df_my_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

df_NI1 = df_myfro[df_myfro['Clas']==1]
df_my_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

#*****************************************************************************\
#*****************************************************************************\
#                               Graphs Setup
#*****************************************************************************\
#*****************************************************************************\
colorh='#2929a3'
colorb='blue'
color1='#1C86EE'
color2='#104E8B'

color3='#FF3333'
color4='#9D1309'

figsize3=(7.5, 5)
fsize0=12
fsize1=14
fsize2=16

path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts_ok/variables/'

#*****************************************************************************\
x1=-15
x2=15
y1=0
y2=0.24
dx=1
bx=np.arange(x1,x2+1,3)
by=np.arange(y1,y2+0.02,0.02)
bins = np.arange(x1, x2+1, dx)

df1=15
df2=30
width = 1

error_config = {'ecolor': '0.4'}
error_config2 = {'ecolor': '0'}

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                   Wind Rose (Range 1-5 degrees)
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

lim_up=5
lim_lo=1

#Extract Dataframe between 1 and 5 degrees
df_posmy= df_myfro[(df_myfro['Dist Front']>=lim_lo) & (df_myfro['Dist Front']<=lim_up)]
df_premy= df_myfro[(df_myfro['Dist Front']<=-lim_lo) & (df_myfro['Dist Front']>=-lim_up)]
df_posyotc= df_yotcfro[(df_yotcfro['Dist Front']>=lim_lo) & (df_yotcfro['Dist Front']<=lim_up)]
df_preyotc= df_yotcfro[(df_yotcfro['Dist Front']<=-lim_lo) & (df_yotcfro['Dist Front']>=-lim_up)]



#*****************************************************************************\
# Wind Rose Set up
#*****************************************************************************\
def new_axes():
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect, axisbg='w')
    fig.add_axes(ax)
    return ax

def set_legend(ax):
    l = ax.legend(borderaxespad=-0.10,title='wind speed (ms$^{-1}$)')
    plt.setp(l.get_texts(), fontsize=12)

opening=1
bins=np.arange(0, 36, 6)
#bins=np.arange(0, 40, 5)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# MAC Ave
#*****************************************************************************\
#*****************************************************************************\
# Wind Direction
#*****************************************************************************\
upre=np.array(df_premy['u 925'])
vpre=np.array(df_premy['v 925'])

wsp_premy=np.sqrt(upre**2 + vpre**2)
#dir_premy=np.arctan2(upre, vpre)*(180/np.pi)+180
dir_premy=np.arctan2(-upre, -vpre)*(180/np.pi)

dir_premy[(upre == 0) & (vpre == 0)]=0

for i in range(0, len(dir_premy)):
    if dir_premy[i]<=0:
        dir_premy[i]+=360

#*****************************************************************************\
upos=np.array(df_posmy['u 925'])
vpos=np.array(df_posmy['v 925'])

wsp_posmy=np.sqrt(upos**2 + vpos**2)
dir_posmy=np.arctan2(-upos, -vpos)*(180/np.pi)

dir_posmy[(upos == 0) & (vpos == 0)]=0

for i in range(0, len(dir_posmy)):
    if dir_posmy[i]<=0:
        dir_posmy[i]+=360

#*****************************************************************************\
# Wind Rose
#*****************************************************************************\
#Create wind speed and direction variables
# ws = random(500)*6
# wd = random(500)*360

ws1=wsp_premy
wd1=dir_premy

wd1= wd1[~np.isnan(ws1)]
ws1= ws1[~np.isnan(ws1)]


ws2=wsp_posmy
wd2=dir_posmy

wd2= wd2[~np.isnan(ws2)]
ws2= ws2[~np.isnan(ws2)]

#*****************************************************************************\
ax = new_axes()
ax.bar(wd1, ws1, normed=True, opening=opening, edgecolor='white',bins=bins)
set_legend(ax)
plt.savefig(path_data_save + 'wrfronts_mypre.eps', format='eps', dpi=1200)

#*****************************************************************************\
ax = new_axes()
ax.bar(wd2, ws2, normed=True, opening=opening, edgecolor='white',bins=bins)
set_legend(ax)

plt.savefig(path_data_save + 'wrfronts_mypost.eps', format='eps', dpi=1200)

#plt.show()

#*****************************************************************************\
# dmy={'Speed MY Pre':ws1, 'Dir MY Pre':wd1}
# df_1= pd.DataFrame(data=dmy)
# df_1.to_csv(path_data + 'wind_dir/df_mypre.csv', sep=',', encoding='utf-8')


# dmy={'Speed MY Post':ws2, 'Dir MY Post':wd2}
# df_2= pd.DataFrame(data=dmy)
# df_2.to_csv(path_data + 'wind_dir/df_mypos.csv', sep=',', encoding='utf-8')

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# MYOTC
#*****************************************************************************\
#*****************************************************************************\
# Wind Direction
#*****************************************************************************\
upre=np.array(df_preyotc['u 925'])
vpre=np.array(df_preyotc['v 925'])

wsp_preyotc=np.sqrt(upre**2 + vpre**2)

#dir_preyotc=np.arctan2(upre, vpre)*(180/np.pi)+180
dir_preyotc=np.arctan2(-upre, -vpre)*(180/np.pi)

dir_preyotc[(upre == 0) & (vpre == 0)]=0

for i in range(0, len(dir_preyotc)):
    if dir_preyotc[i]<=0:
        dir_preyotc[i]+=360

#*****************************************************************************\
upos=np.array(df_posyotc['u 925'])
vpos=np.array(df_posyotc['v 925'])

wsp_posyotc=np.sqrt(upos**2 + vpos**2)
dir_posyotc=np.arctan2(-upos, -vpos)*(180/np.pi)

dir_posyotc[(upos == 0) & (vpos == 0)]=0

for i in range(0, len(dir_posyotc)):
    if dir_posyotc[i]<=0:
        dir_posyotc[i]+=360

#*****************************************************************************\
# Wind Rose
#*****************************************************************************\
ws1=wsp_preyotc
wd1=dir_preyotc

wd1= wd1[~np.isnan(ws1)]
ws1= ws1[~np.isnan(ws1)]

ws2=wsp_posyotc
wd2=dir_posyotc

wd2= wd2[~np.isnan(ws2)]
ws2= ws2[~np.isnan(ws2)]

#*****************************************************************************\
ax = new_axes()
ax.bar(wd1, ws1, normed=True, opening=opening, edgecolor='white',bins=bins)
set_legend(ax)
plt.savefig(path_data_save + 'wrfronts_yotcpre.eps', format='eps', dpi=1200)
#*****************************************************************************\
ax = new_axes()
ax.bar(wd2, ws2, normed=True, opening=opening, edgecolor='white',bins=bins)
set_legend(ax)
plt.savefig(path_data_save + 'wrfronts_yotcpost.eps', format='eps', dpi=1200)

#*****************************************************************************\
# dmy={'Speed YOTC Pre':ws1, 'Dir YOTC Pre':wd1}
# df_1= pd.DataFrame(data=dmy)
# df_1.to_csv(path_data + 'wind_dir/df_yotcpre.csv', sep=',', encoding='utf-8')


# dmy={'Speed YOTC Post':ws2, 'Dir YOTC Post':wd2}
# df_2= pd.DataFrame(data=dmy)
# df_2.to_csv(path_data + 'wind_dir/df_yotcpos.csv', sep=',', encoding='utf-8')
# plt.close()
# plt.close()
# plt.close()
# plt.close()


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                   Temperature (Range 1-5 degrees)
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
yy1=-4
yy2=8

yyy1=0
yyy2=16
by=np.arange(yy1,yy2+2,2)

row = 2
column = 2

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(14,10))
ax1, ax2, ax3,ax4= axes.flat
#*****************************************************************************\
#*****************************************************************************\
# MAC Ave
#*****************************************************************************\
#*****************************************************************************\

df_myclas = df_myfro[np.isfinite(df_myfro['Clas'])]
#*****************************************************************************\
df_mydist = df_myclas[np.isfinite(df_myclas['Dist Front'])]
x1=np.array(df_mydist['Dist Front'])
y1=np.array(df_mydist['T 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([yy1,yy2])
ax1.legend(loc='upper right')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax1.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax1.grid()

#*****************************************************************************\
y1=np.array(df_mydist['Theta v 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax2.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax2.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_ylim([yyy1,yyy2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel(r'$\theta_v$ ($^o$C)',fontsize = 12)
ax2.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax2.grid()

#*****************************************************************************\
#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\

df_yotcclas = df_yotcfro[np.isfinite(df_yotcfro['Clas'])]
#*****************************************************************************\
df_yotcdist = df_yotcclas[np.isfinite(df_yotcclas['Dist Front'])]
x1=np.array(df_yotcdist['Dist Front'])
y1=np.array(df_yotcdist['T 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_ylim([yy1,yy2])
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax3.set_title('YOTC',fontsize = 14 ,fontweight='bold')
ax3.grid()

#*****************************************************************************\

y1=np.array(df_yotcdist['Theta v 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax4.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax4.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_ylim([yyy1,yyy2])
ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
ax4.set_ylabel(r'$\theta_v$ ($^o$C)',fontsize = 12)
ax4.set_title('YOTC',fontsize = 14 ,fontweight='bold')
ax4.grid()

#fig.tight_layout()
fig.savefig(path_data_save + 'temp_fronts.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
# Temperature by Clas
#*****************************************************************************\
#*****************************************************************************\
yy1=-8
yy2=12

by=np.arange(yy1,yy2+2,4)

row = 2
column = 3
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,10))
ax1, ax2, ax3, ax4,ax5,ax6= axes.flat
#*****************************************************************************\
#MAC Ave
#*****************************************************************************\
dfmy=df_my_NI[np.isfinite(df_my_NI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['T 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_ylim([yy1,yy2])
ax1.legend(loc='upper right')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax1.set_title('No Inversion (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax1.grid()
del dfmy, x, y
#*****************************************************************************\
dfmy=df_my_SI[np.isfinite(df_my_SI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['T 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax2.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax2.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_ylim([yy1,yy2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax2.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax2.set_title('Single Inversion (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax2.grid()

#*****************************************************************************\
dfmy=df_my_DL[np.isfinite(df_my_DL['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['T 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_ylim([yy1,yy2])
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax3.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax3.set_title('Decoupled Layer (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax3.grid()
#*****************************************************************************\
#YOTC
#*****************************************************************************\
dfyotc=df_yotc_NI[np.isfinite(df_yotc_NI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['T 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax4.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax4.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_yticks(by)
ax4.set_ylim([yy1,yy2])
ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
ax4.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax4.set_title('No Inversion (YOTC)',fontsize = 14 ,fontweight='bold')
ax4.grid()
del dfyotc, x, y
#*****************************************************************************\
dfyotc=df_yotc_SI[np.isfinite(df_yotc_SI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['T 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax5.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax5.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_yticks(by)
ax5.set_ylim([yy1,yy2])
ax5.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax5.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax5.set_title('Single Inversion (YOTC)',fontsize = 14 ,fontweight='bold')
ax5.grid()

#*****************************************************************************\
dfyotc=df_yotc_DL[np.isfinite(df_yotc_DL['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['T 925'])-273.16

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax6.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax6.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax6.tick_params(axis='both', which='major', labelsize=14)
ax6.set_xticks(bx)
ax6.set_yticks(by)
ax6.set_ylim([yy1,yy2])
ax6.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax6.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax6.set_title('Decoupled Layer (YOTC)',fontsize = 14 ,fontweight='bold')
ax6.grid()
plt.suptitle('925 hPa',fontsize = 14 ,fontweight='bold')
fig.savefig(path_data_save + 'tempclas_fronts.eps', format='eps', dpi=1200)




#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                   Water (Range 1-5 degrees)
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
yy1=50
yy2=120

yyy1=0
yyy2=8


row = 2
column = 2

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(14,10))
ax1, ax2, ax3,ax4= axes.flat
#*****************************************************************************\
#*****************************************************************************\
# MAC Ave
#*****************************************************************************\
#*****************************************************************************\

x1=np.array(df_mydist['Dist Front'])
y1=np.array(df_mydist['RH 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([yy1,yy2])
ax1.legend(loc='upper right')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('RH (%)',fontsize = 12)
ax1.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax1.grid()
#*****************************************************************************\
y1=np.array(df_mydist['Mix R 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax2.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax2.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_ylim([yyy1,yyy2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('mix. ratio (g kg$^{-1}$)',fontsize = 12)
ax2.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax2.grid()




#*****************************************************************************\
#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\
x1=np.array(df_yotcdist['Dist Front'])
y1=np.array(df_yotcdist['RH 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_ylim([yy1,yy2])
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_ylabel('RH (%)',fontsize = 12)
ax3.set_title('YOTC',fontsize = 14 ,fontweight='bold')
ax3.grid()

#*****************************************************************************\

y1=np.array(df_yotcdist['Mix R 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax4.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax4.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_ylim([yyy1,yyy2])
ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
ax4.set_ylabel('mix. ratio (g kg$^{-1}$)',fontsize = 12)
ax4.set_title('YOTC',fontsize = 14 ,fontweight='bold')
ax4.grid()

#fig.tight_layout()
fig.savefig(path_data_save + 'water_fronts.eps', format='eps', dpi=1200)




#*****************************************************************************\
#*****************************************************************************\
# Water by Clas
#*****************************************************************************\
#*****************************************************************************\
yy1=0
yy2=8
row = 2
column = 3
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,10))
ax1, ax2, ax3, ax4,ax5,ax6= axes.flat
#*****************************************************************************\
#MAC Ave
#*****************************************************************************\
dfmy=df_my_NI[np.isfinite(df_my_NI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['Mix R 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([yy1,yy2])
ax1.legend(loc='upper right')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('mix. ratio (g kg$^{-1}$)',fontsize = 12)
ax1.set_title('No Inversion (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax1.grid()
del dfmy, x, y
#*****************************************************************************\
dfmy=df_my_SI[np.isfinite(df_my_SI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['Mix R 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax2.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax2.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_ylim([yy1,yy2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax2.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax2.set_title('Single Inversion (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax2.grid()

#*****************************************************************************\
dfmy=df_my_DL[np.isfinite(df_my_DL['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['Mix R 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_ylim([yy1,yy2])
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax3.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax3.set_title('Decoupled Layer (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax3.grid()
#*****************************************************************************\
#YOTC
#*****************************************************************************\
dfyotc=df_yotc_NI[np.isfinite(df_yotc_NI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['Mix R 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax4.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax4.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_ylim([yy1,yy2])
#ax4.legend(loc='upper right')
ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
ax4.set_ylabel('mix. ratio (g kg$^{-1}$)',fontsize = 12)
ax4.set_title('No Inversion (YOTC)',fontsize = 14 ,fontweight='bold')
ax4.grid()
del dfyotc, x, y
#*****************************************************************************\
dfyotc=df_yotc_SI[np.isfinite(df_yotc_SI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['Mix R 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax5.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax5.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_ylim([yy1,yy2])
ax5.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax5.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax5.set_title('Single Inversion (YOTC)',fontsize = 14 ,fontweight='bold')
ax5.grid()

#*****************************************************************************\
dfyotc=df_yotc_DL[np.isfinite(df_yotc_DL['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['Mix R 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax6.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax6.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax6.tick_params(axis='both', which='major', labelsize=14)
ax6.set_xticks(bx)
ax6.set_ylim([yy1,yy2])
ax6.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax6.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax6.set_title('Decoupled Layer (YOTC)',fontsize = 14 ,fontweight='bold')
ax6.grid()
plt.suptitle('925 hPa',fontsize = 14 ,fontweight='bold')
fig.savefig(path_data_save + 'waterclas_fronts.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Wind
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
yy1=-20
yy2=30

yyy1=0
yyy2=40


row = 2
column = 3

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,10))
ax1, ax2, ax3,ax4,ax5,ax6= axes.flat
#*****************************************************************************\
#*****************************************************************************\
# MAC Ave
#*****************************************************************************\
#*****************************************************************************\

x1=np.array(df_mydist['Dist Front'])
y1=np.array(df_mydist['u 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([yy1,yy2])

ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('ms$^{-1}$',fontsize = 12)
ax1.set_title('u (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax1.grid()
#*****************************************************************************\
y1=np.array(df_mydist['v 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax2.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax2.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_ylim([yy1,yy2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('ms$^{-1}$',fontsize = 12)
ax2.set_title('v (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid()
#*****************************************************************************\

y1=np.sqrt(np.array(df_mydist['v 925'])**2+np.array(df_mydist['u 925'])**2)

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_ylim([yyy1,yyy2])
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_ylabel('ms$^{-1}$',fontsize = 12)
ax3.set_title('wind speed (MAC$_{AVE}$)',fontsize = 14 ,fontweight='bold')
ax3.grid()

#*****************************************************************************\
#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\

x1=np.array(df_yotcdist['Dist Front'])
y1=np.array(df_yotcdist['u 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax4.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax4.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_ylim([yy1,yy2])
ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
ax5.set_ylabel('ms$^{-1}$',fontsize = 12)
ax4.set_title('u (YOTC)',fontsize = 14 ,fontweight='bold')
ax4.grid()
#*****************************************************************************\
y1=np.array(df_yotcdist['v 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax5.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax5.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_ylim([yy1,yy2])
ax5.set_xlabel('Distance from front (deg)',fontsize = 12)
ax5.set_ylabel('ms$^{-1}$',fontsize = 12)
ax5.set_title('v (YOTC)',fontsize = 14 ,fontweight='bold')
ax5.grid()
#*****************************************************************************\

y1=np.sqrt(np.array(df_yotcdist['v 925'])**2+np.array(df_yotcdist['u 925'])**2)

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax6.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax6.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax6.tick_params(axis='both', which='major', labelsize=14)
ax6.set_xticks(bx)
ax6.set_ylim([yyy1,yyy2])
ax6.set_xlabel('Distance from front (deg)',fontsize = 12)
ax6.set_ylabel('ms$^{-1}$',fontsize = 12)
ax6.set_title('wind speed (YOTC)',fontsize = 14 ,fontweight='bold')
ax6.grid()

fig.savefig(path_data_save + 'wind_fronts.eps', format='eps', dpi=1200)

#*****************************************************************************\
#*****************************************************************************\
# Wind by Clas MAC Ave
#*****************************************************************************\
#*****************************************************************************\
yy1=-20
yy2=30

yyy1=0
yyy2=40

row = 3
column = 3
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,15))
ax1, ax2, ax3, ax4,ax5,ax6,ax7,ax8,ax9= axes.flat
#*****************************************************************************\

#*****************************************************************************\
dfmy=df_my_NI[np.isfinite(df_my_NI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['u 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([yy1,yy2])
ax1.set_ylabel('u (ms$^{-1}$)',fontsize = 12)
ax1.set_title('No Inversion',fontsize = 14 ,fontweight='bold')
ax1.grid()
del dfmy, x, y
#*****************************************************************************\
dfmy=df_my_SI[np.isfinite(df_my_SI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['u 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax2.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax2.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_ylim([yy1,yy2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax2.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax2.set_title('Single Inversion',fontsize = 14 ,fontweight='bold')
ax2.grid()

#*****************************************************************************\
dfmy=df_my_DL[np.isfinite(df_my_DL['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['u 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_ylim([yy1,yy2])
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax3.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax3.set_title('Decoupled Layer',fontsize = 14 ,fontweight='bold')
ax3.grid()
#*****************************************************************************\

#*****************************************************************************\
dfmy=df_my_NI[np.isfinite(df_my_NI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['v 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax4.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax4.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_ylim([yy1,yy2])
ax4.legend(loc='upper left')
ax4.set_ylabel('v (ms$^{-1}$)',fontsize = 12)
ax4.set_title('No Inversion',fontsize = 14 ,fontweight='bold')
ax4.grid()
del dfmy, x, y
#*****************************************************************************\
dfmy=df_my_SI[np.isfinite(df_my_SI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['v 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax5.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax5.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_ylim([yy1,yy2])
ax5.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax5.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax5.set_title('Single Inversion',fontsize = 14 ,fontweight='bold')
ax5.grid()

#*****************************************************************************\
dfmy=df_my_DL[np.isfinite(df_my_DL['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.array(dfmy['v 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax6.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax6.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax6.tick_params(axis='both', which='major', labelsize=14)
ax6.set_xticks(bx)
ax6.set_ylim([yy1,yy2])
ax6.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax6.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax6.set_title('Decoupled Layer',fontsize = 14 ,fontweight='bold')
ax6.grid()
#*****************************************************************************\

#*****************************************************************************\
dfmy=df_my_NI[np.isfinite(df_my_NI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.sqrt(np.array(dfmy['v 925'])**2+np.array(dfmy['u 925'])**2)

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax7.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax7.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax7.tick_params(axis='both', which='major', labelsize=14)
ax7.set_xticks(bx)
ax7.set_ylim([yyy1,yyy2])
ax7.set_ylabel('wind speed (ms$^{-1}$)',fontsize = 12)
ax7.set_title('No Inversion',fontsize = 14 ,fontweight='bold')
ax7.grid()
del dfmy, x, y
#*****************************************************************************\
dfmy=df_my_SI[np.isfinite(df_my_SI['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.sqrt(np.array(dfmy['v 925'])**2+np.array(dfmy['u 925'])**2)

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax8.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax8.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax8.tick_params(axis='both', which='major', labelsize=14)
ax8.set_xticks(bx)
ax8.set_ylim([yyy1,yyy2])
ax8.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax8.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax8.set_title('Single Inversion',fontsize = 14 ,fontweight='bold')
ax8.grid()

#*****************************************************************************\
dfmy=df_my_DL[np.isfinite(df_my_DL['Dist Front'])]
x=np.array(dfmy['Dist Front'])
y=np.sqrt(np.array(dfmy['v 925'])**2+np.array(dfmy['u 925'])**2)

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax9.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax9.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax9.tick_params(axis='both', which='major', labelsize=14)
ax9.set_xticks(bx)
ax9.set_ylim([yyy1,yyy2])
ax9.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax9.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax9.set_title('Decoupled Layer',fontsize = 14 ,fontweight='bold')
ax9.grid()

plt.suptitle('MAC$_{AVE}$ (925 hPa)',fontsize = 14 ,fontweight='bold')
fig.savefig(path_data_save + 'windclas_my_fronts.eps', format='eps', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
# Wind by Clas YOTC
#*****************************************************************************\
#*****************************************************************************\
yy1=-20
yy2=30

yyy1=0
yyy2=40

row = 3
column = 3
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,15))
ax1, ax2, ax3, ax4,ax5,ax6,ax7,ax8,ax9= axes.flat
#*****************************************************************************\

#*****************************************************************************\
dfyotc=df_yotc_NI[np.isfinite(df_yotc_NI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['u 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([yy1,yy2])
ax1.set_ylabel('u (ms$^{-1}$)',fontsize = 12)
ax1.set_title('No Inversion',fontsize = 14 ,fontweight='bold')
ax1.grid()
del dfyotc, x, y
#*****************************************************************************\
dfyotc=df_yotc_SI[np.isfinite(df_yotc_SI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['u 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax2.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax2.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_ylim([yy1,yy2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax2.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax2.set_title('Single Inversion',fontsize = 14 ,fontweight='bold')
ax2.grid()

#*****************************************************************************\
dfyotc=df_yotc_DL[np.isfinite(df_yotc_DL['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['u 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_ylim([yy1,yy2])
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax3.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax3.set_title('Decoupled Layer',fontsize = 14 ,fontweight='bold')
ax3.grid()
#*****************************************************************************\

#*****************************************************************************\
dfyotc=df_yotc_NI[np.isfinite(df_yotc_NI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['v 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax4.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax4.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_ylim([yy1,yy2])
ax4.legend(loc='upper left')
ax4.set_ylabel('v (ms$^{-1}$)',fontsize = 12)
ax4.set_title('No Inversion',fontsize = 14 ,fontweight='bold')
ax4.grid()
del dfyotc, x, y
#*****************************************************************************\
dfyotc=df_yotc_SI[np.isfinite(df_yotc_SI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['v 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax5.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax5.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_ylim([yy1,yy2])
ax5.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax5.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax5.set_title('Single Inversion',fontsize = 14 ,fontweight='bold')
ax5.grid()

#*****************************************************************************\
dfyotc=df_yotc_DL[np.isfinite(df_yotc_DL['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.array(dfyotc['v 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax6.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax6.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax6.tick_params(axis='both', which='major', labelsize=14)
ax6.set_xticks(bx)
ax6.set_ylim([yy1,yy2])
ax6.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax6.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax6.set_title('Decoupled Layer',fontsize = 14 ,fontweight='bold')
ax6.grid()
#*****************************************************************************\

#*****************************************************************************\
dfyotc=df_yotc_NI[np.isfinite(df_yotc_NI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.sqrt(np.array(dfyotc['v 925'])**2+np.array(dfyotc['u 925'])**2)

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax7.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax7.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax7.tick_params(axis='both', which='major', labelsize=14)
ax7.set_xticks(bx)
ax7.set_ylim([yyy1,yyy2])
ax7.set_ylabel('wind speed (ms$^{-1}$)',fontsize = 12)
ax7.set_title('No Inversion',fontsize = 14 ,fontweight='bold')
ax7.grid()
del dfyotc, x, y
#*****************************************************************************\
dfyotc=df_yotc_SI[np.isfinite(df_yotc_SI['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.sqrt(np.array(dfyotc['v 925'])**2+np.array(dfyotc['u 925'])**2)

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax8.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax8.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax8.tick_params(axis='both', which='major', labelsize=14)
ax8.set_xticks(bx)
ax8.set_ylim([yyy1,yyy2])
ax8.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax8.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax8.set_title('Single Inversion',fontsize = 14 ,fontweight='bold')
ax8.grid()

#*****************************************************************************\
dfyotc=df_yotc_DL[np.isfinite(df_yotc_DL['Dist Front'])]
x=np.array(dfyotc['Dist Front'])
y=np.sqrt(np.array(dfyotc['v 925'])**2+np.array(dfyotc['u 925'])**2)

bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x, y, statistic=np.std, bins=30)

ax9.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax9.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)

ax9.tick_params(axis='both', which='major', labelsize=14)
ax9.set_xticks(bx)
ax9.set_ylim([yyy1,yyy2])
ax9.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax9.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax9.set_title('Decoupled Layer',fontsize = 14 ,fontweight='bold')
ax9.grid()

plt.suptitle('YOTC (925 hPa)',fontsize = 14 ,fontweight='bold')
fig.savefig(path_data_save + 'windclas_yotc_fronts.eps', format='eps', dpi=1200)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                                   BRN
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
yy1=0
yy2=20
by=np.arange(yy1,yy2+2,2)

row = 2
column = 1

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(7,8))
ax1, ax2 =axes.flat
#*****************************************************************************\
#*****************************************************************************\
# MAC Ave
#*****************************************************************************\
#*****************************************************************************\

df_myclas = df_myfro[np.isfinite(df_myfro['Clas'])]
#*****************************************************************************\
df_mydist = df_myclas[np.isfinite(df_myclas['Dist Front'])]
x1=np.array(df_mydist['Dist Front'])
y1=np.array(df_mydist['BRN 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([yy1,yy2])
ax1.legend(loc='upper right')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax1.set_ylabel('Temp. ($^o$C)',fontsize = 12)
ax1.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax1.grid()

#*****************************************************************************\
#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\
df_yotcclas = df_yotcfro[np.isfinite(df_yotcfro['Clas'])]
df_yotcdist = df_yotcclas[np.isfinite(df_yotcclas['Dist Front'])]
x1=np.array(df_yotcdist['Dist Front'])
y1=np.array(df_yotcdist['BRN 925'])

bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='mean', bins=30)
bin_std, _, _ = stats.binned_statistic(x1, y1, statistic=np.std, bins=30)

ax2.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='prefront',yerr=[np.zeros(15), bin_std[0:df1]], error_kw=error_config)

ax2.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='postfront',yerr=[np.zeros(15), bin_std[df1:df2]], error_kw=error_config)


ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_ylim([yy1,yy2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax2.set_ylabel(r'$\theta_v$ ($^o$C)',fontsize = 12)
ax2.set_title('YOTC',fontsize = 14 ,fontweight='bold')
ax2.grid()

fig.tight_layout()
fig.savefig(path_data_save + 'brn_fronts.eps', format='eps', dpi=1200)
