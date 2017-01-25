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
import scipy.stats as st


base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts_ok/'

#*****************************************************************************\
#Default Info
latMac=-54.50;
lonMac=158.95;

#*****************************************************************************\
#Reading CSV
#*****************************************************************************\
df_front= pd.read_csv(path_data + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')
#*****************************************************************************\
df_yotc= pd.read_csv(path_data + 'MCR/df_era_19952010.csv', sep='\t', parse_dates=['Date'])

df_yotc2= pd.read_csv(path_data + 'MCR/df_era_19952010_5k.csv', sep='\t', parse_dates=['Date'])

df_my= pd.read_csv(path_data + 'MCR/df_macera_19952010_5k.csv', sep='\t', parse_dates=['Date'])

df_yotc= df_yotc.set_index('Date')
df_yotc2= df_yotc2.set_index('Date')
df_my= df_my.set_index('Date')
#*****************************************************************************\
#Merge datraframe mac with
df_yotcfro=pd.concat([df_yotc, df_front],axis=1)
df_yotc2fro=pd.concat([df_yotc2, df_front],axis=1)
df_myfro=pd.concat([df_my, df_front],axis=1)

df_yotcfro['Dist Front']=df_yotcfro['Dist CFront']*-1
df_yotc2fro['Dist Front']=df_yotc2fro['Dist CFront']*-1
df_myfro['Dist Front']=df_myfro['Dist CFront']*-1

df_yotcfro[(df_yotcfro['Dist Front']>10)]=np.nan
df_yotcfro[(df_yotcfro['Dist Front']<-10)]=np.nan
df_yotc2fro[(df_yotc2fro['Dist Front']>10)]=np.nan
df_yotc2fro[(df_yotc2fro['Dist Front']<-10)]=np.nan
df_myfro[(df_myfro['Dist Front']>10)]=np.nan
df_myfro[(df_myfro['Dist Front']<-10)]=np.nan


#*****************************************************************************\
#Clasification by type
#*****************************************************************************\

df_1i= df_yotcfro[np.isfinite(df_yotcfro['1ra Inv'])]
df_yotc_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]


df_1s= df_yotcfro[np.isfinite(df_yotcfro['Strg 1inv'])]
df_yotc_1str = df_1s[np.isfinite(df_1s['Dist Front'])]


df_1i= df_yotc2fro[np.isfinite(df_yotc2fro['1ra Inv'])]
df_yotc2_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]


df_1s= df_yotc2fro[np.isfinite(df_yotc2fro['Strg 1inv'])]
df_yotc2_1str = df_1s[np.isfinite(df_1s['Dist Front'])]


#*****************************************************************************\
df_1i= df_myfro[np.isfinite(df_myfro['1ra Inv'])]
df_my_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

df_1s= df_myfro[np.isfinite(df_myfro['Strg 1inv'])]
df_my_1str = df_1s[np.isfinite(df_1s['Dist Front'])]


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

print per_yotcfro, per_myfro



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

path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts_ok/'

#*****************************************************************************\
x1=-10
x2=10
y1=0
y2=0.24
dx=1
bx=np.arange(x1,x2+1,3)
by=np.arange(y1,y2+0.02,0.02)
bins = np.arange(x1, x2+1, dx)

df1=15
df2=30
width = 1



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                            Inversion Height
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Graphs Setup
#*****************************************************************************\
xx1=-10
xx2=10
dx=1
bx=np.arange(xx1,xx2+1,2)
bins = np.arange(x1, x2+1, dx)

y1=0
y2=4200
by=np.arange(y1,y2+100,600)


df1=10
df2=20
width = 1
row = 1
column = 2

error_config = {'ecolor': '0.4'}
error_config2 = {'ecolor': '0'}

bins=20

#*****************************************************************************\
#*****************************************************************************\
# MAC
#*****************************************************************************\
#*****************************************************************************\

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(11,4))
ax3, ax5 = axes.flat

#*****************************************************************************\
#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\
x1=np.array(df_yotc_1inv['Dist Front'])
y=np.array(df_yotc_1inv['1ra Inv'])

# 1 Inversion
bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y, statistic='mean', bins=bins)
bin_std, _, _ = stats.binned_statistic(x1, y, statistic=np.std, bins=bins)

bin_edges=np.arange(xx1, xx2+1, 1)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='postfront',yerr=[np.zeros(10), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='pretfront',yerr=[np.zeros(10), bin_std[df1:df2]], error_kw=error_config)


ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_ylim([y1,y2])
#ax3.yaxis.set_ticklabels([])
ax3.set_xlabel('Distance to front: cold to warm sector (deg)',fontsize = 12)
ax3.set_ylabel('height (mts.)',fontsize = 12)
ax3.set_title('ERA-i',fontsize = 14 ,fontweight='bold')
ax3.grid()
ax3.set_xlim([xx1,xx2])
ax3.axvline(0, color='k')

#*****************************************************************************\
#*****************************************************************************\
# MAC-YOTC
#*****************************************************************************\
#*****************************************************************************\
x1=np.array(df_my_1inv['Dist Front'])
y=np.array(df_my_1inv['1ra Inv'])

# 1 Inversion
bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y, statistic='mean', bins=bins)
bin_std, _, _ = stats.binned_statistic(x1, y, statistic=np.std, bins=bins)

bin_edges=np.arange(xx1, xx2+1, 1)

ax5.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='postfront',yerr=[np.zeros(10), bin_std[0:df1]], error_kw=error_config)

ax5.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='pretfront',yerr=[np.zeros(10), bin_std[df1:df2]], error_kw=error_config)


ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_yticks(by)
ax5.set_ylim([y1,y2])
#ax5.set_xlim([x1,x2])
ax5.yaxis.set_ticklabels([])
ax5.set_xlabel('Distance to front: cold to warm sector (deg)',fontsize = 12)
#ax5.set_ylabel('Height (mts.)',fontsize = 12)
ax5.legend(loc='upper right')
ax5.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax5.grid()
ax5.set_xlim([xx1,xx2])
ax5.axvline(0, color='k')
#*****************************************************************************\
fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
#fig.savefig(path_data_save + 'heights_fronts_era.eps', format='eps', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Plot line Height
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

x1=np.array(df_yotc_1inv['Dist Front'])
y=np.array(df_yotc_1inv['1ra Inv'])

bin_means_era, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_era, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)

#*****************************************************************************\
x1=np.array(df_yotc2_1inv['Dist Front'])
y=np.array(df_yotc2_1inv['1ra Inv'])

bin_means_era2, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_era2, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)

#*****************************************************************************\
x1=np.array(df_my_1inv['Dist Front'])
y=np.array(df_my_1inv['1ra Inv'])

bin_means_mac, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_mac, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)


bin_edges=np.arange(-9.5, 10.5, 1)
#*****************************************************************************\
fig=plt.figure( facecolor='w',figsize=(10, 6))
ax=fig.add_subplot(111)
ax.plot(bin_edges,bin_means_mac,'-o', color='#1f77b4')
ax.plot(bin_edges,bin_means_era,'-o',color='tomato')
# ax.plot(bin_edges,bin_means_era2,'-o',color='#9edae5')

ax.errorbar(bin_edges, bin_means_mac, yerr=bin_std_mac, fmt='-o',label='MAC', color='#1f77b4')
ax.errorbar(bin_edges, bin_means_era, yerr=bin_std_era*1.2, fmt='-o',label='ERA-i',color='tomato')
# ax.errorbar(bin_edges, bin_means_era2, yerr=bin_std_era2, fmt='-o',label='ERA-i2',color='#9edae5')

ax.set_xticks(np.arange(-10,12,2))
ax.set_xticklabels(np.arange(-10,12,2), size=16)
ax.set_ylim([0,5000])
ax.set_yticks(np.arange(0,5500,500))
ax.set_yticklabels(np.arange(0,5500,500), size=16)
ax.set_ylabel('Height (m)',fontsize = 18)
ax.set_xlabel('Distance to front: cold to warm sector (deg)', size=18)
ax.legend(loc=3,fontsize = 16, numpoints=1)
ax.axvline(0,color='k')
ax.grid()
fig.tight_layout()
fig.savefig(path_data_save + 'heights_cfronts.eps', format='eps', dpi=1200)







#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Plot line Strenght
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

x1=np.array(df_yotc_1inv['Dist Front'])
y=np.array(df_yotc_1inv['Strg 1inv'])

bin_means_era, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_era, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)

#*****************************************************************************\
x1=np.array(df_yotc2_1inv['Dist Front'])
y=np.array(df_yotc2_1inv['Strg 1inv'])

bin_means_era2, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_era2, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)

#*****************************************************************************\
x1=np.array(df_my_1inv['Dist Front'])
y=np.array(df_my_1inv['Strg 1inv'])

bin_means_mac, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_mac, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)


bin_edges=np.arange(-9.5, 10.5, 1)
#*****************************************************************************\
fig=plt.figure( facecolor='w',figsize=(10, 6))
ax=fig.add_subplot(111)
ax.plot(bin_edges,bin_means_mac,'-o', color='#1f77b4')
ax.plot(bin_edges,bin_means_era2,'-o',color='tomato')
# ax.plot(bin_edges,bin_means_era2,'-o',color='#9edae5')

ax.errorbar(bin_edges, bin_means_mac, yerr=bin_std_mac, fmt='-o',label='MAC', color='#1f77b4')
ax.errorbar(bin_edges, bin_means_era2, yerr=bin_std_era2, fmt='-o',label='ERA-i',color='tomato')

ax.set_xticks(np.arange(-10,12,2))
ax.set_xticklabels(np.arange(-10,12,2), size=16)
ax.set_ylim([0,0.12])
ax.set_yticks(np.arange(0,0.14,0.03))
ax.set_yticklabels(np.arange(0,0.14,0.03), size=16)
ax.set_ylabel('strength (K m$^{-1}$)',fontsize = 16)
ax.set_xlabel('Distance to front: cold to warm sector (deg)', size=18)
ax.legend(loc=3,fontsize = 16, numpoints=1)
ax.axvline(0,color='k')
ax.grid()
fig.tight_layout()
#fig.savefig(path_data_save + 'streght_cfronts.eps', format='eps', dpi=1200)




plt.show()
