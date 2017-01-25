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
from datetime import datetime, timedelta

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Scripts/Python/ERA-I/BLH/'

#*****************************************************************************\
#Default Info
latMac=-54.50;
lonMac=158.95;
date_index_12h = pd.date_range('2010-01-01 12:00', periods=730, freq='12H')
#*****************************************************************************\
#Reading Cold Fronts
#*****************************************************************************\
df_front= pd.read_csv(path_data + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')



df_front=df_front.reindex(date_index_12h)
df_front.index.name = 'Date'

#*****************************************************************************\
# Reading BLH ERA-I
#*****************************************************************************\
path_data_erai=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Scripts/Python/ERA-I/BLH/'

matb1= sio.loadmat(path_data_erai+'blh_erai_mac_2010.mat')
blh=matb1['blh'][:]
blh=blh[0,:]
time= matb1['time'][:]

#Building dates from 1900-01-01-00:00
time=time[0,:]
date_ini=datetime(1900, 1, 1) + timedelta(hours=int(time[0])) #hours since
#Date Array
date_erai = pd.date_range(date_ini, periods=len(time), freq='12H')

#*****************************************************************************\
blh_list=blh.tolist()
dy={'BLH':blh_list}
df_blh = pd.DataFrame(data=dy,index=date_erai)

#*****************************************************************************\
#Merge datraframe
#*****************************************************************************\
df_blhfro=pd.concat([df_blh, df_front],axis=1)

df_blhfro['Dist CFront']=df_blhfro['Dist CFront']*-1

df_blhfro[(df_blhfro['Dist CFront']>10)]=np.nan
df_blhfro[(df_blhfro['Dist CFront']<-10)]=np.nan


#*****************************************************************************\
df_blh_cfro = df_blhfro[np.isfinite(df_blhfro['Dist CFront'])]

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
y1=0
y2=2000
dy=200
bx=np.arange(xx1,xx2+1,2)
bins = np.arange(xx1, xx2+1, dx)
by=np.arange(y1,y2+100,dy)

error_config = {'ecolor': '0.4'}
error_config2 = {'ecolor': '0'}

bins=20


df1=10
df2=20

width = 1
#*****************************************************************************\
#*****************************************************************************\
# MAC
#*****************************************************************************\
#*****************************************************************************\
fig=plt.figure(facecolor='w', figsize=(12,8))

#*****************************************************************************\
#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\
x1=np.array(df_blh_cfro['Dist CFront'])
y=np.array(df_blh_cfro['BLH'])

# 1 Inversion
bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y, statistic='mean', bins=bins)
bin_std, _, _ = stats.binned_statistic(x1, y, statistic=np.std, bins=bins)

bin_edges=np.arange(xx1, xx2+1,1)

plt.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='postfront',yerr=[np.zeros(10), bin_std[0:df1]], error_kw=error_config)

plt.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='pretfront',yerr=[np.zeros(10), bin_std[df1:df2]], error_kw=error_config)


plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(bx)
plt.yticks(by)
plt.ylim([y1,y2])
plt.xlim([xx1,xx2])
plt.xlabel('Distance to front: cold to warm sector (deg)',fontsize = 12)
plt.ylabel('height (mts.)',fontsize = 12)
plt.title('ERA-i',fontsize = 14 ,fontweight='bold')
plt.grid()

plt.axvline(0, color='k')

#*****************************************************************************\

#*****************************************************************************\
fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
#fig.savefig(path_data_save + 'heights_fronts_era.eps', format='eps', dpi=1200)
plt.show()
