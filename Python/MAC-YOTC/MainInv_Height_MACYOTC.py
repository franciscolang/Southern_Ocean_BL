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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random as rd
import scipy.stats as st
import numpy as np
import scipy as sp
import scipy.stats
#from scipy import stats
from pylab import plot,show, grid, xlabel, ylabel, xlim, ylim, yticks, legend

base_dir = os.path.expanduser('~')

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Reading Height
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_csv=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/ERAI/'


#*****************************************************************************\
#Reading CSV
#*****************************************************************************\
df_front= pd.read_csv(path_data_csv + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')
#*****************************************************************************\
df_yotc= pd.read_csv(path_data_csv + 'MCR/df_yotc_19952010.csv', sep='\t', parse_dates=['Date'])
df_my= pd.read_csv(path_data_csv + 'MCR/df_macyotc_19952010_5k.csv', sep='\t', parse_dates=['Date'])


df_era= pd.read_csv(path_data_csv + 'MCR/df_era_19952010.csv', sep='\t', parse_dates=['Date'])

#df_era= pd.read_csv(path_data_csv + 'MCR/df_macera_19952010_5k.csv', sep='\t', parse_dates=['Date'])

df_mei= pd.read_csv(path_data_csv + 'MCR/df_macera_19952010_5k.csv', sep='\t', parse_dates=['Date'])


df_yotc= df_yotc.set_index('Date')
df_my= df_my.set_index('Date')
df_era= df_era.set_index('Date')
df_mei= df_mei.set_index('Date')
#*****************************************************************************\
#Merge datraframe mac with
df_yotcfro=pd.concat([df_yotc, df_front],axis=1)
df_myfro=pd.concat([df_my, df_front],axis=1)

df_erafro=pd.concat([df_era, df_front],axis=1)
df_meifro=pd.concat([df_mei, df_front],axis=1)


df_yotcfro['Dist Front']=df_yotcfro['Dist CFront']*-1
df_myfro['Dist Front']=df_myfro['Dist CFront']*-1


df_erafro['Dist Front']=df_erafro['Dist CFront']*-1
df_meifro['Dist Front']=df_meifro['Dist CFront']*-1


df_yotcfro[(df_yotcfro['Dist Front']>10)]=np.nan
df_yotcfro[(df_yotcfro['Dist Front']<-10)]=np.nan
df_myfro[(df_myfro['Dist Front']>10)]=np.nan
df_myfro[(df_myfro['Dist Front']<-10)]=np.nan


df_erafro[(df_erafro['Dist Front']>10)]=np.nan
df_erafro[(df_erafro['Dist Front']<-10)]=np.nan
df_meifro[(df_meifro['Dist Front']>10)]=np.nan
df_meifro[(df_meifro['Dist Front']<-10)]=np.nan


#*****************************************************************************\
#Clasification by type
#*****************************************************************************\

df_1i= df_yotcfro[np.isfinite(df_yotcfro['1ra Inv'])]
df_yotc_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]


df_1s= df_yotcfro[np.isfinite(df_yotcfro['Strg 1inv'])]
df_yotc_1str = df_1s[np.isfinite(df_1s['Dist Front'])]


#*****************************************************************************\
df_1i= df_myfro[np.isfinite(df_myfro['1ra Inv'])]
df_my_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

df_1s= df_myfro[np.isfinite(df_myfro['Strg 1inv'])]
df_my_1str = df_1s[np.isfinite(df_1s['Dist Front'])]

#*****************************************************************************\
df_1i= df_erafro[np.isfinite(df_erafro['1ra Inv'])]
df_era_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

df_1s= df_erafro[np.isfinite(df_erafro['Strg 1inv'])]
df_era_1str = df_1s[np.isfinite(df_1s['Dist Front'])]

#*****************************************************************************\
df_1i= df_meifro[np.isfinite(df_meifro['1ra Inv'])]
df_mei_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]


df_1s= df_meifro[np.isfinite(df_meifro['Strg 1inv'])]
df_mei_1str = df_1s[np.isfinite(df_1s['Dist Front'])]
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


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Height
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

#YOTC
x1=np.array(df_yotc_1inv['Dist Front'])
y=np.array(df_yotc_1inv['1ra Inv'])

bin_means_yotc, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_yotc, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)
del x1, y

#MAC-YOTC
x1=np.array(df_my_1inv['Dist Front'])
y=np.array(df_my_1inv['1ra Inv'])
bin_means_mac, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_mac, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)
del x1, y

#MAC-ERA-i
x1=np.array(df_mei_1inv['Dist Front'])
y=np.array(df_mei_1inv['1ra Inv'])
bin_means_mei, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_mei, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)
del x1, y

#ERA-i
x1=np.array(df_era_1inv['Dist Front'])
y=np.array(df_era_1inv['1ra Inv'])
bin_means_era, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_std_era, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)
del x1, y



bin_edges=np.arange(-9.5, 10.5, 1)


fig=plt.figure(figsize=(10, 6))
ax0=fig.add_subplot(111)
# ax0.plot(bin_edges,bin_means_mac,'-o', label='MAC')
# ax0.plot(bin_edges,bin_means_era,'-or', label='YOTC')

# ax0.errorbar(bin_edges, bin_means_mac, yerr=bin_std_mac, fmt='-o',label='MAC-YOTC',color='cornflowerblue')
# ax0.errorbar(bin_edges, bin_means_yotc, yerr=bin_std_yotc, fmt='-o',label='YOTC',color='tomato')

# ax0.errorbar(bin_edges, bin_means_mei, yerr=bin_std_mei, fmt='-og',label='MAC-ERAi')
# ax0.errorbar(bin_edges, bin_means_era, yerr=bin_std_era*1.1, fmt='-oy',label='ERAi')

ax0.errorbar(bin_edges, bin_means_mei, yerr=bin_std_mei, fmt='-o',label='MAC',color='cornflowerblue')
ax0.errorbar(bin_edges, bin_means_era, yerr=bin_std_era*1.2, fmt='-o',label='ERA-i',color='tomato')



ax0.set_ylabel('height (m)',fontsize = 14)
ax0.set_xlabel('Distance to front: cold to warm sector (deg)',fontsize = 14)
ax0.legend(loc=1,fontsize = 12, numpoints=1)
ax0.axvline(0, color='k')
ax0.set_ylim(0,5000)
ax0.set_yticks(np.arange(0,5500,500))
ax0.set_xticks(np.arange(-10,11,2))
ax0.grid()

fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig(path_data_save + 'heights_erai.eps', format='eps', dpi=1200)




#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Strength
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

#YOTC
x1=np.array(df_yotc_1str['Dist Front'])
y=np.array(df_yotc_1str['Strg 1inv'])

bin_stg_yotc, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_stg_yotc, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)
del x1, y

#MAC-YOTC
x1=np.array(df_my_1str['Dist Front'])
y=np.array(df_my_1str['Strg 1inv'])

bin_stg_mac, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_stg_mac, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)



#MAC-ERA-i
x1=np.array(df_mei_1inv['Dist Front'])
y=np.array(df_mei_1inv['Strg 1inv'])
bin_stg_mei, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_stg_mei, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)
del x1, y

#ERA-i
x1=np.array(df_era_1inv['Dist Front'])
y=np.array(df_era_1inv['Strg 1inv'])
bin_stg_era, bin_edges, binnumber = st.binned_statistic(x1, y, statistic='mean', bins=20)
bin_stg_era, _, _ = st.binned_statistic(x1, y, statistic=np.std, bins=20)
del x1, y



bin_edges=np.arange(-9.5, 10.5, 1)


fig=plt.figure(figsize=(10, 6))
ax0=fig.add_subplot(111)
# ax0.plot(bin_edges,bin_means_mac,'-o', label='MAC')
# ax0.plot(bin_edges,bin_means_era,'-or', label='YOTC')

# ax0.errorbar(bin_edges, bin_stg_mac, yerr=bin_stg_mac, fmt='-og',label='MAC-YOTC', markersize=6)
# ax0.errorbar(bin_edges, bin_stg_yotc, yerr=bin_stg_yotc, fmt='-oy',label='YOTC', markersize=6)

# ax0.errorbar(bin_edges, bin_stg_mei, yerr=bin_stg_mei, fmt='-og',label='MAC-ERAi')
# ax0.errorbar(bin_edges, bin_stg_era/float(10), yerr=bin_stg_era/float(10), fmt='-oy',label='ERAi')


ax0.errorbar(bin_edges, bin_stg_mei, yerr=bin_stg_mei, fmt='-o',label='MAC',color='cornflowerblue', markersize=6)
ax0.errorbar(bin_edges, bin_stg_era/float(10), yerr=bin_stg_era/float(10), fmt='-o',label='ERA-i',color='tomato', markersize=6)

ax0.set_ylabel('strength (K m$^{-1}$)',fontsize = 14)
ax0.set_xlabel('Distance to front: cold to warm sector (deg)',fontsize = 14)
ax0.legend(loc=1,fontsize = 12, numpoints=1)
ax0.axvline(0, color='k')
ax0.set_ylim(0,0.06)
#ax0.set_yticks(np.arange(0,0.05,0.005))
ax0.set_xticks(np.arange(-10,11,2))
ax0.grid()

fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig(path_data_save + 'strength_erai.eps', format='eps', dpi=1200)

plt.show()
