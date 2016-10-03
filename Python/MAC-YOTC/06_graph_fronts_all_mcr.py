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


base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts_ok/MCR/'

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
df_mac= pd.read_csv(path_data + 'MCR/df_mac_19952010_2.csv', sep='\t', parse_dates=['Date'])
df_yotc= pd.read_csv(path_data + 'MCR/df_yotc_19952010_2.csv', sep='\t', parse_dates=['Date'])
df_my= pd.read_csv(path_data + 'MCR/df_macyotc_19952010_2.csv', sep='\t', parse_dates=['Date'])

df_mac= df_mac.set_index('Date')
df_yotc= df_yotc.set_index('Date')
df_my= df_my.set_index('Date')
#*****************************************************************************\
#Merge datraframe mac with
df_macfro=pd.concat([df_mac, df_front],axis=1)
df_yotcfro=pd.concat([df_yotc, df_front],axis=1)
df_myfro=pd.concat([df_my, df_front],axis=1)

df_macfro['Dist Front']=df_macfro['Dist CFront']*-1
df_yotcfro['Dist Front']=df_yotcfro['Dist CFront']*-1
df_myfro['Dist Front']=df_myfro['Dist CFront']*-1




df_macfro[(df_macfro['Dist Front']>10)]=np.nan
df_macfro[(df_macfro['Dist Front']<-10)]=np.nan
df_yotcfro[(df_yotcfro['Dist Front']>10)]=np.nan
df_yotcfro[(df_yotcfro['Dist Front']<-10)]=np.nan
df_myfro[(df_myfro['Dist Front']>10)]=np.nan
df_myfro[(df_myfro['Dist Front']<-10)]=np.nan


#*****************************************************************************\
#Clasification by type
#*****************************************************************************\

df_1i= df_macfro[np.isfinite(df_macfro['1ra Inv'])]
df_mac_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

# df_2i= df_macfro[np.isfinite(df_macfro['2da Inv'])]
# df_mac_2inv = df_2i[np.isfinite(df_2i['Dist Front'])]

df_1s= df_macfro[np.isfinite(df_macfro['Strg 1inv'])]
df_mac_1str = df_1s[np.isfinite(df_1s['Dist Front'])]

# df_2s= df_macfro[np.isfinite(df_macfro['Strg 2inv'])]
# df_mac_2str = df_2s[np.isfinite(df_2s['Dist Front'])]


#*****************************************************************************\
df_BL1 = df_yotcfro[df_yotcfro['Clas']==4]
df_yotc_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

df_DL1 = df_yotcfro[df_yotcfro['Clas']==3]
df_yotc_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

df_SI1 = df_yotcfro[df_yotcfro['Clas']==2]
df_yotc_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

df_NI1 = df_yotcfro[df_yotcfro['Clas']==1]
df_yotc_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

df_1i= df_yotcfro[np.isfinite(df_yotcfro['1ra Inv'])]
df_yotc_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

# df_2i= df_yotcfro[np.isfinite(df_yotcfro['2da Inv'])]
# df_yotc_2inv = df_2i[np.isfinite(df_2i['Dist Front'])]

df_1s= df_yotcfro[np.isfinite(df_yotcfro['Strg 1inv'])]
df_yotc_1str = df_1s[np.isfinite(df_1s['Dist Front'])]

# df_2s= df_yotcfro[np.isfinite(df_yotcfro['Strg 2inv'])]
# df_yotc_2str = df_2s[np.isfinite(df_2s['Dist Front'])]

del df_BL1, df_DL1, df_NI1, df_SI1
#*****************************************************************************\

df_BL1 = df_myfro[df_myfro['Clas']==4]
df_my_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

df_DL1 = df_myfro[df_myfro['Clas']==3]
df_my_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

df_SI1 = df_myfro[df_myfro['Clas']==2]
df_my_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

df_NI1 = df_myfro[df_myfro['Clas']==1]
df_my_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

df_1i= df_myfro[np.isfinite(df_myfro['1ra Inv'])]
df_my_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

# df_2i= df_myfro[np.isfinite(df_myfro['2da Inv'])]
# df_my_2inv = df_2i[np.isfinite(df_2i['Dist Front'])]

df_1s= df_myfro[np.isfinite(df_myfro['Strg 1inv'])]
df_my_1str = df_1s[np.isfinite(df_1s['Dist Front'])]

# df_2s= df_myfro[np.isfinite(df_myfro['Strg 2inv'])]
# df_my_2str = df_2s[np.isfinite(df_2s['Dist Front'])]

#*****************************************************************************\
#Percentages
#*****************************************************************************\
#Total Number Sounding at MAC of 3652
n_macsound=len(df_macfro[np.isfinite(df_macfro['Clas'])])
#Total Number Sounding at YOTC of 1460
n_yotcsound=len(df_yotcfro[np.isfinite(df_yotcfro['Clas'])])
#Total Number Sounding at MAC of 3652
n_mysound=len(df_myfro[np.isfinite(df_myfro['Clas'])])

#Sounding MAC
df_macsound=df_macfro[np.isfinite(df_macfro['Clas'])]
#Sounding YOTC
df_yotcsound=df_yotcfro[np.isfinite(df_yotcfro['Clas'])]
#Sounding MAC-YOTC
df_mysound=df_myfro[np.isfinite(df_myfro['Clas'])]


#Total Number Sounding at MAC with front
n_macfronts=len(df_macsound[np.isfinite(df_macsound['Dist Front'])])
#Total Number Sounding at YOTC with front
n_yotcfronts=len(df_yotcsound[np.isfinite(df_yotcsound['Dist Front'])])
#Total Number Sounding at MAC-YOTC with front
n_myfronts=len(df_mysound[np.isfinite(df_mysound['Dist Front'])])


#Percentages fronts regarding soundings
per_macfro=n_macfronts/float(n_macsound)*100
per_yotcfro=n_yotcfronts/float(n_yotcsound)*100
per_myfro=n_myfronts/float(n_mysound)*100

print per_macfro,per_yotcfro, per_myfro



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
column = 3

error_config = {'ecolor': '0.4'}
error_config2 = {'ecolor': '0'}

bins=20

#*****************************************************************************\
#*****************************************************************************\
# MAC
#*****************************************************************************\
#*****************************************************************************\

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(16,4))
ax1, ax3, ax5 = axes.flat
#*****************************************************************************\
x1=np.array(df_mac_1inv['Dist Front'])
y=np.array(df_mac_1inv['1ra Inv'])


# 1 Inversion
bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y, statistic='mean', bins=bins)
bin_std, _, _ = stats.binned_statistic(x1, y, statistic=np.std, bins=bins)

bin_edges=np.arange(xx1, xx2+1, 1)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='postfront',yerr=[np.zeros(10), bin_std[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='pretfront',yerr=[np.zeros(10), bin_std[df1:df2]], error_kw=error_config)



ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_ylim([y1,y2])
#ax1.set_xlabel('Distance to front:cold to warm sector (deg)',fontsize = 12)
ax1.set_ylabel('height (mts.)',fontsize = 12)
ax1.set_title('MAC',fontsize = 14 ,fontweight='bold')
ax1.grid()
ax1.set_xlim([xx1,xx2])
ax1.axvline(0, color='k')
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
ax3.yaxis.set_ticklabels([])
ax3.set_xlabel('Distance to front: cold to warm sector (deg)',fontsize = 12)
#ax3.set_ylabel('Height (mts.)',fontsize = 12)
ax3.set_title('YOTC',fontsize = 14 ,fontweight='bold')
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
#ax5.set_xlabel('Distance to front:cold to warm sector (deg)',fontsize = 12)
#ax5.set_ylabel('Height (mts.)',fontsize = 12)
ax5.legend(loc='upper right')
ax5.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax5.grid()
ax5.set_xlim([xx1,xx2])
ax5.axvline(0, color='k')
#*****************************************************************************\
fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
fig.savefig(path_data_save + 'heights_fronts.eps', format='eps', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                            Inversion Height Position
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Graphs Setup
#*****************************************************************************\
xx1=-10
xx2=10

y1=0
y2=0.08
dx=1
bx=np.arange(xx1,xx2+1,2)
by=np.arange(y1,y2+0.01,0.01)
#bins = np.arange(xx1, xx2+1, dx)

df1=10
df2=20
width = 1
row = 1
column = 3

error_config = {'ecolor': '0.4'}
error_config2 = {'ecolor': '0'}

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(16,4))
ax1, ax3, ax5 = axes.flat
#*****************************************************************************\
#*****************************************************************************\
# MAC
#*****************************************************************************\
#*****************************************************************************\
#De los Mac Soundings, deja solo los con un front
#np.count_nonzero(~np.isnan(y))
df_macsoundfro=df_macsound[np.isfinite(df_macsound['Dist Front'])]
df_macsoundfro1inv=df_macsoundfro[np.isfinite(df_macsoundfro['1ra Inv'])]

x1=np.array(df_macsoundfro['Dist Front'])

x2=np.array(df_macsoundfro1inv['Dist Front'])
y2=np.array(df_macsoundfro1inv['1ra Inv'])

bin_count2, bin_edges2, _ = stats.binned_statistic(x2, y2, statistic='count', bins=bins)

bin_edges2=np.arange(xx1, xx2+1, 1)

per_count=bin_count2/float(len(x1))

ax1.bar(bin_edges2[0:df1],per_count[0:df1],width,alpha=0.5, color=color1,label='prefront')

ax1.bar(bin_edges2[df1:df2],per_count[df1:df2],width,alpha=0.5, color=color2,label='postfront')

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
#ax1.set_ylim([y1,y2])
#ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('relative frequency',fontsize = 12)
ax1.set_title('MAC',fontsize = 14 ,fontweight='bold')
ax1.grid()
ax1.set_xlim([xx1,xx2])
ax1.axvline(0, color='k')

#*****************************************************************************\
#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\
#De los Mac Soundings, deja solo los con un front
#np.count_nonzero(~np.isnan(y))
df_yotcsoundfro=df_yotcsound[np.isfinite(df_yotcsound['Dist Front'])]
df_yotcsoundfro1inv=df_yotcsoundfro[np.isfinite(df_yotcsoundfro['1ra Inv'])]

x1=np.array(df_yotcsoundfro['Dist Front'])

x2=np.array(df_yotcsoundfro1inv['Dist Front'])
y2=np.array(df_yotcsoundfro1inv['1ra Inv'])

bin_count2, bin_edges2, _ = stats.binned_statistic(x2, y2, statistic='count', bins=bins)

bin_edges2=np.arange(xx1, xx2+1, 1)
#Esta contando los casos cuando no hay inversion
per_count=bin_count2/float(len(x1))


ax3.bar(bin_edges2[0:df1],per_count[0:df1],width,alpha=0.5, color=color1,label='prefront')

ax3.bar(bin_edges2[df1:df2],per_count[df1:df2],width,alpha=0.5, color=color2,label='postfront')

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.yaxis.set_ticklabels([])
#ax1.set_ylim([y1,y2])
#ax3.legend(loc='upper left')
ax3.set_xlabel('Distance to front: cold to warm sector (deg)',fontsize = 12)
#ax3.set_ylabel('Relative frequency',fontsize = 12)
ax3.set_title('YOTC',fontsize = 14 ,fontweight='bold')
ax3.grid()
ax3.set_xlim([xx1,xx2])
ax3.axvline(0, color='k')

#*****************************************************************************\
#*****************************************************************************\
# MAC-YOTC
#*****************************************************************************\
#*****************************************************************************\
df_mysoundfro=df_mysound[np.isfinite(df_mysound['Dist Front'])]
df_mysoundfro1inv=df_mysoundfro[np.isfinite(df_mysoundfro['1ra Inv'])]

x1=np.array(df_mysoundfro['Dist Front'])

x2=np.array(df_mysoundfro1inv['Dist Front'])
y2=np.array(df_mysoundfro1inv['1ra Inv'])

bin_count2, bin_edges2, _ = stats.binned_statistic(x2, y2, statistic='count', bins=bins)

bin_edges2=np.arange(xx1, xx2+1, 1)

per_count=bin_count2/float(len(x1))

ax5.bar(bin_edges2[0:df1],per_count[0:df1],width,alpha=0.5, color=color1,label='prefront')

ax5.bar(bin_edges2[df1:df2],per_count[df1:df2],width,alpha=0.5, color=color2,label='postfront')

ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_yticks(by)
ax5.yaxis.set_ticklabels([])
#ax1.set_ylim([y1,y2])
#ax5.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax5.set_ylabel('Relative frequency',fontsize = 12)
ax5.legend(loc='upper right')
ax5.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax5.grid()
ax5.set_xlim([xx1,xx2])
ax5.axvline(0, color='k')


fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
fig.savefig(path_data_save + 'relatheights_fronts.eps', format='eps', dpi=1200)













#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                            Inversion Streng
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



y1=0
y2=0.14
by=np.arange(y1,y2+0.01,0.03)


df1=10
df2=20
width = 1
row = 1
column = 3

error_config = {'ecolor': '0.4'}
error_config2 = {'ecolor': '0'}

color1='lightgreen'
color2='forestgreen'

#*****************************************************************************\
#*****************************************************************************\
# MAC
#*****************************************************************************\
#*****************************************************************************\


fig, axes = plt.subplots(row, column, facecolor='w', figsize=(16,4))
ax1, ax3, ax5 = axes.flat
#*****************************************************************************\
del x1, y, bin_std
x1=np.array(df_mac_1str['Dist Front'])
y=np.array(df_mac_1str['Strg 1inv'])

# 1 Inversion
bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y, statistic='mean', bins=bins)
bin_std1, _, _ = stats.binned_statistic(x1, y, statistic=np.std, bins=bins)


for i in range(0, len(bin_std1)):
    if bin_std1[i]>=0.1:
        bin_std1[i]=bin_std1[i]/float(3)
    else:
        bin_std1[i]=bin_std1[i]

bin_edges=np.arange(xx1, xx2+1, 1)

ax1.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='postfront',yerr=[np.zeros(10), bin_std1[0:df1]], error_kw=error_config)

ax1.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='prefront',yerr=[np.zeros(10), bin_std1[df1:df2]], error_kw=error_config)


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_ylim([y1,y2])

#ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('strength (K m$^{-1}$)',fontsize = 12)
ax1.set_title('MAC',fontsize = 14 ,fontweight='bold')
ax1.grid()
ax1.set_xlim([xx1,xx2])
ax1.axvline(0, color='k')

#*****************************************************************************\
#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\
x1=np.array(df_yotc_1str['Dist Front'])
y=np.array(df_yotc_1str['Strg 1inv'])

# 1 Inversion
bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y, statistic='mean', bins=bins)
bin_std, _, _ = stats.binned_statistic(x1, y, statistic=np.std, bins=bins)



bin_edges=np.arange(xx1, xx2+1, 1)

ax3.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='postfront',yerr=[np.zeros(10), bin_std[0:df1]], error_kw=error_config)

ax3.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='prefront',yerr=[np.zeros(10), bin_std[df1:df2]], error_kw=error_config)



ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_ylim([y1,y2])
ax3.yaxis.set_ticklabels([])
#ax3.legend(loc='upper left')
ax3.set_xlabel('Distance to front: cold to warm sector (deg)',fontsize = 12)
#ax3.set_ylabel('strength (K m$^{-1}$)',fontsize = 12)
ax3.set_title('YOTC',fontsize = 14 ,fontweight='bold')
ax3.grid()
ax3.set_xlim([xx1,xx2])
ax3.axvline(0, color='k')

#*****************************************************************************\
#*****************************************************************************\
# MAC-YOTC
#*****************************************************************************\
#*****************************************************************************\
x1=np.array(df_my_1str['Dist Front'])
y=np.array(df_my_1str['Strg 1inv'])

# 1 Inversion
bin_means, bin_edges, binnumber = stats.binned_statistic(x1, y, statistic='mean', bins=bins)
bin_std, _, _ = stats.binned_statistic(x1, y, statistic=np.std, bins=bins)

bin_edges=np.arange(xx1, xx2+1, 1)

ax5.bar(bin_edges[0:df1],bin_means[0:df1],width,alpha=0.5, color=color1,label='postfront',yerr=[np.zeros(10), bin_std[0:df1]], error_kw=error_config)

ax5.bar(bin_edges[df1:df2],bin_means[df1:df2],width,alpha=0.5, color=color2,label='prefront',yerr=[np.zeros(10), bin_std[df1:df2]], error_kw=error_config)

ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_ylim([y1,y2])
ax5.yaxis.set_ticklabels([])
#ax5.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax5.set_ylabel('strength (K m$^{-1}$)',fontsize = 12)
ax5.legend(loc='upper right')
ax5.set_title('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
ax5.grid()
ax5.set_xlim([xx1,xx2])
ax5.axvline(0, color='k')

fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
fig.savefig(path_data_save + 'strength_fronts.eps', format='eps', dpi=1200)
plt.show()
