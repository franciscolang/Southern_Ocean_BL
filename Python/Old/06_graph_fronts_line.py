import numpy as np
import scipy.io as sio
import os
from pylab import plot,show, grid
import math
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import stats

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
df_mac= pd.read_csv(path_data + 'df_mac_final.csv', sep='\t', parse_dates=['Date'])
df_yotc= pd.read_csv(path_data + 'df_yotc_all.csv', sep='\t', parse_dates=['Date'])
df_my= pd.read_csv(path_data + 'df_macyotc_final.csv', sep='\t', parse_dates=['Date'])

df_mac= df_mac.set_index('Date')
df_yotc= df_yotc.set_index('Date')
df_my= df_my.set_index('Date')
#*****************************************************************************\
#Merge datraframe mac with
df_macfro=pd.concat([df_mac, df_front],axis=1)
df_yotcfro=pd.concat([df_yotc, df_front],axis=1)
df_myfro=pd.concat([df_my, df_front],axis=1)

#*****************************************************************************\
#Clasification by type
#*****************************************************************************\
df_BL1 = df_macfro[df_macfro['Clas']==4]
df_mac_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

df_DL1 = df_macfro[df_macfro['Clas']==3]
df_mac_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

df_SI1 = df_macfro[df_macfro['Clas']==2]
df_mac_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

df_NI1 = df_macfro[df_macfro['Clas']==1]
df_mac_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

df_1i= df_macfro[np.isfinite(df_macfro['1ra Inv'])]
df_mac_1inv = df_1i[np.isfinite(df_1i['Dist Front'])]

df_2i= df_macfro[np.isfinite(df_macfro['2da Inv'])]
df_mac_2inv = df_2i[np.isfinite(df_2i['Dist Front'])]

df_1s= df_macfro[np.isfinite(df_macfro['Strg 1inv'])]
df_mac_1str = df_1s[np.isfinite(df_1s['Dist Front'])]

df_2s= df_macfro[np.isfinite(df_macfro['Strg 2inv'])]
df_mac_2str = df_2s[np.isfinite(df_2s['Dist Front'])]


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

df_2i= df_yotcfro[np.isfinite(df_yotcfro['2da Inv'])]
df_yotc_2inv = df_2i[np.isfinite(df_2i['Dist Front'])]

df_1s= df_yotcfro[np.isfinite(df_yotcfro['Strg 1inv'])]
df_yotc_1str = df_1s[np.isfinite(df_1s['Dist Front'])]

df_2s= df_yotcfro[np.isfinite(df_yotcfro['Strg 2inv'])]
df_yotc_2str = df_2s[np.isfinite(df_2s['Dist Front'])]

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

df_2i= df_myfro[np.isfinite(df_myfro['2da Inv'])]
df_my_2inv = df_2i[np.isfinite(df_2i['Dist Front'])]

df_1s= df_myfro[np.isfinite(df_myfro['Strg 1inv'])]
df_my_1str = df_1s[np.isfinite(df_1s['Dist Front'])]

df_2s= df_myfro[np.isfinite(df_myfro['Strg 2inv'])]
df_my_2str = df_2s[np.isfinite(df_2s['Dist Front'])]

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

#print per_macfro,per_yotcfro, per_myfro

#Percentages regarding to total number MAC sounding
per_mac_BL=len(df_mac_BL)/float(n_macsound)
per_mac_DL=len(df_mac_DL)/float(n_macsound)
per_mac_NI=len(df_mac_NI)/float(n_macsound)
per_mac_SI=len(df_mac_SI)/float(n_macsound)

#Percentages regarding to total number MAC sounding
per_macfro_BL=len(df_mac_BL)/float(n_macfronts)
per_macfro_DL=len(df_mac_DL)/float(n_macfronts)
per_macfro_NI=len(df_mac_NI)/float(n_macfronts)
per_macfro_SI=len(df_mac_SI)/float(n_macfronts)


#Percentages regarding to total number YOTC sounding
per_yotc_BL=len(df_yotc_BL)/float(n_yotcsound)
per_yotc_DL=len(df_yotc_DL)/float(n_yotcsound)
per_yotc_NI=len(df_yotc_NI)/float(n_yotcsound)
per_yotc_SI=len(df_yotc_SI)/float(n_yotcsound)

#Percentages regarding to total number YOTC sounding
per_yotcfro_BL=len(df_yotc_BL)/float(n_yotcfronts)
per_yotcfro_DL=len(df_yotc_DL)/float(n_yotcfronts)
per_yotcfro_NI=len(df_yotc_NI)/float(n_yotcfronts)
per_yotcfro_SI=len(df_yotc_SI)/float(n_yotcfronts)

#Percentages regarding to total number MAC-YOTC sounding
per_my_BL=len(df_my_BL)/float(n_mysound)
per_my_DL=len(df_my_DL)/float(n_mysound)
per_my_NI=len(df_my_NI)/float(n_mysound)
per_my_SI=len(df_my_SI)/float(n_mysound)

#Percentages regarding to total number MAC-YOTC sounding
per_myfro_BL=len(df_my_BL)/float(n_myfronts)
per_myfro_DL=len(df_my_DL)/float(n_myfronts)
per_myfro_NI=len(df_my_NI)/float(n_myfronts)
per_myfro_SI=len(df_my_SI)/float(n_myfronts)

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

path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts/'

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
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Line graph differences
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
row = 2
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2, ax3, ax4= axes.flat

#*****************************************************************************\
#No Inversion
#*****************************************************************************\
xmac_NI=np.array(df_mac_NI['Dist Front'])
n_macNI, bins=np.histogram(xmac_NI, bins=bins,normed=1)

xyotc_NI=np.array(df_yotc_NI['Dist Front'])
n_yotcNI, bins=np.histogram(xyotc_NI, bins=bins,normed=1)

xmy_NI=np.array(df_my_NI['Dist Front'])
n_myNI, bins=np.histogram(xmy_NI, bins=bins,normed=1)

#*****************************************************************************\
#RMS

rms_macNI=np.sqrt(((n_macNI - n_yotcNI) ** 2).mean(axis=None))
rms_myNI=np.sqrt(((n_myNI - n_yotcNI) ** 2).mean(axis=None))
print rms_macNI, rms_myNI

#*****************************************************************************\

ax1.plot(bins[:-1],n_macNI, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC')
ax1.plot(bins[:-1],n_yotcNI, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC')
ax1.plot(bins[:-1],n_myNI, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$')


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Relative frequency',fontsize = 12)
ax1.set_title('No inversion', fontsize = 14 ,fontweight='bold')
ax1.grid()

#*****************************************************************************\
#Single Inversion
#*****************************************************************************\
xmac_SI=np.array(df_mac_SI['Dist Front'])
n_macSI, bins=np.histogram(xmac_SI, bins=bins,normed=1)

xyotc_SI=np.array(df_yotc_SI['Dist Front'])
n_yotcSI, bins=np.histogram(xyotc_SI, bins=bins,normed=1)

xmy_SI=np.array(df_my_SI['Dist Front'])
n_mySI, bins=np.histogram(xmy_SI, bins=bins,normed=1)

ax2.plot(bins[:-1],n_macSI, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC')
ax2.plot(bins[:-1],n_yotcSI, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC')
ax2.plot(bins[:-1],n_mySI, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$')


ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_xlim([x1,x2])
ax2.set_ylim([y1,y2])
#ax2.legend(loc='upper left')
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('Relative frequency',fontsize = 12)
ax2.set_title('Single inversion', fontsize = 14 ,fontweight='bold')
ax2.grid()


#*****************************************************************************\
#Decoupled Layer
#*****************************************************************************\
xmac_DL=np.array(df_mac_DL['Dist Front'])
n_macDL, bins=np.histogram(xmac_DL, bins=bins,normed=1)

xyotc_DL=np.array(df_yotc_DL['Dist Front'])
n_yotcDL, bins=np.histogram(xyotc_DL, bins=bins,normed=1)

xmy_DL=np.array(df_my_DL['Dist Front'])
n_myDL, bins=np.histogram(xmy_DL, bins=bins,normed=1)

ax3.plot(bins[:-1],n_macDL, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC')
ax3.plot(bins[:-1],n_yotcDL, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC')
ax3.plot(bins[:-1],n_myDL, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$')


ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_xlim([x1,x2])
ax3.set_ylim([y1,y2])
#ax3.legend(loc='upper left')
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_ylabel('Relative frequency',fontsize = 12)
ax3.set_title('Decoupled Layer', fontsize = 14 ,fontweight='bold')
ax3.grid()



#*****************************************************************************\
#Buffer Layer
#*****************************************************************************\
xmac_BL=np.array(df_mac_BL['Dist Front'])
n_macBL, bins=np.histogram(xmac_BL, bins=bins,normed=1)

xyotc_BL=np.array(df_yotc_BL['Dist Front'])
n_yotcBL, bins=np.histogram(xyotc_BL, bins=bins,normed=1)

xmy_BL=np.array(df_my_BL['Dist Front'])
n_myBL, bins=np.histogram(xmy_BL, bins=bins,normed=1)


ax4.plot(bins[:-1],n_macBL, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC')
ax4.plot(bins[:-1],n_yotcBL, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC')
ax4.plot(bins[:-1],n_myBL, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$')


ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_yticks(by)
ax4.set_xlim([x1,x2])
ax4.set_ylim([y1,y2])
#ax4.legend(loc='upper left')
ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
ax4.set_ylabel('Relative frequency',fontsize = 12)
ax4.set_title('Buffer Layer', fontsize = 14 ,fontweight='bold')
ax4.grid()

fig.savefig(path_data + 'plot_clas.eps', format='eps', dpi=1200)
#plt.show()

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
y1=0
y2=0.08
row = 1
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(16,5))
ax1, ax2= axes.flat
#*****************************************************************************\
# MAC
#*****************************************************************************\
df_macsoundfro=df_macsound[np.isfinite(df_macsound['Dist Front'])]
df_macsoundfro1inv=df_macsoundfro[np.isfinite(df_macsoundfro['1ra Inv'])]
df_macsoundfro2inv=df_macsoundfro[np.isfinite(df_macsoundfro['2da Inv'])]
#*****************************************************************************\
x=np.array(df_macsoundfro1inv['Dist Front'])
y=np.array(df_macsoundfro1inv['1ra Inv'])

bin_count, bin_edges, _ = stats.binned_statistic(x, y, statistic='count', bins=30)

per_count_mac1=bin_count/float(len(df_macsoundfro))
#*****************************************************************************\
x=np.array(df_macsoundfro2inv['Dist Front'])
y=np.array(df_macsoundfro2inv['1ra Inv'])

bin_count, bin_edges, _ = stats.binned_statistic(x, y, statistic='count', bins=30)

per_count_mac2=bin_count/float(len(df_macsoundfro))

#*****************************************************************************\
ax1.plot(bin_edges[:-1],per_count_mac1, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC')

ax2.plot(bin_edges[:-1],per_count_mac2, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC')

#*****************************************************************************\
# YOTC
#*****************************************************************************\
df_yotcsoundfro=df_yotcsound[np.isfinite(df_yotcsound['Dist Front'])]
df_yotcsoundfro1inv=df_yotcsoundfro[np.isfinite(df_yotcsoundfro['1ra Inv'])]
df_yotcsoundfro2inv=df_yotcsoundfro[np.isfinite(df_yotcsoundfro['2da Inv'])]
#*****************************************************************************\
x=np.array(df_yotcsoundfro1inv['Dist Front'])
y=np.array(df_yotcsoundfro1inv['1ra Inv'])

bin_count, bin_edges, _ = stats.binned_statistic(x, y, statistic='count', bins=30)

per_count_yotc1=bin_count/float(len(df_yotcsoundfro))
#*****************************************************************************\
x=np.array(df_yotcsoundfro2inv['Dist Front'])
y=np.array(df_yotcsoundfro2inv['1ra Inv'])

bin_count, bin_edges, _ = stats.binned_statistic(x, y, statistic='count', bins=30)

per_count_yotc2=bin_count/float(len(df_yotcsoundfro))

#*****************************************************************************\
ax1.plot(bin_edges[:-1],per_count_yotc1, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC')

ax2.plot(bin_edges[:-1],per_count_yotc2, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC')

#*****************************************************************************\
# MAC-YOTC
#*****************************************************************************\

df_mysoundfro=df_mysound[np.isfinite(df_mysound['Dist Front'])]
df_mysoundfro1inv=df_mysoundfro[np.isfinite(df_mysoundfro['1ra Inv'])]
df_mysoundfro2inv=df_mysoundfro[np.isfinite(df_mysoundfro['2da Inv'])]
#*****************************************************************************\
x=np.array(df_mysoundfro1inv['Dist Front'])
y=np.array(df_mysoundfro1inv['1ra Inv'])

bin_count, bin_edges, _ = stats.binned_statistic(x, y, statistic='count', bins=30)

per_count_my1=bin_count/float(len(df_mysoundfro))
#*****************************************************************************\
x=np.array(df_mysoundfro2inv['Dist Front'])
y=np.array(df_mysoundfro2inv['1ra Inv'])

bin_count, bin_edges, _ = stats.binned_statistic(x, y, statistic='count', bins=30)

per_count_my2=bin_count/float(len(df_mysoundfro))

#*****************************************************************************\
ax1.plot(bin_edges[:-1],per_count_my1, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$')

ax2.plot(bin_edges[:-1],per_count_my2, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$')



#*****************************************************************************\
#*****************************************************************************\
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper right')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Relative frequency',fontsize = 12)
ax1.set_title('Main Inversion', fontsize = 14 ,fontweight='bold')
ax1.grid()

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_xlim([x1,x2])
ax2.set_ylim([y1,y2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('Relative frequency',fontsize = 12)
ax2.set_title('Secondary Inversion', fontsize = 14 ,fontweight='bold')
ax2.grid()

fig.savefig(path_data + 'plot_inversions.eps', format='eps', dpi=1200)


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
x1=-15
x2=15
dx=1
bx=np.arange(x1,x2+1,3)
bins = np.arange(x1, x2+1, dx)

y1=0
y2=2500
df1=15
df2=30
width = 1
row = 2
column = 2

error_config = {'ecolor': '0.4'}
error_config2 = {'ecolor': '0'}


fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2,ax3,ax4= axes.flat

#*****************************************************************************\
#MAC
#*****************************************************************************

x_mac1=np.array(df_mac_1inv['Dist Front'])
y_mac1=np.array(df_mac_1inv['1ra Inv'])

x_mac2=np.array(df_mac_2inv['Dist Front'])
y_mac2=np.array(df_mac_2inv['2da Inv'])

# 1 Inversion
bmean_mac1, bedg_mac1, binnumber = stats.binned_statistic(x_mac1, y_mac1, statistic='mean', bins=30)
bstd_mac1, _, _ = stats.binned_statistic(x_mac1, y_mac1, statistic=np.std, bins=30)
# 2 Inversion
bmean_mac2, bedg_mac2, binnumber = stats.binned_statistic(x_mac2, y_mac2, statistic='mean', bins=30)
bstd_mac2, _, _ = stats.binned_statistic(x_mac2, y_mac2, statistic=np.std, bins=30)


ax1.errorbar(bedg_mac1[:-1],bmean_mac1, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC$',yerr=bstd_mac1)

ax2.errorbar(bedg_mac2[:-1],bmean_mac2, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC$',yerr=bstd_mac1)


ax3.plot(bedg_mac1[:-1],bmean_mac1, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC')

ax4.plot(bedg_mac2[:-1],bmean_mac2, linewidth=2,marker='o',ms=5,ls='dotted', color='b',label='MAC')


#*****************************************************************************\
#YOTC
#*****************************************************************************

x_yotc1=np.array(df_yotc_1inv['Dist Front'])
y_yotc1=np.array(df_yotc_1inv['1ra Inv'])

x_yotc2=np.array(df_yotc_2inv['Dist Front'])
y_yotc2=np.array(df_yotc_2inv['2da Inv'])

# 1 Inversion
bmean_yotc1, bedg_yotc1, binnumber = stats.binned_statistic(x_yotc1, y_yotc1, statistic='mean', bins=30)
bstd_yotc1, _, _ = stats.binned_statistic(x_yotc1, y_yotc1, statistic=np.std, bins=30)
# 2 Inversion
bmean_yotc2, bedg_yotc2, binnumber = stats.binned_statistic(x_yotc2, y_yotc2, statistic='mean', bins=30)
bstd_yotc2, _, _ = stats.binned_statistic(x_yotc2, y_yotc2, statistic=np.std, bins=30)


ax1.errorbar(bedg_yotc1[:-1],bmean_yotc1, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC',yerr=bstd_yotc1)

ax2.errorbar(bedg_yotc2[:-1],bmean_yotc2, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC',yerr=bstd_yotc1)

ax3.plot(bedg_yotc1[:-1],bmean_yotc1, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC')

ax4.plot(bedg_yotc2[:-1],bmean_yotc2, linewidth=2,marker='o',ms=5,ls='dotted', color='r',label='YOTC')
#*****************************************************************************\
#MAC-YOTC
#*****************************************************************************

x_my1=np.array(df_my_1inv['Dist Front'])
y_my1=np.array(df_my_1inv['1ra Inv'])

x_my2=np.array(df_my_2inv['Dist Front'])
y_my2=np.array(df_my_2inv['2da Inv'])

# 1 Inversion
bmean_my1, bedg_my1, binnumber = stats.binned_statistic(x_my1, y_my1, statistic='mean', bins=30)
bstd_my1, _, _ = stats.binned_statistic(x_my1, y_my1, statistic=np.std, bins=30)
# 2 Inversion
bmean_my2, bedg_my2, binnumber = stats.binned_statistic(x_my2, y_my2, statistic='mean', bins=30)
bstd_my2, _, _ = stats.binned_statistic(x_my2, y_my2, statistic=np.std, bins=30)


ax1.errorbar(bedg_my1[:-1],bmean_my1, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$',yerr=bstd_my1)

ax2.errorbar(bedg_my2[:-1],bmean_my2, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$',yerr=bstd_my1)

ax3.plot(bedg_my1[:-1],bmean_my1, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$')

ax4.plot(bedg_my2[:-1],bmean_my2, linewidth=2,marker='o',ms=5,ls='dotted', color='g',label='MAC$_{AVE}$')

#*****************************************************************************\
#*****************************************************************************\
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Height (mts.)',fontsize = 12)
ax1.set_title('Main Inversion', fontsize = 14 ,fontweight='bold')
ax1.grid()


ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_xlim([x1,x2])
ax2.set_ylim([y1,y2])
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('Height (mts.)',fontsize = 12)
ax2.set_title('Secondary Inversion', fontsize = 14 ,fontweight='bold')
ax2.grid()


ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_xlim([x1,x2])
ax3.set_ylim([y1,y2])
ax3.legend(loc='upper left')
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_ylabel('Height (mts.)',fontsize = 12)
ax3.set_title('Main Inversion', fontsize = 14 ,fontweight='bold')
ax3.grid()


ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_xlim([x1,x2])
ax4.set_ylim([y1,y2])
ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
ax4.set_ylabel('Height (mts.)',fontsize = 12)
ax4.set_title('Secondary Inversion', fontsize = 14 ,fontweight='bold')
ax4.grid()


fig.tight_layout()
fig.savefig(path_data + 'plot_heightinversions.eps', format='eps', dpi=1200)

plt.show()

