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

colora='#c6c9d0'
colorb='#67832F'
colorc='tomato'
colord='#1C86EE'



figsize3=(7.5, 5)
fsize0=12
fsize1=14
fsize2=16


path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts/'

#*****************************************************************************\
x1=-15
x2=15
y1=0
y2=0.7
dx=1
bx=np.arange(x1,x2+1,3)
by=np.arange(y1,y2+0.02,0.1)
bins = np.arange(x1, x2+1, dx)

df1=15
df2=30
width = 0.3
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Clas Seasonal
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# MAC
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
dfmac_DJF=df_macfro[(df_macfro.index.month==12) | (df_macfro.index.month==1) | (df_macfro.index.month==2)]
dfmac_MAM=df_macfro[(df_macfro.index.month==3) | (df_macfro.index.month==4) | (df_macfro.index.month==5)]
dfmac_JJA=df_macfro[(df_macfro.index.month==6) | (df_macfro.index.month==7) | (df_macfro.index.month==8)]
dfmac_SON=df_macfro[(df_macfro.index.month==9) | (df_macfro.index.month==10) | (df_macfro.index.month==11)]

#*****************************************************************************\
row = 2
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2, ax3, ax4= axes.flat


#*****************************************************************************\
#No Inversion
#*****************************************************************************\
#DJF
df_NI1 = dfmac_DJF[dfmac_DJF['Clas']==1]
dfmac_DJF_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xmac_NI=np.array(dfmac_DJF_NI['Dist Front'])
n_macNI_DJF, bins=np.histogram(xmac_NI, bins=bins,normed=1)

#MAM
df_NI1 = dfmac_MAM[dfmac_MAM['Clas']==1]
dfmac_MAM_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xmac_NI=np.array(dfmac_MAM_NI['Dist Front'])
n_macNI_MAM, bins=np.histogram(xmac_NI, bins=bins,normed=1)

#JJA
df_NI1 = dfmac_JJA[dfmac_JJA['Clas']==1]
dfmac_JJA_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xmac_NI=np.array(dfmac_JJA_NI['Dist Front'])
n_macNI_JJA, bins=np.histogram(xmac_NI, bins=bins,normed=1)

#SON
df_NI1 = dfmac_SON[dfmac_SON['Clas']==1]
dfmac_SON_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xmac_NI=np.array(dfmac_SON_NI['Dist Front'])
n_macNI_SON, bins=np.histogram(xmac_NI, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'MAC': bins[:-1],
        'DJF': n_macNI_DJF,
        'MAM': n_macNI_MAM,
        'JJA': n_macNI_JJA,
        'SON': n_macNI_SON}

df= pd.DataFrame(raw_data1, columns = ['MAC', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax1.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax1.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax1.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax1.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left',ncol=2)
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Relative frequency',fontsize = 12)
ax1.set_title('No inversion', fontsize = 14 ,fontweight='bold')
ax1.grid()

#*****************************************************************************\
#Single Inversion
#*****************************************************************************\
#DJF
df_SI1 = dfmac_DJF[dfmac_DJF['Clas']==2]
dfmac_DJF_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xmac_SI=np.array(dfmac_DJF_SI['Dist Front'])
n_macSI_DJF, bins=np.histogram(xmac_SI, bins=bins,normed=1)

#MAM
df_SI1 = dfmac_MAM[dfmac_MAM['Clas']==2]
dfmac_MAM_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xmac_SI=np.array(dfmac_MAM_SI['Dist Front'])
n_macSI_MAM, bins=np.histogram(xmac_SI, bins=bins,normed=1)

#JJA
df_SI1 = dfmac_JJA[dfmac_JJA['Clas']==2]
dfmac_JJA_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xmac_SI=np.array(dfmac_JJA_SI['Dist Front'])
n_macSI_JJA, bins=np.histogram(xmac_SI, bins=bins,normed=1)

#SON
df_SI1 = dfmac_SON[dfmac_SON['Clas']==2]
dfmac_SON_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xmac_SI=np.array(dfmac_SON_SI['Dist Front'])
n_macSI_SON, bins=np.histogram(xmac_SI, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'MAC': bins[:-1],
        'DJF': n_macSI_DJF,
        'MAM': n_macSI_MAM,
        'JJA': n_macSI_JJA,
        'SON': n_macSI_SON}

df= pd.DataFrame(raw_data1, columns = ['MAC', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax2.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax2.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax2.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax2.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
#DJF
df_DL1 = dfmac_DJF[dfmac_DJF['Clas']==3]
dfmac_DJF_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xmac_DL=np.array(dfmac_DJF_DL['Dist Front'])
n_macDL_DJF, bins=np.histogram(xmac_DL, bins=bins,normed=1)

#MAM
df_DL1 = dfmac_MAM[dfmac_MAM['Clas']==3]
dfmac_MAM_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xmac_DL=np.array(dfmac_MAM_DL['Dist Front'])
n_macDL_MAM, bins=np.histogram(xmac_DL, bins=bins,normed=1)

#JJA
df_DL1 = dfmac_JJA[dfmac_JJA['Clas']==3]
dfmac_JJA_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xmac_DL=np.array(dfmac_JJA_DL['Dist Front'])
n_macDL_JJA, bins=np.histogram(xmac_DL, bins=bins,normed=1)

#SON
df_DL1 = dfmac_SON[dfmac_SON['Clas']==3]
dfmac_SON_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xmac_DL=np.array(dfmac_SON_DL['Dist Front'])
n_macDL_SON, bins=np.histogram(xmac_DL, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'MAC': bins[:-1],
        'DJF': n_macDL_DJF,
        'MAM': n_macDL_MAM,
        'JJA': n_macDL_JJA,
        'SON': n_macDL_SON}

df= pd.DataFrame(raw_data1, columns = ['MAC', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax3.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax3.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax3.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax3.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
#DJF
df_BL1 = dfmac_DJF[dfmac_DJF['Clas']==4]
dfmac_DJF_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xmac_BL=np.array(dfmac_DJF_BL['Dist Front'])
n_macBL_DJF, bins=np.histogram(xmac_BL, bins=bins,normed=1)

#MAM
df_BL1 = dfmac_MAM[dfmac_MAM['Clas']==4]
dfmac_MAM_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xmac_BL=np.array(dfmac_MAM_BL['Dist Front'])
n_macBL_MAM, bins=np.histogram(xmac_BL, bins=bins,normed=1)

#JJA
df_BL1 = dfmac_JJA[dfmac_JJA['Clas']==4]
dfmac_JJA_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xmac_BL=np.array(dfmac_JJA_BL['Dist Front'])
n_macBL_JJA, bins=np.histogram(xmac_BL, bins=bins,normed=1)

#SON
df_BL1 = dfmac_SON[dfmac_SON['Clas']==4]
dfmac_SON_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xmac_BL=np.array(dfmac_SON_BL['Dist Front'])
n_macBL_SON, bins=np.histogram(xmac_BL, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'MAC': bins[:-1],
        'DJF': n_macBL_DJF,
        'MAM': n_macBL_MAM,
        'JJA': n_macBL_JJA,
        'SON': n_macBL_SON}

df= pd.DataFrame(raw_data1, columns = ['MAC', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax4.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax4.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax4.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax4.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
plt.suptitle('MAC',fontsize = 14 ,fontweight='bold')
##fig.tight_layout()
fig.savefig(path_data + 'seasonalclas_macfronts.eps', format='eps', dpi=1200)

#*****************************************************************************\
# YOTC
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
dfyotc_DJF=df_yotcfro[(df_yotcfro.index.month==12) | (df_yotcfro.index.month==1) | (df_yotcfro.index.month==2)]
dfyotc_MAM=df_yotcfro[(df_yotcfro.index.month==3) | (df_yotcfro.index.month==4) | (df_yotcfro.index.month==5)]
dfyotc_JJA=df_yotcfro[(df_yotcfro.index.month==6) | (df_yotcfro.index.month==7) | (df_yotcfro.index.month==8)]
dfyotc_SON=df_yotcfro[(df_yotcfro.index.month==9) | (df_yotcfro.index.month==10) | (df_yotcfro.index.month==11)]

#*****************************************************************************\
row = 2
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2, ax3, ax4= axes.flat


#*****************************************************************************\
#No Inversion
#*****************************************************************************\
#DJF
df_NI1 = dfyotc_DJF[dfyotc_DJF['Clas']==1]
dfyotc_DJF_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xyotc_NI=np.array(dfyotc_DJF_NI['Dist Front'])
n_yotcNI_DJF, bins=np.histogram(xyotc_NI, bins=bins,normed=1)

#MAM
df_NI1 = dfyotc_MAM[dfyotc_MAM['Clas']==1]
dfyotc_MAM_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xyotc_NI=np.array(dfyotc_MAM_NI['Dist Front'])
n_yotcNI_MAM, bins=np.histogram(xyotc_NI, bins=bins,normed=1)

#JJA
df_NI1 = dfyotc_JJA[dfyotc_JJA['Clas']==1]
dfyotc_JJA_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xyotc_NI=np.array(dfyotc_JJA_NI['Dist Front'])
n_yotcNI_JJA, bins=np.histogram(xyotc_NI, bins=bins,normed=1)

#SON
df_NI1 = dfyotc_SON[dfyotc_SON['Clas']==1]
dfyotc_SON_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xyotc_NI=np.array(dfyotc_SON_NI['Dist Front'])
n_yotcNI_SON, bins=np.histogram(xyotc_NI, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'yotc': bins[:-1],
        'DJF': n_yotcNI_DJF,
        'MAM': n_yotcNI_MAM,
        'JJA': n_yotcNI_JJA,
        'SON': n_yotcNI_SON}

df= pd.DataFrame(raw_data1, columns = ['yotc', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax1.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax1.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax1.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax1.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left',ncol=2)
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Relative frequency',fontsize = 12)
ax1.set_title('No inversion', fontsize = 14 ,fontweight='bold')
ax1.grid()

#*****************************************************************************\
#Single Inversion
#*****************************************************************************\
#DJF
df_SI1 = dfyotc_DJF[dfyotc_DJF['Clas']==2]
dfyotc_DJF_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xyotc_SI=np.array(dfyotc_DJF_SI['Dist Front'])
n_yotcSI_DJF, bins=np.histogram(xyotc_SI, bins=bins,normed=1)

#MAM
df_SI1 = dfyotc_MAM[dfyotc_MAM['Clas']==2]
dfyotc_MAM_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xyotc_SI=np.array(dfyotc_MAM_SI['Dist Front'])
n_yotcSI_MAM, bins=np.histogram(xyotc_SI, bins=bins,normed=1)

#JJA
df_SI1 = dfyotc_JJA[dfyotc_JJA['Clas']==2]
dfyotc_JJA_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xyotc_SI=np.array(dfyotc_JJA_SI['Dist Front'])
n_yotcSI_JJA, bins=np.histogram(xyotc_SI, bins=bins,normed=1)

#SON
df_SI1 = dfyotc_SON[dfyotc_SON['Clas']==2]
dfyotc_SON_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xyotc_SI=np.array(dfyotc_SON_SI['Dist Front'])
n_yotcSI_SON, bins=np.histogram(xyotc_SI, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'yotc': bins[:-1],
        'DJF': n_yotcSI_DJF,
        'MAM': n_yotcSI_MAM,
        'JJA': n_yotcSI_JJA,
        'SON': n_yotcSI_SON}

df= pd.DataFrame(raw_data1, columns = ['yotc', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax2.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax2.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax2.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax2.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
#DJF
df_DL1 = dfyotc_DJF[dfyotc_DJF['Clas']==3]
dfyotc_DJF_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xyotc_DL=np.array(dfyotc_DJF_DL['Dist Front'])
n_yotcDL_DJF, bins=np.histogram(xyotc_DL, bins=bins,normed=1)

#MAM
df_DL1 = dfyotc_MAM[dfyotc_MAM['Clas']==3]
dfyotc_MAM_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xyotc_DL=np.array(dfyotc_MAM_DL['Dist Front'])
n_yotcDL_MAM, bins=np.histogram(xyotc_DL, bins=bins,normed=1)

#JJA
df_DL1 = dfyotc_JJA[dfyotc_JJA['Clas']==3]
dfyotc_JJA_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xyotc_DL=np.array(dfyotc_JJA_DL['Dist Front'])
n_yotcDL_JJA, bins=np.histogram(xyotc_DL, bins=bins,normed=1)

#SON
df_DL1 = dfyotc_SON[dfyotc_SON['Clas']==3]
dfyotc_SON_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xyotc_DL=np.array(dfyotc_SON_DL['Dist Front'])
n_yotcDL_SON, bins=np.histogram(xyotc_DL, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'yotc': bins[:-1],
        'DJF': n_yotcDL_DJF,
        'MAM': n_yotcDL_MAM,
        'JJA': n_yotcDL_JJA,
        'SON': n_yotcDL_SON}

df= pd.DataFrame(raw_data1, columns = ['yotc', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax3.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax3.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax3.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax3.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
#DJF
df_BL1 = dfyotc_DJF[dfyotc_DJF['Clas']==4]
dfyotc_DJF_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xyotc_BL=np.array(dfyotc_DJF_BL['Dist Front'])
n_yotcBL_DJF, bins=np.histogram(xyotc_BL, bins=bins,normed=1)

#MAM
df_BL1 = dfyotc_MAM[dfyotc_MAM['Clas']==4]
dfyotc_MAM_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xyotc_BL=np.array(dfyotc_MAM_BL['Dist Front'])
n_yotcBL_MAM, bins=np.histogram(xyotc_BL, bins=bins,normed=1)

#JJA
df_BL1 = dfyotc_JJA[dfyotc_JJA['Clas']==4]
dfyotc_JJA_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xyotc_BL=np.array(dfyotc_JJA_BL['Dist Front'])
n_yotcBL_JJA, bins=np.histogram(xyotc_BL, bins=bins,normed=1)

#SON
df_BL1 = dfyotc_SON[dfyotc_SON['Clas']==4]
dfyotc_SON_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xyotc_BL=np.array(dfyotc_SON_BL['Dist Front'])
n_yotcBL_SON, bins=np.histogram(xyotc_BL, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'yotc': bins[:-1],
        'DJF': n_yotcBL_DJF,
        'MAM': n_yotcBL_MAM,
        'JJA': n_yotcBL_JJA,
        'SON': n_yotcBL_SON}

df= pd.DataFrame(raw_data1, columns = ['yotc', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax4.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax4.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax4.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax4.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
plt.suptitle('YOTC',fontsize = 14 ,fontweight='bold')
#fig.tight_layout()
fig.savefig(path_data + 'seasonalclas_yotcfronts.eps', format='eps', dpi=1200)
#*****************************************************************************\
# MAC-YOTC
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
dfmy_DJF=df_myfro[(df_myfro.index.month==12) | (df_myfro.index.month==1) | (df_myfro.index.month==2)]
dfmy_MAM=df_myfro[(df_myfro.index.month==3) | (df_myfro.index.month==4) | (df_myfro.index.month==5)]
dfmy_JJA=df_myfro[(df_myfro.index.month==6) | (df_myfro.index.month==7) | (df_myfro.index.month==8)]
dfmy_SON=df_myfro[(df_myfro.index.month==9) | (df_myfro.index.month==10) | (df_myfro.index.month==11)]

#*****************************************************************************\
row = 2
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2, ax3, ax4= axes.flat


#*****************************************************************************\
#No Inversion
#*****************************************************************************\
#DJF
df_NI1 = dfmy_DJF[dfmy_DJF['Clas']==1]
dfmy_DJF_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xmy_NI=np.array(dfmy_DJF_NI['Dist Front'])
n_myNI_DJF, bins=np.histogram(xmy_NI, bins=bins,normed=1)

#MAM
df_NI1 = dfmy_MAM[dfmy_MAM['Clas']==1]
dfmy_MAM_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xmy_NI=np.array(dfmy_MAM_NI['Dist Front'])
n_myNI_MAM, bins=np.histogram(xmy_NI, bins=bins,normed=1)

#JJA
df_NI1 = dfmy_JJA[dfmy_JJA['Clas']==1]
dfmy_JJA_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xmy_NI=np.array(dfmy_JJA_NI['Dist Front'])
n_myNI_JJA, bins=np.histogram(xmy_NI, bins=bins,normed=1)

#SON
df_NI1 = dfmy_SON[dfmy_SON['Clas']==1]
dfmy_SON_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

xmy_NI=np.array(dfmy_SON_NI['Dist Front'])
n_myNI_SON, bins=np.histogram(xmy_NI, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'my': bins[:-1],
        'DJF': n_myNI_DJF,
        'MAM': n_myNI_MAM,
        'JJA': n_myNI_JJA,
        'SON': n_myNI_SON}

df= pd.DataFrame(raw_data1, columns = ['my', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax1.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax1.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax1.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax1.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])


ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left',ncol=2)
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Relative frequency',fontsize = 12)
ax1.set_title('No inversion', fontsize = 14 ,fontweight='bold')
ax1.grid()

#*****************************************************************************\
#Single Inversion
#*****************************************************************************\
#DJF
df_SI1 = dfmy_DJF[dfmy_DJF['Clas']==2]
dfmy_DJF_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xmy_SI=np.array(dfmy_DJF_SI['Dist Front'])
n_mySI_DJF, bins=np.histogram(xmy_SI, bins=bins,normed=1)

#MAM
df_SI1 = dfmy_MAM[dfmy_MAM['Clas']==2]
dfmy_MAM_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xmy_SI=np.array(dfmy_MAM_SI['Dist Front'])
n_mySI_MAM, bins=np.histogram(xmy_SI, bins=bins,normed=1)

#JJA
df_SI1 = dfmy_JJA[dfmy_JJA['Clas']==2]
dfmy_JJA_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xmy_SI=np.array(dfmy_JJA_SI['Dist Front'])
n_mySI_JJA, bins=np.histogram(xmy_SI, bins=bins,normed=1)

#SON
df_SI1 = dfmy_SON[dfmy_SON['Clas']==2]
dfmy_SON_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

xmy_SI=np.array(dfmy_SON_SI['Dist Front'])
n_mySI_SON, bins=np.histogram(xmy_SI, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'my': bins[:-1],
        'DJF': n_mySI_DJF,
        'MAM': n_mySI_MAM,
        'JJA': n_mySI_JJA,
        'SON': n_mySI_SON}

df= pd.DataFrame(raw_data1, columns = ['my', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax2.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax2.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax2.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax2.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
#DJF
df_DL1 = dfmy_DJF[dfmy_DJF['Clas']==3]
dfmy_DJF_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xmy_DL=np.array(dfmy_DJF_DL['Dist Front'])
n_myDL_DJF, bins=np.histogram(xmy_DL, bins=bins,normed=1)

#MAM
df_DL1 = dfmy_MAM[dfmy_MAM['Clas']==3]
dfmy_MAM_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xmy_DL=np.array(dfmy_MAM_DL['Dist Front'])
n_myDL_MAM, bins=np.histogram(xmy_DL, bins=bins,normed=1)

#JJA
df_DL1 = dfmy_JJA[dfmy_JJA['Clas']==3]
dfmy_JJA_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xmy_DL=np.array(dfmy_JJA_DL['Dist Front'])
n_myDL_JJA, bins=np.histogram(xmy_DL, bins=bins,normed=1)

#SON
df_DL1 = dfmy_SON[dfmy_SON['Clas']==3]
dfmy_SON_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

xmy_DL=np.array(dfmy_SON_DL['Dist Front'])
n_myDL_SON, bins=np.histogram(xmy_DL, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'my': bins[:-1],
        'DJF': n_myDL_DJF,
        'MAM': n_myDL_MAM,
        'JJA': n_myDL_JJA,
        'SON': n_myDL_SON}

df= pd.DataFrame(raw_data1, columns = ['my', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax3.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax3.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax3.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax3.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
#DJF
df_BL1 = dfmy_DJF[dfmy_DJF['Clas']==4]
dfmy_DJF_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xmy_BL=np.array(dfmy_DJF_BL['Dist Front'])
n_myBL_DJF, bins=np.histogram(xmy_BL, bins=bins,normed=1)

#MAM
df_BL1 = dfmy_MAM[dfmy_MAM['Clas']==4]
dfmy_MAM_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xmy_BL=np.array(dfmy_MAM_BL['Dist Front'])
n_myBL_MAM, bins=np.histogram(xmy_BL, bins=bins,normed=1)

#JJA
df_BL1 = dfmy_JJA[dfmy_JJA['Clas']==4]
dfmy_JJA_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xmy_BL=np.array(dfmy_JJA_BL['Dist Front'])
n_myBL_JJA, bins=np.histogram(xmy_BL, bins=bins,normed=1)

#SON
df_BL1 = dfmy_SON[dfmy_SON['Clas']==4]
dfmy_SON_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

xmy_BL=np.array(dfmy_SON_BL['Dist Front'])
n_myBL_SON, bins=np.histogram(xmy_BL, bins=bins,normed=1)
#*****************************************************************************\

raw_data1 = {'my': bins[:-1],
        'DJF': n_myBL_DJF,
        'MAM': n_myBL_MAM,
        'JJA': n_myBL_JJA,
        'SON': n_myBL_SON}

df= pd.DataFrame(raw_data1, columns = ['my', 'DJF', 'MAM', 'JJA','SON'])

bar_width=1
bar_l = [i+1 for i in range(len(df['DJF']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

#*****************************************************************************\
ax4.bar(bins[:-1], df['DJF'], bar_width, alpha=0.5, color=colora,label='DJF')
ax4.bar(bins[:-1],df['MAM'], bar_width, alpha=0.5, color=colorb,label='MAM',bottom=df['DJF'])
ax4.bar(bins[:-1],df['JJA'], bar_width, alpha=0.5, color=colorc,label='JJA',bottom=[i+j for i,j in zip(df['DJF'],df['MAM'])])

ax4.bar(bins[:-1],df['SON'], bar_width, alpha=0.5, color=colord,label='SON',bottom=[i+j+z for i,j,z in zip(df['DJF'],df['MAM'],df['JJA'])])

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
plt.suptitle('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
#fig.tight_layout()
fig.savefig(path_data + 'seasonalclas_myfronts.eps', format='eps', dpi=1200)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Clas Daily
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# MAC
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# dfmac_zero=df_macfro[(df_macfro.index.hour==00)]
# dfmac_doce=df_macfro[(df_macfro.index.hour==12)]

# #*****************************************************************************\
# row = 2
# column = 2
# fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
# ax1, ax2, ax3, ax4= axes.flat
# y2=0.4
# by=np.arange(y1,y2+0.1,0.05)
# #*****************************************************************************\
# #No Inversion
# #*****************************************************************************\
# #Zero
# df_NI1 = dfmac_zero[dfmac_zero['Clas']==1]
# dfmac_zero_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

# xmac_NI=np.array(dfmac_zero_NI['Dist Front'])
# n_macNI_zero, bins=np.histogram(xmac_NI, bins=bins,normed=1)

# #Twelwe
# df_NI1 = dfmac_doce[dfmac_doce['Clas']==1]
# dfmac_doce_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

# xmac_NI=np.array(dfmac_doce_NI['Dist Front'])
# n_macNI_doce, bins=np.histogram(xmac_NI, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'MAC': bins[:-1],
#         '00-UTC': n_macNI_zero,
#         '12-UTC': n_macNI_doce}

# df= pd.DataFrame(raw_data1, columns = ['MAC', '00-UTC','12-UTC'])

# bar_width=1

# ax1.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax1.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax1.set_xticks(bx)
# ax1.set_yticks(by)
# ax1.set_xlim([x1,x2])
# ax1.set_ylim([y1,y2])
# ax1.legend(loc='upper left',ncol=2)
# ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax1.set_ylabel('Relative frequency',fontsize = 12)
# ax1.set_title('No inversion', fontsize = 14 ,fontweight='bold')
# ax1.grid()
# #*****************************************************************************\
# #Single Inversion
# #*****************************************************************************\
# #Zero
# df_SI1 = dfmac_zero[dfmac_zero['Clas']==2]
# dfmac_zero_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

# xmac_SI=np.array(dfmac_zero_SI['Dist Front'])
# n_macSI_zero, bins=np.histogram(xmac_SI, bins=bins,normed=1)

# #Twelwe
# df_SI1 = dfmac_doce[dfmac_doce['Clas']==2]
# dfmac_doce_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

# xmac_SI=np.array(dfmac_doce_SI['Dist Front'])
# n_macSI_doce, bins=np.histogram(xmac_SI, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'MAC': bins[:-1],
#         '00-UTC': n_macSI_zero,
#         '12-UTC': n_macSI_doce}

# df= pd.DataFrame(raw_data1, columns = ['MAC', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax2.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax2.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.set_xticks(bx)
# ax2.set_yticks(by)
# ax2.set_xlim([x1,x2])
# ax2.set_ylim([y1,y2])
# ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax2.set_ylabel('Relative frequency',fontsize = 12)
# ax2.set_title('Single inversion', fontsize = 14 ,fontweight='bold')
# ax2.grid()
# #*****************************************************************************\
# #Decoupled Layer
# #*****************************************************************************\
# #Zero
# df_DL1 = dfmac_zero[dfmac_zero['Clas']==3]
# dfmac_zero_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

# xmac_DL=np.array(dfmac_zero_DL['Dist Front'])
# n_macDL_zero, bins=np.histogram(xmac_DL, bins=bins,normed=1)

# #Twelwe
# df_DL1 = dfmac_doce[dfmac_doce['Clas']==3]
# dfmac_doce_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

# xmac_DL=np.array(dfmac_doce_DL['Dist Front'])
# n_macDL_doce, bins=np.histogram(xmac_DL, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'MAC': bins[:-1],
#         '00-UTC': n_macDL_zero,
#         '12-UTC': n_macDL_doce}

# df= pd.DataFrame(raw_data1, columns = ['MAC', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax3.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax3.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax3.tick_params(axis='both', which='major', labelsize=14)
# ax3.set_xticks(bx)
# ax3.set_yticks(by)
# ax3.set_xlim([x1,x2])
# ax3.set_ylim([y1,y2])
# ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax3.set_ylabel('Relative frequency',fontsize = 12)
# ax3.set_title('Decoupled Layer', fontsize = 14 ,fontweight='bold')
# ax3.grid()

# #*****************************************************************************\
# #Buffer Layer
# #*****************************************************************************\
# #Zero
# df_BL1 = dfmac_zero[dfmac_zero['Clas']==4]
# dfmac_zero_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

# xmac_BL=np.array(dfmac_zero_BL['Dist Front'])
# n_macBL_zero, bins=np.histogram(xmac_BL, bins=bins,normed=1)

# #Twelwe
# df_BL1 = dfmac_doce[dfmac_doce['Clas']==4]
# dfmac_doce_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

# xmac_BL=np.array(dfmac_doce_BL['Dist Front'])
# n_macBL_doce, bins=np.histogram(xmac_BL, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'MAC': bins[:-1],
#         '00-UTC': n_macBL_zero,
#         '12-UTC': n_macBL_doce}

# df= pd.DataFrame(raw_data1, columns = ['MAC', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax4.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax4.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax4.tick_params(axis='both', which='major', labelsize=14)
# ax4.set_xticks(bx)
# ax4.set_yticks(by)
# ax4.set_xlim([x1,x2])
# ax4.set_ylim([y1,y2])
# ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax4.set_ylabel('Relative frequency',fontsize = 12)
# ax4.set_title('Buffer Layer', fontsize = 14 ,fontweight='bold')
# ax4.grid()


# plt.suptitle('MAC',fontsize = 14 ,fontweight='bold')
# #fig.tight_layout()

# fig.savefig(path_data + 'dailyclas_macfronts.eps', format='eps', dpi=1200)
# #*****************************************************************************\
# # YOTC
# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\
# dfyotc_zero=df_yotcfro[(df_yotcfro.index.hour==00)]
# dfyotc_doce=df_yotcfro[(df_yotcfro.index.hour==12)]

# #*****************************************************************************\
# row = 2
# column = 2
# fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
# ax1, ax2, ax3, ax4= axes.flat


# #*****************************************************************************\
# #No Inversion
# #*****************************************************************************\
# #Zero
# df_NI1 = dfyotc_zero[dfyotc_zero['Clas']==1]
# dfyotc_zero_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

# xyotc_NI=np.array(dfyotc_zero_NI['Dist Front'])
# n_yotcNI_zero, bins=np.histogram(xyotc_NI, bins=bins,normed=1)

# #Twelwe
# df_NI1 = dfyotc_doce[dfyotc_doce['Clas']==1]
# dfyotc_doce_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

# xyotc_NI=np.array(dfyotc_doce_NI['Dist Front'])
# n_yotcNI_doce, bins=np.histogram(xyotc_NI, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'yotc': bins[:-1],
#         '00-UTC': n_yotcNI_zero,
#         '12-UTC': n_yotcNI_doce}

# df= pd.DataFrame(raw_data1, columns = ['yotc', '00-UTC','12-UTC'])

# bar_width=1

# ax1.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax1.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax1.set_xticks(bx)
# ax1.set_yticks(by)
# ax1.set_xlim([x1,x2])
# ax1.set_ylim([y1,y2])
# ax1.legend(loc='upper left',ncol=2)
# ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax1.set_ylabel('Relative frequency',fontsize = 12)
# ax1.set_title('No inversion', fontsize = 14 ,fontweight='bold')
# ax1.grid()
# #*****************************************************************************\
# #Single Inversion
# #*****************************************************************************\
# #Zero
# df_SI1 = dfyotc_zero[dfyotc_zero['Clas']==2]
# dfyotc_zero_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

# xyotc_SI=np.array(dfyotc_zero_SI['Dist Front'])
# n_yotcSI_zero, bins=np.histogram(xyotc_SI, bins=bins,normed=1)

# #Twelwe
# df_SI1 = dfyotc_doce[dfyotc_doce['Clas']==2]
# dfyotc_doce_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

# xyotc_SI=np.array(dfyotc_doce_SI['Dist Front'])
# n_yotcSI_doce, bins=np.histogram(xyotc_SI, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'yotc': bins[:-1],
#         '00-UTC': n_yotcSI_zero,
#         '12-UTC': n_yotcSI_doce}

# df= pd.DataFrame(raw_data1, columns = ['yotc', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax2.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax2.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.set_xticks(bx)
# ax2.set_yticks(by)
# ax2.set_xlim([x1,x2])
# ax2.set_ylim([y1,y2])
# ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax2.set_ylabel('Relative frequency',fontsize = 12)
# ax2.set_title('Single inversion', fontsize = 14 ,fontweight='bold')
# ax2.grid()
# #*****************************************************************************\
# #Decoupled Layer
# #*****************************************************************************\
# #Zero
# df_DL1 = dfyotc_zero[dfyotc_zero['Clas']==3]
# dfyotc_zero_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

# xyotc_DL=np.array(dfyotc_zero_DL['Dist Front'])
# n_yotcDL_zero, bins=np.histogram(xyotc_DL, bins=bins,normed=1)

# #Twelwe
# df_DL1 = dfyotc_doce[dfyotc_doce['Clas']==3]
# dfyotc_doce_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

# xyotc_DL=np.array(dfyotc_doce_DL['Dist Front'])
# n_yotcDL_doce, bins=np.histogram(xyotc_DL, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'yotc': bins[:-1],
#         '00-UTC': n_yotcDL_zero,
#         '12-UTC': n_yotcDL_doce}

# df= pd.DataFrame(raw_data1, columns = ['yotc', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax3.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax3.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax3.tick_params(axis='both', which='major', labelsize=14)
# ax3.set_xticks(bx)
# ax3.set_yticks(by)
# ax3.set_xlim([x1,x2])
# ax3.set_ylim([y1,y2])
# ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax3.set_ylabel('Relative frequency',fontsize = 12)
# ax3.set_title('Decoupled Layer', fontsize = 14 ,fontweight='bold')
# ax3.grid()

# #*****************************************************************************\
# #Buffer Layer
# #*****************************************************************************\
# #Zero
# df_BL1 = dfyotc_zero[dfyotc_zero['Clas']==4]
# dfyotc_zero_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

# xyotc_BL=np.array(dfyotc_zero_BL['Dist Front'])
# n_yotcBL_zero, bins=np.histogram(xyotc_BL, bins=bins,normed=1)

# #Twelwe
# df_BL1 = dfyotc_doce[dfyotc_doce['Clas']==4]
# dfyotc_doce_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

# xyotc_BL=np.array(dfyotc_doce_BL['Dist Front'])
# n_yotcBL_doce, bins=np.histogram(xyotc_BL, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'yotc': bins[:-1],
#         '00-UTC': n_yotcBL_zero,
#         '12-UTC': n_yotcBL_doce}

# df= pd.DataFrame(raw_data1, columns = ['yotc', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax4.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax4.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax4.tick_params(axis='both', which='major', labelsize=14)
# ax4.set_xticks(bx)
# ax4.set_yticks(by)
# ax4.set_xlim([x1,x2])
# ax4.set_ylim([y1,y2])
# ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax4.set_ylabel('Relative frequency',fontsize = 12)
# ax4.set_title('Buffer Layer', fontsize = 14 ,fontweight='bold')
# ax4.grid()


# plt.suptitle('YOTC',fontsize = 14 ,fontweight='bold')
# #fig.tight_layout()

# fig.savefig(path_data + 'dailyclas_yotcfronts.eps', format='eps', dpi=1200)
# #*****************************************************************************\
# # MAC-YOTC
# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\
# dfmy_zero=df_myfro[(df_myfro.index.hour==00)]
# dfmy_doce=df_myfro[(df_myfro.index.hour==12)]

# #*****************************************************************************\
# row = 2
# column = 2
# fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
# ax1, ax2, ax3, ax4= axes.flat


# #*****************************************************************************\
# #No Inversion
# #*****************************************************************************\
# #Zero
# df_NI1 = dfmy_zero[dfmy_zero['Clas']==1]
# dfmy_zero_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

# xmy_NI=np.array(dfmy_zero_NI['Dist Front'])
# n_myNI_zero, bins=np.histogram(xmy_NI, bins=bins,normed=1)

# #Twelwe
# df_NI1 = dfmy_doce[dfmy_doce['Clas']==1]
# dfmy_doce_NI = df_NI1[np.isfinite(df_NI1['Dist Front'])]

# xmy_NI=np.array(dfmy_doce_NI['Dist Front'])
# n_myNI_doce, bins=np.histogram(xmy_NI, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'my': bins[:-1],
#         '00-UTC': n_myNI_zero,
#         '12-UTC': n_myNI_doce}

# df= pd.DataFrame(raw_data1, columns = ['my', '00-UTC','12-UTC'])

# bar_width=1

# ax1.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax1.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax1.set_xticks(bx)
# ax1.set_yticks(by)
# ax1.set_xlim([x1,x2])
# ax1.set_ylim([y1,y2])
# ax1.legend(loc='upper left',ncol=2)
# ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax1.set_ylabel('Relative frequency',fontsize = 12)
# ax1.set_title('No inversion', fontsize = 14 ,fontweight='bold')
# ax1.grid()
# #*****************************************************************************\
# #Single Inversion
# #*****************************************************************************\
# #Zero
# df_SI1 = dfmy_zero[dfmy_zero['Clas']==2]
# dfmy_zero_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

# xmy_SI=np.array(dfmy_zero_SI['Dist Front'])
# n_mySI_zero, bins=np.histogram(xmy_SI, bins=bins,normed=1)

# #Twelwe
# df_SI1 = dfmy_doce[dfmy_doce['Clas']==2]
# dfmy_doce_SI = df_SI1[np.isfinite(df_SI1['Dist Front'])]

# xmy_SI=np.array(dfmy_doce_SI['Dist Front'])
# n_mySI_doce, bins=np.histogram(xmy_SI, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'my': bins[:-1],
#         '00-UTC': n_mySI_zero,
#         '12-UTC': n_mySI_doce}

# df= pd.DataFrame(raw_data1, columns = ['my', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax2.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax2.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.set_xticks(bx)
# ax2.set_yticks(by)
# ax2.set_xlim([x1,x2])
# ax2.set_ylim([y1,y2])
# ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax2.set_ylabel('Relative frequency',fontsize = 12)
# ax2.set_title('Single inversion', fontsize = 14 ,fontweight='bold')
# ax2.grid()
# #*****************************************************************************\
# #Decoupled Layer
# #*****************************************************************************\
# #Zero
# df_DL1 = dfmy_zero[dfmy_zero['Clas']==3]
# dfmy_zero_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

# xmy_DL=np.array(dfmy_zero_DL['Dist Front'])
# n_myDL_zero, bins=np.histogram(xmy_DL, bins=bins,normed=1)

# #Twelwe
# df_DL1 = dfmy_doce[dfmy_doce['Clas']==3]
# dfmy_doce_DL = df_DL1[np.isfinite(df_DL1['Dist Front'])]

# xmy_DL=np.array(dfmy_doce_DL['Dist Front'])
# n_myDL_doce, bins=np.histogram(xmy_DL, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'my': bins[:-1],
#         '00-UTC': n_myDL_zero,
#         '12-UTC': n_myDL_doce}

# df= pd.DataFrame(raw_data1, columns = ['my', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax3.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax3.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax3.tick_params(axis='both', which='major', labelsize=14)
# ax3.set_xticks(bx)
# ax3.set_yticks(by)
# ax3.set_xlim([x1,x2])
# ax3.set_ylim([y1,y2])
# ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax3.set_ylabel('Relative frequency',fontsize = 12)
# ax3.set_title('Decoupled Layer', fontsize = 14 ,fontweight='bold')
# ax3.grid()

# #*****************************************************************************\
# #Buffer Layer
# #*****************************************************************************\
# #Zero
# df_BL1 = dfmy_zero[dfmy_zero['Clas']==4]
# dfmy_zero_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

# xmy_BL=np.array(dfmy_zero_BL['Dist Front'])
# n_myBL_zero, bins=np.histogram(xmy_BL, bins=bins,normed=1)

# #Twelwe
# df_BL1 = dfmy_doce[dfmy_doce['Clas']==4]
# dfmy_doce_BL = df_BL1[np.isfinite(df_BL1['Dist Front'])]

# xmy_BL=np.array(dfmy_doce_BL['Dist Front'])
# n_myBL_doce, bins=np.histogram(xmy_BL, bins=bins,normed=1)

# #*****************************************************************************\

# raw_data1 = {'my': bins[:-1],
#         '00-UTC': n_myBL_zero,
#         '12-UTC': n_myBL_doce}

# df= pd.DataFrame(raw_data1, columns = ['my', '00-UTC','12-UTC'])

# bar_width=1
# bar_l = [i+1 for i in range(len(df['00-UTC']))]

# ax4.bar(bins[:-1], df['00-UTC'], bar_width, alpha=0.5, color=colorb,label='00-UTC')
# ax4.bar(bins[:-1],df['12-UTC'], bar_width, alpha=0.5, color=colorc,label='12-UTC',bottom=df['00-UTC'])


# ax4.tick_params(axis='both', which='major', labelsize=14)
# ax4.set_xticks(bx)
# ax4.set_yticks(by)
# ax4.set_xlim([x1,x2])
# ax4.set_ylim([y1,y2])
# ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax4.set_ylabel('Relative frequency',fontsize = 12)
# ax4.set_title('Buffer Layer', fontsize = 14 ,fontweight='bold')
# ax4.grid()


# plt.suptitle('MAC$_{AVE}$',fontsize = 14 ,fontweight='bold')
# #fig.tight_layout()

# fig.savefig(path_data + 'dailyclas_myfronts.eps', format='eps', dpi=1200)
plt.show()


