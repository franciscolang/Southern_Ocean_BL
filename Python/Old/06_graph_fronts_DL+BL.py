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

df_DL1 = df_macfro[(df_macfro['Clas']==3) | (df_macfro['Clas']==4)]
#df_DL1 = df_macfro[(df_macfro['Clas']==3)]
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

df_DL1 = df_macfro[(df_yotcfro['Clas']==3) | (df_yotcfro['Clas']==4)]
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


df_DL1 = df_myfro[(df_myfro['Clas']==3) | (df_myfro['Clas']==4)]
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

print per_macfro,per_yotcfro, per_myfro

#Percentages regarding to total number MAC sounding
#per_mac_BL=len(df_mac_BL)/float(n_macsound)
per_mac_DL=len(df_mac_DL)/float(n_macsound)
per_mac_NI=len(df_mac_NI)/float(n_macsound)
per_mac_SI=len(df_mac_SI)/float(n_macsound)

#Percentages regarding to total number MAC sounding
#per_macfro_BL=len(df_mac_BL)/float(n_macfronts)
per_macfro_DL=len(df_mac_DL)/float(n_macfronts)
per_macfro_NI=len(df_mac_NI)/float(n_macfronts)
per_macfro_SI=len(df_mac_SI)/float(n_macfronts)


#Percentages regarding to total number YOTC sounding
#per_yotc_BL=len(df_yotc_BL)/float(n_yotcsound)
per_yotc_DL=len(df_yotc_DL)/float(n_yotcsound)
per_yotc_NI=len(df_yotc_NI)/float(n_yotcsound)
per_yotc_SI=len(df_yotc_SI)/float(n_yotcsound)

#Percentages regarding to total number YOTC sounding
#per_yotcfro_BL=len(df_yotc_BL)/float(n_yotcfronts)
per_yotcfro_DL=len(df_yotc_DL)/float(n_yotcfronts)
per_yotcfro_NI=len(df_yotc_NI)/float(n_yotcfronts)
per_yotcfro_SI=len(df_yotc_SI)/float(n_yotcfronts)

#Percentages regarding to total number MAC-YOTC sounding
#per_my_BL=len(df_my_BL)/float(n_mysound)
per_my_DL=len(df_my_DL)/float(n_mysound)
per_my_NI=len(df_my_NI)/float(n_mysound)
per_my_SI=len(df_my_SI)/float(n_mysound)

#Percentages regarding to total number MAC-YOTC sounding
#per_myfro_BL=len(df_my_BL)/float(n_myfronts)
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

path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts/DL_BL/'

#*****************************************************************************\
x1=-15
x2=15
y1=0
y2=0.16
dx=1
bx=np.arange(x1,x2+1,3)
by=np.arange(y1,y2+0.02,0.02)
bins = np.arange(x1, x2+1, dx)

df1=15
df2=30
width = 1

#*****************************************************************************\
#*****************************************************************************\
#Histogram by type MAC
#*****************************************************************************\
#*****************************************************************************\

row = 2
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2, ax3, ax4= axes.flat

#*****************************************************************************\
x=np.array(df_mac_NI['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax1.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='prefront')
ax1.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='postfront')

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Relative frequency',fontsize = 12)
ax1.set_title('No inversion ('+ str(np.around(per_macfro_NI*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax1.grid()


#*****************************************************************************\
x=np.array(df_mac_SI['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax2.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
ax2.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_xlim([x1,x2])
ax2.set_ylim([y1,y2])
#ax2.legend(loc='upper left')
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('Relative frequency',fontsize = 12)
ax2.set_title('Single inversion ('+ str(np.around(per_macfro_SI*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax2.grid()



#*****************************************************************************\

x=np.array(df_mac_DL['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax3.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
ax3.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_xlim([x1,x2])
ax3.set_ylim([y1,y2])
#ax3.legend(loc='upper left')
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_ylabel('Relative frequency',fontsize = 12)
ax3.set_title('Decoupled layer ('+ str(np.around(per_macfro_DL*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax3.grid()


#*****************************************************************************\
# x=np.array(df_mac_BL['Dist Front'])
# n, bins=np.histogram(x, bins=bins,normed=1)
# pos = list(range(len(n[0:df1])))

# ax4.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
# ax4.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

# ax4.tick_params(axis='both', which='major', labelsize=14)
# ax4.set_xticks(bx)
# ax4.set_yticks(by)
# ax4.set_xlim([x1,x2])
# ax4.set_ylim([y1,y2])
# #ax4.legend(loc='upper left')
# ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax4.set_ylabel('Relative frequency',fontsize = 12)
# ax4.set_title('Buffer layer ('+ str(np.around(per_macfro_BL*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
# ax4.grid()

plt.suptitle('MAC ('+ str(np.around(per_macfro,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
fig.savefig(path_data + 'histclas_macfronts.eps', format='eps', dpi=1200)

#*****************************************************************************\
#*****************************************************************************\
#Histogram by type YOTC
#*****************************************************************************\
#*****************************************************************************\
row = 2
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2, ax3, ax4= axes.flat


#*****************************************************************************\
x=np.array(df_yotc_NI['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax1.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='prefront')
ax1.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='postfront')

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Relative frequency',fontsize = 12)
ax1.set_title('No inversion ('+ str(np.around(per_yotcfro_NI*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax1.grid()


#*****************************************************************************\
x=np.array(df_yotc_SI['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax2.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
ax2.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_xlim([x1,x2])
ax2.set_ylim([y1,y2])
#ax2.legend(loc='upper left')
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('Relative frequency',fontsize = 12)
ax2.set_title('Single inversion ('+ str(np.around(per_yotcfro_SI*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax2.grid()



#*****************************************************************************\

x=np.array(df_yotc_DL['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax3.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
ax3.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_xlim([x1,x2])
ax3.set_ylim([y1,y2])
#ax3.legend(loc='upper left')
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_ylabel('Relative frequency',fontsize = 12)
ax3.set_title('Decoupled layer ('+ str(np.around(per_yotcfro_DL*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax3.grid()


#*****************************************************************************\
# x=np.array(df_yotc_BL['Dist Front'])
# n, bins=np.histogram(x, bins=bins,normed=1)
# pos = list(range(len(n[0:df1])))

# ax4.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
# ax4.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

# ax4.tick_params(axis='both', which='major', labelsize=14)
# ax4.set_xticks(bx)
# ax4.set_yticks(by)
# ax4.set_xlim([x1,x2])
# ax4.set_ylim([y1,y2])
# #ax4.legend(loc='upper left')
# ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax4.set_ylabel('Relative frequency',fontsize = 12)
# ax4.set_title('Buffer layer ('+ str(np.around(per_yotcfro_BL*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
# ax4.grid()

plt.suptitle('YOTC ('+ str(np.around(per_yotcfro,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')

fig.savefig(path_data + 'histclas_yotcfronts.eps', format='eps', dpi=1200)
#*****************************************************************************\
#*****************************************************************************\
#Histogram by type MAC-YOTC
#*****************************************************************************\
#*****************************************************************************\
row = 2
column = 2
fig, axes = plt.subplots(row, column, facecolor='w', figsize=(18,14))
ax1, ax2, ax3, ax4= axes.flat


#*****************************************************************************\
x=np.array(df_my_NI['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax1.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='prefront')
ax1.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='postfront')

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_xlim([x1,x2])
ax1.set_ylim([y1,y2])
ax1.legend(loc='upper left')
ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Relative frequency',fontsize = 12)
ax1.set_title('No inversion ('+ str(np.around(per_myfro_NI*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax1.grid()


#*****************************************************************************\
x=np.array(df_my_SI['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax2.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
ax2.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_xlim([x1,x2])
ax2.set_ylim([y1,y2])
#ax2.legend(loc='upper left')
ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
ax2.set_ylabel('Relative frequency',fontsize = 12)
ax2.set_title('Single inversion ('+ str(np.around(per_myfro_SI*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax2.grid()



#*****************************************************************************\

x=np.array(df_my_DL['Dist Front'])
n, bins=np.histogram(x, bins=bins,normed=1)
pos = list(range(len(n[0:df1])))

ax3.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
ax3.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_xlim([x1,x2])
ax3.set_ylim([y1,y2])
#ax3.legend(loc='upper left')
ax3.set_xlabel('Distance from front (deg)',fontsize = 12)
ax3.set_ylabel('Relative frequency',fontsize = 12)
ax3.set_title('Decoupled layer ('+ str(np.around(per_myfro_DL*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
ax3.grid()


#*****************************************************************************\
# x=np.array(df_my_BL['Dist Front'])
# n, bins=np.histogram(x, bins=bins,normed=1)
# pos = list(range(len(n[0:df1])))

# ax4.bar(bins[0:df1],n[0:df1], width,alpha=0.5, color=color1, label='before')
# ax4.bar(pos,n[df1:df2], width,alpha=0.5, color=color2, label='after')

# ax4.tick_params(axis='both', which='major', labelsize=14)
# ax4.set_xticks(bx)
# ax4.set_yticks(by)
# ax4.set_xlim([x1,x2])
# ax4.set_ylim([y1,y2])
# #ax4.legend(loc='upper left')
# ax4.set_xlabel('Distance from front (deg)',fontsize = 12)
# ax4.set_ylabel('Relative frequency',fontsize = 12)
# ax4.set_title('Buffer layer ('+ str(np.around(per_myfro_BL*100,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')
# ax4.grid()

plt.suptitle('MAC$_{AVE}$ ('+ str(np.around(per_myfro,decimals=1))+'%)',fontsize = 14 ,fontweight='bold')

fig.savefig(path_data + 'histclas_myfronts.eps', format='eps', dpi=1200)
plt.show()
