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
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import plot,show, grid, legend
#from skewt import SkewT
from matplotlib.pyplot import rcParams,figure,show,draw

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/CSV/K/'
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/clusters/WFronts/'



#*****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
#                            MAC Data Original Levels
#*****************************************************************************\
# ****************************************************************************\
path_databom=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/files_bom/'
matb1= sio.loadmat(path_databom+'BOM_1995.mat')
bom_in=matb1['BOM_S'][:]
timesd= matb1['time'][:]
bom=bom_in

for y in range(1996,2011):
    matb= sio.loadmat(path_databom+'BOM_'+str(y)+'.mat')
    bom_r=matb['BOM_S'][:]
    timesd_r= matb['time'][:]
    bom=np.concatenate((bom,bom_r), axis=2)
    timesd=np.concatenate((timesd,timesd_r), axis=1)

ni=bom.shape
#*****************************************************************************\
#Delete special cases
indexdel1=[]
indexdel2=[]
#Delete when array is only NaN
for j in range(0,ni[2]):
    if np.isnan(np.nansum(bom[:,0,j]))==True:
        indexdel1=np.append(indexdel1,j)

bom=np.delete(bom, indexdel1,axis=2)
ni=bom.shape
timesd=np.delete(timesd, indexdel1)

#Delete when max height is lower than 2500 mts
for j in range(0,ni[2]):
    if np.nanmax(bom[:,1,j])<2500:
        indexdel2=np.append(indexdel2,j)

bom=np.delete(bom, indexdel2,axis=2) #(3000,8,3545)
ni=bom.shape
timesd=np.delete(timesd, indexdel2)
#*****************************************************************************\

#Convert Matlab datenum into Python datetime source: https://gist.github.com/vicow)
def datenum_to_datetime(datenum):
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
        + timedelta(days=int(days)) \
        + timedelta(hours=int(hours)) \
        + timedelta(minutes=int(minutes)) \
        + timedelta(seconds=round(seconds)) \
        - timedelta(days=366)


#*****************************************************************************\
#Separation of Variables
pres=bom[:,0,:].reshape(ni[0],ni[2])
hght=bom[:,1,:].reshape(ni[0],ni[2])
temp=bom[:,2,:].reshape(ni[0],ni[2])
dwpo=bom[:,3,:].reshape(ni[0],ni[2])
mixr=bom[:,5,:].reshape(ni[0],ni[2])
wdir_initial=bom[:,6,:].reshape(ni[0],ni[2])
wspd=bom[:,7,:].reshape(ni[0],ni[2])*0.5444444
relh=bom[:,4,:].reshape(ni[0],ni[2])

u=wspd*(np.cos(np.radians(270-wdir_initial)))
v=wspd*(np.sin(np.radians(270-wdir_initial)))
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data YOTC Levels
#*****************************************************************************\
#*****************************************************************************\
#Leyendo Alturas y press
file_levels = np.genfromtxt('./../../Read_Files/YOTC/levels.csv', delimiter=',')
hlev_yotc=file_levels[:,6]
plev_yotc=file_levels[:,4] #value 10 is 925
#*****************************************************************************\
#Interpolation to YOTC Levels
# prutemp=np.empty((len(hlev_yotc),0))
# prumixr=np.empty((len(hlev_yotc),0))
# pruu=np.empty((len(hlev_yotc),0))
# pruv=np.empty((len(hlev_yotc),0))
# prurelh=np.empty((len(hlev_yotc),0))
# prudwpo=np.empty((len(hlev_yotc),0))

# for j in range(0,ni[2]):
# #height initialization
#     x=hght[:,j]
#     x[-1]=np.nan
#     new_x=hlev_yotc
# #Interpolation YOTC levels
#     yt=temp[:,j]
#     rest=interp1d(x,yt)(new_x)
#     prutemp=np.append(prutemp,rest)

#     ym=mixr[:,j]
#     resm=interp1d(x,ym)(new_x)
#     prumixr=np.append(prumixr,resm)

#     yw=u[:,j]
#     resw=interp1d(x,yw)(new_x)
#     pruu=np.append(pruu,resw)

#     yd=v[:,j]
#     resd=interp1d(x,yd)(new_x)
#     pruv=np.append(pruv,resd)

#     yr=relh[:,j]
#     resr=interp1d(x,yr)(new_x)
#     prurelh=np.append(prurelh,resr)

#     ydp=dwpo[:,j]
#     resr=interp1d(x,ydp)(new_x)
#     prudwpo=np.append(prudwpo,resr)

# tempmac_ylev=prutemp.reshape(-1,len(hlev_yotc)).transpose()
# umac_ylev=pruu.reshape(-1,len(hlev_yotc)).transpose()
# vmac_ylev=pruv.reshape(-1,len(hlev_yotc)).transpose()
# mixrmac_ylev=prumixr.reshape(-1,len(hlev_yotc)).transpose()
# relhmac_ylev=prurelh.reshape(-1,len(hlev_yotc)).transpose()
# dwpomac_ylev=prudwpo.reshape(-1,len(hlev_yotc)).transpose()

# wspdmac_ylev=np.sqrt(umac_ylev**2 + vmac_ylev**2)
# wdirmac_ylev=np.arctan2(-umac_ylev, -vmac_ylev)*(180/np.pi)
# wdirmac_ylev[(umac_ylev == 0) & (vmac_ylev == 0)]=0


# relhum_my=relhmac_ylev.T
# temp_my=tempmac_ylev.T
# u_my=umac_ylev.T
# v_my=vmac_ylev.T
# mixr_my=mixrmac_ylev.T
# dwpo_my=dwpomac_ylev.T

# wsp_my=wspdmac_ylev.T
# wdir_my=wdirmac_ylev.T

plev_std=plev_yotc

temp_pres=np.zeros((len(plev_std),ni[2]),'float')
mixr_pres=np.zeros((len(plev_std),ni[2]),'float')
u_pres=np.zeros((len(plev_std),ni[2]),'float')
v_pres=np.zeros((len(plev_std),ni[2]),'float')
relh_pres=np.zeros((len(plev_std),ni[2]),'float')
dwpo_pres=np.zeros((len(plev_std),ni[2]),'float')

for j in range(0,ni[2]):

    yt=temp[~np.isnan(temp[:,j]),j]
    ym=mixr[~np.isnan(mixr[:,j]),j]
    yw=u[~np.isnan(u[:,j]),j]
    yd=v[~np.isnan(v[:,j]),j]
    yr=relh[~np.isnan(relh[:,j]),j]
    yp=dwpo[~np.isnan(dwpo[:,j]),j]
    xp=pres[~np.isnan(temp[:,j]),j]

    temp_interp_pres=si.UnivariateSpline(xp[::-1],yt[::-1],k=5)
    mixr_interp_pres=si.UnivariateSpline(xp[::-1],ym[::-1],k=5)
    u_interp_pres=si.UnivariateSpline(xp[::-1],yw[::-1],k=5)
    v_interp_pres=si.UnivariateSpline(xp[::-1],yd[::-1],k=5)
    relh_interp_pres=si.UnivariateSpline(xp[::-1],yr[::-1],k=5)
    dwpo_interp_pres=si.UnivariateSpline(xp[::-1],yp[::-1],k=5)

    for ind in range(0,len(plev_std)):
        temp_pres[ind,j]=temp_interp_pres(plev_std[ind])
        mixr_pres[ind,j]=mixr_interp_pres(plev_std[ind])
        u_pres[ind,j]=u_interp_pres(plev_std[ind])
        v_pres[ind,j]=v_interp_pres(plev_std[ind])
        relh_pres[ind,j]=relh_interp_pres(plev_std[ind])
        dwpo_pres[ind,j]=dwpo_interp_pres(plev_std[ind])


    relh_pres[relh_pres[:,j]>np.nanmax(yr),j]=np.nan
    relh_pres[relh_pres[:,j]<np.nanmin(yr),j]=np.nan

    temp_pres[temp_pres[:,j]>np.nanmax(yt),j]=np.nan
    temp_pres[temp_pres[:,j]<np.nanmin(yt),j]=np.nan

    u_pres[u_pres[:,j]>np.nanmax(yw),j]=np.nan
    u_pres[u_pres[:,j]<np.nanmin(yw),j]=np.nan
    v_pres[v_pres[:,j]>np.nanmax(yd),j]=np.nan
    v_pres[v_pres[:,j]<np.nanmin(yd),j]=np.nan

    mixr_pres[mixr_pres[:,j]>np.nanmax(ym),j]=np.nan
    mixr_pres[mixr_pres[:,j]<np.nanmin(ym),j]=np.nan

    dwpo_pres[dwpo_pres[:,j]>np.nanmax(yp),j]=np.nan
    dwpo_pres[dwpo_pres[:,j]<np.nanmin(yp),j]=np.nan

    del xp, yt, ym, yw, yd, yr, yp

temp_my=temp_pres.T
u_my=u_pres.T
v_my=v_pres.T
mixr_my=mixr_pres.T
relhum_my=relh_pres.T
dwpo_my=dwpo_pres.T

wsp_my=np.sqrt(u_my**2 + v_my**2)
wdir_my=np.arctan2(-u_my, -v_my)*(180/np.pi)
wdir_my[(u_my == 0) & (v_my == 0)]=0


#*****************************************************************************\
#Cambiar fechas
timestamp = [datenum_to_datetime(t) for t in timesd]
time_my = np.array(timestamp)
time_my_ori = np.array(timestamp)

for i in range(0,ni[2]):
    #Cuando cae 23 horas del 31 de diciembre agrega un anio
    if (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==12:
        y1=time_my[i].year
        time_my[i]=time_my[i].replace(year=y1+1,hour=0, month=1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==1:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==3:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==5:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==7:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==8:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==10:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==4:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==6:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==9:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==11:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==28 and time_my[i].month==2:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    #Bisiesto 2008
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2008:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2004:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2000:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==1996:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
    if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==1992:
        m1=time_my[i].month
        time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)


    #Cuando cae 23 horas, mueve hora a las 00 del dia siguiente
    if time_my[i].hour==23 or time_my[i].hour==22:
        d1=time_my[i].day
        time_my[i]=time_my[i].replace(hour=0,day=d1+1)
    else:
        time_my[i]=time_my[i]
    #Cuando cae 11 horas, mueve hora a las 12 del mismo dia
    if time_my[i].hour==11 or time_my[i].hour==10 or time_my[i].hour==13 or time_my[i].hour==14:
        time_my[i]=time_my[i].replace(hour=12)
    else:
        time_my[i]=time_my[i]

    #Cuando cae 1 horas, mueve hora a las 0 del mismo dia
    if time_my[i].hour==1 or time_my[i].hour==2:
        time_my[i]=time_my[i].replace(hour=0)
    else:
        time_my[i]=time_my[i]

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                          Dataframes 1995-2010                               \
#*****************************************************************************\
#*****************************************************************************\
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
#*****************************************************************************\
#CL3=np.empty(len(df_cluster))*np.nan
#*****************************************************************************\
#Dataframe MAC YOTC levels
t_list=temp_my.tolist()
u_list=u_my.tolist()
v_list=v_my.tolist()
rh_list=relhum_my.tolist()
mr_list=mixr_my.tolist()
dwpo_list=dwpo_my.tolist()
wsp_list=wsp_my.tolist()
wdir_list=wdir_my.tolist()

dmy={'temp':t_list,
'u':u_list,
'v':v_list,
'RH':rh_list,
'dewp':dwpo_list,
'mixr':mr_list,
'wsp':wsp_list,
'wdir':wdir_list}

df_clumac= pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
df_clumac=df_clumac.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

df_clumac=df_clumac.reindex(date_index_all)
df_clumac.index.name = 'Date'
#*****************************************************************************\
#*****************************************************************************\
#Reading FRONTS
#*****************************************************************************\
#*****************************************************************************\
path_data_front=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
df_wfront= pd.read_csv(path_data_front + 'df_wfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_wfront= df_wfront.set_index('Date')

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               4 Clusters
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Reading CSV Cluster
#*****************************************************************************\
#*****************************************************************************\
#925-850-700

df_cluster= pd.read_csv(path_data + 'cluster_warmfront_4k.csv', sep=',', parse_dates=['Date']).set_index('Date')

# K4=np.array(df_cluster['Cluster'])
# dist_clu=np.array(df_cluster['Distance'])

#*****************************************************************************\
#Concadanate dataframes
#*****************************************************************************\
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
df_cluster.reindex(date_index_all)
df_clumac_k=pd.concat([df_clumac, df_cluster],axis=1)

df_myclu_wfro=pd.concat([df_clumac_k, df_wfront],axis=1)

df_myclu_solowfro=df_myclu_wfro[np.isfinite(df_myclu_wfro['Dist WFront'])]

df_myclu_solowfro10=df_myclu_wfro[(df_myclu_wfro['Dist WFront']<=10) & (df_myclu_wfro['Dist WFront']>=-10)]


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Histograms Warm Front
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
xx1=-10
xx2=10
dx=1
bx=np.arange(xx1,xx2+1,2)
width = 1
df1=10
df2=20
color1='#FF3333'
color2='#9D1309'
bins=20

yy1=0
yy2=0.14
by=np.arange(yy1,yy2+0.02,0.02)

#*****************************************************************************\
#*****************************************************************************\
#                                   4K
#*****************************************************************************\
#*****************************************************************************\
#Cambios: df_myclu_solowfro10, creado bins=20, bx,xx1, xx2, df1=15, df2=30



df_1 = df_myclu_wfro[df_myclu_wfro['Cluster']==1]
df_2 = df_myclu_wfro[df_myclu_wfro['Cluster']==2]
df_3 = df_myclu_wfro[df_myclu_wfro['Cluster']==3]
df_4 = df_myclu_wfro[df_myclu_wfro['Cluster']==4]

row = 2
column = 2

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(14,8))
ax1, ax2, ax3,ax4= axes.flat
#*****************************************************************************\
#C1
#*****************************************************************************\



dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==1]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==1]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)


x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)

bin_edges=np.arange(xx1, xx2+1, 1)

ax1.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax1.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_ylim([yy1,yy2])
#ax1.set_xlim([xx1,xx2])
ax1.axvline(0, color='k')
ax1.legend(loc='upper right')
ax1.grid()
#ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Frequency',fontsize = 12)
ax1.set_title('Cluster 1 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')


#*****************************************************************************\
#C2
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==2]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==2]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)
pe2=len(df)/float(len(dfc))
pe2=round(pe2,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax2.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax2.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_ylim([yy1,yy2])
ax2.axvline(0, color='k')
ax2.grid()
#ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax2.set_ylabel('Frequency',fontsize = 12)
ax2.set_title('Cluster 2 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

#*****************************************************************************\
#C3
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==3]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==3]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)
pe2=len(df)/float(len(dfc))
pe2=round(pe2,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax3.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax3.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_ylim([yy1,yy2])
ax3.axvline(0, color='k')
ax3.grid()
ax3.set_xlabel('Distance to front: warm to cold sector (deg)',fontsize = 12)
ax3.set_ylabel('Frequency',fontsize = 12)
ax3.set_title('Cluster 3 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

#*****************************************************************************\
#C4
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==4]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==4]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)
pe2=len(df)/float(len(dfc))
pe2=round(pe2,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax4.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax4.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_yticks(by)
ax4.set_ylim([yy1,yy2])
ax4.set_xlim([xx1,xx2])
ax4.axvline(0, color='k')
ax4.grid()
ax4.set_xlabel('Distance to front: warm to cold sector (deg)',fontsize = 12)
#ax4.set_ylabel('Frequency',fontsize = 12)
ax4.set_title('Cluster 4 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

fig.tight_layout()
plt.savefig(path_data_save + '4K_cluster_warmfront_10deg.eps', format='eps', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               5 Clusters
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

df_cluster= pd.read_csv(path_data + 'cluster_warmfront_5k.csv', sep=',', parse_dates=['Date']).set_index('Date')

# K4=np.array(df_cluster['Cluster'])
# dist_clu=np.array(df_cluster['Distance'])
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
df_cluster.reindex(date_index_all)

#*****************************************************************************\
#Concadanate dataframes
#*****************************************************************************\
df_clumac_k=pd.concat([df_clumac, df_cluster],axis=1)

df_myclu_wfro=pd.concat([df_clumac_k, df_wfront],axis=1)

df_myclu_solowfro=df_myclu_wfro[np.isfinite(df_myclu_wfro['Dist WFront'])]

df_myclu_solowfro10=df_myclu_wfro[(df_myclu_wfro['Dist WFront']<=10) & (df_myclu_wfro['Dist WFront']>=-10)]


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Histograms Warm Front
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
xx1=-10
xx2=10
dx=1
bx=np.arange(xx1,xx2+1,2)
width = 1
df1=10
df2=20

bins=20

yy1=0
yy2=0.14
by=np.arange(yy1,yy2+0.02,0.02)

#*****************************************************************************\
#*****************************************************************************\
#                                   5K
#*****************************************************************************\
#*****************************************************************************\
#Cambios: df_myclu_solowfro10, creado bins=20, bx,xx1, xx2, df1=15, df2=30



df_1 = df_myclu_wfro[df_myclu_wfro['Cluster']==1]
df_2 = df_myclu_wfro[df_myclu_wfro['Cluster']==2]
df_3 = df_myclu_wfro[df_myclu_wfro['Cluster']==3]
df_4 = df_myclu_wfro[df_myclu_wfro['Cluster']==4]
df_5 = df_myclu_wfro[df_myclu_wfro['Cluster']==5]

row = 3
column = 2

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(14,10))
ax1, ax2, ax3, ax4, ax5, ax6= axes.flat
#*****************************************************************************\
#C1
#*****************************************************************************\

dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==1]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==1]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)
pe2=len(df)/float(len(dfc))
pe2=round(pe2,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)

bin_edges=np.arange(xx1, xx2+1, 1)

ax1.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax1.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_ylim([yy1,yy2])
#ax1.set_xlim([xx1,xx2])
ax1.axvline(0, color='k')
ax1.legend(loc='upper right')
ax1.grid()
#ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Frequency',fontsize = 12)
ax1.set_title('Cluster 1 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')


#*****************************************************************************\
#C2
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==2]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==2]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)
pe2=len(df)/float(len(dfc))
pe2=round(pe2,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax2.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax2.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_ylim([yy1,yy2])
ax2.axvline(0, color='k')
ax2.grid()
#ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax2.set_ylabel('Frequency',fontsize = 12)
ax2.set_title('Cluster 2 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

#*****************************************************************************\
#C3
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==3]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==3]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)
pe2=len(df)/float(len(dfc))
pe2=round(pe2,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax3.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax3.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_ylim([yy1,yy2])
ax3.axvline(0, color='k')
ax3.grid()
ax3.set_xlabel('Distance to front: warm to cold sector (deg)',fontsize = 12)
ax3.set_ylabel('Frequency',fontsize = 12)
ax3.set_title('Cluster 3 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

#*****************************************************************************\
#C4
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==4]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==4]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)
pe2=len(df)/float(len(dfc))
pe2=round(pe2,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax4.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax4.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(bx)
ax4.set_yticks(by)
ax4.set_ylim([yy1,yy2])
ax4.set_xlim([xx1,xx2])
ax4.axvline(0, color='k')
ax4.grid()
ax4.set_xlabel('Distance to front: warm to cold sector (deg)',fontsize = 12)
#ax4.set_ylabel('Frequency',fontsize = 12)
ax4.set_title('Cluster 4 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

#*****************************************************************************\
#C5
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==5]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==5]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)
pe2=len(df)/float(len(dfc))
pe2=round(pe2,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax5.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax5.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_xticks(bx)
ax5.set_yticks(by)
ax5.set_ylim([yy1,yy2])
ax5.set_xlim([xx1,xx2])
ax5.axvline(0, color='k')
ax5.grid()
ax5.set_xlabel('Distance to front: warm to cold sector (deg)',fontsize = 12)
#ax4.set_ylabel('Frequency',fontsize = 12)
ax5.set_title('Cluster 5 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

fig.tight_layout()
plt.savefig(path_data_save + '5K_cluster_warmfront_10deg.eps', format='eps', dpi=1200)
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               3 Clusters
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Reading CSV Cluster
#*****************************************************************************\
#*****************************************************************************\
#925-850-700

df_cluster= pd.read_csv(path_data + 'cluster_warmfront_3k.csv', sep=',', parse_dates=['Date']).set_index('Date')
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
df_cluster.reindex(date_index_all)
# K4=np.array(df_cluster['Cluster'])
# dist_clu=np.array(df_cluster['Distance'])

#*****************************************************************************\
#Concadanate dataframes
#*****************************************************************************\

df_clumac_k=pd.concat([df_clumac, df_cluster],axis=1)

df_myclu_wfro=pd.concat([df_clumac_k, df_wfront],axis=1)

df_myclu_solowfro=df_myclu_wfro[np.isfinite(df_myclu_wfro['Dist WFront'])]

df_myclu_solowfro10=df_myclu_wfro[(df_myclu_wfro['Dist WFront']<=10) & (df_myclu_wfro['Dist WFront']>=-10)]


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               Histograms Warm Front
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
xx1=-10
xx2=10
dx=1
bx=np.arange(xx1,xx2+1,2)
width = 1
df1=10
df2=20
bins=20

yy1=0
yy2=0.14
by=np.arange(yy1,yy2+0.02,0.02)

#*****************************************************************************\
#*****************************************************************************\
#                                   4K
#*****************************************************************************\
#*****************************************************************************\
#Cambios: df_myclu_solowfro10, creado bins=20, bx,xx1, xx2, df1=15, df2=30



df_1 = df_myclu_wfro[df_myclu_wfro['Cluster']==1]
df_2 = df_myclu_wfro[df_myclu_wfro['Cluster']==2]
df_3 = df_myclu_wfro[df_myclu_wfro['Cluster']==3]


df_myclu_wfro[np.isfinite(df_myclu_wfro['Cluster'])]

row = 2
column = 2

fig, axes = plt.subplots(row, column, facecolor='w', figsize=(14,8))
ax1, ax2, ax3,ax4= axes.flat
#*****************************************************************************\
#C1
#*****************************************************************************\



dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==1]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==1]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)


x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)

bin_edges=np.arange(xx1, xx2+1, 1)

ax1.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax1.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(bx)
ax1.set_yticks(by)
ax1.set_ylim([yy1,yy2])
#ax1.set_xlim([xx1,xx2])
ax1.axvline(0, color='k')
ax1.legend(loc='upper right')
ax1.grid()
#ax1.set_xlabel('Distance from front (deg)',fontsize = 12)
ax1.set_ylabel('Frequency',fontsize = 12)
ax1.set_title('Cluster 1 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')


#*****************************************************************************\
#C2
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==2]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==2]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)

x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax2.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax2.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(bx)
ax2.set_yticks(by)
ax2.set_ylim([yy1,yy2])
ax2.axvline(0, color='k')
ax2.grid()
#ax2.set_xlabel('Distance from front (deg)',fontsize = 12)
#ax2.set_ylabel('Frequency',fontsize = 12)
ax2.set_title('Cluster 2 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

#*****************************************************************************\
#C3
#*****************************************************************************\
dfc = df_myclu_wfro[df_myclu_wfro['Cluster']==3]
df = df_myclu_solowfro10[df_myclu_solowfro10['Cluster']==3]

pe1=len(dfc)/float(len(df_cluster))
pe1=round(pe1,2)


x1=np.array(df['Dist WFront'])*(-1)
y1=np.array(df['Cluster'])
y1 = np.array(map(int, y1))

bin_count, bin_edges, binnumber = stats.binned_statistic(x1, y1, statistic='count', bins=bins)

bin_perc=bin_count/len(y1)
bin_edges=np.arange(xx1, xx2+1, 1)

ax3.bar(bin_edges[0:df1],bin_perc[0:df1],width,alpha=0.5, color=color1,label='postfront')
ax3.bar(bin_edges[df1:df2],bin_perc[df1:df2],width,alpha=0.5, color=color2,label='prefront')

ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(bx)
ax3.set_yticks(by)
ax3.set_ylim([yy1,yy2])
ax3.axvline(0, color='k')
ax3.grid()
ax3.set_xlabel('Distance to front: warm to cold sector (deg)',fontsize = 12)
ax3.set_ylabel('Frequency',fontsize = 12)
ax3.set_title('Cluster 3 (P$_t$=' + str(pe1)+', N$_t$='+str(len(dfc)) +')' ,fontsize = 14 ,fontweight='bold')

fig.tight_layout()
plt.savefig(path_data_save + '3K_cluster_warmfront_10deg.eps', format='eps', dpi=1200)
plt.show()


