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
from pylab import plot,show, grid, legend
from skewt import SkewT
from matplotlib.pyplot import rcParams,figure,show,draw

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/'
#*****************************************************************************\
#Reading CSV Cluster
#*****************************************************************************\

#925-900-850-800-750-700
df_cluster= pd.read_csv(path_data + 'All_ClusterAnalysis.csv', sep=',', parse_dates=['Date'])
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/Profiles/6Levels/'
CL4=np.array(df_cluster['QCL_2'])
CL5=np.array(df_cluster['QCL_3'])
CL6=np.array(df_cluster['QCL_4'])
CL7=np.array(df_cluster['QCL_5'])
#*****************************************************************************\
#925-850-700
# df_cluster= pd.read_csv(path_data + 'All_ClusterAnalysis.csv', sep=',', parse_dates=['Date'])
# #Reescribe columnas para no cambiar los outputs
# CL4=np.array(df_cluster['QCL_7'])
# CL5=np.array(df_cluster['QCL_8'])
# CL6=np.array(df_cluster['QCL_9'])
# CL7=np.array(df_cluster['QCL_10'])
# path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/Profiles/3Levels/'

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
file_levels = np.genfromtxt('./../Read_Files/YOTC/levels.csv', delimiter=',')
hlev_yotc=file_levels[:,6]
plev_yotc=file_levels[:,4] #value 10 is 925
#*****************************************************************************\
#Interpolation to YOTC Levels
prutemp=np.empty((len(hlev_yotc),0))
prumixr=np.empty((len(hlev_yotc),0))
pruu=np.empty((len(hlev_yotc),0))
pruv=np.empty((len(hlev_yotc),0))
prurelh=np.empty((len(hlev_yotc),0))
prudwpo=np.empty((len(hlev_yotc),0))

for j in range(0,ni[2]):
#height initialization
    x=hght[:,j]
    x[-1]=np.nan
    new_x=hlev_yotc
#Interpolation YOTC levels
    yt=temp[:,j]
    rest=interp1d(x,yt)(new_x)
    prutemp=np.append(prutemp,rest)

    ym=mixr[:,j]
    resm=interp1d(x,ym)(new_x)
    prumixr=np.append(prumixr,resm)

    yw=u[:,j]
    resw=interp1d(x,yw)(new_x)
    pruu=np.append(pruu,resw)

    yd=v[:,j]
    resd=interp1d(x,yd)(new_x)
    pruv=np.append(pruv,resd)

    yr=relh[:,j]
    resr=interp1d(x,yr)(new_x)
    prurelh=np.append(prurelh,resr)

    ydp=dwpo[:,j]
    resr=interp1d(x,ydp)(new_x)
    prudwpo=np.append(prudwpo,resr)

tempmac_ylev=prutemp.reshape(-1,len(hlev_yotc)).transpose()
umac_ylev=pruu.reshape(-1,len(hlev_yotc)).transpose()
vmac_ylev=pruv.reshape(-1,len(hlev_yotc)).transpose()
mixrmac_ylev=prumixr.reshape(-1,len(hlev_yotc)).transpose()
relhmac_ylev=prurelh.reshape(-1,len(hlev_yotc)).transpose()
dwpomac_ylev=prudwpo.reshape(-1,len(hlev_yotc)).transpose()

wspdmac_ylev=np.sqrt(umac_ylev**2 + vmac_ylev**2)
wdirmac_ylev=np.arctan2(-umac_ylev, -vmac_ylev)*(180/np.pi)
wdirmac_ylev[(umac_ylev == 0) & (vmac_ylev == 0)]=0


relhum_my=relhmac_ylev.T
temp_my=tempmac_ylev.T
u_my=umac_ylev.T
v_my=vmac_ylev.T
mixr_my=mixrmac_ylev.T
dwpo_my=dwpomac_ylev.T

wsp_my=wspdmac_ylev.T
wdir_my=wdirmac_ylev.T

#*****************************************************************************\
#*****************************************************************************\
#                          Dataframes 1995-2010                               \
#*****************************************************************************\
#*****************************************************************************\
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
timestamp = [datenum_to_datetime(t) for t in timesd]
time_my = np.array(timestamp)
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
'wdir':wdir_list,
'CL_7':CL7,
'CL_4':CL4,
'CL_5':CL5,
'CL_6':CL6}

df_clumac= pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
df_clumac=df_clumac.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               3 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
# df_CL3_G1 = df_clumac[df_clumac['CL_3']=='1']
# df_CL3_G2 = df_clumac[df_clumac['CL_3']=='2']
# df_CL3_G3 = df_clumac[df_clumac['CL_3']=='3']

# #*****************************************************************************\
# #C1
# #*****************************************************************************\
# df=df_CL3_G1
# RH=np.empty([len(df),91])*np.nan
# U=np.empty([len(df),91])*np.nan
# V=np.empty([len(df),91])*np.nan
# T=np.empty([len(df),91])*np.nan
# DP=np.empty([len(df),91])*np.nan
# MX=np.empty([len(df),91])*np.nan


# for i in range(0,len(df)):
#     RH[i,:]=np.array(df['RH'][i])
#     U[i,:]=np.array(df['u'][i])
#     V[i,:]=np.array(df['v'][i])
#     T[i,:]=np.array(df['temp'][i])
#     DP[i,:]=np.array(df['dewp'][i])
#     MX[i,:]=np.array(df['mixr'][i])

# #Mean profiles
# rhum_CL3_G1=np.nanmean(RH,axis=0)
# temp_CL3_G1=np.nanmean(T,axis=0)
# dewp_CL3_G1=np.nanmean(DP,axis=0)
# mixr_CL3_G1=np.nanmean(MX,axis=0)
# v_CL3_G1=np.nanmean(V,axis=0)
# u_CL3_G1=np.nanmean(U,axis=0)
# wsp_CL3_G1=np.sqrt(u_CL3_G1**2 + v_CL3_G1**2)
# wdir_CL3_G1=np.arctan2(-u_CL3_G1, -v_CL3_G1)*(180/np.pi)
# wdir_CL3_G1[(u_CL3_G1 == 0) & (v_CL3_G1 == 0)]=0
# #*****************************************************************************\
# #C2
# #*****************************************************************************\
# df=df_CL3_G2
# RH=np.empty([len(df),91])*np.nan
# U=np.empty([len(df),91])*np.nan
# V=np.empty([len(df),91])*np.nan
# T=np.empty([len(df),91])*np.nan
# DP=np.empty([len(df),91])*np.nan
# MX=np.empty([len(df),91])*np.nan


# for i in range(0,len(df)):
#     RH[i,:]=np.array(df['RH'][i])
#     U[i,:]=np.array(df['u'][i])
#     V[i,:]=np.array(df['v'][i])
#     T[i,:]=np.array(df['temp'][i])
#     DP[i,:]=np.array(df['dewp'][i])
#     MX[i,:]=np.array(df['mixr'][i])

# #Mean profiles
# rhum_CL3_G2=np.nanmean(RH,axis=0)
# temp_CL3_G2=np.nanmean(T,axis=0)
# dewp_CL3_G2=np.nanmean(DP,axis=0)
# mixr_CL3_G2=np.nanmean(MX,axis=0)
# v_CL3_G2=np.nanmean(V,axis=0)
# u_CL3_G2=np.nanmean(U,axis=0)
# wsp_CL3_G2=np.sqrt(u_CL3_G2**2 + v_CL3_G2**2)
# wdir_CL3_G2=np.arctan2(-u_CL3_G2, -v_CL3_G2)*(180/np.pi)
# wdir_CL3_G2[(u_CL3_G2 == 0) & (v_CL3_G2 == 0)]=0
# #*****************************************************************************\
# #C3
# #*****************************************************************************\
# df=df_CL3_G3
# RH=np.empty([len(df),91])*np.nan
# U=np.empty([len(df),91])*np.nan
# V=np.empty([len(df),91])*np.nan
# T=np.empty([len(df),91])*np.nan
# DP=np.empty([len(df),91])*np.nan
# MX=np.empty([len(df),91])*np.nan


# for i in range(0,len(df)):
#     RH[i,:]=np.array(df['RH'][i])
#     U[i,:]=np.array(df['u'][i])
#     V[i,:]=np.array(df['v'][i])
#     T[i,:]=np.array(df['temp'][i])
#     DP[i,:]=np.array(df['dewp'][i])
#     MX[i,:]=np.array(df['mixr'][i])

# #Mean profiles
# rhum_CL3_G3=np.nanmean(RH,axis=0)
# temp_CL3_G3=np.nanmean(T,axis=0)
# dewp_CL3_G3=np.nanmean(DP,axis=0)
# mixr_CL3_G3=np.nanmean(MX,axis=0)
# v_CL3_G3=np.nanmean(V,axis=0)
# u_CL3_G3=np.nanmean(U,axis=0)
# wsp_CL3_G3=np.sqrt(u_CL3_G2**2 + v_CL3_G2**2)
# wdir_CL3_G3=np.arctan2(-u_CL3_G2, -v_CL3_G2)*(180/np.pi)
# wdir_CL3_G3[(u_CL3_G2 == 0) & (v_CL3_G2 == 0)]=0
# #*****************************************************************************\
# #Plot
# #*****************************************************************************\
# height_m=hlev_yotc
# pressure_pa=plev_yotc
# #C1
# temperature_c=temp_CL3_G1
# dewpoint_c=dewp_CL3_G1
# wsp=wsp_CL3_G1*1.943844
# wdir=wdir_CL3_G1

# mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'3 Clusters')))

# #C2
# temperature_c=temp_CL3_G2
# dewpoint_c=dewp_CL3_G2
# wsp=wsp_CL3_G2*1.943844
# wdir=wdir_CL3_G2

# mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'BoM','C2')))

# #C3
# temperature_c=temp_CL3_G3
# dewpoint_c=dewp_CL3_G3
# wsp=wsp_CL3_G3*1.943844
# wdir=wdir_CL3_G3

# mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'BoM','C3')))


# S=SkewT.Sounding(soundingdata=mydataG1)
# T=SkewT.Sounding(soundingdata=mydataG2)
# Z=SkewT.Sounding(soundingdata=mydataG3)
# S.make_skewt_axes()
# S.add_profile(color='r',bloc=0)
# S.soundingdata=T.soundingdata
# S.add_profile(color='b',bloc=1)
# S.soundingdata=Z.soundingdata
# S.add_profile(color='g',bloc=2)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               4 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
df_CL4_G1 = df_clumac[df_clumac['CL_4']=='1']
df_CL4_G2 = df_clumac[df_clumac['CL_4']=='2']
df_CL4_G3 = df_clumac[df_clumac['CL_4']=='3']
df_CL4_G4 = df_clumac[df_clumac['CL_4']=='4']

#*****************************************************************************\
#C1
#*****************************************************************************\
df=df_CL4_G1
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL4_G1=np.nanmean(RH,axis=0)
temp_CL4_G1=np.nanmean(T,axis=0)
dewp_CL4_G1=np.nanmean(DP,axis=0)
mixr_CL4_G1=np.nanmean(MX,axis=0)
v_CL4_G1=np.nanmean(V,axis=0)
u_CL4_G1=np.nanmean(U,axis=0)
wsp_CL4_G1=np.sqrt(u_CL4_G1**2 + v_CL4_G1**2)
wdir_CL4_G1=np.arctan2(-u_CL4_G1, -v_CL4_G1)*(180/np.pi)
wdir_CL4_G1[(u_CL4_G1 == 0) & (v_CL4_G1 == 0)]=0
#*****************************************************************************\
#C2
#*****************************************************************************\
df=df_CL4_G2
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL4_G2=np.nanmean(RH,axis=0)
temp_CL4_G2=np.nanmean(T,axis=0)
dewp_CL4_G2=np.nanmean(DP,axis=0)
mixr_CL4_G2=np.nanmean(MX,axis=0)
v_CL4_G2=np.nanmean(V,axis=0)
u_CL4_G2=np.nanmean(U,axis=0)
wsp_CL4_G2=np.sqrt(u_CL4_G2**2 + v_CL4_G2**2)
wdir_CL4_G2=np.arctan2(-u_CL4_G2, -v_CL4_G2)*(180/np.pi)
wdir_CL4_G2[(u_CL4_G2 == 0) & (v_CL4_G2 == 0)]=0
#*****************************************************************************\
#C3
#*****************************************************************************\
df=df_CL4_G3
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL4_G3=np.nanmean(RH,axis=0)
temp_CL4_G3=np.nanmean(T,axis=0)
dewp_CL4_G3=np.nanmean(DP,axis=0)
mixr_CL4_G3=np.nanmean(MX,axis=0)
v_CL4_G3=np.nanmean(V,axis=0)
u_CL4_G3=np.nanmean(U,axis=0)
wsp_CL4_G3=np.sqrt(u_CL4_G3**2 + v_CL4_G3**2)
wdir_CL4_G3=np.arctan2(-u_CL4_G3, -v_CL4_G3)*(180/np.pi)
wdir_CL4_G3[(u_CL4_G3 == 0) & (v_CL4_G3 == 0)]=0
#*****************************************************************************\
#C4
#*****************************************************************************\
df=df_CL4_G4
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL4_G4=np.nanmean(RH,axis=0)
temp_CL4_G4=np.nanmean(T,axis=0)
dewp_CL4_G4=np.nanmean(DP,axis=0)
mixr_CL4_G4=np.nanmean(MX,axis=0)
v_CL4_G4=np.nanmean(V,axis=0)
u_CL4_G4=np.nanmean(U,axis=0)
wsp_CL4_G4=np.sqrt(u_CL4_G4**2 + v_CL4_G4**2)
wdir_CL4_G4=np.arctan2(-u_CL4_G4, -v_CL4_G4)*(180/np.pi)
wdir_CL4_G4[(u_CL4_G4 == 0) & (v_CL4_G4 == 0)]=0
#*****************************************************************************\
#Plot
#*****************************************************************************\
height_m=hlev_yotc
pressure_pa=plev_yotc
#C1
temperature_c=temp_CL4_G1
dewpoint_c=dewp_CL4_G1
wsp=wsp_CL4_G1*1.943844
wdir=wdir_CL4_G1

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','C1')))

#C2
temperature_c=temp_CL4_G2
dewpoint_c=dewp_CL4_G2
wsp=wsp_CL4_G2*1.943844
wdir=wdir_CL4_G2

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','C2')))

#C3
temperature_c=temp_CL4_G3
dewpoint_c=dewp_CL4_G3
wsp=wsp_CL4_G3*1.943844
wdir=wdir_CL4_G3

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','C3')))
#C4
temperature_c=temp_CL4_G4
dewpoint_c=dewp_CL4_G4
wsp=wsp_CL4_G4*1.943844
wdir=wdir_CL4_G4

mydataG4=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','C4')))



#Individuals
S1=SkewT.Sounding(soundingdata=mydataG1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + '4K_C1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydataG2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + '4K_C2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydataG3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + '4K_C3.png', format='png', dpi=1200)
S4=SkewT.Sounding(soundingdata=mydataG4)
S4.plot_skewt(color='r')
plt.savefig(path_data_save + '4K_C4.png', format='png', dpi=1200)

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=0)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=1)
S.soundingdata=S4.soundingdata
S.add_profile(color='k',bloc=3)
plt.savefig(path_data_save + '4K.png', format='png', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                              5 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
df_CL5_G1 = df_clumac[df_clumac['CL_5']=='1']
df_CL5_G2 = df_clumac[df_clumac['CL_5']=='2']
df_CL5_G3 = df_clumac[df_clumac['CL_5']=='3']
df_CL5_G4 = df_clumac[df_clumac['CL_5']=='4']
df_CL5_G5 = df_clumac[df_clumac['CL_5']=='5']

#*****************************************************************************\
#C1
#*****************************************************************************\
df=df_CL5_G1
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL5_G1=np.nanmean(RH,axis=0)
temp_CL5_G1=np.nanmean(T,axis=0)
dewp_CL5_G1=np.nanmean(DP,axis=0)
mixr_CL5_G1=np.nanmean(MX,axis=0)
v_CL5_G1=np.nanmean(V,axis=0)
u_CL5_G1=np.nanmean(U,axis=0)
wsp_CL5_G1=np.sqrt(u_CL5_G1**2 + v_CL5_G1**2)
wdir_CL5_G1=np.arctan2(-u_CL5_G1, -v_CL5_G1)*(180/np.pi)
wdir_CL5_G1[(u_CL5_G1 == 0) & (v_CL5_G1 == 0)]=0
#*****************************************************************************\
#C2
#*****************************************************************************\
df=df_CL5_G2
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL5_G2=np.nanmean(RH,axis=0)
temp_CL5_G2=np.nanmean(T,axis=0)
dewp_CL5_G2=np.nanmean(DP,axis=0)
mixr_CL5_G2=np.nanmean(MX,axis=0)
v_CL5_G2=np.nanmean(V,axis=0)
u_CL5_G2=np.nanmean(U,axis=0)
wsp_CL5_G2=np.sqrt(u_CL5_G2**2 + v_CL5_G2**2)
wdir_CL5_G2=np.arctan2(-u_CL5_G2, -v_CL5_G2)*(180/np.pi)
wdir_CL5_G2[(u_CL5_G2 == 0) & (v_CL5_G2 == 0)]=0
#*****************************************************************************\
#C3
#*****************************************************************************\
df=df_CL5_G3
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL5_G3=np.nanmean(RH,axis=0)
temp_CL5_G3=np.nanmean(T,axis=0)
dewp_CL5_G3=np.nanmean(DP,axis=0)
mixr_CL5_G3=np.nanmean(MX,axis=0)
v_CL5_G3=np.nanmean(V,axis=0)
u_CL5_G3=np.nanmean(U,axis=0)
wsp_CL5_G3=np.sqrt(u_CL5_G3**2 + v_CL5_G3**2)
wdir_CL5_G3=np.arctan2(-u_CL5_G3, -v_CL5_G3)*(180/np.pi)
wdir_CL5_G3[(u_CL5_G3 == 0) & (v_CL5_G3 == 0)]=0
#*****************************************************************************\
#C4
#*****************************************************************************\
df=df_CL5_G4
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL5_G4=np.nanmean(RH,axis=0)
temp_CL5_G4=np.nanmean(T,axis=0)
dewp_CL5_G4=np.nanmean(DP,axis=0)
mixr_CL5_G4=np.nanmean(MX,axis=0)
v_CL5_G4=np.nanmean(V,axis=0)
u_CL5_G4=np.nanmean(U,axis=0)
wsp_CL5_G4=np.sqrt(u_CL5_G4**2 + v_CL5_G4**2)
wdir_CL5_G4=np.arctan2(-u_CL5_G4, -v_CL5_G4)*(180/np.pi)
wdir_CL5_G4[(u_CL5_G4 == 0) & (v_CL5_G4 == 0)]=0
#*****************************************************************************\
#C5
#*****************************************************************************\
df=df_CL5_G5
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL5_G5=np.nanmean(RH,axis=0)
temp_CL5_G5=np.nanmean(T,axis=0)
dewp_CL5_G5=np.nanmean(DP,axis=0)
mixr_CL5_G5=np.nanmean(MX,axis=0)
v_CL5_G5=np.nanmean(V,axis=0)
u_CL5_G5=np.nanmean(U,axis=0)
wsp_CL5_G5=np.sqrt(u_CL5_G5**2 + v_CL5_G5**2)
wdir_CL5_G5=np.arctan2(-u_CL5_G5, -v_CL5_G5)*(180/np.pi)
wdir_CL5_G5[(u_CL5_G5 == 0) & (v_CL5_G5 == 0)]=0


#*****************************************************************************\
#Plot
#****************************************************************************\
#C1
temperature_c=temp_CL5_G1
dewpoint_c=dewp_CL5_G1
wsp=wsp_CL5_G1*1.943844
wdir=wdir_CL5_G1

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'5K','C1')))

#C2
temperature_c=temp_CL5_G2
dewpoint_c=dewp_CL5_G2
wsp=wsp_CL5_G2*1.943844
wdir=wdir_CL5_G2

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'5K','C2')))

#C3
temperature_c=temp_CL5_G3
dewpoint_c=dewp_CL5_G3
wsp=wsp_CL5_G3*1.943844
wdir=wdir_CL5_G3

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'5K','C3')))
#C4
temperature_c=temp_CL5_G4
dewpoint_c=dewp_CL5_G4
wsp=wsp_CL5_G4*1.943844
wdir=wdir_CL5_G4

mydataG4=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'5K','C4')))


#C5
temperature_c=temp_CL5_G5
dewpoint_c=dewp_CL5_G5
wsp=wsp_CL5_G5*1.943844
wdir=wdir_CL5_G5

mydataG5=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'5K','C5')))



#Individuals
S1=SkewT.Sounding(soundingdata=mydataG1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydataG2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydataG3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C3.png', format='png', dpi=1200)
S4=SkewT.Sounding(soundingdata=mydataG4)
S4.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C4.png', format='png', dpi=1200)
S5=SkewT.Sounding(soundingdata=mydataG5)
S5.plot_skewt(color='r')
plt.savefig(path_data_save + '5K_C5.png', format='png', dpi=1200)

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=0)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=1)
S.soundingdata=S4.soundingdata
S.add_profile(color='k',bloc=1)
S.soundingdata=S5.soundingdata
S.add_profile(color='sandybrown',bloc=2)
plt.savefig(path_data_save + '5K.png', format='png', dpi=1200)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                              6 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
df_CL6_G1 = df_clumac[df_clumac['CL_6']=='1']
df_CL6_G2 = df_clumac[df_clumac['CL_6']=='2']
df_CL6_G3 = df_clumac[df_clumac['CL_6']=='3']
df_CL6_G4 = df_clumac[df_clumac['CL_6']=='4']
df_CL6_G5 = df_clumac[df_clumac['CL_6']=='5']
df_CL6_G6 = df_clumac[df_clumac['CL_6']=='6']

#*****************************************************************************\
#C1
#*****************************************************************************\
df=df_CL6_G1
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL6_G1=np.nanmean(RH,axis=0)
temp_CL6_G1=np.nanmean(T,axis=0)
dewp_CL6_G1=np.nanmean(DP,axis=0)
mixr_CL6_G1=np.nanmean(MX,axis=0)
v_CL6_G1=np.nanmean(V,axis=0)
u_CL6_G1=np.nanmean(U,axis=0)
wsp_CL6_G1=np.sqrt(u_CL6_G1**2 + v_CL6_G1**2)
wdir_CL6_G1=np.arctan2(-u_CL6_G1, -v_CL6_G1)*(180/np.pi)
wdir_CL6_G1[(u_CL6_G1 == 0) & (v_CL6_G1 == 0)]=0
#*****************************************************************************\
#C2
#*****************************************************************************\
df=df_CL6_G2
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL6_G2=np.nanmean(RH,axis=0)
temp_CL6_G2=np.nanmean(T,axis=0)
dewp_CL6_G2=np.nanmean(DP,axis=0)
mixr_CL6_G2=np.nanmean(MX,axis=0)
v_CL6_G2=np.nanmean(V,axis=0)
u_CL6_G2=np.nanmean(U,axis=0)
wsp_CL6_G2=np.sqrt(u_CL6_G2**2 + v_CL6_G2**2)
wdir_CL6_G2=np.arctan2(-u_CL6_G2, -v_CL6_G2)*(180/np.pi)
wdir_CL6_G2[(u_CL6_G2 == 0) & (v_CL6_G2 == 0)]=0
#*****************************************************************************\
#C3
#*****************************************************************************\
df=df_CL6_G3
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL6_G3=np.nanmean(RH,axis=0)
temp_CL6_G3=np.nanmean(T,axis=0)
dewp_CL6_G3=np.nanmean(DP,axis=0)
mixr_CL6_G3=np.nanmean(MX,axis=0)
v_CL6_G3=np.nanmean(V,axis=0)
u_CL6_G3=np.nanmean(U,axis=0)
wsp_CL6_G3=np.sqrt(u_CL6_G3**2 + v_CL6_G3**2)
wdir_CL6_G3=np.arctan2(-u_CL6_G3, -v_CL6_G3)*(180/np.pi)
wdir_CL6_G3[(u_CL6_G3 == 0) & (v_CL6_G3 == 0)]=0
#*****************************************************************************\
#C4
#*****************************************************************************\
df=df_CL6_G4
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL6_G4=np.nanmean(RH,axis=0)
temp_CL6_G4=np.nanmean(T,axis=0)
dewp_CL6_G4=np.nanmean(DP,axis=0)
mixr_CL6_G4=np.nanmean(MX,axis=0)
v_CL6_G4=np.nanmean(V,axis=0)
u_CL6_G4=np.nanmean(U,axis=0)
wsp_CL6_G4=np.sqrt(u_CL6_G4**2 + v_CL6_G4**2)
wdir_CL6_G4=np.arctan2(-u_CL6_G4, -v_CL6_G4)*(180/np.pi)
wdir_CL6_G4[(u_CL6_G4 == 0) & (v_CL6_G4 == 0)]=0
#*****************************************************************************\
#C5
#*****************************************************************************\
df=df_CL6_G5
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL6_G5=np.nanmean(RH,axis=0)
temp_CL6_G5=np.nanmean(T,axis=0)
dewp_CL6_G5=np.nanmean(DP,axis=0)
mixr_CL6_G5=np.nanmean(MX,axis=0)
v_CL6_G5=np.nanmean(V,axis=0)
u_CL6_G5=np.nanmean(U,axis=0)
wsp_CL6_G5=np.sqrt(u_CL6_G5**2 + v_CL6_G5**2)
wdir_CL6_G5=np.arctan2(-u_CL6_G5, -v_CL6_G5)*(180/np.pi)
wdir_CL6_G5[(u_CL6_G5 == 0) & (v_CL6_G5 == 0)]=0

#*****************************************************************************\
#C6
#*****************************************************************************\
df=df_CL6_G6
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL6_G6=np.nanmean(RH,axis=0)
temp_CL6_G6=np.nanmean(T,axis=0)
dewp_CL6_G6=np.nanmean(DP,axis=0)
mixr_CL6_G6=np.nanmean(MX,axis=0)
v_CL6_G6=np.nanmean(V,axis=0)
u_CL6_G6=np.nanmean(U,axis=0)
wsp_CL6_G6=np.sqrt(u_CL6_G6**2 + v_CL6_G6**2)
wdir_CL6_G6=np.arctan2(-u_CL6_G6, -v_CL6_G6)*(180/np.pi)
wdir_CL6_G6[(u_CL6_G6 == 0) & (v_CL6_G6 == 0)]=0
#*****************************************************************************\
#Plot
#****************************************************************************\
#C1
temperature_c=temp_CL6_G1
dewpoint_c=dewp_CL6_G1
wsp=wsp_CL6_G1*1.943844
wdir=wdir_CL6_G1

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'6K','C1')))

#C2
temperature_c=temp_CL6_G2
dewpoint_c=dewp_CL6_G2
wsp=wsp_CL6_G2*1.943844
wdir=wdir_CL6_G2

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'6K','C2')))

#C3
temperature_c=temp_CL6_G3
dewpoint_c=dewp_CL6_G3
wsp=wsp_CL6_G3*1.943844
wdir=wdir_CL6_G3

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'6K','C3')))
#C4
temperature_c=temp_CL6_G4
dewpoint_c=dewp_CL6_G4
wsp=wsp_CL6_G4*1.943844
wdir=wdir_CL6_G4

mydataG4=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'6K','C4')))


#C5
temperature_c=temp_CL6_G5
dewpoint_c=dewp_CL6_G5
wsp=wsp_CL6_G5*1.943844
wdir=wdir_CL6_G5

mydataG5=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'6K','C5')))

#C6
temperature_c=temp_CL6_G6
dewpoint_c=dewp_CL6_G6
wsp=wsp_CL6_G6*1.943844
wdir=wdir_CL6_G6

mydataG6=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'6K','C6')))


#Individuals
S1=SkewT.Sounding(soundingdata=mydataG1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + '6K_C1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydataG2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + '6K_C2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydataG3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + '6K_C3.png', format='png', dpi=1200)
S4=SkewT.Sounding(soundingdata=mydataG4)
S4.plot_skewt(color='r')
plt.savefig(path_data_save + '6K_C4.png', format='png', dpi=1200)
S5=SkewT.Sounding(soundingdata=mydataG5)
S5.plot_skewt(color='r')
plt.savefig(path_data_save + '6K_C5.png', format='png', dpi=1200)
S6=SkewT.Sounding(soundingdata=mydataG6)
S6.plot_skewt(color='r')
plt.savefig(path_data_save + '6K_C6.png', format='png', dpi=1200)

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=0)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=1)
S.soundingdata=S4.soundingdata
S.add_profile(color='k',bloc=1)
S.soundingdata=S5.soundingdata
S.add_profile(color='sandybrown',bloc=2)
S.soundingdata=S6.soundingdata
S.add_profile(color='cyan',bloc=2)


plt.savefig(path_data_save + '6K.png', format='png', dpi=1200)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                              7 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
df_CL7_G1 = df_clumac[df_clumac['CL_7']=='1']
df_CL7_G2 = df_clumac[df_clumac['CL_7']=='2']
df_CL7_G3 = df_clumac[df_clumac['CL_7']=='3']
df_CL7_G4 = df_clumac[df_clumac['CL_7']=='4']
df_CL7_G5 = df_clumac[df_clumac['CL_7']=='5']
df_CL7_G6 = df_clumac[df_clumac['CL_7']=='6']
df_CL7_G7 = df_clumac[df_clumac['CL_7']=='7']

#*****************************************************************************\
#C1
#*****************************************************************************\
df=df_CL7_G1
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL7_G1=np.nanmean(RH,axis=0)
temp_CL7_G1=np.nanmean(T,axis=0)
dewp_CL7_G1=np.nanmean(DP,axis=0)
mixr_CL7_G1=np.nanmean(MX,axis=0)
v_CL7_G1=np.nanmean(V,axis=0)
u_CL7_G1=np.nanmean(U,axis=0)
wsp_CL7_G1=np.sqrt(u_CL7_G1**2 + v_CL7_G1**2)
wdir_CL7_G1=np.arctan2(-u_CL7_G1, -v_CL7_G1)*(180/np.pi)
wdir_CL7_G1[(u_CL7_G1 == 0) & (v_CL7_G1 == 0)]=0
#*****************************************************************************\
#C2
#*****************************************************************************\
df=df_CL7_G2
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL7_G2=np.nanmean(RH,axis=0)
temp_CL7_G2=np.nanmean(T,axis=0)
dewp_CL7_G2=np.nanmean(DP,axis=0)
mixr_CL7_G2=np.nanmean(MX,axis=0)
v_CL7_G2=np.nanmean(V,axis=0)
u_CL7_G2=np.nanmean(U,axis=0)
wsp_CL7_G2=np.sqrt(u_CL7_G2**2 + v_CL7_G2**2)
wdir_CL7_G2=np.arctan2(-u_CL7_G2, -v_CL7_G2)*(180/np.pi)
wdir_CL7_G2[(u_CL7_G2 == 0) & (v_CL7_G2 == 0)]=0
#*****************************************************************************\
#C3
#*****************************************************************************\
df=df_CL7_G3
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL7_G3=np.nanmean(RH,axis=0)
temp_CL7_G3=np.nanmean(T,axis=0)
dewp_CL7_G3=np.nanmean(DP,axis=0)
mixr_CL7_G3=np.nanmean(MX,axis=0)
v_CL7_G3=np.nanmean(V,axis=0)
u_CL7_G3=np.nanmean(U,axis=0)
wsp_CL7_G3=np.sqrt(u_CL7_G3**2 + v_CL7_G3**2)
wdir_CL7_G3=np.arctan2(-u_CL7_G3, -v_CL7_G3)*(180/np.pi)
wdir_CL7_G3[(u_CL7_G3 == 0) & (v_CL7_G3 == 0)]=0
#*****************************************************************************\
#C4
#*****************************************************************************\
df=df_CL7_G4
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL7_G4=np.nanmean(RH,axis=0)
temp_CL7_G4=np.nanmean(T,axis=0)
dewp_CL7_G4=np.nanmean(DP,axis=0)
mixr_CL7_G4=np.nanmean(MX,axis=0)
v_CL7_G4=np.nanmean(V,axis=0)
u_CL7_G4=np.nanmean(U,axis=0)
wsp_CL7_G4=np.sqrt(u_CL7_G4**2 + v_CL7_G4**2)
wdir_CL7_G4=np.arctan2(-u_CL7_G4, -v_CL7_G4)*(180/np.pi)
wdir_CL7_G4[(u_CL7_G4 == 0) & (v_CL7_G4 == 0)]=0
#*****************************************************************************\
#C5
#*****************************************************************************\
df=df_CL7_G5
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL7_G5=np.nanmean(RH,axis=0)
temp_CL7_G5=np.nanmean(T,axis=0)
dewp_CL7_G5=np.nanmean(DP,axis=0)
mixr_CL7_G5=np.nanmean(MX,axis=0)
v_CL7_G5=np.nanmean(V,axis=0)
u_CL7_G5=np.nanmean(U,axis=0)
wsp_CL7_G5=np.sqrt(u_CL7_G5**2 + v_CL7_G5**2)
wdir_CL7_G5=np.arctan2(-u_CL7_G5, -v_CL7_G5)*(180/np.pi)
wdir_CL7_G5[(u_CL7_G5 == 0) & (v_CL7_G5 == 0)]=0

#*****************************************************************************\
#C6
#*****************************************************************************\
df=df_CL7_G6
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL7_G6=np.nanmean(RH,axis=0)
temp_CL7_G6=np.nanmean(T,axis=0)
dewp_CL7_G6=np.nanmean(DP,axis=0)
mixr_CL7_G6=np.nanmean(MX,axis=0)
v_CL7_G6=np.nanmean(V,axis=0)
u_CL7_G6=np.nanmean(U,axis=0)
wsp_CL7_G6=np.sqrt(u_CL7_G6**2 + v_CL7_G6**2)
wdir_CL7_G6=np.arctan2(-u_CL7_G6, -v_CL7_G6)*(180/np.pi)
wdir_CL7_G6[(u_CL7_G6 == 0) & (v_CL7_G6 == 0)]=0
#*****************************************************************************\
#C7
#*****************************************************************************\
df=df_CL7_G7
RH=np.empty([len(df),91])*np.nan
U=np.empty([len(df),91])*np.nan
V=np.empty([len(df),91])*np.nan
T=np.empty([len(df),91])*np.nan
DP=np.empty([len(df),91])*np.nan
MX=np.empty([len(df),91])*np.nan


for i in range(0,len(df)):
    RH[i,:]=np.array(df['RH'][i])
    U[i,:]=np.array(df['u'][i])
    V[i,:]=np.array(df['v'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])

#Mean profiles
rhum_CL7_G7=np.nanmean(RH,axis=0)
temp_CL7_G7=np.nanmean(T,axis=0)
dewp_CL7_G7=np.nanmean(DP,axis=0)
mixr_CL7_G7=np.nanmean(MX,axis=0)
v_CL7_G7=np.nanmean(V,axis=0)
u_CL7_G7=np.nanmean(U,axis=0)
wsp_CL7_G7=np.sqrt(u_CL7_G7**2 + v_CL7_G7**2)
wdir_CL7_G7=np.arctan2(-u_CL7_G7, -v_CL7_G7)*(180/np.pi)
wdir_CL7_G7[(u_CL7_G7 == 0) & (v_CL7_G7 == 0)]=0
#*****************************************************************************\
#Plot
#****************************************************************************\
#C1
temperature_c=temp_CL7_G1
dewpoint_c=dewp_CL7_G1
wsp=wsp_CL7_G1*1.943844
wdir=wdir_CL7_G1

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'7K','C1')))

#C2
temperature_c=temp_CL7_G2
dewpoint_c=dewp_CL7_G2
wsp=wsp_CL7_G2*1.943844
wdir=wdir_CL7_G2

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'7K','C2')))

#C3
temperature_c=temp_CL7_G3
dewpoint_c=dewp_CL7_G3
wsp=wsp_CL7_G3*1.943844
wdir=wdir_CL7_G3

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'7K','C3')))
#C4
temperature_c=temp_CL7_G4
dewpoint_c=dewp_CL7_G4
wsp=wsp_CL7_G4*1.943844
wdir=wdir_CL7_G4

mydataG4=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'7K','C4')))


#C5
temperature_c=temp_CL7_G5
dewpoint_c=dewp_CL7_G5
wsp=wsp_CL7_G5*1.943844
wdir=wdir_CL7_G5

mydataG5=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'7K','C5')))

#C6
temperature_c=temp_CL7_G6
dewpoint_c=dewp_CL7_G6
wsp=wsp_CL7_G6*1.943844
wdir=wdir_CL7_G6

mydataG6=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'7K','C6')))
#C7
temperature_c=temp_CL7_G7
dewpoint_c=dewp_CL7_G7
wsp=wsp_CL7_G7*1.943844
wdir=wdir_CL7_G7

mydataG7=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'7K','C7')))


#Individuals
S1=SkewT.Sounding(soundingdata=mydataG1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + '7K_C1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydataG2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + '7K_C2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydataG3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + '7K_C3.png', format='png', dpi=1200)
S4=SkewT.Sounding(soundingdata=mydataG4)
S4.plot_skewt(color='r')
plt.savefig(path_data_save + '7K_C4.png', format='png', dpi=1200)
S5=SkewT.Sounding(soundingdata=mydataG5)
S5.plot_skewt(color='r')
plt.savefig(path_data_save + '7K_C5.png', format='png', dpi=1200)
S6=SkewT.Sounding(soundingdata=mydataG6)
S6.plot_skewt(color='r')
plt.savefig(path_data_save + '7K_C6.png', format='png', dpi=1200)
S7=SkewT.Sounding(soundingdata=mydataG7)
S7.plot_skewt(color='r')
plt.savefig(path_data_save + '7K_C7.png', format='png', dpi=1200)



S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=0)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=1)
S.soundingdata=S4.soundingdata
S.add_profile(color='k',bloc=1)
S.soundingdata=S5.soundingdata
S.add_profile(color='sandybrown',bloc=2)
S.soundingdata=S6.soundingdata
S.add_profile(color='cyan',bloc=2)
S.soundingdata=S7.soundingdata
S.add_profile(color='magenta',bloc=2)


plt.savefig(path_data_save + '7K.png', format='png', dpi=1200)
