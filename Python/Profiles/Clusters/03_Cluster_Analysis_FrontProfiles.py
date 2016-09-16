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
#*****************************************************************************\
#Reading CSV Cluster
#*****************************************************************************\
#*****************************************************************************\
#925-850-700
df_cluster= pd.read_csv(path_data + 'All_ClusterAnalysis.csv', sep=',', parse_dates=['Date'])
#Reescribe columnas para no cambiar los outputs
CL3=np.array(df_cluster['QCL_6'])
# CL5=np.array(df_cluster['QCL_8'])
# CL6=np.array(df_cluster['QCL_9'])
# CL7=np.array(df_cluster['QCL_10'])
CL4=np.array(df_cluster['Cluster'])
dist_clu=np.array(df_cluster['Distance'])

path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/Profiles/Fronts'

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
#                          Dataframes 1995-2010                               \
#*****************************************************************************\
#*****************************************************************************\
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
# timestamp = [datenum_to_datetime(t) for t in timesd]
# time_my = np.array(timestamp)
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
'CL_4':CL4,
'dist_clu':dist_clu}

df_clumac= pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
df_clumac=df_clumac.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')
#*****************************************************************************\
#Reading FRONTS
#*****************************************************************************\
path_data_csv=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
df_cfront= pd.read_csv(path_data_csv + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_cfront= df_cfront.set_index('Date')


df_wfront= pd.read_csv(path_data_csv + 'df_wfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_wfront= df_wfront.set_index('Date')

#*****************************************************************************\
#Concadanate dataframes
df_my_wfro=pd.concat([df_clumac, df_wfront],axis=1)
df_my_cfro=pd.concat([df_clumac, df_cfront],axis=1)

#Only Fronts
dfmycfro=df_my_cfro[np.isfinite(df_my_cfro['Dist CFront'])]
dfmywfro=df_my_wfro[np.isfinite(df_my_wfro['Dist WFront'])]

df_fronts=pd.concat([df_my_cfro, df_my_wfro],axis=1)
dfmy_fronts=pd.concat([df_clumac, df_fronts],axis=1)

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                              Profiles 4 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
df_CL4_G1 = dfmywfro[dfmywfro['CL_4']==1]
df_CL4_G2 = dfmywfro[dfmywfro['CL_4']==2]
df_CL4_G3 = dfmywfro[dfmywfro['CL_4']==3]
df_CL4_G4 = dfmywfro[dfmywfro['CL_4']==4]
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



# #Individuals
# S1=SkewT.Sounding(soundingdata=mydataG1)
# S1.plot_skewt(color='r')
# #plt.savefig(path_data_save + '4K_C1.png', format='png', dpi=1200)
# S2=SkewT.Sounding(soundingdata=mydataG2)
# S2.plot_skewt(color='r')
# #plt.savefig(path_data_save + '4K_C2.png', format='png', dpi=1200)
# S3=SkewT.Sounding(soundingdata=mydataG3)
# S3.plot_skewt(color='r')
# #plt.savefig(path_data_save + '4K_C3.png', format='png', dpi=1200)
# S4=SkewT.Sounding(soundingdata=mydataG4)
# S4.plot_skewt(color='r')
# #plt.savefig(path_data_save + '4K_C4.png', format='png', dpi=1200)

# S=SkewT.Sounding(soundingdata=mydataG1)
# S.make_skewt_axes()
# S.add_profile(color='r',bloc=0)
# S.soundingdata=S2.soundingdata
# S.add_profile(color='b',bloc=0)
# S.soundingdata=S3.soundingdata
# S.add_profile(color='g',bloc=1)
# S.soundingdata=S4.soundingdata
# S.add_profile(color='k',bloc=3)
# #plt.savefig(path_data_save + '4K.png', format='png', dpi=1200)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                              Closest WFront Profiles 4 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
height_m=hlev_yotc
pressure_pa=plev_yotc
n_cases=3
#*****************************************************************************\
#C1
#*****************************************************************************\

df=df_CL4_G1.sort('dist_clu', ascending=True)
RH=np.empty([n_cases,91])*np.nan
WSP=np.empty([n_cases,91])*np.nan
DIR=np.empty([n_cases,91])*np.nan
T=np.empty([n_cases,91])*np.nan
DP=np.empty([n_cases,91])*np.nan
MX=np.empty([n_cases,91])*np.nan
Dist=np.empty([n_cases])*np.nan
PosFront=np.empty([n_cases])*np.nan

for i in range(0,3):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist[i]=np.array(df['dist_clu'][i])
    PosFront[i]=np.array(df['Dist WFront'][i])



temperature_c=T[0,:]
dewpoint_c=DP[0,:]
wsp=WSP[0,:]*1.943844
wdir=DIR[0,:]

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[1,:]
dewpoint_c=DP[1,:]
wsp=WSP[1,:]*1.943844
wdir=DIR[1,:]

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[2,:]
dewpoint_c=DP[2,:]
wsp=WSP[2,:]*1.943844
wdir=DIR[2,:]

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))


S1=SkewT.Sounding(soundingdata=mydataG1)
#S1.plot_skewt(color='r')
S2=SkewT.Sounding(soundingdata=mydataG2)
#S2.plot_skewt(color='r')
S3=SkewT.Sounding(soundingdata=mydataG3)
#S3.plot_skewt(color='r')

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
#plt.savefig(path_data_save + '4K_C1_3C_WF.png', format='png', dpi=1200)


#*****************************************************************************\
#C2
#*****************************************************************************\

df=df_CL4_G2.sort('dist_clu', ascending=True)
RH=np.empty([n_cases,91])*np.nan
WSP=np.empty([n_cases,91])*np.nan
DIR=np.empty([n_cases,91])*np.nan
T=np.empty([n_cases,91])*np.nan
DP=np.empty([n_cases,91])*np.nan
MX=np.empty([n_cases,91])*np.nan
Dist=np.empty([n_cases])*np.nan
PosFront=np.empty([n_cases])*np.nan

for i in range(0,3):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist[i]=np.array(df['dist_clu'][i])
    PosFront[i]=np.array(df['Dist WFront'][i])


temperature_c=T[0,:]
dewpoint_c=DP[0,:]
wsp=WSP[0,:]*1.943844
wdir=DIR[0,:]

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[1,:]
dewpoint_c=DP[1,:]
wsp=WSP[1,:]*1.943844
wdir=DIR[1,:]

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[2,:]
dewpoint_c=DP[2,:]
wsp=WSP[2,:]*1.943844
wdir=DIR[2,:]

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))


S1=SkewT.Sounding(soundingdata=mydataG1)
#S1.plot_skewt(color='r')
S2=SkewT.Sounding(soundingdata=mydataG2)
#S2.plot_skewt(color='r')
S3=SkewT.Sounding(soundingdata=mydataG3)
#S3.plot_skewt(color='r')

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
#plt.savefig(path_data_save + '4K_C2_3C_WF.png', format='png', dpi=1200)
#*****************************************************************************\
#C3
#*****************************************************************************\


df=df_CL4_G3.sort('dist_clu', ascending=True)
RH=np.empty([n_cases,91])*np.nan
WSP=np.empty([n_cases,91])*np.nan
DIR=np.empty([n_cases,91])*np.nan
T=np.empty([n_cases,91])*np.nan
DP=np.empty([n_cases,91])*np.nan
MX=np.empty([n_cases,91])*np.nan
Dist=np.empty([n_cases])*np.nan
PosFront=np.empty([n_cases])*np.nan

for i in range(0,3):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist[i]=np.array(df['dist_clu'][i])
    PosFront[i]=np.array(df['Dist WFront'][i])


temperature_c=T[0,:]
dewpoint_c=DP[0,:]
wsp=WSP[0,:]*1.943844
wdir=DIR[0,:]

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[1,:]
dewpoint_c=DP[1,:]
wsp=WSP[1,:]*1.943844
wdir=DIR[1,:]

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[2,:]
dewpoint_c=DP[2,:]
wsp=WSP[2,:]*1.943844
wdir=DIR[2,:]

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))


S1=SkewT.Sounding(soundingdata=mydataG1)
#S1.plot_skewt(color='r')
S2=SkewT.Sounding(soundingdata=mydataG2)
#S2.plot_skewt(color='r')
S3=SkewT.Sounding(soundingdata=mydataG3)
#S3.plot_skewt(color='r')

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
#plt.savefig(path_data_save + '4K_C3_3C_WF.png', format='png', dpi=1200)

#*****************************************************************************\
#C4
#*****************************************************************************\

df=df_CL4_G4.sort('dist_clu', ascending=True)
RH=np.empty([n_cases,91])*np.nan
WSP=np.empty([n_cases,91])*np.nan
DIR=np.empty([n_cases,91])*np.nan
T=np.empty([n_cases,91])*np.nan
DP=np.empty([n_cases,91])*np.nan
MX=np.empty([n_cases,91])*np.nan
Dist=np.empty([n_cases])*np.nan
PosFront=np.empty([n_cases])*np.nan

for i in range(0,3):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist[i]=np.array(df['dist_clu'][i])
    PosFront[i]=np.array(df['Dist WFront'][i])


temperature_c=T[0,:]
dewpoint_c=DP[0,:]
wsp=WSP[0,:]*1.943844
wdir=DIR[0,:]

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[1,:]
dewpoint_c=DP[1,:]
wsp=WSP[1,:]*1.943844
wdir=DIR[1,:]

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[2,:]
dewpoint_c=DP[2,:]
wsp=WSP[2,:]*1.943844
wdir=DIR[2,:]

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))


S1=SkewT.Sounding(soundingdata=mydataG1)
#S1.plot_skewt(color='r')
S2=SkewT.Sounding(soundingdata=mydataG2)
#S2.plot_skewt(color='r')
S3=SkewT.Sounding(soundingdata=mydataG3)
#S3.plot_skewt(color='r')

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
#plt.savefig(path_data_save + '4K_C4_3C_WF.png', format='png', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                              Closest CFront Profiles 4 Cluster
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\

df_CL4_G1 = dfmycfro[dfmycfro['CL_4']==1]
df_CL4_G2 = dfmycfro[dfmycfro['CL_4']==2]
df_CL4_G3 = dfmycfro[dfmycfro['CL_4']==3]
df_CL4_G4 = dfmycfro[dfmycfro['CL_4']==4]

#*****************************************************************************\
#C1
#*****************************************************************************\

df=df_CL4_G1.sort('dist_clu', ascending=True)
RH=np.empty([n_cases,91])*np.nan
WSP=np.empty([n_cases,91])*np.nan
DIR=np.empty([n_cases,91])*np.nan
T=np.empty([n_cases,91])*np.nan
DP=np.empty([n_cases,91])*np.nan
MX=np.empty([n_cases,91])*np.nan
Dist=np.empty([n_cases])*np.nan
PosFront=np.empty([n_cases])*np.nan

for i in range(0,3):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist[i]=np.array(df['dist_clu'][i])
    PosFront[i]=np.array(df['Dist CFront'][i])



temperature_c=T[0,:]
dewpoint_c=DP[0,:]
wsp=WSP[0,:]*1.943844
wdir=DIR[0,:]

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[1,:]
dewpoint_c=DP[1,:]
wsp=WSP[1,:]*1.943844
wdir=DIR[1,:]

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[2,:]
dewpoint_c=DP[2,:]
wsp=WSP[2,:]*1.943844
wdir=DIR[2,:]

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))


S1=SkewT.Sounding(soundingdata=mydataG1)
#S1.plot_skewt(color='r')
S2=SkewT.Sounding(soundingdata=mydataG2)
#S2.plot_skewt(color='r')
S3=SkewT.Sounding(soundingdata=mydataG3)
#S3.plot_skewt(color='r')

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
#plt.savefig(path_data_save + '4K_C1_3C_CF.png', format='png', dpi=1200)


#*****************************************************************************\
#C2
#*****************************************************************************\

df=df_CL4_G2.sort('dist_clu', ascending=True)
RH=np.empty([n_cases,91])*np.nan
WSP=np.empty([n_cases,91])*np.nan
DIR=np.empty([n_cases,91])*np.nan
T=np.empty([n_cases,91])*np.nan
DP=np.empty([n_cases,91])*np.nan
MX=np.empty([n_cases,91])*np.nan
Dist=np.empty([n_cases])*np.nan
PosFront=np.empty([n_cases])*np.nan

for i in range(0,3):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist[i]=np.array(df['dist_clu'][i])
    PosFront[i]=np.array(df['Dist CFront'][i])


temperature_c=T[0,:]
dewpoint_c=DP[0,:]
wsp=WSP[0,:]*1.943844
wdir=DIR[0,:]

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[1,:]
dewpoint_c=DP[1,:]
wsp=WSP[1,:]*1.943844
wdir=DIR[1,:]

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[2,:]
dewpoint_c=DP[2,:]
wsp=WSP[2,:]*1.943844
wdir=DIR[2,:]

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))


S1=SkewT.Sounding(soundingdata=mydataG1)
#S1.plot_skewt(color='r')
S2=SkewT.Sounding(soundingdata=mydataG2)
#S2.plot_skewt(color='r')
S3=SkewT.Sounding(soundingdata=mydataG3)
#S3.plot_skewt(color='r')

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
#plt.savefig(path_data_save + '4K_C2_3C_CF.png', format='png', dpi=1200)
#*****************************************************************************\
#C3
#*****************************************************************************\


df=df_CL4_G3.sort('dist_clu', ascending=True)
RH=np.empty([n_cases,91])*np.nan
WSP=np.empty([n_cases,91])*np.nan
DIR=np.empty([n_cases,91])*np.nan
T=np.empty([n_cases,91])*np.nan
DP=np.empty([n_cases,91])*np.nan
MX=np.empty([n_cases,91])*np.nan
Dist=np.empty([n_cases])*np.nan
PosFront=np.empty([n_cases])*np.nan

for i in range(0,3):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist[i]=np.array(df['dist_clu'][i])
    PosFront[i]=np.array(df['Dist CFront'][i])


temperature_c=T[0,:]
dewpoint_c=DP[0,:]
wsp=WSP[0,:]*1.943844
wdir=DIR[0,:]

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[1,:]
dewpoint_c=DP[1,:]
wsp=WSP[1,:]*1.943844
wdir=DIR[1,:]

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[2,:]
dewpoint_c=DP[2,:]
wsp=WSP[2,:]*1.943844
wdir=DIR[2,:]

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))


S1=SkewT.Sounding(soundingdata=mydataG1)
#S1.plot_skewt(color='r')
S2=SkewT.Sounding(soundingdata=mydataG2)
#S2.plot_skewt(color='r')
S3=SkewT.Sounding(soundingdata=mydataG3)
#S3.plot_skewt(color='r')

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
#plt.savefig(path_data_save + '4K_C3_3C_CF.png', format='png', dpi=1200)

#*****************************************************************************\
#C4
#*****************************************************************************\

df=df_CL4_G4.sort('dist_clu', ascending=True)
RH=np.empty([n_cases,91])*np.nan
WSP=np.empty([n_cases,91])*np.nan
DIR=np.empty([n_cases,91])*np.nan
T=np.empty([n_cases,91])*np.nan
DP=np.empty([n_cases,91])*np.nan
MX=np.empty([n_cases,91])*np.nan
Dist=np.empty([n_cases])*np.nan
PosFront=np.empty([n_cases])*np.nan

for i in range(0,3):
    RH[i,:]=np.array(df['RH'][i])
    WSP[i,:]=np.array(df['wsp'][i])
    DIR[i,:]=np.array(df['wdir'][i])
    T[i,:]=np.array(df['temp'][i])
    DP[i,:]=np.array(df['dewp'][i])
    MX[i,:]=np.array(df['mixr'][i])
    Dist[i]=np.array(df['dist_clu'][i])
    PosFront[i]=np.array(df['Dist CFront'][i])


temperature_c=T[0,:]
dewpoint_c=DP[0,:]
wsp=WSP[0,:]*1.943844
wdir=DIR[0,:]

mydataG1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[1,:]
dewpoint_c=DP[1,:]
wsp=WSP[1,:]*1.943844
wdir=DIR[1,:]

mydataG2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))

temperature_c=T[2,:]
dewpoint_c=DP[2,:]
wsp=WSP[2,:]*1.943844
wdir=DIR[2,:]

mydataG3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'4K','CL1')))


S1=SkewT.Sounding(soundingdata=mydataG1)
#S1.plot_skewt(color='r')
S2=SkewT.Sounding(soundingdata=mydataG2)
#S2.plot_skewt(color='r')
S3=SkewT.Sounding(soundingdata=mydataG3)
#S3.plot_skewt(color='r')

S=SkewT.Sounding(soundingdata=mydataG1)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
#plt.savefig(path_data_save + '4K_C4_3C_CF.png', format='png', dpi=1200)
show()


