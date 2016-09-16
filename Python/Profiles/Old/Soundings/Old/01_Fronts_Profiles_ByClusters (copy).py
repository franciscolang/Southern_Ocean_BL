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
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'

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
'mixr':mr_list,#g/kg
'wsp':wsp_list, #m/s
'wdir':wdir_list} #deg

df_my= pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
df_my=df_my.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')
df_my=df_my.reindex(date_index_all)
df_my.index.name = 'Date'

#*****************************************************************************\
#Reading FRONTS
#*****************************************************************************\
df_cfront= pd.read_csv(path_data + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_cfront= df_cfront.set_index('Date')


df_wfront= pd.read_csv(path_data + 'df_wfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_wfront= df_wfront.set_index('Date')

#*****************************************************************************\
#Concadanate dataframes
df_my_wfro=pd.concat([df_my, df_wfront],axis=1)
df_my_cfro=pd.concat([df_my, df_cfront],axis=1)
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Cold Fronts
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
dfmycfro=df_my_cfro[np.isfinite(df_my_cfro['Dist CFront'])]

path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts_ok/profiles/'
#*****************************************************************************\
# Post Front
#*****************************************************************************\
df_posf = dfmycfro[dfmycfro['Dist CFront']>0]

df=df_posf
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
rhum_posf=np.nanmean(RH,axis=0)
temp_posf=np.nanmean(T,axis=0)
dewp_posf=np.nanmean(DP,axis=0)
mixr_posf=np.nanmean(MX,axis=0)
v_posf=np.nanmean(V,axis=0)
u_posf=np.nanmean(U,axis=0)
wsp_posf=np.sqrt(u_posf**2 + v_posf**2)
wdir_posf=np.arctan2(-u_posf, -v_posf)*(180/np.pi)
wdir_posf[(u_posf == 0) & (v_posf == 0)]=0

#*****************************************************************************\
# Pre Front
#*****************************************************************************\
df_pref = dfmycfro[dfmycfro['Dist CFront']<0]

df=df_pref
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
rhum_pref=np.nanmean(RH,axis=0)
temp_pref=np.nanmean(T,axis=0)
dewp_pref=np.nanmean(DP,axis=0)
mixr_pref=np.nanmean(MX,axis=0)
v_pref=np.nanmean(V,axis=0)
u_pref=np.nanmean(U,axis=0)
wsp_pref=np.sqrt(u_pref**2 + v_pref**2)
wdir_pref=np.arctan2(-u_pref, -v_pref)*(180/np.pi)
wdir_pref[(u_pref == 0) & (v_pref == 0)]=0

#*****************************************************************************\
#Plot
#*****************************************************************************\
height_m=hlev_yotc
pressure_pa=plev_yotc
#Post
temperature_c=temp_posf
dewpoint_c=dewp_posf
wsp=wsp_posf*1.943844
wdir=wdir_posf

mydataposf=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Post Front','')))

#Pre
temperature_c=temp_pref
dewpoint_c=dewp_pref
wsp=wsp_pref*1.943844
wdir=wdir_pref

mydatapref=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Pre Front','')))


S1=SkewT.Sounding(soundingdata=mydataposf)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_postcoldfront.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydatapref)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_precoldfront.png', format='png', dpi=1200)


S=SkewT.Sounding(soundingdata=mydataposf)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
plt.savefig(path_data_save + 'sounding_prepostcold.png', format='png', dpi=1200)


#*****************************************************************************\
# Post Front every 5 deg
#*****************************************************************************\
df_posf1 = dfmycfro[(dfmycfro['Dist CFront']>0) & (dfmycfro['Dist CFront']<=5)]
df_posf2 = dfmycfro[(dfmycfro['Dist CFront']>5) & (dfmycfro['Dist CFront']<=10)]
df_posf3 = dfmycfro[(dfmycfro['Dist CFront']>10) & (dfmycfro['Dist CFront']<=15)]
#*****************************************************************************\

df=df_posf1
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
rhum_posf1=np.nanmean(RH,axis=0)
temp_posf1=np.nanmean(T,axis=0)
dewp_posf1=np.nanmean(DP,axis=0)
mixr_posf1=np.nanmean(MX,axis=0)
v_posf1=np.nanmean(V,axis=0)
u_posf1=np.nanmean(U,axis=0)
wsp_posf1=np.sqrt(u_posf1**2 + v_posf1**2)
wdir_posf1=np.arctan2(-u_posf1, -v_posf1)*(180/np.pi)
wdir_posf1[(u_posf1 == 0) & (v_posf1 == 0)]=0

#*****************************************************************************\

df=df_posf2
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
rhum_posf2=np.nanmean(RH,axis=0)
temp_posf2=np.nanmean(T,axis=0)
dewp_posf2=np.nanmean(DP,axis=0)
mixr_posf2=np.nanmean(MX,axis=0)
v_posf2=np.nanmean(V,axis=0)
u_posf2=np.nanmean(U,axis=0)
wsp_posf2=np.sqrt(u_posf2**2 + v_posf2**2)
wdir_posf2=np.arctan2(-u_posf2, -v_posf2)*(180/np.pi)
wdir_posf2[(u_posf2 == 0) & (v_posf2 == 0)]=0
#*****************************************************************************\

df=df_posf3
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
rhum_posf3=np.nanmean(RH,axis=0)
temp_posf3=np.nanmean(T,axis=0)
dewp_posf3=np.nanmean(DP,axis=0)
mixr_posf3=np.nanmean(MX,axis=0)
v_posf3=np.nanmean(V,axis=0)
u_posf3=np.nanmean(U,axis=0)
wsp_posf3=np.sqrt(u_posf3**2 + v_posf3**2)
wdir_posf3=np.arctan2(-u_posf3, -v_posf3)*(180/np.pi)
wdir_posf3[(u_posf3 == 0) & (v_posf3 == 0)]=0
#*****************************************************************************\
temperature_c=temp_posf1
dewpoint_c=dewp_posf1
wsp=wsp_posf1*1.943844
wdir=wdir_posf1

mydataposf1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Post Front','0-5 (deg)')))

temperature_c=temp_posf2
dewpoint_c=dewp_posf2
wsp=wsp_posf2*1.943844
wdir=wdir_posf2

mydataposf2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Post Front','5-10 (deg)')))

temperature_c=temp_posf3
dewpoint_c=dewp_posf3
wsp=wsp_posf3*1.943844
wdir=wdir_posf3

mydataposf3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Post Front','10-15 (deg)')))

S1=SkewT.Sounding(soundingdata=mydataposf1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_postcoldfront1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydataposf2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_postcoldfront2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydataposf3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_postcoldfront3.png', format='png', dpi=1200)


S=SkewT.Sounding(soundingdata=mydataposf)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
plt.savefig(path_data_save + 'sounding_postcold_3d.png', format='png', dpi=1200)

#*****************************************************************************\
# Pre Front every 5 deg
#*****************************************************************************\
df_pref1 = dfmycfro[(dfmycfro['Dist CFront']<0) & (dfmycfro['Dist CFront']>=-5)]
df_pref2 = dfmycfro[(dfmycfro['Dist CFront']<-5) & (dfmycfro['Dist CFront']>=-10)]
df_pref3 = dfmycfro[(dfmycfro['Dist CFront']<-10) & (dfmycfro['Dist CFront']>=-15)]
#*****************************************************************************\

df=df_pref1
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
rhum_pref1=np.nanmean(RH,axis=0)
temp_pref1=np.nanmean(T,axis=0)
dewp_pref1=np.nanmean(DP,axis=0)
mixr_pref1=np.nanmean(MX,axis=0)
v_pref1=np.nanmean(V,axis=0)
u_pref1=np.nanmean(U,axis=0)
wsp_pref1=np.sqrt(u_pref1**2 + v_pref1**2)
wdir_pref1=np.arctan2(-u_pref1, -v_pref1)*(180/np.pi)
wdir_pref1[(u_pref1 == 0) & (v_pref1 == 0)]=0

#*****************************************************************************\

df=df_pref2
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
rhum_pref2=np.nanmean(RH,axis=0)
temp_pref2=np.nanmean(T,axis=0)
dewp_pref2=np.nanmean(DP,axis=0)
mixr_pref2=np.nanmean(MX,axis=0)
v_pref2=np.nanmean(V,axis=0)
u_pref2=np.nanmean(U,axis=0)
wsp_pref2=np.sqrt(u_pref2**2 + v_pref2**2)
wdir_pref2=np.arctan2(-u_pref2, -v_pref2)*(180/np.pi)
wdir_pref2[(u_pref2 == 0) & (v_pref2 == 0)]=0
#*****************************************************************************\

df=df_pref3
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
rhum_pref3=np.nanmean(RH,axis=0)
temp_pref3=np.nanmean(T,axis=0)
dewp_pref3=np.nanmean(DP,axis=0)
mixr_pref3=np.nanmean(MX,axis=0)
v_pref3=np.nanmean(V,axis=0)
u_pref3=np.nanmean(U,axis=0)
wsp_pref3=np.sqrt(u_pref3**2 + v_pref3**2)
wdir_pref3=np.arctan2(-u_pref3, -v_pref3)*(180/np.pi)
wdir_pref3[(u_pref3 == 0) & (v_pref3 == 0)]=0
#*****************************************************************************\
temperature_c=temp_pref1
dewpoint_c=dewp_pref1
wsp=wsp_pref1*1.943844
wdir=wdir_pref1

mydatapref1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Pre Front','0-5 (deg)')))

temperature_c=temp_pref2
dewpoint_c=dewp_pref2
wsp=wsp_pref2*1.943844
wdir=wdir_pref2

mydatapref2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Pre Front','5-10 (deg)')))

temperature_c=temp_pref3
dewpoint_c=dewp_pref3
wsp=wsp_pref3*1.943844
wdir=wdir_pref3

mydatapref3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Pre Front','10-15 (deg)')))

S1=SkewT.Sounding(soundingdata=mydatapref1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_precoldfront1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydatapref2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_precoldfront2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydatapref3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_precoldfront3.png', format='png', dpi=1200)


S=SkewT.Sounding(soundingdata=mydatapref)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
plt.savefig(path_data_save + 'sounding_precold_3d.png', format='png', dpi=1200)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                           Warm Fronts
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
dfmywfro=df_my_wfro[np.isfinite(df_my_wfro['Dist WFront'])]

path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts_ok/profiles/'
#*****************************************************************************\
# Post Front
#*****************************************************************************\
df_posf = dfmywfro[dfmywfro['Dist WFront']>0]

df=df_posf
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
rhum_posf=np.nanmean(RH,axis=0)
temp_posf=np.nanmean(T,axis=0)
dewp_posf=np.nanmean(DP,axis=0)
mixr_posf=np.nanmean(MX,axis=0)
v_posf=np.nanmean(V,axis=0)
u_posf=np.nanmean(U,axis=0)
wsp_posf=np.sqrt(u_posf**2 + v_posf**2)
wdir_posf=np.arctan2(-u_posf, -v_posf)*(180/np.pi)
wdir_posf[(u_posf == 0) & (v_posf == 0)]=0

#*****************************************************************************\
# Pre Front
#*****************************************************************************\
df_pref = dfmywfro[dfmywfro['Dist WFront']<0]

df=df_pref
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
rhum_pref=np.nanmean(RH,axis=0)
temp_pref=np.nanmean(T,axis=0)
dewp_pref=np.nanmean(DP,axis=0)
mixr_pref=np.nanmean(MX,axis=0)
v_pref=np.nanmean(V,axis=0)
u_pref=np.nanmean(U,axis=0)
wsp_pref=np.sqrt(u_pref**2 + v_pref**2)
wdir_pref=np.arctan2(-u_pref, -v_pref)*(180/np.pi)
wdir_pref[(u_pref == 0) & (v_pref == 0)]=0

#*****************************************************************************\
#Plot
#*****************************************************************************\
height_m=hlev_yotc
pressure_pa=plev_yotc
#Post
temperature_c=temp_posf
dewpoint_c=dewp_posf
wsp=wsp_posf*1.943844
wdir=wdir_posf

mydataposf=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Post Front','')))

#Pre
temperature_c=temp_pref
dewpoint_c=dewp_pref
wsp=wsp_pref*1.943844
wdir=wdir_pref

mydatapref=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Pre Front','')))


S1=SkewT.Sounding(soundingdata=mydataposf)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_postwarmfront.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydatapref)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_prewarmfront.png', format='png', dpi=1200)


S=SkewT.Sounding(soundingdata=mydataposf)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
plt.savefig(path_data_save + 'sounding_prepostwarm.png', format='png', dpi=1200)


#*****************************************************************************\
# Post Front every 5 deg
#*****************************************************************************\
df_posf1 = dfmywfro[(dfmywfro['Dist WFront']>0) & (dfmywfro['Dist WFront']<=5)]
df_posf2 = dfmywfro[(dfmywfro['Dist WFront']>5) & (dfmywfro['Dist WFront']<=10)]
df_posf3 = dfmywfro[(dfmywfro['Dist WFront']>10) & (dfmywfro['Dist WFront']<=15)]
#*****************************************************************************\

df=df_posf1
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
rhum_posf1=np.nanmean(RH,axis=0)
temp_posf1=np.nanmean(T,axis=0)
dewp_posf1=np.nanmean(DP,axis=0)
mixr_posf1=np.nanmean(MX,axis=0)
v_posf1=np.nanmean(V,axis=0)
u_posf1=np.nanmean(U,axis=0)
wsp_posf1=np.sqrt(u_posf1**2 + v_posf1**2)
wdir_posf1=np.arctan2(-u_posf1, -v_posf1)*(180/np.pi)
wdir_posf1[(u_posf1 == 0) & (v_posf1 == 0)]=0

#*****************************************************************************\

df=df_posf2
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
rhum_posf2=np.nanmean(RH,axis=0)
temp_posf2=np.nanmean(T,axis=0)
dewp_posf2=np.nanmean(DP,axis=0)
mixr_posf2=np.nanmean(MX,axis=0)
v_posf2=np.nanmean(V,axis=0)
u_posf2=np.nanmean(U,axis=0)
wsp_posf2=np.sqrt(u_posf2**2 + v_posf2**2)
wdir_posf2=np.arctan2(-u_posf2, -v_posf2)*(180/np.pi)
wdir_posf2[(u_posf2 == 0) & (v_posf2 == 0)]=0
#*****************************************************************************\

df=df_posf3
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
rhum_posf3=np.nanmean(RH,axis=0)
temp_posf3=np.nanmean(T,axis=0)
dewp_posf3=np.nanmean(DP,axis=0)
mixr_posf3=np.nanmean(MX,axis=0)
v_posf3=np.nanmean(V,axis=0)
u_posf3=np.nanmean(U,axis=0)
wsp_posf3=np.sqrt(u_posf3**2 + v_posf3**2)
wdir_posf3=np.arctan2(-u_posf3, -v_posf3)*(180/np.pi)
wdir_posf3[(u_posf3 == 0) & (v_posf3 == 0)]=0
#*****************************************************************************\
temperature_c=temp_posf1
dewpoint_c=dewp_posf1
wsp=wsp_posf1*1.943844
wdir=wdir_posf1

mydataposf1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Post Front','0-5 (deg)')))

temperature_c=temp_posf2
dewpoint_c=dewp_posf2
wsp=wsp_posf2*1.943844
wdir=wdir_posf2

mydataposf2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Post Front','5-10 (deg)')))

temperature_c=temp_posf3
dewpoint_c=dewp_posf3
wsp=wsp_posf3*1.943844
wdir=wdir_posf3

mydataposf3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Post Front','10-15 (deg)')))

S1=SkewT.Sounding(soundingdata=mydataposf1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_postwarmfront1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydataposf2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_postwarmfront2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydataposf3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_postwarmfront3.png', format='png', dpi=1200)


S=SkewT.Sounding(soundingdata=mydataposf)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
plt.savefig(path_data_save + 'sounding_postwarm_3d.png', format='png', dpi=1200)

#*****************************************************************************\
# Pre Front every 5 deg
#*****************************************************************************\
df_pref1 = dfmywfro[(dfmywfro['Dist WFront']<0) & (dfmywfro['Dist WFront']>=-5)]
df_pref2 = dfmywfro[(dfmywfro['Dist WFront']<-5) & (dfmywfro['Dist WFront']>=-10)]
df_pref3 = dfmywfro[(dfmywfro['Dist WFront']<-10) & (dfmywfro['Dist WFront']>=-15)]
#*****************************************************************************\

df=df_pref1
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
rhum_pref1=np.nanmean(RH,axis=0)
temp_pref1=np.nanmean(T,axis=0)
dewp_pref1=np.nanmean(DP,axis=0)
mixr_pref1=np.nanmean(MX,axis=0)
v_pref1=np.nanmean(V,axis=0)
u_pref1=np.nanmean(U,axis=0)
wsp_pref1=np.sqrt(u_pref1**2 + v_pref1**2)
wdir_pref1=np.arctan2(-u_pref1, -v_pref1)*(180/np.pi)
wdir_pref1[(u_pref1 == 0) & (v_pref1 == 0)]=0

#*****************************************************************************\

df=df_pref2
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
rhum_pref2=np.nanmean(RH,axis=0)
temp_pref2=np.nanmean(T,axis=0)
dewp_pref2=np.nanmean(DP,axis=0)
mixr_pref2=np.nanmean(MX,axis=0)
v_pref2=np.nanmean(V,axis=0)
u_pref2=np.nanmean(U,axis=0)
wsp_pref2=np.sqrt(u_pref2**2 + v_pref2**2)
wdir_pref2=np.arctan2(-u_pref2, -v_pref2)*(180/np.pi)
wdir_pref2[(u_pref2 == 0) & (v_pref2 == 0)]=0
#*****************************************************************************\

df=df_pref3
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
rhum_pref3=np.nanmean(RH,axis=0)
temp_pref3=np.nanmean(T,axis=0)
dewp_pref3=np.nanmean(DP,axis=0)
mixr_pref3=np.nanmean(MX,axis=0)
v_pref3=np.nanmean(V,axis=0)
u_pref3=np.nanmean(U,axis=0)
wsp_pref3=np.sqrt(u_pref3**2 + v_pref3**2)
wdir_pref3=np.arctan2(-u_pref3, -v_pref3)*(180/np.pi)
wdir_pref3[(u_pref3 == 0) & (v_pref3 == 0)]=0
#*****************************************************************************\
temperature_c=temp_pref1
dewpoint_c=dewp_pref1
wsp=wsp_pref1*1.943844
wdir=wdir_pref1

mydatapref1=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Pre Front','0-5 (deg)')))

temperature_c=temp_pref2
dewpoint_c=dewp_pref2
wsp=wsp_pref2*1.943844
wdir=wdir_pref2

mydatapref2=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Pre Front','5-10 (deg)')))

temperature_c=temp_pref3
dewpoint_c=dewp_pref3
wsp=wsp_pref3*1.943844
wdir=wdir_pref3

mydatapref3=dict(zip(('hght','pres','temp','dwpt','drct','sknt','StationNumber','SoundingDate'),(height_m,pressure_pa,temperature_c,dewpoint_c,wdir,wsp,'Pre Front','10-15 (deg)')))

S1=SkewT.Sounding(soundingdata=mydatapref1)
S1.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_prewarmfront1.png', format='png', dpi=1200)
S2=SkewT.Sounding(soundingdata=mydatapref2)
S2.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_prewarmfront2.png', format='png', dpi=1200)
S3=SkewT.Sounding(soundingdata=mydatapref3)
S3.plot_skewt(color='r')
plt.savefig(path_data_save + 'sounding_prewarmfront3.png', format='png', dpi=1200)


S=SkewT.Sounding(soundingdata=mydatapref)
S.make_skewt_axes()
S.add_profile(color='r',bloc=0)
S.soundingdata=S2.soundingdata
S.add_profile(color='b',bloc=1)
S.soundingdata=S3.soundingdata
S.add_profile(color='g',bloc=2)
plt.savefig(path_data_save + 'sounding_prewarm_3d.png', format='png', dpi=1200)



