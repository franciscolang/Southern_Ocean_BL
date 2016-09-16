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
from pylab import plot,show, grid, xlabel, ylabel, xlim, ylim


base_dir = os.path.expanduser('~')
Yfin=1995

#*****************************************************************************\
#Default Info
mac = {'name': 'Macquarie Island, Australia', 'lat': -54.62, 'lon': 158.85}
lat_mac = mac['lat']
lon_mac = mac['lon']

ptemp_thold_main=0.010           # K/m
ptemp_thold_sec=0.005            # K/m
shear_thold=0.015
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                            ERA-I Data
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_erai=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/ERAI/'
matb1= sio.loadmat(path_data_erai+'ERAImac_1995.mat')
temp_erai=matb1['temp'][:]
rh_erai=matb1['rh'][:]
q_erai=matb1['q'][:]
u_erai=matb1['u'][:]
v_erai=matb1['v'][:]
time_erai= matb1['time2'][:]

#Pressure Levels
pres_erai=matb1['levels'][:] #hPa
pres_ei=pres_erai[0,::-1]


for y in range(1996,Yfin):
    matb= sio.loadmat(path_data_erai+'ERAImac_'+str(y)+'.mat')
    temp_r=matb['temp'][:]
    rh_r=matb['rh'][:]
    q_r=matb['q'][:]
    u_r=matb['u'][:]
    v_r=matb['v'][:]
    time_r= matb['time2'][:]

    if y==2010:
        rh_r=0.263*pres_ei*q_r*np.exp((17.67*(temp_r-273.16))/(temp_r-29.65))**(-1)*100
        rh_r[rh_r>100]=100
        rh_r[rh_r<0]=0

    temp_erai=np.concatenate((temp_erai,temp_r), axis=0)
    rh_erai=np.concatenate((rh_erai,rh_r), axis=0)
    q_erai=np.concatenate((q_erai,q_r), axis=0)
    v_erai=np.concatenate((v_erai,v_r), axis=0)
    u_erai=np.concatenate((u_erai,u_r), axis=0)
    time_erai=np.concatenate((time_erai,time_r), axis=1)




#*****************************************************************************\

#*****************************************************************************\
#Height Levels
file_levels = np.genfromtxt('./levels.csv', delimiter=',')
hght_ei=file_levels[:,2]*1000 #meters
#*****************************************************************************\
#Building dates from 1800-01-01-00:00
time=time_erai[0,:]
date_ini=datetime(1800, 1, 1) + timedelta(hours=int(time[0])) #hours since
#Date Array
date_erai = pd.date_range(date_ini, periods=len(time), freq='6H')
#*****************************************************************************\
#Rotate from surface to free atmosphere
temp_ei=temp_erai[:,::-1]
u_ei=u_erai[:,::-1]
v_ei=v_erai[:,::-1]
q_ei=q_erai[:,::-1]*1000 #kg/kg=*1000 g/kg
mixr_ei= q_ei*1000 #g/kg
rh_ei=rh_erai[:,::-1]
#*****************************************************************************\
#Variables
#*****************************************************************************\
#Calculate Potential Temperature
theta_ei=(temp_ei)*(1000./pres_ei)**0.287;

#Calculate Virtual Potential Temperature
thetav_ei=(1+0.61*(mixr_ei/1000.))*theta_ei;
tempv_ei=(1+0.61*(mixr_ei/1000.))*temp_ei;

#Calculate Wind Shear and Gradiente Potential Temp
dthetav_ei=np.zeros(theta_ei.shape)
vshear_ei=np.empty(theta_ei.shape)*np.nan

for j in range(0,len(time)):
    for i in range(1,len(pres_ei)-1):
        dthetav_ei[j,i]=(thetav_ei[j,i+1]-thetav_ei[j,i])/float(hght_ei[i+1]-hght_ei[i])
        vshear_ei[j,i]=np.sqrt(float((u_ei[j,i]-u_ei[j,i-1])**2+(v_ei[j,i]-v_ei[j,i-1])**2))/float(hght_ei[i]-hght_ei[i-1])

#Relative Humidity
rh_erai[rh_erai>100]=100
rh_erai[rh_erai<0]=0

# plot(thetav_ei[:,1],'b',temp_ei[:,1],'r')
# show()
#******************************************************************************
#Boundary Layer Height Inversion 1 and 2
#Variables Initialization
sec_ind=np.empty(len(time))*np.nan
main_inv=np.empty(len(time),'float')*np.nan
sec_inv=np.empty(len(time))*np.nan
main_inversion=np.empty(len(time))*np.nan
sec_inversion=np.empty(len(time))*np.nan
main_inv_hght=np.empty(len(time))*np.nan
sec_inv_hght=np.empty(len(time))*np.nan

ei_hght_1inv=np.empty(len(time))*np.nan
ei_hght_2inv=np.empty(len(time))*np.nan
ei_strg_1inv=np.empty(len(time))*np.nan
ei_strg_2inv=np.empty(len(time))*np.nan

yotc_clas=np.empty(len(time))*np.nan



#******************************************************************************
#Main Inversion Position
for ind,line in enumerate(hght_ei):
    if line>=float(100.):
        twenty_y_index=ind
        break
for ind,line in enumerate(hght_ei):
    if line>=2500:
        twoky=ind
        break

main_inv=dthetav_ei[:,twenty_y_index:twoky].argmax(axis=1)
[i for i, j in enumerate(dthetav_ei[:,twenty_y_index:twoky]) if j == main_inv]
main_inv+=twenty_y_index #posicion main inv mas indice de sobre 100 m (3)

# Second Inversion Position
for i in range(0,len(time)):
    for ind in range(twenty_y_index,main_inv[i]):
    #height 2da inv 80% main inv
        if hght_ei[ind]>=(0.8)*hght_ei[main_inv[i]]:
            sec_ind[i]=ind
            break
        else:
            sec_ind[i]=np.nan
    if main_inv[i]==twenty_y_index:
        sec_ind[i]=np.nan
    #calcula la posicion de la sec inv (trata si se puede, si no asigna nan)
    try:
        sec_inv[i]=dthetav_ei[i,twenty_y_index:sec_ind[i]].argmax(0)
        [z for z, j in enumerate(dthetav_ei[i,twenty_y_index:sec_ind[i]]) if j == sec_inv[i]]
        sec_inv[i]+=twenty_y_index
    except:
        sec_inv[i]=np.nan

# main inversion must be > theta_v threshold
ptemp_comp1=dthetav_ei[:,main_inv[:]].diagonal() #extrae diagonal de pot temp

for i in range(0,len(time)):
    if ptemp_comp1[i]<ptemp_thold_main:
        #main_inv[i]=np.nan
        main_inv[i]=-9999 # Cannot convert float NaN to integer
        main_inversion[i]=False
        sec_inv[i]=np.nan
    else:
        main_inv_hght[i]=hght_ei[main_inv[i]]
        ei_strg_1inv[i]=dthetav_ei[i,main_inv[i]]
        main_inversion[i]=True

    if main_inv_hght[i]<=1:
        main_inv_hght[i]=np.nan #Corrige el -9999 para calcular alt

    # secondary inversion must be > theta_v threshold

    if np.isnan(sec_inv[i])==False and dthetav_ei[i,sec_inv[i]]>=ptemp_thold_sec:
        sec_inversion[i]=True
        sec_inv_hght[i]=hght_ei[sec_inv[i]]
        ei_strg_2inv[i]=dthetav_ei[i,sec_inv[i]]
    else:
        sec_inversion[i]=False
        sec_inv_hght[i]=np.nan

    hlev_yotc=hght_ei

    #Clasification
    if sec_inversion[i]==False and main_inversion[i]==True:
        yotc_clas[i]=2


    elif sec_inversion[i]==False and main_inversion[i]==False:
        yotc_clas[i]=1


    elif main_inversion[i]==True and sec_inversion[i]==True and vshear_ei[i,sec_inv[i]]>=shear_thold:
        yotc_clas[i]=4


    else:
        yotc_clas[i]=3

    #Height of Inversions
    ei_hght_1inv=main_inv_hght
    ei_hght_2inv=sec_inv_hght


# ****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
#                            MAC Data Original Levels
#*****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
# ****************************************************************************\
path_databom=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/files_bom/'
matb1= sio.loadmat(path_databom+'BOM_1995.mat')
bom_in=matb1['BOM_S'][:]
timesd= matb1['time'][:]
bom=bom_in

for y in range(1996,Yfin):
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
mixr=bom[:,5,:].reshape(ni[0],ni[2])
wdir_initial=bom[:,6,:].reshape(ni[0],ni[2])
wspd=bom[:,7,:].reshape(ni[0],ni[2])
relh=bom[:,4,:].reshape(ni[0],ni[2])

u=wspd*(np.cos(np.radians(270-wdir_initial)))
v=wspd*(np.sin(np.radians(270-wdir_initial)))




#*****************************************************************************\
#Smooth
#*****************************************************************************\

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

_,leni=pres.shape
temp_s=np.empty(pres.shape)*np.nan

for i in range(0,leni):
    temp_s[:,i] = movingaverage(temp[:,i], 10)

n=0


# plot(temp[:,n],hght[:,n],"k.")
# plot(temp_s[:,n],hght[:,n],"r")
# #xlim(0,1000)
# xlabel('Temperature')
# ylabel('Height')
# grid(True)


#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data ERA-I Levels
#*****************************************************************************\
#*****************************************************************************\
#Interpolation to ERAI Levels
prutemp=np.empty((len(hght_ei),0))*np.nan
#prutemp_un=np.empty((len(hght_ei),0))*np.nan
# prumixr=np.empty((len(hght_ei),0))*np.nan
# pruu=np.empty((len(hght_ei),0))*np.nan
# pruv=np.empty((len(hght_ei),0))*np.nan
# prurelh=np.empty((len(hght_ei),0))*np.nan

temp_hght=np.zeros((len(hght_ei),ni[2]),'float')
temp_pres=np.zeros((len(hght_ei),ni[2]),'float')

print ni[2]

for j in range(0,ni[2]):
#height initialization
    x=hght[:,j]
    x=pres[:,j]
    #x[-1]=np.nan
    #new_x=hght_ei
    #new_xt=hght_ei[0:21]
    new_xt=pres_ei[2:21]

#Interpolation ERAI levels
    yt=temp[~np.isnan(temp_s[:,j]),j]
    # ym=mixr[~np.isnan(mixr[:,j]),j]
    # yw=u[~np.isnan(u[:,j]),j]
    # yd=v[~np.isnan(v[:,j]),j]
    # yr=relh[~np.isnan(relh[:,j]),j]

    x=hght[~np.isnan(temp_s[:,j]),j]
    xp=pres[~np.isnan(temp_s[:,j]),j]

    if max(x)<10000:
        rest=new_xt*np.nan
        # resm=new_xt*np.nan
        # resw=new_xt*np.nan
        # resd=new_xt*np.nan
        # resr=new_xt*np.nan
    else:
        rest=interp1d(xp[::-1],yt[::-1],kind='slinear')(new_xt[::-1])

        # resm=interp1d(x,ym,kind='cubic')(new_xt)
        # resw=interp1d(x,yw,kind='cubic')(new_xt)
        # resd=interp1d(x,yd,kind='cubic')(new_xt)
        # resr=interp1d(x,yr,kind='cubic')(new_xt)



    prutemp=np.append(prutemp,rest)
    # prumixr=np.append(prumixr,resm)
    # pruu=np.append(pruu,resw)
    # pruv=np.append(pruv,resd)
    # prurelh=np.append(prurelh,resr)


    temp_interp_hght=si.UnivariateSpline(x,yt,k=5)
    temp_interp_pres=si.UnivariateSpline(xp[::-1],yt[::-1],k=5)
    for ind in range(0,37):
        temp_hght[ind,j]=temp_interp_hght(hght_ei[ind])
        temp_pres[ind,j]=temp_interp_pres(pres_ei[ind])

    print j, x[-1]


tempmac_ylev=prutemp.reshape(-1,len(new_xt)).transpose()
tempmac_ylev=tempmac_ylev[::-1]
#relhum_my=relhmac_ylev.T
temp_my=tempmac_ylev.T+273.16
temp_hght=temp_hght.T+273.16
temp_pres=temp_pres.T+273.16

# tempi=temp_my.T-273.16

# n=10
# plot(temp_s[:,n],hght[:,n],"b.")
# plot(tempi[:,n],hght_ei[0:21],"k.")
# plot(temp_hght[:,n],hght_ei,"r")
# xlim(-500,500)
# ylim(0,15000)
# xlabel('Temperature')
# ylabel('Height')
# grid(True)
# show()




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
#*****************************************************************************\
#                          Dataframes 1995-2010                               \
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Date index del periodo 2006-2010
date_index_12h = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
#date_index_12h = pd.date_range('1995-01-01 00:00', periods=1462, freq='12H')
# #*****************************************************************************\
#Dataframe ERA-Interim
t_list=temp_ei.tolist()
u_list=u_ei.tolist()
v_list=v_ei.tolist()
rh_list=rh_ei.tolist()
q_list=q_ei.tolist()
mr_list=mixr_ei.tolist()
thetav_list=thetav_ei.tolist()
theta_list=theta_ei.tolist()
dthetav_list=dthetav_ei.tolist()
vertshear_list=vshear_ei.tolist()

dy={'Clas':yotc_clas,
'temp':t_list,
'thetav':thetav_list,
'theta':theta_list,
'dthetav':dthetav_list,
'vertshear':vertshear_list,
'u':u_list,
'v':u_list,
'rh':rh_list,
'q':q_list,
'mixr':mr_list}

df_ei = pd.DataFrame(data=dy,index=date_erai)
df_ei.index.name = 'Date'
#*****************************************************************************\
#Dataframe ei All
df_erai=df_ei.reindex(date_index_12h)
df_erai.index.name = 'Date'


#*****************************************************************************\
dyc={'Clas ERA':yotc_clas,
'temp ERA':t_list,
'thetav ERA':thetav_list,
'theta ERA':theta_list,
'dthetav ERA':dthetav_list,
'vertshear ERA':vertshear_list,
'u ERA':u_list,
'v ERA':u_list,
'rh ERA':rh_list,
'q ERA':q_list,
'mixr ERA':mr_list}

dfc_ei = pd.DataFrame(data=dyc,index=date_erai)
dfc_ei.index.name = 'Date'

dfc_era=dfc_ei.reindex(date_index_12h)
dfc_era.index.name = 'Date'

#*****************************************************************************\
#*****************************************************************************\#*****************************************************************************\
#*****************************************************************************\
#Dataframe MAC ERA-I levels
t_list=temp_my.tolist()
t2_list=temp_hght.tolist()
t3_list=temp_pres.tolist()
# u_list=u_my.tolist()
# v_list=v_my.tolist()
# rh_list=relhum_my.tolist()
# mr_list=mixr_my.tolist()
# q_list=q_my.tolist()
# theta_list=pot_temp_my.tolist()
# thetav_list=pot_temp_v_my.tolist()
# dthetav_list=dthetav_my.tolist()
# vertshear_list=vertshear_my.tolist()

dmy={'temp':t_list,
'temp2':t2_list,
'temp3':t3_list}
# 'thetav':thetav_list,
# 'theta':theta_list,
# 'dthetav':dthetav_list,
# 'vertshear':vertshear_list,
# 'u':u_list,
# 'v':u_list,
# 'rh':rh_list,
# 'q':q_list,
# 'mixr':mr_list}

df_mac_y = pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
dfmy=df_mac_y.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

df_macei=dfmy.reindex(date_index_12h)
df_macei.index.name = 'Date'

#*****************************************************************************\

dmc={'temp MAC':t_list}
# 'thetav MAC':thetav_list,
# 'theta MAC':theta_list,
# 'dthetav MAC':dthetav_list,
# 'vertshear MAC':vertshear_list,
# 'u MAC':u_list,
# 'v MAC':u_list,
# 'rh MAC':rh_list,
# 'q MAC':q_list,
# 'mixr MAC':mr_list}


dfc_m = pd.DataFrame(data=dmc,index=time_my)
# Eliminate Duplicate Soundings
dfc_m=dfc_m.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

dfc_mac=dfc_m.reindex(date_index_12h)
dfc_mac.index.name = 'Date'


#*****************************************************************************\
#q=(float(mixr)/1000.)/(1+(float(mixr)/1000.))

t_list=temp_s.T.tolist()
# u_list=u.T.tolist()
# v_list=v.T.tolist()
# rh_list=relh.T.tolist()
# mr_list=mixr.T.tolist()
#q_list=q.tolist()
pres_list=pres.T.tolist()



#No interpol
dmc2={'pres':pres_list,
'temp':t_list}
# 'u':u_list,
# 'v':u_list,
# 'rh':rh_list,
# 'mixr':mr_list}


dfc_m2 = pd.DataFrame(data=dmc2,index=time_my)
# Eliminate Duplicate Soundings
dfc_m2=dfc_m2.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

dfc_macnint=dfc_m2.reindex(date_index_12h)
dfc_macnint.index.name = 'Date'

#*****************************************************************************\
#*****************************************************************************\
#Combination ERA-I and MAC (Esta es para tomar solo casos donde hay mediciones de ambos el mismo dia y hora)

dfc_macera=pd.concat([dfc_mac,dfc_era],axis=1)

#print np.count_nonzero(~np.isnan(dfc_macera['Clas ERA'])),  np.count_nonzero(~np.isnan(dfc_macera['Clas MAC']))
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Reading Fronts File
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_csv=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
df_front= pd.read_csv(path_data_csv + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')

#Merge datraframe mac with
df_eraifro=pd.concat([df_erai, df_front],axis=1)
df_meifro=pd.concat([df_macei, df_front],axis=1)

df_macerafro=pd.concat([dfc_macera, df_front],axis=1)


#Merge datraframe mac with no interpol

df_macni_fro=pd.concat([dfc_macnint, df_front],axis=1)



#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                       Plot Comparing
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
from matplotlib.projections import register_projection
from skewx_projection_matplotlib_lt_1d4 import SkewXAxes
register_projection(SkewXAxes)

nc=10
Tmacni=np.array(df_macni_fro['temp'][nc])
Pmacni=np.array(df_macni_fro['pres'][nc])
Tmac=np.array(df_meifro['temp'][nc])-273.16
Pmac=pres_ei[2:21]


Tmacun=np.array(df_meifro['temp3'][nc])-273.16
Pmacun=pres_ei


Tera=np.array(df_eraifro['temp'][nc])-273.16
Pera=pres_ei


fig=plt.figure(figsize=(8, 6))
ax0=fig.add_subplot(111, projection='skewx')
plt.grid(True)
ax0.semilogy(Tmacni, Pmacni, '-r',label='MAC Ori')
ax0.semilogy(Tmac, Pmac, '-g',label='MAC Int 1')
ax0.semilogy(Tmacun, Pmacun, '*-y',label='MAC Int 2')
ax0.semilogy(Tera, Pera, 'o-b',label='ERA-I')


l = ax0.axvline(0, color='b')

ax0.yaxis.set_major_formatter(ScalarFormatter())
ax0.set_yticks(np.linspace(100, 1000, 10))
ax0.set_ylim(1050, 100)

ax0.xaxis.set_major_locator(MultipleLocator(10))
ax0.set_xlim(-60, 60)

ax0.set_ylabel('Pressure (hPa)',fontsize = 10)
ax0.set_xlabel('Temperature (C)',fontsize = 10)
plt.legend(loc=2,fontsize = 10)

plt.show()



# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\
# #                                   Profiles and Fronts
# #*****************************************************************************\
# #*****************************************************************************\
# path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/ERAI/'

# #*****************************************************************************\
# #Definition Variables for Loop
# #*****************************************************************************\
# name_var=['rh','q','theta','temp']
# name_var_mac=['rh MAC','q MAC','theta MAC','temp MAC']
# name_var_era=['rh ERA','q ERA','theta ERA','temp ERA']
# name_var_all=['Relative Humidity','Specific Humidity','Pot. Temp','Temperature']

# name_var_sim=['RH','Q','PT','T','VPT']


# units_var=['%','g/kg','K','K']
# z_min=np.array([0,0,270,240])
# z_max=np.array([100,6,310,280])
# formati=['%.0f','%.1f','%.0f','%.0f']
# formati2=['%.0f','%.1f','%.1f','%.1f']

# dx=np.array([2,0.1,0.5,0.5])

# limii=np.array([20,0.6,5,5])

# #for m in range(0,len(name_var)):
# for m in range(3,4):
# #*****************************************************************************\
# # Means Complete Period MAC
# #*****************************************************************************\
#     df=df_meifro
#     bins=np.arange(-10,11,1)

#     df['catdist_fron'] = pd.cut(df['Dist CFront'], bins, labels=bins[0:-1])
#     ncount=pd.value_counts(df['catdist_fron'])

#     MG_mac=np.empty([len(ncount),len(pres_ei)])*np.nan
#     RH=np.empty([max(ncount),len(pres_ei)])*np.nan

#     k1=0
#     k2=0

#     for j in range(-10,10):

#         for i in range(0, len(df)):
#             if df['catdist_fron'][i]==j:
#                 RH[k2,:]=np.array(df[name_var[m]][i])

#                 k2=k2+1
#             MG_mac[k1,:]=np.nanmean(RH, axis=0)
#         k1=k1+1
#         k2=0


# #*****************************************************************************\
# # Means Complete Period ERA-I
# #*****************************************************************************\

#     df=df_eraifro

#     df['catdist_fron'] = pd.cut(df['Dist CFront'], bins, labels=bins[0:-1])
#     ncount=pd.value_counts(df['catdist_fron'])

#     MG_era=np.empty([len(ncount),len(pres_ei)])*np.nan
#     RH=np.empty([max(ncount),len(pres_ei)])*np.nan

#     k1=0
#     k2=0

#     for j in range(-10,10):

#         for i in range(0, len(df)):
#             if df['catdist_fron'][i]==j:
#                 RH[k2,:]=np.array(df[name_var[m]][i])

#                 k2=k2+1
#             MG_era[k1,:]=np.nanmean(RH, axis=0)
#         k1=k1+1
#         k2=0


# #*****************************************************************************\
# # Means Complete Period MAC No Int
# #*****************************************************************************\

#     # df=df_macni_fro

#     # df['catdist_fron'] = pd.cut(df['Dist CFront'], bins, labels=bins[0:-1])
#     # ncount=pd.value_counts(df['catdist_fron'])

#     # MG_macni=np.empty([len(ncount),3000])*np.nan
#     # RH=np.empty([max(ncount),3000])*np.nan

#     # P_macni=np.empty([len(ncount),3000])*np.nan
#     # P=np.empty([max(ncount),3000])*np.nan

#     # k1=0
#     # k2=0

#     # for j in range(-10,10):

#     #     for i in range(0, len(df)):
#     #         if df['catdist_fron'][i]==j:
#     #             RH[k2,:]=np.array(df[name_var[m]][i])
#     #             P[k2,:]=np.array(df['pres'][i])

#     #             k2=k2+1
#     #         MG_macni[k1,:]=np.nanmean(RH, axis=0)
#     #         P_macni[k1,:]=np.nanmean(P, axis=0)
#     #     k1=k1+1
#     #     k2=0






# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\
# # SkewT
# #*****************************************************************************\


#     nc=10

#     T=MG_mac[nc,:]-273.16
#     p=pres_ei
#     Td=MG_era[nc,:]-273.16

#     #TNI=MG_macni[nc,:]
#     #PNI=P_macni[nc,:]

#     fig=plt.figure(figsize=(8, 6))

#     ax0=fig.add_subplot(111, projection='skewx')
#     plt.grid(True)

#     ax0.semilogy(T, p, 'o-r',label='MAC')
#     ax0.semilogy(Td, p, 'o-g',label='ERA-I')
#     #ax0.semilogy(TNI, PNI, 'o-y',label='MAC NI')

#     l = ax0.axvline(0, color='b')

#     ax0.yaxis.set_major_formatter(ScalarFormatter())
#     ax0.set_yticks(np.linspace(100, 1000, 10))
#     ax0.set_ylim(1050, 100)

#     ax0.xaxis.set_major_locator(MultipleLocator(10))
#     ax0.set_xlim(-60, 60)

#     ax0.set_ylabel('Pressure (hPa)',fontsize = 10)
#     ax0.set_xlabel('Temperature (C)',fontsize = 10)
#     plt.legend(loc=2)

#     plt.show()
