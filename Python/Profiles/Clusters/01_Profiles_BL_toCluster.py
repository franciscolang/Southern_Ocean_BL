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
from pylab import plot,show, grid

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/YOTC/mat/'

#*****************************************************************************\
#Default Info
mac = {'name': 'Macquarie Island, Australia', 'lat': -54.62, 'lon': 158.85}
lat_mac = mac['lat']
lon_mac = mac['lon']

ptemp_thold_main=0.010           # K/m
ptemp_thold_sec=0.005            # K/m
shear_thold=0.015               # 1/s

#*****************************************************************************\
#*****************************************************************************\
#                            YOTC Data
#*****************************************************************************\
#*****************************************************************************\
# mat= sio.loadmat(path_data+'yotc_data.mat')
# #*****************************************************************************\
# #Crear fecha de inicio leyendo time
# time= mat['time'][0]
# date_ini=datetime(1900, 1, 1) + timedelta(hours=int(time[0])) #hours since
# #Arreglo de fechas
# date_yotc = pd.date_range(date_ini, periods=len(time), freq='12H')
# #*****************************************************************************\
# #Reading variables
# t_ori= mat['temp'][:] #K
# u_ori= mat['u'][:]
# v_ori= mat['v'][:]
# q_ori= mat['q'][:]

# #rotate from surface to free atmosphere
# temp=t_ori[:,::-1]
# u=u_ori[:,::-1]
# v=v_ori[:,::-1]
# q=q_ori[:,::-1] #kg/kg
# mixr= q*1000 #g/kg


#*****************************************************************************\
#Leyendo Alturas y press
file_levels = np.genfromtxt('./../../Read_Files/YOTC/levels.csv', delimiter=',')
hlev_yotc=file_levels[:,6]
#plev_yotc=file_levels[:,3]
plev_yotc=file_levels[:,4] #value 10 is 925

# #*****************************************************************************\
# g=9.8 #m seg^-2


# #Calculate Virtual Temperature
# temp_v=(temp)*(1000./plev_yotc)**0.287;

# #Calculate VirtualPotential Temperature
# pot_temp_v=(1+0.61*(mixr/1000.))*temp_v;
# pot_temp=(1+0.61*(mixr/1000.))*temp;

# pot_temp_grad=np.zeros(pot_temp.shape)
# yvert_shear=np.empty(pot_temp.shape)*np.nan
# brn_yotca=np.empty(pot_temp.shape)*np.nan
# #Calculate Wind Shear and Gradiente Potential Temp
# for j in range(0,len(time)):
#     for i in range(1,len(hlev_yotc)-1):
#         pot_temp_grad[j,i]=(pot_temp_v[j,i+1]-pot_temp_v[j,i])/float(hlev_yotc[i+1]-hlev_yotc[i])
#         yvert_shear[j,i]=np.sqrt(float((u[j,i]-u[j,i-1])**2+(v[j,i]-v[j,i-1])**2))/float(hlev_yotc[i]-hlev_yotc[i-1])

# for j in range(0,len(time)):
#     for i in range(1,len(hlev_yotc)):
#         brn_yotca[j,i]=(g/float(pot_temp_v[j,0]))*((hlev_yotc[i]-hlev_yotc[0])*(pot_temp_v[j,i]-pot_temp_v[j,0]))/float((u[j,i]-u[j,0])**2+(v[j,i]-v[j,0])**2)

# #Calculate Relative Humidity
# relhum=0.263*plev_yotc*q*np.exp((17.67*(temp-273.16))/(temp-29.65))**(-1)*100
# relhum[relhum>100]=100
# relhum[relhum<0]=0

# #******************************************************************************
# #Boundary Layer Height Inversion 1 and 2
# #Variables Initialization
# sec_ind=np.empty(len(time))
# #main_inv=np.empty(len(time),'float')
# sec_inv=np.empty(len(time))
# main_inversion=np.empty(len(time))
# sec_inversion=np.empty(len(time))
# main_inv_hght=np.empty(len(time))
# sec_inv_hght=np.empty(len(time))
# yotc_clas=np.empty(len(time))
# yotc_depth=np.empty(len(time))
# yotc_hght_1invBL=np.empty(len(time))
# yotc_hght_2invBL=np.empty(len(time))
# yotc_hght_1invDL=np.empty(len(time))
# yotc_hght_2invDL=np.empty(len(time))
# yotc_hght_1inv=np.empty(len(time))
# yotc_hght_2inv=np.empty(len(time))
# yotc_strg_1inv=np.empty(len(time))
# yotc_strg_2inv=np.empty(len(time))

# relhum_yotc=np.empty(relhum.shape)*np.nan
# temp_yotc=np.empty(relhum.shape)*np.nan
# u_yotc=np.empty(relhum.shape)*np.nan
# v_yotc=np.empty(relhum.shape)*np.nan
# mixr_yotc=np.empty(relhum.shape)*np.nan
# pot_temp_v_yotc=np.empty(relhum.shape)*np.nan

# #******************************************************************************
# #Main Inversion Position
# for ind,line in enumerate(hlev_yotc):
#     if line>=float(100.):
#         twenty_y_index=ind
#         break
# for ind,line in enumerate(hlev_yotc):
#     if line>=2500:
#         twoky=ind
#         break

# main_inv=pot_temp_grad[:,twenty_y_index:twoky].argmax(axis=1)
# [i for i, j in enumerate(pot_temp_grad[:,twenty_y_index:twoky]) if j == main_inv]
# main_inv+=twenty_y_index #posicion main inv mas indice de sobre 100 m (3)

# # Second Inversion Position
# for i in range(0,len(time)):
#     for ind in range(twenty_y_index,main_inv[i]):
#     #height 2da inv 80% main inv
#         if hlev_yotc[ind]>=(0.8)*hlev_yotc[main_inv[i]]:
#             sec_ind[i]=ind
#             break
#         else:
#             sec_ind[i]=np.nan
#     if main_inv[i]==twenty_y_index:
#         sec_ind[i]=np.nan
#     #calcula la posicion de la sec inv (trata si se puede, si no asigna nan)
#     try:
#         sec_inv[i]=pot_temp_grad[i,twenty_y_index:sec_ind[i]].argmax(0)
#         [z for z, j in enumerate(pot_temp_grad[i,twenty_y_index:sec_ind[i]]) if j == sec_inv[i]]
#         sec_inv[i]+=twenty_y_index
#     except:
#         sec_inv[i]=np.nan

# # main inversion must be > theta_v threshold
# ptemp_comp1=pot_temp_grad[:,main_inv[:]].diagonal() #extrae diagonal de pot temp
# for i in range(0,len(time)):
#     if ptemp_comp1[i]<ptemp_thold_main:
#         #main_inv[i]=np.nan
#         main_inv[i]=-9999 # Cannot convert float NaN to integer
#         main_inversion[i]=False
#         sec_inv[i]=np.nan
#     else:
#         main_inv_hght[i]=hlev_yotc[main_inv[i]]
#         main_inversion[i]=True

#     if main_inv_hght[i]<=1:
#         main_inv_hght[i]=np.nan #Corrige el -9999 para calcular alt

#     # secondary inversion must be > theta_v threshold

#     if np.isnan(sec_inv[i])==False and pot_temp_grad[i,sec_inv[i]]>=ptemp_thold_sec:
#         sec_inversion[i]=True
#         sec_inv_hght[i]=hlev_yotc[sec_inv[i]]
#     else:
#         sec_inversion[i]=False
#         sec_inv_hght[i]=np.nan

#     #Clasification
#     if sec_inversion[i]==False and main_inversion[i]==True:
#         yotc_clas[i]=2
#         yotc_depth[i]=np.nan
#         yotc_hght_1invBL[i]=np.nan
#         yotc_hght_2invBL[i]=np.nan
#         yotc_hght_1invDL[i]=np.nan
#         yotc_hght_2invDL[i]=np.nan
#         yotc_hght_1inv[i]=hlev_yotc[main_inv[i]]
#         yotc_hght_2inv[i]=np.nan
#         yotc_strg_1inv[i]=pot_temp_grad[i,main_inv[i]]
#         yotc_strg_2inv[i]=np.nan

#     elif sec_inversion[i]==False and main_inversion[i]==False:
#         yotc_clas[i]=1
#         yotc_depth[i]=np.nan
#         yotc_hght_1invBL[i]=np.nan
#         yotc_hght_2invBL[i]=np.nan
#         yotc_hght_1invDL[i]=np.nan
#         yotc_hght_2invDL[i]=np.nan
#         yotc_hght_1inv[i]=np.nan
#         yotc_hght_2inv[i]=np.nan
#         yotc_strg_1inv[i]=np.nan
#         yotc_strg_2inv[i]=np.nan

#     elif main_inversion[i]==True and sec_inversion[i]==True and yvert_shear[i,sec_inv[i]]>=shear_thold:
#         yotc_clas[i]=4
#         yotc_depth[i]=(hlev_yotc[main_inv[i]]-hlev_yotc[sec_inv[i]])
#         yotc_hght_1invBL[i]=hlev_yotc[main_inv[i]]
#         yotc_hght_2invBL[i]=hlev_yotc[sec_inv[i]]
#         yotc_hght_1invDL[i]=np.nan
#         yotc_hght_2invDL[i]=np.nan
#         yotc_hght_1inv[i]=hlev_yotc[main_inv[i]]
#         yotc_hght_2inv[i]=hlev_yotc[sec_inv[i]]
#         yotc_strg_1inv[i]=pot_temp_grad[i,main_inv[i]]
#         yotc_strg_2inv[i]=pot_temp_grad[i,sec_inv[i]]

#     else:
#         yotc_clas[i]=3
#         yotc_hght_1invDL[i]=hlev_yotc[main_inv[i]]
#         yotc_hght_2invDL[i]=hlev_yotc[sec_inv[i]]
#         yotc_depth[i]=(hlev_yotc[main_inv[i]]-hlev_yotc[sec_inv[i]])
#         yotc_hght_1invBL[i]=np.nan
#         yotc_hght_2invBL[i]=np.nan
#         yotc_hght_1inv[i]=hlev_yotc[main_inv[i]]
#         yotc_hght_2inv[i]=hlev_yotc[sec_inv[i]]
#         yotc_strg_1inv[i]=pot_temp_grad[i,main_inv[i]]
#         yotc_strg_2inv[i]=pot_temp_grad[i,sec_inv[i]]

# relhum_yotc=relhum
# temp_yotc=temp
# u_yotc=u
# v_yotc=v
# mixr_yotc=mixr
# pot_temp_v_yotc=pot_temp_v
# dthetav_yotc=pot_temp_grad
# vertshear_yotc=yvert_shear
# pot_temp_yotc=pot_temp
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

nl=640

pres2=pres[0:nl,:]
temp2=temp[0:nl,:]
dwpo2=dwpo[0:nl,:]
mixr2=mixr[0:nl,:]
relh2=relh[0:nl,:]
u2=u[0:nl,:]
v2=v[0:nl,:]

#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data YOTC Levels
#*****************************************************************************\
#*****************************************************************************\
#Standard Pressure Levels
plev_std=np.array([950,925,900,850,800,750,700,650])
#hlev_std=np.array([925,850])

#*****************************************************************************\
#Interpolation to YOTC Levels
#hlev_yotc=hlev_std[::-1]

dimn=ni[2]

# new_ytemp=np.empty((len(hlev_yotc),dimn)) #1 de 3545
# new_ydwpo=np.empty((len(hlev_yotc),dimn))
# new_ymixr=np.empty((len(hlev_yotc),dimn))
# new_yu=np.empty((len(hlev_yotc),dimn))
# new_yv=np.empty((len(hlev_yotc),dimn))
# new_yrelh=np.empty((len(hlev_yotc),dimn))

# #for j in range(0,ni[2]):
# for j in range(0,dimn):
# #for j in range(977,978):
# #hpressure initialization
#     x=pres2[::-1,j]
#     new_x=hlev_yotc

#     y=temp2[::-1,j]
#     new_ytemp[:,j]= sp.interpolate.interp1d(x,y)(new_x)
#     del y

#     y=dwpo2[::-1,j]
#     new_ydwpo[:,j]= sp.interpolate.interp1d(x,y)(new_x)
#     del y
#     y=relh2[::-1,j]
#     new_yrelh[:,j]= sp.interpolate.interp1d(x,y)(new_x)
#     del y
#     y=mixr2[::-1,j]
#     new_ymixr[:,j]= sp.interpolate.interp1d(x,y)(new_x)
#     del y
#     y=u2[::-1,j]
#     new_yu[:,j]= sp.interpolate.interp1d(x,y)(new_x)
#     del y
#     y=v2[::-1,j]
#     new_yv[:,j]= sp.interpolate.interp1d(x,y)(new_x)
#     del y


# mixrmac_ylev=new_ymixr[::-1,:]
# tempmac_ylev=new_ytemp[::-1,:]
# umac_ylev=new_yu[::-1,:]
# vmac_ylev=new_yv[::-1,:]
# relhmac_ylev=new_yrelh[::-1,:]
# dwpomac_ylev=new_ydwpo[::-1,:]


#Interpolation to YOTC Levels

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

tempmac_ylev=temp_pres
umac_ylev=u_pres
vmac_ylev=v_pres
mixrmac_ylev=mixr_pres
relhmac_ylev=relh_pres
dwpomac_ylev=dwpo_pres

wspdmac_ylev=np.sqrt(umac_ylev**2 + vmac_ylev**2)
wdirmac_ylev=np.arctan2(-umac_ylev, -vmac_ylev)*(180/np.pi)
wdirmac_ylev[(umac_ylev == 0) & (vmac_ylev == 0)]=0
#*****************************************************************************\
#Initialization Variables
wdir_my=np.empty(tempmac_ylev.shape)

spec_hum_my=np.empty(tempmac_ylev.shape)
tempv_my=np.empty(tempmac_ylev.shape)
ptemp_v_my=np.empty(tempmac_ylev.shape)
ptemp_my=np.empty(tempmac_ylev.shape)


# relhum_my=np.empty(ni[2])*np.nan
# temp_my=np.empty(ni[2])*np.nan
# u_my=np.empty(ni[2])*np.nan
# v_my=np.empty(ni[2])*np.nan
# mixr_my=np.empty(ni[2])*np.nan
# pot_temp_my=np.empty(ni[2])*np.nan
# pot_temp_v_my=np.empty(ni[2])*np.nan

#*****************************************************************************\
for j in range(0,dimn):
#Calculate new variables
    for i in range(0,len(plev_std)):
        if 0.<=wdirmac_ylev[i,j]<=90.:
            wdir_my[i,j]=wdirmac_ylev[i,j]+270.
        elif 90.<=wdirmac_ylev[i,j]<=360.:
            wdir_my[i,j]=wdirmac_ylev[i,j]-90.

        spec_hum_my[i,j]=(float(mixrmac_ylev[i,j])/1000.)/(1+(float(mixrmac_ylev[i,j])/1000.))

        tempv_my[i,j]=tempmac_ylev[i,j]*float(1+0.61*spec_hum_my[i,j])

        ptemp_my[i,j]=(tempmac_ylev[i,j]+273.16)*((1000./plev_yotc[i])**0.286)

        ptemp_v_my[i,j]=(tempv_my[i,j]+273.16)*((1000./plev_yotc[i])**0.286)


#Variables for Clusters

temp_my=tempmac_ylev.T+273.16
u_my=umac_ylev.T
v_my=vmac_ylev.T
dwp_my=dwpomac_ylev.T+273.16
mixrmy=mixrmac_ylev.T
relh_my=relhmac_ylev.T
#pot_temp_my=ptemp_my.T

relh_sur_my=relh[0,:]
temp_sur_my=temp[0,:]
pres_sur_my=pres[0,:]


#*****************************************************************************\
#Cambiar fechas
timestamp = [datenum_to_datetime(t) for t in timesd]
time_my = np.array(timestamp)
time_my_ori = np.array(timestamp)

# for i in range(0,ni[2]):
#     #Cuando cae 23 horas del 31 de diciembre agrega un anio
#     if (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==12:
#         y1=time_my[i].year
#         time_my[i]=time_my[i].replace(year=y1+1,hour=0, month=1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==1:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==3:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==5:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==7:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==8:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==10:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==4:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==6:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==9:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==11:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==28 and time_my[i].month==2:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     #Bisiesto 2008
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2008:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2004:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2000:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==1996:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==1992:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)


#     #Cuando cae 23 horas, mueve hora a las 00 del dia siguiente
#     if time_my[i].hour==23 or time_my[i].hour==22:
#         d1=time_my[i].day
#         time_my[i]=time_my[i].replace(hour=0,day=d1+1)
#     else:
#         time_my[i]=time_my[i]
#     #Cuando cae 11 horas, mueve hora a las 12 del mismo dia
#     if time_my[i].hour==11 or time_my[i].hour==10 or time_my[i].hour==13 or time_my[i].hour==14:
#         time_my[i]=time_my[i].replace(hour=12)
#     else:
#         time_my[i]=time_my[i]

#     #Cuando cae 1 horas, mueve hora a las 0 del mismo dia
#     if time_my[i].hour==1 or time_my[i].hour==2:
#         time_my[i]=time_my[i].replace(hour=0)
#     else:
#         time_my[i]=time_my[i]

#*****************************************************************************\
#*****************************************************************************\
#                          Dataframes 2006-2010                               \
#*****************************************************************************\
#*****************************************************************************\
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
#*****************************************************************************\
# #Dataframe YOTC 2008-2010
# t_list=temp_yotc.tolist()
# u_list=u_yotc.tolist()
# v_list=v_yotc.tolist()
# rh_list=relhum_yotc.tolist()
# mr_list=mixr_yotc.tolist()
# thetav_list=pot_temp_v_yotc.tolist()
# theta_list=pot_temp_yotc.tolist()
# dthetav_list=dthetav_yotc.tolist()
# vertshear_list=vertshear_yotc.tolist()

# dy={'Clas':yotc_clas,
# 'Depth':yotc_depth,
# '1 Inv BL': yotc_hght_1invBL,
# '2 Inv BL': yotc_hght_2invBL,
# '1 Inv DL': yotc_hght_1invDL,
# '2 Inv DL': yotc_hght_2invDL,
# '1ra Inv': yotc_hght_1inv,
# '2da Inv': yotc_hght_2inv,
# 'Strg 1inv': yotc_strg_1inv,
# 'Strg 2inv': yotc_strg_2inv,
# 'temp':t_list,
# 'thetav':thetav_list,
# 'theta':theta_list,
# 'dthetav':dthetav_list,
# 'vertshear':vertshear_list,
# 'u':u_list,
# 'v':u_list,
# 'RH':rh_list,
# 'mixr':mr_list}

# df_yotc = pd.DataFrame(data=dy,index=date_yotc)
# df_yotc.index.name = 'Date'
#*****************************************************************************\
#Dataframe YOTC All
# df_yotc_all=df_yotc.reindex(date_index_all)
# df_yotc_all.index.name = 'Date'
#*****************************************************************************\
#Dataframe MAC YOTC levels
# t_list=temp_my.tolist()
# u_list=u_my.tolist()
# v_list=v_my.tolist()
# rh_list=relhum_my.tolist()
# mr_list=mixr_my.tolist()
# theta_list=pot_temp_my.tolist()
# thetav_list=pot_temp_v_my.tolist()


dmy={'pres_s':pres_sur_my,
'temp_s':temp_sur_my,
'relh_s':relh_sur_my,
'temp_950':temp_my[:,0],
'temp_925':temp_my[:,1],
'temp_900':temp_my[:,2],
'temp_850':temp_my[:,3],
'temp_800':temp_my[:,4],
'temp_750':temp_my[:,5],
'temp_700':temp_my[:,6],
'temp_650':temp_my[:,7],
'relh_950':relh_my[:,0],
'relh_925':relh_my[:,1],
'relh_900':relh_my[:,2],
'relh_850':relh_my[:,3],
'relh_800':relh_my[:,4],
'relh_750':relh_my[:,5],
'relh_700':relh_my[:,6],
'relh_650':relh_my[:,7],
'u_950':u_my[:,0],
'u_925':u_my[:,1],
'u_900':u_my[:,2],
'u_850':u_my[:,3],
'u_800':u_my[:,4],
'u_750':u_my[:,5],
'u_700':u_my[:,6],
'u_650':u_my[:,7],
'v_950':v_my[:,0],
'v_925':v_my[:,1],
'v_900':v_my[:,2],
'v_850':v_my[:,3],
'v_800':v_my[:,4],
'v_750':v_my[:,5],
'v_700':v_my[:,6],
'v_650':v_my[:,7]}


dfmyclu = pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
dfmy_clu=dfmyclu.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

#dfmy_clu=dfmy.reindex(date_index_all)
dfmy_clu.index.name = 'Date'
del time_my
#*****************************************************************************\
#Saving CSV

dfmy_clu.to_csv('../../../../003 Cluster/dfmy_clu_19952010.csv', sep=',', encoding='utf-8')


















# #*****************************************************************************\
# #*****************************************************************************\
# #Cambiar fechas
# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\timestamp = [datenum_to_datetime(t) for t in timesd]
# time_my = np.array(timestamp)
# time_my_ori = np.array(timestamp)

# for i in range(0,ni[2]):
#     #Cuando cae 23 horas del 31 de diciembre agrega un anio
#     if (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==12:
#         y1=time_my[i].year
#         time_my[i]=time_my[i].replace(year=y1+1,hour=0, month=1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==1:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==3:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==5:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==7:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==8:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==31 and time_my[i].month==10:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==4:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==6:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==9:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==30 and time_my[i].month==11:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)

#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==28 and time_my[i].month==2:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     #Bisiesto 2008
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2008:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2004:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==2000:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==1996:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)
#     if  (time_my[i].hour==23 or time_my[i].hour==22) and time_my[i].day==29 and time_my[i].month==2 and time_my[i].year==1992:
#         m1=time_my[i].month
#         time_my[i]=time_my[i].replace(hour=0, month=m1+1,day=1)


#     #Cuando cae 23 horas, mueve hora a las 00 del dia siguiente
#     if time_my[i].hour==23 or time_my[i].hour==22:
#         d1=time_my[i].day
#         time_my[i]=time_my[i].replace(hour=0,day=d1+1)
#     else:
#         time_my[i]=time_my[i]
#     #Cuando cae 11 horas, mueve hora a las 12 del mismo dia
#     if time_my[i].hour==11 or time_my[i].hour==10 or time_my[i].hour==13 or time_my[i].hour==14:
#         time_my[i]=time_my[i].replace(hour=12)
#     else:
#         time_my[i]=time_my[i]

#     #Cuando cae 1 horas, mueve hora a las 0 del mismo dia
#     if time_my[i].hour==1 or time_my[i].hour==2:
#         time_my[i]=time_my[i].replace(hour=0)
#     else:
#         time_my[i]=time_my[i]

# #*****************************************************************************\
# #Reescribe fechas para ajustarlas a los frentes

# dfmyclu_newd = pd.DataFrame(data=dmy,index=time_my)
# # Eliminate Duplicate Soundings
# dfmyclu_newd=dfmyclu_newd.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

# dfmyclu_newd=dfmyclu_newd.reindex(date_index_all)
# dfmyclu_newd.index.name = 'Date'


# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\
# #Reading Fronts File
# #*****************************************************************************\
# #*****************************************************************************\
# #*****************************************************************************\
# path_data_csv=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
# #*****************************************************************************\
# #Cold Front
# df_cfront= pd.read_csv(path_data_csv + 'df_cfront_19952010.csv', sep='\t', parse_dates=['Date'])
# df_cfront= df_cfront.set_index('Date')

# dfmy_cfro_clu=pd.concat([dfmyclu_newd, df_cfront],axis=1)
# #*****************************************************************************\
# #Warm Front
# df_wfront= pd.read_csv(path_data_csv + 'df_wfront_19952010.csv', sep='\t', parse_dates=['Date'])
# df_wfront= df_wfront.set_index('Date')

# dfmy_wfro_clu=pd.concat([dfmyclu_newd, df_wfront],axis=1)


# #*****************************************************************************\
# #Cold Fronts separation
# #*****************************************************************************\
# dfmy_precfro_clu = dfmy_cfro_clu[dfmy_cfro_clu['Dist CFront']<0]
# dfmy_poscfro_clu = dfmy_cfro_clu[dfmy_cfro_clu['Dist CFront']>0]
# #*****************************************************************************\
# path_data_clu=base_dir+'/Dropbox/Monash_Uni/SO/MAC/003 Cluster/'
# dfmy_poscfro_clu.to_csv(path_data_clu+'dfmy_poscfro_clu.csv', sep=',', encoding='utf-8')
# dfmy_precfro_clu.to_csv(path_data_clu+'dfmy_precfro_clu.csv', sep=',', encoding='utf-8')
