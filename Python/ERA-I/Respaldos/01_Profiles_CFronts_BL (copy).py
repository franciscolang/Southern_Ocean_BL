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
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/ERA-I/'

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
mat= sio.loadmat(path_data+'yotc_data.mat')
#*****************************************************************************\
#Crear fecha de inicio leyendo time
time= mat['time'][0]
date_ini=datetime(1900, 1, 1) + timedelta(hours=int(time[0])) #hours since
#Arreglo de fechas
date_yotc = pd.date_range(date_ini, periods=len(time), freq='12H')
#*****************************************************************************\
#Reading variables
t_ori= mat['temp'][:] #K
u_ori= mat['u'][:]
v_ori= mat['v'][:]
q_ori= mat['q'][:]

#rotate from surface to free atmosphere
temp=t_ori[:,::-1]
u=u_ori[:,::-1]
v=v_ori[:,::-1]
q=q_ori[:,::-1] #kg/kg
mixr= q*1000 #g/kg


#*****************************************************************************\
#Leyendo Alturas y press
file_levels = np.genfromtxt('./../Read_Files/YOTC/levels.csv', delimiter=',')
hlev_yotc=file_levels[:,6]
#plev_yotc=file_levels[:,3]
plev_yotc=file_levels[:,4] #value 10 is 925

#*****************************************************************************\
g=9.8 #m seg^-2


#Calculate Potential Temperature
pot_temp=(temp)*(1000./plev_yotc)**0.287;

#Calculate Virtual Potential Temperature
pot_temp_v=(1+0.61*(mixr/1000.))*pot_temp;
temp_v=(1+0.61*(mixr/1000.))*temp;

pot_temp_grad=np.zeros(pot_temp.shape)
yvert_shear=np.empty(pot_temp.shape)*np.nan
brn_yotca=np.empty(pot_temp.shape)*np.nan
#Calculate Wind Shear and Gradiente Potential Temp
for j in range(0,len(time)):
    for i in range(1,len(hlev_yotc)-1):
        pot_temp_grad[j,i]=(pot_temp_v[j,i+1]-pot_temp_v[j,i])/float(hlev_yotc[i+1]-hlev_yotc[i])
        yvert_shear[j,i]=np.sqrt(float((u[j,i]-u[j,i-1])**2+(v[j,i]-v[j,i-1])**2))/float(hlev_yotc[i]-hlev_yotc[i-1])

for j in range(0,len(time)):
    for i in range(1,len(hlev_yotc)):
        brn_yotca[j,i]=(g/float(pot_temp_v[j,0]))*((hlev_yotc[i]-hlev_yotc[0])*(pot_temp_v[j,i]-pot_temp_v[j,0]))/float((u[j,i]-u[j,0])**2+(v[j,i]-v[j,0])**2)

#Calculate Relative Humidity
relhum=0.263*plev_yotc*q*np.exp((17.67*(temp-273.16))/(temp-29.65))**(-1)*100
relhum[relhum>100]=100
relhum[relhum<0]=0

#******************************************************************************
#Boundary Layer Height Inversion 1 and 2
#Variables Initialization
sec_ind=np.empty(len(time))
#main_inv=np.empty(len(time),'float')
sec_inv=np.empty(len(time))
main_inversion=np.empty(len(time))
sec_inversion=np.empty(len(time))
main_inv_hght=np.empty(len(time))
sec_inv_hght=np.empty(len(time))
yotc_clas=np.empty(len(time))
yotc_depth=np.empty(len(time))
yotc_hght_1invBL=np.empty(len(time))
yotc_hght_2invBL=np.empty(len(time))
yotc_hght_1invDL=np.empty(len(time))
yotc_hght_2invDL=np.empty(len(time))
yotc_hght_1inv=np.empty(len(time))
yotc_hght_2inv=np.empty(len(time))
yotc_strg_1inv=np.empty(len(time))
yotc_strg_2inv=np.empty(len(time))

relhum_yotc=np.empty(relhum.shape)*np.nan
temp_yotc=np.empty(relhum.shape)*np.nan
u_yotc=np.empty(relhum.shape)*np.nan
v_yotc=np.empty(relhum.shape)*np.nan
mixr_yotc=np.empty(relhum.shape)*np.nan
pot_temp_v_yotc=np.empty(relhum.shape)*np.nan

#******************************************************************************
#Main Inversion Position
for ind,line in enumerate(hlev_yotc):
    if line>=float(100.):
        twenty_y_index=ind
        break
for ind,line in enumerate(hlev_yotc):
    if line>=2500:
        twoky=ind
        break

main_inv=pot_temp_grad[:,twenty_y_index:twoky].argmax(axis=1)
[i for i, j in enumerate(pot_temp_grad[:,twenty_y_index:twoky]) if j == main_inv]
main_inv+=twenty_y_index #posicion main inv mas indice de sobre 100 m (3)

# Second Inversion Position
for i in range(0,len(time)):
    for ind in range(twenty_y_index,main_inv[i]):
    #height 2da inv 80% main inv
        if hlev_yotc[ind]>=(0.8)*hlev_yotc[main_inv[i]]:
            sec_ind[i]=ind
            break
        else:
            sec_ind[i]=np.nan
    if main_inv[i]==twenty_y_index:
        sec_ind[i]=np.nan
    #calcula la posicion de la sec inv (trata si se puede, si no asigna nan)
    try:
        sec_inv[i]=pot_temp_grad[i,twenty_y_index:sec_ind[i]].argmax(0)
        [z for z, j in enumerate(pot_temp_grad[i,twenty_y_index:sec_ind[i]]) if j == sec_inv[i]]
        sec_inv[i]+=twenty_y_index
    except:
        sec_inv[i]=np.nan

# main inversion must be > theta_v threshold
ptemp_comp1=pot_temp_grad[:,main_inv[:]].diagonal() #extrae diagonal de pot temp
for i in range(0,len(time)):
    if ptemp_comp1[i]<ptemp_thold_main:
        #main_inv[i]=np.nan
        main_inv[i]=-9999 # Cannot convert float NaN to integer
        main_inversion[i]=False
        sec_inv[i]=np.nan
    else:
        main_inv_hght[i]=hlev_yotc[main_inv[i]]
        main_inversion[i]=True

    if main_inv_hght[i]<=1:
        main_inv_hght[i]=np.nan #Corrige el -9999 para calcular alt

    # secondary inversion must be > theta_v threshold

    if np.isnan(sec_inv[i])==False and pot_temp_grad[i,sec_inv[i]]>=ptemp_thold_sec:
        sec_inversion[i]=True
        sec_inv_hght[i]=hlev_yotc[sec_inv[i]]
    else:
        sec_inversion[i]=False
        sec_inv_hght[i]=np.nan

    #Clasification
    if sec_inversion[i]==False and main_inversion[i]==True:
        yotc_clas[i]=2
        yotc_depth[i]=np.nan
        yotc_hght_1invBL[i]=np.nan
        yotc_hght_2invBL[i]=np.nan
        yotc_hght_1invDL[i]=np.nan
        yotc_hght_2invDL[i]=np.nan
        yotc_hght_1inv[i]=hlev_yotc[main_inv[i]]
        yotc_hght_2inv[i]=np.nan
        yotc_strg_1inv[i]=pot_temp_grad[i,main_inv[i]]
        yotc_strg_2inv[i]=np.nan

    elif sec_inversion[i]==False and main_inversion[i]==False:
        yotc_clas[i]=1
        yotc_depth[i]=np.nan
        yotc_hght_1invBL[i]=np.nan
        yotc_hght_2invBL[i]=np.nan
        yotc_hght_1invDL[i]=np.nan
        yotc_hght_2invDL[i]=np.nan
        yotc_hght_1inv[i]=np.nan
        yotc_hght_2inv[i]=np.nan
        yotc_strg_1inv[i]=np.nan
        yotc_strg_2inv[i]=np.nan

    elif main_inversion[i]==True and sec_inversion[i]==True and yvert_shear[i,sec_inv[i]]>=shear_thold:
        yotc_clas[i]=4
        yotc_depth[i]=(hlev_yotc[main_inv[i]]-hlev_yotc[sec_inv[i]])
        yotc_hght_1invBL[i]=hlev_yotc[main_inv[i]]
        yotc_hght_2invBL[i]=hlev_yotc[sec_inv[i]]
        yotc_hght_1invDL[i]=np.nan
        yotc_hght_2invDL[i]=np.nan
        yotc_hght_1inv[i]=hlev_yotc[main_inv[i]]
        yotc_hght_2inv[i]=hlev_yotc[sec_inv[i]]
        yotc_strg_1inv[i]=pot_temp_grad[i,main_inv[i]]
        yotc_strg_2inv[i]=pot_temp_grad[i,sec_inv[i]]

    else:
        yotc_clas[i]=3
        yotc_hght_1invDL[i]=hlev_yotc[main_inv[i]]
        yotc_hght_2invDL[i]=hlev_yotc[sec_inv[i]]
        yotc_depth[i]=(hlev_yotc[main_inv[i]]-hlev_yotc[sec_inv[i]])
        yotc_hght_1invBL[i]=np.nan
        yotc_hght_2invBL[i]=np.nan
        yotc_hght_1inv[i]=hlev_yotc[main_inv[i]]
        yotc_hght_2inv[i]=hlev_yotc[sec_inv[i]]
        yotc_strg_1inv[i]=pot_temp_grad[i,main_inv[i]]
        yotc_strg_2inv[i]=pot_temp_grad[i,sec_inv[i]]

relhum_yotc=relhum
temp_yotc=temp
u_yotc=u
v_yotc=v
mixr_yotc=mixr
pot_temp_v_yotc=pot_temp_v
dthetav_yotc=pot_temp_grad
vertshear_yotc=yvert_shear
pot_temp_yotc=pot_temp
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
mixr=bom[:,5,:].reshape(ni[0],ni[2])
wdir_initial=bom[:,6,:].reshape(ni[0],ni[2])
wspd=bom[:,7,:].reshape(ni[0],ni[2])
relh=bom[:,4,:].reshape(ni[0],ni[2])

u=wspd*(np.cos(np.radians(270-wdir_initial)))
v=wspd*(np.sin(np.radians(270-wdir_initial)))

#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data YOTC Levels
#*****************************************************************************\
#*****************************************************************************\
#Interpolation to YOTC Levels
prutemp=np.empty((len(hlev_yotc),0))
prumixr=np.empty((len(hlev_yotc),0))
pruu=np.empty((len(hlev_yotc),0))
pruv=np.empty((len(hlev_yotc),0))
prurelh=np.empty((len(hlev_yotc),0))

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

tempmac_ylev=prutemp.reshape(-1,len(hlev_yotc)).transpose()
umac_ylev=pruu.reshape(-1,len(hlev_yotc)).transpose()
vmac_ylev=pruv.reshape(-1,len(hlev_yotc)).transpose()
mixrmac_ylev=prumixr.reshape(-1,len(hlev_yotc)).transpose()
relhmac_ylev=prurelh.reshape(-1,len(hlev_yotc)).transpose()

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
brn_mya=np.empty(tempmac_ylev.shape)

ucomp_my=np.empty(tempmac_ylev.shape)
vcomp_my=np.empty(tempmac_ylev.shape)

ucomp_initial_my=np.empty(tempmac_ylev.shape)
vcomp_initial_my=np.empty(tempmac_ylev.shape)

vert_shear_my=np.empty(tempmac_ylev.shape)*np.nan
ptemp_gmy=np.empty(tempmac_ylev.shape)*np.nan
ptemp_v_gmy=np.empty(tempmac_ylev.shape)*np.nan
main_my_inv=np.empty(ni[2])*np.nan
twenty_my_index=[]
twokmy=[]
sec_my_ind=np.empty(ni[2])*np.nan
sec_my_inv=np.empty(ni[2])*np.nan
ptemp_comp2=np.empty(ni[2])*np.nan
main_my_inv_hght=np.empty(ni[2])*np.nan
main_my_inversion=np.empty(ni[2])*np.nan
sec_my_inv_hght=np.empty(ni[2])*np.nan
sec_my_inversion=np.empty(ni[2])*np.nan
mac_y_clas=np.empty(ni[2])*np.nan
mac_y_hght_1invDL=np.empty(ni[2])*np.nan
mac_y_hght_2invDL=np.empty(ni[2])*np.nan
mac_y_depth=np.empty(ni[2])*np.nan
mac_y_hght_1invBL=np.empty(ni[2])*np.nan
mac_y_hght_2invBL=np.empty(ni[2])*np.nan
mac_y_hght_1inv=np.empty(ni[2])*np.nan
mac_y_hght_2inv=np.empty(ni[2])*np.nan
mac_y_strg_1inv=np.empty(ni[2])*np.nan
mac_y_strg_2inv=np.empty(ni[2])*np.nan


relhum_my=np.empty(ni[2])*np.nan
temp_my=np.empty(ni[2])*np.nan
u_my=np.empty(ni[2])*np.nan
v_my=np.empty(ni[2])*np.nan
mixr_my=np.empty(ni[2])*np.nan
pot_temp_my=np.empty(ni[2])*np.nan
pot_temp_v_my=np.empty(ni[2])*np.nan
brn_my=np.empty(ni[2])*np.nan

#*****************************************************************************\
for j in range(0,ni[2]):
#Calculate new variables
    for i in range(0,len(hlev_yotc)):
        if 0.<=wdirmac_ylev[i,j]<=90.:
            wdir_my[i,j]=wdirmac_ylev[i,j]+270.
        elif 90.<=wdirmac_ylev[i,j]<=360.:
            wdir_my[i,j]=wdirmac_ylev[i,j]-90.

        spec_hum_my[i,j]=(float(mixrmac_ylev[i,j])/1000.)/(1+(float(mixrmac_ylev[i,j])/1000.))

        tempv_my[i,j]=tempmac_ylev[i,j]*float(1+0.61*spec_hum_my[i,j])

        ptemp_my[i,j]=(tempmac_ylev[i,j]+273.16)*((1000./plev_yotc[i])**0.286)

        ptemp_v_my[i,j]=(tempv_my[i,j]+273.16)*((1000./plev_yotc[i])**0.286)

        brn_mya[i,j]=(g/float(ptemp_v_my[0,j]))*((hlev_yotc[i]-hlev_yotc[0])*(ptemp_v_my[i,j]-ptemp_v_my[0,j]))/float((umac_ylev[i,j]-umac_ylev[0,j])**2+(vmac_ylev[i,j]-vmac_ylev[0,j])**2)

    for i in range(0,len(hlev_yotc)-1):
        vert_shear_my[i,j]=np.sqrt(float((umac_ylev[i,j]-umac_ylev[i-1,j])**2+(vmac_ylev[i,j]-vmac_ylev[i-1,j])**2))/float(hlev_yotc[i+1]-hlev_yotc[i])

        ptemp_v_gmy[i,j]=(ptemp_v_my[i+1,j]-ptemp_v_my[i,j])/float(hlev_yotc[i+1]-hlev_yotc[i])

    vert_shear_my[-1,j]=np.nan
    ptemp_v_gmy[-1,j]=np.nan


#*****************************************************************************\
# #Main Inversion Position
for ind,line in enumerate(hlev_yotc):
    if line>=float(100.):
        twenty_my_index=ind
        break
for ind,line in enumerate(hlev_yotc):
    if line>=2500:
        twokmy=ind
        break

main_my_inv=ptemp_gmy[twenty_my_index:twokmy,:].argmax(axis=0)
main_my_inv+=twenty_my_index #posicion main inv mas indice de sobre 100 m (3)

# # Second Inversion Position
for j in range(0,ni[2]):
#for j in range(0,100):
    for ind in range(twenty_my_index,main_my_inv[j]):
    #height 2da inv 80% main inv
        if hlev_yotc[ind]>=(0.8)*hlev_yotc[main_my_inv[j]]:
            sec_my_ind[j]=ind
            break
        else:
            sec_my_ind[j]=np.nan
    if main_my_inv[j]==twenty_my_index:
        sec_my_ind[j]=np.nan
    #calcula la posicion de la sec inv (trata si se puede, si no asigna nan)
    try:
        sec_my_inv[j]=ptemp_v_gmy[twenty_my_index:sec_my_ind[j],j].argmax(0)
        sec_my_inv[j]+=twenty_my_index
    except:
        sec_my_inv[j]=np.nan

# main inversion must be > theta_v threshold
for j in range(0,ni[2]):
#for j in range(0,100):
    ptemp_comp2[j]=ptemp_v_gmy[main_my_inv[j],j]#.diagonal() #extrae diagonal de pot temp
    if ptemp_comp2[j]<ptemp_thold_main:
        #main_inv[i]=np.nan
        main_my_inv[j]=-9999 # Cannot convert float NaN to integer
        main_my_inversion[j]=False
        sec_my_inv[j]=np.nan
    else:
        main_my_inv_hght[j]=hlev_yotc[main_my_inv[j]]
        main_my_inversion[j]=True

    if main_my_inv_hght[j]<=1:
        main_my_inv_hght[j]=np.nan #Corrige el -9999 para calcular alt

# secondary inversion must be > theta_v threshold

    if np.isnan(sec_my_inv[j])==False and ptemp_v_gmy[sec_my_inv[j],j]>=ptemp_thold_sec:
        sec_my_inversion[j]=True
        sec_my_inv_hght[j]=hlev_yotc[sec_my_inv[j]]
    else:
        sec_my_inversion[j]=False
        sec_my_inv_hght[j]=np.nan


# #Clasification
    if sec_my_inversion[j]==False and main_my_inversion[j]==True:
        mac_y_clas[j]=2
        mac_y_depth[j]=np.nan
        mac_y_hght_1invBL[j]=np.nan
        mac_y_hght_2invBL[j]=np.nan
        mac_y_hght_1invDL[j]=np.nan
        mac_y_hght_2invDL[j]=np.nan
        mac_y_hght_1inv[j]=hlev_yotc[main_my_inv[j]]
        mac_y_hght_2inv[j]=np.nan
        mac_y_strg_1inv[j]=ptemp_v_gmy[main_my_inv[j],j]
        mac_y_strg_2inv[j]=np.nan
    elif sec_my_inversion[j]==False and main_my_inversion[j]==False:
        mac_y_clas[j]=1
        mac_y_depth[j]=np.nan
        mac_y_hght_1invBL[j]=np.nan
        mac_y_hght_2invBL[j]=np.nan
        mac_y_hght_1invDL[j]=np.nan
        mac_y_hght_2invDL[j]=np.nan
        mac_y_hght_1inv[j]=np.nan
        mac_y_hght_2inv[j]=np.nan
        mac_y_strg_1inv[j]=np.nan
        mac_y_strg_2inv[j]=np.nan
    elif main_my_inversion[j]==True and sec_my_inversion[j]==True and vert_shear_my[sec_my_inv[j],j]>=shear_thold:
        mac_y_clas[j]=4
        mac_y_depth[j]=(hlev_yotc[main_my_inv[j]]-hlev_yotc[sec_my_inv[j]])
        mac_y_hght_1invBL[j]=hlev_yotc[main_my_inv[j]]
        mac_y_hght_2invBL[j]=hlev_yotc[sec_my_inv[j]]
        mac_y_hght_1invDL[j]=np.nan
        mac_y_hght_2invDL[j]=np.nan
        mac_y_hght_1inv[j]=hlev_yotc[main_my_inv[j]]
        mac_y_hght_2inv[j]=hlev_yotc[sec_my_inv[j]]
        mac_y_strg_1inv[j]=ptemp_v_gmy[main_my_inv[j],j]
        mac_y_strg_2inv[j]=ptemp_v_gmy[sec_my_inv[j],j]
    else:
        mac_y_clas[j]=3
        mac_y_hght_1invDL[j]=hlev_yotc[main_my_inv[j]]
        mac_y_hght_2invDL[j]=hlev_yotc[sec_my_inv[j]]
        mac_y_depth[j]=(hlev_yotc[main_my_inv[j]]-hlev_yotc[sec_my_inv[j]])
        mac_y_hght_1invBL[j]=np.nan
        mac_y_hght_2invBL[j]=np.nan
        mac_y_hght_1inv[j]=hlev_yotc[main_my_inv[j]]
        mac_y_hght_2inv[j]=hlev_yotc[sec_my_inv[j]]
        mac_y_strg_1inv[j]=ptemp_v_gmy[main_my_inv[j],j]
        mac_y_strg_2inv[j]=ptemp_v_gmy[sec_my_inv[j],j]


relhum_my=relhmac_ylev.T
temp_my=tempmac_ylev.T+273.16
u_my=umac_ylev.T
v_my=vmac_ylev.T
mixr_my=mixrmac_ylev.T
pot_temp_v_my=ptemp_v_my.T
pot_temp_my=ptemp_my.T
dthetav_my=ptemp_v_gmy.T
vertshear_my=vert_shear_my.T

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
#                          Dataframes 2006-2010                               \
#*****************************************************************************\
#*****************************************************************************\
#Date index del periodo 2006-2010
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
# #*****************************************************************************\
#Dataframe YOTC 2008-2010
t_list=temp_yotc.tolist()
u_list=u_yotc.tolist()
v_list=v_yotc.tolist()
rh_list=relhum_yotc.tolist()
mr_list=mixr_yotc.tolist()
thetav_list=pot_temp_v_yotc.tolist()
theta_list=pot_temp_yotc.tolist()
dthetav_list=dthetav_yotc.tolist()
vertshear_list=vertshear_yotc.tolist()

dy={'Clas':yotc_clas,
'Depth':yotc_depth,
'1 Inv BL': yotc_hght_1invBL,
'2 Inv BL': yotc_hght_2invBL,
'1 Inv DL': yotc_hght_1invDL,
'2 Inv DL': yotc_hght_2invDL,
'1ra Inv': yotc_hght_1inv,
'2da Inv': yotc_hght_2inv,
'Strg 1inv': yotc_strg_1inv,
'Strg 2inv': yotc_strg_2inv,
'temp':t_list,
'thetav':thetav_list,
'theta':theta_list,
'dthetav':dthetav_list,
'vertshear':vertshear_list,
'u':u_list,
'v':u_list,
'RH':rh_list,
'mixr':mr_list}

df_yotc = pd.DataFrame(data=dy,index=date_yotc)
df_yotc.index.name = 'Date'
#*****************************************************************************\
#Dataframe YOTC All
df_yotc_all=df_yotc.reindex(date_index_all)
df_yotc_all.index.name = 'Date'
#*****************************************************************************\
#Dataframe MAC YOTC levels
t_list=temp_my.tolist()
u_list=u_my.tolist()
v_list=v_my.tolist()
rh_list=relhum_my.tolist()
mr_list=mixr_my.tolist()
theta_list=pot_temp_my.tolist()
thetav_list=pot_temp_v_my.tolist()
dthetav_list=dthetav_my.tolist()
vertshear_list=vertshear_my.tolist()



dmy={'Clas':mac_y_clas,
'Depth':mac_y_depth,
'1 Inv BL': mac_y_hght_1invBL,
'2 Inv BL': mac_y_hght_2invBL,
'1 Inv DL': mac_y_hght_1invDL,
'2 Inv DL': mac_y_hght_2invDL,
'1ra Inv': mac_y_hght_1inv,
'2da Inv': mac_y_hght_2inv,
'Strg 1inv': mac_y_strg_1inv,
'Strg 2inv': mac_y_strg_2inv,
'temp':t_list,
'thetav':thetav_list,
'theta':theta_list,
'dthetav':dthetav_list,
'vertshear':vertshear_list,
'u':u_list,
'v':u_list,
'RH':rh_list,
'mixr':mr_list}

df_mac_y = pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
dfmy=df_mac_y.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

df_macyotc_final=dfmy.reindex(date_index_all)
df_macyotc_final.index.name = 'Date'

#*****************************************************************************\
#Saving CSV

#df_yotc.to_csv('./df_yotc.csv', sep=',', encoding='utf-8')


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Reading Fronts File
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
path_data_csv=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
df_front= pd.read_csv(path_data_csv + 'df_front.csv', sep='\t', parse_dates=['Date'])
df_front= df_front.set_index('Date')

#Merge datraframe mac with
df_yotcfro=pd.concat([df_yotc_all, df_front],axis=1)
df_myfro=pd.concat([df_macyotc_final, df_front],axis=1)


#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                                   Profiles
#*****************************************************************************\
#*****************************************************************************\
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/fronts_ok/profiles/'
#*****************************************************************************\
#MAC Ave
#*****************************************************************************\
df=df_myfro
bins=np.arange(-10,11,1)

df['catdist_fron'] = pd.cut(df['Dist Front'], bins, labels=bins[0:-1])
ncount=pd.value_counts(df['catdist_fron'])


Mrelhum=np.empty([20,91])*np.nan
RH=np.empty([max(ncount),91])*np.nan

Mdthetav=np.empty([20,91])*np.nan
dthetav=np.empty([max(ncount),91])*np.nan

Mtheta=np.empty([20,91])*np.nan
theta=np.empty([max(ncount),91])*np.nan

k1=0
k2=0

for j in range(-10,10):

    for i in range(0, len(df)):
        if df['catdist_fron'][i]==j:
            RH[k2,:]=np.array(df['RH'][i])
            dthetav[k2,:]=np.array(df['dthetav'][i])
            theta[k2,:]=np.array(df['theta'][i])
            k2=k2+1
        Mrelhum[k1,:]=np.nanmean(RH, axis=0)
        Mdthetav[k1,:]=np.nanmean(dthetav, axis=0)
        Mtheta[k1,:]=np.nanmean(theta, axis=0)
        #print j, k2
    k1=k1+1
    k2=0


#Flip West-East (Negative Postfront condition)
Mrelhum=Mrelhum[::-1]
Mdthetav=Mdthetav[::-1]
Mtheta=Mtheta[::-1]
#*****************************************************************************\
# #Rescale
# nlev=27
# height_new=np.arange(0,5001,200) #(22,)
# height_new[0]=10
# hlev=hlev_yotc[0:nlev]

# Mrelhum=Mrelhum[:,0:nlev]
# n1,n2=Mrelhum.shape
# Mdthetav=Mdthetav[:,0:nlev]
# Mtheta=Mtheta[:,0:nlev]


# #Interpolation
# Mrh_my=np.empty([n1,len(height_new)])*np.nan
# Mdthetav_my=np.empty([n1,len(height_new)])*np.nan
# Mtheta_my=np.empty([n1,len(height_new)])*np.nan

# x=hlev
# new_x=height_new

# for i in range(0,n1):
#     y=Mrelhum[i,:]
#     Mrh_my[i,:]= sp.interpolate.interp1d(x, y,kind='cubic')(new_x)
#     del y
#     y=Mdthetav[i,:]
#     Mdthetav_my[i,:]= sp.interpolate.interp1d(x, y,kind='cubic')(new_x)
#     del y
#     y=Mtheta[i,:]
#     Mtheta_my[i,:]= sp.interpolate.interp1d(x, y,kind='cubic')(new_x)
# #*****************************************************************************\
# #Setup
# #*****************************************************************************\
# xtick=[0, height_new[0],height_new[5],height_new[10],height_new[15],height_new[20],height_new[25]]
# #*****************************************************************************\
# #Relative Humidity
# #*****************************************************************************\
# vmin=30
# vmax=100

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mrh_my.T,cmap='jet',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.0f")
# cbar.ax.set_title('(%)', size=12)

# ax.set_title('Relative humidity', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()
# plt.savefig(path_data_save + 'prof_RH_my.eps', format='eps', dpi=1200)

# #*****************************************************************************\
# #Strenght (Virtual Potential Temperature)
# #*****************************************************************************\
# vmin=0
# vmax=0.01

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mdthetav_my.T,cmap='jet',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.3f")
# cbar.ax.set_title('(K m$^{-1}$)', size=12)

# ax.set_title(r'Strength of the inversion ($d\theta_v/dz$)', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()
# plt.savefig(path_data_save + 'prof_strenght_my.eps', format='eps', dpi=1200)
# #*****************************************************************************\
# #Potential Temperature
# #*****************************************************************************\
# vmin=270
# vmax=300

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mtheta_my.T,cmap='jet',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.0f")
# cbar.ax.set_title('(K)', size=12)

# ax.set_title(r'Potential temperature ($\theta$)', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()
# plt.savefig(path_data_save + 'prof_ptemp_my.eps', format='eps', dpi=1200)



# #*****************************************************************************\
# #*****************************************************************************\
# #YOTC
# #*****************************************************************************\
# #*****************************************************************************\
# df=df_yotcfro
# bins=np.arange(-10,11,1)

# df['catdist_fron'] = pd.cut(df['Dist Front'], bins, labels=bins[0:-1])
# ncount=pd.value_counts(df['catdist_fron'])


# Mrelhum=np.empty([20,91])*np.nan
# RH=np.empty([max(ncount),91])*np.nan

# Mdthetav=np.empty([20,91])*np.nan
# dthetav=np.empty([max(ncount),91])*np.nan

# Mtheta=np.empty([20,91])*np.nan
# theta=np.empty([max(ncount),91])*np.nan

# k1=0
# k2=0

# for j in range(-10,10):

#     for i in range(0, len(df)):
#         if df['catdist_fron'][i]==j:
#             RH[k2,:]=np.array(df['RH'][i])
#             dthetav[k2,:]=np.array(df['dthetav'][i])
#             theta[k2,:]=np.array(df['theta'][i])
#             k2=k2+1
#         Mrelhum[k1,:]=np.nanmean(RH, axis=0)
#         Mdthetav[k1,:]=np.nanmean(dthetav, axis=0)
#         Mtheta[k1,:]=np.nanmean(theta, axis=0)
#         #print j, k2
#     k1=k1+1
#     k2=0


# #Flip West-East
# Mrelhum=Mrelhum[::-1]
# Mdthetav=Mdthetav[::-1]
# Mtheta=Mtheta[::-1]
# #*****************************************************************************\
# #Rescale
# nlev=27
# height_new=np.arange(0,5001,200) #(22,)
# height_new[0]=10
# hlev=hlev_yotc[0:nlev]

# Mrelhum=Mrelhum[:,0:nlev]
# n1,n2=Mrelhum.shape
# Mdthetav=Mdthetav[:,0:nlev]
# Mtheta=Mtheta[:,0:nlev]


# #Interpolation
# Mrh_yotc=np.empty([n1,len(height_new)])*np.nan
# Mdthetav_yotc=np.empty([n1,len(height_new)])*np.nan
# Mtheta_yotc=np.empty([n1,len(height_new)])*np.nan

# x=hlev
# new_x=height_new

# for i in range(0,n1):
#     y=Mrelhum[i,:]
#     Mrh_yotc[i,:]= sp.interpolate.interp1d(x, y,kind='cubic')(new_x)
#     del y
#     y=Mdthetav[i,:]
#     Mdthetav_yotc[i,:]= sp.interpolate.interp1d(x, y,kind='cubic')(new_x)
#     del y
#     y=Mtheta[i,:]
#     Mtheta_yotc[i,:]= sp.interpolate.interp1d(x, y,kind='cubic')(new_x)
# #*****************************************************************************\
# #Setup
# #*****************************************************************************\
# xtick=[0, height_new[0],height_new[5],height_new[10],height_new[15],height_new[20],height_new[25]]
# #*****************************************************************************\
# #Relative Humidity
# #*****************************************************************************\
# vmin=30
# vmax=100

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mrh_yotc.T,cmap='jet',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.0f")
# cbar.ax.set_title('(%)', size=12)

# ax.set_title('Relative humidity', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()

# plt.savefig(path_data_save + 'prof_RH_yotc.eps', format='eps', dpi=1200)
# #*****************************************************************************\
# #Strenght (Virtual Potential Temperature)
# #*****************************************************************************\
# vmin=0
# vmax=0.01

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mdthetav_yotc.T,cmap='jet',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.3f")
# cbar.ax.set_title('(K m$^{-1}$)', size=12)

# ax.set_title(r'Strength of the inversion ($d\theta_v/dz$)', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()
# plt.savefig(path_data_save + 'prof_strenght_yotc.eps', format='eps', dpi=1200)
# #*****************************************************************************\
# #Potential Temperature
# #*****************************************************************************\
# vmin=270
# vmax=300

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mtheta_yotc.T,cmap='jet',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.0f")
# cbar.ax.set_title('(K)', size=12)

# ax.set_title(r'Potential temperature ($\theta$)', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()
# plt.savefig(path_data_save + 'prof_ptemp_yotc.eps', format='eps', dpi=1200)

# #*****************************************************************************\
# #*****************************************************************************\
# #Diferences
# #*****************************************************************************\
# #*****************************************************************************\


# Mrh_dif=Mrh_yotc-Mrh_my
# Mdthetav_dif=Mdthetav_yotc-Mdthetav_my
# Mtheta_dif=Mtheta_yotc-Mtheta_my

# #*****************************************************************************\
# #Relative Humidity
# #*****************************************************************************\
# vmin=-25
# vmax=25

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mrh_dif.T,cmap='seismic',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.0f")
# cbar.ax.set_title('(%)', size=12)

# ax.set_title('YOTC minus MAC$_{AVE}$ - relative humidity', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()
# plt.savefig(path_data_save + 'prof_RH_dif.eps', format='eps', dpi=1200)

# #*****************************************************************************\
# #Strenght (Virtual Potential Temperature)
# #*****************************************************************************\
# vmin=-0.007
# vmax=0.007

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mdthetav_dif.T,cmap='seismic',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.3f")
# cbar.ax.set_title('(K m$^{-1}$)', size=12)

# ax.set_title(r'YOTC minus MAC$_{AVE}$ - strength of the inversion ($d\theta_v/dz$)', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()
# plt.savefig(path_data_save + 'prof_strenght_dif.eps', format='eps', dpi=1200)
# #*****************************************************************************\
# #Potential Temperature
# #*****************************************************************************\
# vmin=-5
# vmax=5

# fig, ax = plt.subplots(facecolor='w', figsize=(12,6))
# img = ax.pcolor(Mtheta_dif.T,cmap='seismic',vmin=vmin, vmax=vmax)

# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="6%", pad=0.05)
# cbar = plt.colorbar(img, cax=cax, format="%.0f")
# cbar.ax.set_title('(K)', size=12)

# ax.set_title(r'YOTC minus MAC$_{AVE}$ - potential temperature ($\theta$)', size=18)
# ax.set_xticklabels(np.arange(-15,15,5))
# ax.set_yticklabels(xtick)

# ax.set_ylabel('Altitude (m)', size=18)
# ax.set_xlabel('Position across cold front from cold to warm sector (deg)', size=18)

# ax.margins(0.05)
# plt.tight_layout()
# plt.savefig(path_data_save + 'prof_ptemp_dif.eps', format='eps', dpi=1200)

