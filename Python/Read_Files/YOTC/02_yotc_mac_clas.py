import numpy as np
import scipy.io as sio
from datetime import datetime, timedelta
import pandas as pd
import scipy.io as sio
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator
#from six import StringIO
#import pymeteo.skewt as skewt

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data'

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
mat= sio.loadmat(path_data+'/YOTC/mat/yotc_data.mat')
#*****************************************************************************\
#Crear fecha de inicio leyendo time
time= mat['time'][0]
date_ini=datetime(1900, 1, 1) + timedelta(hours=int(time[0])) #hours since
#Arreglo de fechas
date_range = pd.date_range(date_ini, periods=len(time), freq='12H')
#*****************************************************************************\
#Reading variables
t_ori= mat['temp'][:] #K
u_ori= mat['u'][:]
v_ori= mat['u'][:]
q_ori= mat['q'][:]

#rotate from surface to free atmosphere
temp=t_ori[:,::-1]
u=u_ori[:,::-1]
v=v_ori[:,::-1]
q=q_ori[:,::-1] #kg/kg
mixr= q*1000 #g/kg
#*****************************************************************************\
#Leyendo Alturas y press
file_levels = np.genfromtxt('levels.csv', delimiter=',')
hlev_yotc=file_levels[:,6]
plev_yotc=file_levels[:,3]

#*****************************************************************************\
#Calculate Virtual Temperature
theta=(temp)*(1000./plev_yotc)**0.287;
pot_temp=(1+0.61*(mixr/1000.))*theta;
#Calculate Potential Temperature
pot_temp_grad=np.zeros(pot_temp.shape)
vert_shear=np.empty(pot_temp.shape)
#Calculate Wind Shear

for j in range(0,90):
    for i in range(1,len(hlev_yotc)-1):
        pot_temp_grad[j,i]=pot_temp[j,i+1]-pot_temp[j,i]
        vert_shear[j,i]=np.sqrt((u[j,i]-u[j,i-1])**2+(v[j,i]-v[j,i-1]**2))/(hlev_yotc[i]-hlev_yotc[i-1])

#******************************************************************************
#Boundary Layer Height Inversion 1 and 2
#No he considearo el smoothed porque no tiene sentido aca (cada 2Hpa)

#Variables Initialization
sec_ind=np.empty(len(time))
main_inv=np.empty(len(time))
sec_inv=np.empty(len(time))
main_inversion=np.empty(len(time))
sec_inversion=np.empty(len(time))
minvhgt=[]
sinvhgt=[]  #variable non-np, despues reescribo
yotc_clas=np.empty(len(time))
yotc_depth=np.empty(len(time))
yotc_hght_1invBL=np.empty(len(time))
yotc_hght_2invBL=np.empty(len(time))
yotc_hght_1invDL=np.empty(len(time))
yotc_hght_2invDL=np.empty(len(time))

#Main Inversion Position
for ind,line in enumerate(hlev_yotc):
    if line>=float(100.):
        twenty_m_index=ind
        break
for ind,line in enumerate(hlev_yotc):
    if line>=2500:
        twokm=ind
        break

main_inv=pot_temp_grad[:,twenty_m_index:twokm].argmax(axis=1)
[i for i, j in enumerate(pot_temp_grad[:,twenty_m_index:twokm]) if j == main_inv]
main_inv+=twenty_m_index #posicion main inv mas indice de sobre 100 m (3)

#Second Inversion Position
for i in range(0,len(time)):
    for ind in range(twenty_m_index,main_inv[i]):
    #height 2da inv 80% main inv
        if hlev_yotc[ind]>=(0.8)*hlev_yotc[main_inv[i]]:
            sec_ind[i]=ind
            break
        else:
            sec_ind[i]=np.nan
    if main_inv[i]==twenty_m_index:
        sec_ind[i]=np.nan
    #calcula la posicion de la sec inv (trata si se puede, si no asigna nan)
    try:
        sec_inv[i]=pot_temp_grad[i,twenty_m_index:sec_ind[i]].argmax(0)
        [z for z, j in enumerate(pot_temp_grad[i,twenty_m_index:sec_ind[i]]) if j == sec_inv[i]]
        sec_inv[i]+=twenty_m_index
    except:
        sec_inv[i]=np.nan





    # main inversion must be > theta_v threshold
    if pot_temp_grad[i,main_inv[i]]<ptemp_thold_main:
        main_inv[main_inv==i]=np.nan
        main_inversion[i]=False
        sec_inv[i]=np.nan
    else:
        minvhgt.append(hlev_yotc[main_inv[i]])
        main_inv_hght=np.array(minvhgt)
        main_inversion[i]=True

    # secondary inversion must be > theta_v threshold
    if np.isnan(sec_inv[i])==False and pot_temp_grad[i,sec_inv[i]]>=ptemp_thold_sec:
        sec_inversion[i]=True
        sinvhgt.append(hlev_yotc[sec_inv[i]])
        sec_inv_hght=np.array(sinvhgt)
    else:
        sec_inversion[i]=False

    #Clasification
    if sec_inversion[i]==False and main_inversion[i]==True:
        yotc_clas[i]=2
        yotc_depth[i]=np.nan
        yotc_hght_1invBL[i]=np.nan
        yotc_hght_2invBL[i]=np.nan
        yotc_hght_1invDL[i]=np.nan
        yotc_hght_2invDL[i]=np.nan
    elif sec_inversion[i]==False and main_inversion[i]==False:
        yotc_clas[i]=1
        yotc_depth[i]=np.nan
        yotc_hght_1invBL[i]=np.nan
        yotc_hght_2invBL[i]=np.nan
        yotc_hght_1invDL[i]=np.nan
        yotc_hght_2invDL[i]=np.nan
    elif main_inversion[i]==True and sec_inversion[i]==True and vert_shear[i,sec_inv[i]]>=shear_thold:
        yotc_clas[i]=4
        yotc_depth[i]=(hlev_yotc[main_inv[i]]-hlev_yotc[sec_inv[i]])
        yotc_hght_1invBL[i]=hlev_yotc[main_inv[i]]
        yotc_hght_2invBL[i]=hlev_yotc[sec_inv[i]]
        yotc_hght_1invDL[i]=np.nan
        yotc_hght_2invDL[i]=np.nan
    else:
        yotc_clas[i]=3
        yotc_hght_1invDL[i]=hlev_yotc[main_inv[i]]
        yotc_hght_2invDL[i]=hlev_yotc[sec_inv[i]]
        yotc_depth[i]=np.nan
        yotc_hght_1invBL[i]=np.nan
        yotc_hght_2invBL[i]=np.nan

#np.count_nonzero(~np.isnan(main_inv))

#*****************************************************************************\
##*****************************************************************************\
#Dataframe datos per level
#nlevel=range(1,31)
#Crear listas con variables y unirlas a dataframe 3D
#t_list=temp.tolist()
#u_list=u.tolist()
#v_list=v.tolist()
#q_list=q.tolist()

#data={'temp':t_list, 'q':q_list,'u':u_list, 'v':v_list}
#df=pd.DataFrame(data=data, index=date_range)

