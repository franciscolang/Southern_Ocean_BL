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
mat= sio.loadmat(path_data+'yotc_data.mat')
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
file_levels = np.genfromtxt('./Read_Files/YOTC/levels.csv', delimiter=',')
hlev_yotc=file_levels[:,6]
plev_yotc=file_levels[:,3]

#*****************************************************************************\
#Calculate Virtual Temperature
theta=(temp)*(1000./plev_yotc)**0.287;
pot_temp=(1+0.61*(mixr/1000.))*theta;
#Calculate Potential Temperature
pot_temp_grad=np.zeros(pot_temp.shape)
yvert_shear=np.empty(pot_temp.shape)
#Calculate Wind Shear and Gradiente Potential Temp
for j in range(0,len(time)):
    for i in range(1,len(hlev_yotc)-1):
        pot_temp_grad[j,i]=(pot_temp[j,i+1]-pot_temp[j,i])/float(hlev_yotc[i+1]-hlev_yotc[i])
        yvert_shear[j,i]=np.sqrt(float((u[j,i]-u[j,i-1])**2+(v[j,i]-v[j,i-1])**2))/float(hlev_yotc[i]-hlev_yotc[i-1])

#******************************************************************************
#Boundary Layer Height Inversion 1 and 2
#No he considearo el smoothed porque no tiene sentido aca (cada 2Hpa)

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

# Second Inversion Position
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
    elif sec_inversion[i]==False and main_inversion[i]==False:
        yotc_clas[i]=1
        yotc_depth[i]=np.nan
        yotc_hght_1invBL[i]=np.nan
        yotc_hght_2invBL[i]=np.nan
        yotc_hght_1invDL[i]=np.nan
        yotc_hght_2invDL[i]=np.nan
    elif main_inversion[i]==True and sec_inversion[i]==True and yvert_shear[i,sec_inv[i]]>=shear_thold:
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
#Histogram
n, bin_edges =np.histogram(yotc_clas, bins=[1, 2, 3, 4,5],normed=1)

plt.figure(0)
y=bin_edges[0:4]

#clas = ('No Inv.', 'Single Inv.', 'Decoupled L.', 'Buffer L.')
clas = ('NI', 'SI', 'DL', 'BL')
y_pos = np.arange(len(clas))

plt.barh(y_pos, n, align='center', color='green')
plt.yticks(y_pos, clas)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Performance')
plt.xlim([0,1])
#plt.show()

#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data Original Levels
#*****************************************************************************\
#*****************************************************************************\
path_databom=base_dir+'/Dropbox/Monash_Uni/SO/MAC/MatFiles/files_bom/'
matb= sio.loadmat(path_databom+'BOM_2006-2010.mat')
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

#Leyendo Arreglo de fechas
timesd= matb['timesd'][:]
timestamp = [datenum_to_datetime(t) for t in timesd]
#*****************************************************************************\
bom=matb['SD'][:]#(3000,8,3594)
n=bom.shape
#*****************************************************************************\
#Separation of Variables

pres=bom[:,0,:].reshape(n[0],n[2])
hght=bom[:,1,:].reshape(n[0],n[2])
temp=bom[:,2,:].reshape(n[0],n[2])
mixr=bom[:,5,:].reshape(n[0],n[2])
wdir_initial=bom[:,6,:].reshape(n[0],n[2])
wspd=bom[:,7,:].reshape(n[0],n[2])
#*****************************************************************************\
#Initialization Variables
wdir=np.empty(pres.shape)
spec_hum=np.empty(pres.shape)
tempv=np.empty(pres.shape)
ptemp=np.empty(pres.shape)
ucomp=np.empty(pres.shape)
vcomp=np.empty(pres.shape)
ucomp_initial=np.empty(pres.shape)
vcomp_initial=np.empty(pres.shape)
vert_shear=np.empty(pres.shape)

pres_ave=np.empty(pres.shape)
hght_ave=np.empty(pres.shape)
ptemp_ave=np.empty(pres.shape)
mixr_ave=np.empty(pres.shape)
ucomp_ave=np.empty(pres.shape)
vcomp_ave=np.empty(pres.shape)
wdir_ave=np.empty(pres.shape)
ptemp_gmac=np.empty(pres.shape)

#*****************************************************************************\
#for j in range(0,n[2]):
for j in range(0,10):
#remove any reduntant measurementsat the same pressure

#Calculate new variables
    for i in range(0,len(pres)):
        if 0.<=wdir_initial[i,j]<=90.:
            wdir[i,j]=wdir_initial[i,j]+270.
        elif 90.<=wdir_initial[i,j]<=360.:
            wdir[i,j]=wdir_initial[i,j]-90.

        spec_hum[i,j]=(float(mixr[i,j])/1000.)/(1+(float(mixr[i,j])/1000.))
        tempv[i,j]=temp[i,j]*float(1+0.61*spec_hum[i,j])
        ptemp[i,j]=(tempv[i,j]+273.15)*((1000./pres[i,j])**0.286)
        vcomp[i,j]=-wspd[i,j]*(np.cos(np.radians(wdir[i,j])))
        ucomp[i,j]=-wspd[i,j]*(np.sin(np.radians(wdir[i,j])))
        vcomp_initial[i,j]=-wspd[i,j]*(np.cos(np.radians(wdir_initial[i,j])))
        ucomp_initial[i,j]=-wspd[i,j]*(np.sin(np.radians(wdir_initial[i,j])))
        vert_shear[i,j]=np.sqrt(float((ucomp_initial[i,j]-ucomp_initial[i-1,j])**2+(vcomp_initial[i,j]-vcomp_initial[i-1,j])**2))/float(hght[i,j]-hght[i-1,j])

#Interpolate BoM RH onto ave YOTC height levels

#Smooth data 5 points
    for i in range(0,len(pres)-4):
        ptemp_ave[i+2,j]=ptemp[i,j]*0.2+ptemp[i+1,j]*0.2+ptemp[i+2,j]*0.2+ptemp[i+3,j]*0.2+ptemp[i+4,j]*0.2
        pres_ave[i+2,j]=pres[i,j]*0.2+pres[i+1,j]*0.2+pres[i+2,j]*0.2+pres[i+2,j]*0.2+pres[i+4,j]*0.2
        hght_ave[i+2,j]=hght[i,j]*0.2+hght[i+1,j]*0.2+hght[i+2,j]*0.2+hght[i+2,j]*0.2+hght[i+4,j]*0.2
        mixr_ave[i+2,j]=mixr[i,j]*0.2+mixr[i+1,j]*0.2+mixr[i+2,j]*0.2+mixr[i+2,j]*0.2+mixr[i+4,j]*0.2
        wdir_ave[i+2,j]=wdir[i,j]*0.2+wdir[i+1,j]*0.2+wdir[i+2,j]*0.2+wdir[i+2,j]*0.2+wdir[i+4,j]*0.2
        ucomp_ave[i+2,j]=ucomp[i,j]*0.2+ucomp[i+1,j]*0.2+ucomp[i+2,j]*0.2+ucomp[i+2,j]*0.2+ucomp[i+4,j]*0.2
        vcomp_ave[i+2,j]=vcomp[i,j]*0.2+vcomp[i+1,j]*0.2+vcomp[i+2,j]*0.2+vcomp[i+2,j]*0.2+vcomp[i+4,j]*0.2

#Smooth further by binning every 2 hPa
    bin_size=5
    lower_p=np.empty(n[2])
    higher_p=np.empty(n[2])
    smooth_shear_a=np.empty(n[2])
    smooth_hght_a=np.empty(n[2])
    smooth_pres_a=np.empty(n[2])
    current_bin=[]
    lower_p=np.rint(pres[0,j])
    higher_p=np.rint(lower_p+bin_size)
    largos=np.zeros(((higher_p-500)/bin_size),'float')
    smooth_shear_a=np.empty([len(largos)+1,n[2]])
    #smooth_shear=np.empty([98,10])

    for ii in range(0,len(largos)):
        current_bin=[]
        for jj in range(0,len(pres)):
            if lower_p<pres[jj,j]<=higher_p:
                current_bin=np.append(current_bin,vert_shear[jj,j])
        smooth_shear_a[ii]=np.nanmean(current_bin)
        higher_p-=bin_size
        lower_p-=bin_size

        #smooth_shear[ii,j]=smooth_shear_a[ii]

    #smooth_shear=[]
    #smooth_shear=smooth_shear_a[~np.isnan(smooth_shear_a)]
    #smooth_shear.insert(0,np.nan)

#Gradiente Potential Temp
    for ind in range(0,len(ptemp_ave)-1):
        ptemp_gmac[ind,j]=(ptemp_ave[ind+1,j]-ptemp_ave[ind,j])/(hght_ave[ind+1,j]-hght_ave[ind,j])

    for i in range(0,len(pres)):
        for ind,line in enumerate(hght[i,j]):
            if line>=float(100.):
                twenty_index[j]=ind
            break


#*****************************************************************************\
#Dataframe datos per level
#nlevel=range(1,31)
#Crear listas con variables y unirlas a dataframe 3D
#t_list=temp.tolist()
#u_list=u.tolist()
#v_list=v.tolist()
#q_list=q.tolist()

#data={'temp':t_list, 'q':q_list,'u':u_list, 'v':v_list}
#df=pd.DataFrame(data=data, index=date_range)

