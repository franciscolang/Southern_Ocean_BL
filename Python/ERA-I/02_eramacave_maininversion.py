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
Yfin=2011

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
date_yotc=date_erai
#*****************************************************************************\
#Rotate from surface to free atmosphere
temp_ei=temp_erai[:,::-1]
u_ei=u_erai[:,::-1]
v_ei=v_erai[:,::-1]
q_ei=q_erai[:,::-1]*1000 #kg/kg=*1000 g/kg
mixr_ei= q_ei*1000 #g/kg
rh_ei=rh_erai[:,::-1]




#*****************************************************************************\
#*****************************************************************************\

temp=temp_ei
u=u_ei
v=v_ei
q=q_ei
mixr=mixr_ei


hlev_yotc=hght_ei
plev_yotc=pres_ei


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

#np.count_nonzero(~np.isnan(main_inv))

#*****************************************************************************\
# # #Histogram
# n, bin_edges =np.histogram(yotc_clas, bins=[1, 2, 3, 4,5],normed=1)

# plt.figure(0)
# y=bin_edges[0:4]

# #clas = ('No Inv.', 'Single Inv.', 'Decoupled L.', 'Buffer L.')
# clas = ('NI', 'SI', 'DL', 'BL')
# y_pos = np.arange(len(clas))

# plt.barh(y_pos, n, align='center', color='green')
# plt.yticks(y_pos, clas)
# plt.xticks(np.arange(0, 1.1, 0.1))
# plt.xlabel('Performance')
# plt.xlim([0,1])
# #plt.show()

#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data Original Levels
#*****************************************************************************\
#*****************************************************************************\
# path_databom=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/files_bom/'
# matb= sio.loadmat(path_databom+'BOM_2006-2010.mat')
# #*****************************************************************************\

# bom=matb['SD'][:]#(3000,8,3594)
# ni=bom.shape
# timesd= matb['timesd'][:]
Yfin=2011
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

u=wspd*(np.cos(np.radians(270-wdir_initial)))
v=wspd*(np.sin(np.radians(270-wdir_initial)))
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
#ptemp_gmac=np.empty(pres.shape)
ptemp_gmac=np.empty(pres.shape)
#twenty_index=np.empty([10])
twenty_m_index=[]
twokm=[]
main_m_inv=np.empty([ni[2]])
sec_m_inv=np.empty([ni[2]])
sec_m_ind=np.empty([ni[2]])
ptemp_comp1=np.empty([ni[2]])
main_m_inv_hght=np.empty([ni[2]])
main_m_inversion=np.empty([ni[2]])
sec_m_inversion=np.empty([ni[2]])
sec_m_inv_hght=np.empty([ni[2]])
ftrop=[]
mac_clas=np.empty([ni[2]])
mac_depth=np.empty([ni[2]])
mac_hght_1invBL=np.empty([ni[2]])
mac_hght_2invBL=np.empty([ni[2]])
mac_hght_1invDL=np.empty([ni[2]])
mac_hght_2invDL=np.empty([ni[2]])
mac_hght_1inv=np.empty([ni[2]])
mac_hght_2inv=np.empty([ni[2]])
mac_strg_1inv=np.empty([ni[2]])
mac_strg_2inv=np.empty([ni[2]])
#*****************************************************************************\
for j in range(0,ni[2]):
#for j in range(0,2000):
#remove any reduntant measurementsat the same pressure
#Calculate new variables
    for i in range(0,len(pres)):
        if 0.<=wdir_initial[i,j]<=90.:
            wdir[i,j]=wdir_initial[i,j]+270.
        elif 90.<=wdir_initial[i,j]<=360.:
            wdir[i,j]=wdir_initial[i,j]-90.

        spec_hum[i,j]=(float(mixr[i,j])/1000.)/(1+(float(mixr[i,j])/1000.))
        tempv[i,j]=temp[i,j]*float(1+0.61*spec_hum[i,j])
        ptemp[i,j]=(tempv[i,j]+273.15)*((1000./pres[i,j])**0.286) #Ok
        vcomp[i,j]=-wspd[i,j]*(np.cos(np.radians(wdir[i,j])))
        ucomp[i,j]=-wspd[i,j]*(np.sin(np.radians(wdir[i,j])))
        vcomp_initial[i,j]=-wspd[i,j]*(np.cos(np.radians(wdir_initial[i,j])))
        ucomp_initial[i,j]=-wspd[i,j]*(np.sin(np.radians(wdir_initial[i,j])))
        vert_shear[i,j]=np.sqrt(float((ucomp_initial[i,j]-ucomp_initial[i-1,j])**2+(vcomp_initial[i,j]-vcomp_initial[i-1,j])**2))/float(hght[i,j]-hght[i-1,j])

#Smooth data 5 points
    for i in range(0,len(pres)-4):

        ptemp_ave[i+2,j]=ptemp[i,j]*0.2+ptemp[i+1,j]*0.2+ptemp[i+2,j]*0.2+ptemp[i+3,j]*0.2+ptemp[i+4,j]*0.2

        # pres_ave[i+2,j]=pres[i,j]*0.2+pres[i+1,j]*0.2+pres[i+2,j]*0.2+pres[i+2,j]*0.2+pres[i+4,j]*0.2
        hght_ave[i+2,j]=hght[i,j]*0.2+hght[i+1,j]*0.2+hght[i+2,j]*0.2+hght[i+2,j]*0.2+hght[i+4,j]*0.2
        # mixr_ave[i+2,j]=mixr[i,j]*0.2+mixr[i+1,j]*0.2+mixr[i+2,j]*0.2+mixr[i+2,j]*0.2+mixr[i+4,j]*0.2
        # wdir_ave[i+2,j]=wdir[i,j]*0.2+wdir[i+1,j]*0.2+wdir[i+2,j]*0.2+wdir[i+2,j]*0.2+wdir[i+4,j]*0.2
        # ucomp_ave[i+2,j]=ucomp[i,j]*0.2+ucomp[i+1,j]*0.2+ucomp[i+2,j]*0.2+ucomp[i+2,j]*0.2+ucomp[i+4,j]*0.2
        # vcomp_ave[i+2,j]=vcomp[i,j]*0.2+vcomp[i+1,j]*0.2+vcomp[i+2,j]*0.2+vcomp[i+2,j]*0.2+vcomp[i+4,j]*0.2

    ptemp_ave[0:1,j]=np.nan
    hght_ave[0:1,j]=np.nan

#Smooth further by binning every 2 hPa
    bin_size=5
    lower_p=np.empty(ni[2])
    higher_p=np.empty(ni[2])
    smooth_shear_a=np.empty(ni[2])
    smooth_hght_a=np.empty(ni[2])
    smooth_pres_a=np.empty(ni[2])
    current_bin=[]
    lower_p=np.rint(pres[0,j])
    higher_p=np.rint(lower_p+bin_size)
    largos=np.zeros(((higher_p-500)/bin_size),'float')
    smooth_shear_a=np.empty([len(largos)+1,ni[2]])

    for ii in range(0,len(largos)):
        current_bin=[]
        for jj in range(0,len(pres)):
            if lower_p<pres[jj,j]<=higher_p:
                current_bin=np.append(current_bin,vert_shear[jj,j])
        smooth_shear_a[ii]=np.nanmean(current_bin)
        higher_p-=bin_size
        lower_p-=bin_size

#Gradiente Potential Temp
    for ind in range(0,len(ptemp_ave)-1):
        ptemp_gmac[ind,j]=(ptemp_ave[ind+1,j]-ptemp_ave[ind,j])/float(hght_ave[ind+1,j]-hght_ave[ind,j])


#Main Inversion Position
    for z,line in enumerate(hght[:,j]):
        if line>=float(100.):
            twenty_m_index=np.append(twenty_m_index,z)
            break
    for z,line in enumerate(hght[:,j]):
        if line>=2500:
            twokm=np.append(twokm,z)
            break

#posicion main inv mas indice de sobre 100 m
    main_m_inv[j]=ptemp_gmac[twenty_m_index[j]:twokm[j],j].argmax(axis=0)
    [i for i, k in enumerate(ptemp_gmac[twenty_m_index[j]:twokm[j],j]) if k == main_m_inv[j]]
    main_m_inv[j]+=twenty_m_index[j] #


# Second Inversion Position
    for ind in range(int(twenty_m_index[j]),int(main_m_inv[j])):
    #    print ind
    # height 2da inv 80% main inv
        if hght[ind,j]>=(0.8)*hght[main_m_inv[j],j]:
            sec_m_ind[j]=ind
            break
        else:
            sec_m_ind[j]=np.nan

    if main_m_inv[j]==twenty_m_index[j]:
        sec_m_ind[j]=np.nan
    #calcula la posicion de la sec inv (trata si se puede, si no asigna nan)
    try:
        sec_m_inv[j]=ptemp_gmac[twenty_m_index[j]:sec_m_ind[j],j].argmax(0)
        #[z for z, k in enumerate(ptemp_gmac[twenty_m_index:sec_m_ind[j],j]) if k == sec_m_inv[j]]
        sec_m_inv[j]+=twenty_m_index[j]
    except:
        sec_m_inv[j]=np.nan


# main inversion must be > theta_v threshold
    ptemp_comp1[j]=ptemp_gmac[main_m_inv[j],j]#.diagonal() #extrae diagonal de pot temp
#for i in range(0,len(time)):
    if ptemp_comp1[j]<ptemp_thold_main:
        #main_m_inv[i]=np.nan
        main_m_inv[j]=-9999 # Cannot convert float NaN to integer
        main_m_inversion[j]=False
        sec_m_inv[j]=np.nan
    else:
        main_m_inv_hght[j]=hght[main_m_inv[j],j]
        main_m_inversion[j]=True

    if main_m_inv_hght[j]<=1:
        main_m_inv_hght[j]=np.nan #Corrige el -9999 para calcular alt

 # secondary inversion must be > theta_v threshold
    if np.isnan(sec_m_inv[j])==False and ptemp_gmac[sec_m_inv[j],j]>=ptemp_thold_sec:
        sec_m_inversion[j]=True
        sec_m_inv_hght[j]=hght[sec_m_inv[j],j]
    else:
        sec_m_inversion[j]=False
        sec_m_inv_hght[j]=np.nan
 # height of the free troposphere
    if np.isnan(main_m_inv[j])==False and sec_m_inversion[j]==True:
        for ind,line in enumerate(hght[:,j]):
            if line>=(hght[main_m_inv[j],j]+1000.):
                ftropo[j]=ind
            break

#Clasification
    if sec_m_inversion[j]==False and main_m_inversion[j]==True:
        mac_clas[j]=2
        mac_depth[j]=np.nan
        mac_hght_1invBL[j]=np.nan
        mac_hght_2invBL[j]=np.nan
        mac_hght_1invDL[j]=np.nan
        mac_hght_2invDL[j]=np.nan
        mac_hght_1inv[j]=hght[main_m_inv[j],j]
        mac_hght_2inv[j]=np.nan
        mac_strg_1inv[j]=ptemp_gmac[main_m_inv[j],j]
        mac_strg_2inv[j]=np.nan
    elif sec_m_inversion[j]==False and main_m_inversion[j]==False:
        mac_clas[j]=1
        mac_depth[j]=np.nan
        mac_hght_1invBL[j]=np.nan
        mac_hght_2invBL[j]=np.nan
        mac_hght_1invDL[j]=np.nan
        mac_hght_2invDL[j]=np.nan
        mac_hght_1inv[j]=np.nan
        mac_hght_2inv[j]=np.nan
        mac_strg_1inv[j]=np.nan
        mac_strg_2inv[j]=np.nan
    elif main_m_inversion[j]==True and sec_m_inversion[j]==True and vert_shear[sec_m_inv[j],j]>=shear_thold:
        mac_clas[j]=4
        mac_depth[j]=(hght[main_m_inv[j],j]-hght[sec_m_inv[j],j])
        mac_hght_1invBL[j]=hght[main_m_inv[j],j]
        mac_hght_2invBL[j]=hght[sec_m_inv[j],j]
        mac_hght_1invDL[j]=np.nan
        mac_hght_2invDL[j]=np.nan
        mac_hght_1inv[j]=hght[main_m_inv[j],j]
        mac_hght_2inv[j]=hght[sec_m_inv[j],j]
        mac_strg_1inv[j]=ptemp_gmac[main_m_inv[j],j]
        mac_strg_2inv[j]=ptemp_gmac[sec_m_inv[j],j]
    else:
        mac_clas[j]=3
        mac_hght_1invDL[j]=hght[main_m_inv[j],j]
        mac_hght_2invDL[j]=hght[sec_m_inv[j],j]
        mac_depth[j]=(hght[main_m_inv[j],j]-hght[sec_m_inv[j],j])
        mac_hght_1invBL[j]=np.nan
        mac_hght_2invBL[j]=np.nan
        mac_hght_1inv[j]=hght[main_m_inv[j],j]
        mac_hght_2inv[j]=hght[sec_m_inv[j],j]
        mac_strg_1inv[j]=ptemp_gmac[main_m_inv[j],j]
        mac_strg_2inv[j]=ptemp_gmac[sec_m_inv[j],j]



# #np.count_nonzero(~np.isnan(sec_m_ind))
# #*****************************************************************************\
# # #Histogram
# n, bin_edges =np.histogram(mac_clas, bins=[1, 2, 3, 4,5],normed=1)

# plt.figure(0)
# y=bin_edges[0:4]

# #clas = ('No Inv.', 'Single Inv.', 'Decoupled L.', 'Buffer L.')
# clas = ('NI', 'SI', 'DL', 'BL')
# y_pos = np.arange(len(clas))

# plt.barh(y_pos, n, align='center', color='green')
# plt.yticks(y_pos, clas)
# plt.xticks(np.arange(0, 1.1, 0.1))
# plt.xlabel('Performance')
# plt.xlim([0,1])
# plt.show()

#*****************************************************************************\
#*****************************************************************************\
#                            MAC Data YOTC Levels
#*****************************************************************************\
#*****************************************************************************\
#Interpolation to YOTC Levels
# plt.plot(hght[0:200,1], 'ro')
# plt.show()

# prutemp=np.empty((len(hlev_yotc),0))
# pruwspd=np.empty((len(hlev_yotc),0))
# pruwdir=np.empty((len(hlev_yotc),0))
# prumixr=np.empty((len(hlev_yotc),0))


# for j in range(0,ni[2]):
# #for j in range(0,100):
# #height initialization
#     x=hght[:,j]
#     x[-1]=np.nan
#     new_x=hlev_yotc
# #Interpolation YOTC levels
#     yt=temp[:,j]
#     rest=interp1d(x,yt)(new_x)
#     prutemp=np.append(prutemp,rest)

#     yw=wspd[:,j]
#     resw=interp1d(x,yw)(new_x)
#     pruwspd=np.append(pruwspd,resw)

#     yd=wdir_initial[:,j]
#     resd=interp1d(x,yd)(new_x)
#     pruwdir=np.append(pruwdir,resd)

#     ym=mixr[:,j]
#     resm=interp1d(x,ym)(new_x)
#     prumixr=np.append(prumixr,resm)

# tempmac_ylev=prutemp.reshape(-1,len(hlev_yotc)).transpose()
# wspdmac_ylev=pruwspd.reshape(-1,len(hlev_yotc)).transpose()
# wdirmac_ylev=pruwdir.reshape(-1,len(hlev_yotc)).transpose()
# mixrmac_ylev=prumixr.reshape(-1,len(hlev_yotc)).transpose()

#*****************************************************************************\

temp_pres=np.zeros((len(plev_yotc),ni[2]),'float')
mixr_pres=np.zeros((len(plev_yotc),ni[2]),'float')
u_pres=np.zeros((len(plev_yotc),ni[2]),'float')
v_pres=np.zeros((len(plev_yotc),ni[2]),'float')


for j in range(0,ni[2]):

    yt=temp[~np.isnan(temp[:,j]),j]
    ym=mixr[~np.isnan(mixr[:,j]),j]
    yw=u[~np.isnan(u[:,j]),j]
    yd=v[~np.isnan(v[:,j]),j]

    xp=pres[~np.isnan(temp[:,j]),j]

    temp_interp_pres=si.UnivariateSpline(xp[::-1],yt[::-1],k=5)
    mixr_interp_pres=si.UnivariateSpline(xp[::-1],ym[::-1],k=5)
    u_interp_pres=si.UnivariateSpline(xp[::-1],yw[::-1],k=5)
    v_interp_pres=si.UnivariateSpline(xp[::-1],yd[::-1],k=5)

    for ind in range(0,len(plev_yotc)):
        temp_pres[ind,j]=temp_interp_pres(plev_yotc[ind])
        mixr_pres[ind,j]=mixr_interp_pres(plev_yotc[ind])
        u_pres[ind,j]=u_interp_pres(plev_yotc[ind])
        v_pres[ind,j]=v_interp_pres(plev_yotc[ind])

    temp_pres[temp_pres[:,j]>np.nanmax(yt),j]=np.nan
    temp_pres[temp_pres[:,j]<np.nanmin(yt),j]=np.nan

    u_pres[u_pres[:,j]>np.nanmax(yw),j]=np.nan
    u_pres[u_pres[:,j]<np.nanmin(yw),j]=np.nan
    v_pres[v_pres[:,j]>np.nanmax(yd),j]=np.nan
    v_pres[v_pres[:,j]<np.nanmin(yd),j]=np.nan

    mixr_pres[mixr_pres[:,j]>np.nanmax(ym),j]=np.nan
    mixr_pres[mixr_pres[:,j]<np.nanmin(ym),j]=np.nan

    del xp, yt, ym, yw, yd

tempmac_ylev=temp_pres
umac_ylev=u_pres
vmac_ylev=v_pres
mixrmac_ylev=mixr_pres


wspdmac_ylev=np.sqrt(umac_ylev**2 + vmac_ylev**2)
wdirmac_ylev=np.arctan2(-umac_ylev, -vmac_ylev)*(180/np.pi)
wdirmac_ylev[(umac_ylev == 0) & (vmac_ylev == 0)]=0

#*****************************************************************************\
#Initialization Variables
wdir_my=np.empty(tempmac_ylev.shape)
spec_hum_my=np.empty(tempmac_ylev.shape)
tempv_my=np.empty(tempmac_ylev.shape)
ptemp_my=np.empty(tempmac_ylev.shape)
ptemp_v_my=np.empty(tempmac_ylev.shape)
ucomp_my=np.empty(tempmac_ylev.shape)
vcomp_my=np.empty(tempmac_ylev.shape)
ucomp_initial_my=np.empty(tempmac_ylev.shape)
vcomp_initial_my=np.empty(tempmac_ylev.shape)
vert_shear_my=np.empty(tempmac_ylev.shape)
ptemp_gmy=np.empty(tempmac_ylev.shape)
ptemp_v_gmy=np.empty(tempmac_ylev.shape)
main_my_inv=np.empty(ni[2])
twenty_my_index=[]
twokmy=[]
sec_my_ind=np.empty(ni[2])
sec_my_inv=np.empty(ni[2])
ptemp_comp2=np.empty(ni[2])
main_my_inv_hght=np.empty(ni[2])
main_my_inversion=np.empty(ni[2])
sec_my_inv_hght=np.empty(ni[2])
sec_my_inversion=np.empty(ni[2])
mac_y_clas=np.empty(ni[2])
mac_y_hght_1invDL=np.empty(ni[2])
mac_y_hght_2invDL=np.empty(ni[2])
mac_y_depth=np.empty(ni[2])
mac_y_hght_1invBL=np.empty(ni[2])
mac_y_hght_2invBL=np.empty(ni[2])
mac_y_hght_1inv=np.empty(ni[2])
mac_y_hght_2inv=np.empty(ni[2])
mac_y_strg_1inv=np.empty(ni[2])
mac_y_strg_2inv=np.empty(ni[2])


mac_y_hght_1inv=np.empty(ni[2])*np.nan
mac_y_hght_2inv=np.empty(ni[2])*np.nan
mac_y_strg_1inv=np.empty(ni[2])*np.nan
mac_y_strg_2inv=np.empty(ni[2])*np.nan


main_my_inv=np.empty(ni[2])*np.nan
main_my_inversion=np.empty(ni[2])*np.nan
main_my_inv_hght=np.empty(ni[2])*np.nan




#*****************************************************************************\
for j in range(0,ni[2]):
#for j in range(0,100):
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

        vcomp_my[i,j]=-wspdmac_ylev[i,j]*(np.cos(np.radians(wdir_my[i,j])))
        ucomp_my[i,j]=-wspdmac_ylev[i,j]*(np.sin(np.radians(wdir_my[i,j])))

        vcomp_initial_my[i,j]=-wspdmac_ylev[i,j]*(np.cos(np.radians(wdirmac_ylev[i,j])))
        ucomp_initial_my[i,j]=-wspdmac_ylev[i,j]*(np.sin(np.radians(wdirmac_ylev[i,j])))

    for i in range(0,len(hlev_yotc)-1):
        vert_shear_my[i,j]=np.sqrt(float((ucomp_initial_my[i,j]-ucomp_initial_my[i-1,j])**2+(vcomp_initial_my[i,j]-vcomp_initial_my[i-1,j])**2))/float(hlev_yotc[i+1]-hlev_yotc[i])

        ptemp_v_gmy[i,j]=(ptemp_v_my[i+1,j]-ptemp_v_my[i,j])/float(hlev_yotc[i+1]-hlev_yotc[i])

    vert_shear_my[-1,j]=np.nan
    ptemp_v_gmy[-1,j]=np.nan


ptemp_gmy=ptemp_v_gmy
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

# main inversion must be > theta_v threshold
for j in range(0,ni[2]):
    ptemp_comp2[j]=ptemp_gmy[main_my_inv[j],j]

    if ptemp_comp2[j]>ptemp_thold_main:
        main_my_inv_hght[j]=hlev_yotc[main_my_inv[j]]
        mac_y_strg_1inv[j]=ptemp_comp2[j]
        main_my_inversion[j]=True
    else:
        main_my_inversion[j]=False
        mac_y_strg_1inv[j]=np.nan

#*****************************************************************************\
# #Histogram
# n, bin_edges =np.histogram(mac_y_clas, bins=[1, 2, 3, 4,5],normed=1)

# plt.figure(0)
# y=bin_edges[0:4]

# #clas = ('No Inv.', 'Single Inv.', 'Decoupled L.', 'Buffer L.')
# clas = ('NI', 'SI', 'DL', 'BL')
# y_pos = np.arange(len(clas))

# plt.barh(y_pos, n, align='center', color='green')
# plt.yticks(y_pos, clas)
# plt.xticks(np.arange(0, 1.1, 0.1))
# plt.xlabel('Performance')
# plt.xlim([0,1])
# #plt.show()

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
#date_index_all = pd.date_range('2006-01-01 00:00', periods=3652, freq='12H')

date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')
#*****************************************************************************\
#Dataframe YOTC 2008-2010
dy={'Clas':yotc_clas,
'1ra Inv': yotc_hght_1inv,
'Strg 1inv': yotc_strg_1inv}

df_yotc = pd.DataFrame(data=dy,index=date_yotc)
df_yotc.index.name = 'Date'
#*****************************************************************************\
#Dataframe YOTC All
df_yotc_all=df_yotc.reindex(date_index_all)
df_yotc_all.index.name = 'Date'
#*****************************************************************************\
#Dataframe MAC
dm={'Clas':mac_clas,
'1ra Inv': mac_hght_1inv,
'Strg 1inv': mac_strg_1inv}

df_mac = pd.DataFrame(data=dm,index=time_my)
# Eliminate Duplicate Soundings

dfm=df_mac.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

df_mac_final=dfm.reindex(date_index_all)
df_mac_final.index.name = 'Date'
# count_nan = len(df_my_final) - df_my_final.count()

#*****************************************************************************\
#Dataframe MAC YOTC levels
dmy={'Clas':mac_clas,
'1ra Inv': main_my_inv_hght,
'Strg 1inv': mac_y_strg_1inv}

df_mac_y = pd.DataFrame(data=dmy,index=time_my)
# Eliminate Duplicate Soundings
dfmy=df_mac_y.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

df_macyotc_final=dfmy.reindex(date_index_all)
df_macyotc_final.index.name = 'Date'
#*****************************************************************************\
# #Saving CSV
# path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/MCR/'

# df_yotc.to_csv(path_data_save + 'df_yotc_20082010.csv', sep='\t', encoding='utf-8')
# df_yotc_all.to_csv(path_data_save + 'df_yotc_20062010.csv', sep='\t', encoding='utf-8')
# df_mac_final.to_csv(path_data_save + 'df_mac_20062010.csv', sep='\t', encoding='utf-8')
# df_macyotc_final.to_csv(path_data_save + 'df_macyotc_20062010.csv', sep='\t', encoding='utf-8')

# #*****************************************************************************\
# #*****************************************************************************\
# #Saving CSV
# path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/MCR/'

# df_yotc.to_csv(path_data_save + 'df_era_20082010_5k_2.csv', sep='\t', encoding='utf-8')
# df_yotc_all.to_csv(path_data_save + 'df_era_19952010_5k_2.csv', sep='\t', encoding='utf-8')
# df_mac_final.to_csv(path_data_save + 'df_mac_19952010_5k_2.csv', sep='\t', encoding='utf-8')
# df_macyotc_final.to_csv(path_data_save + 'df_macera_19952010_5k_2.csv', sep='\t', encoding='utf-8')

#*****************************************************************************\
#*****************************************************************************\
#Saving CSV
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/MCR/'

df_yotc.to_csv(path_data_save + 'df_era_20082010.csv', sep='\t', encoding='utf-8')
df_yotc_all.to_csv(path_data_save + 'df_era_19952010.csv', sep='\t', encoding='utf-8')
df_mac_final.to_csv(path_data_save + 'df_mac_19952010.csv', sep='\t', encoding='utf-8')
df_macyotc_final.to_csv(path_data_save + 'df_macera_19952010.csv', sep='\t', encoding='utf-8')

