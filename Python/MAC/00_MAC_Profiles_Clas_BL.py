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
Yfin=2011
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
#                            MAC Data Original Levels
#*****************************************************************************\
#*****************************************************************************\
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
    #if np.nanmax(bom[:,1,j])<2500:
    if np.nanmax(bom[:,1,j])<5000:
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
        #if line>=2500:
        if line>=5000:
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
#Date index del periodo 2006-2010
date_index_all = pd.date_range('1995-01-01 00:00', periods=11688, freq='12H')

#*****************************************************************************\
#Dataframe MAC
dm={'Clas':mac_clas,
'Depth':mac_depth,
'1 Inv BL': mac_hght_1invBL,
'2 Inv BL': mac_hght_2invBL,
'1 Inv DL': mac_hght_1invDL,
'2 Inv DL': mac_hght_2invDL,
'1ra Inv': mac_hght_1inv,
'2da Inv': mac_hght_2inv,
'Strg 1inv': mac_strg_1inv,
'Strg 2inv': mac_strg_2inv}

df_mac = pd.DataFrame(data=dm,index=time_my)
# Eliminate Duplicate Soundings

dfm=df_mac.reset_index().drop_duplicates(cols='index',take_last=True).set_index('index')

df_mac_19952010=dfm.reindex(date_index_all)
df_mac_19952010.index.name = 'Date'

path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'
df_mac_19952010.to_csv(path_data_save + 'df_mac_19952010_5km.csv', sep='\t', encoding='utf-8')
