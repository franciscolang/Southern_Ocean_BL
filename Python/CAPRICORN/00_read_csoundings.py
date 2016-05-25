import numpy as np
import pandas as pd
import csv
import os
from numpy import genfromtxt
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/CAPRICORN/Sounding/'

sounding='201603172130Z'
#*****************************************************************************\
#Numpy
#*****************************************************************************\
data1 = genfromtxt(path_data+'/201603172130Z.csv', delimiter=',')
data2 = genfromtxt(path_data+'/201603180600Z.csv', delimiter=',')
data3 = genfromtxt(path_data+'/201603190600Z.csv', delimiter=',')
data4 = genfromtxt(path_data+'/201603210400Z.csv', delimiter=',')
data5 = genfromtxt(path_data+'/201603220003Z.csv', delimiter=',')
data6 = genfromtxt(path_data+'/201603220236Z.csv', delimiter=',')
data7 = genfromtxt(path_data+'/201603220900Z.csv', delimiter=',')
data8 = genfromtxt(path_data+'/201603221000Z.csv', delimiter=',')
data9 = genfromtxt(path_data+'/201603221348Z.csv', delimiter=',')
data10 = genfromtxt(path_data+'/201603221802Z.csv', delimiter=',')
data11 = genfromtxt(path_data+'/201603222217Z.csv', delimiter=',')
data12 = genfromtxt(path_data+'/201603230151Z.csv', delimiter=',')
data13 = genfromtxt(path_data+'/201603241815Z.csv', delimiter=',')
data14 = genfromtxt(path_data+'/201603242145Z.csv', delimiter=',')
data15 = genfromtxt(path_data+'/201603252157Z.csv', delimiter=',')
data16 = genfromtxt(path_data+'/201603260142Z.csv', delimiter=',')
data17 = genfromtxt(path_data+'/201603260624Z.csv', delimiter=',')
data18 = genfromtxt(path_data+'/201603280925Z.csv', delimiter=',')
data19 = genfromtxt(path_data+'/201603300407Z.csv', delimiter=',')
data20 = genfromtxt(path_data+'/201603312345Z.csv', delimiter=',')
data21 = genfromtxt(path_data+'/201604020600Z.csv', delimiter=',')
data22 = genfromtxt(path_data+'/201604022313Z.csv', delimiter=',')
data23 = genfromtxt(path_data+'/201604041019Z.csv', delimiter=',')
data24 = genfromtxt(path_data+'/201604050018Z.csv', delimiter=',')
data25 = genfromtxt(path_data+'/201604060408Z.csv', delimiter=',')
data26 = genfromtxt(path_data+'/201604070920Z.csv', delimiter=',')
data27 = genfromtxt(path_data+'/201604090703Z.csv', delimiter=',')
data28 = genfromtxt(path_data+'/201604100115Z.csv', delimiter=',')
data29 = genfromtxt(path_data+'/201604110617Z.csv', delimiter=',')
data30 = genfromtxt(path_data+'/201604120118Z.csv', delimiter=',')
data31 = genfromtxt(path_data+'/201604130939Z.csv', delimiter=',')

n=data1.shape
ngr=5000

a = np.empty([ngr-data1.shape[0],n[1]])
a[:] = np.nan
data1 = np.vstack([data1, a])

a = np.empty([ngr-data2.shape[0],n[1]])
a[:] = np.nan
data2 = np.vstack([data2, a])

a = np.empty([ngr-data3.shape[0],n[1]])
a[:] = np.nan
data3 = np.vstack([data3, a])

a = np.empty([ngr-data4.shape[0],n[1]])
a[:] = np.nan
data4 = np.vstack([data4, a])

a = np.empty([ngr-data5.shape[0],n[1]])
a[:] = np.nan
data5 = np.vstack([data5, a])

a = np.empty([ngr-data6.shape[0],n[1]])
a[:] = np.nan
data6 = np.vstack([data6, a])

a = np.empty([ngr-data7.shape[0],n[1]])
a[:] = np.nan
data7 = np.vstack([data7, a])

a = np.empty([ngr-data8.shape[0],n[1]])
a[:] = np.nan
data8 = np.vstack([data8, a])

a = np.empty([ngr-data9.shape[0],n[1]])
a[:] = np.nan
data9 = np.vstack([data9, a])

a = np.empty([ngr-data10.shape[0],n[1]])
a[:] = np.nan
data10 = np.vstack([data10, a])

a = np.empty([ngr-data11.shape[0],n[1]])
a[:] = np.nan
data11 = np.vstack([data11, a])

a = np.empty([ngr-data12.shape[0],n[1]])
a[:] = np.nan
data12 = np.vstack([data12, a])

a = np.empty([ngr-data13.shape[0],n[1]])
a[:] = np.nan
data13 = np.vstack([data13, a])

a = np.empty([ngr-data14.shape[0],n[1]])
a[:] = np.nan
data14 = np.vstack([data14, a])

a = np.empty([ngr-data15.shape[0],n[1]])
a[:] = np.nan
data15 = np.vstack([data15, a])

a = np.empty([ngr-data16.shape[0],n[1]])
a[:] = np.nan
data16 = np.vstack([data16, a])

a = np.empty([ngr-data17.shape[0],n[1]])
a[:] = np.nan
data17 = np.vstack([data17, a])

a = np.empty([ngr-data18.shape[0],n[1]])
a[:] = np.nan
data18 = np.vstack([data18, a])

a = np.empty([ngr-data19.shape[0],n[1]])
a[:] = np.nan
data19 = np.vstack([data19, a])

a = np.empty([ngr-data20.shape[0],n[1]])
a[:] = np.nan
data20 = np.vstack([data20, a])

a = np.empty([ngr-data21.shape[0],n[1]])
a[:] = np.nan
data21 = np.vstack([data21, a])

a = np.empty([ngr-data22.shape[0],n[1]])
a[:] = np.nan
data22 = np.vstack([data22, a])

a = np.empty([ngr-data23.shape[0],n[1]])
a[:] = np.nan
data23 = np.vstack([data23, a])

a = np.empty([ngr-data24.shape[0],n[1]])
a[:] = np.nan
data24 = np.vstack([data24, a])

a = np.empty([ngr-data25.shape[0],n[1]])
a[:] = np.nan
data25= np.vstack([data25, a])

a = np.empty([ngr-data26.shape[0],n[1]])
a[:] = np.nan
data26 = np.vstack([data26, a])

a = np.empty([ngr-data27.shape[0],n[1]])
a[:] = np.nan
data27 = np.vstack([data27, a])

a = np.empty([ngr-data28.shape[0],n[1]])
a[:] = np.nan
data28 = np.vstack([data28, a])

a = np.empty([ngr-data29.shape[0],n[1]])
a[:] = np.nan
data29 = np.vstack([data29, a])

a = np.empty([ngr-data30.shape[0],n[1]])
a[:] = np.nan
data30 = np.vstack([data30, a])

a = np.empty([ngr-data31.shape[0],n[1]])
a[:] = np.nan
data31 = np.vstack([data31, a])

capricorn = np.dstack([data1, data2, data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25,data26,data27,data28,data29,data30,data31])
#*****************************************************************************\
#Dates
#*****************************************************************************\

D=np.array([[2016,3,17,21,30],\
    [2016,3,18, 6,0],\
    [2016,3,19, 6,0],\
    [2016,3,21, 4,0],\
    [2016,3,22, 0,3],\
    [2016,3,22, 2,36],\
    [2016,3,22, 9,0],\
    [2016,3,22, 10,0],\
    [2016,3,22, 13,48],\
    [2016,3,22,18,2],\
    [2016,3,22,22,17],\
    [2016,3,23,1,51],\
    [2016,3,24,18,15],\
    [2016,3,24,21,45],\
    [2016,3,25,21,57],\
    [2016,3,26,1,42],\
    [2016,3,26,6,24],\
    [2016,3,28,9,25],\
    [2016,3,30,4,7],\
    [2016,3,31,23,45],\
    [2016,4,2,6,0],\
    [2016,4,2,23,13],\
    [2016,4,4,10,19],\
    [2016,4,5,0,18],\
    [2016,4,6,4,8],\
    [2016,4,7,9,20],\
    [2016,4,9,7,3],\
    [2016,4,10,1,15],\
    [2016,4,11,6,17],\
    [2016,4,12,1,18],\
    [2016,4,13,9,39],\
    ])


yy=D[:,0].astype(int)
mm=D[:,1].astype(int)
dd=D[:,2].astype(int)
hh=D[:,3].astype(int)
MM=D[:,4].astype(int)

ndates,_=D.shape
mydates=np.array([])
for n in range(0,ndates):
    mydates=np.append(mydates, datetime(yy[n], mm[n], dd[n],hh[n],MM[n],0))

#mydates = mydates[~np.isnan(mydates)]

#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#                               CAPRICORN
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#*****************************************************************************\
#Separation of Variables
ni=capricorn.shape
pres=capricorn[:,7,:].reshape(ni[0],ni[2])
hght=capricorn[:,6,:].reshape(ni[0],ni[2])
temp=capricorn[:,2,:].reshape(ni[0],ni[2])
mixr=capricorn[:,9,:].reshape(ni[0],ni[2])
ucomp=capricorn[:,5,:].reshape(ni[0],ni[2])
vcomp=capricorn[:,4,:].reshape(ni[0],ni[2])

lon=capricorn[0,15,:].reshape(ni[2])
lat=capricorn[0,16,:].reshape(ni[2])
#*****************************************************************************\
#Swap Non Values (-32768) for NaN
ucomp[ucomp==-32768]=np.nan
vcomp[vcomp==-32768]=np.nan
pres[pres==-32768]=np.nan
hght[hght==-32768]=np.nan
temp[temp==-32768]=np.nan
mixr[mixr==-32768]=np.nan

#*****************************************************************************\
ptemp_thold_main=0.010           # K/m
ptemp_thold_sec=0.005            # K/m
shear_thold=0.015               # 1/s
#*****************************************************************************\
spec_hum=np.empty(pres.shape)*np.nan
spec_hum=np.empty(pres.shape)*np.nan
tempv=np.empty(pres.shape)*np.nan
ptemp=np.empty(pres.shape)*np.nan
vert_shear=np.empty(pres.shape)*np.nan
pres_ave=np.empty(pres.shape)*np.nan
hght_ave=np.empty(pres.shape)*np.nan
ptemp_ave=np.empty(pres.shape)*np.nan
ptemp_gmac=np.empty(pres.shape)*np.nan


lower_p=np.empty(ni[2])
higher_p=np.empty(ni[2])
smooth_shear_a=np.empty(ni[2])
smooth_hght_a=np.empty(ni[2])
smooth_pres_a=np.empty(ni[2])


twenty_m_index=[]
twokm=[]

main_m_inv=np.empty(ni[2])*np.nan
sec_m_inv=np.empty(ni[2])*np.nan
sec_m_ind=np.empty(ni[2])*np.nan
ptemp_comp1=np.empty(ni[2])*np.nan
main_m_inv_hght=np.empty(ni[2])*np.nan
main_m_inversion=np.empty(ni[2])*np.nan
sec_m_inversion=np.empty(ni[2])*np.nan
sec_m_inv_hght=np.empty(ni[2])*np.nan

ftrop=[]

capr_clas=np.empty(ni[2])*np.nan
capr_depth=np.empty(ni[2])*np.nan
capr_hght_1invBL=np.empty(ni[2])*np.nan
capr_hght_2invBL=np.empty(ni[2])*np.nan
capr_hght_1invDL=np.empty(ni[2])*np.nan
capr_hght_2invDL=np.empty(ni[2])*np.nan
capr_hght_1inv=np.empty(ni[2])*np.nan
capr_hght_2inv=np.empty(ni[2])*np.nan
capr_strg_1inv=np.empty(ni[2])*np.nan
capr_strg_2inv=np.empty(ni[2])*np.nan



# lower_p=np.empty([6000,1])
# higher_p=np.empty([6000,1])
# smooth_shear_a=np.empty([6000,1])
# smooth_hght_a=np.empty([6000,1])
# smooth_pres_a=np.empty([6000,1])

# spec_hum=np.empty([6000,1])*np.nan
# spec_hum=np.empty([6000,1])*np.nan
# tempv=np.empty([6000,1])*np.nan
# ptemp=np.empty([6000,1])*np.nan
# vert_shear=np.empty([6000,1])*np.nan
# pres_ave=np.empty([6000,1])*np.nan
# hght_ave=np.empty([6000,1])*np.nan
# ptemp_ave=np.empty([6000,1])*np.nan
# ptemp_gmac=np.empty([6000,1])*np.nan

# twenty_m_index=[]
# twokm=[]

# main_m_inv=np.empty([1])*np.nan
# sec_m_inv=np.empty([1])*np.nan
# sec_m_ind=np.empty([1])*np.nan
# ptemp_comp1=np.empty([1])*np.nan
# main_m_inv_hght=np.empty([1])*np.nan
# main_m_inversion=np.empty([1])*np.nan
# sec_m_inversion=np.empty([1])*np.nan
# sec_m_inv_hght=np.empty([1])*np.nan

# ftrop=[]

# capr_clas=np.empty([1])*np.nan
# capr_depth=np.empty([1])*np.nan
# capr_hght_1invBL=np.empty([1])*np.nan
# capr_hght_2invBL=np.empty([1])*np.nan
# capr_hght_1invDL=np.empty([1])*np.nan
# capr_hght_2invDL=np.empty([1])*np.nan
# capr_hght_1inv=np.empty([1])*np.nan
# capr_hght_2inv=np.empty([1])*np.nan
# capr_strg_1inv=np.empty([1])*np.nan
# capr_strg_2inv=np.empty([1])*np.nan


#*****************************************************************************\
#*****************************************************************************\
for j in range(0,ni[2]):
#for j in range(0,1):
    for i in range(0,len(pres)):
        spec_hum[i,j]=(float(mixr[i,j])/1000.)/(1+(float(mixr[i,j])/1000.))

        tempv[i,j]=temp[i,j]*float(1+0.61*spec_hum[i,j])

        ptemp[i,j]=(tempv[i,j])*((1000./pres[i,j])**0.286) #Ok

        vert_shear[i,j]=np.sqrt(float((ucomp[i,j]-ucomp[i-1,j])**2+(vcomp[i,j]-vcomp[i-1,j])**2))/float(hght[i,j]-hght[i-1,j])
#Smooth data 5 points
    for i in range(0,len(pres)-4):
        ptemp_ave[i+2,j]=ptemp[i,j]*0.2+ptemp[i+1,j]*0.2+ptemp[i+2,j]*0.2+ptemp[i+3,j]*0.2+ptemp[i+4,j]*0.2

        hght_ave[i+2,j]=hght[i,j]*0.2+hght[i+1,j]*0.2+hght[i+2,j]*0.2+hght[i+2,j]*0.2+hght[i+4,j]*0.2

#Smooth further by binning every 2 hPa
    bin_size=5

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
        capr_clas[j]=2
        capr_depth[j]=np.nan
        capr_hght_1invBL[j]=np.nan
        capr_hght_2invBL[j]=np.nan
        capr_hght_1invDL[j]=np.nan
        capr_hght_2invDL[j]=np.nan
        capr_hght_1inv[j]=hght[main_m_inv[j],j]
        capr_hght_2inv[j]=np.nan
        capr_strg_1inv[j]=ptemp_gmac[main_m_inv[j],j]
        capr_strg_2inv[j]=np.nan
    elif sec_m_inversion[j]==False and main_m_inversion[j]==False:
        capr_clas[j]=1
        capr_depth[j]=np.nan
        capr_hght_1invBL[j]=np.nan
        capr_hght_2invBL[j]=np.nan
        capr_hght_1invDL[j]=np.nan
        capr_hght_2invDL[j]=np.nan
        capr_hght_1inv[j]=np.nan
        capr_hght_2inv[j]=np.nan
        capr_strg_1inv[j]=np.nan
        capr_strg_2inv[j]=np.nan
    elif main_m_inversion[j]==True and sec_m_inversion[j]==True and vert_shear[sec_m_inv[j],j]>=shear_thold:
        capr_clas[j]=4
        capr_depth[j]=(hght[main_m_inv[j],j]-hght[sec_m_inv[j],j])
        capr_hght_1invBL[j]=hght[main_m_inv[j],j]
        capr_hght_2invBL[j]=hght[sec_m_inv[j],j]
        capr_hght_1invDL[j]=np.nan
        capr_hght_2invDL[j]=np.nan
        capr_hght_1inv[j]=hght[main_m_inv[j],j]
        capr_hght_2inv[j]=hght[sec_m_inv[j],j]
        capr_strg_1inv[j]=ptemp_gmac[main_m_inv[j],j]
        capr_strg_2inv[j]=ptemp_gmac[sec_m_inv[j],j]
    else:
        capr_clas[j]=3
        capr_hght_1invDL[j]=hght[main_m_inv[j],j]
        capr_hght_2invDL[j]=hght[sec_m_inv[j],j]
        capr_depth[j]=(hght[main_m_inv[j],j]-hght[sec_m_inv[j],j])
        capr_hght_1invBL[j]=np.nan
        capr_hght_2invBL[j]=np.nan
        capr_hght_1inv[j]=hght[main_m_inv[j],j]
        capr_hght_2inv[j]=hght[sec_m_inv[j],j]
        capr_strg_1inv[j]=ptemp_gmac[main_m_inv[j],j]
        capr_strg_2inv[j]=ptemp_gmac[sec_m_inv[j],j]

#*****************************************************************************\
#Dataframe CAP
#*****************************************************************************\
dm={'Clas':capr_clas,
'Depth':capr_depth,
'1 Inv BL': capr_hght_1invBL,
'2 Inv BL': capr_hght_2invBL,
'1 Inv DL': capr_hght_1invDL,
'2 Inv DL': capr_hght_2invDL,
'1ra Inv': capr_hght_1inv,
'2da Inv': capr_hght_2inv,
'Strg 1inv': capr_strg_1inv,
'Strg 2inv': capr_strg_2inv,
'Lat':lat,
'Lon':lon}

df_capr = pd.DataFrame(data=dm,index=mydates)
df_capr.index.name = 'Date'

#*****************************************************************************\
#Bar
#*****************************************************************************\
capr_clas=np.array(df_capr['Clas'])
nm, bin_edgesm =np.histogram(capr_clas, bins=[1, 2, 3, 4,5],normed=1)
NI=nm[0]
SI=nm[1]
DL=nm[2]
BL=nm[3]

raw_data = {'tipo': ['CAPR'],
        'NI': NI,
        'SI': SI,
        'DL': DL,
        'BL': BL}

df = pd.DataFrame(raw_data, columns = ['tipo', 'NI', 'SI', 'DL', 'BL'])

f, ax1 = plt.subplots(1, figsize=(10,6))
bar_width = 0.7
bar_l = [i+1 for i in range(len(df['NI']))]
tick_pos = [i+(bar_width/2) for i in bar_l]

ax1.barh(bar_l, df['NI'],label='NI',color='#7C83AF')
ax1.barh(bar_l, df['SI'],label='SI',left=df['NI'],color='#525B92')
ax1.barh(bar_l, df['DL'],label='DL',left=[i+j for i,j in zip(df['NI'],df['SI'])], color='#182157')
ax1.barh(bar_l, df['BL'],label='BL',left=[i+j+k for i,j,k in zip(df['NI'],df['SI'],df['DL'])], color='#080F3A')

plt.yticks(tick_pos, df['tipo'])
ax1.set_xlabel("Absolute Percentage Occurrence")
ax1.set_title('Boundary Layer Categories')
box = ax1.get_position()
ax1.legend(loc='upper center',ncol=4)
plt.ylim([min(tick_pos)-bar_width+0.1, max(tick_pos)+bar_width])
#plt.show()
