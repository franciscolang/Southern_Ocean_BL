import numpy as np
import pandas as pd
import csv
import os
from numpy import genfromtxt
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from pylab import plot,show, grid, legend

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
hght=capricorn[:,6,:].reshape(ni[0],ni[2])/float(1000)
temp=capricorn[:,2,:].reshape(ni[0],ni[2])
mixr=capricorn[:,9,:].reshape(ni[0],ni[2])
ucomp=capricorn[:,5,:].reshape(ni[0],ni[2])
vcomp=capricorn[:,4,:].reshape(ni[0],ni[2])

# lon=capricorn[0,15,:].reshape(ni[2])
# lat=capricorn[0,16,:].reshape(ni[2])
lon=capricorn[:,15,:].reshape(ni[0],ni[2])
lat=capricorn[:,16,:].reshape(ni[0],ni[2])

#*****************************************************************************\
#Swap Non Values (-32768) for NaN
ucomp[ucomp==-32768]=np.nan
vcomp[vcomp==-32768]=np.nan
pres[pres==-32768]=np.nan
hght[hght==-32768]=np.nan
temp[temp==-32768]=np.nan
mixr[mixr==-32768]=np.nan
lon[lon==-32768]=np.nan
lat[lat==-32768]=np.nan

lat2=lat
lon2=lon
hght2=hght
#*****************************************************************************\

p=np.empty(31)*np.nan

for i in range(0, 31):
    p[i]=np.nanargmax(hght[:,i])

for i in range(0, 31):
    lat[p[i]+1:-1,i]=np.nan
    lon[p[i]+1:-1,i]=np.nan
    hght[p[i]+1:-1,i]=np.nan
#*****************************************************************************\
# Temperature v/s pressure
#*****************************************************************************\

import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, MultipleLocator

fig = plt.figure(figsize=(10, 10))

# gs = gridspec.GridSpec(2, 2)

# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])
# ax3 = plt.subplot(gs[2, :])


ax1 = plt.subplot2grid((3,2), (0,0))
ax2 = plt.subplot2grid((3,2), (0,1))
ax3 = plt.subplot2grid((3,2), (1,0),colspan=2, rowspan=2)

# ax1 = plt.subplot2grid((2,3), (0,0))
# ax2 = plt.subplot2grid((2,3), (1,0))
# ax3 = plt.subplot2grid((2,3), (0,1),colspan=2, rowspan=3)


# ax1 = plt.subplot(221)
# ax2 = plt.subplot(223)
# ax3 = plt.subplot(122)


num_plots = 31
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

for i in range(0, num_plots):

    ax1.plot(lon[:,i],hght[:,i])
    ax1.set_xlim(140, 155)
    ax1.set_ylim(0, 30)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Altitude [km]')
    ax1.grid()

    ax2.plot(lat[:,i],hght[:,i])
    #ax1.set_xlim(140, 160)
    ax2.set_ylim(0, 30)
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Altitude [km]')
    ax2.grid()


    ax3.semilogy(temp[:,i], pres[:,i])
    ax3.yaxis.set_major_formatter(ScalarFormatter())
    ax3.set_yticks(np.linspace(100, 1000, 10))
    ax3.set_ylim(1050, 100)
    ax3.set_xlim(200, 310)
    ax3.set_ylabel('Pressure [hPa]')
    ax3.set_xlabel('Temperature [K]')
    ax3.grid()


fig.tight_layout()

# #*****************************************************************************\
# fig = plt.figure(figsize=(10, 10))
# ax3 = plt.subplot(111)
# for i in range(0, num_plots):
#     ax3.semilogy(temp[:,i], pres[:,i])
#     ax3.yaxis.set_major_formatter(ScalarFormatter())
#     ax3.set_yticks(np.linspace(100, 1000, 10))
#     ax3.set_ylim(1050, 100)
#     ax3.set_xlim(200, 320)
#     ax3.set_xlabel('Pressure [hPa]')
#     ax3.set_ylabel('Temperature [K]')
#     ax3.grid()


# fig.tight_layout()
plt.show()
