from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio

base_dir = os.path.expanduser('~')
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/figures/'
#******************************************************************************
# Setup Map
#******************************************************************************

# create new figure, axes instances.
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
map = Basemap(llcrnrlon=130.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=-30., rsphere=(6378137.00,6356752.3142), resolution='l',projection='merc',    lat_0=-20.,lon_0=180.,lat_ts=-50.)

map.drawcoastlines()
map.drawcountries()
map.fillcontinents()
map.drawmapboundary()
# draw parallels
map.drawparallels(np.arange(-90,-10,10),labels=[1,1,0,1])
# draw meridians
map.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])
#map.etopo()
#******************************************************************************
# Maquarie Island
#******************************************************************************
mac = {'name': 'Macquarie Island', 'lat': -54.62, 'lon': 158.85}
lat = mac['lat']
lon = mac['lon']

label = 'MAC'
x,y = map(lon, lat)
map.plot(x, y, 'sg', markersize=5)
#Label Island
plt.text(x+50000, y-100000, label,size=12)

#******************************************************************************
#Reading Mat File with cases
#******************************************************************************

path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/'
mat1= sio.loadmat(path_data+'caso_spared.mat')

lat_low= mat1['lat_low'][:]
lon_low= mat1['lon_low'][:]

Wlon=mat1['Wlon'][:]
Wlat=mat1['Wlat'][:]

Clon=mat1['Clon'][:]
Clat=mat1['Clat'][:]

# ni=5
# Wlon=Wlon[:,ni]
# Wlat=Wlat[:,ni]

#******************************************************************************
#Low centers
#******************************************************************************
xL,yL = map(lon_low, lat_low)
plt.text(xL,yL,'L',fontsize=18,fontweight='bold', ha='center',va='center',color='k')


circle_rad = 15  # This is the radius, in points
plt.plot(xL, yL, 'o', ms=circle_rad * 2, mec='k', mfc='none', mew=2)


#******************************************************************************
#Fronts
#******************************************************************************
xWF,yWF = map(Wlon, Wlat)
xCF,yCF = map(Clon, Clat)

sizeM=50

map.scatter(xWF,yWF,sizeM,marker='o',color='r',label='Warm Fronts', zorder=10)
#map.plot(xWF,yWF,sizeM,'-or',label='Warm Fronts')

map.scatter(xCF,yCF,sizeM,marker='>',color='b',label='Cold Fronts', zorder=10)

plt.legend(loc=3,numpoints=1,fontsize = 11)

plt.savefig(path_data_save + 'map_spared_fronts.eps', format='eps', dpi=1200)


#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
# No Spared
#******************************************************************************
#******************************************************************************
#******************************************************************************
# Setup Map
#******************************************************************************

# create new figure, axes instances.
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
map = Basemap(llcrnrlon=130.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=-30., rsphere=(6378137.00,6356752.3142), resolution='l',projection='merc',    lat_0=-20.,lon_0=180.,lat_ts=-50.)

map.drawcoastlines()
map.drawcountries()
map.fillcontinents()
map.drawmapboundary()
# draw parallels
map.drawparallels(np.arange(-90,-10,10),labels=[1,1,0,1])
# draw meridians
map.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])
#map.etopo()
#******************************************************************************
# Maquarie Island
#******************************************************************************
mac = {'name': 'Macquarie Island', 'lat': -54.62, 'lon': 158.85}
lat = mac['lat']
lon = mac['lon']

label = 'MAC'
x,y = map(lon, lat)
map.plot(x, y, 'sg', markersize=5)
#Label Island
plt.text(x+50000, y-100000, label,size=12)

#******************************************************************************
#Reading Mat File with cases
#******************************************************************************

path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/MatFiles/'
mat1= sio.loadmat(path_data+'caso_nospared.mat')

lat_low= mat1['lat_low'][:]
lon_low= mat1['lon_low'][:]

Wlon=mat1['Wlon'][:]
Wlat=mat1['Wlat'][:]

Clon=mat1['Clon'][:]
Clat=mat1['Clat'][:]

# ni=5
# Wlon=Wlon[:,ni]
# Wlat=Wlat[:,ni]

#******************************************************************************
#Low centers
#******************************************************************************
xL,yL = map(lon_low, lat_low)
plt.text(xL,yL,'L',fontsize=18,fontweight='bold', ha='center',va='center',color='k')


circle_rad = 15  # This is the radius, in points
plt.plot(xL, yL, 'o', ms=circle_rad * 2, mec='k', mfc='none', mew=2)


#******************************************************************************
#Fronts
#******************************************************************************
xWF,yWF = map(Wlon, Wlat)
xCF,yCF = map(Clon, Clat)

sizeM=50

map.scatter(xWF,yWF,sizeM,marker='o',color='r',label='Warm Fronts', zorder=10)
#map.plot(xWF,yWF,sizeM,'-or',label='Warm Fronts')

map.scatter(xCF,yCF,sizeM,marker='>',color='b',label='Cold Fronts', zorder=10)

plt.legend(loc=3,numpoints=1,fontsize = 11)


plt.savefig(path_data_save + 'map_nospared_fronts.eps', format='eps', dpi=1200)

plt.show()
