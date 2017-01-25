from mpl_toolkits.basemap import Basemap, addcyclic
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
from scipy.ndimage.filters import minimum_filter, maximum_filter

base_dir = os.path.expanduser('~')
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/004 Paper/Figures/'
#******************************************************************************
# Setup Map
#******************************************************************************
def extrema(mat,mode='wrap',window=10):
    """find the indices of local extrema (min and max)
    in the input array."""
    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    # (mat == mn) true if pixel is equal to the local in
    # Return the indices of the maxima, minima
    return np.nonzero(mat == mn), np.nonzero(mat == mx)

#******************************************************************************
# Reading NetCDF
#******************************************************************************
data = base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/ERAI/map/2010_03_mslp.nc'
fh =Dataset(data, mode='r')

lons1 = fh.variables['longitude'][:]
lats = fh.variables['latitude'][:]
spres = fh.variables['msl'][:]
time = fh.variables['time'][:]
time_units = fh.variables['time'].units
spres_units = fh.variables['msl'].units
#fh.close()

date_ini=datetime(1900, 1, 1) + timedelta(hours=int(time[0]))
date_erai = pd.date_range(date_ini, periods=len(time), freq='6H')
#56 is '2010-03-15 00:00:00'

prmsl=spres[56,:,:]*0.01
#******************************************************************************
lon_0 = lons1.mean()
lat_0 = lats.mean()

fig=plt.figure()
ax = fig.add_axes([0.05,0.05,0.9,0.85])

# map = Basemap(llcrnrlon=120.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=-30., rsphere=(6378137.00,6356752.3142), resolution='l',projection='lcc',    lat_0=-20.,lon_0=180.,lat_ts=-50.)
# # map = Basemap(llcrnrlon=90.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=-20., rsphere=(6378137.00,6356752.3142), resolution='l',projection='lcc',    lat_0=-20.,lon_0=180.,lat_ts=-50.)
# map.drawparallels(np.arange(-90,-10,10),labels=[0,1,0,0]) #left, right, top or bottom of the plot.
# map.drawmeridians(np.arange(-180,180,10),labels=[0,0,0,1])


map = Basemap(projection='cyl',llcrnrlon=130,llcrnrlat=-70,urcrnrlon=180,urcrnrlat=-30,resolution='l')
# plot (unwarped) rgba image.
im = map.bluemarble(scale=0.5)
map.drawcoastlines(linewidth=0.5,color='0.5')
map.drawmeridians(np.arange(-180,180,10),labels=[0,0,0,1],color='0.5')
map.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0],color='0.5')


local_min, local_max = extrema(prmsl, mode='wrap', window=50)
prmsl, lons = addcyclic(prmsl, lons1)
clevs = np.arange(900,1100.,5)
lons, lats = np.meshgrid(lons, lats)
x, y = map(lons, lats)


cs = map.contour(x,y,prmsl,clevs,colors='w',linewidths=1.)
map.drawcoastlines(linewidth=1.25)
#map.fillcontinents(color='0.8')
#map.etopo()
map.bluemarble()
#map.shadedrelief()

xlows = x[local_min]; xhighs = x[local_max]
ylows = y[local_min]; yhighs = y[local_max]
lowvals = prmsl[local_min]; highvals = prmsl[local_max]
# plot lows as blue L's, with min pressure value underneath.
xyplotted = []
# don't plot if there is already a L or H within dmin meters.
yoffset = 0.022*(map.ymax-map.ymin)
dmin = yoffset
for x,y,p in zip(xlows, ylows, lowvals):
    if x < map.xmax and x > map.xmin and y < map.ymax and y > map.ymin:
        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
        if not dist or min(dist) > dmin:
            plt.text(x,y,'L',fontsize=14,fontweight='bold',
                    ha='center',va='center',color='b')
            plt.text(x,y-yoffset,repr(int(p)),fontsize=9,
                    ha='center',va='top',color='b',
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
            xyplotted.append((x,y))
# plot highs as red H's, with max pressure value underneath.
xyplotted = []
for x,y,p in zip(xhighs, yhighs, highvals):
    if x < map.xmax and x > map.xmin and y < map.ymax and y > map.ymin:
        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
        if not dist or min(dist) > dmin:
            plt.text(x,y,'H',fontsize=14,fontweight='bold',
                    ha='center',va='center',color='tomato',bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
            #plt.text(x,y-yoffset,repr(int(p)),fontsize=9,
            #        ha='center',va='top',color='r',
            #        bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
            xyplotted.append((x,y))


# xi, yi = map(lon, lat)

# cs = map.pcolor(xi,yi,np.squeeze(pres))

# map.drawcoastlines()
# map.drawcountries()

# # Add Colorbar
# cbar = map.colorbar(cs, location='bottom', pad="10%")
# cbar.set_label(spres_units)



#******************************************************************************
# Maquarie Island
#******************************************************************************
mac = {'name': 'Macquarie Island', 'lat': -54.62, 'lon': 158.85}
lat = mac['lat']
lon = mac['lon']

label = 'Macquarie Island'
x,y = map(lon, lat)
map.plot(x, y, 'sg', markersize=5, color='y')
#Label Island
plt.text(x+1, y-1, label,size=12, color='w')

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
# plt.text(xL,yL,'L',fontsize=30,fontweight='bold', ha='center',va='center',color='w')


plt.text(xL,yL,'L',fontsize=14,fontweight='bold', ha='center',va='center',color='b', bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))


# circle_rad = 15  # This is the radius, in points
# plt.plot(xL, yL, 'o', ms=circle_rad * 2, mec='k', mfc='none', mew=2)


#******************************************************************************
#Fronts
#******************************************************************************
xWF,yWF = map(Wlon, Wlat)
xCF,yCF = map(Clon, Clat)

sizeM=50

map.scatter(xWF,yWF,sizeM,marker='o',color='r',label='Warm Fronts', zorder=10)
#map.plot(xWF,yWF,sizeM,'-or',label='Warm Fronts')

map.scatter(xCF,yCF,sizeM,marker='>',color='cyan',label='Cold Fronts', zorder=10)

plt.legend(loc=4,numpoints=1,fontsize = 12)

plt.savefig(path_data_save + 'map_spared_fronts.eps', format='eps', dpi=300, bbox_inches='tight')

plt.show()
