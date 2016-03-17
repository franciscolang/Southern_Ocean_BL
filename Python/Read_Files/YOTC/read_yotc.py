from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#from datetime import datetime
#import pandas as pd
#from glob import glob
#from scipy.interpolate import griddata

file = Dataset('data_files/2009_temp.nc', 'r')

print ' '
print ' '
print '----------------------------------------------------------'
for i,variable in enumerate(file.variables):
    print '   '+str(i),variable
    if i == 4:
        current_variable = variable
print ' '
print 'Variable: ', current_variable.upper()
#print 'File name:  ', current shape

lats  = file.variables['latitude'][:] #(240,0)
lons  = file.variables['longitude'][:] #(121,0)
level  = file.variables['level'][:] #(20,0)
time  = file.variables['time'][:] #(1460,)
temp = file.variables['t'][0,19,:,:] #(1460, 20, 121, 240)
#t = file.variables['t'][:]
#temp=t[0,1,:,:]

t_mac=temp[97,106]

t_units = file.variables['t'].units
file.close

#print len(lon)
print t_units

lon_0 = lons.mean()
lat_0 = lats.mean()
#Australia + New Zeland
#m = Basemap(projection='merc',llcrnrlon=110.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=-5.,resolution='i')
#MAC
#m = Basemap(projection='merc',llcrnrlon=158.,llcrnrlat=-55.,urcrnrlon=160.,urcrnrlat=-54.,resolution='i')
#World
m = Basemap(projection='merc',llcrnrlon=0.,llcrnrlat=-80.,urcrnrlon=360.,urcrnrlat=80.,resolution='i')

# Because our lon and lat variables are 1D, 
# use meshgrid to create 2D arrays 
# Not necessary if coordinates are already in 2D arrays.
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)

# Plot Data
cs = m.pcolor(xi,yi,np.squeeze(temp))

# Add Grid Lines
#m.drawparallels(np.arange(-55,-5,0.5), labels=[1,0,0,0], fontsize=10)
#m.drawmeridians(np.arange(115,180,0.5), labels=[0,0,0,1], fontsize=10)
#m.drawparallels(np.arange(-55,-5,10.), labels=[1,0,0,0], fontsize=10)
#m.drawmeridians(np.arange(115,180,10.), labels=[0,0,0,1], fontsize=10)

m.drawparallels(np.arange(-80,80,30.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(0,180,30.), labels=[0,0,0,1], fontsize=10)


# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')
m.drawcounties()

# Add Colorbar
#cbar = m.colorbar(cs, location='bottom', pad="10%")
#cbar.set_label(t_units)

# Add Title
#plt.title('Temperature')
#plt.show()

tempi = m.contourf(xi,yi,temp)
cb = m.colorbar(tempi,location='bottom', size="5%", pad="15%")
plt.title('Temperature')
cb.set_label('Temperature ' + t_units)
plt.show()
#plt.savefig('2m_temp.png')