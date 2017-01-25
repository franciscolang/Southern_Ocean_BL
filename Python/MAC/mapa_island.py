from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import os


base_dir = os.path.expanduser('~')
path_data_save=base_dir+'/Dropbox/Monash_Uni/SO/MAC/004 Paper/Figures/'
#******************************************************************************
fig = plt.figure()
ax = fig.add_subplot(111)

# map = Basemap(llcrnrlon=140.,llcrnrlat=-70.,urcrnrlon=180.,urcrnrlat=-30., rsphere=(6378137.00,6356752.3142), resolution='l',projection='cyl',    lat_0=-20.,lon_0=180.,lat_ts=-50.)

# map = Basemap(llcrnrlon=100.,llcrnrlat=-70.,urcrnrlon=190.,urcrnrlat=-10., rsphere=(6378137.00,6356752.3142), resolution='l',projection='cyl',    lat_0=-20.,lon_0=180.,lat_ts=-50.)

# map.drawmapboundary()
# map.fillcontinents()
# map.drawcoastlines()
# map.drawparallels(np.arange(-90,-10,10),labels=[0,1,0,0])
# map.drawmeridians(np.arange(-180,180,10),labels=[0,0,0,1])

map = Basemap(projection='cyl',llcrnrlon=110,llcrnrlat=-70,urcrnrlon=180,urcrnrlat=-10,resolution='l')
# plot (unwarped) rgba image.
im = map.bluemarble(scale=0.5)
map.drawcoastlines(linewidth=0.5,color='0.5')
map.drawmeridians(np.arange(-180,180,10),labels=[0,0,0,1],color='0.5')
map.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0],color='0.5')


#******************************************************************************
# Sub Map
#******************************************************************************
axins = zoomed_inset_axes(ax, 50, loc=4)
axins.set_xlim(157, 159)
axins.set_ylim(-56, -54)

plt.xticks(visible=False)
plt.yticks(visible=False)

map2 = Basemap(resolution='h', llcrnrlon=158.75,llcrnrlat=-54.8,urcrnrlon=159.01,urcrnrlat=-54.43, ax=axins)
map2.drawmapboundary()
map2.fillcontinents(color='coral')
map2.drawcoastlines()
map2.drawcountries()

mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
#******************************************************************************
mac = {'name': 'Macquarie Island', 'lat': -54.49, 'lon': 158.962}
lat = mac['lat']
lon = mac['lon']
label = 'station'
x,y = map(lon, lat)
map2.scatter(x,y,30,marker='o',color='b', zorder=10)
plt.text(x-0.1, y+0.028, label,size=10, color='k')

plt.savefig(path_data_save + 'mfig1.eps', format='eps', dpi=200, bbox_inches='tight')
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************

plt.show()
