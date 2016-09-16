import numpy as np
import scipy.io as sio
import pandas as pd
from datetime import datetime
import os
import csv
from matplotlib.ticker import ScalarFormatter, MultipleLocator
import matplotlib.mlab as mlab
import scipy as sp
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as si
from scipy.interpolate import interp1d
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

base_dir = os.path.expanduser('~')
path_data=base_dir+'/Dropbox/Monash_Uni/SO/MAC/Data/00 CSV/'

latMac=-54.50;
lonMac=158.95;
#*****************************************************************************\
#*****************************************************************************\
# Reading CSV with Boundary Layer Clasification
#*****************************************************************************\
#*****************************************************************************\
df_mac_cyc= pd.read_csv(path_data + 'df_maccyc.csv', sep='\t', parse_dates=['Date'])
df_mac_cyc['Date'].index


#*****************************************************************************\
from pylab import plot,show, grid

nc=0
plot(df_mac_cyc['lon'][nc],df_mac_cyc['lat'][nc],'ko',latMac,lonMac,'r*')
grid()
show()
