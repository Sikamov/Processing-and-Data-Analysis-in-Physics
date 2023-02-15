#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:09:03 2019

@author: kirillsikamov
"""


from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import json


with netcdf.netcdf_file('MSR-2.nc', mmap=False) as netcdf_file:
     var = netcdf_file.variables

#Координаты города Оттава
coord= np.array([45.41117, -75.69812])     

#Сохраним значения в файл json
dic = {
  "city": "Ottawa",
  "coordinates": [ coord[0], coord[1]],
  "jan": {
    "min": float( np.min(var['Average_O3_column'].data[::12, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])])),
    "max": float( np.max(var['Average_O3_column'].data[::12, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])])),
    "mean": float( np.mean(var['Average_O3_column'].data[::12, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])]))
  },
  "jul": {
    "min": float(np.min(var['Average_O3_column'].data[6::12, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])])),
    "max": float(np.max(var['Average_O3_column'].data[6::12, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])])),
    "mean": float(np.mean(var['Average_O3_column'].data[6::12, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])]))
  },
  "all": {
    "min": float(np.min(var['Average_O3_column'].data[:, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])])),
    "max": float(np.max(var['Average_O3_column'].data[:, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])])),
    "mean": float(np.mean(var['Average_O3_column'].data[:, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])]))
  }
}
with open('ozon.json', 'w') as file:
    json.dump(dic, file, indent=4)
    
#Построим графики и сохраним 
plt.figure()
plt.plot(var['time'].data-108, var['Average_O3_column'].data[:, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])], label='$O_3$')
plt.plot((var['time'].data-108)[::12], var['Average_O3_column'].data[::12, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])],'o', label='$O_3$jan')
plt.plot((var['time'].data-108)[6::12], var['Average_O3_column'].data[6::12, np.searchsorted(var['latitude'].data, coord[0]),np.searchsorted(var['longitude'].data, coord[1])],'o', label='$O_3$jul')
plt.legend()
plt.grid()
plt.savefig('ozon.png')

   
    