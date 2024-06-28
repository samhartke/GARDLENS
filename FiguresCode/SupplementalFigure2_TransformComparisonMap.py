
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.cm as cm
import os
import random
from datetime import datetime,timedelta
from glob import glob
import intake
import xesmf as xe
import scipy as sp
from analysisFuncs import plot_map, ens_list,gcm_yr_list,regions,templates, getRegion, my_cmap, applyFunc
import matplotlib as mpl
proj = ccrs.PlateCarree()
from dask_jobqueue import PBSCluster
import dask
from dask.distributed import Client


cluster = PBSCluster(
    queue="casper",
    walltime="6:00:00",
    project="P48500028",
    memory="10GB",
    cores=1,
    processes=1,
)

cluster.scale(12)

mask = xr.open_dataset('conus_gmet_mask2.nc')['mask'].values

# # ---- Plot map of change in 2070-2100 projected maximum daily t_mean across U.S. as a result of new transform


# var = 'pcp'
# pred1 = 'pcp_tuv'
# pred2 = 'pcp'

var = 't_mean'
pred1 = 'pALL'
pred2 = 'pALL'
runtag1 = ''
runtag2 = '_analog20'
percentage = False # Decide whether to plot differences as a percentage

print('Comparing %s %s GARD runs with %s %s GARD runs for the variable %s.'%(pred1,runtag1,pred2,runtag2,var))

diff_template = False

if pred1!=pred2:
    comp = '%s to %s'%(pred1,pred2)
    tag = '%svs%s'%(pred1,pred2)
    template = templates['GARD']

elif runtag1!=runtag2:
    comparison='runtag'
    comp = '%s to %s'%(runtag1,runtag2)
    tag = '%svs%s'%(runtag1,runtag2)
    if (runtag1=='_newT10') or (runtag1==''):
        template = '/glade/campaign/ral/hap/hartke/TransformAnalysis/GARD_{gcm}_{var}_{ens}{runtag}.nc'
        print('Using template ',template)
        diff_template=True
    else: template = templates['GARD']

if percentage==True:
    vmin=-10.;vmax=10.
    if var=='pcp':
        vmin=-15.;vmax=15.
else:
    vmin=-5.;vmax=5.

fig = plt.figure(figsize=(14,8))
gcm = 'cesmlens2'
ylim=(25.,50.)

gard_mean = []
for e in ens_list[gcm][:10]:
    if diff_template==True: ds = xr.open_dataset(template.format(gcm=gcm,ens=e,var=var,runtag=runtag1))
    else: ds = xr.open_dataset(template.format(gcm=gcm,ens=e,var=var,runtag=runtag1,pred=pred1,obs='gmet',n=0))
    ds = ds.assign(y=ds.lat[:,0].values).assign(x=ds.lon[0,:].values-360).drop_vars(('lat','lon')).rename({'y':'lat','x':'lon'})
    gard_mean.append(ds[var].sel(time=slice('1980','2100')))

gard_mean_1 = applyFunc(xr.concat(gard_mean,dim='n_ens'),'mean').load()
gard_max_1 = applyFunc(xr.concat(gard_mean,dim='n_ens'),'max').load()
gard_mean = None

print('Datasets retrieved')
template = templates['GARD']
diff_template = False
print(template)

gard_mean = []
for e in ens_list[gcm][:10]:
    if diff_template==True: ds = xr.open_dataset(template.format(gcm=gcm,ens=e,var=var,runtag=runtag2))
    else: ds = xr.open_dataset(templates['GARD'].format(gcm=gcm,ens=e,var=var,runtag=runtag2,pred=pred2,obs='gmet',n=0))
    ds = ds.assign(y=ds.lat[:,0].values).assign(x=ds.lon[0,:].values-360).drop_vars(('lat','lon')).rename({'y':'lat','x':'lon'})
    gard_mean.append(ds[var].sel(time=slice('1980','2100')))

gard_mean_2 = applyFunc(xr.concat(gard_mean,dim='n_ens'),'mean').load()
gard_max_2 = applyFunc(xr.concat(gard_mean,dim='n_ens'),'max').load()
gard_mean = None

print('Datasets retrieved')

ax1 = fig.add_subplot(2,2,1,projection=proj)
diff = gard_mean_2.sel(year=slice('1980','2014')).mean(('year','n_ens')) - gard_mean_1.sel(year=slice('1980','2014')).mean(('year','n_ens'))
if percentage == True:
    diff = 100*diff/gard_mean_1.sel(year=slice('1980','2014')).mean(('year','n_ens'))
plot_map(diff.where(mask),ax=ax1,cmap='seismic',vmin=vmin,vmax=vmax,ylim=ylim,colorbar=False,left=False,bottom=False)
ax1.set_title('Change in 1980-2014 annual mean %s\nfrom %s'%(var,comp))

ax1 = fig.add_subplot(2,2,2,projection=proj)
diff = gard_max_2.sel(year=slice('1980','2014')).mean(('year','n_ens')) - gard_max_1.sel(year=slice('1980','2014')).mean(('year','n_ens'))
if percentage == True:
    diff = 100*diff/gard_max_1.sel(year=slice('1980','2014')).mean(('year','n_ens'))
plot_map(diff.where(mask),ax=ax1,cmap='seismic',vmin=vmin,vmax=vmax,ylim=ylim,colorbar=False,left=False,bottom=False)
ax1.set_title('Change in 1980-2014 annual max %s\nfrom %s'%(var,comp))

ax1 = fig.add_subplot(2,2,3,projection=proj)
diff = gard_mean_2.sel(year=slice('2070','2100')).mean(('year','n_ens')) - gard_mean_1.sel(year=slice('2070','2100')).mean(('year','n_ens'))
if percentage == True:
    diff = 100*diff/gard_mean_1.sel(year=slice('2070','2100')).mean(('year','n_ens'))
plot_map(diff.where(mask),ax=ax1,cmap='seismic',vmin=vmin,vmax=vmax,ylim=ylim,colorbar=False,left=False,bottom=False)
ax1.set_title('Change in 2070-2100 annual mean %s\nfrom %s'%(var,comp))

ax1 = fig.add_subplot(2,2,4,projection=proj)
diff = gard_max_2.sel(year=slice('2070','2100')).mean(('year','n_ens')) - gard_max_1.sel(year=slice('2070','2100')).mean(('year','n_ens'))
if percentage == True:
    diff = 100*diff/gard_max_1.sel(year=slice('2070','2100')).mean(('year','n_ens'))
plot_map(diff.where(mask),ax=ax1,cmap='seismic',vmin=vmin,vmax=vmax,ylim=ylim,colorbar=False,left=False,bottom=False)
ax1.set_title('Change in 2070-2100 annual max %s\nfrom %s'%(var,comp))


if var=='pcp': units = 'mm/d'
elif (var=='t_mean') or (var=='t_range'): units = 'C'
if percentage==True: units = '%'    
ax2 = fig.add_axes([0.37, 0.09, 0.25, 0.02])
cb = mpl.colorbar.ColorbarBase(ax2, orientation='horizontal',cmap='seismic',
                               norm=mpl.colors.Normalize(vmin, vmax),  # vmax and vmin
                               extend='both',label='change in %s [%s]'%(var,units))

if percentage==True: tag = tag+'_perc'
plt.savefig('figures/TransformComparisonMap_%s_%s_%s.jpg'%(gcm,var,tag),bbox_inches='tight',dpi=1200)

plt.show()


