import xarray as xr
from matplotlib import rcParams
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import numpy as np
import os
from matplotlib.colors import ListedColormap

cmap = plt.get_cmap('jet')#'Blues'
my_cmap = cmap(np.arange(cmap.N)) # Get the colormap colors
my_cmap[:,-1] = 1 - np.flip(np.geomspace(0.01, 1, cmap.N, endpoint=False)) # Set alpha
my_cmap = ListedColormap(my_cmap) # Create new colormap


params2 = {
    'font.family': 'sans serif',
    'lines.markersize': 2,
    'lines.linewidth': 2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.facecolor': 'whitesmoke',
    'axes.linewidth': 1.5,
    'legend.title_fontsize': 15,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.major.size': 8,'xtick.minor.size': 5,'xtick.major.width': 1.5,'xtick.minor.width': 1.,
    'ytick.major.size': 8,'ytick.minor.size': 5,'ytick.major.width': 1.5,'ytick.minor.width': 1.,
    'legend.frameon': False,
}

plt.rcParams.update(params2)

def label_plot(ax, proj, top=False,bottom=True,left=True,right=False):
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.7)
    ax.add_feature(cfeature.OCEAN, color="white", edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.7)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.25, color='grey', alpha=1., linestyle='--')
    gl.top_labels = top
    gl.bottom_labels = bottom
    gl.right_labels = right
    gl.left_labels = left


def plot_map(da, ax=None, proj=None, yaxis="lat", xaxis="lon", cmap=None, vmax=None, vmin=None, colorbar=True, top=False,bottom=True,left=True,right=False,
             colorbar_label="Precipitation [mm]", xlim=None, ylim=None,alpha=1.0,shrink=0.6):
    
    if proj is None: proj=ccrs.PlateCarree()
    if ax is None: ax = plt.axes(projection=proj)

    if colorbar == True: qm = da.plot.pcolormesh(xaxis, yaxis, cmap=cmap, ax=ax, transform=proj, vmax=vmax, vmin=vmin,alpha=alpha, add_colorbar=True,cbar_kwargs={'shrink':shrink})
    else: qm = da.plot.pcolormesh(xaxis, yaxis, cmap=cmap, ax=ax, transform=proj, vmax=vmax, vmin=vmin, add_colorbar=False,alpha=alpha)
    label_plot(ax, proj, top=top, bottom=bottom, left=left, right=right)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if colorbar:
        qm.colorbar.set_label(colorbar_label)


# # -----------------------------------------------------------------

def getCDF(data):

    # use np.percentile to calculate empirical cdf
    x = np.append(np.arange(0.5,100.,0.5),[99.8,99.9])
    cdf = data.chunk(dict(time=-1)).quantile(x/100.)
    #cdf = np.array(list([np.percentile(data,i) for i in x]))
    
    return(x,cdf)


# # -----------------------------------------------------------------
# Function to find annual climate metric, func, from a dataset, ds

def applyFunc(ds,func,thr=0.,time_var='time'):
    
    if time_var!='time':
        ds = ds.rename({time_var:'time'})
    
    # # first check if variable seems to be in K and convert to C if so
    # if ds.mean()>250.:
    #     ds = ds - 273.15
    
    if func == 'std':
        result = ds.groupby('time.year').std(skipna=True)
    
    elif func == 'sum':
        result = ds.groupby('time.year').sum(skipna=True)
    
    elif func == 'mean':
        result = ds.groupby('time.year').mean(skipna=True)
    
    elif func == 'max':
        result = ds.groupby('time.year').max(skipna=True)
    
    elif func == 'min':
        result = ds.groupby('time.year').min(skipna=True)
    
    elif func == 'q95':
        result = ds.chunk(dict(time=-1)).groupby('time.year').quantile(0.95)
    
    elif func == 'q99':
        result = ds.chunk(dict(time=-1)).groupby('time.year').quantile(0.99)
        
    elif func == 'count':
        result = ds.where(ds>thr).groupby('time.year').count()
    
    elif func == 'cwd':
        # find highest number of consecutive wet day spell each year
        t=3
        cwdmax = t
        windows = []
        while cwdmax==t:
            cwd = ds.where(ds>thr).rolling(time=t,center=True).count()
            cwd = cwd.groupby('time.year').max()
            windows.append(cwd)
            cwdmax = cwd.max()
            t+=1
            
        result = ds.where(ds>thr).rolling(time=30,center=True).count()
        result = result.groupby('time.year').max()
    
    elif func == 'cdd':
        # find highest number of consecutive dry day spell each year
        result = ds.where(ds==0.).rolling(time=90,center=True).count()
        result = result.groupby('time.year').max()
    
    elif func == 'r95ptot':
        if thr==0.:
            print('No threshold given. Calculating from ds.')
            thr = ds.chunk(dict(time=-1)).groupby('time.year').quantile(0.95)
            years = thr.year.values
            if len(years)>30:
                thr = thr.sel(year=slice(years[0],years[30])).mean('year')
            else: thr = thr.mean('year')
        
        annualtot = ds.groupby('time.year').sum().values
        result = ds.where(ds>thr).groupby('time.year').sum()/annualtot
    
    elif func == 'r99ptot':
        if thr==0.:
            print('No threshold given. Calculating from ds.')
            thr = ds.chunk(dict(time=-1)).groupby('time.year').quantile(0.99)
            years = thr.year.values
            thr = thr.sel(year=slice(years[0],years[30])).mean('year')
        
        annualtot = ds.groupby('time.year').sum().values
        result = ds.where(ds>thr).groupby('time.year').sum()/annualtot
    
    elif func == 'trend':
        result = ds.polyfit(dim="year", deg=1,skipna=True).polyfit_coefficients.sel(degree=1)
    
    else:
        print('This climate metric is not recognized')
    
    return(result)


# # -----------------------------------------------------------------
# # Function to read in statistics file for specific datasets

outfile = '/glade/campaign/ral/hap/hartke/stat_files/{product}_{scen}_{gcm}_{e}_*_*_stats.nc'
from glob import glob

def getStats(product,gcm='CESM2',e='1231_01',scen='historical',func='mean',r=None,time=None):
    
    print(glob(outfile.format(product=product,scen=scen,gcm=gcm,e=e))[0])
    
    try:
        ds = xr.open_dataset(glob(outfile.format(product=product,scen=scen,gcm=gcm,e=e))[0])
        try:
            ds = ds[func]
            if r!=None:
                ds = getRegion(ds,r)
            if time!=None:
                ds = ds.sel(year=slice(str(time[0]),str(time[1])))
        
            return(ds)
    
        except: print('%s does not exist in %s %s %s stats file.'%(func,scen,gcm,e))
        
    except: print('Stat file does not exist for %s %s %s run.'%(scen,gcm,e))

    

# # -----------------------------------------------------------------

def getRegion(ds, coords,latname='lat',lonname='lon',time=0.,time_var='time'):

    if time_var!='time':
        ds = ds.rename({time_var:'time'})
    
    if latname!='lat':
        ds = ds.rename({latname:'lat'})
        
    if lonname!='lon':
        ds = ds.rename({lonname:'lon'})
    
    ds = ds.reindex(lat=np.sort(ds.lat))
    
    if len(coords)==2:
        
        if ds.lon.max().values>180.:
            dsnew = ds.sel(lat=coords[0],lon=coords[1]+360.,method='nearest')
        
        else:
            dsnew = ds.sel(lat=coords[0],lon=coords[1],method='nearest')
    
    elif len(coords)==4:
        
        if ds.lon.max().values>180.:
            dsnew = ds.sel(lat=slice(coords[0],coords[1]),lon=slice(coords[2]+360.,coords[3]+360.))
        
        elif ds.lon[0].values<ds.lon[-1].values:
            dsnew = ds.sel(lat=slice(coords[0],coords[1]),lon=slice(coords[2],coords[3]))
        
        else:
            print(ds.lon[0].values,ds.lon[-1].values)
            dsnew = ds.sel(lat=slice(coords[0],coords[1]),lon=slice(coords[3],coords[2]))
    
    else:
        print('Coordinate format not recognized')
    
    if time!=0.:
        dsnew = dsnew.sel(time=slice(time[0],time[1]))

    if time_var!='time':
        dsnew = dsnew.rename({'time':time_var})
            
    return(dsnew)

# # ----------------------------------------------------------


## --------------------------------------------------------------------------------------     
# VALIDATION/OBSERVATION PRODUCTS
gmet_file = "/glade/campaign/ral/hap/anewman/conus_v1p2/eighth/v2_landmask/conus_daily_eighth_{yr}0101_{yr}1231_{ens_mem}.nc4"
gmetAK_file = "/glade/campaign/ral/hap/hartke/gmet/alaska_rectilinear_daily_{yr}.nc"
gmetHI_file = "/glade/campaign/ral/hap/anewman/ncar_hawaii_ensemble/hawaii_daily_{yr}_{ens_mem}.nc4"
prism_file = "/glade/campaign/ral/hap/common/prism/data/{var}/PRISM_daily_{var2}_{yr}.nc"
era5_file = '/glade/campaign/ral/hap/hartke/era5/era5_daily_{yr}{runtag}.nc'
livneh_file = '/glade/campaign/ral/hap/common/Livneh_met_updated/precip/livneh_unsplit_precip.2021-05-02.{yr}.nc'
livneh_temp_file = '/glade/campaign/ral/hap/common/Livneh_met_updated/temp_and_wind/livneh_lusu_2020_temp_and_wind.2021-05-02.{yr}.nc'
nclimgrid_file = '/glade/campaign/ral/hap/hartke/era5/'

# GCMS
canesm_template = '/glade/campaign/ral/hap/hartke/canesm5/canesm_daily_{yr1}_{yr2}_{ens}{runtag}.nc'
cesm_template = "/glade/campaign/ral/hap/hartke/cesmlens2/cesm_daily_{yr1}_{yr2}_{ens}{runtag}.nc"
ecearth_template = "/glade/campaign/ral/hap/hartke/ecearth3/ecearth3_daily_{yr1}_{yr2}_{ens}{runtag}.nc"

# GARD-LENS
gard_final = '/glade/campaign/ral/hap/hartke/GARD_LENS_final/{var}/GARDLENS_{gcm}_{ens}_{var}_{styr}_2100_{runtag}.nc'
gard_template = '/glade/campaign/ral/hap/hartke/gard_output/{gcm}_{obs}/{ens}_{pred}/processed{runtag}_{var}{n}.nc'
gard_raw = '/glade/campaign/ral/hap/hartke/gard_output/{gcm}_{obs}/{ens}_{pred}/gard{runtag}_out_{var}.nc'
gard_error = '/glade/campaign/ral/hap/hartke/gard_output/{gcm}_{obs}/{ens}_{pred}/gard{runtag}_out_{var}_errors.nc'
gard_logistic = '/glade/campaign/ral/hap/hartke/gard_output/{gcm}_{obs}/{ens}_{pred}/gard{runtag}_out_{var}_logistic.nc'

# DOWNSCALED DATASETS
ds_wd = '/glade/campaign/ral/hap/gutmann/downscaled_data/'
maca_file = ds_wd + 'MACA/macav2livneh_{var}_{gcm}_{ens}_{scen}_{yr1}_{yr2}_CONUS_daily.nc'
dbcca_file = ds_wd + 'ORNL_DBCCA/{gcm}_{scen}_r1i1p1f1_DBCCA_{valdata}_VIC4_{var}_{yr1}.nc'
regcm_file = ds_wd + 'ORNL_RegCM/{gcm}_{scen}_r1i1p1f1_RegCM_{valdata}_VIC4_{var}_{yr1}.nc'
gmfd_file = '/glade/campaign/collections/rda/data/ds314.0/0.25deg/daily/{var}_0p25_daily_{yr}-{yr}.nc'
star_file = '/glade/campaign/ral/hap/common/STAR-ESDM/{scen}/{gcm}/{var}/downscaled.{gcm}.{ens}.{var}.{scen}.gn.nclimgrid.star.{yr}.tllc.nc'
loca_template = "/glade/campaign/ral/hap/anewman/loca2/{gcm}/0p0625deg/{ens}/{scen}/{var}/{var}.{gcm}.{scen}.{ens}.{yr}.LOCA_16thdeg_{version}.nc"
wrf_file = '/glade/campaign/ral/hap/hartke/wrf/wrf_daily_{yr}.nc'
nex_file = ds_wd + 'NASA_NEX_BCSD/{var}/{scen}/{var}_day_{gcm}_{scen}_{ens}_{g}_{yr}.nc'

# Define file name templates and study region coordinates

ens_list = {'cesmlens2':[],
           'canesm5':[],
           'ecearth3':[]}

# write list of LENS2 ensemble members
for styr in (1231, 1251, 1281, 1301):
    for e in range(1,21): ens_list['cesmlens2'].append("%d_%02d"%(styr,e)) 
for styr in np.arange(1001,1192,10):
    if (styr-1)%20==0: e = int((styr-1001)/20 + 1)
    else: e = int((styr-1011)/20 + 1)
    ens_list['cesmlens2'].append("%d_%02d"%(styr,e))

loca_ens = [filename for filename in os.listdir('/glade/campaign/ral/hap/anewman/loca2/CESM2-LENS/0p0625deg/')]


canesm_dir = '/glade/collections/cmip/CMIP6/CMIP/CCCma/CanESM5/historical/'
ens_list['canesm5'] = np.sort([filename for filename in os.listdir(canesm_dir) if len(glob(canesm_dir+'%s/*day*/pr/gn/v20*/pr/pr_*day*_CanESM5_historical_%s_gn_18500101-20141231.nc'%(filename,filename)))>0])


ssp_dir = '/glade/collections/cmip/CMIP6/ScenarioMIP/EC-Earth-Consortium/EC-Earth3/ssp370/'
gcm_file = '{ens}/day/ta/{g}/v20210101/ta/ta_day_{gcm}_{scen}_{ens}_{g}_20150101-20151231.nc'
ens_list['ecearth3'] = np.sort([filename for filename in os.listdir(ssp_dir) if os.path.exists(ssp_dir+gcm_file.format(ens=filename,scen='ssp370',gcm='EC-Earth3',g='gr'))])


templates = {'GMET':gmet_file,
             'GMET_AK':gmetAK_file,
             'GMET_HI':gmetHI_file,
             'PRISM':prism_file,
             'GMFD':gmfd_file,
             'GARD':gard_template,
             'GARDfinal':gard_final,
             'GARDraw':gard_raw,
             'GARDerr':gard_error,
             'GARDlog':gard_logistic,
             'CESM':cesm_template,
             'cesmlens2':cesm_template,
             'canesm5':canesm_template,
             'ecearth3':ecearth_template,
             'LOCA':loca_template,
             'CanESM':canesm_template,
             'ERA5':era5_file,
             'WRF':wrf_file,
             'ClimGrid':'/glade/campaign/ral/hap/hartke/nClimGrid/ncdd-{yr}{mnth}-grd-scaled.nc',
             'Livneh':livneh_file,
             'LivnehTemp':livneh_temp_file,
             'NEX':nex_file,
             'MACA':maca_file,
             'DBCCA':dbcca_file,
             'REGCM':regcm_file,
             'STAR':star_file,
            }


gcm_yr_list = {'cesmlens2':np.concatenate((np.arange(1850,2011,10),np.arange(2015,2100,10))),
               'ecearth3':np.concatenate((np.arange(1970,2011,10),[2015],np.arange(2020,2100,10))),
               'canesm5':np.concatenate((np.arange(1901,2012,10),[2015],np.arange(2021,2100,10))),
               'canesm5_2':np.concatenate(([1950],np.arange(1961,2012,10),[2015],np.arange(2021,2100,10))),
              }

# gcm_yr_list = {'cesmlens2':np.concatenate((np.arange(1850,2011,10),np.arange(2015,2100,10))),
#               'ecearth3':np.concatenate((np.arange(1970,2011,10),[2015],np.arange(2020,2100,10))),
#               'canesm5':np.concatenate(([1850],np.arange(1861,2012,10),[2015],np.arange(2021,2100,10))),
#               'canesm5_2':np.concatenate(([1950],np.arange(1961,2012,10),[2015],np.arange(2021,2100,10))),
#              }

canesm_regdata = ('r11i1p2f1','r12i1p2f1','r13i1p2f1','r14i1p2f1','r15i1p2f1','r16i1p2f1','r17i1p2f1','r18i1p2f1',
                  'r19i1p2f1','r20i1p2f1','r21i1p2f1','r22i1p2f1','r23i1p2f1','r24i1p2f1','r25i1p2f1')

regions = {'WA':(45.5, 49., -124., -116.),
           'WI':(42.5, 47., -93., -87.),
          'SE':(30.5, 35., -90., -82.),
          'CO':(37, 41, -109, -102),
           'FR':(39,41,-106,-104),
           'LA':(29, 33, -94, -89.),
           'NOLA':(29.95, -90.07),
           'SEAT':(47.61, -122.33),
           'DENV':(39.74, -104.99),
           'HONL':(21.31, -157.85), # Honolulu
           'POHA':(19.75, -155.54), #Pohakuloa Training Area
           'YUMA':(32.69,-114.63),
           'NYNY':(40.71,-74.01),
           'MIAM':(25.76, -80.19),
           'CHIC':(41.88, -87.63),
           'DOVR':(39.128, -75.47),
           'CONUS':(25., 50., -126., -67.),
           'AK':(52.,72.,-170.,-130.),
           'HI':(18.5,23.,-160.,-154.),
           'FTWN':(64.83,-147.64),
           'ANC':(61.22,-149.9),
           'FTLB':(35.14,-79.),
          }

reg_names = {'WA':'Washington',
             'WI':'Wisconsin',
             'SE':'Southeast',
             'CO':'Colorado',
             'LA':'Louisiana',
             'CONUS':'Contiguous U.S.',
        }


pcp_var_names = {'cesmlens2':'PRECT',
                 'ERA5':'PRECT',
                 'ecearth3':'pr',
                 'canesm5':'pr',
                 'Livneh':'PRCP',
                 'PRISM':'PR',
                 'PRISM2':'ppt',
}

t_mean_var_names = {'cesmlens2':'TREFHT',
                 'ERA5':'T2',
                 'ecearth3':'tas',
                 'canesm5':'tas',
                 'Livneh':'PRCP',
                 'PRISM':'T2M',
                    'PRISM2':'tmean',
}

var_name = {'pcp':pcp_var_names,
            't_mean':t_mean_var_names,
}

gcm_dict = {'cesmlens2':'CESM2-LE',
           'canesm5':'CanESM5',
           'ecearth3':'EC-Earth3'}


