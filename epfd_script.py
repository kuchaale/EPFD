from netCDF4 import Dataset
import numpy as np #version 1.9.1
import sys
from scipy.interpolate import interp1d
import time
import bottleneck as bn #version 1.0.0

def c_diff(arr, h):
#	print arr.shape
	d_arr = np.copy(arr)
	d_arr[0,...] = (arr[1,...]-arr[0,...])/(h[1]-h[0])
	d_arr[-1,...] = (arr[-1,...]-arr[-2,...])/(h[-1]-h[-2])
	d_arr[1:-1,...] = (arr[2:,...]-arr[0:-2,...])/(np.reshape(h[2:]-h[0:-2],(arr.shape[0]-2,1,1)))
	return d_arr

def rmv_mean(arr):
	return arr-bn.nanmean(arr,axis=3)[...,np.newaxis]

def interp(lev, data, lev_int):
	f = interp1d(lev[::-1],data[:,::-1,:],axis=1)
	return f(lev_int[::-1])[:,::-1,:]

in_file = sys.argv[1] #input file
scale_by_sqrt_p = True #scaled by square root of (1000/p)
inter_bool = True #interpolate to regular vertical grid

dataset = Dataset(in_file, 'r')

t = dataset.variables['t'][:]
u = dataset.variables['u'][:]
v = dataset.variables['v'][:]
o = dataset.variables['omega'][:]
lon = dataset.variables['lon'][:]
lat = dataset.variables['lat'][:]
lev = dataset.variables['level'][:]
tim = dataset.variables['time'][:]
units = dataset.variables['time'].units

nlev = lev.shape[0]
nlat = lat.shape[0]
nlon = lon.shape[0]
ntime = tim.shape[0]

theta = t*(np.reshape(lev,(1,nlev,1,1))/1000.)**(-0.286)
theta_zm = bn.nanmean(theta, axis = 3)
loglevel = np.log(lev)

THETAp = np.transpose(c_diff(np.transpose(theta_zm,[1,0,2]), loglevel),[1,0,2])
THETAp /= 100.*np.reshape(lev,(1,nlev,1))

Uza = rmv_mean(u)
Vza = rmv_mean(v)
THETAza = rmv_mean(theta)

UV = Uza*Vza
VTHETA = Vza*THETAza

UVzm = bn.nanmean(UV, axis=3)
VTHETAzm = bn.nanmean(VTHETA,axis=3)

#constants
a = 6.37122e06 
PI = np.pi
phi = lat*PI/180.0     
acphi=a*np.cos(phi)       
asphi=a*np.sin(phi)       
omega = 7.2921e-5      
f = 2*omega*np.sin(phi) 
latfac=acphi*np.cos(phi)


Fphi = -UVzm*np.reshape(latfac,(1,1,nlat))
Fp = np.reshape(f*acphi,(1,1,nlat))*VTHETAzm/THETAp


Fdiv1 = np.transpose(c_diff(np.transpose(Fphi,[2,1,0]), asphi),[2,1,0])
Fdiv2 = np.transpose(c_diff(np.transpose(Fp,[1,0,2]), lev*100.),[1,0,2])
Fdiv = Fdiv1 + Fdiv2


#residual circulation
Vzm = bn.nanmean(v, axis = 3)
V_res = Vzm - np.transpose(c_diff(np.transpose(VTHETAzm/THETAp,[1,0,2]), lev*100.),[1,0,2])
Ozm = bn.nanmean(o, axis = 3)
O_res = Ozm + np.transpose(c_diff(np.transpose((VTHETAzm*np.reshape(np.cos(phi),(1,1,nlat)))/THETAp,[2,1,0]),asphi),[2,1,0])


if inter_bool:
	lev_int = 10**np.linspace(2,-0.8,15)  
	Fp_int = interp(lev, Fp, lev_int)
	Fphi_int = interp(lev, Fphi, lev_int)  
	Fdiv_int = interp(lev, Fdiv, lev_int)
	V_res_int = interp(lev, V_res, lev_int)
	O_res_int = interp(lev, O_res, lev_int)
	nlev2 = lev_int.shape[0]       
else:
	nlev2 = nlev
	lev_int = lev
	Fp_int = np.copy(Fp)
	Fphi_int = np.copy(Fphi) 
	Fdiv_int = np.copy(Fdiv)
	V_res_int = np.copy(V_res) 
	O_res_int = np.copy(O_res)


Fp_int = Fp_int*np.reshape(np.cos(phi),(1,1,nlat))   
Fphi_int = Fphi_int/a

Fp_int = Fp_int/1.0e4
Fphi_int = Fphi_int/PI

if scale_by_sqrt_p:
	rhofac = np.sqrt(100./lev_int) 
	Fp_int = Fp_int*np.reshape(rhofac,(1,nlev2,1))
	Fphi_int = Fphi_int*np.reshape(rhofac,(1,nlev2,1))

#output NetCDF file
f = Dataset('EPFD_'+in_file, 'w', format='NETCDF3_CLASSIC')
f.createDimension('lat', nlat)
f.createDimension('lev', nlev2)
f.createDimension('time', None)

latitude = f.createVariable('lat', np.float32, ('lat',))
levels = f.createVariable('lev', np.float32, ('lev',))
t = f.createVariable('time', np.float64, ('time',))

fx = f.createVariable('Fphi', np.float64, ('time','lev','lat',))
fy = f.createVariable('Fp', np.float64, ('time','lev','lat',))
fd = f.createVariable('Fdiv', np.float64, ('time','lev','lat',))
oc = f.createVariable('o_res', np.float64, ('time','lev','lat',))
vc = f.createVariable('v_res', np.float64, ('time','lev','lat',))

f.description = 'MERRA EPFD (6-hourly)'
f.history = 'Created ' + time.ctime(time.time())
f.source = 'netCDF4 python module'
t.units = units#'days since 1979-01-01 00:00:00.0'
latitude.units = 'degrees north'

levels.units = 'hPa'
fx.units = 'kg/s^2'
fy.units = 'kg/s^2'
fd.units = 'm/s^2'
oc.units = 'Pa/day'
vc.units = 'm/s'

t[:] = tim
latitude[:] = lat
levels[:] = lev_int

fx[:] = Fphi_int
fy[:] = Fp_int
fd[:] = Fdiv_int
oc[:] = O_res_int
vc[:] = V_res_int

f.close()			

print 'done'
