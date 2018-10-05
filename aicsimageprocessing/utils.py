import numpy as np

rads_per_deg = 0.0174533

def cart2sph(x,y,z, degrees_or_radians = 'radians'):
    ### Angles are in radians
    
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    
    if degrees_or_radians == 'degrees':
        azimuth = azimuth/rads_per_deg
        elevation = elevation/rads_per_deg
    
    return azimuth, elevation, r

def sph2cart(azimuth,elevation,r, degrees_or_radians = 'radians'):
    ### Angles are in radians
    
    if degrees_or_radians == 'degrees':
        azimuth = azimuth * rads_per_deg
        elevation = elevation * rads_per_deg
    
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z
