import numpy as np
from astropy.time import Time
import exoplanet as xo 
import lightkurve as lk


def model_data_residuals(param_lists, lc):
    """
    Does an initial quality check of the Kepler data to see if the noise is Gaussian. Uses a simplified model of the system 
    and NASA exoplanet archive values 
    
    Parameters
    ----------
    param_lists : `dict` 
        Dictionary of list of parameters for each planet
    lc : :class:`~lightkurve.Lightcurve` 
        Stitched light curve of planetary system
    
    
    Returns 
    -------
    model : ``
        Modeled light curve of planetary system using simplified orbits and NASA archive values 
    difference : ``
        Residuals between the model and Kepler data 
    std_calculated : ``
        Standard deviation of the residuals 
    mean_tweaked : ``
        Mean of the residuals 
    """
    
    lc.sort(keys='time')
    
    # model orbit for each of the planets using initial guesses from NASA exoplanet archive 
    orbits = {}
    for i in range(len(param_lists['pl_letter'])):
        orbits[param_lists['pl_letter'][i]] = xo.orbits.SimpleTransitOrbit(period=param_lists["pl_orbper"][i], 
                                                                           t0=(Time(param_lists["pl_tranmid"] [i],
                                                                                    format="jd").bkjd),                           
                                                                           b=param_lists['pl_imppar'][i], 
                                                                           duration=param_lists['pl_trandur'][i]/24,
                                                                           r_star=param_lists['st_rad'][i],
                                                                           ror=param_lists['pl_ratror'][i])
    t = lc.time.value
    u = [0.3, 0.2] # initial limb darkening guesses 
    
    # model each light curve using orbit 
    light_curves = {}
    for i in range(len(param_lists['pl_letter'])):
        light_curves[param_lists['pl_letter'][i]] = (xo.LimbDarkLightCurve(*u).get_light_curve(
                                                    orbit=orbits[param_lists['pl_letter'][i]],
                                                    r=param_lists['st_rad'][i]*param_lists['pl_ratror'][i],
                                                    t=t).eval())
        
    # summing up the light curves to get one model light curve     
    model = sum(light_curves.values())
    model = model.flatten()
    
    # getting flux from Kepler dataset 
    dataset = np.array(lc.flux)
    
    # finding difference between Kepler data and model 
    # if these residuals fit a Gaussian, that implies the noise of the Kepler dataset is Gaussian 
    difference = (dataset - model)
   
    # calculate the standard deviation of the difference 
    diff_sorted = np.sort(difference)
    N = len(diff_sorted)
    p = np.arange(N)
    f = lambda x: np.interp(x, p, diff_sorted)
    one_sigma_pos = f((0.8413)*N)
    one_sigma_neg = f((1-0.8413)*N)
    std_calculated = (one_sigma_pos - one_sigma_neg)/2
    
    # find the mean of the distribution of the difference by isolating +/- 1 sigma of the data 
    mean_tweaked = np.mean(difference[((np.median(difference) - std_calculated) < difference) 
                                      & (difference < (np.median(difference) + std_calculated))])
    
    return model, difference, std_calculated, mean_tweaked



def remove_outliers(sigma, lc, difference, std_calculated, mean_tweaked):
        
    flags_sigma = ((mean_tweaked-(sigma*std_calculated)) > difference) | ((mean_tweaked+(sigma*std_calculated)) < difference)
        
    lc_clean = lc[~flags_sigma]
        
    lc_final = np.ma.filled(lc_clean, fill_value=np.nan).remove_nans()
        
    lc_final.sort(keys='time')
    
    return lc_final, flags_sigma
    
    