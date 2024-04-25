import numpy as np
from astropy.time import Time
import exoplanet as xo 
import lightkurve as lk


# add docstring to these!! 

def model_data_residuals(param_lists, lc):
    constant = 1

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
    u = [0.3, 0.2]

    light_curves = {}
    for i in range(len(param_lists['pl_letter'])):
        light_curves[param_lists['pl_letter'][i]] = (xo.LimbDarkLightCurve(*u).get_light_curve(
                                                    orbit=orbits[param_lists['pl_letter'][i]],
                                                    r=param_lists['st_rad'][i]*param_lists['pl_ratror'][i],
                                                    t=t).eval())
        
    dataset = np.array(lc.flux)
    model = sum(light_curves.values())
    model = model.flatten()
    difference = (dataset - model)
    
    diff_sorted = np.sort(difference)
    N = len(diff_sorted)
    p = np.arange(N)
    f = lambda x: np.interp(x, p, diff_sorted)
    one_sigma_pos = f((0.8413)*N)
    one_sigma_neg = f((1-0.8413)*N)
    std_calculated = (one_sigma_pos - one_sigma_neg)/2
    
    mean_tweaked = np.mean(difference[((np.median(difference) - std_calculated) < difference) 
                                      & (difference < (np.median(difference) + std_calculated))])
    
    return model, difference, std_calculated, mean_tweaked



def remove_outliers(sigma, lc, difference, std_calculated, mean_tweaked):
        
    flags_sigma = ((mean_tweaked-(sigma*std_calculated)) > difference) | ((mean_tweaked+(sigma*std_calculated)) < difference)
        
    lc_clean = lc[~flags_sigma]
        
    lc_final = np.ma.filled(lc_clean, fill_value=np.nan).remove_nans()
        
    return lc_final, flags_sigma
    
    