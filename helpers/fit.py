import numpy as np
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.pymc3 import terms, GaussianProcess
from astropy.time import Time
import astropy.units as units
from astropy.units import Quantity
import platform
import multiprocessing
import plots


def gp(lc, param_lists, complicated_gp=False):

    with pm.Model() as model:

        # Gaussian process 
        if complicated_gp == True: 
            peak = plots.periodogram_lomb_scargle(lc)
            log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(lc['flux_err'])), sd=1)
            sigma_rot = pm.InverseGamma("sigma_rot", **pmx.estimate_inverse_gamma_parameters(1, 5))
            log_prot = pm.Normal("log_prot", mu=np.log(peak['period']), sd=0.02)
            prot = pm.Deterministic("prot", tt.exp(log_prot))
            log_Q0 = pm.Normal("log_Q0", mu=0, sd=2)
            log_dQ = pm.Normal("log_dQ", mu=0, sd=2)
            f = pm.Uniform("f", lower=0.01, upper=1)
            
            kernel = terms.RotationTerm(sigma=sigma_rot,period=prot,
                                        Q0=tt.exp(log_Q0),dQ=tt.exp(log_dQ),f=f)
            
            gaussian_vars = [log_jitter, sigma_rot, log_prot, log_Q0, log_dQ, f]
                
        else: 
            # A jitter term describing excess white noise
            log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(lc['flux_err'])), sd=2)
    
            # The standard deviation of the process
            # defined as sqrt(S0 * w0 * Q)
            # where S0 is related to the power when w = 0 
            # and w0 is the undamped angular frequency
            log_sigma_gp = pm.Normal("log_sigma_gp", mu=np.log(np.mean(lc['flux_err'])), sd=10)
        
            # the undamped period of the oscillator
            # 2pi / w0 
            log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10) # in days, constrain a little bit 
        
            log_Q0 = pm.Normal("log_Q0", mu=0, sd=2)
    
            # GP model for the light curve
            kernel = terms.SHOTerm(sigma=tt.exp(log_sigma_gp),
                               rho=tt.exp(log_rho_gp),
                               Q=tt.exp(log_Q0))
            
            gaussian_vars = [log_jitter, log_sigma_gp, log_rho_gp, log_Q0]
            
        
        t = np.array(lc['time'].value)
        err = np.array(lc['flux_err'].value)
        gp = GaussianProcess(kernel, 
                             t=t, 
                             diag=lc['flux_err'].value**2 + tt.exp(2*log_jitter), 
                             quiet=False)
        gp.marginal("transit_obs", observed=lc['flux'].value - np.mean(lc['flux'].value))
        pm.Deterministic("gp_pred", gp.predict(lc['flux'].value - np.mean(lc['flux'].value)))
        
        freq = np.linspace(0.01,10,5000)
        pm.Deterministic("psd", kernel.get_psd(freq))
        
        # Optimize the MAP solution
        
        start = model.test_point

        map_soln = start
        map_soln = pmx.optimize(start=map_soln, vars=gaussian_vars[:len(gaussian_vars)//2]) 
        map_soln = pmx.optimize(start=map_soln, vars=gaussian_vars[len(gaussian_vars)//2:]) 
        map_soln = pmx.optimize(start=map_soln)
    
    return map_soln


def _to_unitless_day(data):
    if isinstance(data, Quantity):
        return data.to(units.day).value
    elif not np.isscalar(data):
        return np.asarray([_to_unitless_day(item) for item in data]).flatten()
    else:
        return data


def create_transit_windows(lc, period, transit_time, duration):
        """Returns a boolean array that is ``True`` during transits and
        ``False`` elsewhere.

        This method supports multi-planet systems by allowing ``period``,
        ``transit_time``, and ``duration`` to be array-like lists of parameters.

        Parameters
        ----------
        period : `~astropy.units.Quantity`, float, or array-like
            Period(s) of the transits.
        duration : `~astropy.units.Quantity`, float, or array-like
            Duration(s) of the transits.
        transit_time : `~astropy.time.Time`, float, or array-like
            Transit midpoint(s) of the transits.

        Returns
        -------
        transit_mask : np.array of bool
            Mask that flags transits. Mask is ``True`` where there are transits.

        """
        # Convert Quantity objects to floats in units "day"
        period = _to_unitless_day(period)
        duration = _to_unitless_day(duration)

        # If ``transit_time`` is a ``Quantity```, attempt converting it to a ``Time`` object
        if isinstance(transit_time, Quantity):
            transit_time = Time(transit_time, format=lc.time.format, scale=lc.time.scale)

        # Ensure all parameters are 1D-arrays
        period = np.atleast_1d(period)
        duration = np.atleast_1d(duration)
        transit_time = np.atleast_1d(transit_time)

        # Make sure all params have the same number of entries
        n_planets = len(period)
        if any(len(param) != n_planets for param in [duration, transit_time]):
            raise ValueError(
                "period, duration, and transit_time must have "
                "the same number of values."
            )

        # Initialize an empty cadence mask
        in_transit = np.empty(len(lc), dtype=bool)
        in_transit[:] = False

        # Create the transit mask
        for per, dur, tt in zip(period, duration, transit_time):
            if isinstance(tt, Time):
                # If a `Time` is passed, ensure it has the right format & scale
                tt = Time(tt, format=lc.time.format, scale=lc.time.scale).value
            hp = per / 2.0
            in_transit |= np.abs((lc.time.value - tt + hp) % per - hp) < 5 * dur

        return in_transit


def optimise_model(lc, initial_guesses, gp_map_soln, texp=0.5 / 24, u_init=[0.3, 0.2], start=None, use_mass=False, log_density=True, include_depth=True, log_depth=True, complicated_gp=False):
    """Optimise a transit model to fit some data

    Parameters
    ----------
    lc : :class:`~lightkurve.Lightcurve`
        The lightcurve data
    initial_guesses : `dict`
        Dictionary of initial guesses
    texp : `float`, optional
        Exposure time in units of days
    u_init : `list`, optional
        Initial limb darkening guesses, by default [0.3, 0.2]

    Returns
    -------
    model
        PyMC3 model
    map_soln : `dict`
        Dictionary of optimised parameters
    """
    n_planets = len(initial_guesses["pl_orbper"])
    
    if lc.time.format == 'bkjd':
        t0s_bjd = Time(initial_guesses["pl_tranmid"], format="jd").bkjd
    elif lc.time.format == 'btjd':
        t0s_bjd = Time(initial_guesses["pl_tranmid"], format="jd").btjd
    else: 
        t0s_bjd = None
        print('error. could not identify time system used for light curve (e.g., BKJD or BTJD)')
    
    mass_star = np.array(initial_guesses['st_mass'])
    depths = np.array(initial_guesses['pl_trandep'])/100

    with pm.Model() as model:

        # The baseline flux
        mean = pm.Normal("mean", mu=1.0, sd=1.0) 

        # The time of a reference transit for each planet
        t0 = pm.Normal("t0", mu=t0s_bjd, sd=1.0, shape=n_planets)

        # The log period; also tracking the period itself
        logP = pm.Normal("logP", mu=np.log(initial_guesses["pl_orbper"]), sd=0.1, shape=n_planets)
        period = pm.Deterministic("period", tt.exp(logP))

        # The Kipping (2013) parameterization for quadratic limb darkening parameters
        # An uninformative prior for quadratic limb darkening parameters, flat distribution 
        # u1 (scalar) – The first limb darkening coefficient
        # u2 (scalar) – The second limb darkening coefficient
        u = xo.distributions.QuadLimbDark("u", testval=u_init)
        model_lightcurve = xo.LimbDarkLightCurve(u[0], u[1])
        
        # stellar density 
        initial_dens = initial_guesses["st_dens"][0] if initial_guesses["berger_dens"][0] == -1.0\
            else initial_guesses["berger_dens"][0]
        
        if log_density == True:
            log_rho_star = pm.Normal("log_rho_star",mu=np.log(initial_dens), sd=1)
            rho_star = pm.Deterministic("rho_star", tt.exp(log_rho_star))
            
            system_vars = [log_rho_star]
            
        else: 
            rho_star = pm.Normal("rho_star", mu=initial_dens, sd=1)
            xo.units.with_unit(rho_star, (units.g/(units.cm**3)))
            
            system_vars = [rho_star]
        
        # calculating the stellar radius from density and mass 
        if use_mass == True: 
            r_star = pm.Deterministic("r_star", (((mass_star/rho_star)/((4/3)*np.pi))**(1/3)) * ((units.solMass / (units.g/units.cm**3))**(1/3)).to(units.solRad))
        
        else: 
            r_star = pm.Normal("r_star", mu=initial_guesses["st_rad"][0], sd=0.1)
            
        if include_depth == True:
            
            if log_depth == True: 
                log_depth = pm.Normal("log_depth", mu=np.log(depths), sd=1.0, shape=n_planets)
                depth = pm.Deterministic("depth", tt.exp(log_depth))
                
                system_vars.append(log_depth)
                
            else: 
                depth = pm.Normal("depth", mu=depths, sd=2.0, shape=n_planets)
                
                system_vars.append(depth)
            
            b = pm.Uniform("b", lower=0, upper=1, shape=n_planets, testval=initial_guesses["pl_imppar"])
        
            ror = pm.Deterministic("ror", model_lightcurve.get_ror_from_approx_transit_depth(depth, b))
            
            system_vars.extend([b, t0, mean, logP, r_star, ror, u])
            
        else:
            # the radius ratio between the planet and star 
            ror = pm.Uniform("ror", lower=0.001, upper=0.1, shape=n_planets, testval=initial_guesses["pl_ratror"])
        
            # the impact parameter, uniformly distributed between 0 and 1+ror, where ror is the radius ratio between planet and star
            b = xo.distributions.ImpactParameter("b", ror=ror, shape=n_planets, testval=initial_guesses["pl_imppar"]) 
            
            system_vars.extend([b, t0, mean, logP, r_star, ror, u])
        
        # the radius of the orbiting body, in units of solar radii 
        r_p = pm.Deterministic('r_p', ror*r_star)
        system_vars.append(r_p)

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, r_star=r_star, rho_star=rho_star)

        # Compute the model light curve using starry
        light_curves = model_lightcurve.get_light_curve(
            orbit=orbit, r=r_p, t=lc["time"].value, texp=texp*units.day) 
        light_curve = tt.sum(light_curves, axis=-1) + mean 

        # Here we track the value of the model light curve for plotting purposes
        pm.Deterministic("light_curves", light_curves)
        
        # Gaussian process 
        if complicated_gp == True: 
            log_jitter = gp_map_soln["log_jitter"]
            sigma_rot = gp_map_soln["sigma_rot"]
            log_prot = gp_map_soln["log_prot"]
            prot = gp_map_soln["prot"]
            log_Q0 = gp_map_soln["log_Q0"]
            log_dQ = gp_map_soln["log_dQ"]
            f = gp_map_soln["f"]
            
            kernel = terms.RotationTerm(sigma=sigma_rot,period=prot,
                                        Q0=tt.exp(log_Q0),dQ=tt.exp(log_dQ),f=f)
    
        else: 
            # A jitter term describing excess white noise
            log_jitter = gp_map_soln["log_jitter"]
    
            # The standard deviation of the process
            # defined as sqrt(S0 * w0 * Q)
            # where S0 is related to the power when w = 0 
            # and w0 is the undamped angular frequency
            log_sigma_gp = gp_map_soln["log_sigma_gp"]
        
            # the undamped period of the oscillator
            # 2pi / w0 
            log_rho_gp = gp_map_soln["log_rho_gp"]
        
            log_Q0 = gp_map_soln["log_Q0"]
    
            # GP model for the light curve
            kernel = terms.SHOTerm(sigma=tt.exp(log_sigma_gp),
                               rho=tt.exp(log_rho_gp),
                               Q=tt.exp(log_Q0))
        
        t = np.array(lc['time'].value)
        err = np.array(lc['flux_err'].value)
        gp = GaussianProcess(kernel, 
                             t=t, 
                             diag=lc['flux_err'].value**2 + tt.exp(2*log_jitter), 
                             quiet=True)
        gp.marginal("transit_obs", observed=lc['flux'].value - light_curve)
        pm.Deterministic("gp_pred", gp.predict(lc['flux'].value - light_curve))
        
        # Optimize the MAP solution
        
        if start is None:
            start = model.test_point

        map_soln = start
        map_soln = pmx.optimize(start=map_soln, vars=system_vars[:len(system_vars)//2]) 
        map_soln = pmx.optimize(start=map_soln, vars=system_vars[len(system_vars)//2:]) 
        map_soln = map_soln = pmx.optimize(start=map_soln, vars=[r_p])
        map_soln = pmx.optimize(start=map_soln)
    
    _, mask = plots.plot_fittedlightcurves(lc, initial_guesses, map_soln)
    plots.folded_plots(lc, initial_guesses, map_soln)
    
    return map_soln, model, mask 

def sample_posteriors(
    model,
    map_soln,
    tune=1500,
    draws=1000,
    target_accept=0.97,
    cores=1 if platform.system() == "Darwin" else 12,
    chains=12):
    """Sample the posteriors of a given model

    Parameters
    ----------
    model
        PyMC3 Model
    map_soln : `dict`
        Dictionary of optimised parameters
    tune : `int`, optional
        How many tuning steps, by default 1500
    draws : `int`, optional
        How many draws, by default 1000
    cores : `int`, optional
        How many cores to use, by default 6
    chains : `int`, optional
        How many chains to run, by default 2
    target_accept : `float`, optional
        Acceptance rate, from 0 to 1, by default 0.97


    Returns
    -------
    trace : arviz.InferenceData
        Sampled posteriors
    """
    with model:
        trace = pmx.sample(
            tune=tune,
            draws=draws,
            start=map_soln,
            cores=cores,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            idata_kwargs=dict(log_likelihood=False),
            mp_ctx=multiprocessing.get_context("forkserver"),
        )

    return trace
