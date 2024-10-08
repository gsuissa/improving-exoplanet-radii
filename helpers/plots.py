import numpy as np 
import arviz as az
import corner
import matplotlib.pyplot as plt 
import exoplanet as xo 
import lightkurve as lk 
import astropy.units as units
from astropy.time import Time 
from astropy.timeseries import BoxLeastSquares
from scipy.stats import norm 

# as docstring to all of these 

def diagnostic_plots(lc, lc_final, transits, transit_windows, param_lists, model, difference, std_calculated, mean_tweaked, sigma, flags_nsigma):
    
    system_name = param_lists['pl_name'][0][0:-2]
    
    # Model/data comparison plot
    p = lc.scatter()
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    plt.plot(lc.time.value,lc.flux.value,color='k',marker='o',ls='',label='data')
    plt.plot(lc.time.value,difference,marker='o',ls='',color='green',label='difference')
    plt.plot(lc.time.value, model+1,color='orange',lw=3,label='model')
    plt.xlabel(p.get_xlabel())
    plt.ylabel(p.get_ylabel())
    plt.xlim(min(lc.time.value), min(lc.time.value)+100)
    plt.ylim(mean_tweaked-(10*std_calculated), mean_tweaked+(10*std_calculated))
    plt.legend(loc='lower left',fontsize=12)
    plt.show()
    
    # Cumulative distribution function plot
    fig, ax = plt.subplots()
    diff_sorted = np.sort(1-difference)
    norm_cdf = norm.cdf((diff_sorted - (1 - mean_tweaked))/std_calculated)
    N = len(diff_sorted)
    p = np.arange(N)
    plt.plot(p, diff_sorted,label='residuals',color='deeppink',zorder=1)
    plt.plot(norm_cdf*N, diff_sorted, label='gaussian',color='k',zorder=0)
    plt.ylim(1-mean_tweaked-5*std_calculated, 1-mean_tweaked+5*std_calculated)
    plt.axhline(1-mean_tweaked-std_calculated,color='k',ls='dashed')
    plt.axhline(1-mean_tweaked+std_calculated,color='k',ls='dashed')
    plt.xlabel("Frequency")
    plt.ylabel("Flux")
    plt.legend(fontsize=12)
    plt.savefig('output/'+system_name+'/long/'+system_name+'_plots/diagnostic_plots/cdf.png', dpi=300, bbox_inches="tight")
    
    # Residuals histogram plot
    fig, ax = plt.subplots()
    if np.log10(len(lc.time)) >= 6: 
        bins_number = 20000
    elif 5 < np.log10(len(lc.time)) < 6: 
        bins_number = 10000
    else:
        bins_number = 1000
    h, bins = np.histogram(1-difference, bins=bins_number)
    plt.hist(1-difference, bins=bins, density=True, color='cornflowerblue')
    plt.plot(bins, norm(1-mean_tweaked, std_calculated).pdf(bins), linewidth=2, color='black',label='gaussian')
    plt.axvline(1-mean_tweaked-(sigma*std_calculated),ls='dashed',color='k')
    plt.axvline(1-mean_tweaked+(sigma*std_calculated),ls='dashed',color='k',label="{0}-sigma".format(sigma))
    plt.ylim(1e-6,1e4)
    plt.xlim(1-mean_tweaked-(10*std_calculated), 1-mean_tweaked+(10*std_calculated))
    plt.yscale('log')
    plt.xlabel("Residuals")
    plt.ylabel("Probability")
    plt.legend(fontsize=12,loc='upper right')
    plt.savefig('output/'+system_name+'/long/'+system_name+'_plots/diagnostic_plots/hist.png', dpi=300, bbox_inches="tight")
    
    # Highlighting outliers plot
    fig, ax = plt.subplots(figsize=(10,4))
    plt.scatter(lc.time.value, lc.flux.value,color='k',label='data',s=0.5,alpha=0.5)
    plt.plot(lc.time.value, model+1,color='deeppink',label='model')
    plt.scatter(lc.time.value[flags_nsigma], lc.flux.value[flags_nsigma],label='{0}-sigma outliers'.format(sigma),color='blue')
    plt.ylim(mean_tweaked-(10*std_calculated), mean_tweaked+(10*std_calculated))
    plt.xlim(min(lc.time.value), min(lc.time.value)+200)
    plt.xlabel("Time [BKJD]")
    plt.ylabel("Normalized flux")
    plt.legend(fontsize=10,loc='lower right')
    plt.show()
    
    # Outlier distribution plot
    fig, ax = plt.subplots()
    plt.scatter(difference[flags_nsigma],model[flags_nsigma],label='outliers',color='k')
    plt.xlabel("difference")
    plt.ylabel("model")
    plt.legend(fontsize=12)
    plt.show()
    
    # Phase folded light curves plot
    fig, ax = plt.subplots(len(param_lists['pl_letter']), figsize=(12,20))
    with plt.style.context(lk.MPLSTYLE):
        for i in range(len(param_lists['pl_name'])):
            if lc.time.format == 'bkjd':
                epoch_time = Time(param_lists["pl_tranmid"][i],format="jd").bkjd
            elif lc.time.format == 'btjd':
                epoch_time = Time(param_lists["pl_tranmid"][i],format="jd").btjd
            else: 
                epoch_time = None
                print('error. could not identify time system used for light curve (e.g., BKJD or BTJD)')
            ax[i] = lc_final.fold(period=param_lists["pl_orbper"][i],
                               epoch_time=epoch_time).scatter(ax=ax[i], label=param_lists["pl_name"][i], alpha=0.04, color='k')
            ax[i].legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('output/'+system_name+'/long/'+system_name+'_plots/diagnostic_plots/folded.png', dpi=300, bbox_inches="tight")
    
    # Transit windows plot 
    plt.figure(figsize=(10,4))
    plt.scatter(lc_final.time.value, lc_final.flux.value, label='data')
    plt.scatter(transit_windows.time.value, transit_windows.flux.value, label='transit windows')
    plt.scatter(transits.time.value, transits.flux.value, label='transits')
    plt.plot(lc.time.value, model+1, label='model', color='k')
    plt.xlim(min(lc_final.time.value), min(lc_final.time.value)+20)
    plt.legend(fontsize=10,loc='lower left')
    plt.show()
            
        
def periodogram_lomb_scargle(lc_final):
    x = lc_final['time'].value 
    y = lc_final['flux'].value 

    results = xo.estimators.lomb_scargle_estimator(x, y, max_peaks=5, min_period=0.1, max_period=100.0, samples_per_peak=50)

    peak = results["peaks"][0]
    freq, power = results["periodogram"]
    plt.plot(1 / freq, power, "k")
    plt.axvline(peak["period"], color="k", lw=4, alpha=0.3)
    plt.xlim((1 / freq).min(), (1 / freq).max())
    plt.yticks([])
    plt.xlabel("period [days]")
    _ = plt.ylabel("power")
    plt.show()
    return peak
    

def periodogram_boxleastsquares(lc_final, lower_period, upper_period):

    x = lc_final['time'].value 
    y = lc_final['flux'].value 

    m = np.zeros(len(x), dtype=bool)
    period_grid = np.exp(np.linspace(np.log(lower_period), np.log(upper_period), 5000))
    bls_results = []
    periods = []
    t0s = []
    depths = []
    
    # Compute the periodogram for each planet by iteratively masking out
    # transits from the higher signal to noise planets. Here we're assuming
    # that we know that there are exactly two planets.
    for i in range(10):
        bls = BoxLeastSquares(x[~m], y[~m])
        bls_power = bls.power(period_grid, 0.1, oversample=20)
        bls_results.append(bls_power)

        # Save the highest peak as the planet candidate
        index = np.argmax(bls_power.power)
        periods.append(bls_power.period[index])
        t0s.append(bls_power.transit_time[index])
        depths.append(bls_power.depth[index])

        # Mask the data points that are in transit for this candidate
        m |= bls.transit_mask(x, periods[-1], 0.5, t0s[-1])
        
    fig, axes = plt.subplots(len(bls_results), 2, figsize=(15, 20))

    for i in range(len(bls_results)):
         # Plot the periodogram
        ax = axes[i, 0]
        ax.axvline(np.log10(periods[i]), color="C1", lw=5, alpha=0.8)
        ax.plot(np.log10(bls_results[i].period), bls_results[i].power, "k")
        ax.annotate("period = {0:.4f} d".format(periods[i]),
            (0, 1),xycoords="axes fraction",xytext=(5, -5),textcoords="offset points",
            va="top",
            ha="left",
            fontsize=12)
        ax.set_ylabel("bls power")
        ax.set_yticks([])
        ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
        if i < len(bls_results) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("log10(period)")

        # Plot the folded transit
        ax = axes[i, 1]
        p = periods[i]
        x_fold = (x - t0s[i] + 0.5 * p) % p - 0.5 * p
        m = np.abs(x_fold) < 0.5
        ax.plot(x_fold[m], y[m], ".k")

        # Overplot the phase binned light curve
        bins = np.linspace(-0.51, 0.51, 100)
        denom, _ = np.histogram(x_fold, bins)
        num, _ = np.histogram(x_fold, bins, weights=y)
        denom[num == 0] = 1.0
        #ax.scatter(0.5 * (bins[1:] + bins[:-1]), num / denom, color="C1")

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel("relative flux [ppt]")
        if i < len(bls_results) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time since transit")

    _ = fig.subplots_adjust(hspace=0.02)
    plt.tight_layout()
    plt.show() 


def plot_fittedlightcurves(transit_windows, notransits, param_lists, map_soln):
    system_name = param_lists['pl_name'][0][0:-2]
    
    t = transit_windows["time"].value
    y = transit_windows["flux"].value
    gp_mod = map_soln["gp_pred"] + map_soln["mean"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    
    # plotting gp_model against data 
    ax = axes[0]
    ax.scatter(transit_windows["time"].value, transit_windows["flux"].value, color="k", label="data (transits removed)", s=2)
    ax.plot(t, gp_mod, color="green", label="GP model",lw=2.5)
    ax.legend(fontsize=8, loc=3, bbox_to_anchor=(1, 0.4))
    ax.set_ylabel("normalized flux")
    ax.yaxis.set_label_coords(-0.15, 0.5)

    # plotting de-trended data 
    ax = axes[1]
    ax.scatter(t, y - gp_mod, color="k", label="de-trended data", s=2)
    for i, l in enumerate(param_lists['pl_letter']):
        ax.plot(t, map_soln["light_curves"][:, i], label="planet {0}".format(l), lw=2)
    ax.legend(fontsize=8, loc=3, bbox_to_anchor=(1, 0.2))
    ax.set_ylabel("de-trended flux")
    
    # plotting residuals 
    ax = axes[2]
    mod = gp_mod + np.sum(map_soln["light_curves"], axis=-1)
    rms = np.sqrt(np.median((y-mod)**2))
    mask = np.abs(y-mod) < 7 * rms
    ax.scatter(t, y - mod, color="k", label='residuals', s=2, zorder=2)
    plt.plot(t[~mask], (y-mod)[~mask], "xr", label="outliers",zorder=1)
    ax.axhline(0, color="#aaaaaa", lw=1)
    ax.legend(fontsize=8, loc=3, bbox_to_anchor=(1, 0.4))
    ax.set_ylabel("residuals")
    
    plt.xlim(min(t), min(t)+200)
    plt.xlabel("time [BKJD]")
    plt.tight_layout()
    plt.savefig('output/'+system_name+'/long/'+system_name+'_plots/model_plots/detrended_fitted.png', dpi=300, bbox_inches="tight")

    return fig, mask 


def folded_plots(lc_final, param_lists, map_soln):
    system_name = param_lists['pl_name'][0][0:-2]
    
    detrended_data = lk.LightCurve(time=lc_final['time'], flux=lc_final['flux']-map_soln["gp_pred"])
    gp_model = lk.LightCurve(time=lc_final['time'], flux=map_soln['gp_pred']+map_soln['mean'])
    
    fig, ax = plt.subplots(len(param_lists['pl_letter']), figsize=(12,20))
    with plt.style.context(lk.MPLSTYLE):
        for n, letter in enumerate(param_lists['pl_letter']):

            model = lk.LightCurve(time=lc_final['time'], flux=map_soln["light_curves"][:,n]+map_soln['mean'])
            lc_final.fold(period=map_soln['period'][n], epoch_time=map_soln['t0'][n]).scatter(ax=ax[n], alpha=1, label='data',color='steelblue',s=20)
            detrended_data.fold(period=map_soln['period'][n], epoch_time=map_soln['t0'][n]).scatter(ax=ax[n], label='stellar variability removed', color='palevioletred',s=20)
            model.fold(period=map_soln['period'][n], epoch_time=map_soln['t0'][n]).plot(ax=ax[n],color='k',lw=2,label='model')
            ax[n].set_xlim(-0.5,0.5)
            ax[n].legend(loc='lower right',fontsize=12)
            ax[n].set_title('planet '+ str(letter),fontsize=20)
        
        plt.tight_layout()  
        plt.savefig('output/'+system_name+'/long/'+system_name+'_plots/model_plots/detrended_folded.png', dpi=300, bbox_inches="tight")
        
    
    
def plot_psd(map_soln):
    fig, ax = plt.subplots()
    plt.title("initial psd")
    plt.text(.02, .01, 'Q={0}'.format(
        np.round(np.exp(map_soln['log_Q0']),decimals=2)), ha='left', va='bottom', transform=ax.transAxes)
    freq = np.linspace(0.01,10,5000)
    plt.loglog(freq, map_soln['psd'], ":k", label="full model")
    plt.xlim(freq.min(), freq.max())
    plt.xlabel("frequency [1 / day]")
    plt.ylabel("power [day flux$^2$]")
    plt.show()


def diagnostic_plots_refined(lc_final, param_lists, map_soln):
    system_name = param_lists['pl_name'][0][0:-2]
    
    t = lc_final["time"].value
    y = lc_final["flux"].value
    gp_mod = map_soln["gp_pred"]
    model = np.sum(map_soln["light_curves"], axis=-1)
    detrended_flux = y - gp_mod
    difference = detrended_flux - model 
    
    # Model/data comparison plot
    p = lc_final.scatter()
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    plt.plot(t,y,color='k',marker='o',ls='',label='data')
    plt.plot(t,difference,marker='o',ls='',color='green',label='difference')
    for i, l in enumerate(param_lists['pl_letter']):
        plt.plot(t, map_soln["light_curves"][:, i]+map_soln['mean'], label="planet {0} model".format(l))
    plt.xlabel(p.get_xlabel())
    plt.ylabel(p.get_ylabel())
    plt.xlim(min(lc_final.time.value), min(lc_final.time.value)+150)
    plt.legend(loc='lower left', fontsize=12)
    plt.show()

    # Cumulative distribution plot
    diff_sorted = np.sort(difference)
    N = len(diff_sorted)
    p = np.arange(N)
    f = lambda x: np.interp(x, p, diff_sorted)
    one_sigma_pos = f((0.8413)*N)
    one_sigma_neg = f((1-0.8413)*N)
    std_calculated = (one_sigma_pos - one_sigma_neg)/2
    norm_cdf = norm.cdf((diff_sorted - map_soln['mean'])/std_calculated)
    
    plt.plot(p, diff_sorted,label='residuals',color='deeppink',zorder=1)
    plt.plot(norm_cdf*N, diff_sorted, label='gaussian',color='k',zorder=0)
    plt.ylim(map_soln['mean']-5*std_calculated, map_soln['mean']+5*std_calculated)
    plt.xlabel("Frequency")
    plt.ylabel("Flux")
    plt.legend(fontsize=12)
    plt.savefig('output/'+system_name+'/long/'+system_name+'_plots/diagnostic_plots_refined/cdf.png', dpi=300, bbox_inches="tight")
    
    # Residuals histogram plot
    fig, ax = plt.subplots(figsize=(7,5))
    h, bins = np.histogram(difference, bins=100)
    plt.hist(difference, bins=bins, density=True,color='cornflowerblue')
    plt.plot(bins, norm(map_soln['mean'], std_calculated).pdf(bins), linewidth=2, color='black',label='gaussian')
    plt.axvline(map_soln['mean']-(5*std_calculated),color='k', ls='dashed')
    plt.axvline(map_soln['mean']+(5*std_calculated),ls='dashed',color='k',label="5-sigma")
    plt.ylim(1e-6,1e5)
    plt.xlim(map_soln['mean']-(10*std_calculated), map_soln['mean']+(10*std_calculated))
    plt.yscale('log')
    plt.xlabel("Residuals")
    plt.ylabel("Probability")
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig('output/'+system_name+'/long/'+system_name+'_plots/diagnostic_plots_refined/hist.png', dpi=300, bbox_inches="tight")
    
def trace_plots(output_folder, system_id, cadence, trace):
    az.rcParams["plot.max_subplots"] = 200
    axes = az.plot_trace(trace, compact=False, var_names=['mean', 't0', 'period', 'rho_star', 'r_star', 'u', 'depth', 'b', 'ror','r_p'])
    fig = axes.ravel()[0].figure
    fig.tight_layout()
    plt.show()

    _ = corner.corner(trace, var_names=['mean', 't0', 'period', 'rho_star', 'r_star', 'u', 'depth', 'b', 'ror','r_p'],show_titles=True, title_fmt='0.4f')
    plt.show()

    