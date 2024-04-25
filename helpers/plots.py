import numpy as np 
import matplotlib.pyplot as plt 
import exoplanet as xo 
import lightkurve as lk 
import astropy.units as units
from astropy.time import Time 
from astropy.timeseries import BoxLeastSquares
from scipy.stats import norm 

# as docstring to all of these 

def diagnostic_plots(lc, lc_final, param_lists, model, difference, std_calculated, mean_tweaked, sigma, flags_nsigma):
    
    p = lc.scatter()
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    plt.plot(lc.time.value,lc.flux.value,color='k',marker='o',ls='',label='data')
    plt.plot(lc.time.value,difference,marker='o',ls='',color='green',label='difference')
    plt.plot(lc.time.value, model+1,color='orange',lw=3,label='model')
    plt.xlabel(p.get_xlabel())
    plt.ylabel(p.get_ylabel())
    plt.xlim(600,700)
    plt.ylim(0.998, 1.001)
    plt.legend(loc='lower left',fontsize=12)
    plt.show()
    
    fig, ax = plt.subplots()
    diff_sorted = np.sort(difference)
    norm_cdf = norm.cdf((diff_sorted - mean_tweaked)/std_calculated)
    N = len(diff_sorted)
    p = np.arange(N)
    plt.plot(p, diff_sorted,label='residuals cdf')
    plt.plot(norm_cdf*N, diff_sorted, label='gaussian cdf')
    plt.ylim(0.9995,1.0005)
    plt.axhline(mean_tweaked,color='k')
    plt.axhline(mean_tweaked-std_calculated,color='k')
    plt.axhline(mean_tweaked+std_calculated,color='k')
    plt.xlabel("frequency")
    plt.ylabel("flux")
    plt.legend(fontsize=12)
    plt.show()
    
    fig, ax = plt.subplots()
    h, bins = np.histogram(difference, bins=1000)
    plt.hist(difference, bins=bins, density=True)
    plt.plot(bins, norm(mean_tweaked, std_calculated).pdf(bins), linewidth=2, color='black',label='gaussian')
    plt.axvline(mean_tweaked-(sigma*std_calculated),color='red')
    plt.axvline(mean_tweaked+(sigma*std_calculated),color='red',label="{0} sigma".format(sigma))
    plt.ylim(1e-6,1e4)
    plt.xlim(0.999,1.001)
    plt.yscale('log')
    plt.xlabel("difference")
    plt.ylabel("probability")
    plt.legend(fontsize=12)
    plt.show()
    
    fig, ax = plt.subplots()
    plt.scatter(lc.time.value, lc.flux.value,label='difference')
    plt.scatter(lc.time.value[flags_nsigma], lc.flux.value[flags_nsigma],label='outliers')
    plt.plot(lc.time.value, model+1,color='orange',label='model')
    plt.ylim(0.997, 1.003)
    plt.xlim(600,700)
    plt.xlabel("time")
    plt.ylabel("flux")
    plt.legend(fontsize=12)
    plt.show()
    
    fig, ax = plt.subplots()
    plt.scatter(difference[flags_nsigma],model[flags_nsigma],label='outliers',color='k')
    #plt.xlim(0.999,1.0005)
    plt.xlabel("difference")
    plt.ylabel("model")
    plt.legend(fontsize=12)
    plt.show()
    
    for i in range(len(param_lists['pl_name'])):
        ax = lc_final.fold(period=param_lists["pl_orbper"][i],
                 epoch_time=Time(param_lists["pl_tranmid"][i],
                                 format="jd").bkjd).scatter(label=param_lists["pl_name"][i], alpha=0.05)
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


def plot_fittedlightcurves(lc_final, param_lists, map_soln):

    t = lc_final["time"].value
    y = lc_final["flux"].value
    gp_mod = map_soln["gp_pred"] + map_soln["mean"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    
    # plotting gp_model against data 
    ax = axes[0]
    ax.plot(t, y, "k", label="data")
    ax.plot(t, gp_mod, color="C2", label="gp model")
    ax.legend(fontsize=10)
    ax.set_ylabel("relative flux")

    # plotting de-trended data 
    ax = axes[1]
    ax.plot(t, y - gp_mod, "k", label="de-trended data")
    for i, l in enumerate(param_lists['pl_letter']):
        ax.plot(t, map_soln["light_curves"][:, i], label="planet {0}".format(l))
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("de-trended flux [ppt]")
    
    # plotting residuals 
    ax = axes[2]
    mod = gp_mod + np.sum(map_soln["light_curves"], axis=-1)
    rms = np.sqrt(np.median((y-mod)**2))
    mask = np.abs(y-mod) < 7 * rms
    ax.plot(t, y - mod, "k")
    plt.plot(t[~mask], (y-mod)[~mask], "xr", label="outliers")
    ax.axhline(0, color="#aaaaaa", lw=1)
    ax.set_ylabel("residuals [ppt]")


    plt.xlim(400,620)
    plt.xlabel("time [days]")
    plt.tight_layout()

    return fig, mask 


def folded_plots(lc_final, param_lists, map_soln):
    detrended_data = lk.LightCurve(time=lc_final['time'], flux=lc_final['flux']-map_soln["gp_pred"])

    for n, letter in enumerate(param_lists['pl_letter']):

        model = lk.LightCurve(time=lc_final['time'], flux=map_soln["light_curves"][:,n]+map_soln['mean'])
    
        fig, ax = plt.subplots(figsize=(12,4))
        lc_final.fold(period=map_soln['period'][n], epoch_time=map_soln['t0'][n]).scatter(ax=ax, alpha=1, label='raw data')
        detrended_data.fold(period=map_soln['period'][n], epoch_time=map_soln['t0'][n]).scatter(ax=ax, label='data $-$ gp model')
        model.fold(period=map_soln['period'][n], epoch_time=map_soln['t0'][n]).plot(ax=ax,color='k',lw=2,label='model')
        plt.xlim(-0.75,0.75)
        plt.legend(fontsize=12, loc='lower right')
        plt.title('planet '+ str(letter))
        plt.annotate("radius = {0:.4f} R$\oplus$".format(map_soln['r_p'][n]*units.solRad.to(units.earthRad)),
                     (0, 0),xycoords="axes fraction",xytext=(7,20),textcoords="offset points",va="top",ha="left",fontsize=12)
    
    
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
    t = lc_final["time"].value
    y = lc_final["flux"].value
    gp_mod = map_soln["gp_pred"]
    model = np.sum(map_soln["light_curves"], axis=-1)
    detrended_flux = y - gp_mod
    difference = detrended_flux - model 
    
    p = lc_final.scatter()
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    plt.plot(t,y,color='k',marker='o',ls='',label='data')
    plt.plot(t,difference,marker='o',ls='',color='green',label='difference')
    for i, l in enumerate(param_lists['pl_letter']):
        plt.plot(t, map_soln["light_curves"][:, i]+map_soln['mean'], label="planet {0} model".format(l))
    plt.xlabel(p.get_xlabel())
    plt.ylabel(p.get_ylabel())
    plt.xlim(550,700)
    #plt.ylim(0.998, 1.001)
    plt.legend(loc='lower left', fontsize=12)
    plt.show()

    diff_sorted = np.sort(difference)
    N = len(diff_sorted)
    p = np.arange(N)
    f = lambda x: np.interp(x, p, diff_sorted)
    one_sigma_pos = f((0.8413)*N)
    one_sigma_neg = f((1-0.8413)*N)
    std_calculated = (one_sigma_pos - one_sigma_neg)/2
    
    norm_cdf = norm.cdf((diff_sorted - map_soln['mean'])/std_calculated)

    plt.plot(p, diff_sorted,label='residuals cdf')
    plt.plot(norm_cdf*N, diff_sorted, label='gaussian cdf')
    #plt.ylim(0.9995,1.0005)
    plt.xlabel("frequency")
    plt.ylabel("flux")
    plt.legend(fontsize=12)
    plt.show()
    
    sigma=5
    
    fig, ax = plt.subplots(figsize=(7,5))
    h, bins = np.histogram(difference, bins=100)
    plt.hist(difference, bins=bins, density=True)
    plt.plot(bins, norm(map_soln['mean'], std_calculated).pdf(bins), linewidth=2, color='black',label='gaussian')
    plt.axvline(map_soln['mean']-(sigma*std_calculated),color='red')
    plt.axvline(map_soln['mean']+(sigma*std_calculated),color='red',label="{0} sigma".format(sigma))
    plt.ylim(1e-6,1e5)
    #plt.xlim(0.999,1.001)
    plt.yscale('log')
    plt.xlabel("difference")
    plt.ylabel("probability")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    

    