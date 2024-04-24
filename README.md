# Obtaining Higher Precision Planetary Radius Measurements 
Leveraging planetary multiplicity to get better fits for exoplanet radii, in order to better understand the radius valley and planetary compositions. 

## Motivation 
One approach to probing planetary interiors is achieving higher precision of bulk planetary parameters, particularly planetary radius measurements. Precise radius measurements for transiting planets are difficult to obtain due to degeneracies between the transit depth, impact parameter, and limb-darkening coefficients. These degeneracies can be broken by considering multi-transiting planetary systems, since they share the same stellar radius and limb-darkening coefficients! 

## About this model 
This photodynamical model is a pipeline that uses the `exoplanet`, `lightkurve`, and `PyMC3` packages to fit Kepler systems for the density of the host star, impact parameters, limb-darkening coefficients, and planet-to-star radius ratios all simultaneously. [An earlier iteration of this project](https://github.com/TomWagg/radius-valley) was first initiated by Tom Wagg. Thank you to both my advisor Professor Eric Agol (University of Washington) for his guidance and to Tom! 



