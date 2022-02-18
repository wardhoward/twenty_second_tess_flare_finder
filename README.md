# twenty_second_tess_flare_finder
[under construction] A Python-based set of tools developed for TESS GO 3174 in order to find flares in 20 second cadence and 2 min cadence TESS light curves. A tool to estimate the flare temperature from the T and Evryscope g' passbands is also included.

# Quick start:
Initial flare candidates are found using the identify_tess_flare_candidates_step_1 code, which has a lot of helper functions at the top and the main code at the bottom. It can be run as 
> python2 identify_tess_flare_candidates_step_1.py &

Place TESS light curve fits files downloaded from MAST (either 2 min or 20 sec cadence) into a directory ./tess_lcvs. The code will automatically loop through each TIC ID identified in the directory's FITS files and search for flare candidates from that TIC ID. Candidates are output in the machine-readable file initial_tess_flare_candidates.csv. If multiple sectors of FITS files for a TIC are present, they will be temporarily merged into a single light curve prior to the flare search. Also prior to the flare search, out-of-flare variability in the TESS light curve will be de-trended with an adaptive-windowing Savitsky-Golay filter defined in the helper functions and described in Howard & MacGregor (2021), ApJ (in press). Candidates are defined as 4.5-sigma excursions above the photometric noise. Flare start, peak, and stop times are found with the function:
> low_times, flare_times, upp_times, flare_signifs = get_flare_candidates_20sec(x_vals, y_vals, avoid_mask, 4.5)

where 4.5-sigma is currently hard-coded (easy to change) and two versions of the get_flare_candidates() function are defined in the helper function section. One version is get_flare_candidates_20sec() for fast-cadence light curves and the other is get_flare_candidates_2min() for 2 min cadence. The main difference is in better outlier suppression for 20 sec cadence where it is needed. You will need to specify which version you want to use by commenting out one of the two options. The 2 min version is currently uncommented at line 748.

## Vetting
Next, candidates must be vetted by eye to remove spurious detections. I find doing this as a three step process is helpful. First, click through the full light curves with XXX.py to determine and then exclude from further consideration which non-flaring stars are included in the results. Once that is done, then click through each flare candidate in each light curve with XXX.py and make a CSV or TXT list of which candidates are valid. Finally, look through the TESS Target Pixel Files of what is left if uncertainty remains about particular candidates.

Vetting steps

Confirmation steps

Temperature steps
