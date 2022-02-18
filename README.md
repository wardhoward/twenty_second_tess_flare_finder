# twenty_second_tess_flare_finder
[under construction] A Python-based set of tools developed for TESS GO 3174 in order to find flares in 20 second cadence and 2 min cadence TESS light curves. A tool to estimate the flare temperature from the T and Evryscope g' passbands is also included.

# Quick start:
Initial flare candidates are found using the identify_flare_candidates_step_1a code, which has a lot of helper functions at the top and the main code at the bottom. It can be run as 
> python2 identify_tess_flare_candidates_step_1.py &

Place TESS light curve fits files downloaded from MAST (either 2 min or 20 sec cadence) into a directory ./tess_lcvs. The code will automatically loop through each TIC identified in the dir as it looks for flare candidates and output the candidates as the machine-readable file initial_tess_flare_candidates.csv. If multiple sectors of FITS files for a TIC are present, they will be temporarily merged into a single light curve prior to the flare search. Candidates are defined as 4.5-sigma excursions above the photometric noise. Flare start, peak, and stop times are found with the function:
> low_times, flare_times, upp_times, flare_signifs = get_flare_candidates(x_vals, y_vals, avoid_mask, 4.5)

where 4.5-sigma is currently hard-coded (easy to change) and two versions of the get_flare_candidates() function are defined in the helper function section. One version is get_flare_candidates_20sec() for fast-cadence light curves and the other is get_flare_candidates() for 2 min cadence. The main difference is in better outlier suppression for 20 sec cadence where it is needed. You will need to specify which version you want to use by commenting out one of the two options. The 2 min version is currently uncommented at line XXX.

Vetting steps

Confirmation steps

Temperature steps
