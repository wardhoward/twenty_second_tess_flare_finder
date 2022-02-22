# twenty_second_tess_flare_finder
[under construction] A Python-based set of tools developed for TESS GO 3174 in order to find flares in 20 second cadence and 2 min cadence TESS light curves. A tool to estimate the flare temperature from the T and Evryscope g' passbands is also included.

## Finding the initial set of flare candidates:
Initial flare candidates are found using the identify_tess_flare_candidates_step_1 code, which has a lot of helper functions at the top and the main code at the bottom. It can be run as 
> python2 identify_tess_flare_candidates_step_1.py &

Place TESS light curve fits files downloaded from MAST (either 2 min or 20 sec cadence) into a directory ./tess_lcvs. The code will automatically loop through each TIC ID identified in the directory's FITS files and search for flare candidates from that TIC ID. Candidates are output in the machine-readable file initial_tess_flare_candidates.csv. If multiple sectors of FITS files for a TIC are present, they will be temporarily merged into a single light curve prior to the flare search. Also prior to the flare search, out-of-flare variability in the TESS light curve will be de-trended with an adaptive-windowing Savitsky-Golay filter defined in the helper functions and described in Howard & MacGregor (2021), ApJ (in press). Candidates are defined as 4.5-sigma excursions above the photometric noise. Flare start, peak, and stop times are found with the function:
> low_times, flare_times, upp_times, flare_signifs = get_flare_candidates_20sec(x_vals, y_vals, avoid_mask, 4.5)

where 4.5-sigma is currently hard-coded (easy to change) and two versions of the get_flare_candidates() function are defined in the helper function section. One version is get_flare_candidates_20sec() for fast-cadence light curves and the other is get_flare_candidates_2min() for 2 min cadence. The main difference is in better outlier suppression for 20 sec cadence where it is needed. You will need to specify which version you want to use by commenting out one of the two options. The 2 min version is currently uncommented at line 748.

## Vetting
Next, candidates must be vetted by eye to remove spurious detections. I find doing this as a three step process is helpful. First, click through the full light curves with vet_out_FP_only_lcvs_tess_step_2.py to determine and then exclude from further consideration which non-flaring stars are included in the results. To save a level 2 events file, uncomment lines 554-560. Once that is done, then click through each flare candidate in each light curve with flarebyflare_tess_lv2_initial_flare_candidates_step_3.py and make a CSV or TXT list of which specific flare candidates from each star are valid (e.g. list_of_byhand_flarebyflare_FPs.csv). Once that is made, uncomment lines 425-433 and lines 615-616 to read it in and make the cleaned level 3 flare events file. You will also need a CSV list of stellar distances (pc) and T-mag values to compute quiescent luminosities. Finally, look through the TESS Target Pixel Files of what is left if uncertainty remains about particular candidates.

## Temperature
The Evryscope database can be queried at the coordinates and times of the TESS flares to find simultaneous flare events. Convert the Evryscope times from CTIO-based UT times to TESS Barycentric dates. Once both TESS band and g' band fluxes are available, use the suite of tools in XXX to estimate their color-temperatures. The function get_teff() takes the ratio of fluxes in each band at each timestamp (interpolated onto the same x-axis) and converts to the blackbody temperature that produces that ratio of fluxes. The ratios themselves are computed with the function 
> ratio_erg_arr, ratio_erg_arr2, n_flares, tot_ratio, tot_sampl_ratio, FWHM_ratio, FWHM_sampl_ratio, EVRY_Erg, TESS_Erg, Evry_Erg_FWHM = compute_ratio(i, x_tess, y_tess, y_tess_err, x_evry, y_evry, y_evry_regrid, y_evry_err_regrid, err_tess_Q_0, tess_Q_0, err_evr_Q_0, evr_Q_0, ESTART, ESTOP, FWHM_start_val, FWHM_stop_val, False)
> tot_ratio_err = np.absolute(tot_ratio)*np.sqrt((err_evr_Q_0/evr_Q_0)\**2.0 + (err_tess_Q_0/tess_Q_0)\**2.0)

Several of these inputs are not necessary for most user use-cases and the code can be adapted.
> temperature_arr = get_teff(ratio_erg_arr)
