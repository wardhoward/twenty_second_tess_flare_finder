from astropy.stats import LombScargle
from astropy.io import fits
from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
import scipy.signal
from scipy.interpolate import interp1d
import glob
import copy
import os
from astropy.time import Time
from scipy.optimize import curve_fit
print ("\n")

def SG_lcv_smoother(x_in, y_in, est_period):
    #print(est_period,"d")

    if est_period<1.0:
        if len(y_in)<101:
            filt_sap_norm_flux = np.median(y_in)*np.ones(len(y_in))
        else:
            filt_sap_norm_flux = scipy.signal.savgol_filter(y_in, 101, 2)
    elif (est_period>=1.0) and (est_period<2.0):
        if len(y_in)<151:
            filt_sap_norm_flux = np.median(y_in)*np.ones(len(y_in))
        else:
            filt_sap_norm_flux = scipy.signal.savgol_filter(y_in, 151, 2)
    elif (est_period>=2.0) and (est_period<4.0):
        if len(y_in)<201:
            filt_sap_norm_flux = np.median(y_in)*np.ones(len(y_in))
        else:
            filt_sap_norm_flux = scipy.signal.savgol_filter(y_in, 201, 2)
    else:
        if len(y_in)<401:
            filt_sap_norm_flux = np.median(y_in)*np.ones(len(y_in))
        else:
            filt_sap_norm_flux = scipy.signal.savgol_filter(y_in, 401, 1)
        
    return (x_in, y_in, filt_sap_norm_flux)

def obtain_lc_breakpoints(in_tbjd):

    tbjd = copy.deepcopy(in_tbjd)
    index = np.argsort(tbjd)
    tbjd = tbjd[index]
    
    # break points?
    tbjd_hr = copy.deepcopy(tbjd)*24.0
    tbjd_hr -= np.nanmin(tbjd_hr)
    diff_in_hr = np.absolute(np.roll(tbjd_hr,1)-tbjd_hr)

    diff_in_hr[0] = 0.0
    diff_in_hr[-1] = 0.0

    #print np.max(diff_in_hr), np.min(diff_in_hr), np.mean(diff_in_hr)

    list_breakpoints = diff_in_hr[diff_in_hr>12.0]
    #print np.max(diff_in_hr[(tbjd>1489.67)&(tbjd<1491.88)])
    #exit()
    tbjd_breaks=[]
    for j in range(len(list_breakpoints)):
        BREAKPOINT = list_breakpoints[j]
        #print tbjd[diff_in_hr==BREAKPOINT][0]
        tbjd_breaks.append(tbjd[diff_in_hr==BREAKPOINT][0])
    tbjd_breaks=np.array(tbjd_breaks)

    break_index = np.argsort(tbjd_breaks)
    tbjd_breaks = tbjd_breaks[break_index]
    
    return tbjd_breaks

def systematics_removal_by_breakpoint(in_tbjd, in_flux, in_avoid_mask, tbjd_breaks, PROT):

    deflared_tbjd = copy.deepcopy(in_tbjd[in_avoid_mask==0])
    deflared_flux = copy.deepcopy(in_flux[in_avoid_mask==0])

    LAST_INDEX = np.arange(len(tbjd_breaks))[-1]
    for j in range(len(tbjd_breaks)+1):

        if j!=0:
            PREV_BRK = tbjd_breaks[j-1]
        else:
            PREV_BRK=np.min(in_tbjd)
            
        if j==0:
            BRK = tbjd_breaks[j]
            #print BRK
            deflared_tbjd_pt, deflared_flux_pt, SG_deflared_flux_pt = SG_lcv_smoother(deflared_tbjd[deflared_tbjd<BRK], deflared_flux[deflared_tbjd<BRK], PROT)
        elif (j!=0) and (j<=LAST_INDEX):
            BRK = tbjd_breaks[j]

            orig_inX = copy.deepcopy(in_tbjd[(in_tbjd>PREV_BRK)&(in_tbjd<BRK)])
            orig_inY = copy.deepcopy(in_flux[(in_tbjd>PREV_BRK)&(in_tbjd<BRK)])
            
            inX = copy.deepcopy(deflared_tbjd[(deflared_tbjd>PREV_BRK)&(deflared_tbjd<BRK)])
            inY = copy.deepcopy(deflared_flux[(deflared_tbjd>PREV_BRK)&(deflared_tbjd<BRK)])
            inX = np.concatenate((inX, orig_inX[:10], orig_inX[-10:]))
            inY = np.concatenate((inY, orig_inY[:10], orig_inY[-10:]))
            inX_ind = np.argsort(inX)
            inX =inX[inX_ind]
            inY =inY[inX_ind]
            
            deflared_tbjd_pt, deflared_flux_pt, SG_deflared_flux_pt = SG_lcv_smoother(inX, inY, PROT)
            #deflared_tbjd_pt, deflared_flux_pt, SG_deflared_flux_pt = SG_lcv_smoother(deflared_tbjd[(deflared_tbjd>PREV_BRK)&(deflared_tbjd<BRK)], deflared_flux[(deflared_tbjd>PREV_BRK)&(deflared_tbjd<BRK)], PROT)
        else:
            BRK = tbjd_breaks[-1]
            deflared_tbjd_pt, deflared_flux_pt, SG_deflared_flux_pt = SG_lcv_smoother(deflared_tbjd[deflared_tbjd>BRK], deflared_flux[deflared_tbjd>BRK], PROT)

        #plt.scatter(deflared_tbjd_pt, deflared_flux_pt,s=2,color="black")
        #plt.scatter(deflared_tbjd_pt, SG_deflared_flux_pt, s=2, color="orange")
        #plt.show()
        #exit()
            
        if j==0:
            all_sections_deflared_tbjd = np.array(copy.deepcopy(deflared_tbjd_pt))
            all_sections_deflared_flux = np.array(copy.deepcopy(deflared_flux_pt))
            all_sections_SG_deflared_flux = np.array(copy.deepcopy(SG_deflared_flux_pt))
        else:
            all_sections_deflared_tbjd = np.concatenate((all_sections_deflared_tbjd, deflared_tbjd_pt))
            all_sections_deflared_flux = np.concatenate((all_sections_deflared_flux, deflared_flux_pt))
            all_sections_SG_deflared_flux = np.concatenate((all_sections_SG_deflared_flux, SG_deflared_flux_pt))

    all_index = np.argsort(all_sections_deflared_tbjd)
    all_sections_deflared_tbjd = all_sections_deflared_tbjd[all_index]
    all_sections_SG_deflared_flux = all_sections_SG_deflared_flux[all_index]

    interp_SG = interp1d(all_sections_deflared_tbjd, all_sections_SG_deflared_flux, kind="linear", bounds_error=False, fill_value=1.0)
    SG_model_flux = interp_SG(in_tbjd)

    #plt.plot(all_sections_deflared_tbjd, all_sections_SG_deflared_flux,marker="+",ls="none", color="black")
    #plt.plot(in_tbjd, SG_model_flux, marker="+", ls="none", color="orange")
    #plt.show()
    #exit()
    
    return (in_tbjd, SG_model_flux)

def get_tess_lcv(fits_table_filename):

    hdulist = fits.open(fits_table_filename)  # open a FITS file

    data = hdulist[1].data  # assume the first extension is a table

    # get star time series 
    tess_bjd = hdulist[1].data['TIME']
    sap_flux = hdulist[1].data['SAP_FLUX']
    sap_flux_err = hdulist[1].data['SAP_FLUX_ERR']
    
    sap_norm_flux=sap_flux/np.nanmedian(sap_flux)
    sap_norm_flux_err=sap_flux_err/np.nanmedian(sap_flux)

    pdcsap_flux = hdulist[1].data['PDCSAP_FLUX']
    flags = hdulist[1].data['QUALITY']

    # convert to numpy arrays:
    tess_bjd=np.array(tess_bjd)
    sap_norm_flux=np.array(sap_norm_flux)
    sap_norm_flux_err = np.array(sap_norm_flux_err)
    flags=np.array(flags)

    temp = np.vstack((tess_bjd,sap_norm_flux,sap_norm_flux_err,flags)).T
    tess_bjd = np.array([x[0] for x in temp if np.isnan(x[1])==False])
    sap_norm_flux = np.array([x[1] for x in temp if np.isnan(x[1])==False])
    sap_norm_flux_err = np.array([x[2] for x in temp if np.isnan(x[1])==False])
    flags = np.array([x[3] for x in temp if np.isnan(x[1])==False])
    
    #optional smoothing with SG filter:
    #sg_whitened_flux, sg_filter = SG_lcv_smoother(tess_bjd,sap_norm_flux,fits_table_filename)
    
    #return (tess_bjd, sap_norm_flux, flags)
    return (tess_bjd, sap_norm_flux, sap_norm_flux_err, flags)

def build_tess_lightcurve(TID,PATH):
    #print glob.glob(PATH+"*"+str(TID)+"*lc.fits")
    #exit()
    tess_lcvs = glob.glob(PATH+"*lc.fits") #load every TESS lcv, select each flare star from this list
    tic_id_arr=[]
    for i in range(len(tess_lcvs)):
        #print int(tess_lcvs[i].split("-")[2])
        tic_id_arr.append(int(tess_lcvs[i].split("-")[2]))
    tic_id_arr=np.array(tic_id_arr)
    indices_tic_id = np.arange(len(tic_id_arr))
    
    targ_tic_ids = copy.deepcopy(tic_id_arr)[tic_id_arr==TID]
    targ_inds = indices_tic_id[tic_id_arr==TID]
    #print TID, targ_inds
    #exit()
    #print ""

    sector_list=[]
    
    count=0
    for j in targ_inds:
        #print j,TID,tic_id_arr[j]
        #print int(tess_lcvs[j].split("-")[-4].replace("s",""))
        sector_list.append(int(tess_lcvs[j].split("-")[-4].replace("s","")))
        tess_bjd_part, sap_flux_part, sap_flux_err_part, flags_part = get_tess_lcv(tess_lcvs[j])
        if count==0:
            tess_bjd=tess_bjd_part
            sap_flux=sap_flux_part
            sap_flux_err=sap_flux_err_part
            flags=flags_part
        else:
            tess_bjd=np.concatenate((tess_bjd,tess_bjd_part))
            sap_flux=np.concatenate((sap_flux,sap_flux_part))
            sap_flux_err=np.concatenate((sap_flux_err,sap_flux_err_part))
            flags=np.concatenate((flags,flags_part))
        count+=1

    tess_bjd = copy.deepcopy(tess_bjd)
    sap_flux = copy.deepcopy(sap_flux)
    sap_flux_err = copy.deepcopy(sap_flux_err)
    flags = copy.deepcopy(flags)

    if len(flags) != len(tess_bjd):
        print "FlagError. Exiting now"
        exit()
        
    first_sector = np.min(sector_list)
    last_sector = np.max(sector_list)

    return (tess_bjd,sap_flux,sap_flux_err,flags)

def get_flare_candidates_2min(x_vals, y_vals, avoid_mask, signif):

    #WARNING: do not use on 20 sec TESS data! Use get_flare_candidates20sec() instead.
    
    x_val_index_array = np.arange(len(x_vals))
    
    # reinforce time-sorted lc x vals
    timesort_inds = np.argsort(x_vals)
    x_vals = x_vals[timesort_inds]
    y_vals = y_vals[timesort_inds]
    avoid_mask = avoid_mask[timesort_inds]
    
    one_sigma = np.std(y_vals[avoid_mask==0])
    sigma_y_vals = (y_vals - np.median(y_vals[avoid_mask==0])) / one_sigma

    candidate_list = copy.deepcopy(x_vals[sigma_y_vals>=4.5])

    plausible_candidates=[]
    plausible_sigma_vals=[]
    plausible_ampl_vals=[]
    for cand_index in range(len(candidate_list)):
        CAND_TIME = candidate_list[cand_index]
        XVAL_INDEX = list(x_vals).index(CAND_TIME)
        SIGMA_VAL = sigma_y_vals[XVAL_INDEX] #assoc. significance of cand
        AMPL_VAL = y_vals[XVAL_INDEX] #assoc. peak flux of cand
        
        candidate_sigma = copy.deepcopy(sigma_y_vals[(XVAL_INDEX-3):(XVAL_INDEX+4)])
        if len(candidate_sigma[candidate_sigma>=2.5]) < 3:
            continue
        else:
            plausible_candidates.append(CAND_TIME)
            plausible_sigma_vals.append(SIGMA_VAL)
            plausible_ampl_vals.append(AMPL_VAL)
    # convert to np array for future slicing and indexing purposes
    plausible_candidates = np.array(plausible_candidates)
    plausible_sigma_vals = np.array(plausible_sigma_vals)
    plausible_ampl_vals = np.array(plausible_ampl_vals)
    plausible_index = np.arange(len(plausible_candidates))

    
    low_times=[]
    upp_times=[]
    for plaus_cand_index in range(len(plausible_candidates)):
        CAND_TIME = plausible_candidates[plaus_cand_index]

        times_before_cand = x_vals[(x_vals<CAND_TIME)&(x_vals>(-0.05+CAND_TIME))]
        times_after_cand = x_vals[(x_vals>CAND_TIME)&(x_vals<(0.17+CAND_TIME))]

        signifs_before_cand = sigma_y_vals[(x_vals<CAND_TIME)&(x_vals>(-0.05+CAND_TIME))]
        signifs_after_cand = sigma_y_vals[(x_vals>CAND_TIME)&(x_vals<(0.17+CAND_TIME))]

        indices_before = np.arange(len(times_before_cand))
        indices_after = np.arange(len(times_after_cand))
        
        # find start time of flare (sigma<1):
        if len(times_before_cand)==0:
            LOW_TIME = CAND_TIME - (2.0/(60.0*24.0))
        elif len(times_before_cand[signifs_before_cand<0.99])<3:
            LOW_TIME = np.max(times_before_cand) - (2.0/(60.0*24.0))
        else:
            LOW_TIME = np.max(times_before_cand[signifs_before_cand<0.99])

        # find stop time of flare (sigma<1):
        if len(times_after_cand)==0:
            UPP_TIME = CAND_TIME+ (10.0/(60.0*24.0))
        elif len(times_after_cand[signifs_after_cand<0.99])<3:
            UPP_TIME = np.min(times_after_cand) + (10.0/(60.0*24.0))
        else:
            UPP_TIME = 0.0104+np.min(times_after_cand[signifs_after_cand<0.99])
        
        low_times.append(LOW_TIME)
        upp_times.append(UPP_TIME)
    low_times=np.array(low_times)
    upp_times=np.array(upp_times)

    # deduplicate using error bars, keeping highest sigma one
    rejected_indices=np.array([999999])
    for plaus_cand_index in range(len(plausible_candidates)):

        if plaus_cand_index in rejected_indices:
            continue
        
        CAND_TIME = plausible_candidates[plaus_cand_index]
        CAND_SIGMA = plausible_sigma_vals[plaus_cand_index]
        CAND_LOW = low_times[plaus_cand_index]
        CAND_UPP = upp_times[plaus_cand_index]
        

        NUM_CANDS = len(plausible_candidates[(plausible_candidates>=CAND_LOW)&(plausible_candidates<=CAND_UPP)])
        
        if NUM_CANDS > 1:
            sub_plausible_index = plausible_index[(plausible_candidates>=CAND_LOW)&(plausible_candidates<=CAND_UPP)]
            
            best_sub_plausible_index = sub_plausible_index[list(plausible_sigma_vals[sub_plausible_index]).index(np.max(plausible_sigma_vals[sub_plausible_index]))]
            rejected_indices_pt = sub_plausible_index[sub_plausible_index!=best_sub_plausible_index] # concat these to eliminated invalid values
            rejected_indices = np.concatenate((rejected_indices,rejected_indices_pt))
            
    dedup_indices = np.array([x for x in plausible_index if not x in rejected_indices])

    if len(plausible_index)==0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    #print "\nplaus",plausible_index
    #print "\nreject",rejected_indices
    #print "\ndedup",dedup_indices
    #exit()
    
    low_times = low_times[dedup_indices]
    plausible_candidates = plausible_candidates[dedup_indices]
    upp_times = upp_times[dedup_indices]
    plausible_sigma_vals = plausible_sigma_vals[dedup_indices]
    
    """
    # diagnostic plots of detections
    for u in range(len(low_times)):
        plt.axvspan(low_times[u], upp_times[u], facecolor="grey")
    plt.axhline(2.5,color="blue")
    plt.plot(x_vals, sigma_y_vals, marker="+", ls="none", color="black")
    plt.plot(plausible_candidates, plausible_sigma_vals, marker="o", ls="none", color="red")
    plt.ylabel("$\sigma_\mathrm{flux}$",fontsize=12)
    plt.show()
    exit()
    """
    
    return (low_times, plausible_candidates, upp_times, plausible_sigma_vals)

def get_flare_candidates_20sec(x_vals, y_vals, avoid_mask, signif):

    x_val_index_array = np.arange(len(x_vals))
    
    # reinforce time-sorted lc x vals
    timesort_inds = np.argsort(x_vals)
    x_vals = x_vals[timesort_inds]
    y_vals = y_vals[timesort_inds]
    avoid_mask = avoid_mask[timesort_inds]
    
    one_sigma = np.std(y_vals[avoid_mask==0])
    sigma_y_vals = (y_vals - np.median(y_vals[avoid_mask==0])) / one_sigma

    candidate_list = copy.deepcopy(x_vals[sigma_y_vals>=4.5])

    plausible_candidates=[]
    plausible_sigma_vals=[]
    plausible_ampl_vals=[]
    for cand_index in range(len(candidate_list)):
        CAND_TIME = candidate_list[cand_index]
        XVAL_INDEX = list(x_vals).index(CAND_TIME)
        SIGMA_VAL = sigma_y_vals[XVAL_INDEX] #assoc. significance of cand
        AMPL_VAL = y_vals[XVAL_INDEX] #assoc. peak flux of cand
        
        candidate_sigma = copy.deepcopy(sigma_y_vals[(XVAL_INDEX-3):(XVAL_INDEX+4)])
        if len(candidate_sigma[candidate_sigma>=3.0]) < 3:
            continue
        else:
            plausible_candidates.append(CAND_TIME)
            plausible_sigma_vals.append(SIGMA_VAL)
            plausible_ampl_vals.append(AMPL_VAL)
    # convert to np array for future slicing and indexing purposes
    plausible_candidates = np.array(plausible_candidates)
    plausible_sigma_vals = np.array(plausible_sigma_vals)
    plausible_ampl_vals = np.array(plausible_ampl_vals)
    plausible_index = np.arange(len(plausible_candidates))

    
    low_times=[]
    upp_times=[]
    for plaus_cand_index in range(len(plausible_candidates)):
        CAND_TIME = plausible_candidates[plaus_cand_index]

        times_before_cand = x_vals[(x_vals<CAND_TIME)&(x_vals>(-0.05+CAND_TIME))]
        times_after_cand = x_vals[(x_vals>CAND_TIME)&(x_vals<(0.17+CAND_TIME))]

        signifs_before_cand = sigma_y_vals[(x_vals<CAND_TIME)&(x_vals>(-0.05+CAND_TIME))]
        signifs_after_cand = sigma_y_vals[(x_vals>CAND_TIME)&(x_vals<(0.17+CAND_TIME))]

        indices_before = np.arange(len(times_before_cand))
        indices_after = np.arange(len(times_after_cand))
        
        rolling_avs_before = np.array([np.nanmean(signifs_before_cand[(x-3):(x+4)]) for x in indices_before[3:]]) # currently a 2 min rolling average
        rolling_avs_before= np.concatenate((np.zeros(3), rolling_avs_before))
        
        rolling_avs_after = np.array([np.nanmean(signifs_after_cand[(x-3):(x+4)]) for x in indices_after[3:]]) # currently a 2 min rolling average
        rolling_avs_after= np.concatenate((3.1*np.ones(3), rolling_avs_after))

        if len(times_before_cand)!=len(rolling_avs_before):
            LOW_TIME = CAND_TIME - (2.0/(60.0*24.0))
            #print times_before_cand
            #print "rolling ave",np.array([np.nanmean(signifs_before_cand[(x-3):(x+4)]) for x in indices_before[3:]])
            #exit()
            
        elif len(times_before_cand[rolling_avs_before<0.1])<3:
            #print "\nbefore",times_before_cand[rolling_avs_before<0.1]
            #print rolling_avs_before
            #print "sigma",signifs_before_cand
            #print "aves",np.array([signifs_before_cand[(x-3):(x+4)] for x in indices_before])
            LOW_TIME = np.max(times_before_cand) - (2.0/(60.0*24.0))
        else:
            LOW_TIME = times_before_cand[rolling_avs_before<0.1][-2]
            
        if len(times_after_cand)!=len(rolling_avs_after):
            UPP_TIME = CAND_TIME+ (20.0/(60.0*24.0))
        elif len(times_after_cand[rolling_avs_after<0.1])<3:
            UPP_TIME = np.min(times_after_cand) + (20.0/(60.0*24.0))
        else:
            UPP_TIME = 0.0104+times_after_cand[rolling_avs_after<0.1][2]
        
        low_times.append(LOW_TIME)
        upp_times.append(UPP_TIME)
    low_times=np.array(low_times)
    upp_times=np.array(upp_times)

    # deduplicate using error bars, keeping highest sigma one
    rejected_indices=np.array([999999])
    for plaus_cand_index in range(len(plausible_candidates)):

        if plaus_cand_index in rejected_indices:
            continue
        
        CAND_TIME = plausible_candidates[plaus_cand_index]
        CAND_SIGMA = plausible_sigma_vals[plaus_cand_index]
        CAND_LOW = low_times[plaus_cand_index]
        CAND_UPP = upp_times[plaus_cand_index]
        

        NUM_CANDS = len(plausible_candidates[(plausible_candidates>=CAND_LOW)&(plausible_candidates<=CAND_UPP)])
        
        if NUM_CANDS > 1:
            sub_plausible_index = plausible_index[(plausible_candidates>=CAND_LOW)&(plausible_candidates<=CAND_UPP)]
            
            best_sub_plausible_index = sub_plausible_index[list(plausible_sigma_vals[sub_plausible_index]).index(np.max(plausible_sigma_vals[sub_plausible_index]))]
            rejected_indices_pt = sub_plausible_index[sub_plausible_index!=best_sub_plausible_index] # concat these to eliminated invalid values
            rejected_indices = np.concatenate((rejected_indices,rejected_indices_pt))
            
    dedup_indices = np.array([x for x in plausible_index if not x in rejected_indices])

    if len(plausible_index)==0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    #print "\nplaus",plausible_index
    #print "\nreject",rejected_indices
    #print "\ndedup",dedup_indices
    #exit()
    
    low_times = low_times[dedup_indices]
    plausible_candidates = plausible_candidates[dedup_indices]
    upp_times = upp_times[dedup_indices]
    plausible_sigma_vals = plausible_sigma_vals[dedup_indices]
    
    """
    # diagnostic plots of detections
    for u in range(len(low_times)):
        plt.axvspan(low_times[u], upp_times[u], facecolor="grey")
    plt.axhline(2.5,color="blue")
    plt.plot(x_vals, sigma_y_vals, marker="+", ls="none", color="black")
    plt.plot(plausible_candidates, plausible_sigma_vals, marker="o", ls="none", color="red")
    plt.ylabel("$\sigma_\mathrm{flux}$",fontsize=12)
    plt.show()
    exit()
    """
    
    return (low_times, plausible_candidates, upp_times, plausible_sigma_vals)

def cut_outliers_before_smooth(x_vals, sap_flux):
    
    # loop the following lines by breakpoint (test on i-28):
    LL = np.median(sap_flux) - 3*np.std(sap_flux) #account for transits
    cut_index = np.arange(len(sap_flux)).astype(int)
    xval_breaks = obtain_lc_breakpoints(x_vals) #yes, declared 2X for now
    xval_breaks = np.concatenate((np.array([np.min(x_vals)]), xval_breaks)) #hacky, yeah...
    #print i_prime, xval_breaks
    #exit()

    
    yes_in_cut_index=[]
    no_in_cut_index=[]
    for IXV in range(len(xval_breaks)):
        if xval_breaks[-1]==xval_breaks[IXV]: #if current val == last in arr
            XV_SECT = x_vals[(x_vals>=xval_breaks[IXV])]
        else:
            XV_SECT = x_vals[(x_vals>=xval_breaks[IXV])&(x_vals<xval_breaks[IXV+1])]
        XVmin = np.min(XV_SECT)
        XVmax = np.max(XV_SECT)
        
        sap_flux_cut, lower_clip, upper_clip = sigmaclip(sap_flux[(x_vals>=XVmin)&(x_vals<XVmax)], 2.5, 2.5)

        temp_yes_cutind = copy.deepcopy(cut_index[(sap_flux>=lower_clip)&(sap_flux<=upper_clip)&(x_vals>=XVmin)&(x_vals<XVmax)])
        temp_no_cutind = copy.deepcopy(cut_index[((sap_flux<lower_clip)|(sap_flux>upper_clip))&(x_vals>=XVmin)&(x_vals<XVmax)])
        if len(yes_in_cut_index)==0:
            yes_in_cut_index = copy.deepcopy(temp_yes_cutind)
            no_in_cut_index = copy.deepcopy(temp_no_cutind)
        else:
            yes_in_cut_index = np.concatenate((yes_in_cut_index, temp_yes_cutind))
            no_in_cut_index = np.concatenate((no_in_cut_index, temp_no_cutind))
            
    avoid_mask = np.zeros(len(x_vals))
    avoided_times = copy.deepcopy(x_vals[no_in_cut_index])
    #exit()
    for n in range(len(x_vals)):
        #print n,len(x_vals)
        X_VAL = copy.deepcopy(x_vals[n])
        diff_array = 24.0*60.0*np.absolute(X_VAL - avoided_times)
        if len(diff_array)>0:
            if (np.min(diff_array) < 20.0): #mins
                avoid_mask[n] = 1

    return avoid_mask

def fortify_lcv(avoid_mask, xval_breaks):

    #fortify first and last epochs in section
    avoid_mask[:10]=0
    avoid_mask[-10:]=0
    for XV in np.sort(xval_breaks):
        #plt.axvline(XV, color="red")
        XV_IND = list(x_vals).index(XV)
        avoid_mask[XV_IND] = 0
        avoid_mask[XV_IND-1] = 0 #last epoch of section before break

    return avoid_mask

def estimate_prot(in_x, in_y, TID):
    mod_in_x = copy.deepcopy(in_x)
    mod_in_x-=np.min(mod_in_x)
    
    mod_in_y = copy.deepcopy(in_y)
    mod_in_y -=np.median(in_y)
    std_in_y = 2.0*np.std(mod_in_y)

    mod_in_x = mod_in_x[mod_in_y>-std_in_y]
    mod_in_y = mod_in_y[mod_in_y>-std_in_y]
    
    frequency = np.linspace(0.1, 10.0, 10000)
    period_array = 1.0/frequency
    power = LombScargle(mod_in_x, mod_in_y).power(frequency)

    P_ROT = period_array[list(power).index(np.max(power))]

    per_mask = np.zeros(len(period_array))
    if P_ROT<2.0:
        per_mask[np.absolute(period_array-P_ROT)<0.25] = 1
    elif (P_ROT>2.0) and (P_ROT<6.0):
        per_mask[np.absolute(period_array-P_ROT)<1.0] = 1
    else:
        per_mask[np.absolute(period_array-P_ROT)<1.5] = 1
        
    PWR_STD = np.std(power[per_mask==0])
    P_SNR = np.round(np.max(power)/PWR_STD, 1)

    # check i=48
    if P_SNR<50:
        P_FLAG=1
        y="no"
    else:
        P_FLAG=0
        y="yes"

    """
    fig, ax = plt.subplots(figsize=(11, 4))
    plt.axis('off')

    ax1 = fig.add_subplot(121)

    plt.title(str(i_prime)+"  S/N = "+str(P_SNR)+" Real = "+y)
    plt.axvline(P_ROT, color="red")
    plt.plot(period_array[per_mask==0], power[per_mask==0],marker="o",ms=2, ls="none")
    plt.plot(period_array[per_mask==1], power[per_mask==1])
    
    ax2 = fig.add_subplot(122)
    plt.plot(mod_in_x % P_ROT, mod_in_y, marker="+", ls="none", color="black", alpha=0.333)
    plt.ylim(-3.0*np.std(mod_in_y), 3.0*np.std(mod_in_y))
    
    plt.tight_layout()
    plt.show()
    """
    
    return (P_ROT, P_SNR, P_FLAG)

"""
prot_index=[]
prot_tid=[]
prot_val=[]
prot_desig=[]
with open("p_rot_info_20s_cadence.csv","r") as PROT_INFO:
    next(PROT_INFO)
    for lines in PROT_INFO:
        prot_index.append(int(lines.split(",")[0]))
        prot_tid.append(int(lines.split(",")[1]))
        prot_val.append(float(lines.split(",")[2]))
        prot_desig.append(str(lines.split(",")[3]))
prot_index = np.array(prot_index)
prot_tid = np.array(prot_tid)
prot_val = np.array(prot_val)
prot_desig = np.array(prot_desig)
"""


successes = np.array(glob.glob("./tess_lcvs/tess*_lc.fits"))
sectors = np.array([int(x.split("-")[1].replace("s","")) for x in successes]).astype(int)

successes = np.sort(np.unique(np.array([int(x.split("-")[2]) for x in successes])))

outfile_str = "initial_tess_flare_candidates.csv"
os.system("rm ./initial_tess_flare_candidates.csv")

for i_prime in range(len(successes)):

    TID = int(successes[i_prime])

    try:
        tess_bjd, sap_flux, sap_flux_err, flags = build_tess_lightcurve(TID, "./tess_lcvs/")
    except (UnboundLocalError):
        print "UnboundLocalError",TID
        continue
    sort_index = np.argsort(tess_bjd)
    tess_bjd = tess_bjd[sort_index]
    sap_flux = sap_flux[sort_index]
    sap_flux_err = sap_flux_err[sort_index]
    flags = flags[sort_index]
    
    # bad times
    # plt axvspans of these values only and look for patterns
    
    mask = np.zeros(len(tess_bjd))
    mask[(tess_bjd>2174.21)&(tess_bjd<2174.23)]=1
    mask[(tess_bjd>2099.72)&(tess_bjd<2102.35)]=1
    mask[(tess_bjd>2306.81)&(tess_bjd<2309.43)]=1
    mask[(tess_bjd>2319.9)&(tess_bjd<2323.1)]=1
    mask[(tess_bjd>2169.58)&(tess_bjd<2176.6)]=1
    mask[(tess_bjd>2126.73)&(tess_bjd<2130.27)]=1
    mask[(tess_bjd>2185.7)&(tess_bjd<2187.32)]=1
    mask[(tess_bjd>2112.92)&(tess_bjd<2115.6)]=1
    mask[(tess_bjd>2268.5)&(tess_bjd<2272.1)]=1 #bad pixels
    mask[(tess_bjd>2186.7)&(tess_bjd<2187.3)]=1
    mask[(tess_bjd>2228.07)&(tess_bjd<2229.31)]=1
    mask[(tess_bjd>2253.79)&(tess_bjd<2256.16)]=1
    mask[(tess_bjd>2241.37)&(tess_bjd<2242.52)]=1
    mask[(tess_bjd>2266.24)&(tess_bjd<2267.3)]=1
    mask[(tess_bjd>2142.5)&(tess_bjd<2143.78)]=1
    mask[(tess_bjd>2280.35)&(tess_bjd<2282.01)]=1
    mask[(tess_bjd>2157.02)&(tess_bjd<2158.64)]=1
    mask[(tess_bjd>2141.5)&(tess_bjd<2143.4)]=1
    mask[(tess_bjd>2098.5)&(tess_bjd<2100.5)]=1
    mask[(tess_bjd>2293.5)&(tess_bjd<2295.95)]=1
    mask[(tess_bjd>2281.94)&(tess_bjd<2282.64)]=1
    mask[(tess_bjd>2071.51)&(tess_bjd<2074.4)]=1
    mask[(tess_bjd>2084.3)&(tess_bjd<2087.4)]=1
    mask[(tess_bjd>2156.59)&(tess_bjd<2157.1)]=1

    #in transit data

    mask[(tess_bjd>1531.5)&(tess_bjd<1536.9)]=1
    mask[(tess_bjd>1463.7)&(tess_bjd<1464.1)]=1
    mask[(tess_bjd>1421.09)&(tess_bjd<1422.6)]=1
    mask[(tess_bjd>2247.0)&(tess_bjd<2248.05)]=1
    mask[(tess_bjd>1491.45)&(tess_bjd<1492.9)]=1
    mask[(tess_bjd>1504.5)&(tess_bjd<1505.7)]=1
    mask[(tess_bjd>2229.1)&(tess_bjd<2230.1)]=1
    mask[(tess_bjd>2242.3)&(tess_bjd<2243.3)]=1
    #mask[(tess_bjd>)&(tess_bjd<)]=1
    mask[(tess_bjd>1790.87)&(tess_bjd<1792.5)]=1
    #mask[(tess_bjd>)&(tess_bjd<)]=1
    #mask[(tess_bjd>)&(tess_bjd<)]=1
    
    #remove systematics affected times:
    tess_bjd = tess_bjd[mask==0]
    sap_flux = sap_flux[mask==0]
    sap_flux_err = sap_flux_err[mask==0]
    flags = flags[mask==0]

    tess_bjd = tess_bjd[flags==0]
    sap_flux = sap_flux[flags==0]
    sap_flux_err = sap_flux_err[flags==0]
    flags = flags[flags==0]
    
    x_vals = copy.deepcopy(tess_bjd)
    y_vals = copy.deepcopy(sap_flux)

    obs_in_minutes = len(x_vals)*2.0 #approx, some obs 10 sec off
    obs_in_days = np.round(obs_in_minutes/(24.0*60.0), 4)

    #if making FFDs, make this file too:
    #obs_info = str(TID)+","+str(obs_in_days)+"\n"
    #with open("toi_obs_times.csv","a") as ADDENDUM:
    #    ADDENDUM.write(obs_info)
    #continue   
    
    P_ROT, P_SNR, P_FLAG= estimate_prot(x_vals, y_vals, TID)

    #set low SNR and long-period rotators to 2 d periods.
    PROT = copy.deepcopy(P_ROT)
    PROT=0.5*PROT #to account for multi-spot rotators
    if P_SNR<50.0:
        PROT=2.0
    if PROT>2.0:
        PROT=2.0
    
    print i_prime, TID, PROT
    
    #apply smoothing for pass 1 (just to get a flatter light curve prior to flare ID)

    avoid_mask = cut_outliers_before_smooth(x_vals, sap_flux)
    
    x_vals_reduce = copy.deepcopy(x_vals[avoid_mask==0])
    y_vals_reduce = copy.deepcopy(y_vals[avoid_mask==0])
    

    xval_breaks = obtain_lc_breakpoints(x_vals) #(re)set here to keep following logic consistent with earlier versions of the code
    
    avoid_mask = fortify_lcv(avoid_mask, xval_breaks) #fortify first and last epochs in each section
    
    x_vals, SG_model_flux = systematics_removal_by_breakpoint(x_vals, y_vals, avoid_mask, xval_breaks, PROT)

    """
    # for testing purposes:
    plt.plot(x_vals, y_vals, marker="+",ls="none",color="black")
    plt.plot(x_vals[avoid_mask==0], SG_model_flux[avoid_mask==0], marker="+",ls="none",color="orange")
    plt.plot(x_vals[avoid_mask==1], SG_model_flux[avoid_mask==1], marker="+",ls="none",color="salmon")
    plt.show()
    #continue
    #exit()
    """
    
    y_vals = y_vals - SG_model_flux

    # perform flare search here
    # uncomment either the 2min or 20sec version (note the 20 sec version works on both, but is slower):
    low_times, flare_times, upp_times, flare_signifs = \
        get_flare_candidates_2min(x_vals, y_vals, avoid_mask, 4.5)
    
    #low_times, flare_times, upp_times, flare_signifs = \
    #    get_flare_candidates_20sec(x_vals, y_vals, avoid_mask, 4.5)

    for cands_index in range(len(low_times)):
        info = str(i_prime)+","+str(TID)+","+str(np.round(P_ROT,5))+","+str(np.round(P_SNR,1))+","+str(low_times[cands_index])+","+str(flare_times[cands_index])+","+str(upp_times[cands_index])+","+str(np.round(flare_signifs[cands_index],1))+"\n"
        with open(outfile_str, "a") as OUTFILE:
            OUTFILE.write(info)

    """
    # for inspection purposes:
    fig, ax = plt.subplots(figsize=(7, 6.5))
    plt.axis('off')

    ax1 = fig.add_subplot(211)
    
    plt.title(i_prime)
    plt.plot(tess_bjd, sap_flux, marker="+",ls="none", color="black", alpha=0.999)
    plt.xlabel("TBJD [d]")
    plt.xlabel("SAP Flux")

    ax2 = fig.add_subplot(212)
    plt.title(i_prime)
    plt.plot(tess_bjd % PROT, sap_flux, marker="+",ls="none", color="black", alpha=0.999)
    plt.xlabel("TBJD [d]")
    plt.xlabel("SAP Flux")
    plt.tight_layout()
    
    plt.show()
    """
    
    #exit()
