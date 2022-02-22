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

def get_Q0_luminosity(d, inband_mag, bandpass):

    TESS_0 = 4.03*(10.0**(-6.0)) #Sullivan 2015, 4.03 x 10^-6 erg /s cm2
    TESS_0 = TESS_0 * (10.0**(-7.0)) #J/erg, now in J/(s cm^2) or W/cm^2
    TESS_0 = TESS_0 * (10.0**4.0) #cm^2 / m^2, now in W/m^2

    #print TESS_0 #W/m^2
    #print np.log10(TESS_0)


    meters_per_pc = 3.086*(10.0**(16.0))
    #d = 1.3018 #Proxima dist, parsec
    d*=meters_per_pc #parsecs to meters
    d*=100.0 #meters to cm

    mu=0.15 # eff g' in micrometers from Gemini
    g0_flux = 5.41*(10.0**-8.0) # in W/m^2/mu
    watt_to_erg_per_sec = 10.0**7.0 # J/sec to ergs/sec
    meter2_to_cm2 = 10.0**(-4.0) # m^2 / cm^2

    if bandpass == "g-mag":
        g0_flux_erg = g0_flux*mu*watt_to_erg_per_sec*meter2_to_cm2
    elif bandpass == "T-mag":
        g0_flux_erg = TESS_0*watt_to_erg_per_sec*meter2_to_cm2
    else:
        print ("No such bandpass",bandpass)
        print ("Exiting now.")
        exit()

    #from apparent mag to stellar flux:
    F_star = g0_flux_erg * 10.0**((inband_mag)/(-2.5))

    Q_0 = 4.0*np.pi*(d**2.0)*F_star #quiesc. flux in g' in erg/s

    return Q_0

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
    avoid_times = copy.deepcopy(in_tbjd[in_avoid_mask==1])

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
            deflared_tbjd_pt, deflared_flux_pt, SG_deflared_flux_pt = SG_lcv_smoother(deflared_tbjd[(deflared_tbjd>PREV_BRK)&(deflared_tbjd<BRK)], deflared_flux[(deflared_tbjd>PREV_BRK)&(deflared_tbjd<BRK)], PROT)
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
    extend_avoid_mask = np.zeros(len(all_sections_deflared_tbjd))
    
    #define new mask, and remove anything within 0.005 d before flare stop and after flare end, removing SG artifacts
    for AV_TIME in avoid_times:
        diff_deflared_tbjd = np.absolute(AV_TIME-all_sections_deflared_tbjd)
        extend_avoid_mask[diff_deflared_tbjd<0.01]=1
    #all_sections_deflared_tbjd
    #all_sections_SG_deflared_flux
    
    interp_SG = interp1d(all_sections_deflared_tbjd[extend_avoid_mask==0], all_sections_SG_deflared_flux[extend_avoid_mask==0], kind="linear", bounds_error=False, fill_value=1.0)
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

def get_lcv_CSV(TID):

    file_str = glob.glob("./lcvs_asCSV/tic-"+str(TID)+"_pressed_lc.csv")[0]
    
    tess_bjd=[]
    sap_flux=[]
    sap_flux_err=[]
    flags=[]
    with open(file_str,"r") as INFILE:
        for lines in INFILE:
            tess_bjd.append(float(lines.split(",")[0]))
            sap_flux.append(float(lines.split(",")[1]))
            sap_flux_err.append(float(lines.split(",")[2]))
            flags.append(float(lines.split(",")[3].rstrip("\n")))
    tess_bjd = np.array(tess_bjd)
    sap_flux = np.array(sap_flux)
    sap_flux_err = np.array(sap_flux_err)
    flags = np.array(flags)

    return (tess_bjd, sap_flux, sap_flux_err, flags)


def get_flare_candidates(x_vals, y_vals, avoid_mask, signif):

    x_val_index_array = np.arange(len(x_vals))
    
    # reinforce time-sorted lc x vals
    timesort_inds = np.argsort(x_vals)
    x_vals = x_vals[timesort_inds]
    y_vals = y_vals[timesort_inds]
    avoid_mask = avoid_mask[timesort_inds]
    
    one_sigma = np.std(y_vals[avoid_mask==0])
    sigma_y_vals = (y_vals - np.median(y_vals[avoid_mask==0])) / one_sigma

    candidate_list = copy.deepcopy(x_vals[sigma_y_vals>=3.0])

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

suppl_TIC=[]
suppl_tmag=[]
suppl_dist=[]
with open("filled_dists_tmags_suppl.csv","r") as INFILE_2:
    for lines in INFILE_2:
        suppl_TIC.append(int(lines.split(",")[0]))
        suppl_tmag.append(float(lines.split(",")[1]))
        suppl_dist.append(float(lines.split(",")[2]))
suppl_TIC=np.array(suppl_TIC)
suppl_tmag=np.array(suppl_tmag)
suppl_dist=np.array(suppl_dist)

"""
FP_array=[]
with open("list_of_byhand_flarebyflare_FPs.csv","r") as INFILE:
    next(INFILE)
    for lines in INFILE:
        DESIG = str(lines.split(",")[1].replace(" ","").rstrip("\n"))
        if "n" in DESIG:
            #print DESIG
            FP_array.append(int(lines.split(",")[0]))
FP_array = np.array(FP_array)
"""

uniq_init_i_prime=[]
init_i_prime=[]
initial_TID=[]
init_Prot=[]
init_Psnr=[]
init_low_times=[]
init_flare_times=[]
init_upp_times=[]
init_flare_signifs=[]
with open("lv2_initial_tess_flare_candidates.csv","r") as INFILE:
    for lines in INFILE:
        uniq_init_i_prime.append(int(lines.split(",")[0]))
        init_i_prime.append(int(lines.split(",")[1]))
        initial_TID.append(int(lines.split(",")[2]))
        init_Prot.append(float(lines.split(",")[3]))
        init_Psnr.append(float(lines.split(",")[4]))
        init_low_times.append(float(lines.split(",")[5]))
        init_flare_times.append(float(lines.split(",")[6]))
        init_upp_times.append(float(lines.split(",")[7]))
        init_flare_signifs.append(float(lines.split(",")[8].rstrip("\n")))
uniq_init_i_prime=np.array(uniq_init_i_prime)
init_i_prime=np.array(init_i_prime)
initial_TID=np.array(initial_TID)
init_Prot=np.array(init_Prot)
init_Psnr=np.array(init_Psnr)
init_low_times=np.array(init_low_times)
init_flare_times=np.array(init_flare_times)
init_upp_times=np.array(init_upp_times)
init_flare_signifs=np.array(init_flare_signifs)


init_log_Q0=[]
for i in range(len(uniq_init_i_prime)):
    TID = initial_TID[i]
    DIST = suppl_dist[suppl_TIC==TID][0]
    TMAG = suppl_tmag[suppl_TIC==TID][0]

    if DIST>1.0:
        Q_0 = get_Q0_luminosity(DIST, TMAG, "T-mag")
    else:
        Q_0 = 1
    init_log_Q0.append(np.round(np.log10(Q_0),2))
init_log_Q0=np.array(init_log_Q0)

sort_uniq_i_prime = np.sort(np.unique(init_i_prime))

for i_prime in sort_uniq_i_prime:

    #if i_prime != 35: #havent done 144 yet
    #    continue

    TID = int(initial_TID[init_i_prime==i_prime][0])

    Q0 = 10.0**init_log_Q0[initial_TID==TID][0]
    LOG_Q0 = np.round(np.log10(Q0), 2)
    
    #if flarewise_TID != TID:
    #    continue

    P_ROT = init_Prot[init_i_prime==i_prime][0]
    P_SNR = init_Psnr[init_i_prime==i_prime][0]
    PROT = copy.deepcopy(P_ROT)
    PROT=0.5*PROT #to account for multi-spot rotators
    if P_SNR<50.0:
        PROT=2.0
    if PROT>2.0:
        PROT=2.0
        
    print i_prime, TID, PROT

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

    # initial_TID, init_low_times, init_flare_times, init_upp_times, init_flare_signifs
    
    avoid_mask = np.zeros(len(x_vals))
    target_flstarts = init_low_times[initial_TID==TID]
    target_flpeaks = init_flare_times[initial_TID==TID]
    target_flstops = init_upp_times[initial_TID==TID]
    target_signifs = init_flare_signifs[initial_TID==TID]
    target_longindex = uniq_init_i_prime[initial_TID==TID]
    
    for sp_ind in range(len(target_flstarts)):
        avoid_mask[(x_vals>=target_flstarts[sp_ind]) & (x_vals<=target_flstops[sp_ind])] = 1

    xval_breaks = obtain_lc_breakpoints(x_vals)
    
    x_vals, SG_model_flux = systematics_removal_by_breakpoint(x_vals, y_vals, avoid_mask, xval_breaks, PROT)
    
    uncorr_y_vals = copy.deepcopy(y_vals)
    y_vals = y_vals - SG_model_flux

    for sp_ind in range(len(target_flstarts)):
        FLSTART = target_flstarts[sp_ind]
        FLPEAK = target_flpeaks[sp_ind]
        FLSTOP = target_flstops[sp_ind]
        FLSIGNIFS = target_signifs[sp_ind]
        LONG_INDEX = target_longindex[sp_ind]

        if LONG_INDEX in FP_array:
            continue
        
        x_fl = copy.deepcopy(x_vals[(x_vals>=FLSTART)&(x_vals<=FLSTOP)])
        y_fl = copy.deepcopy(y_vals[(x_vals>=FLSTART)&(x_vals<=FLSTOP)])
        x_fl_insec = x_fl*24.0*3600.0
        x_fl_insec -= np.min(x_fl_insec)

        ED = np.round(np.trapz(y_fl, x_fl_insec))
        if ED<1:
            ED=1.0

        ERG = np.round(np.log10((ED*Q0)/0.187),2)

        AMPL = np.max(y_fl)

        #if (DESIG=="FP"):
        #    continue
        
        info_out = str(LONG_INDEX)+","+str(i_prime)+","+str(TID)+","+str(P_ROT)+","+str(P_SNR)+","+str(FLSTART)+","+str(FLPEAK)+","+str(FLSTOP)+","+str(FLSIGNIFS)+","+str(LOG_Q0)+","+str(ED)+","+str(ERG)+","+str(np.round(AMPL,4))+"\n"
        with open("lv3_final_tess_flare_candidates.csv","a") as OUTFILE_FIN:
            OUTFILE_FIN.write(info_out)

        
        fig, ax = plt.subplots(figsize=(9, 6.5))
        plt.axis('off')

        ax1 = fig.add_subplot(312)
        DESIG="N/A"
        
        for sp_ind2 in range(len(target_flstarts)):
            plt.axvspan(target_flstarts[sp_ind2], target_flstops[sp_ind2], facecolor="grey", alpha=0.333)
        plt.plot(x_vals, uncorr_y_vals, marker="+",ls="none",color="black")
        plt.plot(x_vals, SG_model_flux, marker="+",ls="none",color="orange")

        plt.axvline(target_flstarts[sp_ind],ls="--",color="red")
        plt.axvline(target_flstops[sp_ind],ls="--",color="red")
    
        #plt.xlabel("TBJD [d]")
        plt.ylabel("SAP Flux")
        plt.xlim(target_flstarts[sp_ind]-0.04, target_flstops[sp_ind]+0.04)
        MX = 1.05*np.nanmax(uncorr_y_vals[(x_vals>=(target_flstarts[sp_ind]-0.01))&(x_vals<=(target_flstops[sp_ind]+0.01))])
        MN = 0.95*np.nanmin(uncorr_y_vals[(x_vals>=(target_flstarts[sp_ind]-0.01))&(x_vals<=(target_flstops[sp_ind]+0.01))])
        plt.ylim(MN, MX)
        
        ax2 = fig.add_subplot(311)
        plt.title("L="+str(LONG_INDEX)+",  I_P="+str(i_prime)+",  DES="+DESIG+",  Ebol="+str(ERG)+" erg")
        
        for sp_ind2 in range(len(target_flstarts)):
            plt.axvspan(target_flstarts[sp_ind2], target_flstops[sp_ind2], facecolor="grey", alpha=0.333)
        plt.plot(x_vals, y_vals, marker="+",ls="none",color="black")

        plt.axvline(target_flstarts[sp_ind],ls="--",color="red")
        plt.axvline(target_flstops[sp_ind],ls="--",color="red")
        
        #plt.xlabel("TBJD [d]")
        plt.ylabel("SAP Flux")
        plt.xlim(target_flstarts[sp_ind]-0.04, target_flstops[sp_ind]+0.04)
        MX = 1.1*np.nanmax(y_vals[(x_vals>=(target_flstarts[sp_ind]-0.04))&(x_vals<=(target_flstops[sp_ind]+0.04))])
        MN = 0.9*np.nanmin(y_vals[(x_vals>=(target_flstarts[sp_ind]-0.04))&(x_vals<=(target_flstops[sp_ind]+0.04))])
        plt.ylim(MN, MX)

        ax3 = fig.add_subplot(313)

        plt.plot(x_vals, uncorr_y_vals, marker="+",ls="none",color="black")
        plt.plot(x_vals, SG_model_flux, marker="+",ls="none",color="orange")
    
        for u in range(len(target_flpeaks)):
            plt.axvline(target_flpeaks[u], linewidth=2.5, color="red",zorder=0.01)
    
        plt.xlabel("TBJD [d]")
        plt.ylabel("SAP Flux")
    
        plt.tight_layout()
    
        plt.show()
        #exit()
        
