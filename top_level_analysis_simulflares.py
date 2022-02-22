import glob
import copy
import numpy as np
import os
import astropy.table as tbl
from astropy import time, coordinates as coord, units as u
from paper_data_read_functions import read_table_one
from paper_data_read_functions import read_table_two as read_table_two
from astropy.stats import LombScargle
from astropy.io import fits
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
import scipy.signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def linear(x, m, b):
    return m*x + b

def davenport_flare_model(epochs, width, ampl, peak_time):

    if width==0.0 and ampl==0.0 and peak_time==0.0:
        return np.zeros(len(epochs))
    elif ampl<0.000001:
        ampl=100000.0
    else:
        pass
    
    # See Davenport et al 2014, ApJ 797, 2
    t = copy.deepcopy(epochs)
    t/=width
    #assume a gaussian-distribution 1-5mins for risetime, 20-30mins decay
    try:
        flare_y = 0.689*np.exp(-1.6*(t-peak_time),dtype=np.float64) + 0.303*np.exp(-0.2783*(t-peak_time),dtype=np.float64)
    except (RuntimeWarning):
        #print np.exp(-1.6*(t-peak_time),dtype=np.float64)
        #print np.exp(-0.2783*(t-peak_time),dtype=np.float64)
        exit()
    flare_y[(t-peak_time)<0] = 1.0 + (1.941)*(t-peak_time)[(t-peak_time)<0] - 0.175*((t-peak_time)[(t-peak_time)<0]**2) - 2.246*((t-peak_time)[(t-peak_time)<0]**3) - 1.125*((t-peak_time)[(t-peak_time)<0]**4)

    flare_y[flare_y<0.000001]= 0.000001 #set all flare points that would otherwise be negative to val near zero
    flare_y = flare_y*ampl #adjust flare peak amplitude
    #rise; FWHMs not minutes
    return flare_y

def two_peak_dav_flare_model(epochs, width1, ampl1, peak_time1, width2, ampl2, peak_time2):
    flare_y1 = davenport_flare_model(epochs, width1, ampl1, peak_time1)
    flare_y2 = davenport_flare_model(epochs, width2, ampl2, peak_time2)
    flare_y = flare_y1 + flare_y2
    return flare_y

def three_peak_dav_flare_model(epochs, width1, ampl1, peak_time1, width2, ampl2, peak_time2, width3, ampl3, peak_time3):
    flare_y1 = davenport_flare_model(epochs, width1, ampl1, peak_time1)
    flare_y2 = davenport_flare_model(epochs, width2, ampl2, peak_time2)
    flare_y3 = davenport_flare_model(epochs, width3, ampl3, peak_time3)
    flare_y = flare_y1 + flare_y2 + flare_y3
    return flare_y

def four_peak_dav_flare_model(epochs, width1, ampl1, peak_time1, width2, ampl2, peak_time2, width3, ampl3, peak_time3, width4, ampl4, peak_time4):
    flare_y1 = davenport_flare_model(epochs, width1, ampl1, peak_time1)
    flare_y2 = davenport_flare_model(epochs, width2, ampl2, peak_time2)
    flare_y3 = davenport_flare_model(epochs, width3, ampl3, peak_time3)
    flare_y4 = davenport_flare_model(epochs, width3, ampl3, peak_time3)
    flare_y = flare_y1 + flare_y2 + flare_y3 + flare_y4
    return flare_y

def compute_1sigma_CI(input_array):

    sorted_input_array = np.sort(input_array)

    low_ind = int(0.16*len(input_array))
    high_ind = int(0.84*len(input_array))

    bot_val = (sorted_input_array[:low_ind])[-1]
    top_val = (sorted_input_array[high_ind:])[0]

    bot_arr_err = abs(np.nanmedian(input_array) - bot_val)
    top_arr_err = abs(np.nanmedian(input_array) - top_val)

    return (np.nanmedian(input_array), bot_arr_err, top_arr_err)

def get_ATLAS_gmag():
    atl_TIC=[]
    atl_RA=[]
    atl_Dec=[]
    atl_g=[]
    atl_dg=[]
    atl_r=[]
    atl_dr=[]
    atl_i=[]
    atl_di=[]
    atl_dist=[]
    with open("EVRYFLARE_atlas_xmatch.csv","r") as INFILE:
        next(INFILE)
        for lines in INFILE:
            atl_TIC.append(int(lines.split(",")[0]))
            atl_RA.append(float(lines.split(",")[1]))
            atl_Dec.append(float(lines.split(",")[2]))
            atl_g.append(float(lines.split(",")[3]))
            atl_dg.append(float(lines.split(",")[4]))
            atl_r.append(float(lines.split(",")[5]))
            atl_dr.append(float(lines.split(",")[6]))
            atl_i.append(float(lines.split(",")[7]))
            atl_di.append(float(lines.split(",")[8]))
            atl_dist.append(float(lines.split(",")[9]))
    atl_TIC=np.array(atl_TIC)
    atl_RA=np.array(atl_RA)
    atl_Dec=np.array(atl_Dec)
    atl_g=np.array(atl_g)
    atl_dg=np.array(atl_dg)
    atl_r=np.array(atl_r)
    atl_dr=np.array(atl_dr)
    atl_i=np.array(atl_i)
    atl_di=np.array(atl_di)
    atl_dist=np.array(atl_dist)
    
    return (atl_TIC, atl_g, atl_dg)

def get_tess_lcv(fits_table_filename):

    hdulist = fits.open(fits_table_filename)  # open a FITS file

    data = hdulist[1].data  # assume the first extension is a table
    print hdulist[1].columns
    exit()
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
    
    first_sector = np.min(sector_list)
    last_sector = np.max(sector_list)

    return (tess_bjd,sap_flux,sap_flux_err,first_sector,last_sector)

_lambda = np.linspace(1.0, 1500.0, num=1000) # nm
Lambda = copy.deepcopy(_lambda)

def blackbody_spectrum(_lambda, temp):
    __lambda = copy.deepcopy(_lambda)
    __lambda *= (10.0**(-9.0)) # from nm to m
    h = 6.62607004 * 10.0**(-34.0) # Planck's constant m^2 kg / s
    c = 3.0*(10.0**8.0) #m/s
    k_B = 1.38064852 * 10.0**(-23.0) # Boltzmann constant (m^2 kg / s^2 K)
    T = temp #9000.0 # K
    flux = 2.0*h*(c**2.0) / ((__lambda**5.0)*(np.exp(h*c/(__lambda*k_B*T),dtype=np.float128)-1.0))

    return flux

# Green filters:
filt_wavelength=[]
filt_response=[]
with open("./run4/ctio_omega_g.csv","r") as FILTER:
    for lines in FILTER:
        filt_wavelength.append(float(lines.split(",")[0]))
        filt_response.append(0.01*float(lines.split(",")[1].rstrip("\n")))
filt_wavelength=np.array(filt_wavelength)
filt_response=np.array(filt_response)

CCD_wavelength=[]
CCD_response=[]
with open("./run4/ML290050_QE.csv","r") as CCD:
    for lines in CCD:
        CCD_wavelength.append(float(lines.split(",")[0]))
        CCD_response.append(0.01*float(lines.split(",")[1].rstrip("\n")))
CCD_wavelength=np.array(CCD_wavelength)
CCD_response=np.array(CCD_response)

interp_filt=interp1d(filt_wavelength,filt_response,kind="linear",fill_value=0.0,bounds_error=False)
interp_CCD=interp1d(CCD_wavelength,CCD_response,kind="linear",fill_value=0.0,bounds_error=False)

g_response_fn = interp_filt(Lambda)*interp_CCD(Lambda)

#A = np.trapz(g_response_fn,Lambda)
#B = np.trapz(filt_response,filt_wavelength)

in_gband = g_response_fn[g_response_fn > 0.05]
g_response_fn/=np.average(in_gband)

# TESS response function:
TESS_wavelength=[]
TESS_response=[]
with open("./run4/tess-response-function-v1.0.csv","r") as FILTER:
    next(FILTER) #1
    next(FILTER) #2
    next(FILTER) #3
    next(FILTER) #4
    next(FILTER) #5
    next(FILTER) #6
    next(FILTER) #7
    next(FILTER) #8
    next(FILTER) #9
    
    for lines in FILTER:
        TESS_wavelength.append(float(lines.split(",")[0]))
        TESS_response.append(float(lines.split(",")[1].rstrip("\n")))
TESS_wavelength=np.array(TESS_wavelength)
TESS_response=np.absolute(np.array(TESS_response))

interp_TESS = interp1d(TESS_wavelength,TESS_response,kind="linear",fill_value=0.0,bounds_error=False)

TESS_response_fn = interp_TESS(Lambda)

in_Tband = TESS_response_fn[TESS_response_fn > 0.05]
TESS_response_fn/=np.average(in_Tband)

BB_temps = np.linspace(500.0,50000.0,num=4000) # set to 2000 steps!!!

ratios=[]
arr_estfrac_in_g=[]
arr_estfrac_in_T=[]
for i in range(len(BB_temps)):
    flux = blackbody_spectrum(_lambda, BB_temps[i])

    flux_in_g = flux*g_response_fn
    flux_in_T = flux*TESS_response_fn

    area_in_g = np.trapz(flux_in_g, _lambda)
    area_in_T = np.trapz(flux_in_T, _lambda)
    area_in_tot = np.trapz(flux, _lambda)

    estfrac_in_g = area_in_g/area_in_tot
    estfrac_in_T = area_in_T/area_in_tot
    
    ratios.append(area_in_g/area_in_T)
    arr_estfrac_in_g.append(estfrac_in_g)
    arr_estfrac_in_T.append(estfrac_in_T)
    
    #print BB_temps[i],"K"
ratios=np.array(ratios)
arr_estfrac_in_g = np.array(arr_estfrac_in_g)
arr_estfrac_in_T = np.array(arr_estfrac_in_T)

ratios[0] = 0.0
ratios[-1]=10.0

plt.plot(ratios, BB_temps)
plt.xlabel("Ratio")
plt.ylabel("Teff")
plt.show()

plt.plot(BB_temps, arr_estfrac_in_T, color="firebrick")
plt.plot(BB_temps, arr_estfrac_in_g, color="royalblue")
plt.text(17235, 0.15, "Evryscope bandpass", color="royalblue",fontsize=14)
plt.text(5495.0, 0.54, "TESS bandpass", color="firebrick",fontsize=14)

plt.xlabel("$T_\mathrm{Eff}$ [K]",fontsize=14)
plt.ylabel("Fraction of total blackbody flux in band",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
#plt.savefig("fraction_flux_inband.png")
plt.show()
#exit()

get_teff = interp1d(ratios, BB_temps, kind="linear")
#exit()

######################################################################

def fast_pre_whiten_blur(epochs,mags,white_cutoff):
    smoothed_vals=[]
    for i in np.arange(epochs.shape[0]):
        gauss = np.exp(-0.5*(((epochs[i]*np.ones(epochs.shape[0])-epochs)/white_cutoff)**2.0))
        epoch_tot = np.sum(gauss*mags)    
        epoch_weight = np.sum(gauss) #weight further away points more heavily

        smoothed_vals.append(epoch_tot/(1.0*epoch_weight))

    return np.array(smoothed_vals).astype(float)

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

def correct_source_brightness(input_ticid, new_g, new_dg, new_T, new_dT):

    if input_ticid==294750180:
        new_g=11.452
        new_dg=0.041
        new_T=8.924 #9.015 (EXO-FOP combination) #8.924 (mine)
        new_dT=0.006
    elif input_ticid==5796048:
        new_g=13.441
        new_dg=0.015
        new_T=9.959
        new_dT=0.034
    elif input_ticid==206327797:
        new_g=12.94
        new_dg=0.02
        new_T=8.288
        new_dT=0.007
    elif input_ticid==220433364:
        new_g=11.377
        new_dg=0.026
        new_T= 8.255
        new_dT=0.008
    elif input_ticid==392756613:
        new_g= 10.182 #and 12.750 | 0.013 (apass dr9)
        new_dg=0.083
        new_T=7.154
        new_dT=0.023
    elif input_ticid==142086812:
        new_g=10.866
        new_dg=0.067
        new_T= 8.194
        new_dT=0.02
    elif input_ticid==441398770:
        new_g=10.177
        new_dg=0.026
        new_T=7.396 #orig value
        new_dT=0.04 #orig value
    elif input_ticid==192543856:
        new_g=14.575
        new_dg=0.014
        new_T=10.9065
        new_dT=0.0087
    else:
        pass
        
    return (new_g, new_dg, new_T, new_dT)

def get_evr_lc(evr_filestring):
    evr_mjd=[]
    evr_gmag=[]
    evr_gmag_err=[]
    evr_snr=[]
    evr_lim_mag=[]
    with open(evr_filestring,"r") as INFILE:
        next(INFILE)
        for lines in INFILE:
            evr_mjd.append(float(lines.split(",")[0]))
            evr_gmag.append(float(lines.split(",")[1]))
            evr_gmag_err.append(float(lines.split(",")[2]))
            evr_snr.append(float(lines.split(",")[3]))
            evr_lim_mag.append(float(lines.split(",")[4].rstrip("\n")))
    evr_mjd=np.array(evr_mjd)
    evr_gmag=np.array(evr_gmag)-np.nanmedian(evr_gmag)
    evr_gmag_err=np.array(evr_gmag_err)
    evr_snr=np.array(evr_snr)
    evr_lim_mag=np.array(evr_lim_mag)
    evr_flags=np.zeros(len(evr_snr))
    evr_flags[evr_snr<10.0] = 1.0

    return (evr_mjd, evr_gmag, evr_gmag_err, evr_flags)

def get_prewh_lcv(targ):
    tbjd=[]
    tmags=[]
    with open(targ,"r") as INFILE:
        for lines in INFILE:
            tbjd.append(float(lines.split(",")[0]))
            tmags.append(float(lines.split(",")[2]))
    tbjd = np.array(tbjd) + 2457000.0 - 2400000.5
    tmags = np.array(tmags)
    
    return (tbjd, tmags)

def load_J2000_coords():

    gi_tic=[]
    gi_ra=[]
    gi_dec=[]
    with open("wshoward_evryflare_tess_gi_prop_c3.csv","r") as INFILE:
        next(INFILE)
        for lines in INFILE:
            gi_tic.append(int(lines.split(",")[0]))
            gi_ra.append(float(lines.split(",")[1]))
            gi_dec.append(float(lines.split(",")[2]))
    gi_tic=np.array(gi_tic).astype(int)
    gi_ra=np.array(gi_ra)
    gi_dec=np.array(gi_dec)
    
    return (gi_tic, gi_ra, gi_dec)

def load_MAST_data():
    ID=[]
    RA=[]
    Dec=[]
    MatchID=[]
    MatchRa=[]
    MatchDEC=[]
    dstArcSec=[]
    gmag=[]
    e_gmag=[]
    Tmag=[]
    e_Tmag=[]
    mass=[]
    e_mass=[]
    d=[]
    e_d=[]
    
    with open("reduced_MAST_Crossmatch_CTL_evryflare_III.csv","r") as INFILE:
        next(INFILE)
        for lines in INFILE:
            ID.append(int(lines.split(",")[0]))
            RA.append(float(lines.split(",")[1]))
            Dec.append(float(lines.split(",")[2]))
            MatchID.append(int(lines.split(",")[3]))
            MatchRa.append(float(lines.split(",")[4]))
            MatchDEC.append(float(lines.split(",")[5]))
            #dstArcSec.append(float(lines.split(",")[6]))
            #gmag.append(float(lines.split(",")[7]))
            #e_gmag.append(float(lines.split(",")[8]))
            Tmag.append(float(lines.split(",")[9]))
            e_Tmag.append(float(lines.split(",")[10]))
            #mass.append(float(lines.split(",")[11]))
            #e_mass.append(float(lines.split(",")[12]))
            d.append(float(lines.split(",")[13]))
            e_d.append(float(lines.split(",")[14]))
    MAST_ID=np.array(ID)
    MAST_RA=np.array(RA)
    MAST_Dec=np.array(Dec)
    MatchID=np.array(MatchID)
    MatchRa=np.array(MatchRa)
    MatchDEC=np.array(MatchDEC)
    #MAST_dstArcSec=np.array(dstArcSec)
    #MAST_gmag=np.array(gmag)
    #MAST_e_gmag=np.array(e_gmag)
    MAST_Tmag=np.array(Tmag)
    MAST_e_Tmag=np.array(e_Tmag)
    #MAST_mass=np.array(mass)
    #MAST_e_mass=np.array(e_mass)
    MAST_d=np.array(d)
    MAST_e_d=np.array(e_d)
    
    return (MatchID, MAST_Tmag, MAST_e_Tmag, MAST_d, MAST_e_d, MAST_RA, MAST_Dec)

def clean_evr_lc(i, evr_mjd_window, evr_fracflux, evr_fracflux_err):
    
    flag_bool = np.zeros(len(evr_mjd_window)).astype(int)
    cp_evr_mjd_window = copy.deepcopy(evr_mjd_window)
    cp_evr_fracflux = copy.deepcopy(evr_fracflux)
    cp_evr_fracflux_err = copy.deepcopy(evr_fracflux_err)
    
    if i==50:
        flag_bool[(evr_fracflux>0.3519)&(evr_mjd_window<0.0299)] = 1
        flag_bool[(evr_fracflux>2.3)&(evr_mjd_window>0.1139)] = 1
    elif i==76:
        flag_bool[(evr_mjd_window>0.085)] = 1
    elif i==110:
        flag_bool[(evr_fracflux>0.37)&(evr_mjd_window>0.083)] = 1
    elif i==161:
        flag_bool[(evr_mjd_window<0.01633)] = 1
    elif i==168:
        flag_bool[(evr_fracflux>0.76)] = 1
    elif i==233:
        flag_bool[(evr_mjd_window<0.1014)] = 1
    elif i==300:
        flag_bool[(evr_fracflux>1.313)] = 1
    elif i==500:
        flag_bool[(evr_fracflux<-0.11)] = 1
    elif i==553:
        flag_bool[(evr_fracflux>0.202)] = 1
        flag_bool[(evr_fracflux>0.097)&(evr_mjd_window>0.088)] = 1
    elif i==688: #ratchet offset?
        flag_bool[(evr_mjd_window>0.0784)] = 1
    elif i==729:
        flag_bool[(evr_fracflux>1.444)&(evr_mjd_window<0.0027)] = 1
    elif i==25:
        flag_bool[(evr_fracflux>3.0)] = 1
    else:
        pass
        
    return (cp_evr_mjd_window[flag_bool==0], cp_evr_fracflux[flag_bool==0], cp_evr_fracflux_err[flag_bool==0])

def adjust_evr_by_target(i, evr_mjd_window, evr_fracflux):

    i_list = []
    evr_start_list = []
    evr_stop_list = []
    evr_FWHM_start = []
    evr_FWHM_stop = []
    CW_start = []
    CW_stop = []
    with open("good_simulflares.csv","r") as INFILE:
        next(INFILE)
        for lines in INFILE:
            i_list.append(int(lines.split(",")[0]))
            evr_start_list.append(float(lines.split(",")[1]))
            evr_stop_list.append(float(lines.split(",")[2]))
            evr_FWHM_start.append(float(lines.split(",")[3]))
            evr_FWHM_stop.append(float(lines.split(",")[4]))
            CW_start.append(float(lines.split(",")[5]))
            CW_stop.append(float(lines.split(",")[6].rstrip("\n")))
    i_list = np.array(i_list)
    evr_start_list = np.array(evr_start_list)
    evr_stop_list = np.array(evr_stop_list)
    evr_FWHM_start = np.array(evr_FWHM_start)
    evr_FWHM_stop = np.array(evr_FWHM_stop)
    CW_start = np.array(CW_start)
    CW_stop = np.array(CW_stop)
    
    ESTART = evr_start_list[i_list==i][0]
    ESTOP = evr_stop_list[i_list==i][0]
    FWHM_START = evr_FWHM_start[i_list==i][0]
    FWHM_STOP = evr_FWHM_stop[i_list==i][0]
    CW_START = CW_start[i_list==i][0]
    CW_STOP = CW_stop[i_list==i][0]
    if i==50:
        EMEAN = np.nanmedian(evr_fracflux[(evr_mjd_window<=ESTART)])
    elif i==553:
        EMEAN = np.nanmedian(evr_fracflux[(evr_mjd_window>=ESTOP)])
    else:
        EMEAN = np.nanmedian(evr_fracflux[(evr_mjd_window<=ESTART)|(evr_mjd_window>=ESTOP)])

    evr_fracflux = evr_fracflux - EMEAN
    
    return (evr_mjd_window, evr_fracflux, ESTART, ESTOP, FWHM_START, FWHM_STOP,CW_START,CW_STOP)

#i, tess_mjd_window, tess_fracflux, inwindow_fracflux_err, tess_mjd_raw_window, tess_raw_fracflux, inwindow_raw_fracflux_err, NEWSTART, NEWSTOP

def adjust_tess_by_target(i, tess_mjd_window, tess_fracflux, inwindow_fracflux_err, tess_mjd_raw_window, tess_raw_fracflux, inwindow_raw_fracflux_err, NEWSTART, NEWSTOP):
    
    if i==17:
        tess_mjd_window = copy.deepcopy(tess_mjd_raw_window)
        tess_fracflux = copy.deepcopy(tess_raw_fracflux)
        inwindow_fracflux_err = copy.deepcopy(inwindow_raw_fracflux_err)
    elif i==110:
        tess_mjd_window =tess_mjd_window[3:]
        tess_fracflux = tess_fracflux[3:]
        inwindow_fracflux_err = inwindow_fracflux_err[3:]
    elif i==233:
        x_vals = copy.deepcopy(tess_mjd_window[(tess_mjd_window<=NEWSTART)|(tess_mjd_window>=(1.05*NEWSTOP))])
        y_vals = copy.deepcopy(tess_fracflux[(tess_mjd_window<=NEWSTART)|(tess_mjd_window>=(1.05*NEWSTOP))])
        popt, pcov = curve_fit(linear, x_vals, y_vals)

        tess_fracflux = tess_fracflux - linear(tess_mjd_window, *popt)
        
    elif i==606:
        tess_mjd_window = tess_mjd_window[tess_fracflux>-0.08555]
        tess_fracflux= tess_fracflux[tess_fracflux>-0.08555]
        inwindow_fracflux_err = inwindow_fracflux_err[tess_fracflux>-0.08555]
    elif i==661:
        tess_mjd_window = tess_mjd_window[tess_fracflux>-0.03]
        tess_fracflux= tess_fracflux[tess_fracflux>-0.03]
        inwindow_fracflux_err = inwindow_fracflux_err[tess_fracflux>-0.03]
    else:
        pass
    
    return (tess_mjd_window, tess_fracflux, inwindow_fracflux_err)

def get_JR_revised_spt():
    # JR_tic_id,JR_SpT,updated_T_eff,T_eff_EvryFlare_I,updated_masses=get_JR_revised_spt()
    JR_tic_id=[]
    JR_SpT=[]
    updated_SpT=[]
    updated_T_eff=[]
    T_eff_EvryFlare_I=[]
    updated_masses=[]
    spot_increases=[] #assuming non-0K spots...
    spot_temp=[]
    with open("EvryFlare_II_Table_Jeff_SpT_addendum.csv","r") as INPUT_FILE:
        next(INPUT_FILE)
        for lines in INPUT_FILE:
            JR_tic_id.append(int(lines.split(",")[0]))
            JR_SpT.append(lines.split(",")[1])
            updated_SpT.append(lines.split(",")[2])
            updated_T_eff.append(float(lines.split(",")[3]))
            T_eff_EvryFlare_I.append(float(lines.split(",")[4]))
            updated_masses.append(float(lines.split(",")[5]))
            spot_increases.append(float(lines.split(",")[6]))
            spot_temp.append(float(lines.split(",")[7]))
    JR_tic_id=np.array(JR_tic_id)
    JR_SpT=np.array(JR_SpT)
    updated_SpT=np.array(updated_SpT)
    updated_T_eff=np.array(updated_T_eff)
    T_eff_EvryFlare_I=np.array(T_eff_EvryFlare_I)
    updated_masses=np.array(updated_masses)
    spot_increases=np.array(spot_increases)
    spot_temp=np.array(spot_temp)
    
    return (JR_tic_id,JR_SpT,updated_SpT,updated_T_eff,T_eff_EvryFlare_I,updated_masses,spot_increases,spot_temp)

def fit_flare_coeffs(i,x_tess, y_tess):
    
    guess_width=0.01
    guess_ampl=np.max(y_tess)
    guess_peak_time=100.0*x_tess[list(y_tess).index(np.max(y_tess))]
    guess = [guess_width, guess_ampl, guess_peak_time]
    popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=guess, method="dogbox", maxfev=5000)
    popt_tess_s2 = [0.0, 0.0, 0.0]
    popt_tess_s3 = [0.0, 0.0, 0.0]

    #popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
    #print popt_tess_both
    #exit()
    popt_tess_all = popt_tess_s1 + popt_tess_s2 + popt_tess_s3

    sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    #print "during",i
    if i==0:
        ###################################
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=[0.0034679, 0.8512746, 16.5644449], method="dogbox", maxfev=5000)
    
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)

        x_tess_lv2 =x_tess[20:]
        y_tess_lv2 =y_tess[20:]-sampled_flare1[20:]
        popt_tess_s2, pcov = curve_fit(davenport_flare_model, x_tess_lv2, y_tess_lv2, p0=[0.004, 0.06, 20.56], method="dogbox", maxfev=5000)

        popt_tess_s3 = [0.0, 0.0, 0.0]
        #####################################
    if i==4:
        ###################################
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=[0.005, 0.6, 11.5], method="dogbox", maxfev=5000)
        ###################################
    if i==17:
        ###################################
        
        popt_tess_s1 = [0.004, 0.50124442, 12.25]
        popt_tess_s2 = [0.005, 0.23, 11.25]
        popt_tess_s3 = [0.004, 0.04, 20.25]

        popt_tess_both = popt_tess_s1 + popt_tess_s2
        popt_tess_all = popt_tess_s1 + popt_tess_s2 + popt_tess_s3
    
        #popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=5000)
        popt_tess_all, pcov = curve_fit(three_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_all, method="dogbox", maxfev=5000)

        sampled_flare = three_peak_dav_flare_model(x_tess, *popt_tess_all)

        popt_tess_s1 = popt_tess_all[:3]
        popt_tess_s2 = popt_tess_all[3:6]
        popt_tess_s3 = popt_tess_all[6:]

        #print popt_tess_s1,popt_tess_s2,popt_tess_s3
        #sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)
        #sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)

        #x_tess_lv2 =x_tess[(x_tess<0.0385)|(x_tess>0.0502)]
        #y_tess_lv2 =y_tess[(x_tess<0.0385)|(x_tess>0.0502)]-sampled_flare1[(x_tess<0.0385)|(x_tess>0.0502)]
        #x_tess_lv3 = x_tess[(x_tess>0.0768)]
        #y_tess_lv3 = y_tess[(x_tess>0.0768)] - sampled_flare[(x_tess>0.0768)]
        #popt, pcov = curve_fit(linear, x_tess_lv3, y_tess_lv3)
        #y_tess_lv3 = y_tess_lv3 - linear(x_tess_lv3, *popt)
    
        #sampled_flare3 = davenport_flare_model(x_tess_lv3, *popt_tess_s3)
        #popt_tess_s2, pcov = curve_fit(davenport_flare_model, x_tess_lv2, y_tess_lv2, p0=popt_tess_s2, method="dogbox", maxfev=5000)
        #sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        #sampled_flare = sampled_flare1 + sampled_flare2
    if i==19:
        popt_tess_s1 = [0.01629248, 0.27271776, 3.8689474]
        popt_tess_s2 = [0.01629248, 0.08, 4.7]
        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=5000)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
        #print popt_tess_s1, popt_tess_s2
        #exit()
        #popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=guess, method="dogbox", maxfev=5000)
        
        #sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        #x_tess_lv2 =x_tess[(x_tess>0.0682)&(x_tess<0.151)]
        #y_tess_lv2 =0.02+y_tess[(x_tess>0.0682)&(x_tess<0.151)]-sampled_flare1[(x_tess>0.0682)&(x_tess<0.151)]
        #sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        
    if i==25: #2-component flare! 
        popt_tess_s1 = [0.003, 0.26, 18.8]
        popt_tess_s2 = [0.00639, 0.03, 10.3]
        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=5000)

        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)
        
        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
        
    if i==26:
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=[0.0015, 0.42, 39.0], method="dogbox", maxfev=5000)
        popt_tess_s2 = [0.0328580702, 0.0159421932, 2.15520069]
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=5000)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
        
        #sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)
        
        #popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=5000)
    if i==32:
        y_tess+=0.0037

        #popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=guess, method="dogbox", maxfev=5000)
        
        #popt_tess_s1 = [0.00789164, 0.22163762, 7.68]
        #popt_tess_s1 = [0.00839963, 0.21, 7.22963689]

        #popt_tess_s1 = [0.0045, 0.31, 13.6]
        popt_tess_s2 = [0.006, 0.015, 14.2]
        popt_tess_s3 = [0.00789164, 0.015, 14.3]

        #x_tess_lv1 = x_tess[(x_tess>0.0385)&(x_tess<0.1106)]
        #y_tess_lv1 = y_tess[(x_tess>0.0385)&(x_tess<0.1106)]
        #y_tess_lv1 = y_tess_lv1[(x_tess_lv1<0.07965)|(x_tess_lv1>0.09489)]
        #x_tess_lv1 = x_tess_lv1[(x_tess_lv1<0.07965)|(x_tess_lv1>0.09489)]
        
        #popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess_lv1, y_tess_lv1, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        #sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)

        popt_tess_both = popt_tess_s2 + popt_tess_s3
        #popt_tess_all = popt_tess_s1 + popt_tess_s2 + popt_tess_s3
        
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        sampled_flare3 = davenport_flare_model(x_tess, *popt_tess_s3)

        x_tess_lv2 = x_tess[(x_tess>0.0798) & (x_tess<0.1367)]
        y_tess_lv2 = y_tess[(x_tess>0.0798) & (x_tess<0.1367)] - sampled_flare1[(x_tess>0.0798) & (x_tess<0.1367)]
        popt, pcov = curve_fit(linear, x_tess_lv2, y_tess_lv2)
        y_tess_lv2 = y_tess_lv2 - linear(x_tess_lv2, *popt) + 0.003

        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess_lv2, y_tess_lv2, p0=popt_tess_both, method="dogbox", maxfev=5000)

        #sampled_flare = two_peak_dav_flare_model(x_tess_lv2, *popt_tess_both)

        popt_tess_s2 =popt_tess_both[:3]
        popt_tess_s3 = popt_tess_both[3:]
    if i==43:
        y_tess+=0.005
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=[0.00675, 0.197, 8.753], method="dogbox", maxfev=5000)
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)

        y_tess_lv2 = y_tess[x_tess>0.0975] - sampled_flare1[x_tess>0.0975]
        x_tess_lv2 = x_tess[x_tess>0.0975]

        #popt_tess_s2, pcov = curve_fit(davenport_flare_model, x_tess_lv2, y_tess_lv2, p0=popt_tess_s2, method="dogbox", maxfev=5000)
        sampled_flare2 = davenport_flare_model(x_tess_lv2, *popt_tess_s2)
        sampled_flare3 = davenport_flare_model(x_tess_lv2, *popt_tess_s3)

        popt_tess_s1 = [0.00675, 0.197, 8.753]
        popt_tess_s2 = [0.00675, 0.02, 15.3]
        popt_tess_s3 = [0.011, 0.019, 11.9]
        popt_tess_all = popt_tess_s1 + popt_tess_s2 + popt_tess_s3
        
        popt_tess_all, pcov = curve_fit(three_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_all, method="dogbox", maxfev=5000)

        sampled_flare = three_peak_dav_flare_model(x_tess, *popt_tess_all)

        popt_tess_s1 = popt_tess_all[:3]
        popt_tess_s2 = popt_tess_all[3:6]
        popt_tess_s3 = popt_tess_all[6:]
        
    if i==47:
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=guess, method="dogbox", maxfev=5000)

        popt_tess_s1 = [0.005, 0.14, 11.2]
        popt_tess_s2 = [0.009, 0.15, 6.8]

        popt_tess_both = popt_tess_s1 + popt_tess_s2
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=5000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)
        
        #sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        #sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==50:
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=guess, method="dogbox", maxfev=5000)

        popt_tess_s1 = [0.004, 0.17, 14.2]
        popt_tess_s2 = [0.007, 0.05, 10.5]
        popt_tess_s3 = [0.007, 0.04, 12.5]
        #popt_tess_s4 = [0.01, 0.07, 11.9]
        
        popt_tess_all = popt_tess_s1 + popt_tess_s2 + popt_tess_s3
        popt_tess_all, pcov = curve_fit(three_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_all, method="dogbox", maxfev=5000)
        sampled_flare = three_peak_dav_flare_model(x_tess, *popt_tess_all)
        
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        sampled_flare3 = davenport_flare_model(x_tess, *popt_tess_s3)
        #sampled_flare4 = davenport_flare_model(x_tess, *popt_tess_s4)
        #popt_tess_s1 = popt_tess_both[:3]
        #popt_tess_s2 = popt_tess_both[3:]
        
        popt_tess_s1 = popt_tess_all[:3]
        popt_tess_s2 = popt_tess_all[3:6]
        popt_tess_s3 = popt_tess_all[6:]
    if i==70:
        y_tess+=0.009

        guess=[0.00229722244, 0.182745872, 24.3002867]
        popt_tess_s2 = [0.002, 0.03, 36.3002867]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=[0.00229722244, 0.182745872, 24.3002867], method="dogbox", maxfev=5000)

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=5000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==76:
        y_tess+=0.001

        popt_tess_s1 = [0.005, 0.12, 11.4]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==101:
        pass #already good
    if i==102:
        y_tess+=0.008

        popt_tess_s1 = [0.00442403451, 0.108719041, 12.892075]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==107:
        y_tess+=0.008
        popt_tess_s1 = [0.00431918650, 0.104348780, 13.2017324]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==110:
        y_tess+=0.008
        
        popt_tess_s1 = [0.005, 0.11, 9.1]
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
        
    if i==120:
        y_tess-=0.003

        popt_tess_s1=[0.007, 0.09, 8.45]
        popt_tess_s2 = [0.007, 0.05, 9.3]

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)
        #sampled_flare =sampled_flare1 + sampled_flare2

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==134:
        y_tess-=0.0075

        popt_tess_s1 = [0.001186, 0.148995, 47.326]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==157:
        y_tess+=0.002

        popt_tess_s1 = [0.00143852084, 0.112102761, 38.8070939]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)

    if i==161:
        popt_tess_s1 = [0.002191, 0.1165006, 25.59755]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==167:
        y_tess+=0.001
        
        popt_tess_s1 = [0.00349, 0.0958, 16.0803]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==168:
        y_tess+=0.003
        
        popt_tess_s1 = [0.02058028,  0.0648666, 3.3742464]
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==169:
        y_tess+=0.004
        
        popt_tess_s1 = [0.003066, 0.07493, 18.63976]
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==175:
        y_tess-=0.006
        
        popt_tess_s1 = [0.0024316817, 0.070339, 22.9532]
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==178:
        y_tess+=0.003
        
        popt_tess_s1 = [0.02464795, 0.0695005, 2.70170385]
        popt_tess_s2 = [0.01, 0.011, 2.2]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)
        #sampled_flare =sampled_flare1 + sampled_flare2

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==233:
        y_tess += 0.002

        popt_tess_s1 = [0.0319, 0.0367, 6.5384]
        popt_tess_s2 = [0.0319, 0.012, 9.5]
        
        #popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)

        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)
        #sampled_flare =sampled_flare1 + sampled_flare2

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==290:

        popt_tess_s1 = [0.00205, 0.04515, 27.59]
        popt_tess_s2 = [0.0176, 0.0080413, 3.7374]
        popt_tess_s3 = [0.003, 0.003, 38.7]
        
        #popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        sampled_flare3 = davenport_flare_model(x_tess, *popt_tess_s3)

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)

        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)
        #sampled_flare =sampled_flare1 + sampled_flare2
        
        y_tess_lv2 = y_tess[x_tess>0.1] - sampled_flare[x_tess>0.1] + 0.0004
        x_tess_lv2 = x_tess[x_tess>0.1]

        popt_tess_s3, pcov = curve_fit(davenport_flare_model, x_tess_lv2, y_tess_lv2, p0=popt_tess_s3, method="dogbox", maxfev=5000)

        sampled_flare3 = davenport_flare_model(x_tess, *popt_tess_s3)
        sampled_flare = sampled_flare1 + sampled_flare2 + sampled_flare3
        
        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]

    if i==346:
        popt_tess_s1 = [0.0035, 0.032, 16.3]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==377:
        popt_tess_s1 = [0.0024687, 0.0442, 22.677496]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==379:
        y_tess+=0.003

        popt_tess_s1 = [0.00239, 0.039525, 23.63298]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==445:
        popt_tess_s1 = [0.005126, 0.0263, 11.192]

        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==482:
        y_tess+=0.005
        popt_tess_s1 = [0.00343, 0.0369, 16.35998]
        popt_tess_s2 = [0.008, 0.002, 9.1]
        popt_tess_s3 = [0.008, 0.001, 9.9]

        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        sampled_flare3 = davenport_flare_model(x_tess, *popt_tess_s3)
        
        popt_tess_all = list(popt_tess_s1) + list(popt_tess_s2) + list(popt_tess_s3)
        popt_tess_all, pcov = curve_fit(three_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_all, method="dogbox", maxfev=10000)
        sampled_flare = three_peak_dav_flare_model(x_tess, *popt_tess_all)
        #sampled_flare = sampled_flare1 + sampled_flare2 + sampled_flare3

        popt_tess_s1 = popt_tess_all[:3]
        popt_tess_s2 = popt_tess_all[3:6]
        popt_tess_s3 = popt_tess_all[6:]
    if i==500:
        y_tess+=0.002
        
        popt_tess_s1 = [0.00449, 0.031179, 12.607]

        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==553:
        y_tess+=0.001
        
        popt_tess_s1 = [0.00389, 0.02138, 13.97]

        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==606:
        popt_tess_s1 = [0.0026, 0.018, 21.7288]
        popt_tess_s2 = [0.008, 0.004, 3.1]

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        #sampled_flare = sampled_flare1 + sampled_flare2
        
        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==608:
        y_tess+=0.0065

        popt_tess_s1 = [0.0055, 0.024036, 9.906]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_tess, *popt_tess_s1)
    if i==661:
        y_tess+=0.0004

        popt_tess_s1 = [0.00315, 0.01044, 17.68]
        popt_tess_s2 = [0.0021, 0.005, 39.5]
        
        popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==674:
        y_tess+=0.001

        popt_tess_s1 = [0.0046, 0.00917, 11.96]
        popt_tess_s2 = [0.0046, 0.00917, 3.1]
        popt_tess_s3 = [0.006, 0.006, 6.1]
        #popt_tess_s4 = [0.0035, 0.006, 18.5]

        popt_tess_all = list(popt_tess_s1) + list(popt_tess_s2) + list(popt_tess_s3)
        popt_tess_all, pcov = curve_fit(three_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_all, method="dogbox", maxfev=10000)
        sampled_flare = three_peak_dav_flare_model(x_tess, *popt_tess_all)

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        sampled_flare3 = davenport_flare_model(x_tess, *popt_tess_s3)

        popt_tess_s1 = popt_tess_all[:3]
        popt_tess_s2 = popt_tess_all[3:6]
        popt_tess_s3 = popt_tess_all[6:]
    if i==688:
        
        popt_tess_s1 = [0.0025, 0.00998, 22.361]
        popt_tess_s2 = [0.0056, 0.00569, 10.648]

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
        
    if i==692:
        y_tess+=0.001

        popt_tess_s1 = [0.00599, 0.009, 9.564]
        popt_tess_s2 = [0.00599, 0.005, 13.1]

        #popt_tess_s1, pcov = curve_fit(davenport_flare_model, x_tess, y_tess, p0=popt_tess_s1, method="dogbox", maxfev=5000)

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        #sampled_flare = sampled_flare1 + sampled_flare2

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==719:
        popt_tess_s1 = [0.002412, 0.19, 23.747]
        popt_tess_s2 = [0.014107, 0.029, 4.697]

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        #sampled_flare = sampled_flare1 + sampled_flare2

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==721:
        popt_tess_s1 = [0.0016665, 0.204520515, 33.5056041]
        popt_tess_s2 = [0.0039262, 0.066710746, 16.2837634]

        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)

        #rsmpl_x_tess = np.linspace(np.min(x_tess), np.max(x_tess), num=1000)
        
        #sampled_flare = sampled_flare1 + sampled_flare2

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==735:
        #y_tess+=0.001
        
        popt_tess_s1 = [0.00141, 0.1395, 39.5078]
        popt_tess_s2 = [0.00329, 0.03369, 18.3782]
        
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        #sampled_flare = sampled_flare1 + sampled_flare2

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
    if i==769:
        
        popt_tess_s1 = [0.00579, 0.0249, 10.204]
        popt_tess_s2 = [0.000583, 0.03001, 107.689]
        
        sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
        sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
        #sampled_flare = sampled_flare1 + sampled_flare2

        popt_tess_both = list(popt_tess_s1) + list(popt_tess_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_tess, y_tess, p0=popt_tess_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_tess, *popt_tess_both)

        popt_tess_s1 = popt_tess_both[:3]
        popt_tess_s2 = popt_tess_both[3:]
        
    
    #print "_s1",popt_tess_s1
    #print "_s2",popt_tess_s2
    #exit()
    plt.title(i)
    #plt.plot(x_tess_lv2, y_tess_lv2, marker="o",ls="none",color="black")
    
    plt.plot(x_tess, y_tess, marker="o",ls="none",color="black")
    plt.plot(x_tess, sampled_flare, ls="-",color="cornflowerblue")
    #plt.plot(x_tess, sampled_flare2, ls="-",color="green")

    #plt.plot(x_tess, sampled_flare3, ls="-",color="magenta")
    #plt.plot(x_tess, sampled_flare4, ls="-",color="darkorange")

    #plt.plot(x_tess_lv2, y_tess_lv2,marker="o",ls="none",color="black")
    #plt.plot(x_tess_lv2, sampled_flare2, ls="-",color="green")
    #plt.plot(x_tess_lv2, sampled_flare3, ls="-",color="green")
    #plt.plot(x_tess, sampled_flare3, ls="-",color="magenta")
    #plt.show()

    #plt.plot(x_tess_lv3, y_tess_lv3,marker="o",ls="none",color="black")
    #plt.plot(x_tess_lv3, sampled_flare3, ls="-",color="green")
    #plt.show()
    #exit()
    plt.close("all")
    
    return (popt_tess_s1, popt_tess_s2, popt_tess_s3, x_tess, y_tess)

def fit_evr_flare_coeffs(i,x_evry, y_evry):
    
    guess_width=0.01
    guess_ampl=np.max(y_tess)
    guess_peak_time=100.0*x_tess[list(y_tess).index(np.max(y_tess))]
    guess = [guess_width, guess_ampl, guess_peak_time]
    popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=guess, method="dogbox", maxfev=5000)
    popt_evry_s2 = [0.0, 0.0, 0.0]
    popt_evry_s3 = [0.0, 0.0, 0.0]

    #popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
    #print popt_evry_both
    #exit()
    popt_evry_all = popt_evry_s1 + popt_evry_s2 + popt_evry_s3

    sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    
    if i==0:
        ###################################

        popt_evry_s1 = [0.003125, 8.683676, 18.2728]
        popt_evry_s2 = [0.007228, 0.396217, 11.5625]

        #popt_evry_s1 = [0.004, 7.1, 14.3]
        #popt_evry_s2 = [0.007228, 0.396217, 11.5625]
        
        #popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)

        
        #sampled_flare1 = davenport_flare_model(x_evry, *popt_evry_s1)
        #sampled_flare2 = davenport_flare_model(x_evry, *popt_evry_s2)
        #sampled_flare = sampled_flare1 + sampled_flare2

        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        popt_tess_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=5000)
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]

        """
        x_evry_lv2 =x_evry[20:]
        y_evry_lv2 =y_evry[20:]-sampled_flare1[20:]
        popt_evry_s2, pcov = curve_fit(davenport_flare_model, x_evry_lv2, y_evry_lv2, p0=[0.004, 0.06, 20.56], method="dogbox", maxfev=5000)
        """
        
        popt_evry_s3 = [0.0, 0.0, 0.0]
        #####################################
        
    if i==4:
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=[0.0023738, 8.809, 23.674], method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
        
        #####################################
    if i==10:
        popt_evry_s1 = [0.0037849, 2.512, 14.9501]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)

        #####################################
    if i==16:
        #popt_evry_s1 = [0.003, 2.2, 18.7]
        popt_evry_s1 = [0.00274, 2.024, 20.73]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==17:
        popt_evry_s1 = [0.004, 1.8, 12.25]
        popt_evry_s2 = [0.006, 1.0, 9.25]
        popt_evry_s3 = [0.004, 0.1, 20.25]

        popt_evry_all = popt_evry_s1 + popt_evry_s2 + popt_evry_s3
    
        popt_evry_all, pcov = curve_fit(three_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_all, method="dogbox", maxfev=5000)

        sampled_flare = three_peak_dav_flare_model(x_evry, *popt_evry_all)

        popt_evry_s1 = popt_evry_all[:3]
        popt_evry_s2 = popt_evry_all[3:6]
        popt_evry_s3 = popt_evry_all[6:]
    if i==19:
        
        popt_evry_s1 = [0.0103542, 2.1842492, 5.9017898]
        popt_evry_s2 = [0.00561, 0.635549, 13.003]
        
        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=5000)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
        
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)
    if i==25:
        popt_evry_s1 = [0.00111, 4.32, 50.53]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
        
    if i==26:

        popt_evry_s1 = [0.0010179, 4.814, 57.005]
        popt_evry_s2 = [0.0, 0.0, 0.0] #[0.0013318, 0.4363, 37.7087]

        #popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        
        #sampled_flare1 = davenport_flare_model(x_evry, *popt_evry_s1)

        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=5000)

        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    if i==32:
        popt_evry_s1 = [0.008009, 1.25, 7.63896]

        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==43:
        popt_evry_s1 = [0.005198, 0.948, 11.28]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
        
    if i==47:

        popt_evry_s1 = [0.0028, 0.445, 20.22]
        popt_evry_s2 = [0.0014, 0.565, 42.63]

        popt_evry_both = popt_evry_s1 + popt_evry_s2
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=5000)
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)
        
        #sampled_flare1 = davenport_flare_model(x_evry, *popt_evry_s1)
        #sampled_flare2 = davenport_flare_model(x_evry, *popt_evry_s2)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    if i==50:
        y_evry-=0.099
        
        popt_evry_s1 = [0.00296018, 2.12933, 19.3987]
        popt_evry_s2 = [0.00501267, 0.32247, 14.6907]
        
        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=5000)
        
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)
        
        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    if i==70:
        popt_evry_s1 = [0.0013, 2.375, 44.157]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==76:
        popt_evry_s1 = [0.0028, 1.507, 19.9606]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==101:
        popt_evry_s1 = [0.000445, 8.986, 124.3]

        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.0469)&(x_evry<0.06625)], y_evry[(x_evry>0.0469)&(x_evry<0.06625)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==102:
        popt_evry_s1 = [0.001877, 1.808, 30.123]

        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==110:
        popt_evry_s1 = [0.0013847, 1.39, 32.73]

        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==120:
        #popt_evry_s1 = [0.004, 0.7, 25.7]

        #popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        
        #sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)

        popt_evry_s1= [0.00206, 0.5393, 27.82]
        popt_evry_s2 = [0.0101, 0.48453, 6.14]

        sampled_flare1 = davenport_flare_model(x_evry, *popt_evry_s1)
        sampled_flare2 = davenport_flare_model(x_evry, *popt_evry_s2)

        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)
        #sampled_flare =sampled_flare1 + sampled_flare2

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    if i==134:
        popt_evry_s1 = [0.000467, 4.74, 119.3]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==157:
        popt_evry_s1 = [0.0077, 1.199, 7.446]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==161:
        popt_evry_s1 = [0.0011155, 0.51661, 48.91089]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==167:
        popt_evry_s1 =  [0.0035, 0.615, 15.9]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==168:
        popt_evry_s1 =  [0.025, 0.2815, 2.728]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==169:
        popt_evry_s1 =  [0.001397, 1.065, 40.62]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.04)&(x_evry<0.0867)], y_evry[(x_evry>0.04)&(x_evry<0.0867)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==175:
        popt_evry_s1 = [0.001037, 0.94, 53.7]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==178:
        popt_evry_s1 = [0.0162, 0.3412, 3.91]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.026)&(x_evry<0.1116)], y_evry[(x_evry>0.026)&(x_evry<0.1116)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==233:
        popt_evry_s1 = [0.0288, 0.114, 7.07]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.14)], y_evry[(x_evry>0.14)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==290:
        popt_evry_s1 = [0.0024, 0.4058, 23.17]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry<0.0888)], y_evry[(x_evry<0.0888)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==300:
        
        popt_evry_s1 = [0.0017, 1.859, 33.8]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.055)&(x_evry<0.064)], y_evry[(x_evry>0.055)&(x_evry<0.064)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==346:
        popt_evry_s1 = [0.00065, 0.735, 87.116]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==377:
        popt_evry_s1 = [0.0027, 0.32, 20.64]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==379:
        popt_evry_s1 = [0.00239, 0.395, 23.6]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==445:
        popt_evry_s1 = [0.002, 0.2, 28.5]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==482:
        popt_evry_s1 = [0.00212, 0.464, 26.31]
        popt_evry_s2 = [0.01008, 0.071, 7.38]

        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=5000)
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    if i==500: #2 peaks?
        popt_evry_s1 = [0.001226, 0.583, 45.566]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==553:
        popt_evry_s1 = [0.00239, 0.2037, 22.438]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==606:
        popt_evry_s1 = [0.001385, 0.1636, 39.84]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==608:
        popt_evry_s1 = [0.0025, 0.2759, 21.94]
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.04949)&(x_evry<0.0657)], y_evry[(x_evry>0.04949)&(x_evry<0.0657)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
        
    if i==661:
        popt_evry_s1 = [0.001785, 0.12, 31.42]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.0504)&(x_evry<0.0664)], y_evry[(x_evry>0.0504)&(x_evry<0.0664)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==674:
        popt_evry_s1 = [0.0009224, 0.149, 59.96]
        popt_evry_s2 = [0.0005856, 0.317, 21.57]

        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    if i==688:
        popt_evry_s1 = [0.003266, 0.090, 17.093]
        popt_evry_s2 = [0.000423, 0.316, 137.96]

        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=10000)
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    if i==692:
        popt_evry_s1 = [0.00187, 0.11119, 30.05]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.0504)&(x_evry<0.0664)], y_evry[(x_evry>0.0504)&(x_evry<0.0664)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==719:
        popt_evry_s1 = [0.0115, 0.41448198, 5.01409503]
        popt_evry_s2 = [0.00029, 1.933, 199.6]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare1 = davenport_flare_model(x_evry, *popt_evry_s1)
        sampled_flare2 = davenport_flare_model(x_evry, *popt_evry_s2)
        sampled_flare = sampled_flare1 + sampled_flare2

        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry, y_evry, p0=popt_evry_both, method="dogbox", maxfev=10000)
        #x_evry_up = np.linspace(np.min(x_evry),np.max(x_evry),num=1000)
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    if i==721:
        popt_evry_s1 = [0.0012, 5.713, 46.24]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.0504)&(x_evry<0.0664)], y_evry[(x_evry>0.0504)&(x_evry<0.0664)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==735:
        popt_evry_s1 = [0.00127, 2.028, 44.8]
        
        popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry[(x_evry>0.0529)&(x_evry<0.0618)], y_evry[(x_evry>0.0529)&(x_evry<0.0618)], p0=popt_evry_s1, method="dogbox", maxfev=5000)
        sampled_flare = davenport_flare_model(x_evry, *popt_evry_s1)
    if i==769:

        popt_evry_s1 =  [0.00588, 0.488, 10.00308]
        popt_evry_s2 = [0.000207, 1.58, 295.124590]
        
        #popt_evry_s1, pcov = curve_fit(davenport_flare_model, x_evry, y_evry, p0=popt_evry_s1, method="dogbox", maxfev=5000)
        #sampled_flare1 = davenport_flare_model(x_evry, *popt_evry_s1)
        #sampled_flare2 = davenport_flare_model(x_evry, *popt_evry_s2)
        #sampled_flare = sampled_flare1 + sampled_flare2

        popt_evry_both = list(popt_evry_s1) + list(popt_evry_s2)
        popt_evry_both, pcov = curve_fit(two_peak_dav_flare_model, x_evry[(x_evry>0.0504)&(x_evry<0.09)], y_evry[(x_evry>0.0504)&(x_evry<0.09)], p0=popt_evry_both, method="dogbox", maxfev=10000)
        #x_evry_up = np.linspace(np.min(x_evry),np.max(x_evry),num=1000)
        sampled_flare = two_peak_dav_flare_model(x_evry, *popt_evry_both)

        popt_evry_s1 = popt_evry_both[:3]
        popt_evry_s2 = popt_evry_both[3:]
    
    #print "_s1",popt_evry_s1
    #print "_s2",popt_evry_s2
    #print "_s3",popt_evry_s3
    #exit()
    plt.title(i)
    #plt.plot(x_evry_lv2, y_evry_lv2, marker="o",ls="none",color="black")
    
    plt.plot(x_evry, y_evry, marker="o",ls="none",color="black")
    plt.plot(x_evry, sampled_flare, ls="-",color="cornflowerblue")
    #plt.plot(x_evry, sampled_flare2, ls="-",color="green")

    #plt.plot(x_evry, sampled_flare3, ls="-",color="magenta")
    #plt.plot(x_evry, sampled_flare4, ls="-",color="darkorange")

    #plt.plot(x_evry_lv2, y_evry_lv2,marker="o",ls="none",color="black")
    #plt.plot(x_evry_lv2, sampled_flare2, ls="-",color="green")
    #plt.plot(x_evry_lv2, sampled_flare3, ls="-",color="green")
    #plt.plot(x_evry, sampled_flare3, ls="-",color="magenta")
    #plt.show()

    #plt.plot(x_evry_lv3, y_evry_lv3,marker="o",ls="none",color="black")
    #plt.plot(x_evry_lv3, sampled_flare3, ls="-",color="green")
    #plt.show()
    #exit()
    plt.clf()
    plt.close("all")
    
    return (popt_evry_s1, popt_evry_s2, popt_evry_s3, x_evry, y_evry)



def get_Q_0_errors(g_mags_val, e_mags_val, TESS_mags_val, e_TESS_mags_val, dist_pc, e_dist_pc):

    n_trials = 1000

    dist1_sampler = np.random.normal(dist_pc, e_dist_pc, n_trials)
    dist2_sampler = np.random.normal(dist_pc, e_dist_pc, n_trials)
    gmag_sampler = np.random.normal(g_mags_val, e_mags_val, n_trials)
    Tmag_sampler = np.random.normal(TESS_mags_val, e_TESS_mags_val, n_trials)

    evry_quiesc=[]
    tess_quiesc=[]
    for n in np.arange(n_trials):
        evr_Q_0 = get_Q0_luminosity(dist1_sampler[n], gmag_sampler[n], "g-mag") #g-mag
        tess_Q_0 = get_Q0_luminosity(dist2_sampler[n], Tmag_sampler[n], "T-mag")

        evry_quiesc.append(evr_Q_0)
        tess_quiesc.append(tess_Q_0)
    evry_quiesc= np.array(evry_quiesc)
    tess_quiesc= np.array(tess_quiesc)

    evry_std_Q_0 = np.std(evry_quiesc)
    tess_std_Q_0 = np.std(tess_quiesc)

    evr_Q_0 = get_Q0_luminosity(dist_pc, g_mags_val, "g-mag") #g-mag

    tess_Q_0 = get_Q0_luminosity(dist_pc, TESS_mags_val, "T-mag") #T-mag

    """
    plt.axvline(evr_Q_0,color="black")
    plt.axvline(evr_Q_0 + evry_std_Q_0,color="grey")
    plt.axvline(evr_Q_0 - evry_std_Q_0,color="grey")
    plt.hist(evry_quiesc)
    plt.show()

    plt.axvline(tess_Q_0,color="black")
    plt.axvline(tess_Q_0 + tess_std_Q_0,color="grey")
    plt.axvline(tess_Q_0 - tess_std_Q_0,color="grey")
    plt.hist(tess_quiesc,color="firebrick")
    plt.show()
    #exit()
    """
    
    return (evr_Q_0, evry_std_Q_0, tess_Q_0, tess_std_Q_0)

def offsets_y_evry():
    off_i=[]
    off_topFF=[]
    off_botFF=[]
    with open("y_offsets_simul.csv","r") as INFILE:
        next(INFILE)
        for lines in INFILE:
            off_i.append(int(lines.split(",")[0]))
            off_topFF.append(float(lines.split(",")[1]))
            off_botFF.append(float(lines.split(",")[2]))
    off_i=np.array(off_i)
    off_topFF=np.array(off_topFF)
    off_botFF=np.array(off_botFF)

    return (off_i, off_topFF, off_botFF)

def get_fracflux_scaling_err(mx_evry, my_evry, my_evry_err, ESTART, ESTOP):

    my_evry_init = copy.deepcopy(my_evry) #preserve values for resets
    mx_evry_init = copy.deepcopy(mx_evry) #preserve values for resets
    my_evry_err_init = copy.deepcopy(my_evry_err) #preserve values for resets
    
    mx_evry_sub = mx_evry[(mx_evry<ESTART) | (mx_evry>ESTOP)]
    my_evry_sub = my_evry[(mx_evry<ESTART) | (mx_evry>ESTOP)]
    my_evry_sub_err = my_evry_err[(mx_evry<ESTART) | (mx_evry>ESTOP)]

    sub_flag = np.ones(len(my_evry_sub))
    sub_index = np.arange(len(my_evry_sub))
    n_trials=1000
    mag_offset_distrib=[]
    for b in range(n_trials):
        my_evry = copy.deepcopy(my_evry_init)
        mx_evry = copy.deepcopy(mx_evry_init)
        
        sub_flag = np.ones(len(my_evry_sub))
        mc_my_evry_sub = copy.deepcopy(my_evry_sub)
        #mc_my_evry_sub = np.random.normal(my_evry_sub, my_evry_sub_err)
        width = int(np.floor(len(sub_index)/5.0))
        
        start_pos =  int(np.floor(len(sub_flag)*np.random.random()))

        sub_flag[start_pos:(start_pos+width)]=-1.0
        
        MED = np.nanmedian(mc_my_evry_sub[sub_flag>0.0])
        
        my_evry_old = copy.deepcopy(my_evry)
        my_evry = my_evry - MED

        """
        plt.axvline(ESTART,color="grey")
        plt.axvline(ESTOP,color="grey")
        plt.plot(mx_evry, my_evry_old, marker="o",ls="none",color="green")
        plt.plot(mx_evry_sub,my_evry_sub, marker="o",ls="none",color="red")
        plt.axhline(MED)
        plt.gca().invert_yaxis()
        plt.show()
        exit()
        """
        mc_evr_fracflux = (-1.0) + 2.512**(-my_evry)
        mc_evr_fracflux_sub = mc_evr_fracflux[(mx_evry<ESTART) | (mx_evry>ESTOP)]
        mc_evr_fracflux = mc_evr_fracflux - np.nanmedian(mc_evr_fracflux_sub[sub_flag>0.0])
        
        mag_offset_distrib.append(np.nanmax(mc_evr_fracflux))
            
    mag_offset_distrib = np.array(mag_offset_distrib)

    #plt.hist(mag_offset_distrib,color="green")
    #plt.show()
    
    mag_offset_err = np.std(mag_offset_distrib)
    mag_offset_med = np.median(mag_offset_distrib)

    #if (offset_med+0.5*mag_offset_err)>mag_offset_err:
    #    mag_offset_err=offset_med+mag_offset_err

    return mag_offset_err

def compute_ratio(i, x_tess, y_tess, y_tess_err, x_evry, y_evry, y_evry_regrid, y_evry_err_regrid, err_tess_Q_0, tess_Q_0, err_evr_Q_0, evr_Q_0, ESTART, ESTOP, FWHM_start_val, FWHM_stop_val, MC_BOOL):
    
    popt_tess_s1, popt_tess_s2, popt_tess_s3, x_tess, y_tess = fit_flare_coeffs(i, x_tess, y_tess)

    # number of peaks in flare model (1,2, or 3 peaks allowed):
    n1 = np.sum(np.unique(popt_tess_s1))
    n2 = np.sum(np.unique(popt_tess_s2))
    n3 = np.sum(np.unique(popt_tess_s3))
    if n1>0.0:
        n1=1
    if n2>0.0:
        n2=1
    if n3>0.0:
        n3=1
    n_flares = int(n1 + n2 + n3)
    #print n1, n2, n3, "tot", n_flares
    #exit()
    #print popt_tess_s2
    #exit()

    popt_tess_all = list(popt_tess_s1) + list(popt_tess_s2) + list(popt_tess_s3)
    
    sampled_flare1 = davenport_flare_model(x_tess, *popt_tess_s1)
    sampled_flare2 = davenport_flare_model(x_tess, *popt_tess_s2)
    sampled_flare3 = davenport_flare_model(x_tess, *popt_tess_s3)
    sampled_flare = sampled_flare1 + sampled_flare2 + sampled_flare3

    popt_evry_s1, popt_evry_s2, popt_evry_s3, x_evry, y_evry = fit_evr_flare_coeffs(i, x_evry, y_evry)
    
    def rescale_three_peak_flare_fit(epochs, yscale):
        flare_y = three_peak_dav_flare_model(epochs, *popt_tess_all)
        flare_y*=yscale
    
        return flare_y

    guess_scale = 1.1*np.nanmax(y_evry)/np.nanmax(y_tess)

    popt_evr_scale, pcov = curve_fit(rescale_three_peak_flare_fit, x_evry, y_evry, p0=guess_scale, method="dogbox", maxfev=5000)
    #sampled_evr_flare = rescale_three_peak_flare_fit(x_evry, *popt_evr_scale)

    upsampl_x_evry = np.linspace(np.min(x_evry),np.max(x_evry),num=1000)
    upsampl_evr_flare = rescale_three_peak_flare_fit(upsampl_x_evry, *popt_evr_scale)

    # compute flare blackbody evolution:
    
    sampled_flare1 = davenport_flare_model(x_tess, *popt_evry_s1)
    sampled_flare2 = davenport_flare_model(x_tess, *popt_evry_s2)
    sampled_flare3 = davenport_flare_model(x_tess, *popt_evry_s3)
    sampled_evr_flare = sampled_flare1 + sampled_flare2 + sampled_flare3
    
    #sampled_evr_flare = rescale_three_peak_flare_fit(x_tess, *popt_evr_scale)
    sampled_tess_flare = sampled_flare
    
    evry_ED_arr = 120.0*sampled_evr_flare
    tess_ED_arr = 120.0*sampled_tess_flare

    #evr_Q_0, err_evr_Q_0, tess_Q_0, err_tess_Q_0
    sampl_evr_Q_0 = np.random.normal(evr_Q_0, err_evr_Q_0)
    sampl_tess_Q_0 = np.random.normal(tess_Q_0, err_tess_Q_0)

    if MC_BOOL==True:
        evry_erg_sampl = sampl_evr_Q_0*np.array(evry_ED_arr)
        tess_erg_sampl = sampl_tess_Q_0*np.array(tess_ED_arr)
    else:
        evry_erg_sampl = evr_Q_0*np.array(evry_ED_arr)
        tess_erg_sampl = tess_Q_0*np.array(tess_ED_arr)
    
    evry_ED_data =120.0*y_evry_regrid
    tess_ED_data = 120.0*y_tess
    err_evry_ED_data =120.0*y_evry_err_regrid
    err_tess_ED_data = 120.0*y_tess_err

    evry_erg_data = evr_Q_0*evry_ED_data
    tess_erg_data = tess_Q_0*tess_ED_data

    EVRY_CUT = np.nanmedian(evry_ED_data[(x_tess<ESTART)|(x_tess>ESTOP)]) + 0.5*np.nanstd(evry_ED_data[(x_tess<ESTART)|(x_tess>ESTOP)])
    TESS_CUT = np.nanmedian(tess_ED_data[(x_tess<ESTART)|(x_tess>ESTOP)]) + 0.5*np.nanstd(tess_ED_data[(x_tess<ESTART)|(x_tess>ESTOP)])

    ratio_erg_arr = evry_erg_data/tess_erg_data
    ratio_erg_arr[ratio_erg_arr<0.00001]=0.00001
    ratio_erg_arr[ratio_erg_arr>9.0]=10.0
    ratio_erg_arr[(evry_ED_data<EVRY_CUT) | (tess_ED_data<TESS_CUT)] = 0.00001

    ratio_erg_arr2 = evry_erg_sampl/tess_erg_sampl
    ratio_erg_arr2[ratio_erg_arr2<0.00001]=0.00001
    ratio_erg_arr2[ratio_erg_arr2>9.0]=9.0
    ratio_erg_arr2[(evry_ED_arr<EVRY_CUT) | (tess_ED_arr<TESS_CUT)] = 0.00001

    #total area-under-curve, data
    x_tess_insec = x_tess*24.0*3600.0
    x_evry_insec = x_evry*24.0*3600.0
    x_tess_insec = x_tess_insec - np.min(x_tess_insec)
    x_evry_insec = x_evry_insec - np.min(x_tess_insec)

    x_tess_upsmpl = np.linspace(np.nanmin(x_tess),np.nanmax(x_tess),num=1000)
    x_tess_upsmpl_insec = x_tess_upsmpl*24.0*3600.0
    
    upsampled_evry_flare1 = davenport_flare_model(x_tess_upsmpl, *popt_evry_s1)
    upsampled_evry_flare2 = davenport_flare_model(x_tess_upsmpl, *popt_evry_s2)
    upsampled_evry_flare3 = davenport_flare_model(x_tess_upsmpl, *popt_evry_s3)
    upsampled_evr_flare = upsampled_evry_flare1 + upsampled_evry_flare2 + upsampled_evry_flare3
    
    upsampled_tess_flare1 = davenport_flare_model(x_tess_upsmpl, *popt_tess_s1)
    upsampled_tess_flare2 = davenport_flare_model(x_tess_upsmpl, *popt_tess_s2)
    upsampled_tess_flare3 = davenport_flare_model(x_tess_upsmpl, *popt_tess_s3)
    upsampled_tess_flare = upsampled_tess_flare1 + upsampled_tess_flare2 + upsampled_tess_flare3
    
    #sampled_evr_flare = rescale_three_peak_flare_fit(x_tess, *popt_evr_scale)
    sampled_tess_flare = sampled_flare
    
    ED_EVRY_tot = np.trapz(y_evry[(x_evry>=ESTART) & (x_evry<=ESTOP)], x_evry_insec[(x_evry>=ESTART) & (x_evry<=ESTOP)])
    
    ED_TESS_tot = np.trapz(y_tess[(x_tess>=ESTART) & (x_tess<=ESTOP)], x_tess_insec[(x_tess>=ESTART) & (x_tess<=ESTOP)])
    
    #total area-under-curve, model
    # models do not need event starts or stops, provides a better measurement of all the flare flux across whole emission timescale- remember to write up in paper!!!
    
    ED_EVRY_sampltot = np.trapz(upsampled_evr_flare[(x_tess_upsmpl>=ESTART) & (x_tess_upsmpl<=ESTOP)], x_tess_upsmpl_insec[(x_tess_upsmpl>=ESTART) & (x_tess_upsmpl<=ESTOP)]) # done in TESS timestamps
    
    ED_TESS_sampltot = np.trapz(upsampled_tess_flare[(x_tess_upsmpl>=ESTART) & (x_tess_upsmpl<=ESTOP)], x_tess_upsmpl_insec[(x_tess_upsmpl>=ESTART) & (x_tess_upsmpl<=ESTOP)])
    
 
    #FWHM area-under-curve, data
    ED_EVRY_FWHM = np.trapz(y_evry[(x_evry>=FWHM_start_val) & (x_evry<=FWHM_stop_val)], x_evry_insec[(x_evry>=FWHM_start_val) & (x_evry<=FWHM_stop_val)])
    
    ED_TESS_FWHM = np.trapz(y_tess[(x_tess>=FWHM_start_val) & (x_tess<=FWHM_stop_val)], x_tess_insec[(x_tess>=FWHM_start_val) & (x_tess<=FWHM_stop_val)])
    
    #FWHM area-under-curve, model
    """
    if MC_BOOL==True:
        plt.plot(x_tess_upsmpl_insec[(x_tess_upsmpl>=FWHM_start_val) & (x_tess_upsmpl<=FWHM_stop_val)], upsampled_evr_flare[(x_tess_upsmpl>=FWHM_start_val) & (x_tess_upsmpl<=FWHM_stop_val)], marker="o",ls="none",color="red")
        plt.show()
        exit()
    """
    
    ED_EVRY_FWHMsampl = np.trapz(upsampled_evr_flare[(x_tess_upsmpl>=FWHM_start_val) & (x_tess_upsmpl<=FWHM_stop_val)], x_tess_upsmpl_insec[(x_tess_upsmpl>=FWHM_start_val) & (x_tess_upsmpl<=FWHM_stop_val)]) # done in TESS timestamps
    
    ED_TESS_FWHMsampl = np.trapz(upsampled_tess_flare[(x_tess_upsmpl>=FWHM_start_val) & (x_tess_upsmpl<=FWHM_stop_val)], x_tess_upsmpl_insec[(x_tess_upsmpl>=FWHM_start_val) & (x_tess_upsmpl<=FWHM_stop_val)])


    if MC_BOOL==False:
        tot_ratio = (ED_EVRY_tot*evr_Q_0)/(ED_TESS_tot*tess_Q_0)
        tot_sampl_ratio = (ED_EVRY_sampltot*evr_Q_0)/(ED_TESS_sampltot*tess_Q_0)
        FWHM_ratio = (ED_EVRY_FWHM*evr_Q_0)/(ED_TESS_FWHM*tess_Q_0)
        FWHM_sampl_ratio = (ED_EVRY_FWHMsampl*evr_Q_0)/(ED_TESS_FWHMsampl*tess_Q_0)
        EVRY_erg = np.round(ED_EVRY_tot*evr_Q_0,2)
        TESS_erg = np.round(ED_TESS_tot*tess_Q_0,2)

        Evry_Erg_FWHM = ED_EVRY_FWHMsampl*evr_Q_0
    else:
        tot_ratio = (ED_EVRY_tot*evr_Q_0)/(ED_TESS_tot*tess_Q_0)
        tot_sampl_ratio = (ED_EVRY_sampltot*sampl_evr_Q_0)/(ED_TESS_sampltot*sampl_tess_Q_0)
        FWHM_ratio = (ED_EVRY_FWHM*evr_Q_0)/(ED_TESS_FWHM*tess_Q_0) #fix this?
        FWHM_sampl_ratio = (ED_EVRY_FWHMsampl*sampl_evr_Q_0)/(ED_TESS_FWHMsampl*sampl_tess_Q_0)
        #print "EVRY ED",ED_EVRY_FWHMsampl, "TESS ED",ED_TESS_FWHMsampl,"Q0",sampl_evr_Q_0,sampl_tess_Q_0
        EVRY_erg = ED_EVRY_tot*sampl_evr_Q_0
        TESS_erg = ED_TESS_tot*sampl_tess_Q_0

        Evry_Erg_FWHM = ED_EVRY_FWHMsampl*evr_Q_0
    
    return (ratio_erg_arr, ratio_erg_arr2, n_flares, tot_ratio, tot_sampl_ratio, FWHM_ratio, FWHM_sampl_ratio, EVRY_erg, TESS_erg, Evry_Erg_FWHM)

def convert_SpT_to_mass(input_SpT):

    #input_SpT
    SpT_ind_array = [s.replace(".","").replace("K","0_").replace("M","1_") for s in input_SpT]
    SpT_ind_array = np.array([s.replace("_",".") for s in SpT_ind_array])
    #print "\nSpT",input_SpT
    #print SpT_ind_array
    #exit()
    
    SpT_list = ["K4","K5","K7","M0","M1","M2","M3","M4","M5","M6","M7"]

    mass_list = [0.75, 0.7, 0.63, 0.59, 0.54, 0.42, 0.29, 0.2, 0.15, 0.12, 0.11] #Kraus Hillenbrand 2007

    spt_ind_list = [0.4, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    
    interp_spt = interp1d(spt_ind_list, mass_list, kind="linear")

    stellar_mass = interp_spt(SpT_ind_array)

    #print "mass",stellar_mass
    
    return stellar_mass

def get_SpT():
    SpT_TIC = []
    SpT_values = []
    with open("simulflares_jeff_plus_simbad_SpT.csv","r") as INFILE:
        next(INFILE)
        for lines in INFILE:
            SpT_TIC.append(int(lines.split(",")[0]))
            SpT_values.append(str(lines.split(",")[3].rstrip("\n")))
    SpT_TIC=np.array(SpT_TIC)
    SpT_values=np.array(SpT_values)

    SpT_masses = convert_SpT_to_mass(SpT_values)
    #exit()
    
    return (SpT_TIC, SpT_values, SpT_masses)

def get_Prot():
    index_Prot=[]
    TIC_Prot=[]
    Prot_VAL=[]
    with open("short_simulflares_rot_periods.csv","r") as INFILE:
        next(INFILE)
        for lines in INFILE:
            index_Prot.append(int(lines.split(",")[0]))
            TIC_Prot.append(int(lines.split(",")[1]))
            Prot_VAL.append(float(lines.split(",")[2]))
    index_Prot=np.array(index_Prot)
    TIC_Prot=np.array(TIC_Prot)
    Prot_VAL=np.array(Prot_VAL)

    return (index_Prot, TIC_Prot, Prot_VAL)

def get_rossby_number(masses, periods):

    log_conv_ovrtn_tscale = 1.16 - 1.49*np.log10(masses)-0.54*(np.log10(masses)**2.0) #days, Eq. 11 in Wright et al. (2011), valid 0.09 < M < 1.36 sol. mass

    conv_ovrtn_tscale = 10.0**log_conv_ovrtn_tscale

    rossby_num = np.round(periods/conv_ovrtn_tscale,4)

    return rossby_num

TID=[]
RA=[]
DEC=[]
YEAR=[]
MON=[]
DAY=[]
HR=[]
MIN=[]
SEC=[]
JD=[]
JD_start=[]
JD_stop=[]
GMAG=[]
with open("./run4/dedup_combined_K7_M5_vetted_tess_flares.csv","r") as INFILE:
    for lines in INFILE:
        TID.append(int(lines.split(",")[0]))
        RA.append(float(lines.split(",")[1]))
        DEC.append(float(lines.split(",")[2]))
        YEAR.append(int(lines.split(",")[3]))
        MON.append(int(lines.split(",")[4]))
        DAY.append(int(lines.split(",")[5]))
        HR.append(int(lines.split(",")[6]))
        MIN.append(int(lines.split(",")[7]))
        SEC.append(float(lines.split(",")[8]))
        JD.append(float(lines.split(",")[9]))
        JD_start.append(float(lines.split(",")[10]))
        JD_stop.append(float(lines.split(",")[11]))
        GMAG.append(float(lines.split(",")[12]))
TID = np.array(TID)
RA = np.array(RA)
DEC = np.array(DEC)
YEAR = np.array(YEAR)
MON = np.array(MON)
DAY = np.array(DAY)
HR = np.array(HR)
MIN = np.array(MIN)
SEC = np.array(SEC)
MJD = np.array(JD) - 2400000.5
MJD_start = np.array(JD_start) - 2400000.5
MJD_stop = np.array(JD_stop) - 2400000.5
GMAG = np.array(GMAG)

# read in EVR flare data tables
tid_TI, apassid_TI, ra_TI, dec_TI, M_g_TI, SpT_TI, Q_0_TI, gmags_TI, TESS_mags_TI, distances_TI, peak_flux_TI, contrast_TI, ED_TI, EVR_log_energies_TI, event_start_TI, event_time_TI, event_stop_TI, FWHM_TI, impulsiveness_TI, signif_TI = read_table_one("./run4/EVR_Flares_Sect1-6_Table_I.csv")

tid_TII,apassid_TII,ra_TII,dec_TII,M_g_TII,SpT_TII,N_flares_TII,obs_time_TII,EVR_alpha_TII,EVR_alpha_err_low_TII,EVR_alpha_err_high_TII,EVR_beta_TII,EVR_beta_err_low_TII,EVR_beta_err_high_TII,superflares_TII,superflares_err_low_TII,superflares_err_high_TII,mean_log_E_TII,max_log_E_TII,mean_contrast_TII,max_contrast_TII,P_rot_TII,g_mags_TII,TESS_mags_TII = read_table_two("./run4/postref_EVR_Flare_Stars_Sect1-6_Table_II.csv")

JR_tic_id,JR_SpT,updated_SpT,updated_T_eff,T_eff_EvryFlare_I,updated_masses,spot_increases,spot_temp=get_JR_revised_spt()

atl_TIC, atl_g, atl_dg = get_ATLAS_gmag()

gi_tic, gi_ra, gi_dec = load_J2000_coords()

MatchID, MAST_Tmag, MAST_e_Tmag, MAST_d, MAST_e_d, MAST_RA, MAST_Dec = load_MAST_data()

off_i, off_topFF, off_botFF = offsets_y_evry()

SpT_TIC, SpT_values, SpT_masses = get_SpT()

index_Prot, TIC_Prot, Prot_values = get_Prot()

unique_TIC_Prot = np.unique(TIC_Prot)
unique_masses=[]
unique_SpT=[]
unique_Prot=[]
for ind in range(len(unique_TIC_Prot)):
    unique_masses.append(SpT_masses[SpT_TIC==unique_TIC_Prot[ind]][0])
    unique_SpT.append(SpT_values[SpT_TIC==unique_TIC_Prot[ind]][0])
    unique_Prot.append(Prot_values[TIC_Prot==unique_TIC_Prot[ind]][0])
unique_masses=np.array(unique_masses)
unique_SpT=np.array(unique_SpT)
unique_Prot=np.array(unique_Prot)

unique_rossby_num = get_rossby_number(unique_masses, unique_Prot) #dont forget some targets will have MASS but no PROT!!!!!!!!

# special photometry:
list_of_dirs2 = glob.glob("./special/special_TIC*csv")
S_valid_TID_YMD=[]
for i in range(len(list_of_dirs2)):
    IDstr = list_of_dirs2[i].replace("./special/special_TIC-","").replace("_lc.csv","")
    S_valid_TID_YMD.append(IDstr)
S_valid_TID_YMD=np.array(S_valid_TID_YMD).astype(str)

#efte photometry:
list_of_dirs2 = glob.glob("./run4/TIC*csv")
N_valid_TID_YMD=[]
for i in range(len(list_of_dirs2)):
    IDstr = list_of_dirs2[i].replace("./run4/TIC-","").replace("_lc.csv","")
    N_valid_TID_YMD.append(IDstr)
N_valid_TID_YMD=np.array(N_valid_TID_YMD).astype(str)

# write out useful info and data into files:
os.system("rm evryflare_III_table_I_v2.csv")

header = "i, TIC_ID, RA, Dec, Year, Mon, Day, HR, Min, g_mags, TESS_mags, distance, SpT, mass, Prot, Ro, Evry_Erg, e_Evry_Erg, TESS_erg, e_TESS_Erg, evr_peakFF, tess_peakFF, n_peaks, tot_BB_data, e_tot_BB_data, tot_BB_data_trap, e_tot_BB_data_trap, E_tot_BB_data_trap, tot_BB_sampl, e_tot_BB_sampl, E_tot_BB_sampl, FWHM_BB_data, e_FWHM_BB_data, FWHM_BB_sampl, e_FWHM_BB_sampl, E_FWHM_BB_sampl, FWHM, impulse"+"\n"
with open("evryflare_III_table_I_v2.csv","a") as OUTFILE_TABLE:
    OUTFILE_TABLE.write(header)

already_found = [25, 32, 43, 47, 50, 70, 76, 101, 102, 120, 161, 167, 168, 169, 175, 178, 233, 290, 346, 377, 379, 445, 482, 500, 553, 606, 608, 661, 674, 688, 692, 719, 769]

newly_found = [0, 4, 10, 16, 17, 19, 26, 110, 134, 157, 721]

print (len(already_found)+len(newly_found),"simultaneous flares total.\n")

fl_temperatures=[]
for i in range(len(TID)):

    # which starts and stops do we really want to use? EVRY or TESS?
    
    if i!=0:
        continue
    
    if (i not in already_found) and (i not in newly_found):
        continue
    
    g_mags_val = atl_g[atl_TIC==TID[i]][0]
    e_mags_val = atl_dg[atl_TIC==TID[i]][0]
    TESS_mags_val = MAST_Tmag[MatchID==TID[i]][0]
    #print TESS_mags_val
    #exit()
    e_TESS_mags_val = MAST_e_Tmag[MatchID==TID[i]][0]
    dist_pc = MAST_d[MatchID==TID[i]][0]
    e_dist_pc = MAST_e_d[MatchID==TID[i]][0]
    RA_val = MAST_RA[MatchID==TID[i]][0]
    Dec_val = MAST_Dec[MatchID==TID[i]][0]
    SpT = SpT_values[SpT_TIC==TID[i]][0]
    mass = SpT_masses[SpT_TIC==TID[i]][0]
    try:
        Prot = unique_Prot[unique_TIC_Prot==TID[i]][0]
        Ro = unique_rossby_num[unique_TIC_Prot==TID[i]][0]
    except(IndexError, ValueError):
        Prot = -1.0
        Ro = -1.0
        
    g_mags_val, e_mags_val, TESS_mags_val, e_TESS_mags_val = \
            correct_source_brightness(TID[i], g_mags_val, e_mags_val, TESS_mags_val, e_TESS_mags_val)

    #g_mags_val = 11.6
    #TESS_mags_val = 9.2841
    #evry_ff_adj = 1.14
    #tess_ff_adj = 1.39
    
    # get errors in quiescent luminosities:
    evr_Q_0, err_evr_Q_0, tess_Q_0, err_tess_Q_0 = \
        get_Q_0_errors(g_mags_val, e_mags_val, TESS_mags_val, e_TESS_mags_val, dist_pc, e_dist_pc)
    
    if i in already_found:
        
        TID_ACTUAL_YMD = str(TID[i])+"_"+str(YEAR[i])+"-"+str(MON[i])+"-"+str(DAY[i])
        print (TID_ACTUAL_YMD+" "+str(HR[i])+" "+str(MIN[i]))
        evr_filestring = glob.glob("./run4/TIC-"+str(TID_ACTUAL_YMD)+"*.csv")[0]
    
        evr_mjd, evr_gmag, evr_gmag_err, evr_flags = get_evr_lc(evr_filestring)
        actual_evr_gmag = copy.deepcopy(evr_gmag)
        
        #START=MJD_start[i]
        #STOP=MJD_stop[i]
        
    if i in newly_found:
        if len(str(MON[i]))<2:
            MONSTR = "0"+str(MON[i])
        else:
            MONSTR = str(MON[i])
        if len(str(DAY[i]))<2:
            DAYSTR = "0"+str(DAY[i])
        else:
            DAYSTR = str(DAY[i])
        TID_ACTUAL_YMD = str(TID[i])+"_"+str(YEAR[i])+str(MONSTR)+str(DAYSTR)

        print (TID_ACTUAL_YMD+" "+str(HR[i])+" "+str(MIN[i]))
        evr_filestring = glob.glob("./special/special_TIC-"+str(TID_ACTUAL_YMD)+"*.csv")[0]

        evr_mjd=[]
        evr_gmag=[]
        with open(evr_filestring,"r") as INFILE:
            for lines in INFILE:
                evr_mjd.append(float(lines.split(",")[0]))
                evr_gmag.append(float(lines.split(",")[1]))
        evr_mjd = np.array(evr_mjd)
        evr_gmag2 = copy.deepcopy(np.array(evr_gmag))
        
        evr_gmag = np.array(evr_gmag)-np.nanmedian(sigmaclip(evr_gmag,3.0,3.0)[0])
        
        if np.all(np.isnan(evr_gmag))==True:
            evr_gmag = evr_gmag2-np.nanmedian(evr_gmag2)

        actual_evr_gmag = evr_gmag + g_mags_val
        evr_gmag_err = 0.15*np.ones(len(evr_gmag))
        evr_gmag_err[actual_evr_gmag<=12.0] = 0.05
    
    #####    INSERT FLARE PARAMS    #####
    START=MJD_start[i]
    STOP=MJD_stop[i]

    WIND_START = START - 0.055
    WIND_STOP = STOP + 0.025

    lcv_TID = glob.glob("./run4/whtd_star/whtd_star*_TIC-"+str(TID[i])+"_*.csv")[0]
    
    tbjd, tmags = get_prewh_lcv(lcv_TID)

    PATH2 = "/home/wshoward/data1/tess/AAS233_figs/evr_flarestars_tess_lcvs/"
    
    tess_bjd2, sap_flux, sap_flux_err, first_sector, last_sector = \
            build_tess_lightcurve(TID[i], PATH2)
    tess_bjd2 = tess_bjd2 + 2457000.0 - 2400000.5

    tess_bjd_ext = np.concatenate((np.array([np.min(tbjd)]),tess_bjd2,np.array([np.max(tbjd)])))
    sap_flux_err_ext = np.concatenate((np.array([0.0]),sap_flux_err,np.array([0.0])))
    interp_tess_err = interp1d(tess_bjd_ext, sap_flux_err_ext, kind="linear")
    fracflux_tess_err = interp_tess_err(tbjd) #frac flux
    
    tess_conv_mags = -2.5*np.log10(sap_flux)
    #tess_conv_magerrs = -2.5*np.log10(sap_flux_err)
    #sap_frac_flux = sap_flux - 1.0
    
    # interp frac flux errors to tess y grid:
    

    old_inflare_tbjd = tbjd[(tbjd>START) & (tbjd<STOP)]
    old_inflare_tmags = tmags[(tbjd>START) & (tbjd<STOP)]

    inwindow_tbjd = tbjd[(tbjd>WIND_START) & (tbjd<WIND_STOP)]
    inwindow_tmags = tmags[(tbjd>WIND_START) & (tbjd<WIND_STOP)]
    inwindow_fracflux_err = fracflux_tess_err[(tbjd>WIND_START) & (tbjd<WIND_STOP)]

    inwindow_tess_conv_mags = tess_conv_mags[(tess_bjd2>WIND_START) & (tess_bjd2<WIND_STOP)]
    inwindow_tess_bjd2 = tess_bjd2[(tess_bjd2>WIND_START) & (tess_bjd2<WIND_STOP)]
    inwindow_raw_fracflux_err = copy.deepcopy(sap_flux_err)[(tess_bjd2>WIND_START) & (tess_bjd2<WIND_STOP)]
    
    evr_mjd_window =evr_mjd[(evr_mjd>WIND_START)&(evr_mjd<WIND_STOP)]-np.min(inwindow_tbjd)
    
    #evr_gmag = evr_gmag + 10.5
    #plt.plot(evr_mjd_window[evr_mjd_window>0.12],evr_gmag[(evr_mjd>WIND_START)&(evr_mjd<WIND_STOP)][evr_mjd_window>0.12],marker="o",ls="none",color="limegreen")
    #plt.gca().invert_yaxis()
    #plt.show()

    evr_fracflux = (-1.0) + 2.512**(-evr_gmag[(evr_mjd>WIND_START)&(evr_mjd<WIND_STOP)])
    evr_fracflux_err = (-1.0) + 2.512**(evr_gmag_err[(evr_mjd>WIND_START)&(evr_mjd<WIND_STOP)])
    
    evr_mjd_window, evr_fracflux, evr_fracflux_err = clean_evr_lc(i, evr_mjd_window, evr_fracflux, evr_fracflux_err)

    evr_mjd_window_insec = 24.0*3600.0*evr_mjd_window #in seconds

    NEWSTART = START - np.min(inwindow_tbjd)
    NEWSTOP = STOP - np.min(inwindow_tbjd)
    
    tess_mjd_window = inwindow_tbjd-np.min(inwindow_tbjd)
    tess_fracflux = (-1.0) + 2.512**(-inwindow_tmags)
    inwindow_fracflux_err = inwindow_fracflux_err
    
    tess_mjd_raw_window =  inwindow_tess_bjd2 - np.nanmin(inwindow_tbjd)
    tess_raw_fracflux = (-1.0) + 2.512**(-inwindow_tess_conv_mags)
    inwindow_raw_fracflux_err = inwindow_raw_fracflux_err
    
    tess_mjd_window, tess_fracflux, inwindow_fracflux_err = \
        adjust_tess_by_target(i, tess_mjd_window, tess_fracflux, inwindow_fracflux_err, tess_mjd_raw_window, tess_raw_fracflux, inwindow_raw_fracflux_err, NEWSTART, NEWSTOP)


    inflare_tbjd = copy.deepcopy(tess_mjd_window[(tess_mjd_window>=NEWSTART)&(tess_mjd_window<=NEWSTOP)])
    inflare_fflux = copy.deepcopy(tess_fracflux[(tess_mjd_window>=NEWSTART)&(tess_mjd_window<=NEWSTOP)])
    
    tess_mjd_window_insec = 24.0*3600.0*tess_mjd_window #in seconds

    evr_mjd_window,evr_fracflux,ESTART,ESTOP,FWHM_start_val,FWHM_stop_val,CW_START,CW_STOP = \
            adjust_evr_by_target(i, evr_mjd_window, evr_fracflux)

    ######
    my_evry = actual_evr_gmag[(evr_mjd>WIND_START)&(evr_mjd<WIND_STOP)]
    mx_evry = copy.deepcopy(evr_mjd[(evr_mjd>WIND_START)&(evr_mjd<WIND_STOP)])-np.min(inwindow_tbjd)
    my_evry_err = evr_gmag_err[(evr_mjd>WIND_START)&(evr_mjd<WIND_STOP)]


    
    ######
    mag_offset_err = get_fracflux_scaling_err(mx_evry, my_evry, my_evry_err, ESTART, ESTOP)
    ######
    
    x_tess = tess_mjd_window[(tess_mjd_window>=CW_START)&(tess_mjd_window<=CW_STOP)]
    y_tess = tess_fracflux[(tess_mjd_window>=CW_START)&(tess_mjd_window<=CW_STOP)]
    y_tess_err = inwindow_fracflux_err[(tess_mjd_window>=CW_START)&(tess_mjd_window<=CW_STOP)]
    
    x_evry = evr_mjd_window[(evr_mjd_window>=CW_START)&(evr_mjd_window<=CW_STOP)]
    y_evry = evr_fracflux[(evr_mjd_window>=CW_START)&(evr_mjd_window<=CW_STOP)]
    y_evry_err = evr_fracflux_err[(evr_mjd_window>=CW_START)&(evr_mjd_window<=CW_STOP)]

    #y_evry*=(evry_ff_adj)
    #y_tess*=(tess_ff_adj)
    
    cut_index = np.array([v for v in np.arange(len(y_evry)) if np.isnan(y_evry[v])==False]).astype(int)

    y_evry = y_evry[cut_index]
    y_evry_err = y_evry_err[cut_index]
    x_evry = x_evry[cut_index]
    

    index = np.argsort(x_evry)
    y_evry = y_evry[index]
    y_evry_err = y_evry_err[index]
    x_evry = x_evry[index]


    y_evry_sub = y_evry[(x_evry<ESTART) | (x_evry>ESTOP)]
    y_evry_sub_err = y_evry_err[(x_evry<ESTART) | (x_evry>ESTOP)]
    sub_flag = np.ones(len(y_evry_sub))
    sub_index = np.arange(len(y_evry_sub))

    n_trials=1000
    
    offset_distrib=[]
    for b in range(n_trials):
        sub_flag = np.ones(len(y_evry_sub))
        #mc_y_evry_sub = copy.deepcopy(y_evry_sub)
        mc_y_evry_sub = np.random.normal(y_evry_sub, y_evry_sub_err)
        width = int(np.floor(len(sub_index)/5.0))
        
        start_pos =  int(np.floor(len(sub_flag)*np.random.random()))
        #print start_pos
        #exit()
        #continue
        sub_flag[start_pos:(start_pos+width)]=-1.0
        #mc_y_evry_sub 
        MED = np.median(mc_y_evry_sub[sub_flag>0.0])
        offset_distrib.append(MED)
            
    offset_distrib = np.array(offset_distrib)
    offset_err = np.std(offset_distrib)
    offset_med = np.median(offset_distrib)
    if (offset_med+0.5*offset_err)>offset_err:
        offset_err=offset_med+offset_err
    
    np.random.shuffle(offset_distrib)
    
    interp_y_evry = interp1d(x_evry,y_evry,fill_value=0.00001,bounds_error=False,kind="linear")
    interp_y_evry_ERR = interp1d(x_evry,y_evry_err,fill_value=0.00001,bounds_error=False,kind="linear")
    
    y_evry_regrid = interp_y_evry(x_tess)
    y_evry_err_regrid = interp_y_evry_ERR(x_tess)

    #######
    init_y_tess = copy.deepcopy(y_tess)
    init_y_evry = copy.deepcopy(y_evry)
    init_y_evry_regrid = copy.deepcopy(y_evry_regrid)

    popt_evry_s1, popt_evry_s2, popt_evry_s3, x_evry, y_evry = fit_evr_flare_coeffs(i, x_evry, y_evry)
    popt_tess_s1, popt_tess_s2, popt_tess_s3, x_tess, y_tess = fit_flare_coeffs(i, x_tess, y_tess)
    
    ratio_erg_arr, ratio_erg_arr2, n_flares, tot_ratio, tot_sampl_ratio, FWHM_ratio, FWHM_sampl_ratio, EVRY_Erg, TESS_Erg, Evry_Erg_FWHM = compute_ratio(i, x_tess, copy.deepcopy(y_tess), y_tess_err, x_evry, copy.deepcopy(y_evry), copy.deepcopy(y_evry_regrid), y_evry_err_regrid, err_tess_Q_0, tess_Q_0, err_evr_Q_0, evr_Q_0, ESTART, ESTOP, FWHM_start_val, FWHM_stop_val, False)

    #info = str(i)+","+str(Evry_Erg_FWHM)+"\n"
    #with open("addendum_FWHM_energy.csv","a") as ROUTFILE:
    #    ROUTFILE.write(info)
    #continue

    tot_ratio_err = np.absolute(tot_ratio)*np.sqrt((err_evr_Q_0/evr_Q_0)**2.0 + (err_tess_Q_0/tess_Q_0)**2.0)
    #tot_sampl_ratio_err =
    FWHM_ratio_err =np.absolute(FWHM_ratio)*np.sqrt((err_evr_Q_0/evr_Q_0)**2.0 + (err_tess_Q_0/tess_Q_0)**2.0)
    #FWHM_sampl_ratio = 
    #print "formal err",tot_ratio_err,FWHM_ratio_err
    ### iteratio:

    """
    for b in range(len(x_tess)):
        ffx_info = str(x_tess[b])+","+str(y_tess[b])+","+str(y_tess_err[b])+","+"0\n"
        with open(str(i)+"_flare_fluxes_lc.csv","a") as OUTFILE:
            OUTFILE.write(ffx_info)
    for b in range(len(x_evry)):
        ffx_info = str(x_evry[b])+","+str(y_evry[b])+","+str(y_evry_err[b])+","+"1\n"
        with open(str(i)+"_flare_fluxes_lc.csv","a") as OUTFILE:
            OUTFILE.write(ffx_info)
    """
    
    n_trials = 200
    
    all_ratio_erg=[]
    errors_tot_ratio=[]
    errors_tot_sampl_ratio=[]
    errors_FWHM_ratio=[]
    errors_FWHM_sampl_ratio=[]
    errors_EVRY_Erg=[]
    errors_TESS_Erg=[]
    for trial in range(n_trials):
        mc_y_tess = copy.deepcopy(init_y_tess)
        
        new_zpoint = offset_distrib[trial]
        
        mc_y_evry = init_y_evry - new_zpoint
        
        mc_y_evry_regrid = init_y_evry_regrid - new_zpoint

        mc_y_evry_regrid = np.random.normal(mc_y_evry_regrid, mag_offset_err)

        SIGN = 1.0 - 2.0*np.random.random(len(mc_y_evry_regrid))
        
        SIGN[SIGN < 0.0] = -1.0
        SIGN[SIGN >= 0.0] = 1.0
        
        mc_y_evry_regrid = np.random.normal(mc_y_evry_regrid, 0.999*y_evry_err_regrid)
        #mc_y_evry_regrid = mc_y_evry_regrid + SIGN*y_evry_err_regrid*np.random.random(len(y_evry_err_regrid))
        
        try:
            mc_ratio_erg_arr, mc_ratio_erg_arr2, mc_n_flares, mc_tot_ratio, mc_tot_sampl_ratio, mc_FWHM_ratio, mc_FWHM_sampl_ratio, mc_EVRY_Erg, mc_TESS_Erg, mc_Evry_Erg_FWHM = compute_ratio(i, x_tess, mc_y_tess, y_tess_err, x_evry, mc_y_evry, mc_y_evry_regrid, y_evry_err_regrid, err_tess_Q_0, tess_Q_0, err_evr_Q_0, evr_Q_0, ESTART, ESTOP, FWHM_start_val,FWHM_stop_val, True)
        except (RuntimeError):
            continue

        errors_tot_ratio.append(mc_tot_ratio)
        errors_tot_sampl_ratio.append(mc_tot_sampl_ratio)
        errors_FWHM_ratio.append(mc_FWHM_ratio)
        errors_FWHM_sampl_ratio.append(mc_FWHM_sampl_ratio)
        #print mc_FWHM_sampl_ratio
        errors_EVRY_Erg.append(mc_EVRY_Erg)
        errors_TESS_Erg.append(mc_TESS_Erg)
        
        if len(all_ratio_erg)==0:
            all_ratio_erg = mc_ratio_erg_arr2
        else:
            all_ratio_erg = np.vstack((all_ratio_erg, mc_ratio_erg_arr2))

    ERR_tot_ratio=np.nanstd(errors_tot_ratio)
    ERR_tot_sampl_ratio=np.nanstd(errors_tot_sampl_ratio)
    ERR_FWHM_ratio=np.nanstd(errors_FWHM_ratio)
    ERR_FWHM_sampl_ratio=np.nanstd(errors_FWHM_sampl_ratio)

    e_EVRY_Erg = np.round(np.nanstd(errors_EVRY_Erg),2)
    e_TESS_Erg = np.round(np.nanstd(errors_TESS_Erg),2)
    
    tot_ratio_upp = tot_ratio + ERR_tot_ratio
    tot_sampl_ratio_upp = tot_sampl_ratio + ERR_tot_sampl_ratio
    FWHM_ratio_upp = FWHM_ratio + ERR_FWHM_ratio
    FWHM_sampl_ratio_upp = FWHM_sampl_ratio + ERR_FWHM_sampl_ratio
    tot_ratio_low = tot_ratio - ERR_tot_ratio
    tot_sampl_ratio_low = tot_sampl_ratio - ERR_tot_sampl_ratio
    FWHM_ratio_low = FWHM_ratio - ERR_FWHM_ratio
    FWHM_sampl_ratio_low = FWHM_sampl_ratio - ERR_FWHM_sampl_ratio

    if tot_sampl_ratio>3.0:
        tot_sampl_ratio = 3.0
    if tot_sampl_ratio<0.00001:
        tot_sampl_ratio = 0.00001
    if FWHM_sampl_ratio>3.0:
        FWHM_sampl_ratio=3.0
    if FWHM_sampl_ratio<0.00001:
        FWHM_sampl_ratio = 0.00001
    if tot_ratio>3.0:
        tot_ratio = 3.0
    if tot_ratio<0.00001:
        tot_ratio = 0.00001
    
    if tot_ratio_upp>3.0:
        tot_ratio_upp = 3.0
    if tot_sampl_ratio_upp>3.0:
        tot_sampl_ratio_upp = 3.0
    if FWHM_ratio_upp>3.0:
        FWHM_ratio_upp = 3.0
    if FWHM_sampl_ratio_upp>3.0:
        FWHM_sampl_ratio_upp = 3.0
    if tot_ratio_low>3.0:
        tot_ratio_low = 3.0
    if tot_sampl_ratio_low>3.0:
        tot_sampl_ratio_low = 3.0
    if FWHM_ratio_low>3.0:
        FWHM_ratio_low = 3.0
    if FWHM_sampl_ratio_low>3.0:
        FWHM_sampl_ratio_low = 3.0
    if tot_ratio_upp<0.00001:
        tot_ratio_upp = 0.00001
    if tot_sampl_ratio_upp<0.00001:
        tot_sampl_ratio_upp = 0.00001
    if FWHM_ratio_upp<0.00001:
        FWHM_ratio_upp = 0.00001
    if FWHM_sampl_ratio_upp<0.00001:
        FWHM_sampl_ratio_upp = 0.00001
    if tot_ratio_low<0.00001:
        tot_ratio_low = 0.00001
    if tot_sampl_ratio_low<0.00001:
        tot_sampl_ratio_low = 0.00001
    if FWHM_ratio_low<0.00001:
        FWHM_ratio_low = 0.00001
    if FWHM_sampl_ratio_low<0.00001:
        FWHM_sampl_ratio_low = 0.00001
    
    tot_temp_upp = get_teff(tot_ratio_upp)
    tot_sampl_temp_upp = get_teff(tot_sampl_ratio_upp)
    FWHM_temp_upp = get_teff(FWHM_ratio_upp)
    FWHM_sampl_temp_upp = get_teff(FWHM_sampl_ratio_upp)
    tot_temp_low = get_teff(tot_ratio_low)
    tot_sampl_temp_low = get_teff(tot_sampl_ratio_low)
    FWHM_temp_low = get_teff(FWHM_ratio_low)
    FWHM_sampl_temp_low = get_teff(FWHM_sampl_ratio_low)
    
    #exit()
    
    ratio_erg_upp2 = []
    ratio_erg_low2 = []
    for k in range(len(all_ratio_erg.T)):
        try:
            target_array = all_ratio_erg[:,k]
        except(IndexError):
            print (all_ratio_erg)
            print (all_ratio_erg.shape)
            print (len(all_ratio_erg.T))
            print (k)
            exit()
        #print np.nanstd(target_array), np.nanstd(target_array[target_array<2.14])
        RATMED = np.nanmedian(target_array[target_array<2.14])
        RATSTD = np.nanstd(target_array[target_array<2.14])
        ratio_erg_upp2.append(RATMED + RATSTD)
        ratio_erg_low2.append(RATMED - RATSTD)
    ratio_erg_upp2 = np.array(ratio_erg_upp2)
    ratio_erg_low2 = np.array(ratio_erg_low2)
    
    #exit()

    # full errors, including systematic offsets:
    BOT_total_evr_y_err = np.sqrt(y_evry_err_regrid**2.0 + (np.ones(len(y_evry_err_regrid))*offset_err)**2.0)
    TOP_total_evr_y_err = np.sqrt(y_evry_err_regrid**2.0 + (np.ones(len(y_evry_err_regrid))*offset_err)**2.0)
    
    BOT_err_ratio_erg_arr= np.absolute(ratio_erg_arr)*np.sqrt((BOT_total_evr_y_err/y_evry_regrid)**2.0 + (y_tess_err/y_tess)**2.0 + (err_evr_Q_0/evr_Q_0)**2.0 + (err_tess_Q_0/tess_Q_0)**2.0 + (mag_offset_err/y_evry_regrid)**2.0)

    TOP_err_ratio_erg_arr= np.absolute(ratio_erg_arr)*np.sqrt((TOP_total_evr_y_err/y_evry_regrid)**2.0 + (y_tess_err/y_tess)**2.0 + (err_evr_Q_0/evr_Q_0)**2.0 + (err_tess_Q_0/tess_Q_0)**2.0 + (mag_offset_err/y_evry_regrid)**2.0)

    # formal errors only:
    BOT_err_ratio_formal = np.absolute(ratio_erg_arr)*np.sqrt((y_evry_err_regrid/y_evry_regrid)**2.0 + (y_tess_err/y_tess)**2.0 + (err_evr_Q_0/evr_Q_0)**2.0 + (err_tess_Q_0/tess_Q_0)**2.0)

    TOP_err_ratio_formal = np.absolute(ratio_erg_arr)*np.sqrt((y_evry_err_regrid/y_evry_regrid)**2.0 + (y_tess_err/y_tess)**2.0 + (err_evr_Q_0/evr_Q_0)**2.0 + (err_tess_Q_0/tess_Q_0)**2.0)
    
    #######
    
    temp_arr = get_teff(ratio_erg_arr)
    temp_index = np.arange(len(temp_arr))
    GOOD_T_IND = temp_index[temp_arr<48000.0]
    ratio_erg_upp = ratio_erg_arr + TOP_err_ratio_erg_arr
    ratio_erg_low = ratio_erg_arr - BOT_err_ratio_erg_arr
    formal_ratio_erg_upp = ratio_erg_arr + TOP_err_ratio_formal
    formal_ratio_erg_low = ratio_erg_arr - BOT_err_ratio_formal
    ratio_erg_upp[ratio_erg_upp<0.00001] =0.00001
    ratio_erg_low[ratio_erg_low<0.00001] =0.00001
    ratio_erg_upp[ratio_erg_upp>3.0] =3.0
    ratio_erg_low[ratio_erg_low>3.0] =3.0

    
    formal_ratio_erg_upp[formal_ratio_erg_upp<0.00001] =0.00001
    formal_ratio_erg_low[formal_ratio_erg_low<0.00001] =0.00001
    formal_ratio_erg_upp[formal_ratio_erg_upp>3.0] =3.0
    formal_ratio_erg_low[formal_ratio_erg_low>3.0] =3.0
    
    temp_upp_err = get_teff(ratio_erg_upp)
    temp_low_err = get_teff(ratio_erg_low)

    temp_upp_err = np.absolute(temp_upp_err - temp_arr)
    temp_low_err = np.absolute(temp_low_err - temp_arr)

    formal_temp_upp_err = get_teff(formal_ratio_erg_upp)
    formal_temp_low_err = get_teff(formal_ratio_erg_low)

    formal_temp_upp_err = np.absolute(formal_temp_upp_err - temp_arr)
    formal_temp_low_err = np.absolute(formal_temp_low_err - temp_arr)

    actual_FWHM_temp = np.nanmean(temp_arr[(x_tess>FWHM_start_val)&(x_tess<FWHM_stop_val)])
    actual_tot_temp = np.nanmean(temp_arr[(x_tess>ESTART)&(x_tess<ESTOP)])

    #####
    tot_BB_data = actual_tot_temp
    tot_BB_data_trap =get_teff(tot_ratio)
    e_tot_BB_data_trap_upp = np.absolute(tot_BB_data_trap - tot_temp_upp)
    e_tot_BB_data_trap_low = np.absolute(tot_BB_data_trap - tot_temp_low)
    #e_tot_BB_data = 
    tot_BB_sampl = get_teff(tot_sampl_ratio)
    e_tot_BB_sampl_upp = np.absolute(tot_BB_sampl - tot_sampl_temp_upp)
    e_tot_BB_sampl_low = np.absolute(tot_BB_sampl - tot_sampl_temp_low)
    FWHM_BB_data =actual_FWHM_temp
    #e_FWHM_BB_data =e_tot_BB_data
    FWHM_BB_sampl = get_teff(FWHM_sampl_ratio)
    e_FWHM_BB_sampl_upp = np.absolute(FWHM_BB_sampl - FWHM_sampl_temp_upp)
    e_FWHM_BB_sampl_low = np.absolute(FWHM_BB_sampl - FWHM_sampl_temp_low)

    #print FWHM_BB_sampl,e_FWHM_BB_sampl_low,e_FWHM_BB_sampl_upp
    #print tot_BB_sampl,e_tot_BB_sampl_low,e_tot_BB_sampl_upp
    #exit()
    #####
    
    mc_tot_temps = temp_arr[(x_tess>FWHM_start_val)&(x_tess<FWHM_stop_val)]
    mc_FWHM_temps = temp_arr[(x_tess>ESTART)&(x_tess<ESTOP)]

    n_trials = 1000

    mc_tot_temp_arr=[]
    mc_FWHM_temp_arr=[]
    for t in range(n_trials):
        mc_tot_temps = (temp_arr[(x_tess>ESTART)&(x_tess<ESTOP)]-temp_low_err[(x_tess>ESTART)&(x_tess<ESTOP)]) + (temp_upp_err[(x_tess>ESTART)&(x_tess<ESTOP)]+temp_low_err[(x_tess>ESTART)&(x_tess<ESTOP)])*np.random.random(len(temp_arr[(x_tess>ESTART)&(x_tess<ESTOP)]))

        mc_FWHM_temps = (temp_arr[(x_tess>FWHM_start_val)&(x_tess<FWHM_stop_val)]-temp_low_err[(x_tess>FWHM_start_val)&(x_tess<FWHM_stop_val)]) + (temp_upp_err[(x_tess>FWHM_start_val)&(x_tess<FWHM_stop_val)]+temp_low_err[(x_tess>FWHM_start_val)&(x_tess<FWHM_stop_val)])*np.random.random(len(temp_arr[(x_tess>FWHM_start_val)&(x_tess<FWHM_stop_val)]))

        mc_tot_temp_arr.append(np.mean(mc_tot_temps))
        mc_FWHM_temp_arr.append(np.mean(mc_FWHM_temps))
    mc_tot_temp_arr=np.array(mc_tot_temp_arr)
    mc_FWHM_temp_arr=np.array(mc_FWHM_temp_arr)
    
    mc_tot_temp_arr = [x for x in mc_tot_temp_arr if (x>1000) and (x<48000)]
    mc_FWHM_temp_arr = [x for x in mc_FWHM_temp_arr if (x>1000) and (x<48000)]

    e_tot_BB_data=np.nanstd(mc_tot_temp_arr)
    e_FWHM_BB_data=np.nanstd(mc_FWHM_temp_arr)
    
    temp_arr2 = get_teff(ratio_erg_arr2)
    temp_index2 = np.arange(len(temp_arr2))
    GOOD_T_IND2 = temp_index2[temp_arr2<48000.0]

    ### MODEL UNCERTAINTIES ###
    #ratio_erg_upp2 = ratio_erg_arr2 + TOP_err_ratio_erg_arr
    #ratio_erg_low2 = ratio_erg_arr2 - BOT_err_ratio_erg_arr
    formal_ratio_erg_upp2 = ratio_erg_arr2 + TOP_err_ratio_formal
    formal_ratio_erg_low2 = ratio_erg_arr2 - BOT_err_ratio_formal
    ratio_erg_upp2[ratio_erg_upp2<0.00001] =0.00001
    ratio_erg_low2[ratio_erg_low2<0.00001] =0.00001
    ratio_erg_upp2[ratio_erg_upp2>3.0] =3.0
    ratio_erg_low2[ratio_erg_low2>3.0] =3.0

    #formal_ratio_erg_upp2[formal_ratio_erg_upp2<0.00001] =0.00001
    #formal_ratio_erg_low2[formal_ratio_erg_low2<0.00001] =0.00001
    #formal_ratio_erg_upp2[formal_ratio_erg_upp2>3.0] =3.0
    #formal_ratio_erg_low2[formal_ratio_erg_low2>3.0] =3.0
    
    temp_upp_err2 = get_teff(ratio_erg_upp2)
    temp_low_err2 = get_teff(ratio_erg_low2)

    temp_upp_err2 = np.absolute(temp_upp_err2 - temp_arr2)
    temp_low_err2 = np.absolute(temp_low_err2 - temp_arr2)

    #formal_temp_upp_err2 = get_teff(formal_ratio_erg_upp2)
    #formal_temp_low_err2 = get_teff(formal_ratio_erg_low2)

    #formal_temp_upp_err2 = np.absolute(formal_temp_upp_err2 - temp_arr2)
    #formal_temp_low_err2 = np.absolute(formal_temp_low_err2 - temp_arr2)
    ###
    
    best_fit_temp = np.round(np.median(temp_arr),-2)
    #print best_fit_temp,np.median(ratio_erg_arr)
    fl_temperatures.append(best_fit_temp)
    #if best_fit_temp>39000.0:
    #    print i, best_fit_temp,"K"
    #    exit()
    #continue


    plt.title(i)
    plt.plot(24.0*x_tess[GOOD_T_IND],temp_arr[GOOD_T_IND],marker="o",color="black",ls="none")
    plt.fill_between(24.0*x_tess[GOOD_T_IND],temp_arr[GOOD_T_IND]-temp_low_err[GOOD_T_IND],temp_arr[GOOD_T_IND]+temp_upp_err[GOOD_T_IND],color="lightgrey")
    plt.errorbar(24.0*x_tess[GOOD_T_IND],temp_arr[GOOD_T_IND],yerr=[formal_temp_low_err[GOOD_T_IND],formal_temp_upp_err[GOOD_T_IND]],marker="o",color="black",ls="none")
    

    plt.fill_between(24.0*x_tess[GOOD_T_IND2],temp_arr2[GOOD_T_IND2]-temp_low_err2[GOOD_T_IND2],temp_arr2[GOOD_T_IND2]+temp_upp_err2[GOOD_T_IND2],color="lightblue")
    plt.plot(24.0*x_tess[GOOD_T_IND2],temp_arr2[GOOD_T_IND2],marker="o",ms=0,ls="-",linewidth=2,color="royalblue")

    #plt.plot(24.0*x_tess[GOOD_T_IND2],temp_arr2[GOOD_T_IND2],marker="o",color="grey",ls="none")

    plt.ylim(-25.0,1.1*np.max(temp_arr[GOOD_T_IND]))
    
    plt.xlabel("Time elapsed [hr]",fontsize=14)
    plt.ylabel("Temperature [K]",fontsize=14)
    plt.tight_layout()
    #plt.savefig(str(i)+"_flare_BB_temp.png")
    plt.show()
    plt.close("all")
    
    rmstr = "rm "+str(i)+"_flaretemp_data_lc.csv"
    os.system(rmstr)
    rmstr = "rm "+str(i)+"_flaretemp_model_lc.csv"
    os.system(rmstr)
    rmstr = "rm "+str(i)+"_flare_fits_lc.csv"
    os.system(rmstr)
    
    for t in range(len(x_tess[GOOD_T_IND])):
        wo_data_times = np.array(x_tess[GOOD_T_IND])[t]
        wo_data_temps = np.array(temp_arr[GOOD_T_IND])[t]
        wo_data_lowerr = np.array(temp_low_err[GOOD_T_IND])[t]
        wo_data_upperr = np.array(temp_upp_err[GOOD_T_IND])[t]
        wo_data_formal_lowerr = np.array(formal_temp_low_err[GOOD_T_IND])[t]
        wo_data_formal_upperr = np.array(formal_temp_upp_err[GOOD_T_IND])[t]

        d_info = str(wo_data_times)+","+str(wo_data_temps)+","+str(wo_data_lowerr)+","+str(wo_data_upperr)+","+str(wo_data_formal_lowerr)+","+str(wo_data_formal_upperr)+"\n"
        #with open(str(i)+"_flaretemp_data_lc.csv","a") as DATA_OUTFILE:
        #    DATA_OUTFILE.write(d_info)
            
    for t in range(len(x_tess[GOOD_T_IND2])):
        wo_model_times = np.array(x_tess[GOOD_T_IND2])[t]
        wo_model_temps = np.array(temp_arr2[GOOD_T_IND2])[t]
        wo_model_lowerr = np.array(temp_low_err2[GOOD_T_IND2])[t]
        wo_model_upperr = np.array(temp_upp_err2[GOOD_T_IND2])[t]

        m_info = str(wo_model_times)+","+str(wo_model_temps)+","+str(wo_model_lowerr)+","+str(wo_model_upperr)+"\n"
        #with open(str(i)+"_flaretemp_model_lc.csv","a") as MODEL_OUTFILE:
        #    MODEL_OUTFILE.write(m_info)

    popt_evry_s1, popt_evry_s2, popt_evry_s3, x_evry, alt_y_evry = fit_evr_flare_coeffs(i, x_evry, init_y_evry)
    popt_tess_s1, popt_tess_s2, popt_tess_s3, x_tess, alt_y_tess = fit_flare_coeffs(i, x_tess, init_y_tess)
    popt_evry_all = list(popt_evry_s1) + list(popt_evry_s2) + list(popt_evry_s3)
    popt_tess_all = list(popt_tess_s1) + list(popt_tess_s2) + list(popt_tess_s3)
    
    fit_times = np.linspace(np.nanmin(x_tess), np.nanmax(x_tess), num=1000)
    fit_evry_fracflux = three_peak_dav_flare_model(fit_times, *popt_evry_all)
    fit_tess_fracflux = three_peak_dav_flare_model(fit_times, *popt_tess_all)
    
    for t in range(len(fit_times)):
        wo_fit_times = np.array(fit_times)[t]
        wo_fit_evry_fracflux = np.array(fit_evry_fracflux)[t]
        wo_fit_tess_fracflux = np.array(fit_tess_fracflux)[t]

        f_info = str(wo_fit_times)+","+str(wo_fit_evry_fracflux)+","+str(wo_fit_tess_fracflux)+"\n"
        #with open(str(i)+"_flare_fits_lc.csv","a") as FIT_OUTFILE:
        #    FIT_OUTFILE.write(f_info)


    FWHM = (FWHM_stop_val-FWHM_start_val)*24.0*60.0 #in mins
    evr_peakFF = np.round(np.nanmax(evr_fracflux[(evr_mjd_window>FWHM_start_val)&(evr_mjd_window<FWHM_stop_val)]),2)
    tess_peakFF = np.round(np.nanmax(tess_fracflux[(tess_mjd_window>FWHM_start_val)&(tess_mjd_window<FWHM_stop_val)]),2)

    impulse = np.round(evr_peakFF/FWHM,4)
    
    # write out useful info and data into files:
    # "i,TID, RA_val, Dec_val, YEAR, MON, DAY, HR, MIN, g_mags, TESS_mags, distance, SpT, mass, Prot, Ro, Evry_Erg, e_Evry_Erg, TESS_erg, e_TESS_Erg, evr_peakFF, tess_peakFF, n_peaks, tot_BB_data, e_tot_BB_data, tot_BB_sampl, e_tot_BB_sampl, E_tot_BB_sampl, FWHM_BB_data, e_FWHM_BB_data, FWHM_BB_sampl, e_FWHM_BB_sampl, E_FWHM_BB_sampl, FWHM, impulse"

    tab_info = str(i)+","+str(TID[i])+","+str(np.round(RA_val,5))+","+str(np.round(Dec_val,5))+","+str(YEAR[i])+","+str(MON[i])+","+str(DAY[i])+","+str(HR[i])+","+str(MIN[i])+","+str(g_mags_val)+","+str(np.round(TESS_mags_val,3))+","+str(np.round(dist_pc,2))+","+str(SpT)+","+str(np.round(mass,3))+","+str(Prot)+","+str(np.round(Ro,3))+","+str(np.round(np.log10(EVRY_Erg),2))+","+str(np.round(np.log10(e_EVRY_Erg),2))+","+str(np.round(np.log10(TESS_Erg),2))+","+str(np.round(np.log10(e_TESS_Erg),2))+","+str(evr_peakFF)+","+str(tess_peakFF)+","+str(n_flares)+","+str(int(np.round(tot_BB_data, -2)))+","+str(int(np.round(e_tot_BB_data,-2)))+","+str(int(np.round(tot_BB_data_trap,-2)))+","+str(int(np.round(e_tot_BB_data_trap_low,-2)))+","+str(int(np.round(e_tot_BB_data_trap_upp,-2)))+","+str(int(np.round(tot_BB_sampl,-2)))+","+str(int(np.round(e_tot_BB_sampl_low,-2)))+","+str(int(np.round(e_tot_BB_sampl_upp,-2)))+","+str(int(np.round(FWHM_BB_data,-2)))+","+str(int(np.round(e_FWHM_BB_data,-2)))+","+str(int(np.round(FWHM_BB_sampl,-2)))+","+str(int(np.round(e_FWHM_BB_sampl_low,-2)))+","+str(int(np.round(e_FWHM_BB_sampl_upp,-2)))+","+str(FWHM)+","+str(impulse)+"\n"

    with open("evryflare_III_table_I_v2.csv","a") as OUTFILE_TABLE:
        OUTFILE_TABLE.write(tab_info)


    continue
    #exit()
    #exit()
    evi = np.argsort(x_evry) #evryscope x-index    

    """
    fig, ax = plt.subplots(figsize=(7,5))
    plt.axis('off')

    plt.title(str(i)+" TIC "+TID_ACTUAL_YMD.replace("_","  UT: ")+" "+str(HR[i])+":"+str(MIN[i]))
    
    ax1 = fig.add_subplot(111)
    
    ax1.plot(x_evry*24.0, y_evry, marker="o",ls="none", color="cornflowerblue")
    ax1.plot(x_tess*24.0, sampled_evr_flare, ls="-", color="darkblue")
    #ax1.plot(upsampl_x_evry*24.0, upsampl_evr_flare, ls="-", color="grey")
    plt.ylabel("$\Delta$F/F in $g^{\prime}$-band",color="royalblue",fontsize=14)
    plt.yticks(color="royalblue",fontsize=12)
    #plt.xticks(color="black",fontsize=12)
    plt.xlabel("Elapsed Time [hr]",fontsize=14)
    
    ax3 = ax1.twinx()
    ax3.plot(x_tess*24.0, y_tess, marker="o",ls="none", color="firebrick")
    ax3.plot(x_tess*24.0, sampled_flare, ls="-", color="darkorange")
    plt.yticks(color="firebrick",fontsize=12)
    plt.ylabel("$\Delta$F/F in TESS band",color="firebrick",fontsize=14)
    plt.tight_layout()
    #plt.savefig("/home/wshoward/Desktop/repeating_flare_"+str(TID_ACTUAL_YMD)+".png")
    plt.show()
    """

    """
    
    #def ampl_only_flare_model1(epochs, ampl):
    #    return davenport_flare_model(epochs, popt_tess_s1[0], ampl, popt_tess_s1[2])

    #continue
    #exit()
    
    fig, ax = plt.subplots(figsize=(8,5))
    plt.axis('off')

    ax1 = fig.add_subplot(211)
    plt.title(str(i)+" "+str(TID[i]))

    ax1.axhline(0.5*np.nanmax(evr_fracflux),color="darkorange")

    #ax1.axvline(fl_start_val,color="green")
    #ax1.axvline(fl_stop_val,color="green")
    ax1.axvline(FWHM_start_val,color="salmon")
    ax1.axvline(FWHM_stop_val,color="salmon")
    
    info = str(evr_peakFF)+","+str(tess_peakFF)+","+str(impulse)+","+str(best_fit_temp)+"\n"
    with open("play_with_flares.csv","a") as OUT:
        OUT.write(info)
        
    #peaktime=evr_mjd_window[list(evr_fracflux).index(evr_peakFF)]
    
    #ax1.plot(peaktime,peakFF,marker="o",ms=15,color="red")
    #ax1.axvline(evr_mjd_window[list(evr_fracflux).index(MAX_VAL)],color="black")

    #ax1.plot(evr_xgrid_insec/(24.0*3600.0),evr_ygrid_insec,marker="o",ms=2,ls="none",color="mediumslateblue")
    #ax1.plot(tess_xgrid_insec/(24.0*3600.0),tess_ygrid_insec,marker="o",ms=2,ls="none",color="firebrick")
    
    #ax1.plot(tess_mjd_window, tess_fracflux, marker="+", ms=9, ls="none",color="black")

    ax1.plot(tess_mjd_window, tess_fracflux, marker="+", ms=9, ls="none",color="brown")
    
    ax1.plot(inflare_tbjd, inflare_fflux, marker="+", ms=9, ls="none",color="red")

    ax1.plot(evr_mjd_window, evr_fracflux, marker="o",ls="none",color="mediumslateblue")
    
    #info = str(i)+", "+str(TID[i])+", "+str(TID_ACTUAL_YMD)+", "+str(np.min(inwindow_tbjd))+", \n"
    #with open("list_multiband_flares.csv","a") as OUTFILE:
    #    OUTFILE.write(info)

    #plt.xticks(rotation=90.0)
    plt.xlabel("BMJD",fontsize=12)
    plt.ylabel("$\Delta$ F/F",fontsize=12)
    #plt.gca().invert_yaxis()

    ax2 = fig.add_subplot(212)
    
    ax2.plot(tbjd, tmags, marker="+", ls="none",color="black")
    ax2.plot(old_inflare_tbjd, old_inflare_tmags, marker="+", ls="none",color="red")
    #plt.xticks(rotation=90.0)
    plt.xlabel("TBJD",fontsize=12)
    plt.ylabel("TESS-mags",fontsize=12)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
    plt.close("all")

"""

"""
plt.hist(fl_temperatures,range=[4000,35000])
plt.xlabel("Color-temperature [K]",fontsize=14)
plt.ylabel("# flares",fontsize=14)
#plt.savefig("/home/wshoward/Desktop/BB_temps.png")
plt.show()
"""
