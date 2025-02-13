"""
Script to download the SDSS-V MWM/BOSS spectra
"""

# import the necessary modules
import numpy as np
from astropy.io import fits
from astropy.table import Table

# load in the file of stars you want the spectra for

file = 'APOGEE_BOSS.fits'
path = '/Users/dhortadarrington/Documents/Projects/Lux-BOSS/data/'+str(file)
tb = fits.open(path)
dat = tb[1].data 

# mask out any bad or missing values
mask = (dat['snr_boss']>20)
print('Initial size of sample: '+str(len(dat)))
print('Size of sample with good stellar parameters: '+str(len(dat[mask])))

dat = dat[mask]
dat = dat[:10000]

# command line command to get spectra for one star 
# wget --spider https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/000+02/apStar-dr17-2M17335483-2753043.fits
# the "redux" path gets you the raw visit spectra, which isn't combined. Below is the path needed to get the combined spectra

# command line command to get spectra for one star 
# wget --spider https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/000+02/apStar-dr17-2M17335483-2753043.fits
# the "redux" path gets you the raw visit spectra, which isn't combined. Below is the path needed to get the combined spectra

print('Downloading spectra for '+str(len(dat))+' stars')

# Code to do bulk download
# master = 'wget -P /Users/dhortadarrington/Documents/Projects/Lux-BOSS/spec/ -np -xnH --cut-dirs 9 --no-check-certificate --user sdss5 --password panoPtic-5 -r https://data.sdss5.org/sas/sdsswork/mwm/spectro/astra/'

master = '/mwm/spectro/astra/'

#example spectra file link: https://data.sdss5.org/sas/sdsswork/mwm/spectro/astra/0.6.0/spectra/star/00/00/mwmStar-0.6.0-102600000.fits

# astra_version = dat['v_astra']
astra_version = '0.6.0'
data_type = 'spectra'
object_type = 'star'
filetype = dat['filetype']
sdss_id = dat['sdss_id']


paths = []
for indx, i in enumerate(dat):
    first_dir_name = str(sdss_id[indx])[-4:-2]
    second_dir_name = str(sdss_id[indx])[-2:]
    paths.append(master+astra_version+str('/')+data_type+str('/')+object_type+str('/')+\
    first_dir_name+str('/')+second_dir_name+str('/')+'mwmStar'+str('-')+\
    astra_version+str('-')+str(sdss_id[indx])+str('.fits'))                

    
# save the file with all the spectra
savepath = '/Users/dhortadarrington/Documents/Projects/Lux-BOSS/spec/'
file_name = 'APOGEE_BOSS'
np.savetxt(savepath+'rsync-'+str(file_name)+'.txt', paths, fmt = "%s") 

###################### After getting the wget file
# you run in the command line, followed by 
# rsync -avz --files-from=rsync.txt rsync://sdss5@dtn.sdss.org/sdsswork sdsswork

# this will download all the spectra files into the folder "spectra-reference-aspcapStar/" within the directory you are in


#####
# if by any chance the spectra are downloaded like on the SAS into subdirectories

# you can then run:
# " find . -name '*.fits' -exec mv {} /Users/dhortadarrington/Documents/Projects/El-Cañón/spec \; "
# to get all the files in a single directory

########################## save the input training data file with the stellar parameters of interest
# this will need to be modified depending on what you wanna train on

id = dat['sdss_id']
teff = dat['teff']
logg = dat['logg']
feh = dat['fe_h']
mgh = dat['mg_h']
alh = dat['al_h']
nah = dat['na_h']
ch = dat['c_h']
nh = dat['n_h']
sih = dat['si_h']
nih = dat['ni_h']
mnh = dat['mn_h']
ceh = dat['ce_h']
coh = dat['co_h']
crh = dat['cr_h']
cuh = dat['cu_h']
kh = dat['k_h']
ndh = dat['nd_h']
oh = dat['o_h']
ph = dat['p_h']
sh = dat['s_h']
tih = dat['ti_h']

e_teff = dat['e_teff']
e_logg = dat['e_logg']
e_feh = dat['e_fe_h']
e_mgh = dat['e_mg_h']
e_alh = dat['e_al_h']
e_nah = dat['e_na_h']
e_ch = dat['e_c_h']
e_nh = dat['e_n_h']
e_sih = dat['e_si_h']
e_nih = dat['e_ni_h']
e_mnh = dat['e_mn_h']
e_ceh = dat['e_ce_h']
e_coh = dat['e_co_h']
e_crh = dat['e_cr_h']
e_cuh = dat['e_cu_h']
e_kh = dat['e_k_h']
e_ndh = dat['e_nd_h']
e_oh = dat['e_o_h']
e_ph = dat['e_p_h']
e_sh = dat['e_s_h']
e_tih = dat['e_ti_h']



t = Table([id, teff, logg, feh, mgh, alh, nah, ch, nh, sih, nih, mnh, ceh, coh, crh, cuh, kh, ndh, oh, ph, sh, tih,\
          e_teff, e_logg, e_feh, e_mgh, e_alh, e_nah, e_ch, e_nh, e_sih, e_nih, e_mnh, e_ceh, e_coh, e_crh, e_cuh, e_kh, e_ndh, e_oh, e_ph, e_sh, e_tih],\
       names=('sdss_id', 'teff', 'logg', 'feh', 'mgh', 'alh', 'nah', 'ch', 'nh', 'sih', 'nih', 'mnh', 'ceh', 'coh', 'crh', 'cuh', 'kh', 'ndh', 'oh', 'ph', 'sh', 'tih',\
          'e_teff', 'e_logg', 'e_feh', 'e_mgh', 'e_alh', 'e_nah', 'e_ch', 'e_nh', 'e_sih', 'e_nih', 'e_mnh', 'e_ceh', 'e_coh', 'e_crh', 'e_cuh', 'e_kh', 'e_ndh', 'e_oh', 'e_ph', 'e_sh', 'e_tih'))

savepath = '/Users/dhortadarrington/Documents/Projects/Lux-BOSS/data/'
t.write(savepath+'training-set-'+str(file_name)+'.fits', format='fits', overwrite=True)
