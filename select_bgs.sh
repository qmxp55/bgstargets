#!/bin/bash

module load python/3.7-anaconda-2019.07
source deactivate
conda deactivate 
conda activate DESI_BGS_omar

export DR='dr9k'
export outdir='/global/cscratch1/sd/qmxp55/bgstargets_output'
export version='0.1.0'
export Nran='3'

mkdir -p ${outdir}/${DR}/${version}



#run below to generate catalogue and randoms without rmag limit -- this might take a while!
#python bin/getSweeps.py -dr ${DR}-north -OF ${outdir}/${DR}/ -rlim None 
#python bin/getBGSBITS.py -IF ${outdir}/${DR}/${DR}-north_sweep_whole.npy -OF ${outdir}/${DR}/${version}/bgstargets-north.npy
#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-north.npy -OF ${outdir}/${DR}/${version}/extra-north_n256.npy

#python bin/getSweeps.py -dr ${DR}-south -OF ${outdir}/${DR}/ -rlim None 
#python bin/getBGSBITS.py -IF ${outdir}/${DR}/${DR}-south_sweep_whole.npy -OF ${outdir}/${DR}/${version}/bgstargets-south.npy
#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-south.npy -OF ${outdir}/${DR}/${version}/extra-south_n256.npy

#python bin/getRandoms.py -dr ${DR} -OF ${outdir}/${DR}/
#python bin/getExtra.py -IF ${outdir}/${DR}/${DR}_random_N${Nran}.npy -OF ${outdir}/${DR}/extra_random_N${Nran}_n256.npy -random True


#------------------------------------------------
#run below to generate catalogue and randoms with rmag limit of 21
 
#python bin/getSweeps.py -dr ${DR}-north -OF ${outdir}/${DR}/
python bin/getBGSBITS.py -IF ${outdir}/${DR}/${DR}-north_sweep_whole_r21.0.npy -OF ${outdir}/${DR}/${version}/bgstargets-north.npy
#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-north.npy -OF ${outdir}/${DR}/${version}/extra-north_n256.npy

#python bin/getSweeps.py -dr ${DR}-south -OF ${outdir}/${DR}/
python bin/getBGSBITS.py -IF ${outdir}/${DR}/${DR}-south_sweep_whole_r21.0.npy -OF ${outdir}/${DR}/${version}/bgstargets-south.npy
#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-south.npy -OF ${outdir}/${DR}/${version}/extra-south_n256.npy

#python bin/getRandoms.py -dr ${DR} -OF ${outdir}/${DR}/
#python bin/getExtra.py -IF ${outdir}/${DR}/${DR}_random_N${Nran}.npy -OF ${outdir}/${DR}/extra_random_N${Nran}_n256.npy -random True

#------------------------------------------------


#export mastercat='/global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0/sweep-190p000-200p005.fits'
#export mastercat='/global/cscratch1/sd/qmxp55/sweep_files/dr8_sweep_174.0_186.0_-3.0_2.0_vdebug.npy'
#export mastercat='/global/cscratch1/sd/qmxp55/bgstargets_output/dr8/dr8_sweep_whole_maskbitsources.npy'

#python bin/getExtra.py -IF ${mastercat} -OF ${outdir}/${DR}/extra_dr8_sweep_whole_maskbitsources_n256.npy -random True

#python bin/getBGSBITS.py -IF ${mastercat} -OF ${outdir}/${DR}/${version}/bgstargets-north.npy

#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-north.npy -OF ${outdir}/${DR}/${version}/extra-north_n256.npy

#======================
#get all sweeps objects within a patch for north/south analysis
#python bin/getSweeps.py -dr ${DR}-north -OF ${outdir}/${DR}/ -rlim None -patch 100,285,29,35
#python bin/getSweeps.py -dr ${DR}-south -OF ${outdir}/${DR}/ -rlim None -patch 100,285,29,35

#python bin/getBGSBITS.py -IF ${outdir}/${DR}/${DR}-north_sweep_100.0_285.0_29.0_35.0.npy -OF ${outdir}/${DR}/${version}/bgstargets-north_100.0_285.0_29.0_35.0.npy
#python bin/getBGSBITS.py -IF ${outdir}/${DR}/${DR}-south_sweep_100.0_285.0_29.0_35.0.npy -OF ${outdir}/${DR}/${version}/bgstargets-south_100.0_285.0_29.0_35.0.npy

#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-north_100.0_285.0_29.0_35.0.npy -OF ${outdir}/${DR}/${version}/extra-north_100.0_285.0_29.0_35.0_n256.npy
#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-south_100.0_285.0_29.0_35.0.npy -OF ${outdir}/${DR}/${version}/extra-south_100.0_285.0_29.0_35.0_n256.npy
