#!/bin/bash

module load python/3.7-anaconda-2019.07
source deactivate
conda deactivate 
conda activate DESI_BGS_omar

export DR='dr9'
export outdir='/global/cscratch1/sd/qmxp55/bgstargets_output'
export version='0.1.0'
export Nran='1'

mkdir -p ${outdir}/${DR}/${version}


#------------------------------------------------
#run below to generate catalogue and randoms with rmag limit of 21
 
#python bin/getSweeps.py -dr ${DR}-north -OF ${outdir}/${DR}/
#python bin/getBGSBITS.py -IF ${outdir}/${DR}/${DR}-north_sweep_whole_r21.0.npy -OF ${outdir}/${DR}/${version}/bgstargets-north.npy
#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-north.npy -OF ${outdir}/${DR}/${version}/extra-north_n256.npy

#python bin/getSweeps.py -dr ${DR}-south -OF ${outdir}/${DR}/
#python bin/getBGSBITS.py -IF ${outdir}/${DR}/${DR}-south_sweep_whole_r21.0.npy -OF ${outdir}/${DR}/${version}/bgstargets-south.npy
#python bin/getExtra.py -IF ${outdir}/${DR}/${version}/bgstargets-south.npy -OF ${outdir}/${DR}/${version}/extra-south_n256.npy

python bin/getRandoms.py -dr ${DR} -OF ${outdir}/${DR}/ -Nran ${Nran}
python bin/getExtra.py -IF ${outdir}/${DR}/${DR}_random_N${Nran}.npy -OF ${outdir}/${DR}/extra_random_N${Nran}_n256.npy -random True

#get extra cols for SGA V3.0
#python bin/getExtra.py -IF /global/cfs/cdirs/cosmo/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.fits -OF #${outdir}/SGA_regions.npy
