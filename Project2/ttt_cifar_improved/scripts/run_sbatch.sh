#!/bin/bash

common_corruptions=(brightness contrast defocus_blur elastic_transform fog frost gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate shot_noise snow zoom_blur)

# ------------------------

for corrupt in ${common_corruptions[@]}; do
	echo "corrupt: ${corrupt}"
	sbatch --export=CORRUPT=${corrupt} scripts/sbatch_ttt.run
done