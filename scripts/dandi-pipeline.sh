#!/bin/bash
#
# Arguments:
# dandi-pipeline.sh <directory> <stain> <sample>
#
# where <directory> is the path to the channel to be processed
#       <stain> is the name of the stain associated with the channel
#
# The pipeline should be run using the chunglab-stack
#
# Environment variables:
# $ILLUM_CORR - the path to the illumination correction function
# $N_WORKERS - the number of workers to use
# $DANDI_ROOT - root directory of dandiset
# $SUBJECT - the subject that was the source for the sample
# $PSNR - snr for compression (default = 80)
# $TEMPLATE - the JSON template for the sidecars
# $JP2K_ROOT - place to put JPEG 2000 files.
#
RAW_PATH=$1
if [ ! -d "$RAW_PATH" ];
then
  echo "$RAW_PATH does not exist. Exiting."
  exit 1
fi
STAIN=$2
SAMPLE=$3
if [ -z "$PSNR" ]; then export PSNR=80; fi
if [ -z "$TEMPLATE" ]; then
  TEMPLATE=/media/share10/lee/data/2021-01-24_150slab/dataset/sub-mgh191021520/ses-20210124/microscopy/sub-mgh191021520_ses-20210124_sample-150slab_SPIM.json
fi
METADATA_FILE=$(dirname "$RAW_PATH")/metadata.txt
X_STEP_SIZE=$(dandi-metadata get-x-step-size "$METADATA_FILE")
Y_VOXEL_SIZE=$(dandi-metadata get-y-voxel-size "$METADATA_FILE")
if [ -z "$ILLUM_CORR" ]; then
  ILLUM_CORR=/media/share10/lee/illum/ospim1-2021-03-09.tiff
fi
if [ -z "$N_WORKERS" ]; then
  N_WORKERS=48
fi
if [ -z "$JP2K_ROOT" ]; then
  JP2K_ROOT="$RAW_PATH"_jp2k
fi
#
#----- Run parameters
#
set -e
echo "--------- Run parameters --------"
echo "Path:          $RAW_PATH"
echo "Stain:         $STAIN"
echo "Sample:        $SAMPLE"
echo "Metadata file: $METADATA_FILE"
echo "X_STEP_SIZE:   $X_STEP_SIZE"
echo "Y_VOXEL_SIZE:  $Y_VOXEL_SIZE"
echo "ILLUM_CORR:    $ILLUM_CORR"
echo "N_WORKERS:     $N_WORKERS"
echo "DANDI_ROOT:    $DANDI_ROOT"
echo "JP2K_ROOT      $JP2K_ROOT"
echo "--------------------------------"
#
# Loop over each .dcimg file
#
ALL_DCIMGS=`find $RAW_PATH -wholename "**/*.dcimg"`

CHUNK_NUMBER=0
for DCIMG_PATH in $ALL_DCIMGS;
do
  CHUNK_NUMBER=$(($CHUNK_NUMBER + 1))
  dcimg_filename=$(basename "$DCIMG_PATH")
  z=$(echo "$dcimg" | cut -d. -f1)
  rest=$(dirname "$DCIMG_PATH")
  xy=`basename $rest`
  rest=`dirname $rest`
  x=`basename $rest`
  jp2k_path="$JP2K_ROOT"/x/xy/z
  mkdir -p "$jp2k_path"
  #
  # Write the JPEG files
  #
  dcimg2jp2 \
  --input $DCIMG_PATH \
  --output-pattern $jp2k_path/img_%05d.jp2 \
  --n-workers $N_WORKERS \
  --psnr $PSNR

  target_name="$DANDI_ROOT"/$(dandi-metadata target-file \
      --subject $SUBJECT \
      --sample $SAMPLE \
      --source-path $(dirname "$RAW_PATH") \
      --stain $STAIN \
      --chunk $CHUNK_NUMBER )
  #
  # Write the NGFF volume
  #
  mkdir -p $(dirname "$target_name")
  volume_path="$target_name".ngff
  sidecar_path="$target_name".json
  transform_path=${target_name::-4}transforms.json
    dcimg2oblique \
      --n-writers 11 \
      --n-workers $N_WORKERS \
	    --rotate-90 3 \
	    --flip-ud \
	    --input "$jp2k_path"/"img_*.jp2" \
	    --output "$volume_path" \
	     --destripe-method wavelet --sigma1 128 --sigma2 512 --wavelet db5 --crossover 10 \
	     --flat $ILLUM_CORR --dark 100 \
	    --levels 7 \
	    --jp2k \
	    --ngff
	#
	# Write the sidecar.
	#
	dandi-metadata write-sidecar \
	  --template "$TEMPLATE" \
	  --metadata-file "$METADATA_FILE" \
	  --volume "$volume_path" \
	  --volume-format ngff \
	  --stain "$STAIN" \
	  --output "$sidecar_path"
	#
	# Write the transform file
	#
	dandi-metadata write-transform \
	  --input "$DCIMG_PATH" \
	  --output "$transform_path" \
	  --y-voxel-size "$Y_VOXEL_SIZE" \
	  --target-reference-frame slab"$SAMPLE" \
	  $ALL_DCIMGS
	#
	# At this point, the dcimg file could be deleted... not yet though
	#
done