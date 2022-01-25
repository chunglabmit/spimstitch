#!/bin/bash

###################
#
# Dragonfly pipeline
#
# This pipeline converts a series of .ims files output by Dragonfly to our
# lab's BlockFS format for display in Neuroglancer. It uses the Terastitcher
# XML file as a guide for how to do the stitching.
#
# Usage:
# cd <ims-directory>
# dragonfly-pipeline <xml-file> <alignment-channel> <n-channels>
#
# where:
#   xml-file is the Terastitcher XML file output by dragonfly
#   alignment-channel is the 1-based index of the channel used for alignment
#   n-channels is the # of channels captured
#
ORIG_XML=$1
ALIGNMENT_CHANNEL=$2
N_CHANNELS=$3
#
# Flags for this script
#
set -e # Exit on error
set -x # Print executed commands
#
# First, perform some massaging of the XML file
#
cp "$ORIG_XML" import.xml
# Set the root for TeraStitcher to this directory
xmlstarlet ed -L -u '//TeraStitcher/stacks_dir/@value' -v "$(pwd)" import.xml
# Reverse the X direction by changing sign on voxel size
X_VOXEL_SIZE="-$(xmlstarlet sel -t -v '//TeraStitcher/voxel_dims/@H' import.xml)"
xmlstarlet ed -L -u '//TeraStitcher/voxel_dims/@H' -v $X_VOXEL_SIZE import.xml
#
# Now run the alignment
#
oblique-align \
  --imaris \
  --terastitcher-xml import.xml \
  --output alignment.json \
  --align-xz \
  --sigma 5 \
  --window-size 51,51,51 \
  --sample-count 100 \
  --min-correlation .97 \
  --channel $ALIGNMENT_CHANNEL
#
# For each channel, output "Channel-$CHANNEL_precomputed_stitched" volume
#
for CHANNEL_NUMBER in $(seq 1 $N_CHANNELS)
do
  CHANNEL_NAME=Channel-"$CHANNEL_NUMBER"_precomputed_stitched
  imaris2stitched \
    --terastitcher-xml import.xml \
    --alignment alignment.json \
    --output $CHANNEL_NAME \
    --channel $CHANNEL_NUMBER \
    --levels 7
done
