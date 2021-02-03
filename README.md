# spimstitch
Stitch images taken on a diagonal using the SPIM microscope

## Usage

The easiest way to use the pipeline is to run the 
`spimstitch-pipeline.sh` script. Pip installs this so it is runnable.
These are easy directions for use in the Chung lab,
 e.g. for paths, `/path-to/root-dir/Ex_488_Em_1` and
`/path-to/root-dir/Ex_562_Em_2` and a standard 4x objective with
Y voxel size of 1.8 µm and X step size 2.0 µm (don't type the dollar
sign):
```bash
$ cd /path-to/root-dir
$ source /home/build/anaconda3/bin/activate chunglab-stack
$ spimstitch-pipeline Ex_488_Em_1
$ spimstitch-pipeline Ex_562_Em_1
```

You can set the Y voxel size and X step size for the script. This
is done with environment variables. For instance, for a X step size
of 2.56 µm and Y voxel size of 3.62µm:

```bash
$ export X_STEP_SIZE=2.56
$ export Y_VOXEL_SIZE=3.62
$ cd /path-to/root-dir
...
```
## Illumination correction and destriping
The pipeline will automatically create an illumination correction image
by default. To override this behavior, set the environment variable,
"ILLUM_CORR" to the path to the illumination correction file. For instance,

```bash
$ export ILLUM_CORR=/path-to/Ex_488_Em_1-illuc.tiff
...
```
The default filename for the illumination correction function is the
channel name + "-illuc.tiff" (as seen above)

There are two methods for destriping in the pipeline, wavelet and
lightsheet. "lightsheet" is used by default, but if you want to use
wavelet (which preserves the autofluorescence signal), you can define
the "USE_WAVELETS" environment variable to choose that method. You should
also specify a background value that will be used as the darkfield value
in destriping. A typical value is somewher between 40 and 200.

For wavelet:
```bash
$ export USE_WAVELETS=1
$ export BACKGROUND=100 # for instance
```

## Alignment

The pipeline recalculates the Y_VOXEL_SIZE by default. This value is
applied to all subvolumes when stitching. If you are processing multiple
channels for a pipeline, you should use the alignment from the first
for subsequent channels. The alignment is stored in a file whose name
is the channel name + "-align.json" (for instance "Ex_488_Em_1-align.json").

It's best not to export this value to ensure that it doesn't inadvertently
get applied to an unrelated volume. An example series of invocations:

```bash
$ spimstitch-pipeline.sh Ex_488_Em_1
$ ALIGN_FILE=Ex_488_Em_1-align.json spimstitch-pipeline.sh Ex_562_Em_2
```

## Individual commands

### dcimg2tif

**dcimg2tif** converts a .dcimg file to a .tiff stack.

Usage:
```bash
dcimg2tif \
  --input <dcimg-file> \
  --output-pattern <output-pattern> \
  [--compression <compression>] \
  [--n-workers <n-workers>] \
  [--rotate-90 <rotate-90>] \
  [--flip-ud] \
  [--start <start>] \
  [--stop <stop>]
```

where

* **dcimg-file** is the name of the .dcimg file to be converted

* **output-pattern** is the pattern for filenames for the .tiff files.
  The z-index of each tiff file is substituted in the pattern, for
  instance, "img_%05d.tiff".

* **compression** is the tiff file compression level: 0-9, default=3

* **n-workers** is the number of worker processes to use, default is
  a single process. **dcimg2tif** is I/O bound, so it's generally
  inefficient to use all cores in a machine and a number between
  6 and 20 is probably a good choice.
  
* **rotate-90** the number of 90 degree clockwise rotations of each
  plane. For the oblique spim machines in the Chung Lab, this number
  should be "3". Default is 0.
  
* **flip-ud** if present, this flips the image in the Y direction after
  rotating. This switch should be present for the oblique spim
  machines in the Chung Lab.
  
* **start** The starting frame to be extracted. Defaults to 0

* **stop** One past the last frame to be extracted. Defaults to
  the number of frames in the .dcimg file.
  
### stack2oblique

The stack2oblique command converts a stack of .tiff files to
an de-obliqued precomputed volume, accounting for the 45° tilt of
the camera relative to the stage motion.

Usage:

```bash
stack2oblique \
  --input <input-pattern> \
  --output <output-path> \
  [--levels <levels>] \
  [--log-level <log-level>] \
  [--n-writers <n-writers>] \
  [--n-workers <n-workers>] \
```

where

* **input-pattern** is a glob expression to collect all of the stack
  frames, for instance, "Ex_488_Em_1_destriped/img*.tiff" (be sure
  to put the expression in quotes on the command-line)
  
* **output-path** is the path to the precomputed directory for the
  output volume.
  
* **levels** is the number of Neuroglancer pyramid levels to be created.
  A single level (1) is sufficient for use in **oblique2stitched**,
  but the default (5) may be better if you want to view the
  intermediate volume in Neuroglancer.
  
* **log-level** is the level for logging output. Possible values are
  "DEBUG", "INFO", "WARNING" and "ERROR". The default is "WARNING".
  
* **n-writers** is the number of writer processes to use. The default
  is 12, unless your computer has fewer CPUs.
  
* **n-workers** is the number of worker processes to use for reading
  files. The default is 12, unless your computer has fewer CPUs.
  
### oblique2stitched

The **oblique2stitched** command converts a group of oblique volumes
into a single stitched volume. The oblique volumes must be organized
in a hierarchy:

```text
/<x>
   /<x>_<y>
     /<z>
```
where <x>, <y> and <z> are the x, y and z coordinates of the
start of the oblique volumes in 10ths of a micron.

Usage:

```bash
oblique2stitched \
  --input <input-path> \
  --output <output-path> \
  [--levels <levels>] \
  [--log-level <log-level>] \
  [--n-writers <n-writers>] \
  [--n-workers <n-workers>] \
  [--silent] \
  [--x-step-size <x-step-size>] \
  [--y-voxel-size <y-voxel-size>] \
  [--z-offset <z-offset>] \
  [--output-size <output-size>] \
  [--output-offset <output-offset>]
```

where

* **input-path** is the path to the directory hierarchy of
  Neuroglancer subvolumes to be stitched
  
* **output-path** is the path to the Neuroglancer volume to be created.

* **levels** is the number of Neuroglancer pyramid levels to be created.
  The default is 5.
  
* **log-level** is the level for logging, one of "DEBUG", "INFO",
  "WARNING" or "ERROR". The default is "WARNING".
  
* **n-writers** is the number of writer processes to use. The default
  is the lesser of 12 or the number of CPUs on the computer.
  
* **n-workers** is the number of worker processes to use. The default
  is the lesser of 12 or the number of CPUs on the computer.
  
* **silent** if present will suppress printing of the progress bar

* **x-step-size** is the X step size for the stage in microns

* **y-voxel-size** is the size of a voxel for the CCD camera in microns

* **z-offset** is the offset between subvolumes in the Z direction in
  pixels.
  
* **output-size** is the size of the volume to create as x,y,z in
  pixels. The default is the entire volume.
  
* **output-offset** is the offset of the written volume with respect
  to the subvolumes. The format is x,y,z (in pixels). The default is
  0,0,0
  
### oblique-illum-corr

Compute an illumination correction image from one or more .dcimg files.
The resulting image can be used as the "--flat" input to pystripe.

The algorithm is as follows:

* Create a histogram at every pixel of the values from a subset
  of the frames in all of the .dcimg files that are above the
  background value
  
* At every pixel, take the value at a percentile of the histogram.
  This percentile should be fairly high (95% to 99.9%) to collect
  true foreground pixels (e.g. pixels within cells). This is
  the intermediate image.
  
* Fit the intermediate image to the function, 
  **A** *x*²+ **B** *x* + **C** *y* using a RANSAC estimator.

Usage:

```bash
oblique-illum-corr \
  --output <output-file> \
  [--intermediate-output <intermediate-output-file>] \
  [--n-frames <n-frames>] \
  [--n-bins <n-bins>] \
  [--values-per-bin <values-per-bin>] \
  [--min-samples <min-samples>] \
  [--percentile <percentile>] \
  [--background <background>] \
  [--rotate-90 <rotate-90>] \
  [--flip-ud] \
  <dcimg-file> [<dcimg-file>...]
```

where
* **output-file** is the name of the .tiff file to be written

* **intermediate-output** is the name of a .tiff file that holds
  the values at the percentile of the per-pixel histogram.
  
* **n-frames** is the number of frames to select at random from among
  all the frames of all the dcimg files. The default is all of them.
  
* **n-bins** is the number of bins per pixel in the histogram. This
  number should be high enough to capture the dynamic range, but
  low enough not to use too much memory. The memory consumed is
  4 * image-width * image-height * n-bins. The default is 1024.
  
* **values-per-bin** is the number of intensity values per bin. The
  dynamic range of the histogram is n-bins * values-per-bin* with
  values above this being clipped. For instance, if n-bins is 4
  and values-per-bin is 1024, the dynamic range is 0 to 4095.
  
* **min-samples** is the minimum number of samples to take in each
  RANSAC round. The default is 20.
  
* **percentile** is the percentile value to take from the histogram
  of pixel values. For instance, a percentile of "95" will take the
  95%th brightest value at each pixel. The default is 95.
  
* **background** is the background cutoff. Pixels aren't included in
  the histogram if they fall below this number. The default is 150.
  
* **rotate-90** the number of 90 degree clockwise rotations of each
  plane. For the oblique spim machines in the Chung Lab, this number
  should be "3". Default is 0.
  
* **flip-ud** if present, this flips the image in the Y direction after
  rotating. This switch should be present for the oblique spim
  machines in the Chung Lab.
  
* **dcimg-file** is the path to a .dcimg file to include in the
  calculation. Multiple .dcimg files can be specified on the command
  line.
  
### oblique-align

**oblique-align** estimates a y pixel size by aligning subvolumes.
The algorithm is to find bright points at random in the overlapping
regions between subvolumes and perform a gradient descent of the
Pearson correlation coefficient between the two overlapping regions.

The result is the median value among all alignment estimates that
have a final Pearson correlation coefficient greater than a cutoff
value. The estimates and the result are written to a .json file.
The voxel size can be read out of this file using the command:

```bash
Y_VOXEL_SIZE=`python -c "import json;print(json.load(open('"<align-file>"'))['voxel_size'])"`
```

Usage:

```bash
oblique-align \
  --input <precomputed-path> \
  --output <align-file> \
  [--voxel-size <y-voxel-size>] \
  [--x-step-size <x-step-size>] \
  [--is-oblique] \
  [--n-cores <n-workers>] \
  [--sigma <sigma>] \
  [--sample-count <sample-count>] \
  [--window-size <window-size>] \
  [--blob-detection-window-size <blob-detection-window-size>] \
  [--border-size <border-size>] \
  [--min-correlation <min-correlation>]
```

where

* **precomputed-path** is the root path to the precomputed volumes.
  All precomputed subvolumes in all subdirectories under this will be
  used to compute the estimate. See [oblique2stitched](#oblique2stitched)
  for details on directory layout.
  
* **align-file** is the JSON file containing the calculated alignment.

* **y-voxel-size** the nominal y voxel size. The alignment will improve
  upon this value.
  
* **x-step-size** the x step size in microns.

* **is-oblique** this flag should be specified for volumes created by
  **stack2oblique**.
  
* **n-cores** the number of processes to use when doing the gradient
  descents.
  
* **sigma** the smoothing sigma for the blob detector and for the
  image fed into the gradient descent. The value is in microns. The
  default is 2.5
  
* **sample-count** the number of samples to be taken per overlapped
  region. The default is 20.
  
* **window-size** the window size for the Pearson correlation calculation
  as "x,y,z". The default is 21,21,21. All values must be odd.
  
* **blob-detection-window-size** the window size for the blob detector.
  Random points are chosen in the overlap region and blobs are
  detected within this window around the points. The format is x,y,z
  and the default is 64,64,64.
  
* **border-size** is a border to each side of the target for the
  gradient descent. A smaller number fetches fewer pixels initially,
  but a larger number fetches more pixels, allowing for a longer
  travel along the gradient before fetching more pixels. The format
  is x,y,z and the default is 10,10,10.
  
* **min-correlation** is the minimum allowed Pearson correlation
  coefficient. All estimates whose correlation coefficient is below
  this value are discarded. The default is .95.
  
## oblique-deskew

This command adjusts for large differences between x-step-size (times sqrt(2))
and y-voxel-size. *oblique2stitched* will build a volume whose X-Z plane is
square, even though the data shape is a parallelogram, resulting in a lengthening
along the X-Z axis and a truncation along the X+Z axis. *oblique-deskew* builds
a new volume with this truncation corrected.

Usage:
```bash
oblique-deskew \
    --input <input-path> \
    --output <output-path> \
    --x-step-size <x-step-size> \
    --y-voxel-size <y-voxel-size> \
    [--levels <levels>] \
    [--n-cores <n-cores>] \
    [--n-writers <n-writers>]
```

where

* *input-path* is the path to the precomputed volume, e.g. as output by
  *oblique2stitched*
  
* *output-path* is the path to the blockfs Neuroglancer volume to be created.

* *x-step-size* is the stepper size in the X direction in microns

* *y-voxel-size* is the size of a pixel on the camera in microns

* *levels* is the number of pyramid levels to be written to the output. The
  default is 5.
  
* *n-cores* is the number of cores to use to read and assemble blocks. The default
  is the number of CPUs in the server
  
* *n-writers* is the number of writer processes to use when writing the output.
  The default is either 11 or the number of CPUs, whichever is lower.
