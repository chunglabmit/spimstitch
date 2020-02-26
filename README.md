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