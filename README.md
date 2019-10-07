# kitti_instance_RGBD_utils


## branches
| branch          | description                     | owner | Update |
|-----------------|---------------------------------|-------|--------|
| master          | fork from Jerrypiglet/kitti_instance_RGBD_utils  | youyi |        |
| test_heatmap    | run testing                     | youyi |    |
| dump_data       | dump TUM dataset using kitti scripts | youyi |        |

A data preparation script for instance-wise temporal RGB-D/3D data on KITTI.

Each sample consists of one car sequence with:
- N RGB frames cropped out;
- N reprojected sparse depth from KITTI raw data;
- Style and pose initialization from MV3D dataset for the first frame in the sequence;
- 2D/3D bounding boxes;
- Silhouette(s) for some frames (not for all samples because only 200 frames of KITTI are semantically labelled).

## Requirements    

KITTI raw, depth, semantic, detection dataset; MV3D dataset.

Mayavi(http://docs.enthought.com/mayavi/mayavi/mlab.html) is required to run the code.

Quick installation in **Python2** with Conda:

https://stackoverflow.com/questions/41960672/how-to-install-mayavi-trait-backends

## Usage

Check the ``kitti_lidar_reproj_ipynb.ipynb`` file for a demo.

## Obsolete

> python kitti_lidar.py --fdir [input dir of kitti drive] --outdir [output dir]

Example: 

> conda activate mayavi

> python kitti_lidar.py --fdir /home/rzhu/Documents/kitti_dataset/raw/2011_09_26/2011_09_26_drive_0005_sync/ --outdir ./
