# Data processor

## KITTI dataset
### Raw data structure
- Download raw data from [here](http://www.cvlibs.net/datasets/kitti/raw_data.php).
- Download odometry data (color) from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- Copy the ground truth poses from `deepFEPE/datasets/kitti_gt_poses`.
```
`-- KITTI (raw data, odometry sequences, GT poses)
|   |-- raw
|   |   |-- 2011_09_26_drive_0020_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 2011_09_28_drive_0001_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 2011_09_29_drive_0004_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 2011_09_30_drive_0016_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 2011_10_03_drive_0027_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |-- sequences
|   |   |-- 00/
|   |   |-- ...
|   |   |-- 10/
|   |-- poses
|   |   |-- 00.txt
|   |   |-- ...
|   |   |-- 10.txt

```
### Processing command
**``WE ARE NOT FILTERING STATIC FRAMES FOR THE ODO DATASET!``**
Set ``--with_pose`` ``--with_X`` ``--with_sift`` to decide whether to dump pose files, rectified lidar points, and SIFT kps/des and corres.
```
python dump_img_odo_tum.py --dump --dataset_dir /media/yoyee/Big_re/kitti/data_odometry_color/dataset/ \
--with_pose --with_sift \
--dump_root /media/yoyee/Big_re/kitti/kitti_dump/odo_corr_dump_siftIdx_npy_delta1235810_test_0713 \
--num_threads=8  --img_height 376 --img_width 1241 --dataloader_name kitti_seq_loader --cam_id '02'
```

## ApolloScape dataset
### Raw data structure
- Download raw data (Training data, Road11.tar.gz) from [here](http://apolloscape.auto/self_localization.html) or use the following script.
```
python apollo/download.py -h
python apollo/download.py --dataset_dir /media/yoyee/Big_re/apollo/train_seq_1 --if_download
# change the name to Road11.tar.gz
tar zxf Road11.tar.gz
```

### Processing command
```
python dump_img_odo_tum.py  --dump --dataset_dir /media/yoyee/Big_re/apollo/train_seq_1/  --dataloader_name  apollo_train_loader  --with_pose    --with_sift --dump_root /media/yoyee/Big_re/apollo/apollo_dump/train_seq_1/   --num_threads=1  --cam_id 5  --img_height 2710 --img_width  3384 
```

## EuRoC dataset
To be done.
## TUM dataset
To be done.

## Visualize dataset
Refer to https://github.com/eric-yyjau/kitti_instance_RGBD_utils for some code snippets.


