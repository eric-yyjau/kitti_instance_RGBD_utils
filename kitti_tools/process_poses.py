# extract tars


import subprocess
import glob
import argparse
import logging
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foo")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/data/tum/raw_sequences/",
        help="path to dataset",
    )
    args = parser.parse_args()
    print(args)

    if_match_stamps = False
    if_cvt_kitti = True

    folders = glob.glob(f"{args.dataset_dir}/**/")
    for folder in folders:
        # subprocess.run(f"tar -zxf {f}", shell=True, check=True)

        # associate timestamps of poses
        print(folder)
        gt_file = "groundtruth_filter.txt"

        if if_match_stamps:
            subprocess.run(
                f"python associate.py {folder}/rgb.txt {folder}/groundtruth.txt --first_only --max_difference 0.08 --save_file {folder}/{gt_file}",
                shell=True,
                check=True,
            )
        #

        if if_cvt_kitti:
            assert (
                Path(folder) / gt_file
            ).exists(), f"{(Path(folder) / gt_file)} not exists"
            # process files
            logging.info(f"generate kitti format gt pose: {folder}")
            subprocess.run(
                f"evo_traj tum {str(Path(folder)/gt_file)} --save_as_kitti",
                shell=True,
                check=True,
            )  # https://github.com/MichaelGrupp/evo
            filename = gt_file[:-3] + "kitti"
            print(f"cp {filename} {Path(folder)/filename}")
            subprocess.run(
                f"cp {filename} {Path(folder)/filename}", shell=True, check=True
            )
