# extract tars

import subprocess
import glob


if __name__ == "__main__":
    # download
    sequences = [
        "machine_hall/MH_01_easy/MH_01_easy.zip",
        "machine_hall/MH_02_easy/MH_02_easy.zip",
        "machine_hall/MH_03_medium/MH_03_medium.zip",
        "machine_hall/MH_04_difficult/MH_04_difficult.zip",
        "machine_hall/MH_05_difficult/MH_05_difficult.zip",
        "vicon_room1/V1_01_easy/V1_01_easy.zip",
        "vicon_room1/V1_01_easy/V1_01_easy.zip",
        "vicon_room1/V1_02_medium/V1_02_medium.zip",
        "vicon_room1/V1_03_difficult/V1_03_difficult.zip",
        "vicon_room2/V2_01_easy/V2_01_easy.zip",
        "vicon_room2/V2_02_medium/V2_02_medium.zip",
        "vicon_room2/V2_03_difficult/V2_03_difficult.zip",

    ]
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip
    # http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip

    # base_path = "https://vision.in.tum.de/rgbd/dataset/"
    base_path = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"

    if_download = True
    if_untar = True

    if if_download:
        for seq in sequences:
            subprocess.run(f"wget {base_path + seq}", shell=True, check=True)

    if if_untar:
        # unzip
        tar_files = glob.glob("*.zip")
        for f in tar_files:
            subprocess.run(f"unzip {f}", shell=True, check=True)
