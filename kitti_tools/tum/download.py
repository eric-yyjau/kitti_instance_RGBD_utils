# extract tars

import subprocess
import glob
tar_files = glob.glob("*.tgz")
for f in tar_files:
    subprocess.run(f"tar -zxf {f}", shell=True, check=True)
