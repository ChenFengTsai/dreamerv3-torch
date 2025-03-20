import subprocess

command = "/home/richtsai1103/CRL/dreamerv3-torch/xvfb_run.sh python3 /home/richtsai1103/CRL/dreamerv3-torch/dreamer.py --configs metaworld --task metaworld_pick-place-v2 --logdir ./logdir/metaworld_pick-place-v2/ --device cuda:3"

subprocess.run(command, shell=True)
