import os
import os.path as osp
from argparse import Namespace


def create_if_not_exist(dirpath: str):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)


def create_dirs(args: Namespace):
    create_if_not_exist(args.checkpoints_dir)
    create_if_not_exist(args.log_dir)
