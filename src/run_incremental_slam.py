import logging
logging.basicConfig(level=logging.INFO)

import argparse
import torch
import numpy as np
import cv2
import time
import json
import open3d as o3d
from run_optimization import view
from incremental_slam import IncrementalSLAM

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resolution', default=800, type=int, help="View window resolution")
    parser.add_argument('--learning-rate', default=5, type=float,
                        help="ADAM learning rate.")
    parser.add_argument('--cam-dist-weight', default=0.25, type=float,
                        help="Mutual camera distance cost weight.")
    parser.add_argument('--cam-dir-weight', default=0.25, type=float,
                        help="Horizontal camera alinment cost weight.")
    parser.add_argument('--json-recording', help='Read viewing directions from json file.')
    args = parser.parse_args()
    return args


def read_record(record):
    record = json.loads(record)

    frame_id = record['video_frame']
    point_ids = record['point_ids']
    directions = np.zeros([len(record['point_ids']), 3], dtype=np.float32)

    for dir, view_dir in zip(directions, record['directions']):
        dir[:] = np.asarray(view_dir)

    return frame_id, point_ids, directions


def main():
    args = parseargs()
    print('ARGS', args)

    slam = IncrementalSLAM()

    start_frames = 20
    optimization_interval = 5
    optimized_cams = 40
    slam_cam_id = -1
    point_ids_map = {}
    skip = 6
    with open(args.json_recording, 'r') as f:
        for i in range(1200):
            f.readline()

        for record_id, record in enumerate(f):
            if record_id % skip != 0:
                continue
            frame_id, point_ids, directions = read_record(record)
            directions = [directions[i] for i in range(len(point_ids)) if i % 1 == 0]
            point_ids = [point_ids[i] for i in range(len(point_ids)) if i % 1 == 0]

            slam_point_ids = [point_ids_map[i] if i in point_ids_map else None for i in point_ids ]
            if slam_cam_id < start_frames:
                slam_cam_id, slam_point_ids = slam.add_camera(
                    np.asarray([slam_cam_id, 0, 0]), np.zeros(3),
                    directions, slam_point_ids,
                    camera_dist=1, camera_dist_weight=1)
            else:
                last_pos = 2*slam.c_pos[slam_cam_id] - slam.c_pos[slam_cam_id-1]
                last_rot = slam.c_rot[slam_cam_id] + np.random.normal(size=3) * 0.1
                slam_cam_id, slam_point_ids = slam.add_camera(
                    last_pos, last_rot,
                    directions, slam_point_ids,
                    camera_dist=1, camera_dist_weight=0.0000001)

            for s_id, c_id in zip(slam_point_ids, point_ids):
                point_ids_map[c_id] = s_id

            if slam_cam_id > start_frames and slam_cam_id % optimization_interval == 0:
                slam.optimize_both(range(max(0, slam_cam_id - optimized_cams), slam_cam_id + 1))


if __name__ == "__main__":
    main()
