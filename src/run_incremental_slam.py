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
    parser.add_argument('--out-laz', help='Output laz file name.')
    parser.add_argument('--start-frames', default=20, type=int, help='Frames for SLAM initialization.')
    parser.add_argument('--optimization-interval', default=5, type=int, help='Run SLAM optimization each n frames.')
    parser.add_argument('--optimized-cams', default=40, type=int, help='Run pose estimation on last n cameras.')
    parser.add_argument('--geojson', help='Geojson with camera positions. Can be used to show ground truth trajectory.')

    args = parser.parse_args()
    return args


def read_record(record):
    record = json.loads(record)

    frame_id = record['video_frame']
    point_ids = record['point_ids']
    directions = np.zeros([len(record['point_ids']), 3], dtype=np.float32)

    for dir, view_dir in zip(directions, record['directions']):
        dir[:] = np.asarray(view_dir)
        dir[2] *= -1

    return frame_id, point_ids, directions


def read_geo_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)

    c_pos = []
    for camera in data['features']:
        c_id = camera['id']
        c_pos.append(np.asarray(camera['geometry']['coordinates']))

    c_pos = np.stack(c_pos, axis=0)
    return c_pos


def main():
    args = parseargs()
    print('ARGS', args)

    slam = IncrementalSLAM()

    c_pos_gt = read_geo_json(args.geojson) if args.geojson else [None] * 100000
    if c_pos_gt is not None:
        c_pos_gt -= np.mean(c_pos_gt, axis=0, keepdims=True)

    slam_cam_id = -1
    point_ids_map = {}
    skip = 1
    first = True
    with open(args.json_recording, 'r') as f:
        #for i in range(500):
        #    f.readline()

        for record_id, record in enumerate(f):
            if record_id % skip != 0:
                continue
            frame_id, point_ids, directions = read_record(record)
            directions = [directions[i] for i in range(len(point_ids)) if i % 1 == 0]
            point_ids = [point_ids[i] for i in range(len(point_ids)) if i % 1 == 0]

            slam_point_ids = [point_ids_map[i] if i in point_ids_map else None for i in point_ids]
            if slam_cam_id < args.start_frames:
                slam_cam_id, slam_point_ids = slam.add_camera(
                    c_pos_gt[record_id], np.zeros(3),
                    directions, slam_point_ids,
                    camera_dist=1, camera_dist_weight=10000, c_pos_gt=c_pos_gt[record_id])
            else:
                last_pos = 2*slam.c_pos[slam_cam_id] - slam.c_pos[slam_cam_id-1]
                last_rot = slam.c_rot[slam_cam_id] + np.random.normal(size=3) * 0.1
                slam_cam_id, slam_point_ids = slam.add_camera(
                    last_pos, last_rot,
                    directions, slam_point_ids,
                    camera_dist=1, camera_dist_weight=0.0, c_pos_gt=c_pos_gt[record_id])

            for s_id, c_id in zip(slam_point_ids, point_ids):
                point_ids_map[c_id] = s_id

            if slam_cam_id > args.start_frames and slam_cam_id % args.optimization_interval == 0:
                if first:
                    first = False
                    slam.optimize_both(range(0, slam_cam_id + 1))
                else:
                    slam.optimize_both(range(max(0, slam_cam_id - args.optimized_cams), slam_cam_id + 1))
                #slam.optimize_both(range(max(0, slam_cam_id - args.optimized_cams), slam_cam_id + 1), new_stuff=False)
                #if slam_cam_id % (optimization_interval * 10) == 0:
                #    slam.optimize_both(range(slam_cam_id + 1), new_stuff=False, iterations=2000, episode_lr=0.2)
                if args.out_laz:
                    slam.save_laz(args.out_laz)



if __name__ == "__main__":
    main()
