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
    parser.add_argument('--opt-iterations', default=250, type=int, help='Output laz file name.')
    parser.add_argument('--start-frames', default=20, type=int, help='Frames for SLAM initialization.')
    parser.add_argument('--skip-first-frames', default=0, type=int, help='How many frames to skip at the beginning of the video sequence.')
    parser.add_argument('--frame-rate-subsampling', default=1, type=int, help='Take every n(th) frame from video sequence.')
    parser.add_argument('--optimization-interval', default=5, type=int, help='Run SLAM optimization each n frames.')
    parser.add_argument('--optimized-cams', default=40, type=int, help='Run pose estimation on last n cameras.')
    parser.add_argument('--geojson', help='Geojson with camera positions. Can be used to show ground truth trajectory.')
    parser.add_argument('--optimal-camera-distance', type=float, help='If specify, try to drop some input frames to maintain this distance between cameras.')
    parser.add_argument('--maximum-skip-frames', default=20, type=int, help='Maximum number of frames that can be skipped when useing --optimal-camera-distance.')
    parser.add_argument('--point-subsampling', default=1, type=int, help='Reduction of points by this factor.')

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
    if args.geojson:
        c_pos_gt -= np.mean(c_pos_gt, axis=0, keepdims=True)

    slam_cam_id = -1
    point_ids_map = {}
    first = True
    last_pos = None
    with open(args.json_recording, 'r') as f:
        for i in range(args.skip_first_frames):
            f.readline()

        for record_id, record in enumerate(f):
            if record_id % args.frame_rate_subsampling != 0:
                continue

            frame_id, point_ids, directions = read_record(record)
            if args.point_subsampling > 1:
                directions = [directions[i] for i in range(len(point_ids)) if i % args.point_subsampling == 0]
                point_ids = [point_ids[i] for i in range(len(point_ids)) if i % args.point_subsampling == 0]

            print(record_id, frame_id)
            slam_point_ids = [point_ids_map[i] if i in point_ids_map else None for i in point_ids]

            if slam_cam_id < args.start_frames:
                if c_pos_gt[record_id] is not None:
                    cam_pos = c_pos_gt[slam_cam_id+1]
                else:
                    cam_pos = np.asarray([slam_cam_id+1, 0, 0])

                if last_pos is None:
                    last_pos = cam_pos

                camera_dist = np.linalg.norm(last_pos-cam_pos)

                slam_cam_id, slam_point_ids = slam.add_camera(
                    cam_pos, np.zeros(3),
                    directions, slam_point_ids,
                    c_dist=camera_dist, c_dist_weight=0.00001, c_pos_gt=c_pos_gt[record_id], c_pos_weight=0)
                last_pos = slam.camera.c_pos[slam_cam_id].detach().cpu().numpy()
            else:
                new_pos = (2*slam.camera.c_pos[slam_cam_id] - slam.camera.c_pos[slam_cam_id-1]).detach().cpu().numpy()
                last_rot = slam.camera.c_rot[slam_cam_id].detach().cpu().numpy() + np.random.normal(size=3) * 0.01
                camera_dist = np.linalg.norm(last_pos - cam_pos)
                slam_cam_id, slam_point_ids = slam.add_camera(
                    new_pos, last_rot,
                    directions, slam_point_ids,
                    c_dist=camera_dist, c_dist_weight=0.0, c_pos_gt=c_pos_gt[record_id], c_pos_weight=0)
                last_pos = slam.camera.c_pos[slam_cam_id].detach().cpu().numpy()

            for s_id, c_id in zip(slam_point_ids, point_ids):
                point_ids_map[c_id] = s_id

            if slam_cam_id == args.start_frames or slam_cam_id > args.start_frames and slam_cam_id % args.optimization_interval == 0:
                if first:
                    first = False
                    slam.triangulate()
                    slam.optimize_both(range(0, slam_cam_id + 1), iterations=1600, new_cams=False)
                else:
                    slam.optimize_both(range(max(0, slam_cam_id - args.optimized_cams), slam_cam_id + 1), iterations=args.opt_iterations)
                #slam.optimize_both(range(max(0, slam_cam_id - args.optimized_cams), slam_cam_id + 1), new_stuff=False)
                #if slam_cam_id % (optimization_interval * 10) == 0:
                #    slam.optimize_both(range(slam_cam_id + 1), new_stuff=False, iterations=2000, episode_lr=0.2)
                if args.out_laz:
                    slam.save_laz(args.out_laz)


if __name__ == "__main__":
    main()
