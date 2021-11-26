import logging
from hmac import new

logging.basicConfig(level=logging.INFO)

import argparse
import numpy as np
import cv2
from collections import defaultdict
from pytorch3d import transforms
import torch
import json
import time
from scipy import signal, spatial


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-video', required=True,  help="Video to process.")
    parser.add_argument('-o', '--output-file', help="Output json file.")
    parser.add_argument('--frame-rotation', default=0, type=float, help="Rotate the video.")
    parser.add_argument('--frame-speed-file', help="csv with frame, speed (in m/s) values on each line.")
    parser.add_argument('--video-subsampling', default=1, type=int, help="How the input video is subsampled compared to it's original version.")

    args = parser.parse_args()
    return args


class FeatureTracker:
    def __init__(self):
        self.frame_history = []
        self.frame_id = []

        self.bf_interval = 25
        self.bf_overlap = 20
        self.bf_max_error = 2

        self.points = np.zeros([0, 1, 2], dtype=np.float32)
        self.points_speed = np.zeros([0, 1, 2], dtype=np.float32)
        self.point_ids = []
        self.point_history = defaultdict(dict)

        self.point_counter = 0
        self.frame_counter = 0

        self.min_point_distance = 8

        self.feature_params = dict(maxCorners=3000,
                              qualityLevel=0.02,
                              minDistance=self.min_point_distance,
                              blockSize=5)

        self.point_speed_momentum = 0.8

        self.lk_params = {'winSize': (9, 9),
                          'maxLevel': 2,
                          'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    def add_frame(self, img, frame_id):
        self.frame_history.append(img)
        self.frame_id.append(frame_id)

        if len(self.frame_history) == self.bf_interval:
            return self.__run_bf_tracking()
        else:
            return None

    def __density_filter(self, points, frame, max_local_points=2):
        points = points.astype(int).squeeze(1)
        pos = points[:, 1] * frame.shape[1] + points[:, 0]
        density = np.bincount(pos, minlength=frame.shape[0]*frame.shape[1])
        density = density.reshape(frame.shape[0], frame.shape[1])
        kernel = np.ones([self.min_point_distance, 1])
        density = signal.convolve2d(density, kernel, mode='same')
        density = signal.convolve2d(density, kernel.T, mode='same')
        density = density.reshape(frame.shape[0] * frame.shape[1])
        point_score = np.maximum(density[pos] - max_local_points, 0) / (density[pos] + 0.01)
        survivors = np.random.uniform(size=pos.shape[0]) >= point_score
        return survivors

    def __get_initial_speeds(self, p1, s1, p2):
        if p1.shape[0] == 0 or p2.shape[0] == 0:
            return np.zeros(p2.shape)
        else:
            dist = spatial.distance.cdist(p2.reshape(-1, 2), p1.reshape(-1, 2), metric='sqeuclidean')
            nearest = np.argmin(dist, axis=1)
            s2 = [s1[i] for i in nearest]
            return np.stack(s2, axis=0)

    def __run_bf_tracking(self):

        # detect new points in the oldest frame
        mask = np.ones(self.frame_history[0].shape, dtype=np.uint8)
        for x, y in self.point_history[self.frame_id[0]].values():
            cv2.circle(mask, (int(x), int(y)), self.min_point_distance + 2, (0, 0, 0), -1)
        mask[int(mask.shape[0]*0.55):] = 0

        p0 = cv2.goodFeaturesToTrack(self.frame_history[0], mask=mask, **self.feature_params)
        if p0 is None:
            p0 = np.zeros([0, 1, 2], dtype=np.float32)

        # forward track all points through all frames
        forward_points = [np.concatenate([self.points, p0], axis=0)]

        forward_points_speed = [np.concatenate([self.points_speed,
                                                self.__get_initial_speeds(self.points, self.points_speed, p0)], axis=0)]
        last_frame = self.frame_history[0]
        for frame in self.frame_history[1:]:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                last_frame, frame, forward_points[-1], forward_points[-1] + forward_points_speed[-1], **self.lk_params)
            forward_points_speed.append(
                (1 - self.point_speed_momentum) * forward_points_speed[-1] +
                self.point_speed_momentum * (p1 - forward_points[-1]))

            forward_points.append(p1)
            last_frame = frame

        # backward track all points through all frames
        backward_points = [forward_points[-1]]
        backward_points_speed = -forward_points_speed[-1]
        last_frame = self.frame_history[-1]
        for frame in self.frame_history[-1::-1]:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                last_frame, frame, backward_points[-1], backward_points[-1] + backward_points_speed, **self.lk_params)
            backward_points_speed = (1 - self.point_speed_momentum) * backward_points_speed + \
                                   self.point_speed_momentum * (p1 - backward_points[-1])
            backward_points.append(p1)
            last_frame = frame
        backward_points = backward_points[::-1]

        # check returned points if close to origin
        distances = np.sum((forward_points[0].reshape(-1, 2) - backward_points[0].reshape(-1, 2)) ** 2, axis=1) ** 0.5
        survivors = np.logical_and(distances < self.bf_max_error, self.__density_filter(forward_points[0], self.frame_history[0]))

        for i in range(len(forward_points)):
            forward_points[i] = forward_points[i][survivors]

        self.point_ids = list(self.point_ids) + [None] * p0.shape[0]
        self.point_ids = [self.point_ids[i] for i in np.nonzero(survivors)[0]]

        for i in range(len(self.point_ids)):
            if self.point_ids[i] is None:
                self.point_ids[i] = self.point_counter
                self.point_counter += 1
                for frame_points, f_i in zip(forward_points, self.frame_id):
                    self.point_history[f_i][self.point_ids[i]] = frame_points[i, 0].tolist()
            else:
                for frame_points, f_i in zip(forward_points[self.bf_overlap:], self.frame_id[self.bf_overlap:]):
                    self.point_history[f_i][self.point_ids[i]] = frame_points[i, 0].tolist()

        self.points = forward_points[-self.bf_overlap]
        self.points_speed = forward_points_speed[-self.bf_overlap][survivors]

        results = (self.frame_id[:-self.bf_overlap], [self.point_history[f_i] for f_i in self.frame_id[:-self.bf_overlap]])

        self.frame_history = self.frame_history[-self.bf_overlap:]
        self.frame_id = self.frame_id[-self.bf_overlap:]
        return results

from models.matching import Matching
from models.utils import frame2tensor

class SuperGlueTracker:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.y_range = [0.1, 0.64]
        torch.set_grad_enabled(False)
        nms_radius = 4
        keypoint_threshold = 0.003
        max_keypoints = 4000
        sinkhorn_iterations = 20
        match_threshold = 0.4
        superglue = 'outdoor'
        self.config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self.matching = Matching(self.config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']

        self.last_frame_sg_info = None
        self.last_frame_kp_ids = []
        self.last_frame_id = []
        self.keypoint_counter = 0

    def add_frame(self, img, frame_id):
        vertical_offset = int(img.shape[0] * self.y_range[0] + 0.5)
        img = img[vertical_offset:int(img.shape[0] * self.y_range[1] + 0.5)]
        img_tensor = frame2tensor(img, self.device)

        if self.last_frame_sg_info is None:
            pred = self.matching.superpoint({'image': img_tensor})
            self.last_frame_sg_info = {k+'0': pred[k] for k in self.keys}
            self.last_frame_sg_info['image0'] = img_tensor

            self.last_frame_id = frame_id
            self.last_frame_kp_ids = [None] * self.last_frame_sg_info['keypoints0'][0].shape[0]
            return None

        pred = self.matching({**self.last_frame_sg_info, 'image1': img_tensor})
        kpts0 = self.last_frame_sg_info['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        for i in range(matches.shape[0]):
            if self.last_frame_kp_ids[i] is None and matches[i] > -1:
                self.last_frame_kp_ids[i] = self.keypoint_counter
                self.keypoint_counter += 1

        keypoint_results = {self.last_frame_kp_ids[i]: (kp[0], kp[1] + vertical_offset) for i, kp in enumerate(kpts0) if self.last_frame_kp_ids[i] is not None}
        results = ([self.last_frame_id],
                   [keypoint_results])

        self.last_frame_sg_info = {k + '0': pred[k + '1'] for k in self.keys}
        self.last_frame_sg_info['image0'] = img_tensor
        last_frame_kp_ids = [None] * kpts1.shape[0]
        for i, id in enumerate(self.last_frame_kp_ids):
            if id is not None:
                last_frame_kp_ids[matches[i]] = id
        self.last_frame_kp_ids = last_frame_kp_ids
        self.last_frame_id = frame_id

        return results


class ORBTracker:
    def __init__(self):
        self.frame_history = []
        self.frame_id = []
        self.orb = cv2.ORB_create(5000, 1.4, nlevels=4, firstLevel=0, WTA_K=4)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, True)

    def add_frame(self, img, frame_id):
        self.frame_history.append(img)
        self.frame_id.append(frame_id)

        if len(self.frame_history) < 2:
            return None

        kps1, desc1 = self.orb.detectAndCompute(img, None)
        kps0, desc0 = self.orb.detectAndCompute(self.frame_history[-2], None)

        matches = self.matcher.match(desc0, desc1)
        img1 = np.stack([img] * 3, axis=2)
        img0 = np.stack([self.frame_history[-2]] * 3, axis=2)

        matches = sorted(matches, key=lambda m: m.distance)
        final_img = cv2.drawMatches(img0, kps0, img1, kps1, matches[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        #for kp in kps:
        #    cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), (255, 0, 0))
        cv2.imshow('img', final_img)
        key = cv2.waitKey()
        if key == 27:
            exit(-1)

        return None


def get_rot_matrix(yaw, pitch):
    R1 = np.asarray(
        [[np.cos(yaw), -np.sin(yaw), 0],
         [np.sin(yaw), np.cos(yaw), 0],
         [0, 0, 1]])

    R2 = np.asarray(
        [[np.cos(pitch), 0, -np.sin(pitch)],
         [0, 1, 0],
         [np.sin(pitch), 0, np.cos(pitch)]])

    return R2 @ R1


def positions_to_view_direction(points, width, height):
    points = points.copy()

    points[:, 0] = points[:, 0] / width * np.pi * 2 - np.pi
    points[:, 1] = (1 - points[:, 1] / height) * np.pi

    directions = np.zeros([points.shape[0], 3])
    directions[:, 0] = np.cos(points[:, 0]) * np.sin(points[:, 1])
    directions[:, 1] = -np.sin(points[:, 0]) * np.sin(points[:, 1])
    directions[:, 2] = np.cos(points[:, 1])
    #unit_vector = np.zeros([3, 1])
    #unit_vector[0, 0] = 1
    #for p, d in zip(points, directions):
    #    #R = get_rot_matrix(p[0], p[1])
    #    d[...] = np.asarray([np.cos(p[0]) * np.sin(p[1]), -np.sin(p[0]) * np.sin(p[1]), np.cos(p[1])])

    return directions

def get_frame_distances(file_name):
    frame_rate = 30
    last_speed = 0
    last_distance = 0
    distances = []
    with open(file_name, 'r') as f:
        for line in f:
            spd, frm = line.split()
            spd = float(spd)
            frm = int(frm)
            while len(distances) <= frm:
                last_distance = last_distance + last_speed / frame_rate
                distances.append(last_distance)
            last_speed = spd

    return distances


def main():
    args = parseargs()
    print('ARGS', args)

    if args.output_file:
        output_file = open(args.output_file, 'w')
    else:
        output_file = None


    if args.frame_speed_file:
        distances = get_frame_distances(args.frame_speed_file)

    print(distances)

    frame_history = {}
    frame_points = {}
    point_history = defaultdict(list)
    video = cv2.VideoCapture(args.input_video)

    skip_frames = 1

    last_distance = -100
    frame_id = 0
    if frame_id > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    tracker = SuperGlueTracker()
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print('died')
            break

        frame_id += args.video_subsampling
        if frame_id % skip_frames != 0:
            continue
        if last_distance + 0.7 > distances[frame_id]:
            continue
        last_distance = distances[frame_id]

        if args.frame_rotation != 0:
            shift = int(args.frame_rotation * frame.shape[1])
            frame = np.concatenate([frame[:, shift:], frame[:, :shift]], axis=1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        draw_frame = np.stack([frame, frame, frame], axis=2)

        results = tracker.add_frame(frame, frame_id)
        frame_history[frame_id] = draw_frame

        if results is None:
            continue

        for f_id, f_points in zip(results[0], results[1]):
            frame_points[f_id] = f_points.keys()

        for f_id, f_points in zip(results[0], results[1]):
            for p_id in f_points:
                point_history[p_id].append(f_points[p_id])

            draw_frame = frame_history[f_id]
            del frame_history[f_id]

            points = np.zeros([len(frame_points[f_id]), 2])
            for p, p_id in zip(points, frame_points[f_id]):
                p[0] = point_history[p_id][-1][0]
                p[1] = point_history[p_id][-1][1]

            directions = positions_to_view_direction(points, frame.shape[1], frame.shape[0])

            if output_file:
                directions = [(d[0], d[1], d[2]) for d in directions]
                print(json.dumps({'video_frame': f_id, 'point_ids': list(frame_points[f_id]), 'directions': directions}),
                      file=output_file)

            print(f_id, len(frame_points[f_id]), distances[f_id])
            for p_id in frame_points[f_id]:
                if len(point_history[p_id]) > 1:
                    old_pos = point_history[p_id][-2]
                    for new_pos in point_history[p_id][-1:]:
                        a, b = old_pos
                        c, d = new_pos
                        cv2.line(draw_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
                        #cv2.circle(draw_frame, (int(a), int(b)), 1, (0, 0, 255), -1)
                        old_pos = new_pos


                a, b = point_history[p_id][-1]
                cv2.circle(draw_frame, (int(a), int(b)), 2, (255, 0, 0), -1)

            cv2.imshow('vid', draw_frame)
            key = cv2.waitKey(3)
            if key == 27:
                exit(-1)


if __name__ == "__main__":
    main()
