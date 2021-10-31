import logging
logging.basicConfig(level=logging.INFO)

import argparse
import torch
import numpy as np
import cv2
import time
from collections import defaultdict
from pytorch3d import transforms
import open3d as o3d


def view(cam_trajectory, points, camera_point_assignment, resolution=1600, center=None, size=None, relations=False, errors=False, p_i=None, point_colors=None):

    if point_colors is not None:
        pass
    elif p_i is not None:
        max_count = 10
        counts = np.bincount(p_i, minlength=p_i.shape[0])
        counts = np.minimum(counts, max_count)
        colors = [(255 - 255 * i / max_count, 0, 255 * i / max_count) for i in range(max_count + 1)]
        point_colors = [colors[c] for c in counts]
    else:
        point_colors = [(0, 0, 255) for c in p_i]


    cam_trajectory = cam_trajectory.copy()[:, :2]
    points = points.copy()[:, :2]
    all = cam_trajectory[:, :2]

    if center is None:
        center = (np.max(all, axis=0, keepdims=True) + np.min(all, axis=0, keepdims=True)) / 2
    if size is None:
        size = np.max(np.linalg.norm(all - center, axis=1)) * 1.1

    img = np.zeros([resolution, resolution, 3], dtype=np.uint8)
    cam_trajectory = (cam_trajectory - center) / size / 2 + 0.5
    points = (points - center) / size / 2 + 0.5
    cam_trajectory *= resolution
    points *= resolution

    if relations:
        for start, camera_points in zip(cam_trajectory, camera_point_assignment):
            for end in points[camera_points > 0]:
                cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (128, 128, 128))

    else:
        for p, c in zip(points, point_colors):
            cv2.circle(img, (int(p[0]), int(p[1])), 1, c, -1)

    for p in cam_trajectory:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

    return img, center, size


class IncrementalSLAM:
    def __init__(self):
        self.c_count = 0
        self.p_count = 0
        self.v_count = 0
        self.c_max = 20000
        self.p_max = 1000000
        self.v_max = 10000000

        self.c_pos = np.zeros([self.c_max, 3], dtype=np.float32)
        self.c_dist = np.zeros([self.c_max], dtype=np.float32)
        self.c_dist_weight = np.zeros([self.c_max], dtype=np.float32)
        self.c_rot = np.zeros([self.c_max, 3], dtype=np.float32)
        self.p_pos = np.zeros([self.p_max, 3], dtype=np.float32)

        self.cam_point_map = defaultdict(set)
        self.point_cam_map = defaultdict(set)
        self.view_map = {}
        self.view_dir = np.zeros([self.v_max, 3], dtype=np.float32)

        self.rejected_points = set()
        self.optimized_points = set()
        self.optimized_cams = set()

        self.error_power = 0.1
        self.lr = 0.2
        self.cam_dist_weight = 0.25
        self.rot_scale = 0.05
        self.resolution = 1200
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (self.resolution, self.resolution))

    def add_camera(self, position, rotation, view_vectors, point_ids, camera_dist, camera_dist_weight, base_point_distance=10):
        '''

        :param view_vectors: [n, 3] unit view direction vectors
        :param point_ids: [n] ids of viewed points - new points should have ID -1
        :return: camera_id, [n] point id for all points
        '''
        cam_id = self.c_count
        self.c_count += 1

        self.c_pos[cam_id] = position
        self.c_dist[cam_id] = camera_dist
        self.c_dist_weight[cam_id] = camera_dist_weight
        self.c_rot[cam_id] = rotation

        R = transforms.euler_angles_to_matrix(torch.tensor(rotation * self.rot_scale), convention='XYZ').numpy().squeeze()

        for i, view in enumerate(view_vectors):
            p_id = point_ids[i]
            if p_id is None:
                p_id = self.p_count
                self.p_count += 1
                point_ids[i] = p_id
                self.p_pos[p_id] = position + (R @ view_vectors[i].reshape(1, 3).squeeze()) * base_point_distance

            self.view_dir[self.v_count] = view_vectors[i]
            self.view_map[(cam_id, p_id)] = self.v_count
            self.point_cam_map[p_id].add(cam_id)
            self.v_count += 1

        self.cam_point_map[cam_id] = set(point_ids)

        return cam_id, point_ids

    def problem_loss(self, c_pos, c_rot, c_i, p_pos, p_i, view_dir,
                     c_dist, c_dist_weight):

        cam_dist_cost = (torch.sum((c_pos[1:] - c_pos[:-1]) ** 2, axis=1) ** 0.5 - c_dist[:-1]) ** 2
        cam_dist_cost = cam_dist_cost * c_dist_weight[:-1]

        c_pos = c_pos[c_i]
        p_pos = p_pos[p_i]

        c_rot = transforms.euler_angles_to_matrix(c_rot * self.rot_scale, convention='XYZ')
        view_dir = c_rot[c_i].bmm(view_dir.reshape(-1, 3, 1))

        dir = p_pos - c_pos
        dir = dir / torch.norm(dir, dim=1, keepdim=True)
        view_loss = 1.000001 - dir.reshape(-1, 1, 3).bmm(view_dir.reshape(-1, 3, 1))
        view_loss = view_loss ** self.error_power

        opt = torch.sum(view_loss) + self.cam_dist_weight * torch.sum(cam_dist_cost)
        return opt, view_loss

    def optimize_both(self, optimized_cam_ids):
        last_time = time.time()

        optimized_cam_ids = list(optimized_cam_ids)
        optimized_point_ids = set().union(*[self.cam_point_map[i] for i in optimized_cam_ids]) - self.rejected_points

        min_point_constraint_count = 3
        optimized_point_ids = {p_id for p_id in optimized_point_ids if len(self.point_cam_map[p_id]) >= min_point_constraint_count}

        c_dist = torch.tensor(self.c_dist[:self.c_count]).float().to(self.device)
        c_dist_weight = torch.tensor(self.c_dist_weight[:self.c_count]).float().to(self.device)

        c_pos = torch.tensor(self.c_pos[:self.c_count]).float().to(self.device)
        c_rot = torch.tensor(self.c_rot[:self.c_count]).float().to(self.device)
        p_pos = torch.tensor(self.p_pos[:self.p_count]).float().to(self.device)

        c_pos_orig = torch.tensor(self.c_pos[:self.c_count]).float().to(self.device)
        c_rot_orig = torch.tensor(self.c_rot[:self.c_count]).float().to(self.device)
        p_pos_orig = torch.tensor(self.p_pos[:self.p_count]).float().to(self.device)

        c_i = []
        p_i = []
        for cam_id in range(self.c_count):
            optimized_cam_points = sorted(list(self.cam_point_map[cam_id] & optimized_point_ids))
            cam_id = [cam_id] * len(optimized_cam_points)
            c_i += cam_id
            p_i += optimized_cam_points

        v_i = [self.view_map[c, p] for c, p in zip(c_i, p_i)]
        view_dir = torch.tensor(self.view_dir[v_i]).float().to(self.device)

        optimized_point_ids = sorted(list(optimized_point_ids))
        optimized_points_new_id = range(len(optimized_point_ids))
        optimized_points_id_map = {orig_id: new_id for orig_id, new_id in zip(optimized_point_ids, optimized_points_new_id)}
        p_i = [optimized_points_id_map[orig_id] for orig_id in p_i]

        p_pos = p_pos[optimized_point_ids]
        p_pos_orig = p_pos_orig[optimized_point_ids]

        print(f'Cameras: {c_pos.shape[0]}, Points: {p_pos.shape[0]}, Connections: {view_dir.shape[0]}')

        c_i = torch.tensor(c_i).to(self.device)
        p_i = torch.tensor(p_i).to(self.device)

        c_pos.requires_grad = True
        c_rot.requires_grad = True
        p_pos.requires_grad = True

        old_cams = torch.tensor(sorted(list(self.optimized_cams))).long().to(self.device)
        old_points = torch.tensor(sorted([optimized_points_id_map[i] for i in self.optimized_points & set(optimized_point_ids)])).long().to(self.device)
        static_cams = sorted(list(set(range(self.c_count)) - set(optimized_cam_ids)))
        static_cams = torch.tensor(static_cams).long().to(self.device)

        params = [c_pos, c_rot, p_pos]
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.90, 0.99))

        self.max_iter = 850
        self.init_iter = 50
        lr_schedule = [self.lr * 2] * self.init_iter + ((np.linspace(0.0001 * 2, 1, self.max_iter - self.init_iter) ** 0.4) * self.lr).tolist()
        #lr_schedule = ((np.linspace(0.0001 * 2, 1, self.max_iter) ** 0.4) * self.lr).tolist()

        for i in range(self.max_iter):
            for g in optimizer.param_groups:
                g['lr'] = lr_schedule[i]

            optimizer.zero_grad()
            loss, view_loss = self.problem_loss(
                c_pos, c_rot, c_i, p_pos, p_i, view_dir, c_dist, c_dist_weight)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().item()

            with torch.no_grad():
                if self.optimized_cams and i < 50:
                    c_pos[old_cams] = c_pos_orig[old_cams]
                    c_rot[old_cams] = c_rot_orig[old_cams]
                    p_pos[old_points] = p_pos_orig[old_points]
                else:
                    c_pos[static_cams] = c_pos_orig[static_cams]
                    c_rot[static_cams] = c_rot_orig[static_cams]

            if i % 50 == 0:
                print(i, loss)
                img, _, _ = view(
                    c_pos.detach().cpu().numpy()[-100:], p_pos.detach().cpu().numpy(), None,
                    resolution=self.resolution, p_i=p_i.detach().cpu().numpy())
                self.out.write(img)

                cv2.imshow('result', img)

                '''view_loss = view_loss.detach().cpu().numpy()
                p_i_cpu = p_i.detach().cpu().numpy()

                errors = 1 - np.asarray([np.median(view_loss[p_i_cpu == p_id]) ** 10 for p_id in optimized_points_new_id])
                print(errors.min(), errors.max())
                print(' '.join([f'{e}' for e in errors[:100]]))
                errors = np.arccos(errors) * 180 * np.pi
                print(errors.min(), errors.max())
                print(' '.join([f'{e}' for e in errors[:100]]))
                errors = np.minimum(errors, 20)
                errors /= 20
                point_colors = [(int(255 * (1-error)), 0, int(255 * error)) for error in errors]

                img2, _, _ = view(
                    c_pos.detach().cpu().numpy()[-100:], p_pos.detach().cpu().numpy(), None,
                    resolution=self.resolution, point_colors=point_colors)

                cv2.imshow('result2', img2)'''

                key = cv2.waitKey(5)
                if key == 27:
                    break
                elif key == ord('s'):
                    pcd = o3d.geometry.PointCloud()
                    c = c_pos.detach().cpu().numpy()
                    p = p_pos.detach().cpu().numpy()
                    all_positions = np.concatenate([c, p], axis=0)
                    cam_colors = c.copy()
                    point_colors = p.copy()
                    cam_colors[:, 0] = 1
                    cam_colors[:, 1] = 0
                    cam_colors[:, 2] = 0
                    point_colors[:, 0] = 0
                    point_colors[:, 1] = 0
                    point_colors[:, 2] = 1
                    all_colors = np.concatenate([cam_colors, point_colors], axis=0)

                    pcd.points = o3d.utility.Vector3dVector(all_positions)
                    pcd.colors = o3d.utility.Vector3dVector(all_colors)
                    o3d.visualization.draw_geometries([pcd], width=1600, height=1200, point_show_normal=False)

                last_time = time.time()

        view_loss = view_loss.detach().cpu().numpy()
        p_i_cpu = p_i.detach().cpu().numpy()
        errors = 1 - np.asarray([np.median(view_loss[p_i_cpu == p_id]) ** 10 for p_id in optimized_points_new_id])
        errors = np.arccos(errors) * 180 * np.pi
        self.rejected_points |= set(np.asarray(optimized_point_ids)[errors > 7].tolist())

        self.c_pos[:self.c_count] = c_pos.detach().cpu().numpy()
        self.c_rot[:self.c_count] = c_rot.detach().cpu().numpy()
        self.p_pos[optimized_point_ids] = p_pos.detach().cpu().numpy()
        self.optimized_points |= set(optimized_point_ids)
        self.optimized_cams |= set(optimized_cam_ids)

        for c in self.c_rot[:self.c_count]:
            print(c * self.rot_scale * 180 / np.pi)



    def optimize_cameras(self, cam_id):
        pass

    def optimize_points(self, cam_id):
        pass


