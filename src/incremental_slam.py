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
import laspy
from slam_primitives import CalibratedCamera, CameraMotionSpeedConstraint, CameraPositionConstraint


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
    points[:, 1] *= -1
    cam_trajectory[:, 1] *= -1
    all = cam_trajectory[:, :2]

    if center is None:
        center = (np.max(all, axis=0, keepdims=True) + np.min(all, axis=0, keepdims=True)) / 2
    if size is None:
        size = np.max(np.linalg.norm(all - center, axis=1)) * 1.5

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


def two_line_intersections(p1, u1, p2, u2, min_distance=5, max_distance=500):
    p = p1 - p2
    t2 = (p.dot(u1) * u1.dot(u2) / u1.dot(u1) - p.dot(u2)) / (u1.dot(u2)**2 / u1.dot(u1) - u2.dot(u2))
    t1 = (u2.dot(u1)*t2 - p.dot(u1)) / u1.dot(u1)
    t2 = max(t2, min_distance)
    t1 = max(t1, min_distance)
    t2 = min(t2, max_distance)
    t1 = min(t1, max_distance)
    P1 = p1 + u1 * t1
    P2 = p2 + u2 * t2
    PI = (P1 + P2) / 2.0
    if np.isnan(PI).any():
        PI = p1 + u1
    return PI


class IncrementalSLAM:
    def __init__(self):
        self.min_point_constraint_count = 3
        self.error_power = 1
        self.lr = 0.3
        self.cam_dist_weight = 0.25
        self.resolution =  600
        self.c_max =   1000
        self.p_max =  50000
        self.v_max = 500000

        self.out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                                   (self.resolution, self.resolution))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.camera = CalibratedCamera(self.device, error_power=self.error_power, c_max=self.c_max, v_max=self.v_max)
        self.camera_speed_constraint = CameraMotionSpeedConstraint(self.device, self.camera)
        self.camera_position_contsraint = CameraPositionConstraint(self.device, self.camera)

        self.p_count = 0
        self.p_pos = torch.zeros([self.p_max, 3], dtype=torch.float, device=self.device)

        self.rejected_points = torch.zeros([self.p_max], dtype=torch.bool, device=self.device)
        self.optimized_points = torch.zeros([self.p_max], dtype=torch.bool, device=self.device)

        self.new_cameras = []

    def save_laz(self, file_name):
        relevant_points = torch.logical_and(self.optimized_points, torch.logical_not(self.rejected_points)).cpu().numpy()
        header = laspy.LasHeader(version='1.4', point_format=3)
        las_data = laspy.LasData(header)

        all_positions = np.concatenate([self.p_pos[relevant_points].detach().cpu().numpy(),
                                        self.camera.all_positions().detach().cpu().numpy(),
                                        self.camera_position_contsraint.all_positions().cpu().numpy()], axis=0)

        las_data.x = all_positions[:, 0]
        las_data.y = all_positions[:, 1]
        las_data.z = all_positions[:, 2]
        las_data.classification = [1] * relevant_points.sum() + [2] * self.camera.c_count + [3] * self.camera.c_count
        las_data.write(file_name)

    def triangulate_new_points(self, point_ids):
        with torch.no_grad():
            view_mask = torch.isin(self.camera.views_p_id[:self.camera.p_count], point_ids)
            views, c_pos = self.camera.get_world_views(view_mask)
            views = views.cpu().numpy()
            c_pos = c_pos.cpu().numpy()
            p_ids = self.camera.views_p_id[:view_mask.shape[0]][view_mask].cpu().numpy()

            p_id_lists = defaultdict(list)
            for i in range(p_ids.shape[0]):
                p_id_lists[p_ids[i]].append(i)

            for p_id in p_id_lists:
                if len(p_id_lists[p_id]) > 1:
                    i1 = sorted(p_id_lists[p_id])[0]
                    i2 = sorted(p_id_lists[p_id])[-1]
                    self.p_pos[p_id] = torch.from_numpy(two_line_intersections(c_pos[i1], views[i1], c_pos[i2], views[i2]))
                    if torch.isnan(self.p_pos[p_id]).any():
                        pass

    def add_camera(self, position, rotation, view_vectors, point_ids, c_dist, c_dist_weight,
                   c_pos_gt=None, c_pos_weight=0):
        '''
        :param position: [3] Camera position vector.
        :param rotation: [3] Camera rotation vector in euler angles
        :param view_vectors: view_vectors: [n, 3] unit view direction vectors 
        :param point_ids: point_ids: [n] ids of viewed points - new points should have ID None
        :param c_dist: distance from the last camera
        :param c_dist_weight: weight of the distance - use 0 to ignore the ground truth distance
        :param c_pos_gt: [3] Ground truth camera position.
        :param c_pos_weight: Weight of the ground truth position.
        :return: camera_id, [n] point id for all points
        '''


        for i in range(len(point_ids)):
            p_id = point_ids[i]
            if p_id is None:
                p_id = self.p_count
                self.p_count += 1
                point_ids[i] = p_id

        position = torch.from_numpy(position).float().to(self.device)
        rotation = torch.from_numpy(rotation).float().to(self.device)
        view_vectors = torch.from_numpy(view_vectors).float().to(self.device)
        if c_pos_gt is None:
            c_pos_gt = position
        else:
            c_pos_gt = torch.from_numpy(c_pos_gt).float().to(self.device)
        torch_point_ids = torch.from_numpy(np.asarray(point_ids)).long().to(self.device)

        cam_id = self.camera.add_camera(position, rotation, torch_point_ids, view_vectors)
        self.camera_position_contsraint.add_camera(cam_id, c_pos_gt, weight=c_pos_weight)
        self.camera_speed_constraint.add_camera(cam_id, c_dist, weight=c_dist_weight)

        self.new_cameras.append(cam_id)
        return cam_id, point_ids

    def problem_loss(self, c_pos, c_rot, c_i, p_pos, p_i, view_dir,
                     c_dist, c_pos_gt, c_dist_weight):

        dist_cost = self.camera_speed_constraint.loss()
        pos_cost = self.camera_position_contsraint.loss()
        reprojection_loss = self.camera.loss(self.p_pos)

        opt = dist_cost + pos_cost + reprojection_loss
        return opt

    #def remove_underground_points(self, optimized_point_ids):
    #    for p_id in set(optimized_point_ids) - self.rejected_points:
    #        if self.p_pos[p_id, 2] < self.c_pos[list(self.point_cam_map[p_id]), 2].min() - 10:
    #            self.rejected_points.add(p_id)

    #def remove_distant_points(self, optimized_point_ids, max_distance=100):
    #    for p_id in set(optimized_point_ids) - self.rejected_points:
    #        if np.linalg.norm(self.p_pos[p_id] - np.mean(self.c_pos[list(self.point_cam_map[p_id])], axis=0)) > max_distance:
    #            self.rejected_points.add(p_id)

    def triangulate(self):
        with torch.no_grad():
            point_ids = torch.arange(0, self.p_count, device=self.device)
            view_mask = torch.isin(self.camera.views_p_id[:self.camera.p_count], point_ids)
            views, c_pos = self.camera.get_world_views(view_mask)
            p_ids = self.camera.views_p_id[:view_mask.shape[0]][view_mask].cpu().numpy()

            p_id_lists = defaultdict(list)
            for i in range(p_ids.shape[0]):
                p_id_lists[p_ids[i]].append(i)

            for p_id in p_id_lists:
                if len(p_id_lists[p_id]) > 1:
                    i1 = sorted(p_id_lists[p_id])[len(p_id_lists[p_id]) // 2]
                    self.p_pos[p_id] = c_pos[i1] + 10 * views[i1]
        #self.triangulate_new_points(torch.arange(0, self.p_count, device=self.device))
        pass

    def optimize_both(self, optimized_cam_ids, new_cams=True, iterations=150, episode_lr=1, show_interval=50):
        last_time = time.time()
        if new_cams:
            self.max_iter = iterations
            self.init_iter = 20
            lr_schedule = (np.linspace(0.001, 0.000001, self.init_iter) ** 0.5 * episode_lr).tolist() \
                          + ((np.linspace(0.0001, 1, self.init_iter) ** 0.5) * self.lr * episode_lr).tolist() \
                          + ((np.linspace(1, 0.0001, self.max_iter - 2 * self.init_iter) ** 0.4) * self.lr * episode_lr).tolist()
        else:
            self.max_iter = iterations
            self.init_iter = 0
            lr_schedule = ((np.linspace(0.0001, 1, self.init_iter) ** 0.5) * self.lr * episode_lr).tolist()\
                          + ((np.linspace(1, 0.0001, self.max_iter - self.init_iter) ** 0.4) * self.lr * episode_lr).tolist()

        with torch.no_grad():
            view_point_ids = self.camera.all_view_point_ids()
            point_counts = torch.bincount(view_point_ids, minlength=self.p_count)
            unwanted_points = torch.logical_or(point_counts < self.min_point_constraint_count, self.rejected_points[:self.p_count])
            new_points = torch.logical_and(torch.logical_not(unwanted_points), torch.logical_not(self.optimized_points[:self.p_count]))
            print('NEW points', new_points.sum())
            self.optimized_points[:self.p_count] = torch.logical_or(self.optimized_points[:self.p_count], new_points)

            if new_cams:
                new_cameras = np.asarray(self.new_cameras)
                new_cameras = torch.from_numpy(new_cameras).long().to(self.device)
                self.camera.prepare_optimization(new_cameras, None, unwanted_points.nonzero()[:, 0])
            else:
                optimized_cam_ids = np.asarray(optimized_cam_ids)
                optimized_cam_ids = torch.from_numpy(optimized_cam_ids).long().to(self.device)
                self.camera.prepare_optimization(optimized_cam_ids, None, unwanted_points.nonzero()[:, 0])

        if new_cams:
            params = [self.camera.c_pos, self.camera.c_rot]
            self.camera.c_pos.requires_grad = True
            self.camera.c_rot.requires_grad = True
            self.p_pos.requires_grad = False
        else:
            params = [self.camera.c_pos, self.camera.c_rot, self.p_pos]
            self.camera.c_pos.requires_grad = True
            self.camera.c_rot.requires_grad = True
            self.p_pos.requires_grad = True
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.90, 0.99))

        print(f'Init time: {time.time() - last_time:.3f}')
        last_time = time.time()

        for i in range(self.max_iter):
            for g in optimizer.param_groups:
                g['lr'] = lr_schedule[i]

            optimizer.zero_grad()
            loss = self.camera.loss(self.p_pos) + self.camera_position_contsraint.loss() + self.camera_speed_constraint.loss()
            loss.backward()
            self.camera.mask_gradients()
            optimizer.step()
            loss = loss.cpu().item()

            with torch.no_grad():
                if new_cams and i == self.init_iter:
                    self.triangulate_new_points(new_points.nonzero()[:, 0])
                    params = [self.camera.c_pos, self.camera.c_rot, self.p_pos]
                    self.camera.c_pos.requires_grad = True
                    self.camera.c_rot.requires_grad = True
                    self.p_pos.requires_grad = True
                    optimizer = torch.optim.Adam(params, lr=lr_schedule[i], betas=(0.90, 0.99))

            if i % show_interval == 0:
                print(f'{i}, {loss}, {show_interval / (time.time() - last_time):.1f}')

                #for c in c_rot.detach().cpu().numpy()[:self.c_count]:
                #    print(c * self.rot_scale * 180 / np.pi)
                #print()
                #print('==============================')

                img, _, _ = view(
                    self.camera.c_pos[optimized_cam_ids].detach().cpu().numpy(), self.p_pos[self.optimized_points][-4000:].detach().cpu().numpy(), None,
                    resolution=self.resolution, p_i=self.camera.opt_p_id.detach().cpu().numpy())
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
                '''elif key == ord('s'):
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
                    '''
                last_time = time.time()

        '''view_loss = view_loss.detach().cpu().numpy()
        p_i_cpu = p_i_t.detach().cpu().numpy()
        errors = np.asarray([np.median(view_loss[p_i_cpu == p_id]) ** (1.0/self.error_power) for p_id in optimized_points_new_id])
        if self.use_dot_product:
            errors = np.arccos(1 - errors) * 180 / np.pi
        else:
            errors = np.arcsin(errors) * 180 / np.pi
        #print(' '.join([f'{e}' for e in errors[-100:]]))
        rejected_points = errors > 1
        print(f'Rejected: {np.mean(rejected_points) * 100:.2f}% --- {np.sum(rejected_points)}/{rejected_points.shape[0]}')
        self.rejected_points |= set(np.asarray(optimized_point_ids)[rejected_points].tolist())
        self.c_pos[:self.c_count] = c_pos.detach().cpu().numpy()
        self.c_rot[:self.c_count] = c_rot.detach().cpu().numpy()
        self.p_pos[optimized_point_ids] = p_pos.detach().cpu().numpy()

        self.optimized_cams |= set(optimized_cam_ids)'''

        #cam_pos_error = np.linalg.norm(self.camera.c_pos[max(0, self.camera.c_count-20):self.camera.c_count] - self.camera_position_contsraint.positions[max(0, self.camera.c_count-20):self.c_count], axis=1)
        #cam_pos_error = np.mean(cam_pos_error)
        #self.new_cameras = []
        #print(f'Camera position error at camera {self.c_count}: {cam_pos_error}')
        #self.remove_underground_points(optimized_point_ids)
        #self.remove_distant_points(optimized_point_ids, max_distance=100)

        #for c in self.c_rot[:self.c_count]:
        #    print(c * self.rot_scale * 180 / np.pi)

    def optimize_cameras(self, cam_id):
        pass

    def optimize_points(self, cam_id):
        pass


