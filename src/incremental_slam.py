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
    return PI

class IncrementalSLAM:
    def __init__(self):
        self.c_count = 0
        self.p_count = 0
        self.v_count = 0
        self.c_max = 20000
        self.p_max = 5000000
        self.v_max = 50000000

        self.c_pos = np.zeros([self.c_max, 3], dtype=np.float32)
        self.c_pos_gt = np.zeros([self.c_max, 3], dtype=np.float32)
        self.c_pos_weight = np.zeros([self.c_max], dtype=np.float32)
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

        self.error_power = 1
        self.lr = 0.3
        self.cam_dist_weight = 0.25
        self.rot_scale = 0.05
        self.resolution = 1200
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (self.resolution, self.resolution))

        self.use_dot_product = False

    def save_laz(self, file_name):
        relevant_points = sorted(list(self.optimized_points - self.rejected_points))
        header = laspy.LasHeader(version='1.4', point_format=3)
        las_data = laspy.LasData(header)
        las_data.x = np.concatenate((self.p_pos[relevant_points, 0], self.c_pos[:self.c_count, 0], self.c_pos_gt[:self.c_count, 0]), axis=0)
        las_data.y = np.concatenate((self.p_pos[relevant_points, 1], self.c_pos[:self.c_count, 1], self.c_pos_gt[:self.c_count, 1]), axis=0)
        las_data.z = np.concatenate((self.p_pos[relevant_points, 2], self.c_pos[:self.c_count, 2], self.c_pos_gt[:self.c_count, 2]), axis=0)
        las_data.classification = [1] * len(relevant_points) + [2] * self.c_count + [3] * self.c_count
        las_data.R = [255] * len(relevant_points) + [0] * self.c_count   + [255] * self.c_count
        las_data.G = [255] * len(relevant_points) + [255] * self.c_count + [0] * self.c_count
        las_data.B = [255] * len(relevant_points) + [0] * self.c_count   + [0] * self.c_count
        las_data.write(file_name)



    def triangulate_new_point(self, p_id, c_pos, c_rot):
        point_cameras = sorted(list(self.point_cam_map[p_id]))
        c1 = point_cameras[0]
        c2 = point_cameras[-1]
        R1 = transforms.euler_angles_to_matrix(torch.tensor(c_rot[c1] * self.rot_scale),
                                               convention='XYZ').numpy().squeeze()
        R2 = transforms.euler_angles_to_matrix(torch.tensor(c_rot[c2] * self.rot_scale),
                                               convention='XYZ').numpy().squeeze()
        u1 = (R1 @ self.view_dir[self.view_map[c1, p_id]].reshape(3, 1)).squeeze()
        u2 = (R2 @ self.view_dir[self.view_map[c2, p_id]].reshape(3, 1)).squeeze()

        return two_line_intersections(c_pos[c1], u1, c_pos[c2], u2)

    def add_camera(self, position, rotation, view_vectors, point_ids, camera_dist, camera_dist_weight,
                   base_point_distance=42, c_pos_gt=None):
        '''

        :param view_vectors: [n, 3] unit view direction vectors
        :param point_ids: [n] ids of viewed points - new points should have ID -1
        :return: camera_id, [n] point id for all points
        '''
        cam_id = self.c_count
        self.c_count += 1

        self.c_pos[cam_id] = position
        if c_pos_gt is not None:
            self.c_pos_gt[cam_id] = c_pos_gt
        else:
            self.c_pos_gt[cam_id] = position

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
                self.p_pos[p_id] = position + (R @ view_vectors[i].reshape(3, 1)).squeeze() * base_point_distance

            self.view_dir[self.v_count] = view_vectors[i]
            self.view_map[(cam_id, p_id)] = self.v_count
            self.point_cam_map[p_id].add(cam_id)
            self.v_count += 1

        self.cam_point_map[cam_id] = set(point_ids)

        return cam_id, point_ids

    def problem_loss(self, c_pos, c_rot, c_i, p_pos, p_i, view_dir,
                     c_dist, c_pos_gt, c_dist_weight):

        #cam_dist_cost = (torch.sum((c_pos[1:] - c_pos[:-1]) ** 2, axis=1) ** 0.5 - c_dist[:-1]) ** 2
        #cam_dist_cost = cam_dist_cost * c_dist_weight[:-1]

        cam_pos_cost = torch.norm(c_pos - c_pos_gt, dim=1)**2 * c_dist_weight

        c_pos = c_pos[c_i]
        p_pos = p_pos[p_i]

        c_rot = transforms.euler_angles_to_matrix(c_rot * self.rot_scale, convention='XYZ')
        view_dir = c_rot[c_i].bmm(view_dir.reshape(-1, 3, 1))

        c_p_dir = p_pos - c_pos
        if self.use_dot_product:
            c_p_dir = c_p_dir / torch.norm(c_p_dir, dim=1, keepdim=True)
            view_loss = 1.000001 - c_p_dir.reshape(-1, 1, 3).bmm(view_dir.reshape(-1, 3, 1))
            view_loss = view_loss ** self.error_power
        else:
            cross = torch.cross(c_p_dir, view_dir.reshape(-1, 3), dim=1) / torch.norm(c_p_dir, dim=1, keepdim=True)
            view_loss = torch.norm(cross, dim=1)
            view_loss = view_loss ** self.error_power
            view_loss *= (c_p_dir.reshape(-1, 1, 3).bmm(view_dir.reshape(-1, 3, 1)).reshape(-1).detach() > 0).float()

        opt = torch.sum(view_loss) + self.cam_dist_weight * torch.sum(cam_pos_cost)
        return opt, view_loss

    def remove_underground_points(self, optimized_point_ids):
        for p_id in set(optimized_point_ids) - self.rejected_points:
            if self.p_pos[p_id, 2] < self.c_pos[list(self.point_cam_map[p_id]), 2].min() - 10:
                self.rejected_points.add(p_id)

    def remove_distant_points(self, optimized_point_ids, max_distance=100):
        for p_id in set(optimized_point_ids) - self.rejected_points:
            if np.linalg.norm(self.p_pos[p_id] - np.mean(self.c_pos[list(self.point_cam_map[p_id])], axis=0)) > max_distance:
                self.rejected_points.add(p_id)


    def optimize_both(self, optimized_cam_ids, new_stuff=True, iterations=150, episode_lr=1, show_interval=50):
        optimized_cam_ids = sorted(list(set(optimized_cam_ids)))

        last_time = time.time()
        if self.optimized_cams and new_stuff:
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

        optimized_point_ids = set().union(*[self.cam_point_map[i] for i in optimized_cam_ids]) - self.rejected_points
        min_point_constraint_count = 2
        optimized_point_ids = {p_id for p_id in optimized_point_ids if len(self.point_cam_map[p_id]) >= min_point_constraint_count}

        c_i = []
        p_i = []
        for cam_id in range(self.c_count):
            optimized_cam_points = sorted(list(self.cam_point_map[cam_id] & optimized_point_ids))
            cam_id = [cam_id] * len(optimized_cam_points)
            c_i += cam_id
            p_i += optimized_cam_points

        optimized_point_ids = sorted(list(optimized_point_ids))
        optimized_points_new_id = range(len(optimized_point_ids))
        optimized_points_id_map = {orig_id: new_id for orig_id, new_id in zip(optimized_point_ids, optimized_points_new_id)}

        if self.optimized_cams and new_stuff:
            c_i_t = [c_i[i] for i in range(len(p_i)) if p_i[i] in self.optimized_points]
            p_i_t = [i for i in p_i if i in self.optimized_points]
        else:
            c_i_t = c_i
            p_i_t = p_i

        v_i = [self.view_map[c, p] for c, p in zip(c_i_t, p_i_t)]
        p_i_t = [optimized_points_id_map[orig_id] for orig_id in p_i_t]
        view_dir = torch.tensor(self.view_dir[v_i]).float().to(self.device)
        c_i_t = torch.tensor(c_i_t).to(self.device)
        p_i_t = torch.tensor(p_i_t).to(self.device)

        # move camera information to torch/GPU
        c_dist = torch.tensor(self.c_dist[:self.c_count]).float().to(self.device)
        c_dist_weight = torch.tensor(self.c_dist_weight[:self.c_count]).float().to(self.device)
        c_pos = torch.tensor(self.c_pos[:self.c_count]).float().to(self.device)
        c_pos_gt = torch.tensor(self.c_pos_gt[:self.c_count]).float().to(self.device)
        c_rot = torch.tensor(self.c_rot[:self.c_count]).float().to(self.device)
        c_pos_orig = torch.tensor(self.c_pos[:self.c_count]).float().to(self.device)
        c_rot_orig = torch.tensor(self.c_rot[:self.c_count]).float().to(self.device)

        # move point information to torch/GPU
        p_pos = torch.tensor(self.p_pos[optimized_point_ids]).float().to(self.device)

        print(f'Cameras: {c_pos.shape[0]}, Points: {p_pos.shape[0]}, Connections: {view_dir.shape[0]}')

        c_pos.requires_grad = True
        c_rot.requires_grad = True
        p_pos.requires_grad = True

        old_cams = torch.tensor(sorted(list(self.optimized_cams))).long().to(self.device)
        old_points = torch.tensor(sorted([optimized_points_id_map[i] for i in self.optimized_points & set(optimized_point_ids)])).long().to(self.device)
        static_cams = torch.tensor(sorted(list(set(range(self.c_count)) - set(optimized_cam_ids)))).long().to(self.device)

        if self.optimized_cams and new_stuff:
            params = [c_pos, c_rot]
        else:
            params = [c_pos, c_rot, p_pos]
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.90, 0.99))

        print(f'Init time: {time.time() - last_time:.3f}')
        last_time = time.time()

        for i in range(self.max_iter):
            for g in optimizer.param_groups:
                g['lr'] = lr_schedule[i]

            optimizer.zero_grad()
            loss, view_loss = self.problem_loss(
                c_pos, c_rot, c_i_t, p_pos, p_i_t, view_dir, c_dist, c_pos_gt, c_dist_weight)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().item()

            with torch.no_grad():
                if self.optimized_cams and new_stuff and i < self.init_iter:
                    c_pos[old_cams] = c_pos_orig[old_cams]
                    c_rot[old_cams] = c_rot_orig[old_cams]
                else:
                    c_pos[static_cams] = c_pos_orig[static_cams]
                    c_rot[static_cams] = c_rot_orig[static_cams]

                if self.optimized_cams and new_stuff and i == self.init_iter:
                    c_pos_cpu = c_pos.detach().cpu().numpy()
                    c_rot_cpu = c_rot.detach().cpu().numpy()
                    for new_point_id in set(p_i) - self.optimized_points:
                        self.p_pos[new_point_id] = self.triangulate_new_point(
                            new_point_id, c_pos_cpu, c_rot_cpu)
                    p_pos = torch.tensor(self.p_pos[optimized_point_ids]).float().to(self.device)
                    p_pos.requires_grad = True

                    v_i = [self.view_map[c, p] for c, p in zip(c_i, p_i)]
                    p_i = [optimized_points_id_map[orig_id] for orig_id in p_i]
                    view_dir = torch.tensor(self.view_dir[v_i]).float().to(self.device)
                    c_i_t = torch.tensor(c_i).to(self.device)
                    p_i_t = torch.tensor(p_i).to(self.device)
                    params = [c_pos, c_rot, p_pos]
                    optimizer = torch.optim.Adam(params, lr=lr_schedule[i], betas=(0.90, 0.99))

            if i % show_interval == 0:
                #for c in c_rot.detach().cpu().numpy()[:self.c_count]:
                #    print(c * self.rot_scale * 180 / np.pi)
                #print()
                #print('==============================')

                print(f'{i}, {loss}, {show_interval / (time.time() - last_time):.1f}')
                last_time = time.time()
                img, _, _ = view(
                    c_pos.detach().cpu().numpy()[-100:], p_pos.detach().cpu().numpy(), None,
                    resolution=self.resolution, p_i=p_i_t.detach().cpu().numpy())
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

        view_loss = view_loss.detach().cpu().numpy()
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

        self.optimized_points |= set(optimized_point_ids)
        self.optimized_cams |= set(optimized_cam_ids)

        cam_pos_error = np.linalg.norm(self.c_pos[max(0, self.c_count-20):self.c_count] - self.c_pos_gt[max(0, self.c_count-20):self.c_count], axis=1)
        cam_pos_error = np.mean(cam_pos_error)
        print(f'Camera position error at camera {self.c_count}: {cam_pos_error}')
        self.remove_underground_points(optimized_point_ids)
        self.remove_distant_points(optimized_point_ids, max_distance=100)

        #for c in self.c_rot[:self.c_count]:
        #    print(c * self.rot_scale * 180 / np.pi)

    def optimize_cameras(self, cam_id):
        pass

    def optimize_points(self, cam_id):
        pass


