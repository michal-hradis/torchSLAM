import logging
logging.basicConfig(level=logging.INFO)

import torch
from collections import defaultdict
from pytorch3d import transforms


class CameraMotionSpeedConstraint:
    def __init__(self, device, camera):
        self.device = device
        self.camera = camera

        self.distances = torch.zeros([camera.c_pos.shape[0]], dtype=torch.float32, device=device)
        self.weights = torch.zeros([camera.c_pos.shape[0]], dtype=torch.float32, device=device)

        self.last_cost = None

    def add_camera(self, cam_id, distance, weight=1):
        self.distances[cam_id] = distance
        self.weights[cam_id] = weight

    def loss(self):
        c_pos = self.camera.c_pos[:self.camera.c_count]
        distances = self.distances[1:self.camera.c_count]
        weights = self.weights[1:self.camera.c_count]
        cost = (torch.sum((c_pos[1:] - c_pos[:-1]) ** 2, axis=1) ** 0.5 - distances) ** 2
        self.last_cost = cost * weights
        return torch.sum(self.last_cost)


class CameraPositionConstraint:
    def __init__(self, device, camera):
        self.device = device
        self.camera = camera

        self.positions = torch.zeros(camera.c_pos.shape, dtype=torch.float32, device=device)
        self.weights = torch.zeros([camera.c_pos.shape[0]], dtype=torch.float32, device=device)

        self.last_cost = None

    def add_camera(self, cam_id, position, weight=1):
        self.positions[cam_id] = position
        self.weights[cam_id] = weight

    def loss(self):
        c_pos = self.camera.c_pos[:self.camera.c_count]
        positions = self.positions[:self.camera.c_count]
        weights = self.weights[:self.camera.c_count]
        cost = torch.sum((c_pos - positions) ** 2, axis=1)
        self.last_cost = cost * weights
        return torch.sum(self.last_cost)

    def all_positions(self):
        return self.positions[:self.camera.c_count]


class CalibratedCamera:
    def __init__(self, device, error_power=0.5, c_max=10000, v_max=1000000):
        self.use_dot_product = False
        self.rot_scale = 0.05
        self.error_power = error_power
        self.c_max = c_max
        self.p_max = v_max
        self.device = device

        self.c_count = 0
        self.p_count = 0
        self.c_pos = torch.zeros([self.c_max, 3], dtype=torch.float, device=self.device)
        self.c_rot = torch.zeros([self.c_max, 3], dtype=torch.float, device=self.device)

        self.views = torch.zeros([self.p_max, 3], dtype=torch.float, device=self.device)
        self.views_p_id = torch.zeros([self.p_max], dtype=torch.long, device=self.device)
        self.views_c_id = torch.zeros([self.p_max], dtype=torch.long, device=self.device)

        self.c_points = defaultdict(set)

        self.opt_c_id = None
        self.opt_p_id = None
        self.opt_views = None
        self.opt_sleeping_cams = None

    def add_camera(self, position, rotation, views_p_id, views):
        '''

        :param position: [1, 3] - initial camera position vector
        :param rotation: [1, 3] - initial camera rotation vector as euler angles 'XYZ'
        :param views_p_id: [n] - array of point ids for points viewed from the camera
        :param views: [n, 3] - point view directions as unit vectors for each point viewed from the camere in the camera coordinate frame
        :return: int - new camera id
        '''
        cam_id = self.c_count
        self.c_count += 1

        with torch.no_grad():
            self.c_pos[cam_id] = position
            self.c_rot[cam_id] = rotation
            self.views_c_id[self.p_count:self.p_count + views_p_id.shape[0]] = cam_id
            self.views_p_id[self.p_count:self.p_count + views_p_id.shape[0]] = views_p_id
            self.views[self.p_count:self.p_count + views_p_id.shape[0]] = views
            self.p_count += views_p_id.shape[0]

        return cam_id

    def prepare_optimization(self, c_id_request: torch.Tensor, p_id_request: torch.Tensor, p_id_rejected: torch.Tensor):
        '''

        :param c_id_request: [n] long tensor of optimized camera ids
        :param p_id_request: [n] long tensor of optimized point ids (it is used to
        :param p_id_rejected: [n] long tensor of points which should not be used
        :return: N/A
        '''
        with torch.no_grad():
            if c_id_request is None and p_id_request is None:
                raise NotImplementedError('At least one of c_id_request and p_id_request has to be specified')
            cp_mask = None
            if c_id_request is not None:
                cp_mask = torch.isin(self.views_c_id[:self.p_count], c_id_request)
            if p_id_request is not None:
                p_mask = torch.isin(self.views_p_id[:self.p_count], p_id_request)
                if cp_mask is None:
                    cp_mask = p_mask
                else:
                    cp_mask = torch.logical_or(cp_mask, p_mask)

            if p_id_rejected is not None:
                r_mask = torch.isin(self.views_p_id[:self.p_count], p_id_rejected, invert=True)
                cp_mask = torch.logical_and(cp_mask, r_mask)

            self.opt_c_static = torch.isin(
                torch.arange(0, self.c_max, dtype=torch.long, device=self.device), c_id_request, invert=True)

            self.opt_c_id = self.views_c_id[:self.p_count][cp_mask]
            self.opt_p_id = self.views_p_id[:self.p_count][cp_mask]
            self.opt_views = self.views[:self.p_count][cp_mask, :]

    def distortion_model(self, views):
        return views

    def loss(self, p_pos):
        c_pos = self.c_pos[self.opt_c_id]
        c_rot = transforms.euler_angles_to_matrix(self.c_rot[:self.c_count] * self.rot_scale, convention='XYZ')
        c_rot = c_rot[self.opt_c_id]

        p_pos = p_pos[self.opt_p_id]

        views = self.distortion_model(self.opt_views)
        view_dir = c_rot.bmm(views.reshape(-1, 3, 1))

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

        opt = torch.sum(view_loss)
        return opt

    def all_positions(self):
        return self.c_pos[:self.c_count]

    def all_view_point_ids(self):
        return self.views_p_id[:self.p_count]

    def get_world_views(self, view_ids):
        R = transforms.euler_angles_to_matrix(self.c_rot[self.views_c_id[:view_ids.shape[0]][view_ids]] * self.rot_scale, convention='XYZ')
        u = R.bmm(self.views[:view_ids.shape[0]][view_ids].reshape(-1, 3, 1)).squeeze(2)
        return u, self.c_pos[self.views_c_id[:view_ids.shape[0]][view_ids]]

    def mask_gradients(self):
        self.c_pos.grad[self.opt_c_static] = 0
        self.c_rot.grad[self.opt_c_static] = 0


class PerspectiveCamera:
    def __init__(self, device, error_power=0.5, c_max=10000, v_max=1000000):
        self.use_dot_product = False
        self.rot_scale = 0.05
        self.error_power = error_power
        self.c_max = c_max
        self.p_max = v_max
        self.device = device

        self.c_count = 0
        self.p_count = 0
        self.c_pos = torch.zeros([self.c_max, 3], dtype=torch.float, device=self.device)
        self.c_rot = torch.zeros([self.c_max, 3], dtype=torch.float, device=self.device)
        self.c_focal = torch.zeros([self.c_max, 3], dtype=torch.float, device=self.device)

        self.views = torch.zeros([self.p_max, 3], dtype=torch.float, device=self.device)
        self.views_p_id = torch.zeros([self.p_max], dtype=torch.long, device=self.device)
        self.views_c_id = torch.zeros([self.p_max], dtype=torch.long, device=self.device)

        self.c_points = defaultdict(set)

        self.opt_c_id = None
        self.opt_p_id = None
        self.opt_views = None
        self.opt_sleeping_cams = None

    def add_camera(self, position, rotation, views_p_id, views):
        '''

        :param position: [1, 3] - initial camera position vector
        :param rotation: [1, 3] - initial camera rotation vector as euler angles 'XYZ'
        :param views_p_id: [n] - array of point ids for points viewed from the camera
        :param views: [n, 3] - point view directions as unit vectors for each point viewed from the camere in the camera coordinate frame
        :return: int - new camera id
        '''
        cam_id = self.c_count
        self.c_count += 1

        with torch.no_grad():
            self.c_pos[cam_id] = position
            self.c_rot[cam_id] = rotation
            self.views_c_id[self.p_count:self.p_count + views_p_id.shape[0]] = cam_id
            self.views_p_id[self.p_count:self.p_count + views_p_id.shape[0]] = views_p_id
            self.views[self.p_count:self.p_count + views_p_id.shape[0]] = views
            self.p_count += views_p_id.shape[0]

        return cam_id

    def prepare_optimization(self, c_id_request: torch.Tensor, p_id_request: torch.Tensor, p_id_rejected: torch.Tensor):
        '''

        :param c_id_request: [n] long tensor of optimized camera ids
        :param p_id_request: [n] long tensor of optimized point ids (it is used to
        :param p_id_rejected: [n] long tensor of points which should not be used
        :return: N/A
        '''
        with torch.no_grad():
            if c_id_request is None and p_id_request is None:
                raise NotImplementedError('At least one of c_id_request and p_id_request has to be specified')
            cp_mask = None
            if c_id_request is not None:
                cp_mask = torch.isin(self.views_c_id[:self.p_count], c_id_request)
            if p_id_request is not None:
                p_mask = torch.isin(self.views_p_id[:self.p_count], p_id_request)
                if cp_mask is None:
                    cp_mask = p_mask
                else:
                    cp_mask = torch.logical_or(cp_mask, p_mask)

            if p_id_rejected is not None:
                r_mask = torch.isin(self.views_p_id[:self.p_count], p_id_rejected, invert=True)
                cp_mask = torch.logical_and(cp_mask, r_mask)

            self.opt_c_static = torch.isin(
                torch.arange(0, self.c_max, dtype=torch.long, device=self.device), c_id_request, invert=True)

            self.opt_c_id = self.views_c_id[:self.p_count][cp_mask]
            self.opt_p_id = self.views_p_id[:self.p_count][cp_mask]
            self.opt_views = self.views[:self.p_count][cp_mask, :]

    def distortion_model(self, views):
        return views

    def loss(self, p_pos):
        c_pos = self.c_pos[self.opt_c_id]
        c_rot = transforms.euler_angles_to_matrix(self.c_rot[:self.c_count] * self.rot_scale, convention='XYZ')
        c_rot = c_rot[self.opt_c_id]

        p_pos = p_pos[self.opt_p_id]

        views = self.distortion_model(self.opt_views)
        view_dir = c_rot.bmm(views.reshape(-1, 3, 1))

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

        opt = torch.sum(view_loss)
        return opt

    def all_positions(self):
        return self.c_pos[:self.c_count]

    def all_view_point_ids(self):
        return self.views_p_id[:self.p_count]

    def get_world_views(self, view_ids):
        R = transforms.euler_angles_to_matrix(self.c_rot[self.views_c_id[:view_ids.shape[0]][view_ids]] * self.rot_scale, convention='XYZ')
        u = R.bmm(self.views[:view_ids.shape[0]][view_ids].reshape(-1, 3, 1)).squeeze(2)
        return u, self.c_pos[self.views_c_id[:view_ids.shape[0]][view_ids]]

    def mask_gradients(self):
        self.c_pos.grad[self.opt_c_static] = 0
        self.c_rot.grad[self.opt_c_static] = 0
