import logging
logging.basicConfig(level=logging.INFO)

import argparse
import torch
import numpy as np
import cv2
import time
from scipy import spatial
from pytorch3d import transforms
import json
import open3d as o3d

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resolution', default=800, type=int, help="View window resolution")
    parser.add_argument('-l', '--trajectory-length', default=500, type=int, help="Generated trajectory key-frame count.")
    parser.add_argument('-m', '--trajectory-momentum', default=0.95, type=float, help="Generated trajectory movement momentum.")
    parser.add_argument('--trajectory-acceleration', default=0.8, type=float,
                        help="Generated trajectory acceleration standard deviation.")

    parser.add_argument('--point-distance', default=10, type=float,
                        help="Point to camera distance standard deviation.")
    parser.add_argument('--points-per-camera', default=15, type=float,
                        help="Point generated per camera.")

    parser.add_argument('--point-visibility-falloff', default=8, type=float,
                        help="How distant cameras see the point.")

    parser.add_argument('--view-noise', default=0.02, type=float,
                        help="Point location error in radians. (points projected to camera)")
    parser.add_argument('--cam-noise', default=100, type=float,
                        help="Initial camera position noise.")
    parser.add_argument('--point-noise', default=100, type=float,
                        help="Initial point position noise.")
    parser.add_argument('--world-shift', default=10, type=float,
                        help="Initial shift in world position estimate.")

    parser.add_argument('--learning-rate', default=5, type=float,
                        help="ADAM learning rate.")
    parser.add_argument('--cam-dist-weight', default=0.25, type=float,
                        help="Mutual camera distance cost weight.")
    parser.add_argument('--cam-dir-weight', default=0.25, type=float,
                        help="Horizontal camera alinment cost weight.")

    parser.add_argument('--json-recording', help='Read viewing directions from json file.')

    args = parser.parse_args()
    return args


def range_coeff(x, distance, range):
    return np.exp(-np.log2(distance / x) ** 2 / range)


def generate_cam_trajectory(length=30, momentum=0.94, acc_sdev=0.6, acc2_sdev=0.0):
    positions = np.zeros([length, 3], dtype=np.float32)
    positions_2 = np.zeros([length, 3], dtype=np.float32)
    last_pos = np.zeros([3], dtype=np.float32)
    last_pos_2 = np.zeros([3], dtype=np.float32)

    last_velocity = np.asarray([10, 0, 0], dtype=np.float32)
    last_velocity_2 = np.asarray([10, 0, 0], dtype=np.float32)

    for p, p2 in zip(positions, positions_2):
        p[...] = last_pos + last_velocity
        p2[...] = last_pos_2 + last_velocity_2
        last_pos = p
        last_pos_2 = p2

        acc = np.random.normal(size=[2]) * acc_sdev
        last_velocity[:2] = last_velocity[:2] * momentum + acc
        last_velocity_2[:2] = last_velocity_2[:2] * momentum + acc + np.random.normal(size=[2]) * acc2_sdev

    return positions, positions_2


def generate_points(trajectory, trajectory_2, point_camera_distance=5, points_per_camera=10, camera_point_visibility_distance=10):
    points = np.zeros([trajectory.shape[0]*points_per_camera, 3])
    points_2 = np.zeros([trajectory.shape[0]*points_per_camera, 3])
    for i in range(trajectory.shape[0]):
        p = np.random.normal(size=[points_per_camera, 3]) * point_camera_distance
        p[:, 2] *= 0.1

        points[i*points_per_camera:i*points_per_camera+points_per_camera] = \
            trajectory[i].reshape(1, 3) + p
        points_2[i*points_per_camera:i*points_per_camera+points_per_camera] = \
            trajectory_2[i].reshape(1, 3) + p

    camera_point_assignment = np.zeros([trajectory.shape[0], points.shape[0]])
    for i in range(trajectory.shape[0]):
        camera_point_assignment[i, i*points_per_camera:i*points_per_camera+points_per_camera] = 1

    distances = spatial.distance_matrix(trajectory, points, p=2)
    prob = range_coeff(distances, camera_point_visibility_distance, 1) * 0.7
    print(prob.mean())
    prob = prob > np.random.uniform(0, 1, size=prob.shape)
    print(np.mean(prob))
    camera_point_assignment[prob] = 1

    return points, points_2, camera_point_assignment


def view(cam_trajectory, points, points2, camera_point_assignment, resolution=1600, center=None, size=None, relations=False, errors=False):
    cam_trajectory = cam_trajectory.copy()[:, :2]
    points = points.copy()[:, :2]
    points2 = points2.copy()[:, :2]
    all = cam_trajectory[:, :2]

    if center is None:
        center = (np.max(all, axis=0, keepdims=True) + np.min(all, axis=0, keepdims=True)) / 2
    if size is None:
        size = np.max(np.linalg.norm(all - center, axis=1)) * 1.1

    img = np.zeros([resolution, resolution, 3], dtype=np.uint8)
    cam_trajectory = (cam_trajectory - center) / size / 2 + 0.5
    points = (points - center) / size / 2 + 0.5
    points2 = (points2 - center) / size / 2 + 0.5
    cam_trajectory *= resolution
    points *= resolution
    points2 *= resolution

    if relations:
        for start, camera_points in zip(cam_trajectory, camera_point_assignment):
            for end in points[camera_points > 0]:
                cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (128, 128, 128))

    if errors:
        for p1, p2 in zip(points, points2):
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (128, 128, 128))
            cv2.circle(img, (int(p1[0]), int(p1[1])), 1, (0, 0, 255), -1)
            cv2.circle(img, (int(p2[0]), int(p2[1])), 1, (255, 0, 0), -1)
    else:
        for p in points:
            cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)

    for p in cam_trajectory:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

    return img, center, size


def get_viewing_directions(cam_trajectory, points, noise_sdev=0.1):
    directions = cam_trajectory.reshape(cam_trajectory.shape[0], 1, 3) - points.reshape(1, points.shape[0], 3)
    directions = directions / np.linalg.norm(directions, axis=2, keepdims=True)
    noise = np.random.normal(size=directions.shape)
    noise = noise ** 4 * np.sign(noise) * noise_sdev
    torch_c_rotation = transforms.euler_angles_to_matrix(torch.tensor(noise), convention='XYZ')
    torch_c_rotation = torch_c_rotation.reshape(-1, 3, 3)
    directions = torch_c_rotation.bmm(torch.tensor(directions).reshape(-1, 3, 1)).cpu().numpy().reshape(directions.shape[0], directions.shape[1], 3)
    return directions


def problem_loss(torch_cameras, torch_c_rotation, torch_camera_i, torch_points, torch_point_i, torch_directions,
                 camera_distance, camera_up, cam_dist_weight, cam_dir_weight):
    cam_flat = torch_cameras[torch_camera_i]
    points_flat = torch_points[torch_point_i]

    # cameras should be roughly horizontal
    torch_c_rotation = transforms.euler_angles_to_matrix(torch_c_rotation * 0.05, convention='XYZ')
    real_up = torch_c_rotation.bmm(camera_up.reshape(-1, 3, 1))
    up_score = 1.001 - camera_up.reshape(-1, 1, 3).bmm(real_up)

    # rotate camera point views
    torch_directions = torch_c_rotation[torch_camera_i].bmm(torch_directions.reshape(-1, 3, 1))

    # get current point view directions
    dir = cam_flat - points_flat
    dir = dir / (torch.norm(dir, dim=1, keepdim=True) + 1e-5)

    # point view loss
    prod = 1.001 - torch.bmm(dir.reshape(-1, 1, 3), torch_directions.reshape(-1, 3, 1))
    prod = prod ** 0.1

    # consecutive camera pair distance loss
    cam_dist = torch.sum((torch.sum((torch_cameras[1:] - torch_cameras[:-1]) ** 2, axis=1) ** 0.5 - camera_distance)**2)

    # final combined loss
    opt = torch.sum(prod) + cam_dist_weight * cam_dist + cam_dir_weight * torch.sum(up_score)
    return opt


def generate_data(args):
    cameras, cameras_2 = generate_cam_trajectory(args.trajectory_length, momentum=args.trajectory_momentum,
                                                 acc_sdev=args.trajectory_acceleration)

    points, points_2, camera_point_assignment = generate_points(cameras, cameras_2,
                                                                point_camera_distance=args.point_distance,
                                                                points_per_camera=args.points_per_camera,
                                                                camera_point_visibility_distance=args.point_visibility_falloff)

    print('Connections: ', np.sum(camera_point_assignment), np.sum(camera_point_assignment) / args.trajectory_length)
    directions = get_viewing_directions(cameras, points, noise_sdev=args.view_noise)

    noisy_cameras = cameras_2.copy()
    noisy_cameras[1:-1] += np.random.normal(size=(cameras.shape[0] - 2, 3)) * args.cam_noise
    noisy_cameras[1:-1, 1] += args.world_shift
    noisy_points = points_2 + np.random.normal(size=points.shape) * args.point_noise
    noisy_points[:, 1] += args.world_shift

    return cameras, directions, noisy_cameras, points, noisy_points, camera_point_assignment


def read_json(args):
    records = []
    max_records = 200
    with open(args.json_recording, 'r') as f:
        for i, line in enumerate(f):
            if i % 30 == 0:
                records.append(json.loads(line))
                if len(records) == max_records:
                    break
    cam_ids = dict()
    point_ids = dict()

    for i, record in enumerate(records):
        if record['video_frame'] not in cam_ids:
            cam_ids[record['video_frame']] = len(cam_ids)
        for p_id in record['point_ids']:
            if p_id not in point_ids:
                point_ids[p_id] = len(point_ids)

    cameras = np.zeros([len(cam_ids), 3], dtype=np.float32)
    cameras[:, 0] = -np.linspace(0, cameras.shape[0], cameras.shape[0])
    directions = np.zeros([len(cam_ids), len(point_ids), 3], dtype=np.float32)
    points = np.zeros([len(point_ids), 3], dtype=np.float32)
    camera_point_assignment = np.zeros([len(cam_ids), len(point_ids)], dtype=np.float32)

    for record in records:
        c_id = cam_ids[record['video_frame']]
        for p_id, view_dir in zip(record['point_ids'], record['directions']):
            p_id = point_ids[p_id]
            directions[c_id, p_id, :] = np.asarray(view_dir)
            camera_point_assignment[c_id, p_id] = 1

    for p_id, p in enumerate(points):
        p[...] = np.mean(cameras[camera_point_assignment[:, p_id] > 0], axis=0)
        p += np.random.normal(size=(3,)) * 3

    return cameras, directions, cameras, points, points, camera_point_assignment


def main():
    args = parseargs()
    print('ARGS', args)

    if not args.json_recording:
        cameras, directions, noisy_cameras, points, noisy_points, camera_point_assignment = generate_data(args)

    else:
        cameras, directions, noisy_cameras, points, noisy_points, camera_point_assignment = read_json(args)

    img, center, size = view(cameras, points, noisy_points, camera_point_assignment, resolution=args.resolution,
                             relations=True)
    cv2.imwrite('gt.jpg', img)
    cv2.imshow('GT', img)
    img, _, _ = view(noisy_cameras, noisy_points, points, camera_point_assignment,
                     resolution=args.resolution, center=center, size=size, errors=True)
    cv2.imshow('result', img)
    cv2.waitKey()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    torch_camera_distance = torch.tensor(np.sum((cameras[1:] - cameras[0:-1])**2, axis=1)**0.5).float().to(device)
    torch_camera_up = torch.zeros(cameras.shape).float().to(device)
    torch_camera_up[:, 2] = 1

    torch_cameras = torch.tensor(noisy_cameras).float().to(device)
    torch_c_rotation = torch.zeros(noisy_cameras.shape).float().to(device)
    torch_points = torch.tensor(noisy_points).float().to(device)
    torch_directions = torch.tensor(directions[camera_point_assignment > 0]).reshape(-1, 3).float().to(device)
    c, p = np.nonzero(camera_point_assignment)
    torch_camera_i = torch.tensor(c).to(device)
    torch_point_i = torch.tensor(p).to(device)

    torch_cameras.requires_grad = True
    torch_points.requires_grad = True
    torch_c_rotation.requires_grad = True

    params = [torch_points]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)


    loss_obj = [0]
    def closure():
        optimizer.zero_grad()
        loss = problem_loss(torch_cameras, torch_c_rotation, torch_camera_i, torch_points, torch_point_i,
                            torch_directions, torch_camera_distance, torch_camera_up,
                            args.cam_dist_weight, args.cam_dir_weight)
        loss.backward()
        loss_obj[0] = loss.cpu().item()
        return loss

    out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (args.resolution, args.resolution))

    last_time = time.time()
    for i in range(1000000):
        optimizer.step(closure)

        if i % 20 == 0:
            img, _, _ = view(
                torch_cameras.detach().cpu().numpy(), torch_points.detach().cpu().numpy(), points, camera_point_assignment,
                resolution=args.resolution, center=center, size=size, errors=False)
            out.write(img)

        if time.time() - last_time > 0.1:
            print(i, loss_obj)
            img, _, _ = view(
                torch_cameras.detach().cpu().numpy(), torch_points.detach().cpu().numpy(), points, camera_point_assignment,
                resolution=args.resolution, errors=False)
            cv2.imshow('result', img)
            key = cv2.waitKey(5)
            if key == 27:
                break
            elif key == ord(' '):
                params = [torch_points, torch_cameras, torch_c_rotation]
                optimizer = torch.optim.Adam(params, lr=args.learning_rate)
            elif key == ord('s'):
                pcd = o3d.geometry.PointCloud()
                c = torch_cameras.detach().cpu().numpy()
                p = torch_points.detach().cpu().numpy()
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


if __name__ == "__main__":
    main()
