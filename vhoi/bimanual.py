import argparse
from collections import defaultdict
import json
import os
from typing import Tuple

import numpy as np
import zarr

from pyrutils.geometric import bounding_boxes_from_keypoints


def extract_2d_bounding_boxes_from_json_file(filepath: str) -> Tuple[list, list]:
    with open(filepath, mode='r') as f:
        bbs_data = json.load(f)
    bbs = []
    certainties = []
    for bb_data in bbs_data:
        bb = bb_data['bounding_box']
        h, w = bb['h'], bb['w']
        x_min, x_max = bb['x'] - w / 2, bb['x'] + w / 2
        y_min, y_max = bb['y'] - h / 2, bb['y'] + h / 2
        bb = [x_min, y_min, x_max, y_max]
        for candidate in bb_data['candidates']:
            bbs.append(bb)
            certainties.append(candidate['certainty'])
    return bbs, certainties


def extract_3d_bounding_boxes_info_from_json_file(filepath: str):
    with open(filepath, mode='r') as f:
        bbs_data = json.load(f)
    certainties = []
    instance_names = []
    for bb_data in bbs_data:
        instance_name = bb_data['instance_name']
        if 'RightHand' in instance_name or 'LeftHand' in instance_name:
            continue
        instance_names.append(instance_name)
        certainties.append(bb_data['certainty'])
    return certainties, instance_names


def extract_hand_keypoints_from_json_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(filepath, mode='r') as f:
        hands_data = json.load(f)
    # A few files have more than one element in this list, but it seems to me that the first detection is usually
    # best.
    hands_data = hands_data[0]
    lh_kps, rh_kps = [], []
    for key, value in sorted(hands_data.items(), key=lambda v: int(v[0].split('_')[1])):
        confidence = value['confidence']
        if confidence:
            x, y = value['x'], value['y']
        else:  # keypoint was not detected
            x, y = float('nan'), float('nan')
        if 'LHand' in key:
            lh_kps.append([x, y])
        elif 'RHand' in key:
            rh_kps.append([x, y])
    lh_kps, rh_kps = np.array(lh_kps, dtype=np.float32), np.array(rh_kps, dtype=np.float32)
    return lh_kps, rh_kps


def from_pixel_proportion_to_pixel_value(keypoints: np.ndarray, img_dims: tuple) -> np.ndarray:
    """Convert coordinates in pixel proportion to pixel values."""
    img_h, img_w = img_dims
    img_dims = np.array([img_w, img_h], dtype=np.float32)
    keypoints *= img_dims
    return keypoints


def track_all_2d_objects(args):
    objects_2d_root_dir = args.objects_2d_root_dir
    img_dims = args.img_dims
    threshold = args.threshold
    save_root = args.save_root
    for dirpath, _, filenames in os.walk(objects_2d_root_dir):
        if filenames:
            dirpath3d = dirpath.replace('2d', '3d')
            tracked_objects, num_frames = track_2d_objects(dirpath, dirpath3d)
            tracked_objects = post_process_tracked_objects(tracked_objects, num_frames, img_dims, threshold)
            if save_root is not None:
                subject, task, take = dirpath.split('/')[-4:-1]
                save_dir = os.path.join(save_root, subject, task)
                os.makedirs(save_dir, exist_ok=True)
                save_file = os.path.join(save_dir, take)
                np.save(save_file, tracked_objects)


def track_2d_objects(objects2d_dir: str, objects3d_dir: str):
    """Track Bimanual objects along the video based on the 2D bounding boxes in each frame.

    We track objects in a video by matching the certainty of the 2D bounding boxes with the certainty of the 3D
    bounding boxes. The reason we do that is because the 3D bounding boxes are already tracked.
    """
    filenames = sorted(os.listdir(objects2d_dir))  # filenames are the same in both directories
    objects = defaultdict(list)  # ID -> List[int, BoundingBox]
    for frame, filename in enumerate(filenames):
        filepath = os.path.join(objects2d_dir, filename)
        bbs_2d, certainties_2d = extract_2d_bounding_boxes_from_json_file(filepath)
        filepath = os.path.join(objects3d_dir, filename)
        certainties_3d, instance_names_3d = extract_3d_bounding_boxes_info_from_json_file(filepath)
        for certainty_3d, instance_name_3d in zip(certainties_3d, instance_names_3d):
            try:
                index = certainties_2d.index(certainty_3d)
            except ValueError:
                continue
            certainties_2d.pop(index)
            bb_2d = bbs_2d.pop(index)
            objects[instance_name_3d].append([frame, bb_2d])
    return objects, len(filenames)


def post_process_tracked_objects(tracked_objects, num_frames, img_dims, threshold: float = 0.5):
    """Post-process the tracked objects.

    Fill out empty frames with NaN, remove badly tracked objects, and convert coordinates from pixel proportion to
    pixel values.
    """
    tracked_objects = {k: v for k, v in tracked_objects.items() if len(v) >= round(threshold * num_frames)}
    tracked = np.full([num_frames, len(tracked_objects), 4], fill_value=np.nan, dtype=np.float32)
    for i, (_, bbs) in enumerate(tracked_objects.items()):
        for frame, bb in bbs:
            tracked[frame, i, :] = bb
    original_shape = tracked.shape
    tracked = from_pixel_proportion_to_pixel_value(tracked.reshape(-1, 2), img_dims).reshape(*original_shape)
    return tracked


def track_all_2d_hands(args):
    hands_2d_root_dir = args.hands_2d_root_dir
    img_dims = args.img_dims
    save_root = args.save_root
    for dirpath, _, filenames in os.walk(hands_2d_root_dir):
        if filenames:
            tracked_hands, num_frames = track_2d_hands(dirpath)
            tracked_hands = post_process_tracked_hands(tracked_hands, num_frames, img_dims)
            if save_root is not None:
                subject, task, take = dirpath.split('/')[-4:-1]
                save_dir = os.path.join(save_root, subject, task)
                os.makedirs(save_dir, exist_ok=True)
                save_file = os.path.join(save_dir, take)
                np.save(save_file, tracked_hands)


def track_2d_hands(dirpath: str):
    """Track Bimanual hands bounding boxes based on the provided keypoints."""
    filenames = sorted(os.listdir(dirpath))
    hands = defaultdict(list)  # ID -> List[int, BoundingBox]
    for frame, filename in enumerate(filenames):
        filepath = os.path.join(dirpath, filename)
        lh_kps, rh_kps = extract_hand_keypoints_from_json_file(filepath)
        lh_bb, rh_bb = bounding_boxes_from_keypoints(lh_kps), bounding_boxes_from_keypoints(rh_kps)
        hands['left_hand'].append([frame, lh_bb])
        hands['right_hand'].append([frame, rh_bb])
    return hands, len(filenames)


def post_process_tracked_hands(tracked_hands, num_frames, img_dims):
    """Fill out empty frames with NaN values and convert coordinates from pixel proportion to pixel values."""
    tracked = np.full([num_frames, len(tracked_hands), 4], fill_value=np.nan, dtype=np.float32)
    for i, (_, bbs) in enumerate(sorted(tracked_hands.items())):
        for frame, bb in bbs:
            tracked[frame, i, :] = bb
    original_shape = tracked.shape
    tracked = from_pixel_proportion_to_pixel_value(tracked.reshape(-1, 2), img_dims).reshape(*original_shape)
    return tracked


def process_ground_truth_single_hand_labels(hand: list):
    start_frames = hand[::2]
    end_frames = [start_frame - 1 for start_frame in start_frames[1:]]
    start_frames = start_frames[:-1]
    actions = [action_id if action_id is not None else -1 for action_id in hand[1::2]]
    framewise_actions = []
    for action, start_frame, end_frame in zip(actions, start_frames, end_frames):
        framewise_actions += [action] * (end_frame - start_frame + 1)
    return framewise_actions


def process_ground_truth_hand_labels(data: dict):
    left_hand = data['left_hand']
    lh_actions = process_ground_truth_single_hand_labels(left_hand)
    right_hand = data['right_hand']
    rh_actions = process_ground_truth_single_hand_labels(right_hand)
    return lh_actions, rh_actions


def collect_ground_truth_hand_labels(args):
    ground_truth_labels_root_dir = args.ground_truth_labels_root_dir
    ground_truth_data = defaultdict(dict)
    for dirpath, dirnames, filenames in os.walk(ground_truth_labels_root_dir):
        if filenames:
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                with open(filepath, mode='r') as f:
                    data = json.load(f)
                lh_actions, rh_actions = process_ground_truth_hand_labels(data)
                subject_id, task_id = dirpath.split(sep='/')[-2:]
                take_id = filename.split(sep='.')[0]
                video_id = f'{subject_id}-{task_id}-{take_id}'
                ground_truth_data[video_id]['left_hand'] = lh_actions
                ground_truth_data[video_id]['right_hand'] = rh_actions
    if (save_dir := args.save_dir) is not None:
        filename = 'bimacs_ground_truth_labels.json'
        filepath = os.path.join(save_dir, filename)
        with open(filepath, mode='w') as f:
            json.dump(ground_truth_data, f, indent=2)


def collect_videos_fps(args):
    rgbd_root_dir = args.rgbd_root_dir
    video_id_to_fps = {}
    for dirpath, dirnames, filenames in os.walk(rgbd_root_dir):
        if filenames and dirpath.endswith('rgb'):
            filename = 'metadata.csv'
            filepath = os.path.join(dirpath, filename)
            fps = None
            with open(filepath, mode='r') as f:
                for line in f:
                    name, _, value = line.strip().split(sep=',')
                    if name == 'fps':
                        fps = int(value)
                        break
            subject_id, task_id, take_id = dirpath.split(sep='/')[-4:-1]
            video_id = f'{subject_id}-{task_id}-{take_id}'
            video_id_to_fps[video_id] = fps
    if (save_dir := args.save_dir) is not None:
        filename = 'bimacs_video_id_to_video_fps.json'
        filepath = os.path.join(save_dir, filename)
        with open(filepath, mode='w') as f:
            json.dump(video_id_to_fps, f, indent=2)


def collect_bimanual_bounding_boxes(args):
    tracked_objects_dir = args.tracked_objects_dir
    tracked_hands_dir = args.tracked_hands_dir
    save_root = args.save_root

    save_path = os.path.join(save_root, 'bounding_boxes.zarr')
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=False)
    for dirpath, dirnames, filenames in os.walk(tracked_hands_dir):
        if filenames:
            subject, task = dirpath.split('/')[-2:]
            for filename in filenames:
                take = filename.split(sep='.')[0]
                video_id = f'{subject}-{task}-{take}'
                if video_id not in root:
                    filepath = os.path.join(tracked_hands_dir, subject, task, take + '.npy')
                    tracked_hands = np.load(filepath)
                    filepath = os.path.join(tracked_objects_dir, subject, task, take + '.npy')
                    tracked_objects = np.load(filepath)
                    features_group = root.create_group(video_id)
                    features_group.array('left_hand', tracked_hands[:, 0], chunks=False, dtype=np.float32)
                    features_group.array('right_hand', tracked_hands[:, 1], chunks=False, dtype=np.float32)
                    features_group.array('objects', tracked_objects, chunks=False, dtype=np.float32)
                    print(f'Processed bounding boxes for video {video_id}')


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Pre-process Bimanual dataset functions.')
    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')
    # Track 2D objects in the Bimanual Actions dataset.
    parser_track_2d_objects = subparsers.add_parser('track_2d_objects',
                                                    help='Track the objects in the Bimanual Actions dataset. We '
                                                         'save the x_min, y_min, x_max, y_max coordinates of the '
                                                         'bounding boxes. The coordinates are converted from pixel '
                                                         'proportion to pixel value.')

    parser_track_2d_objects.add_argument('--objects_2d_root_dir', type=str, required=True,
                                         help='The \'bimacs_derived_data_2d_objects\' directory. Note that you should '
                                              'also have the \'bimacs_derived_data_3d_objects\' directory alongside '
                                              'it.')
    parser_track_2d_objects.add_argument('--img_dims', nargs=2, default=(480, 640), type=int,
                                         help='The height and width of the Bimanual dataset.')
    parser_track_2d_objects.add_argument('--threshold', default=0.5, type=float,
                                         help='Objects that have been tracked for less than threshold * num frames in '
                                              'the video are excluded.')
    parser_track_2d_objects.add_argument('--save_root', type=str,
                                         help='Path of directory to save the tracked objects. We replicate the '
                                              'structure of \'bimacs_derived_data_2d_objects\' inside the save_root '
                                              'directory, but simplify it by making it like '
                                              '.../subject_a/task_b/take_c.npy, where take_c.npy contains all tracked '
                                              'object as an array of shape (num_frames, num_objects, 4).')
    parser_track_2d_objects.set_defaults(func=track_all_2d_objects)
    # Track 2D bounding boxes of person's hands.
    parser_track_2d_hands = subparsers.add_parser('track_2d_hands',
                                                  help='Track the hands in the Bimanual Actions dataset. We '
                                                       'save the x_min, y_min, x_max, y_max coordinates of the '
                                                       'bounding boxes. The coordinates are converted from pixel '
                                                       'proportion to pixel value.')

    parser_track_2d_hands.add_argument('--hands_2d_root_dir', type=str, required=True,
                                       help='The \'bimacs_derived_data_hand_pose\' directory.')
    parser_track_2d_hands.add_argument('--img_dims', nargs=2, default=(480, 640), type=int,
                                       help='The height and width of the Bimanual dataset.')
    parser_track_2d_hands.add_argument('--save_root', type=str,
                                       help='Path of directory to save the tracked hands. We replicate the '
                                            'structure of \'bimacs_derived_data_hand_pose\' inside the save_root '
                                            'directory, but simplify it by making it like '
                                            '.../subject_a/task_b/take_c.npy, where take_c.npy contains all tracked '
                                            'hands as an array of shape (num_frames, 2, 4). take_c[:, 0, :] is the '
                                            'left hand whereas take_c[:, 1, :] is the right hand.')
    parser_track_2d_hands.set_defaults(func=track_all_2d_hands)
    # Collect tracked bounding boxes into a single location
    parser_bounding_boxes = subparsers.add_parser('collect_bounding_boxes',
                                                  help='Collect the tracked hands and objects bounding boxes into a '
                                                       'single location.')
    parser_bounding_boxes.add_argument('--tracked_objects_dir', type=str, required=True,
                                       help='Path to root directory containing all tracked 2D objects. This is the '
                                            '\'bimacs_derived_data_2d_objects_tracked\' dir.')
    parser_bounding_boxes.add_argument('--tracked_hands_dir', type=str, required=True,
                                       help='Path to root directory containing all tracked hands. This is the '
                                            '\'bimacs_derived_data_hand_pose_tracked\' dir.')
    parser_bounding_boxes.add_argument('--save_root', type=str,
                                       help='Path to directory to save the bounding boxes. A bounding_boxes.zarr '
                                            'folder is created inside save_root.')
    parser_bounding_boxes.set_defaults(func=collect_bimanual_bounding_boxes)
    # Collect ground-truth labels for hands
    parser_collect_hand_labels = subparsers.add_parser('collect_hand_labels',
                                                       help='Collect the ground-truth action for both hands into a '
                                                            'single file.')
    parser_collect_hand_labels.add_argument('--ground_truth_labels_root_dir', type=str, required=True,
                                            help='The \'bimacs_rgbd_data_ground_truth\' directory.')
    parser_collect_hand_labels.add_argument('--save_dir', type=str,
                                            help='Directory to save single .json file with all ground-truth labels '
                                                 'in it.')
    parser_collect_hand_labels.set_defaults(func=collect_ground_truth_hand_labels)
    # Create video id to video FPS dictionary
    parser_collect_videos_fps = subparsers.add_parser('collect_videos_fps',
                                                      help='Collect the recording FPS of each video into a '
                                                           'single file.')
    parser_collect_videos_fps.add_argument('--rgbd_root_dir', type=str, required=True,
                                           help='Path to the \'rgbd\' directory.')
    parser_collect_videos_fps.add_argument('--save_dir', type=str,
                                           help='Directory to save single .json file with all videos FPS.')
    parser_collect_videos_fps.set_defaults(func=collect_videos_fps)
    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
