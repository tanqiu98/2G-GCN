"""CAD-120 dataset feature pre-processing code.

The functions here process the raw features provided by [1] into a dictionary mapping the video ID
to a CAD120Video object. Each CAD120Video object contains a list of segments, each segmented represented by a
CAD120VideoSegment object. The CAD120VideoSegment objects contain the features for the segments. The features are:
- skeleton: 630 features.
- skeleton-object: 400 features.
- object: 180 features.
- object-object: 200 features.
- temporal skeleton-skeleton: 160 features.
- temporal object-object: 40 features,
and class-related information:
- sub-activity: class index.
- next sub-activity: class index of the sub-activity in the next video segment.
- object affordance: affordance index for each object.
- next object affordance: affordance index for each object in the next segment.

The temporal features connect segment t to segment t + 1; in that case, the temporal feature connecting segment t
to segment t + 1 is stored in the features of segment t + 1. Thus, the first segment has no temporal features.

[1] Koppula, H. S., Gupta, R., & Saxena, A. (2013). Learning human activities and object affordances
from RGB-D videos. The International Journal of Robotics Research, 32(8), 951–970.
"""
from collections import defaultdict
from itertools import chain
import os
import pickle
from typing import Optional

import numpy as np

from vhoi.cad120classes import CAD120Video, CAD120VideoSegment


def parse_features_binary_svm_format_file(filepath: str, cad120_video=None):
    cad120_video = cad120_video if cad120_video is not None else CAD120Video()
    filename = os.path.basename(filepath)
    filename_info, _ = filename.split(sep='.')
    filename_parts = filename_info.split(sep='_')
    segment_id = int(filename_parts[-1])
    cad120_video_segment = cad120_video[segment_id]
    if len(filename_parts) == 2:
        parse_nontemporal_features_binary_svm_format_file(filepath, cad120_video_segment=cad120_video_segment)
    elif len(filename_parts) == 3:
        parse_temporal_features_binary_svm_format_file(filepath, cad120_video_segment=cad120_video_segment)
    else:
        raise ValueError(f'Could not parse {filepath}')
    return cad120_video


def parse_nontemporal_features_binary_svm_format_file(filepath: str, cad120_video_segment=None):
    cad120_video_segment = cad120_video_segment if cad120_video_segment is not None else CAD120VideoSegment()
    with open(filepath, mode='r') as f:
        first_line = f.readline()
        first_line_info = [int(value) for value in first_line.strip().split(sep=' ')]
        num_objects, num_object_edges, num_skeleton_object_edges, _, _, _ = first_line_info
        # Object Node Features
        for _ in range(num_objects):
            line = f.readline().strip().split(sep=' ')
            affordance_class, object_id, *features = line
            affordance_class, object_id = int(affordance_class), int(object_id)
            cad120_video_segment.object_affordance[object_id] = affordance_class
            features = parse_colon_separated_values(features)
            cad120_video_segment.object_features[object_id] = features
        # Skeleton Node Features
        line = f.readline().strip().split(sep=' ')
        subactivity_class, _, *features = line
        cad120_video_segment.subactivity = int(subactivity_class)
        features = parse_colon_separated_values(features)
        cad120_video_segment.skeleton_features = features
        # Object-Object Features
        for _ in range(num_object_edges):
            line = f.readline().strip().split(sep=' ')
            _, _, object_1_id, object_2_id, *features = line
            object_1_id, object_2_id = int(object_1_id), int(object_2_id)
            features = parse_colon_separated_values(features)
            cad120_video_segment.object_object_features[(object_1_id, object_2_id)] = features
        # Skeleton-Object Features
        for _ in range(num_skeleton_object_edges):
            line = f.readline().strip().split(sep=' ')
            _, _, object_id, *features = line
            object_id = int(object_id)
            features = parse_colon_separated_values(features)
            cad120_video_segment.skeleton_object_features[object_id] = features
    return cad120_video_segment


def parse_temporal_features_binary_svm_format_file(filepath: str, cad120_video_segment=None):
    cad120_video_segment = cad120_video_segment if cad120_video_segment is not None else CAD120VideoSegment()
    with open(filepath, mode='r') as f:
        first_line = f.readline()
        first_line_info = [int(value) for value in first_line.strip().split(sep=' ')]
        num_objects, num_skeletons, _, segment_2_number = first_line_info
        # Temporal Object-Object Features
        for _ in range(num_objects):
            line = f.readline().strip().split(sep=' ')
            object_id, features = parse_temporal_object_object_line(line)
            cad120_video_segment.object_temporal_features[object_id] = features
        # Temporal Skeleton-Skeleton Features
        assert num_skeletons == 1, f'Each video should have only one person. Video {filepath} has {num_skeletons}.'
        line = f.readline().strip().split(sep=' ')
        features = parse_temporal_skeleton_skeleton_line(line)
        cad120_video_segment.skeleton_temporal_features = features
    return cad120_video_segment


def parse_temporal_object_object_line(line: list):
    affordance_class_1, affordance_class_2, object_id, *features = line
    if ':' in object_id:
        features = [object_id] + features
        object_id = affordance_class_2
    if ':' in affordance_class_2:
        features = [affordance_class_2] + features
        object_id = affordance_class_1
    if ':' in affordance_class_1:
        raise ValueError('No object ID.')
    object_id, features = int(object_id), parse_colon_separated_values(features)
    return object_id, features


def parse_temporal_skeleton_skeleton_line(line: list) -> list:
    subactivity_class_1, subactivity_class_2, skeleton_id, *features = line
    if ':' in skeleton_id:
        features = [skeleton_id] + features
    if ':' in subactivity_class_2:
        features = [subactivity_class_2] + features
    if ':' in subactivity_class_1:
        features = [subactivity_class_1] + features
    features = parse_colon_separated_values(features)
    return features


def parse_colon_separated_values(values: list) -> list:
    return [int(value.split(sep=':')[1]) for value in values]


def parse_features_binary_svm_format_dir(path: str):
    data = defaultdict(CAD120Video)
    features_binary_svm_format_files = os.listdir(path)
    for file in features_binary_svm_format_files:
        video_id, *_ = file.split(sep='_')
        cad120_video = data[video_id]
        filepath = os.path.join(path, file)
        parse_features_binary_svm_format_file(filepath, cad120_video=cad120_video)
    return data


def parse_annotation(annotation_dir_path, data):
    subject_folders = os.listdir(annotation_dir_path)
    for subject_folder in subject_folders:
        subject_dir = os.path.join(annotation_dir_path, subject_folder)
        activity_folders = os.listdir(subject_dir)
        for activity_folder in activity_folders:
            activity_dir = os.path.join(subject_dir, activity_folder)
            activity_label_file = os.path.join(activity_dir, 'activityLabel.txt')
            with open(activity_label_file, mode='r') as f:
                for line in f:
                    info = line.split(sep=',')[:-1]
                    video_id, objects_info = info[0], info[3:]
                    object_types = {}
                    for object_info in objects_info:
                        object_id, object_type = object_info.split(sep=':')
                        object_types[int(object_id)] = object_type
                    for video_segment in data[video_id]._video_segments:
                        video_segment.object_type = object_types
            labeling_file = os.path.join(activity_dir, 'labeling.txt')
            with open(labeling_file, mode='r') as f:
                last_video_id, last_segment_id, skip_remaining_video_lines = -1, -1, False
                for line in f:
                    info = line.strip().split(sep=',')
                    video_id, start_frame, end_frame, subactivity_name, *affordance_names = info
                    start_frame, end_frame = int(start_frame), int(end_frame)
                    affordance_names = {object_id: affordance_name
                                        for object_id, affordance_name in enumerate(affordance_names, start=1)}
                    if video_id != last_video_id:
                        last_video_id = video_id
                        last_segment_id = 0
                    if last_segment_id >= len(data[video_id]._video_segments):
                        print(f'There are more annotated segments than segment features provided for video {video_id}'
                              f' | {activity_dir}')
                        continue
                    data[video_id]._video_segments[last_segment_id].start_frame = start_frame
                    data[video_id]._video_segments[last_segment_id].end_frame = end_frame
                    data[video_id]._video_segments[last_segment_id].subactivity_name = subactivity_name
                    data[video_id]._video_segments[last_segment_id].object_affordance_name = affordance_names
                    last_segment_id += 1
    return data


def parse_cad120_information(features_binary_svm_format_path: str, annotation_dir_path: str,
                             save_path: Optional[str] = None):
    """Main function to pre-process the CAD-120 dataset provided raw information.

    First, process the directory 'features_binary_svm_format_path' to obtain segment-level features. Then, update
    the extracted features with metadata from the 'annotation_dir_path'. Save the pre-processed pickled features and
    metadata to save_path (if a save_path is specified).
    """
    data = parse_features_binary_svm_format_dir(features_binary_svm_format_path)
    for cad120_video in data.values():
        cad120_video.from_dict_to_list()
        cad120_video.update_next_labels()
    data = parse_annotation(annotation_dir_path, data=data)
    if save_path is not None:
        save_file = os.path.join(save_path, 'cad120data.pickle')
        with open(save_file, mode='wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return data


def convert_from_world_to_image_coordinates(world_coordinates):
    """Convert from 3D world coordinates to 2D image coordinates.

    Arguments:
        world_coordinates: Tensor of shape (num_points, 3) containing the x, y, z world coordinates of a set of points.
    Returns:
        A tensor of shape (num_points, 2) containing the x, y image coordinates of the input points.
    """
    horizontal_scale = 640 / 1.1147
    vertical_scale = 480 / 0.8336
    depth = world_coordinates[:, -1:]
    x = world_coordinates[:, :1] * horizontal_scale / depth + 320
    y = -world_coordinates[:, 1:2] * vertical_scale / depth + 240
    image_coordinates = np.concatenate([x, y], axis=-1)
    return image_coordinates


def read_skeleton(filepath: str):
    with open(filepath, mode='r') as f:
        skeleton_per_frame = []
        for line in f:
            if line == 'END':
                break
            skeleton_frame = line.split(sep=',')[1:-1]
            skeleton_frame = [float(v) for v in skeleton_frame]
            joints = []
            for i in chain(range(10, 151, 14), range(154, 167, 4)):
                joint_frame_info = skeleton_frame[i:i + 4]
                joints.append(joint_frame_info)
            skeleton_frame = np.stack(joints, axis=0)
            skeleton_per_frame.append(skeleton_frame)
        skeleton_per_frame = np.stack(skeleton_per_frame, axis=0)
    return skeleton_per_frame


def compute_subactivities_statistics(data_path: str, video_id_to_subject_id: dict):
    with open(data_path, mode='rb') as f:
        data = pickle.load(f)
    data_per_subject = defaultdict(list)
    for video_id, video_data in data.items():
        # The original dataset contains features for a video with ID 0505003751 which doesn't correspond to any
        # subject in the dataset.
        subject_id = video_id_to_subject_id.get(video_id, None)
        if subject_id is None:
            continue
        data_per_subject[subject_id].append(video_data)
    statistics_per_subject = {}
    for subject_id, videos in data_per_subject.items():
        # if subject_id in {'Subject1', 'Subject3', 'Subject4'}:
        #     subject_id = 'Subject[1/3/4]'
        # if subject_id in {'Subject1', 'Subject3', 'Subject5'}:
        #     subject_id = 'Subject[1/3/5]'
        # if subject_id in {'Subject1', 'Subject4', 'Subject5'}:
        #     subject_id = 'Subject[1/4/5]'
        # if subject_id in {'Subject3', 'Subject4', 'Subject5'}:
        #     subject_id = 'Subject[3/4/5]'
        # subject_id = 'Subject[1/3/4/5]'
        for video in videos:
            for video_segment in video:
                start_frame, end_frame = video_segment.start_frame, video_segment.end_frame
                if start_frame is None or end_frame is None:
                    continue
                duration = end_frame - start_frame + 1
                subactivity = video_segment.subactivity_name
                statistics_per_subject.setdefault(subject_id, {}).setdefault(subactivity, []).append(duration)
    for subject_id, activities in sorted(statistics_per_subject.items()):
        print(f'{subject_id}')
        for activity_id, durations in sorted(activities.items()):
            print(f'\t{activity_id:10}: {np.mean(durations):6.2f} +/- {np.std(durations):6.2f}')
