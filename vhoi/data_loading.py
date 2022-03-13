from cmath import nan
from functools import partial
import json
import os
import pickle
import random
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import zarr

from pyrutils.itertools import run_length_encoding
import pyrutils.torch.fetchers as fetchers
import pyrutils.torch.forwarders as forwarders
from pyrutils.torch.train_utils import numpy_to_torch
from pyrutils.utils import read_dictionary


def load_cad120_training_data(data_path, model_name, model_input_type, test_subject_id, video_id_to_subject_id,
                              data_path_zarr=None, batch_size=8, val_fraction=0.2, seed=42, debug=False,
                              scaling_strategy=None, sigma: float = 0.0, downsampling: int = 1):
    root = None
    if data_path_zarr is not None:
        root = zarr.open(data_path_zarr, mode='r')
    with open(data_path, mode='rb') as f:
        data = pickle.load(f)
    training_data = []
    for video_id, video_data in data.items():
        # The original dataset contains features for a video with ID 0505003751 which doesn't correspond to any
        # subject in the dataset.
        subject_id = video_id_to_subject_id.get(video_id, None)
        if subject_id is None or subject_id == test_subject_id:
            continue
        if root is not None:
            training_data_datum = [root[video_id + '/skeleton'][:], root[video_id + '/objects'][:],
                                   root[video_id + '/skeleton_bounding_box'][:],
                                   root[video_id + '/objects_bounding_box'][:],
                                   root[video_id + '/skeleton_2d_pose'][:],
                                   video_data]
        else:
            training_data_datum = video_data
        training_data.append(training_data_datum)
    training_data, val_data = split_train_test(training_data, test_fraction=val_fraction, seed=seed)
    if debug:
        training_data = training_data[:4]
        val_data = val_data[:1]
    train_loader, scalers, _ = create_data_loader(training_data, model_name, model_input_type, 'cad120',
                                                  batch_size=batch_size, shuffle=True,
                                                  scaling_strategy=scaling_strategy, sigma=sigma,
                                                  downsampling=downsampling, test_data=False)
    val_loader, _, _ = create_data_loader(val_data, model_name, model_input_type, 'cad120', batch_size=len(val_data),
                                          shuffle=False, scalers=scalers, sigma=sigma, downsampling=downsampling,
                                          test_data=False)
    input_size = input_size_from_data_loader(train_loader, model_name, model_input_type)
    data_info = {'input_size': input_size}
    return train_loader, val_loader, data_info, scalers


def load_bimanual_training_data(data_path, data_path_zarr, data_path_bbs_zarr, data_path_hps_zarr, model_name: str, model_input_type: str,
                                test_subject_id: int, video_id_to_video_fps: dict, batch_size: int = 8,
                                val_fraction: float = 0.2, seed: int = 42, debug: bool = False, scaling_strategy=None,
                                sigma: float = 0.0, downsampling: int = 1):
    with open(data_path, mode='rb') as f:
        data = json.load(f)
    #with open(video_id_to_video_fps, mode='rb') as v:
    #    video_id_to_video_fps = json.load(v)
    root = zarr.open(data_path_zarr, mode='r')
    root_bbs = zarr.open(data_path_bbs_zarr, mode='r')
    root_hps = zarr.open(data_path_hps_zarr, mode='r')
    training_data = []
    for video_id, hands_ground_truth in data.items():
        subject, task, take = video_id.split(sep='-')
        subject_id = int(subject.split(sep='_')[1])
        if subject_id == test_subject_id:
            continue
        left_hand_features = root[video_id]['left_hand'][:]
        right_hand_features = root[video_id]['right_hand'][:]
        object_features = root[video_id]['objects'][:]
        left_hand_bbs = root_bbs[video_id]['left_hand'][:]
        right_hand_bbs = root_bbs[video_id]['right_hand'][:]
        objects_bbs = root_bbs[video_id]['objects'][:]
        left_hand_hps = root_hps[video_id]['left_hand'][:]
        right_hand_hps = root_hps[video_id]['right_hand'][:]
        video_fps = video_id_to_video_fps[video_id]
        if video_fps == 15:  # Some videos were mistakenly collected at 15 FPS.
            left_hand_features = np.repeat(left_hand_features, repeats=2, axis=0)
            right_hand_features = np.repeat(right_hand_features, repeats=2, axis=0)
            object_features = np.repeat(object_features, repeats=2, axis=0)
            left_hand_bbs = np.repeat(left_hand_bbs, repeats=2, axis=0)
            right_hand_bbs = np.repeat(right_hand_bbs, repeats=2, axis=0)
            objects_bbs = np.repeat(objects_bbs, repeats=2, axis=0)
            left_hand_hps = np.repeat(left_hand_hps, repeats=2, axis=0)
            right_hand_hps = np.repeat(right_hand_hps, repeats=2, axis=0)
            hands_ground_truth['left_hand'] = np.repeat(hands_ground_truth['left_hand'], repeats=2, axis=0)
            hands_ground_truth['right_hand'] = np.repeat(hands_ground_truth['right_hand'], repeats=2, axis=0)
        training_data.append([left_hand_features, right_hand_features, object_features, hands_ground_truth,
                              left_hand_bbs, right_hand_bbs, objects_bbs, left_hand_hps, right_hand_hps])
    training_data, val_data = split_train_test(training_data, test_fraction=val_fraction, seed=seed)
    if debug:
        training_data = training_data[:4]
        val_data = val_data[:1]
    train_loader, scalers, _ = create_data_loader(training_data, model_name, model_input_type, 'bimanual',
                                                  batch_size=batch_size, shuffle=True,
                                                  scaling_strategy=scaling_strategy, sigma=sigma,
                                                  downsampling=downsampling, test_data=False)
    val_loader, _, _ = create_data_loader(val_data, model_name, model_input_type, 'bimanual', batch_size=len(val_data),
                                          shuffle=False, scalers=scalers, sigma=sigma, downsampling=downsampling,
                                          test_data=False)
    input_size = input_size_from_data_loader(train_loader, model_name, model_input_type)
    data_info = {'input_size': input_size}
    return train_loader, val_loader, data_info, scalers


def load_mphoi_training_data(data_path, data_path_zarr, data_path_obbs_zarr, data_path_hbbs_zarr, data_path_hps_zarr, model_name: str, 
                             model_input_type: str, test_subject_id: int, batch_size: int = 8, val_fraction: float = 0.2, seed: int = 42, 
                             debug: bool = False, scaling_strategy=None, sigma: float = 0.0, downsampling: int = 1):
    with open(data_path, mode='rb') as f:
        data = json.load(f)
    root = zarr.open(data_path_zarr, mode='r')
    root_obbs = zarr.open(data_path_obbs_zarr, mode='r')
    root_hbbs = zarr.open(data_path_hbbs_zarr, mode='r')
    root_hps = zarr.open(data_path_hps_zarr, mode='r')
    training_data = []
    for video_id, human_ground_truth in data.items():
        subject_id, task, take = video_id.split(sep='-')
        first_sub, second_sub = int(subject_id[-2]), int(subject_id[-1])
        first_test_sub, second_test_sub = int(test_subject_id[-2]), int(test_subject_id[-1])
        if (first_sub-first_test_sub)*(second_sub-second_test_sub)*(first_sub-second_test_sub)*(second_sub-first_test_sub) == 0:
            continue
        Human1_features = root[video_id]['Human1'][:]
        Human2_features = root[video_id]['Human2'][:]
        object_features = root[video_id]['objects'][:]
        Human1_bbs = root_hbbs[video_id]['Human1'][:]
        Human2_bbs = root_hbbs[video_id]['Human2'][:]
        objects_bbs = root_obbs[video_id]['objects'][:]
        Human1_hps = root_hps[video_id]['Human1'][:]
        Human2_hps = root_hps[video_id]['Human2'][:]
        training_data.append([Human1_features, Human2_features, object_features, human_ground_truth,
                              Human1_bbs, Human2_bbs, objects_bbs, Human1_hps, Human2_hps])
    training_data, val_data = split_train_test(training_data, test_fraction=val_fraction, seed=seed)
    if debug:
        training_data = training_data[:4]
        val_data = val_data[:1]
    train_loader, scalers, _ = create_data_loader(training_data, model_name, model_input_type, 'mphoi',
                                                  batch_size=batch_size, shuffle=True,
                                                  scaling_strategy=scaling_strategy, sigma=sigma,
                                                  downsampling=downsampling, test_data=False)
    val_loader, _, _ = create_data_loader(val_data, model_name, model_input_type, 'mphoi', batch_size=len(val_data),
                                          shuffle=False, scalers=scalers, sigma=sigma, downsampling=downsampling,
                                          test_data=False)
    input_size = input_size_from_data_loader(train_loader, model_name, model_input_type)
    data_info = {'input_size': input_size}
    return train_loader, val_loader, data_info, scalers


def load_training_data(data, model_name, model_input_type, batch_size: int = 8, val_fraction: float = 0.2,
                       seed: int = 42, debug: bool = False, sigma: float = 0.0):
    data_path, data_path_zarr = data.path, data.path_zarr
    test_subject_id = data.cross_validation_test_subject
    scaling_strategy = data.scaling_strategy
    downsampling = data.downsampling
    if 'BimanualActions' in data_path:
        with open(data.video_id_to_video_fps, mode='r') as f:
            video_id_to_video_fps = json.load(f)
        data_path_bbs_zarr = data.path_bb_zarr
        data_path_hps_zarr = data.path_hp_zarr
        train_loader, val_loader, data_info, scalers = \
            load_bimanual_training_data(data_path, data_path_zarr, data_path_bbs_zarr, data_path_hps_zarr, model_name, model_input_type,
                                        test_subject_id=test_subject_id, video_id_to_video_fps=video_id_to_video_fps,
                                        batch_size=batch_size, val_fraction=val_fraction, seed=seed, debug=debug,
                                        scaling_strategy=scaling_strategy, sigma=sigma, downsampling=downsampling)
    elif 'MPHOI' in data_path:
        data_path_obbs_zarr = data.path_obb_zarr
        data_path_hbbs_zarr = data.path_hbb_zarr
        data_path_hps_zarr = data.path_hps_zarr
        train_loader, val_loader, data_info, scalers = \
            load_mphoi_training_data(data_path, data_path_zarr, data_path_obbs_zarr, data_path_hbbs_zarr, data_path_hps_zarr, 
                                        model_name, model_input_type, test_subject_id=test_subject_id, batch_size=batch_size, 
                                        val_fraction=val_fraction, seed=seed, debug=debug, scaling_strategy=scaling_strategy, 
                                        sigma=sigma, downsampling=downsampling)
    
    else:  # CAD-120
        video_id_to_subject_id = read_dictionary(data.video_id_to_subject_id)
        train_loader, val_loader, data_info, scalers = \
            load_cad120_training_data(data_path, model_name, model_input_type,
                                      test_subject_id=test_subject_id,
                                      video_id_to_subject_id=video_id_to_subject_id,
                                      data_path_zarr=data_path_zarr,
                                      batch_size=batch_size,
                                      val_fraction=val_fraction, seed=seed,
                                      debug=debug,
                                      scaling_strategy=scaling_strategy,
                                      sigma=sigma, downsampling=downsampling)
    return train_loader, val_loader, data_info, scalers


def load_cad120_testing_data(data_path, model_name, model_input_type, test_subject_id, video_id_to_subject_id,
                             data_path_zarr=None, batch_size=8, scalers=None, downsampling: int = 1):
    root = None
    if data_path_zarr is not None:
        root = zarr.open(data_path_zarr, mode='r')
    with open(data_path, mode='rb') as f:
        data = pickle.load(f)
    testing_data = []
    test_ids = []
    for video_id, video_data in data.items():
        # The original dataset contains features for a video with ID 0505003751 which doesn't correspond to any
        # subject in the dataset.
        subject_id = video_id_to_subject_id.get(video_id, None)
        if subject_id is None or subject_id != test_subject_id:
            continue
        if root is not None:
            testing_data_datum = [root[video_id + '/skeleton'][:], root[video_id + '/objects'][:],
                                  root[video_id + '/skeleton_bounding_box'][:],
                                  root[video_id + '/objects_bounding_box'][:],
                                  root[video_id + '/skeleton_2d_pose'][:],
                                  video_data]
        else:
            testing_data_datum = video_data
        testing_data.append(testing_data_datum)
        test_ids.append(video_id)
    test_loader, _, segmentations = create_data_loader(testing_data, model_name, model_input_type, 'cad120',
                                                       batch_size=batch_size, shuffle=False, scalers=scalers,
                                                       downsampling=downsampling, test_data=True)
    input_size = input_size_from_data_loader(test_loader, model_name, model_input_type)
    data_info = {'input_size': input_size}
    return test_loader, data_info, segmentations, test_ids


def load_bimanual_testing_data(data_path, data_path_zarr, data_path_bbs_zarr, data_path_hps_zarr, model_name: str, model_input_type: str,
                               test_subject_id: int, video_id_to_video_fps: dict, batch_size: int,
                               scalers: Optional[dict] = None, downsampling: int = 1):
    with open(data_path, mode='rb') as f:
        data = json.load(f)
    root = zarr.open(data_path_zarr, mode='r')
    root_bbs = zarr.open(data_path_bbs_zarr, mode='r')
    root_hps = zarr.open(data_path_hps_zarr, mode='r')
    testing_data, test_ids = [], []
    for video_id, hands_ground_truth in data.items():
        subject, task, take = video_id.split(sep='-')
        subject_id = int(subject.split(sep='_')[1])
        if subject_id != test_subject_id:
            continue
        left_hand_features = root[video_id]['left_hand'][:]
        right_hand_features = root[video_id]['right_hand'][:]
        object_features = root[video_id]['objects'][:]
        left_hand_bbs = root_bbs[video_id]['left_hand'][:]
        right_hand_bbs = root_bbs[video_id]['right_hand'][:]
        objects_bbs = root_bbs[video_id]['objects'][:]
        left_hand_hps = root_hps[video_id]['left_hand'][:]
        right_hand_hps = root_hps[video_id]['right_hand'][:]
        video_fps = video_id_to_video_fps[video_id]
        if video_fps == 15:  # Some videos were mistakenly collected at 15 FPS.
            left_hand_features = np.repeat(left_hand_features, repeats=2, axis=0)
            right_hand_features = np.repeat(right_hand_features, repeats=2, axis=0)
            object_features = np.repeat(object_features, repeats=2, axis=0)
            left_hand_bbs = np.repeat(left_hand_bbs, repeats=2, axis=0)
            right_hand_bbs = np.repeat(right_hand_bbs, repeats=2, axis=0)
            objects_bbs = np.repeat(objects_bbs, repeats=2, axis=0)
            left_hand_hps = np.repeat(left_hand_hps, repeats=2, axis=0)
            right_hand_hps = np.repeat(right_hand_hps, repeats=2, axis=0)
            hands_ground_truth['left_hand'] = np.repeat(hands_ground_truth['left_hand'], repeats=2, axis=0)
            hands_ground_truth['right_hand'] = np.repeat(hands_ground_truth['right_hand'], repeats=2, axis=0)
        testing_data.append([left_hand_features, right_hand_features, object_features, hands_ground_truth,
                             left_hand_bbs, right_hand_bbs, objects_bbs, left_hand_hps, right_hand_hps])
        test_ids.append(video_id)
    test_loader, _, segmentations = create_data_loader(testing_data, model_name, model_input_type, 'bimanual',
                                                       batch_size=batch_size, shuffle=False, scalers=scalers,
                                                       downsampling=downsampling, test_data=True)
    input_size = input_size_from_data_loader(test_loader, model_name, model_input_type)
    data_info = {'input_size': input_size}
    return test_loader, data_info, segmentations, test_ids


def load_mphoi_testing_data(data_path, data_path_zarr, data_path_obbs_zarr, data_path_hbbs_zarr, data_path_hps_zarr,
                            model_name: str, model_input_type: str, test_subject_id: int, batch_size: int, 
                            scalers: Optional[dict] = None, downsampling: int = 1):
    with open(data_path, mode='rb') as f:
        data = json.load(f)
    root = zarr.open(data_path_zarr, mode='r')
    root_obbs = zarr.open(data_path_obbs_zarr, mode='r')
    root_hbbs = zarr.open(data_path_hbbs_zarr, mode='r')
    root_hps = zarr.open(data_path_hps_zarr, mode='r')
    testing_data, test_ids = [], []
    for video_id, human_ground_truth in data.items():
        subject_id, task, take = video_id.split(sep='-')
        if subject_id != test_subject_id:
            continue
        Human1_features = root[video_id]['Human1'][:]
        Human2_features = root[video_id]['Human2'][:]
        object_features = root[video_id]['objects'][:]
        Human1_bbs = root_hbbs[video_id]['Human1'][:]
        Human2_bbs = root_hbbs[video_id]['Human2'][:]
        objects_bbs = root_obbs[video_id]['objects'][:]
        Human1_hps = root_hps[video_id]['Human1'][:]
        Human2_hps = root_hps[video_id]['Human2'][:]
        testing_data.append([Human1_features, Human2_features, object_features, human_ground_truth,
                            Human1_bbs, Human2_bbs, objects_bbs, Human1_hps, Human2_hps])
        test_ids.append(video_id)
    test_loader, _, segmentations = create_data_loader(testing_data, model_name, model_input_type, 'mphoi',
                                                       batch_size=batch_size, shuffle=False, scalers=scalers,
                                                       downsampling=downsampling, test_data=True)
    input_size = input_size_from_data_loader(test_loader, model_name, model_input_type)
    data_info = {'input_size': input_size}
    return test_loader, data_info, segmentations, test_ids


def load_testing_data(data, model_name: str, model_input_type: str, batch_size: int, scalers: Optional[dict] = None):
    data_path, data_path_zarr = data.path, data.path_zarr
    test_subject_id = data.cross_validation_test_subject
    downsampling = data.get('downsampling', default_value=1)
    if 'BimanualActions' in data_path:
        with open(data.video_id_to_video_fps, mode='r') as f:
            video_id_to_video_fps = json.load(f)
        data_path_bbs_zarr = data.get('path_bb_zarr', default_value=None)
        data_path_hps_zarr = data.get('path_hp_zarr', default_value=None)
        if data_path_bbs_zarr is None:
            data_path_bbs_zarr = os.path.join(os.path.dirname(data_path_zarr), 'bounding_boxes.zarr')
        if data_path_hps_zarr is None:
            data_path_hps_zarr = os.path.join(os.path.dirname(data_path_zarr), 'hands_pose.zarr')
        test_loader, data_info, segmentations, test_ids = \
            load_bimanual_testing_data(data_path, data_path_zarr, data_path_bbs_zarr, data_path_hps_zarr, model_name, model_input_type,
                                       test_subject_id=test_subject_id, video_id_to_video_fps=video_id_to_video_fps,
                                       batch_size=batch_size, scalers=scalers, downsampling=downsampling)
    elif 'MPHOI' in data_path:
        data_path_obbs_zarr = data.get('path_obb_zarr', default_value=None)
        data_path_hbbs_zarr = data.get('path_hbb_zarr', default_value=None)
        data_path_hps_zarr = data.get('path_hps_zarr', default_value=None)
        if data_path_obbs_zarr is None:
            data_path_obbs_zarr = os.path.join(os.path.dirname(data_path_zarr), 'object_bounding_boxes.zarr')
        if data_path_hbbs_zarr is None:
            data_path_hbbs_zarr = os.path.join(os.path.dirname(data_path_zarr), 'human_bounding_boxes.zarr')
        if data_path_hps_zarr is None:
            data_path_hps_zarr = os.path.join(os.path.dirname(data_path_zarr), 'human_pose.zarr')
        test_loader, data_info, segmentations, test_ids = \
            load_mphoi_testing_data(data_path, data_path_zarr, data_path_obbs_zarr, data_path_hbbs_zarr, data_path_hps_zarr,
                                    model_name, model_input_type, test_subject_id=test_subject_id, batch_size=batch_size, 
                                    scalers=scalers, downsampling=downsampling)
    else:  # CAD-120
        video_id_to_subject_id = read_dictionary(data.video_id_to_subject_id)
        test_loader, data_info, segmentations, test_ids = \
            load_cad120_testing_data(data_path, model_name, model_input_type,
                                     test_subject_id=test_subject_id, video_id_to_subject_id=video_id_to_subject_id,
                                     data_path_zarr=data_path_zarr, batch_size=batch_size, scalers=scalers,
                                     downsampling=downsampling)
    return test_loader, data_info, segmentations, test_ids


def split_train_test(training_data: list, test_fraction: float = 0.2, seed: int = 42):
    random.seed(seed)
    random.shuffle(training_data)
    num_testing_videos = round(len(training_data) * test_fraction)
    testing_data = training_data[:num_testing_videos]
    training_data = training_data[num_testing_videos:]
    return training_data, testing_data


def create_data_loader(data, model_name: str, model_input_type: str, dataset_name: str, batch_size: int, shuffle: bool,
                       scaling_strategy: Optional[str] = None, scalers: Optional[dict] = None, sigma: float = 0.0,
                       downsampling: int = 1, test_data: bool = False):
    if dataset_name.lower() == 'cad120':
        x, y = assemble_tensors(data, model_name, model_input_type, sigma=sigma, downsampling=downsampling,
                                test_data=test_data)
    elif dataset_name.lower() == 'mphoi':
        x, y = assemble_mphoi_tensors(data, model_name, sigma=sigma, downsampling=downsampling, test_data=test_data)
    else:
        x, y = assemble_bimanual_tensors(data, model_name, sigma=sigma, downsampling=downsampling, test_data=test_data)
    x, scalers = maybe_scale_input_tensors(x, model_name, scaling_strategy=scaling_strategy, scalers=scalers)
    x = [np.nan_to_num(ix, copy=False, nan=0.0) for ix in x]
    x, y = numpy_to_torch(*x), numpy_to_torch(*y)
    dataset = TensorDataset(*(x + y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                             pin_memory=False, drop_last=False)
    segmentations = assemble_segmentations(data, model_name, dataset_name=dataset_name)
    return data_loader, scalers, segmentations


def assemble_segmentations(data, model_name: str, dataset_name: str):
    segmentations = None
    if model_name == '2G-GCN':
        if dataset_name.lower() == 'cad120':
            segmentations = assemble_cad120_segmentations_from_frame_level_features(data)
    return segmentations


def assemble_cad120_segmentations_from_frame_level_features(data):
    segmentations = []
    for _, _, _, _, _, video_data in data:
        segmentations.append([])
        for video_segment in video_data:
            start_frame, end_frame = video_segment.start_frame, video_segment.end_frame
            if start_frame is None or end_frame is None:
                continue
            start_frame -= 1
            end_frame -= 1
            segmentations[-1].append((start_frame, end_frame))
    return segmentations


def maybe_scale_input_tensors(x: list, model_name: str, scaling_strategy: Optional[str] = None,
                              scalers: Optional[dict] = None):
    there_are_no_scalers = scalers is None or not scalers
    if there_are_no_scalers:
        scalers = {}
        if scaling_strategy is None:
            return x, scalers
    if model_name in {'2G-GCN', 'bimanual_baseline', 'cad120_baseline'}:
        x_human, human_scaler = x[0], scalers.get('human_scaler', None)
        x_human, human_scaler = scale_array(x_human, scaler=human_scaler, scaling_strategy=scaling_strategy)

        x_object, object_scaler = x[1], scalers.get('object_scaler', None)
        x_object, object_scaler = scale_array(x_object, scaler=object_scaler, scaling_strategy=scaling_strategy)

        scalers = {'human_scaler': human_scaler, 'object_scaler': object_scaler}
        x = [x_human, x_object] + x[2:]
    return x, scalers


def scale_array(x, scaler=None, scaling_strategy='standard'):
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1])
    if scaler is None:
        if scaling_strategy == 'standard':
            scaler = StandardScaler().fit(x)
        else:
            raise ValueError(f'scaling_strategy must be standard and not {scaling_strategy}.')
    x = scaler.transform(x)
    x = x.reshape(*x_shape)
    return x, scaler


def assemble_tensors(data: list, model_name: str, model_input_type: str, sigma: float = 0.0, downsampling: int = 1,
                     test_data: bool = False):
    if model_name in {'2G-GCN', 'cad120_baseline'}:
        xs, ys = assemble_frame_level_recurrent_human(data, downsampling=downsampling, test_data=test_data)
        if model_name == '2G-GCN':
            if sigma:
                ys[2] = ignore_last_step_end_flag(ys[2])
            ys[2] = smooth_segmentation(ys[2], sigma)
            ys_budget = ys[2]
            xs_objects, ys_objects = assemble_frame_level_recurrent_objects(data, downsampling=downsampling,
                                                                            test_data=test_data)
            if sigma:
                ys_objects[2] = ignore_last_step_end_flag_general(ys_objects[2])
            ys_objects[2] = smooth_segmentation(ys_objects[2], sigma)
            ys_objects_budget = ys_objects[2]
            xs_ho_dists = assemble_human_object_distances(data, downsampling=downsampling)
            xs_oo_dists = assemble_object_object_distances(data, downsampling=downsampling)
            # Update tensor sequence to: x_human, x_objects, x_objects_mask, x_human_segment, x_objects_segment.
            xs = xs[:1] + xs_objects[:2] + xs[1:] + xs_objects[2:]
            ys = ([ys_budget] + [ys_objects_budget] + ys[2:] + ys_objects[2:] +
                  ys[:2] + ys_objects[:2] +
                  ys[:2] + ys_objects[:2])
            xs_steps = assemble_num_steps(data, downsampling=downsampling)
            xs += [xs_ho_dists, xs_oo_dists, xs_steps]
            xs = add_fake_dimension_to_human_tensors(xs, [0, 3])
            ys = add_fake_dimension_to_human_tensors(ys, [0, 2, 4, 5, 8, 9])
        elif model_name == 'cad120_baseline':
            xs_objects, ys_objects = assemble_frame_level_recurrent_objects(data, downsampling=downsampling,
                                                                            test_data=test_data)
            xs = xs[:1] + xs_objects[:2]
            xs = add_fake_dimension_to_human_tensors(xs, [0])
            ys = ys[:1] + ys_objects[:1]
            ys = add_fake_dimension_to_human_tensors(ys, [0])
    else:
        raise ValueError(f'{model_name} is not an option for model name.')
    return xs, ys


def add_fake_dimension_to_human_tensors(tensors, indices):
    for index in indices:
        tensors[index] = np.expand_dims(tensors[index], axis=2)
    return tensors


def assemble_bimanual_tensors(data: list, model_name: str, sigma: float = 0.0, downsampling: int = 1,
                              test_data: bool = False):
    xs, ys = assemble_bimanual_frame_level_recurrent_human(data, downsampling=downsampling, test_data=test_data)
    xs_objects = assemble_bimanual_frame_level_recurrent_objects(data, downsampling=downsampling)
    if model_name == '2G-GCN':
        if sigma:
            ys[2] = ignore_last_step_end_flag_general(ys[2])
        ys[2] = smooth_segmentation(ys[2], sigma)
        ys_budget = ys[2]
        xs_hh_dists = assemble_bimanual_human_human_distances(data, downsampling=downsampling)
        xs_ho_dists = assemble_bimanual_human_object_distances(data, downsampling=downsampling)
        xs_oo_dists = assemble_bimanual_object_object_distances(data, downsampling=downsampling)
        xs_steps = assemble_num_steps(data, downsampling=downsampling)
        xs = xs[:1] + xs_objects + xs[1:] + [xs_hh_dists, xs_ho_dists, xs_oo_dists, xs_steps]
        ys = [ys_budget] + ys[2:] + ys[:2]
        ys += ys[-2:]
    elif model_name == 'bimanual_baseline':
        xs, ys = xs[:1], ys[:1]
        xs = xs + xs_objects
    else:
        raise ValueError(f'Bimanual code not implemented for {model_name} yet.')
    return xs, ys


def assemble_mphoi_tensors(data: list, model_name: str, sigma: float = 0.0, downsampling: int = 1,
                              test_data: bool = False):
    xs, ys = assemble_mphoi_frame_level_recurrent_human(data, downsampling=downsampling, test_data=test_data)
    xs_objects = assemble_mphoi_frame_level_recurrent_objects(data, downsampling=downsampling)
    if model_name == '2G-GCN':
        if sigma:
            ys[2] = ignore_last_step_end_flag_general(ys[2])
        ys[2] = smooth_segmentation(ys[2], sigma)
        ys_budget = ys[2]
        xs_hh_dists = assemble_mphoi_human_human_distances(data, downsampling=downsampling)
        xs_ho_dists = assemble_mphoi_human_object_distances(data, downsampling=downsampling)
        xs_oo_dists = assemble_mphoi_object_object_distances(data, downsampling=downsampling)
        xs_steps = assemble_num_steps(data, downsampling=downsampling)
        xs = xs[:1] + xs_objects + xs[1:] + [xs_hh_dists, xs_ho_dists, xs_oo_dists, xs_steps]
        ys = [ys_budget] + ys[2:] + ys[:2]
        ys += ys[-2:]
    else:
        raise ValueError(f'MPHOI code not implemented for {model_name} yet.')
    return xs, ys


def ignore_last_step_end_flag(x):
    """x is a tensor of shape (num_examples, num_steps)."""
    end_frame_examples, end_frame_steps = np.nonzero(x == 1.0)
    acc_lengths = 0
    for m, length in run_length_encoding(end_frame_examples):
        acc_lengths += length
        end_frame_index = acc_lengths - 1
        end_frame_index = end_frame_steps[end_frame_index]
        x[m, end_frame_index] = 0.0
    return x


def ignore_last_step_end_flag_general(x):
    """x is a tensor of shape (num_examples, num_steps, num_entities)."""
    num_entities = x.shape[-1]
    for e in range(num_entities):
        x[:, :, e] = ignore_last_step_end_flag(x[:, :, e])
    return x


def smooth_segmentation(x, sigma: float):
    """Smooth an input segmentation.

    Arguments:
        x - A tensor of shape (num_examples, num_steps).
        sigma - Gaussian smoothing value.
    Returns:
        The smoothed segmentation, a tensor of shape (num_examples, num_steps).
    """
    if sigma:
        missing_indices = x == -1.0
        x[missing_indices] = 0.0
        x = np.clip(gaussian_filter1d(x, sigma=sigma, axis=1, mode='constant') * 2.5 * sigma, 0.0, 1.0)
        x[missing_indices] = -1.0
    return x


def assemble_frame_level_recurrent_human(data, downsampling: int = 1, test_data: bool = False):
    # Process input tensors (visual features)
    xs_human = []
    xs_pose = []
    x_obb= []
    max_len, max_len_downsampled = 0, 0
    for human_features, _, _, objects_bounding_box, skeleton_pose, _ in data:
        max_len = max(max_len, human_features.shape[0])
        human_features = human_features[downsampling - 1::downsampling]
        max_len_downsampled = max(max_len_downsampled, human_features.shape[0])
        xs_human.append(human_features)
        skeleton_pose = skeleton_pose[downsampling - 1::downsampling]/100
        xs_pose.append(skeleton_pose)
        objects_bounding_box = objects_bounding_box[downsampling - 1::downsampling]/100
        x_obb.append(objects_bounding_box)
    # xs_obb: reshape the object bounding box
    bbb = list()
    for i, video in enumerate(x_obb):
        bb = list()
        for j, frame in enumerate(video):
            b = np.zeros((5,4))    # max_no_objects = 5
            n = len(frame)
            if n!=5:
                b[0:n,] = frame 
            else:
                b = frame
            b = b.reshape(10,2)
            bb.append(b)
        bbb.append(bb)
    xs_obb = bbb

    # xs_human: add various features to human features
    xs_human_copy = xs_human
    xx = list()
    for i, video in enumerate(xs_human_copy):
        x = list()
        for j, frame in enumerate(video):
            pose = xs_pose[i][j]       # skeletons (9*2)
            obb = xs_obb[i][j]         # object bounding box (10*2)
            # velocity
            if j+1 < len(video):
                next_pose = xs_pose[i][j+1]
                sk_velo = (next_pose - pose)*100
                next_obb = xs_obb[i][j+1]
                obb_velo = (next_obb - obb)*100                     
            else:
                sk_velo = np.zeros((9, 2))
                obb_velo = np.zeros((10, 2))
            skvelo = np.hstack((pose, sk_velo)) # 9*4
            obbvelo = np.hstack((obb, obb_velo)) # 10*4
            pv = skvelo.reshape(1, -1)
            ov = obbvelo.reshape(1, -1)
            # concatenate all fetures
            args = (frame, pv[0], ov[0])
            human_cat = np.concatenate(args)
            x.append(human_cat)
        x = np.array(x)
        xx.append(x)
    xs_human = xx

    human_feature_size = xs_human[-1].shape[-1] # human_feature[2048] + skvelo[36] + obbvelo[40] = 2124-dimension
    x_human = np.full([len(xs_human), max_len_downsampled, human_feature_size], fill_value=np.nan, dtype=np.float32)
    for m, x_h in enumerate(xs_human):
        x_human[m, :x_h.shape[0], :] = x_h
    xs = [x_human]
    # Process output classes and extra tensors
    y_rec_subactivity = np.full([x_human.shape[0], max_len], fill_value=-1, dtype=np.int64)
    y_pred_subactivity = np.full_like(y_rec_subactivity, fill_value=-1)
    for m, (_, _, _, _, _, video_data) in enumerate(data):
        for i, video_segment in enumerate(video_data):
            start_frame, end_frame = video_segment.start_frame, video_segment.end_frame
            if start_frame is None or end_frame is None:
                continue
            start_frame -= 1
            end_frame -= 1
            subactivity = video_segment.subactivity - 1
            y_rec_subactivity[m, start_frame:end_frame + 1] = subactivity
            next_subactivity = video_segment.next_subactivity
            next_subactivity = next_subactivity - 1 if next_subactivity is not None else -1
            y_pred_subactivity[m, start_frame:end_frame + 1] = next_subactivity
    x_subactivity_segmentation = segmentation_from_output_class(y_rec_subactivity[:, downsampling - 1::downsampling],
                                                                segmentation_type='input')
    xs.append(x_subactivity_segmentation)
    if not test_data:
        y_rec_subactivity = y_rec_subactivity[:, downsampling - 1::downsampling]
        y_pred_subactivity = y_pred_subactivity[:, downsampling - 1::downsampling]
    y_subactivity_segmentation = segmentation_from_output_class(y_rec_subactivity, segmentation_type='output')
    ys = [y_rec_subactivity, y_pred_subactivity, y_subactivity_segmentation]
    return xs, ys


def assemble_bimanual_frame_level_recurrent_human(data, downsampling: int = 1, test_data: bool = False):
    # Input
    xs_lh, xs_rh = [], []
    xs_lhp, xs_rhp = [], []
    x_obb= []
    max_len, max_len_downsampled = 0, 0
    for left_hand, right_hand, _, _, _, _, objects_bounding_box, left_hand_pose, right_hand_pose in data:
        # hands feature
        max_len = max(max_len, left_hand.shape[0])
        left_hand = left_hand[downsampling - 1::downsampling]
        right_hand = right_hand[downsampling - 1::downsampling]
        max_len_downsampled = max(max_len_downsampled, left_hand.shape[0])
        xs_lh.append(left_hand)
        xs_rh.append(right_hand)
        # hands pose
        left_hand_pose = left_hand_pose[downsampling - 1::downsampling]/100
        right_hand_pose = right_hand_pose[downsampling - 1::downsampling]/100
        xs_lhp.append(left_hand_pose)
        xs_rhp.append(right_hand_pose)
        # objects bounding box
        objects_bounding_box = objects_bounding_box[downsampling - 1::downsampling]/100
        x_obb.append(objects_bounding_box)
    # xs_obb: reshape the object bounding box
    bbb = list()
    for i, video in enumerate(x_obb):
        bb = list()
        for j, frame in enumerate(video):
            b = np.zeros((9,4))  # max_no_objects = 9
            n = len(frame)
            if n!=9:
                b[0:n,] = frame 
            else:
                b = frame
            b = b.reshape(18,2)
            bb.append(b)
        bbb.append(bb)
    xs_obb = bbb
    # add context features to xs_lh, xs_rh
    ll = list()
    rr = list()
    keypoints = [0, 4, 8, 12, 16, 20]         # hand keypoints
    for i, video in enumerate(xs_lh):
        l = list()
        r = list()
        for j, frame in enumerate(video):
            lhp = xs_lhp[i][j][keypoints]     # left hand pose  (6*2)
            rhp = xs_rhp[i][j][keypoints]     # right hand pose (6*2)
            obb = xs_obb[i][j]                # object bounding box (18*2)
            # velocity
            if j+1 < len(video):
                next_lhp = xs_lhp[i][j+1][keypoints]
                lhp_velo = (next_lhp - lhp)*100
                next_rhp = xs_rhp[i][j+1][keypoints]
                rhp_velo = (next_rhp - rhp)*100
                next_obb = xs_obb[i][j+1]
                obb_velo = (next_obb - obb)*100                     
            else:
                lhp_velo = np.zeros((6, 2))
                rhp_velo = np.zeros((6, 2))
                obb_velo = np.zeros((18, 2))
            lhpvelo = np.hstack((lhp, lhp_velo)) # 6*4
            rhpvelo = np.hstack((rhp, rhp_velo)) # 6*4
            obbvelo = np.hstack((obb, obb_velo)) # 18*4
            lpv = lhpvelo.reshape(1, -1)
            rpv = rhpvelo.reshape(1, -1)
            obv = obbvelo.reshape(1, -1)
            # concatenate all fetures
            args = (lpv[0], rpv[0], obv[0])
            context = np.concatenate(args) # 120
            lh_con = np.concatenate((frame, context)) # 2168
            rh_con = np.concatenate((xs_rh[i][j], context)) # 2168
            l.append(lh_con)
            r.append(rh_con)
        l = np.array(l)
        r = np.array(r)
        ll.append(l)
        rr.append(r)
    xs_lh = ll
    xs_rh = rr
    feature_size = xs_lh[0].shape[-1]
    x_hs = np.full([len(xs_lh), max_len_downsampled, 2, feature_size], fill_value=np.nan, dtype=np.float32)
    for m, (lh, rh) in enumerate(zip(xs_lh, xs_rh)):
        x_hs[m, :len(lh), 0] = lh
        x_hs[m, :len(rh), 1] = rh
    xs = [x_hs]
    # Output
    y_rec_hs = np.full([len(x_hs), max_len, 2], fill_value=-1, dtype=np.int64)
    y_pred_hs = np.full_like(y_rec_hs, fill_value=-1)
    for m, (_, _, _, video_hands_ground_truth, _, _, _, _, _) in enumerate(data):
        # Left Hand
        y_lh = video_hands_ground_truth['left_hand']
        y_rec_hs[m, :len(y_lh), 0] = y_lh
        rle = list(run_length_encoding(y_lh))
        y_lh_p = []
        for (_, previous_length), (next_label, _) in zip(rle[:-1], rle[1:]):
            y_lh_p += [next_label] * previous_length
        y_pred_hs[m, :len(y_lh_p), 0] = y_lh_p
        # Right Hand
        y_rh = video_hands_ground_truth['right_hand']
        y_rec_hs[m, :len(y_rh), 1] = y_rh
        rle = list(run_length_encoding(y_rh))
        y_rh_p = []
        for (_, previous_length), (next_label, _) in zip(rle[:-1], rle[1:]):
            y_rh_p += [next_label] * previous_length
        y_pred_hs[m, :len(y_rh_p), 1] = y_rh_p
    x_hs_segmentation = segmentation_from_output_class(y_rec_hs[:, downsampling - 1::downsampling],
                                                       segmentation_type='input')
    xs.append(x_hs_segmentation)
    if not test_data:
        y_rec_hs = y_rec_hs[:, downsampling - 1::downsampling]
        y_pred_hs = y_pred_hs[:, downsampling - 1::downsampling]
    y_hs_segmentation = segmentation_from_output_class(y_rec_hs, segmentation_type='output')
    ys = [y_rec_hs, y_pred_hs, y_hs_segmentation]
    return xs, ys


def assemble_mphoi_frame_level_recurrent_human(data, downsampling: int = 1, test_data: bool = False):
    # Input
    xs_h1, xs_h2 = [], []
    xs_h1p, xs_h2p = [], []
    x_obb= []
    max_len, max_len_downsampled = 0, 0
    for Human1, Human2, _, _, _, _, objects_bounding_box, Human1_pose, Human2_pose in data:
        # human feature
        max_len = max(max_len, Human1.shape[0])
        Human1 = Human1[downsampling - 1::downsampling]
        Human2 = Human2[downsampling - 1::downsampling]
        max_len_downsampled = max(max_len_downsampled, Human1.shape[0])
        xs_h1.append(Human1)
        xs_h2.append(Human2)
        # human pose
        Human1_pose = Human1_pose[downsampling - 1::downsampling]/1000
        Human2_pose = Human2_pose[downsampling - 1::downsampling]/1000
        xs_h1p.append(Human1_pose)
        xs_h2p.append(Human2_pose)
        # objects bounding box
        objects_bounding_box = objects_bounding_box[downsampling - 1::downsampling]/1000
        x_obb.append(objects_bounding_box)
    # xs_obb: reshape the object bounding box
    bbb = list()
    for i, video in enumerate(x_obb):
        bb = list()
        for j, frame in enumerate(video):
            b = np.zeros((4,4))  # max_no_objects = 4
            n = len(frame)
            if n!=4:
                b[0:n,] = frame 
            else:
                b = frame
            b = b.reshape(8,2)
            bb.append(b)
        bbb.append(bb)
    xs_obb = bbb
    # add context features to xs_h1, xs_h2
    hhh1 = list()
    hhh2 = list()
    keypoints = [1, 2, 4, 6, 7, 11, 13, 14, 27] # upper body keypoints
    for i, video in enumerate(xs_h1):
        hh1 = list()
        hh2 = list()
        for j, frame in enumerate(video):
            h1p = xs_h1p[i][j][keypoints]     # Human1 pose  (9*2)
            h2p = xs_h2p[i][j][keypoints]     # Human2 pose  (9*2)
            obb = xs_obb[i][j]                # object bounding box (8*2)
            # velocity
            if j+1 < len(video):
                next_h1p = xs_h1p[i][j+1][keypoints]
                h1p_velo = (next_h1p - h1p)*100
                next_h2p = xs_h2p[i][j+1][keypoints]
                h2p_velo = (next_h2p - h2p)*100
                next_obb = xs_obb[i][j+1]
                obb_velo = (next_obb - obb)*100
            else:
                h1p_velo = np.zeros((9, 2))
                h2p_velo = np.zeros((9, 2))
                obb_velo = np.zeros((8, 2))
            h1pvelo = np.hstack((h1p, h1p_velo)) # 9*4
            h2pvelo = np.hstack((h2p, h2p_velo)) # 9*4
            obbvelo = np.hstack((obb, obb_velo)) # 8*4
            lpv = h1pvelo.reshape(1, -1)
            rpv = h2pvelo.reshape(1, -1)
            obv = obbvelo.reshape(1, -1)
            # concatenate all fetures
            args = (lpv[0], rpv[0], obv[0])
            context = np.concatenate(args) # 104
            h1_con = np.concatenate((frame, context)) # 2152
            h2_con = np.concatenate((xs_h2[i][j], context)) # 2152
            hh1.append(h1_con)
            hh2.append(h2_con)
        hh1 = np.array(hh1)
        hh2 = np.array(hh2)
        hhh1.append(hh1)
        hhh2.append(hh2)
    xs_h1 = hhh1
    xs_h2 = hhh2
    feature_size = xs_h1[0].shape[-1]
    x_hs = np.full([len(xs_h1), max_len_downsampled, 2, feature_size], fill_value=np.nan, dtype=np.float32)
    for m, (h1, h2) in enumerate(zip(xs_h1, xs_h2)):
        x_hs[m, :len(h1), 0] = h1
        x_hs[m, :len(h2), 1] = h2
    xs = [x_hs]
    # Output
    y_rec_hs = np.full([len(x_hs), max_len, 2], fill_value=-1, dtype=np.int64)
    y_pred_hs = np.full_like(y_rec_hs, fill_value=-1)
    for m, (_, _, _, video_hands_ground_truth, _, _, _, _, _) in enumerate(data):
        # Human1
        y_h1 = video_hands_ground_truth['Human1']
        y_rec_hs[m, :len(y_h1), 0] = y_h1
        rle = list(run_length_encoding(y_h1))
        y_h1_p = []
        for (_, previous_length), (next_label, _) in zip(rle[:-1], rle[1:]):
            y_h1_p += [next_label] * previous_length
        y_pred_hs[m, :len(y_h1_p), 0] = y_h1_p
        # Human2
        y_h2 = video_hands_ground_truth['Human2']
        y_rec_hs[m, :len(y_h2), 1] = y_h2
        rle = list(run_length_encoding(y_h2))
        y_h2_p = []
        for (_, previous_length), (next_label, _) in zip(rle[:-1], rle[1:]):
            y_h2_p += [next_label] * previous_length
        y_pred_hs[m, :len(y_h2_p), 1] = y_h2_p
    x_hs_segmentation = segmentation_from_output_class(y_rec_hs[:, downsampling - 1::downsampling],
                                                       segmentation_type='input')
    xs.append(x_hs_segmentation)
    if not test_data:
        y_rec_hs = y_rec_hs[:, downsampling - 1::downsampling]
        y_pred_hs = y_pred_hs[:, downsampling - 1::downsampling]
    y_hs_segmentation = segmentation_from_output_class(y_rec_hs, segmentation_type='output')
    ys = [y_rec_hs, y_pred_hs, y_hs_segmentation]
    return xs, ys


def segmentation_from_output_class(y, segmentation_type='input'):
    x_segmentation = np.array(y, dtype=np.float32)
    original_missing_mask = y == -1.0
    x_segmentation = np.where(original_missing_mask, np.nan, x_segmentation)
    end_indices = (x_segmentation[:, 1:] - x_segmentation[:, :-1]) != 0.0
    end_indices = np.concatenate([end_indices, np.full_like(end_indices, fill_value=True)[:, -1:]], axis=1)
    x_segmentation[end_indices] = 1.0
    x_segmentation[~end_indices & ~np.isnan(x_segmentation)] = 0.0
    x_segmentation[np.isnan(x_segmentation)] = 1.0
    if segmentation_type == 'output':
        x_segmentation[original_missing_mask] = -1.0
    return x_segmentation


def assemble_frame_level_recurrent_objects(data, downsampling: int = 1, test_data: bool = False):
    # Process input tensors (visual features)
    xs_objects = []
    max_len, max_num_objects = 0, 0
    max_len_downsampled = 0
    for _, object_features, _, _, _, _ in data:
        max_len = max(max_len, object_features.shape[0])
        max_num_objects = max(max_num_objects, object_features.shape[1])
        object_features = object_features[downsampling - 1::downsampling]
        max_len_downsampled = max(max_len_downsampled, object_features.shape[0])
        xs_objects.append(object_features)
    object_feature_size = xs_objects[-1].shape[-1]
    x_objects = np.full([len(xs_objects), max_len_downsampled, max_num_objects, object_feature_size],
                        fill_value=np.nan, dtype=np.float32)
    x_objects_mask = np.zeros([len(xs_objects), max_num_objects], dtype=np.float32)
    for m, x_o in enumerate(xs_objects):
        x_objects[m, :x_o.shape[0], :x_o.shape[1], :] = x_o
        x_objects_mask[m, :x_o.shape[1]] = 1.0
    xs = [x_objects, x_objects_mask]
    # Process output classes and extra tensors
    y_rec_affordance = np.full([x_objects.shape[0], max_len, max_num_objects], fill_value=-1, dtype=np.int64)
    y_pred_affordance = np.full_like(y_rec_affordance, fill_value=-1)
    for m, (_, _, _, _, _, video_data) in enumerate(data):
        for i, video_segment in enumerate(video_data):
            start_frame, end_frame = video_segment.start_frame, video_segment.end_frame
            if start_frame is None or end_frame is None:
                continue
            start_frame -= 1
            end_frame -= 1
            affordances = video_segment.object_affordance
            for object_id, object_affordance in affordances.items():
                y_rec_affordance[m, start_frame:end_frame + 1, object_id - 1] = object_affordance - 1
            next_affordances = video_segment.next_object_affordance
            for object_id, object_affordance in next_affordances.items():
                y_pred_affordance[m, start_frame:end_frame + 1, object_id - 1] = object_affordance - 1
    x_affordance_segmentation = segmentation_from_output_class(y_rec_affordance[:, downsampling - 1::downsampling],
                                                               segmentation_type='input')
    xs.append(x_affordance_segmentation)
    if not test_data:
        y_rec_affordance = y_rec_affordance[:, downsampling - 1::downsampling]
        y_pred_affordance = y_pred_affordance[:, downsampling - 1::downsampling]
    y_affordance_segmentation = segmentation_from_output_class(y_rec_affordance, segmentation_type='output')
    ys = [y_rec_affordance, y_pred_affordance, y_affordance_segmentation]
    return xs, ys


def assemble_bimanual_frame_level_recurrent_objects(data, downsampling: int = 1):
    xs_objects = []
    max_len, max_len_downsampled, max_num_objects = 0, 0, 0
    for _, _, objects, _, _, _, _, _, _ in data:
        max_len = max(max_len, objects.shape[0])
        max_num_objects = max(max_num_objects, objects.shape[1])
        objects = objects[downsampling - 1::downsampling]
        max_len_downsampled = max(max_len_downsampled, objects.shape[0])
        xs_objects.append(objects)
    feature_size = xs_objects[-1].shape[-1]
    x_objects = np.full([len(xs_objects), max_len_downsampled, max_num_objects, feature_size],
                        fill_value=np.nan, dtype=np.float32)
    x_objects_mask = np.zeros([len(xs_objects), max_num_objects], dtype=np.float32)
    for m, x_o in enumerate(xs_objects):
        x_objects[m, :x_o.shape[0], :x_o.shape[1], :] = x_o
        x_objects_mask[m, :x_o.shape[1]] = 1.0
    xs = [x_objects, x_objects_mask]
    return xs


def assemble_mphoi_frame_level_recurrent_objects(data, downsampling: int = 1):
    xs_objects = []
    max_len, max_len_downsampled, max_num_objects = 0, 0, 0
    for _, _, objects, _, _, _, _, _, _ in data:
        max_len = max(max_len, objects.shape[0])
        max_num_objects = max(max_num_objects, objects.shape[1])
        objects = objects[downsampling - 1::downsampling]
        max_len_downsampled = max(max_len_downsampled, objects.shape[0])
        xs_objects.append(objects)
    feature_size = xs_objects[-1].shape[-1]
    x_objects = np.full([len(xs_objects), max_len_downsampled, max_num_objects, feature_size],
                        fill_value=np.nan, dtype=np.float32)
    x_objects_mask = np.zeros([len(xs_objects), max_num_objects], dtype=np.float32)
    for m, x_o in enumerate(xs_objects):
        x_objects[m, :x_o.shape[0], :x_o.shape[1], :] = x_o
        x_objects_mask[m, :x_o.shape[1]] = 1.0
    xs = [x_objects, x_objects_mask]
    return xs


def compute_centroid(bounding_boxes):
    """Compute centroids of an array of bounding boxes.

    Arguments:
        bounding_boxes - ndarray of shape (num_bounding_boxes, 4).
    Returns:
        An ndarray of shape (num_bounding_boxes, 2).
    """
    x = (bounding_boxes[..., :1] + bounding_boxes[..., 2:3]) / 2
    y = (bounding_boxes[..., 1:2] + bounding_boxes[..., 3:4]) / 2
    return np.concatenate([x, y], axis=-1)


def apply_positional_encoding(x_features, dimension=2048):
    """Given x, y locations, compute the positional encoding of it.

    Arguments:
        x_features - ndarray of shape (*, 2).
        dimension - Final dimension of the positional encoding feature. Must be divisible by 4.
    Returns
        An ndarray of shape (*, dimension) containing the positional encoding of the input features.
    """
    m = np.array([1e4], dtype=np.float32)
    indices = 4 * np.arange(0, dimension // 4, dtype=np.float32) / dimension
    factor = 1 / (m ** indices)
    result = []
    for coord_index in range(2):
        pre_result = x_features[..., coord_index:coord_index + 1] * factor
        result.append(np.sin(pre_result))
        result.append(np.cos(pre_result))
    result = np.concatenate(result, axis=-1)
    return result


def assemble_human_object_distances(data, downsampling: int = 1):
    cad120_dims = np.array([640, 480], dtype=np.float32)
    max_len, max_num_objects = 0, 0
    all_dists = []
    for _, _, skeleton_bounding_box, objects_bounding_box, _, _ in data:
        skeleton_bounding_box = skeleton_bounding_box[downsampling - 1::downsampling]
        objects_bounding_box = objects_bounding_box[downsampling - 1::downsampling]
        objects_centroid = compute_centroid(objects_bounding_box)
        skeleton_centroid = compute_centroid(skeleton_bounding_box) / cad120_dims
        dists = np.linalg.norm(objects_centroid - np.expand_dims(skeleton_centroid, axis=1),
                               ord=2, axis=-1)
        dists = np.expand_dims(dists, axis=1)
        all_dists.append(dists)
        max_len = max(max_len, objects_bounding_box.shape[0])
        max_num_objects = max(max_num_objects, objects_bounding_box.shape[1])
    tensor_shape = [len(all_dists), max_len, 1, max_num_objects]
    x_ho_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)
    for m, x_ho_d in enumerate(all_dists):
        x_ho_dists[m, :x_ho_d.shape[0], :, :x_ho_d.shape[2]] = x_ho_d
    return x_ho_dists


def assemble_bimanual_human_human_distances(data, downsampling: int = 1):
    bimanual_dims = np.array([640, 480], dtype=np.float32)
    max_len, max_num_humans = 0, 2
    all_dists = []
    for _, _, _, _, lh_bb, rh_bb, _, _, _ in data:
        lh_bb = lh_bb[downsampling - 1::downsampling]
        lh_centroids = compute_centroid(lh_bb) / bimanual_dims
        rh_bb = rh_bb[downsampling - 1::downsampling]
        rh_centroids = compute_centroid(rh_bb) / bimanual_dims
        dists = np.linalg.norm(lh_centroids - rh_centroids, ord=2, axis=-1)
        all_dists.append(dists)
        max_len = max(max_len, lh_bb.shape[0])
    tensor_shape = [len(all_dists), max_len, max_num_humans, max_num_humans]
    x_hh_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)
    for m, x_hh_d in enumerate(all_dists):
        x_hh_dists[m, :x_hh_d.shape[0], 0, 1] = x_hh_d
        x_hh_dists[m, :x_hh_d.shape[0], 1, 0] = x_hh_d
        x_hh_dists[m, :x_hh_d.shape[0], 0, 0] = 0.0
        x_hh_dists[m, :x_hh_d.shape[0], 1, 1] = 0.0
    return x_hh_dists


def assemble_bimanual_human_object_distances(data, downsampling: int = 1):
    bimanual_dims = np.array([640, 480], dtype=np.float32)
    max_len, max_num_humans, max_num_objects = 0, 2, 0
    lh_dists, rh_dists = [], []
    for _, _, _, _, lh_bb, rh_bb, obj_bbs, _, _ in data:
        lh_bb = lh_bb[downsampling - 1::downsampling]
        lh_centroids = compute_centroid(lh_bb) / bimanual_dims
        rh_bb = rh_bb[downsampling - 1::downsampling]
        rh_centroids = compute_centroid(rh_bb) / bimanual_dims
        obj_bbs = obj_bbs[downsampling - 1::downsampling]
        obj_centroids = compute_centroid(obj_bbs) / bimanual_dims
        lh_objs_dists = np.linalg.norm(obj_centroids - np.expand_dims(lh_centroids, axis=1), ord=2, axis=-1)
        lh_dists.append(lh_objs_dists)
        rh_objs_dists = np.linalg.norm(obj_centroids - np.expand_dims(rh_centroids, axis=1), ord=2, axis=-1)
        rh_dists.append(rh_objs_dists)
        max_len = max(max_len, lh_bb.shape[0])
        max_num_objects = max(max_num_objects, obj_bbs.shape[1])
    tensor_shape = [len(lh_dists), max_len, max_num_humans, max_num_objects]
    x_ho_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)
    for m, (lh_d, rh_d) in enumerate(zip(lh_dists, rh_dists)):
        x_ho_dists[m, :lh_d.shape[0], 0, :lh_d.shape[1]] = lh_d
        x_ho_dists[m, :rh_d.shape[0], 1, :rh_d.shape[1]] = rh_d
    return x_ho_dists


def assemble_bimanual_object_object_distances(data, downsampling: int = 1):
    bimanual_dims = np.array([640, 480], dtype=np.float32)
    max_len, max_num_objects = 0, 0
    all_dists = []
    for _, _, _, _, _, _, obj_bbs, _, _ in data:
        obj_bbs = obj_bbs[downsampling - 1::downsampling]
        objs_centroid = compute_centroid(obj_bbs) / bimanual_dims
        num_objects = objs_centroid.shape[1]
        dists = []
        for k in range(num_objects):
            kth_object_centroid = objs_centroid[:, k:k + 1]
            kth_dist = np.linalg.norm(objs_centroid - kth_object_centroid, ord=2, axis=-1)
            dists.append(kth_dist)
        dists = np.stack(dists, axis=1)
        all_dists.append(dists)
        max_len = max(max_len, obj_bbs.shape[0])
        max_num_objects = max(max_num_objects, num_objects)
    tensor_shape = [len(all_dists), max_len, max_num_objects, max_num_objects]
    x_oo_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)
    for m, x_oo_d in enumerate(all_dists):
        x_oo_dists[m, :x_oo_d.shape[0], :x_oo_d.shape[1], :x_oo_d.shape[2]] = x_oo_d
    return x_oo_dists


def assemble_object_object_distances(data, downsampling: int = 1):
    max_len, max_num_objects = 0, 0
    all_dists = []
    for _, _, _, objects_bounding_box, _, _ in data:
        objects_bounding_box = objects_bounding_box[downsampling - 1::downsampling]
        objects_centroid = compute_centroid(objects_bounding_box)
        num_objects = objects_centroid.shape[1]
        dists = []
        for k in range(num_objects):
            kth_object_centroid = objects_centroid[:, k:k + 1]
            kth_dist = np.linalg.norm(objects_centroid - kth_object_centroid, ord=2, axis=-1)
            dists.append(kth_dist)
        dists = np.stack(dists, axis=1)
        all_dists.append(dists)
        max_len = max(max_len, objects_bounding_box.shape[0])
        max_num_objects = max(max_num_objects, objects_bounding_box.shape[1])
    tensor_shape = [len(all_dists), max_len, max_num_objects, max_num_objects]
    x_oo_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)
    for m, x_oo_d in enumerate(all_dists):
        x_oo_dists[m, :x_oo_d.shape[0], :x_oo_d.shape[1], :x_oo_d.shape[2]] = x_oo_d
    return x_oo_dists


def assemble_mphoi_human_human_distances(data, downsampling: int = 1):
    mphoi_dims = np.array([3840, 2160], dtype=np.float32)
    max_len, max_num_humans = 0, 2
    all_dists = []
    for _, _, _, _, h1_bb, h2_bb, _, _, _ in data:
        h1_bb = h1_bb[downsampling - 1::downsampling]
        h1_centroids = compute_centroid(h1_bb) / mphoi_dims
        h2_bb = h2_bb[downsampling - 1::downsampling]
        h2_centroids = compute_centroid(h2_bb) / mphoi_dims
        dists = np.linalg.norm(h1_centroids - h2_centroids, ord=2, axis=-1)
        all_dists.append(dists)
        max_len = max(max_len, h1_bb.shape[0])
    tensor_shape = [len(all_dists), max_len, max_num_humans, max_num_humans]
    x_hh_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)
    for m, x_hh_d in enumerate(all_dists):
        x_hh_dists[m, :x_hh_d.shape[0], 0, 1] = x_hh_d
        x_hh_dists[m, :x_hh_d.shape[0], 1, 0] = x_hh_d
        x_hh_dists[m, :x_hh_d.shape[0], 0, 0] = 0.0
        x_hh_dists[m, :x_hh_d.shape[0], 1, 1] = 0.0
    return x_hh_dists


def assemble_mphoi_human_object_distances(data, downsampling: int = 1):
    mphoi_dims = np.array([3840, 2160], dtype=np.float32)
    max_len, max_num_humans, max_num_objects = 0, 2, 0
    h1_dists, h2_dists = [], []
    for _, _, _, _, h1_bb, h2_bb, obj_bbs, _, _ in data:
        h1_bb = h1_bb[downsampling - 1::downsampling]
        h1_centroids = compute_centroid(h1_bb) / mphoi_dims
        h2_bb = h2_bb[downsampling - 1::downsampling]
        h2_centroids = compute_centroid(h2_bb) / mphoi_dims
        obj_bbs = obj_bbs[downsampling - 1::downsampling]
        obj_centroids = compute_centroid(obj_bbs) / mphoi_dims
        h1_objs_dists = np.linalg.norm(obj_centroids - np.expand_dims(h1_centroids, axis=1), ord=2, axis=-1)
        h1_dists.append(h1_objs_dists)
        h2_objs_dists = np.linalg.norm(obj_centroids - np.expand_dims(h2_centroids, axis=1), ord=2, axis=-1)
        h2_dists.append(h2_objs_dists)
        max_len = max(max_len, h1_bb.shape[0])
        max_num_objects = max(max_num_objects, obj_bbs.shape[1])
    tensor_shape = [len(h1_dists), max_len, max_num_humans, max_num_objects]
    x_ho_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)
    for m, (h1_d, h2_d) in enumerate(zip(h1_dists, h2_dists)):
        x_ho_dists[m, :h1_d.shape[0], 0, :h1_d.shape[1]] = h1_d
        x_ho_dists[m, :h2_d.shape[0], 1, :h2_d.shape[1]] = h2_d
    return x_ho_dists


def assemble_mphoi_object_object_distances(data, downsampling: int = 1):
    mphoi_dims = np.array([3840, 2160], dtype=np.float32)
    max_len, max_num_objects = 0, 0
    all_dists = []
    for _, _, _, _, _, _, obj_bbs, _, _ in data:
        obj_bbs = obj_bbs[downsampling - 1::downsampling]
        objs_centroid = compute_centroid(obj_bbs) / mphoi_dims
        num_objects = objs_centroid.shape[1]
        dists = []
        for k in range(num_objects):
            kth_object_centroid = objs_centroid[:, k:k + 1]
            kth_dist = np.linalg.norm(objs_centroid - kth_object_centroid, ord=2, axis=-1)
            dists.append(kth_dist)
        dists = np.stack(dists, axis=1)
        all_dists.append(dists)
        max_len = max(max_len, obj_bbs.shape[0])
        max_num_objects = max(max_num_objects, num_objects)
    tensor_shape = [len(all_dists), max_len, max_num_objects, max_num_objects]
    x_oo_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)
    for m, x_oo_d in enumerate(all_dists):
        x_oo_dists[m, :x_oo_d.shape[0], :x_oo_d.shape[1], :x_oo_d.shape[2]] = x_oo_d
    return x_oo_dists


def assemble_num_steps(data, downsampling: int = 1):
    xs_steps = []
    for x, *_ in data:
        num_steps = len(x[downsampling - 1::downsampling])
        xs_steps.append(num_steps)
    xs_steps = np.array(xs_steps, dtype=np.float32)
    return xs_steps


def select_model_data_fetcher(model_name: str, model_input_type: str, **kwargs):
    model_to_data_fetcher = {
        'bimanual_baseline': partial(fetchers.multiple_input_multiple_output, n=3),
        'cad120_baseline': partial(fetchers.multiple_input_multiple_output, n=3),
        '2G-GCN': partial(gcn_fetcher, **kwargs),
    }
    return model_to_data_fetcher[model_name]


def select_model_data_feeder(model_name: str, model_input_type: str, **kwargs):
    model_to_data_forwarder = {
        'bimanual_baseline': forwarders.multiple_input_forward,
        'cad120_baseline': forwarders.multiple_input_forward,
        '2G-GCN': partial(gcn_forward, **kwargs),
    }
    return model_to_data_forwarder[model_name]


def gcn_forward(model, data, **kwargs):
    input_human_segmentation = kwargs.get('input_human_segmentation', False)
    impose_segmentation_pattern = kwargs.get('impose_segmentation_pattern', 0)
    if impose_segmentation_pattern:
        if impose_segmentation_pattern == 1:
            human_segmentation = torch.ones(data[0].size()[:-1], dtype=data[0].dtype, device=data[0].device)
        else:
            raise ValueError(f'Segmentation pattern can only be 1, not {impose_segmentation_pattern}')
    elif input_human_segmentation:
        human_segmentation = data[3]
    else:
        human_segmentation = None
    model_kwargs = {
        'x_human': data[0],
        'x_objects': data[1],
        'objects_mask': data[2],
        'human_segmentation': human_segmentation,
    }
    dataset_name = kwargs.get('dataset_name', 'cad120')
    if dataset_name == 'cad120':
        input_object_segmentation = kwargs.get('input_object_segmentation', False)
        if impose_segmentation_pattern:
            if impose_segmentation_pattern == 1:
                object_segmentation = torch.ones(data[1].size()[:-1], dtype=data[1].dtype, device=data[1].device)
            else:
                raise ValueError(f'Segmentation pattern can only be 1, not {impose_segmentation_pattern}')
        elif input_object_segmentation:
            object_segmentation = data[4]
        else:
            object_segmentation = None
        model_kwargs['objects_segmentation'] = object_segmentation
        human_human_distances = human_object_distances = object_object_distances = None
        if kwargs.get('make_attention_distance_based', False):
            human_object_distances = data[5]
            object_object_distances = data[6]
    else:
        human_human_distances = human_object_distances = object_object_distances = None
        if kwargs.get('make_attention_distance_based', False):
            human_human_distances = data[4]
            human_object_distances = data[5]
            object_object_distances = data[6]
    model_kwargs['human_human_distances'] = human_human_distances
    model_kwargs['human_object_distances'] = human_object_distances
    model_kwargs['object_object_distances'] = object_object_distances
    model_kwargs['steps_per_example'] = data[7]
    model_kwargs['inspect_model'] = kwargs.get('inspect_model', False)
    return model(**model_kwargs)


def gcn_fetcher(dataset, device, **kwargs):
    data = []
    data.append(dataset[0].to(device))
    data.append(dataset[1].to(device))
    data.append(dataset[2].to(device))
    if kwargs.get('input_human_segmentation', False):
        data.append(dataset[3].to(device))
    else:
        data.append(dataset[3])
    dataset_name = kwargs.get('dataset_name', 'cad120')
    if dataset_name == 'cad120':
        if kwargs.get('input_object_segmentation', False):
            data.append(dataset[4].to(device))
        else:
            data.append(dataset[4])
        if kwargs.get('make_attention_distance_based', False):
            data.append(dataset[5].to(device))
            data.append(dataset[6].to(device))
        else:
            data.append(dataset[5])
            data.append(dataset[6])
        targets = [target.to(device) for target in dataset[8:]]
    else:  # bimanual & mphoi
        if kwargs.get('make_attention_distance_based', False):
            data.append(dataset[4].to(device))
            data.append(dataset[5].to(device))
            data.append(dataset[6].to(device))
        else:
            data.append(dataset[4])
            data.append(dataset[5])
            data.append(dataset[6])
        targets = [target.to(device) for target in dataset[8:]]
    data.append(dataset[7].to(device))
    return data, targets


def determine_num_classes(model_name: str, model_input_type: str, dataset_name: str):
    if model_name in {'2G-GCN', 'bimanual_baseline', 'cad120_baseline'}:
        if dataset_name.lower() == 'bimanual':
            return 14, None
        elif dataset_name.lower() == 'mphoi':
            return 13, None
        else:
            return 10, 12
    if model_input_type == 'human':
        return 10
    else:
        return 12


def input_size_from_data_loader(data_loader: DataLoader, model_name: str, model_input_type: str):
    if model_name in {'2G-GCN', 'bimanual_baseline', 'cad120_baseline'}:
        human_input_size = data_loader.dataset[0][0].size(-1)
        object_input_size = data_loader.dataset[0][1].size(-1)
        return human_input_size, object_input_size
    else:
        raise ValueError(f'{model_name} is not an option for model name.')

