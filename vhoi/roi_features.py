import argparse
import os

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import zarr

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from pyrutils.geometric import alter_bounding_boxes_size, bounding_boxes_from_keypoints
from vhoi.cad120 import convert_from_world_to_image_coordinates, read_skeleton


def get_predictor(config_filepath):
    cfg = get_cfg()
    cfg.merge_from_file(config_filepath)
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weights
    cfg.MODEL.WEIGHTS = 'http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl'
    predictor = DefaultPredictor(cfg)
    return predictor


def extract_roi_features(predictor, raw_image, raw_boxes=None, verbose=False):
    """Extract ROI features from an image using a pre-trained predictor.

    This function extracts ROI features from bounding boxes in an image.

    Arguments:
        predictor - A detectron2.engine.DefaultPredictor object.
        raw_image - Numpy array of shape (raw_height, raw_width, 3) and dtype uint8 containing the BGR values of
            the image.
        raw_boxes - Numpy array of shape (num_boxes, 4) containing the top-left (x1, y1) and bottom-right (x2, y2)
            coordinates of the bounding boxes in raw_image coordinates. If None, use the RPN proposals as bounding
            boxes.
        verbose - If True, print extra information along the process.
    """
    # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    raw_height, raw_width = raw_image.shape[:2]
    if verbose:
        print('Original image size:', (raw_height, raw_width))
    with torch.no_grad():
        # Pre-processing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        if verbose:
            print('Transformed image size:', image.shape[:2])
        # Scale boxes
        new_height, new_width = image.shape[:2]
        scale_x = new_width / raw_width
        scale_y = new_height / raw_height
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
        # ----
        image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
        inputs = [{'image': image, 'height': raw_height, 'width': raw_width}]
        images = predictor.model.preprocess_image(inputs)
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[feature_name] for feature_name in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(features, proposal_boxes)
        roi_features = box_features.mean(dim=[2, 3])  # pooled to 1x1
        if verbose:
            print('Pooled RoI features size:', roi_features.shape)
        return roi_features


def extract_instances_object(predictor, roi_features, raw_boxes, raw_height, raw_width, verbose=True):
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    with torch.no_grad():
        pred_class_logits, _ = predictor.model.roi_heads.box_predictor(roi_features)
    if verbose:
        print('Predicted logits size:', pred_class_logits.shape)
    pred_class_prob = nn.functional.softmax(pred_class_logits, dim=-1)
    pred_scores, pred_classes = pred_class_prob[..., :-1].max(dim=-1)
    # Detectron2 Formatting (for visualization only)
    instances = Instances(
        image_size=(raw_height, raw_width),
        pred_boxes=raw_boxes,
        scores=pred_scores,
        pred_classes=pred_classes
    )
    return instances


def read_raw_boxes(annotation_dir, video_id):
    """Read all object bounding boxes from a video.

    Returns:
        A numpy array of shape (num_frames, num_objects, 4), containing the x_min, y_min, x_max, y_max values of
        the bounding boxes.
    """
    object_files = sorted(filename for filename in os.listdir(annotation_dir)
                          if filename.startswith(video_id) and 'obj' in filename)
    raw_boxes = []
    for object_file in object_files:
        raw_single_boxes, last_frame = [], 0
        object_file_path = os.path.join(annotation_dir, object_file)
        with open(object_file_path, mode='r') as f:
            for line in f:
                line = line.split(sep=',')[:-1]
                frame_num, obj_coords = int(line[0]), line[2:6]
                if last_frame == frame_num:
                    continue
                last_frame = frame_num
                obj_coords = [float(obj_coord) for obj_coord in obj_coords]
                if obj_coords:
                    raw_single_boxes.append(obj_coords)
        raw_single_boxes = np.stack(raw_single_boxes, axis=0)
        raw_boxes.append(raw_single_boxes)
    raw_boxes = np.stack(raw_boxes, axis=1)
    raw_boxes = np.where(raw_boxes == 0.0, np.nan, raw_boxes)
    return raw_boxes


def read_skeleton_raw_box(annotation_dir, video_id, upper_body_only=True):
    """Read all skeleton bounding boxes from a video.

    Returns:
        A numpy array of shape (num_frames, 4), containing the x_min, y_min, x_max, y_max values of the skeleton
        bounding boxes.
    """
    image_skeleton = read_skeleton_image_pose(annotation_dir, video_id, upper_body_only=upper_body_only)
    # CAD-120 skeleton bounding box misses a lot of the person, so we enlarge it by 20%.
    skeleton_bounding_box = []
    for skeleton in image_skeleton:
        skeleton = alter_bounding_boxes_size(bounding_boxes_from_keypoints(skeleton), alter_percentage=120)
        skeleton_bounding_box.append(skeleton)
    skeleton_bounding_box = np.stack(skeleton_bounding_box, axis=0)
    return skeleton_bounding_box


def read_skeleton_image_pose(annotation_dir, video_id, upper_body_only=True):
    filepath = os.path.join(annotation_dir, video_id + '.txt')
    world_skeleton = read_skeleton(filepath)
    world_skeleton, world_skeleton_conf = world_skeleton[..., :-1], world_skeleton[..., -1:]
    world_skeleton_shape = world_skeleton.shape
    world_skeleton = world_skeleton.reshape(-1, world_skeleton_shape[-1])
    image_skeleton = convert_from_world_to_image_coordinates(world_skeleton)
    image_skeleton = image_skeleton.reshape(*world_skeleton_shape[:-1], -1)
    image_skeleton = image_skeleton * world_skeleton_conf
    if upper_body_only:
        upper_body_joints = [0, 1, 2, 3, 4, 5, 6, 11, 12]
        image_skeleton = image_skeleton[:, upper_body_joints]
    image_skeleton = np.where(image_skeleton == 0.0, np.nan, image_skeleton)
    return image_skeleton


def extract_cad120_visual_features_from_video(images_dir, annotation_dir, video_id, predictor):
    raw_obj_boxes = read_raw_boxes(annotation_dir, video_id)
    raw_skeleton_boxes = read_skeleton_raw_box(annotation_dir, video_id, upper_body_only=True)
    raw_skeleton_boxes = np.expand_dims(raw_skeleton_boxes, axis=1)
    rgb_files = sorted(filename for filename in os.listdir(images_dir) if filename.startswith('RGB'))
    num_boxes_frames, num_skeleton_frames, num_frames = len(raw_obj_boxes), len(raw_skeleton_boxes), len(rgb_files)
    error_msg = f'Mismatch between number of read frames. Video {images_dir}' \
                f'\nObject: {num_boxes_frames}\nSkeleton: {num_skeleton_frames}\nRGB: {num_frames}'
    assert num_boxes_frames == num_skeleton_frames == num_frames, error_msg
    num_boxes = raw_obj_boxes.shape[1]
    skeleton_features = np.full([num_frames, 2048], fill_value=np.nan, dtype=np.float32)
    obj_features = np.full([num_frames, num_boxes, 2048], fill_value=np.nan, dtype=np.float32)
    for frame_num, rgb_file in enumerate(rgb_files):
        raw_image_path = os.path.join(images_dir, rgb_file)
        raw_image = cv.imread(raw_image_path)
        raw_obj_boxes_frame = raw_obj_boxes[frame_num]
        raw_skeleton_box_frame = raw_skeleton_boxes[frame_num]
        raw_boxes_frame = np.concatenate([raw_skeleton_box_frame, raw_obj_boxes_frame], axis=0)
        roi_features = extract_roi_features(predictor, raw_image, raw_boxes_frame, verbose=False)
        if np.any(~np.isnan(raw_skeleton_box_frame)):
            skeleton_features[frame_num] = roi_features[0].cpu().numpy()
        for k, obj_bbs in enumerate(raw_obj_boxes_frame):
            if np.any(~np.isnan(obj_bbs)):
                obj_features[frame_num, k] = roi_features[k + 1].cpu().numpy()
    return skeleton_features, obj_features


def extract_cad120_bounding_boxes(annotation_dir: str, video_id: str):
    """Extract bounding boxes of skeleton and objects in a video.

    Arguments:
        annotation_dir - Directory containing information about the skeleton pose and object bounding boxes throughout
            the specified video.
        video_id - A unique identifier for the video.
    Returns:
        Two numpy arrays. The first contains the bounding boxes of the skeleton and is of shape (num_frames, 4), and
        the second contains the bounding boxes of the objects and is of shape (num_frames, num_objects, 4).
    """
    raw_skeleton_boxes = read_skeleton_raw_box(annotation_dir, video_id, upper_body_only=True)
    raw_obj_boxes = read_raw_boxes(annotation_dir, video_id)
    return raw_skeleton_boxes, raw_obj_boxes


def extract_cad120_visual_features(args):
    all_images_dir = args.all_images_dir
    all_annotation_dir = args.all_annotation_dir
    save_file = args.save_file
    add_positional_features = args.add_positional_features

    store = zarr.DirectoryStore(save_file)
    root = zarr.group(store=store, overwrite=False)
    predictor = get_predictor(args.config_filepath)
    images_subject_folders = sorted(os.listdir(all_images_dir))
    annotation_subject_folders = sorted(os.listdir(all_annotation_dir))
    for images_subject_folder, annotation_subject_folder in zip(images_subject_folders, annotation_subject_folders):
        images_subject_dir = os.path.join(all_images_dir, images_subject_folder)
        images_activity_folders = sorted(os.listdir(images_subject_dir))
        annotation_subject_dir = os.path.join(all_annotation_dir, annotation_subject_folder)
        annotation_activity_folders = sorted(os.listdir(annotation_subject_dir))
        for images_activity_folder, annotation_activity_folder in zip(images_activity_folders,
                                                                      annotation_activity_folders):
            images_activity_dir = os.path.join(images_subject_dir, images_activity_folder)
            video_ids = set(os.listdir(images_activity_dir))
            annotation_activity_dir = os.path.join(annotation_subject_dir, annotation_activity_folder)
            for video_id in video_ids:
                if video_id not in root:
                    images_dir = os.path.join(images_activity_dir, video_id)
                    skeleton_features, object_features = \
                        extract_cad120_visual_features_from_video(images_dir, annotation_activity_dir,
                                                                  video_id, predictor)
                    features_group = root.create_group(video_id)
                    features_group.array('skeleton', skeleton_features, chunks=False, dtype=np.float32)
                    features_group.array('objects', object_features, chunks=False, dtype=np.float32)
                    print(f'Processed features for video {images_dir}')
            if add_positional_features:
                for video_id in video_ids:
                    skeleton_boxes, objects_boxes = extract_cad120_bounding_boxes(annotation_activity_dir, video_id)
                    features_group = root[video_id]
                    if 'skeleton_bounding_box' not in features_group:
                        features_group.array('skeleton_bounding_box', skeleton_boxes, chunks=False, dtype=np.float32)
                    if 'objects_bounding_box' not in features_group:
                        features_group.array('objects_bounding_box', objects_boxes, chunks=False, dtype=np.float32)
                    if 'skeleton_2d_pose' not in features_group:
                        skeleton_2d_pose = read_skeleton_image_pose(annotation_activity_dir, video_id,
                                                                    upper_body_only=True)
                        features_group.array('skeleton_2d_pose', skeleton_2d_pose, chunks=False, dtype=np.float32)


def _extract_bimanual_visual_features(predictor, dirpath, tracked_hands, tracked_objects):
    rgb_files = sorted(filename for filename in os.listdir(dirpath) if filename.endswith('.png'))
    num_frames, num_objects = len(rgb_files), len(tracked_objects[0])
    error_msg = f'Mismatch between number of read frames. Video {dirpath}' \
                f'\nObjects: {len(tracked_objects)}\nHands: {len(tracked_hands)}\nRGB: {num_frames}'
    assert len(tracked_objects) == len(tracked_hands) == len(rgb_files), error_msg
    lh_vf = np.full([num_frames, 2048], fill_value=np.nan, dtype=np.float32)
    rh_vf = np.full([num_frames, 2048], fill_value=np.nan, dtype=np.float32)
    objs_vf = np.full([num_frames, num_objects, 2048], fill_value=np.nan, dtype=np.float32)
    for frame_num, rgb_file in enumerate(rgb_files):
        image_path = os.path.join(dirpath, rgb_file)
        image = cv.imread(image_path)
        hands_bbs = tracked_hands[frame_num]
        objs_bbs = tracked_objects[frame_num]
        bbs = np.concatenate([hands_bbs, objs_bbs], axis=0)
        roi_features = extract_roi_features(predictor, image, bbs, verbose=False)
        if np.any(~np.isnan(hands_bbs[0])):
            lh_vf[frame_num] = roi_features[0].cpu().numpy()
        if np.any(~np.isnan(hands_bbs[1])):
            rh_vf[frame_num] = roi_features[1].cpu().numpy()
        for k, obj_bbs in enumerate(objs_bbs):
            if np.any(~np.isnan(obj_bbs)):
                objs_vf[frame_num, k] = roi_features[k + 2].cpu().numpy()
    return lh_vf, rh_vf, objs_vf


def extract_bimanual_visual_features(args):
    rgbd_dir = args.rgbd_dir
    tracked_objects_dir = args.tracked_objects_dir
    tracked_hands_dir = args.tracked_hands_dir
    save_root = args.save_root

    save_path = os.path.join(save_root, 'faster_rcnn.zarr')
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=False)
    predictor = get_predictor(args.config_filepath)
    for dirpath, dirnames, filenames in os.walk(rgbd_dir):
        if dirpath.endswith('/rgb'):
            subject, task, take = dirpath.split('/')[-4:-1]
            video_id = f'{subject}-{task}-{take}'
            if video_id not in root:
                filepath = os.path.join(tracked_hands_dir, subject, task, take + '.npy')
                tracked_hands = np.load(filepath)
                filepath = os.path.join(tracked_objects_dir, subject, task, take + '.npy')
                tracked_objects = np.load(filepath)
                lh_vf, rh_vf, objs_vf = _extract_bimanual_visual_features(predictor, dirpath,
                                                                          tracked_hands, tracked_objects)
                features_group = root.create_group(video_id)
                features_group.array('left_hand', lh_vf, chunks=False, dtype=np.float32)
                features_group.array('right_hand', rh_vf, chunks=False, dtype=np.float32)
                features_group.array('objects', objs_vf, chunks=False, dtype=np.float32)
                print(f'Processed visual features for video {video_id}')


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Functions to Extract Visual Features.')
    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')
    # Extract Faster R-CNN 2D visual features from videos.
    # CAD-120 dataset.
    parser_cad120 = subparsers.add_parser('cad-120',
                                          help='Extract visual features for the CAD-120 dataset.')

    parser_cad120.add_argument('--all_images_dir', type=str, required=True,
                               help='Path to RGB-D_Images directory of CAD-120.')
    parser_cad120.add_argument('--all_annotation_dir', type=str, required=True,
                               help='Path to annotations directory of CAD-120.')
    parser_cad120.add_argument('--config_filepath', type=str, required=True,
                               help='Path to \'faster_rcnn_R_101_C4_caffe.yaml\' file.')
    parser_cad120.add_argument('--save_file', type=str, required=True,
                               help='Path to .zarr \'file\' to create hierarchical structure to save the features.')
    parser_cad120.add_argument('--add_positional_features', action='store_true',
                               help='Whether to extract positional features in addition to the visual ones.')
    parser_cad120.set_defaults(func=extract_cad120_visual_features)
    # Bimanual Actions dataset.
    parser_bimanual = subparsers.add_parser('bimanual',
                                            help='Extract visual features for the Bimanual Actions dataset.')

    parser_bimanual.add_argument('--rgbd_dir', type=str, required=True,
                                 help='Path to root directory containing all RGB-D images. It should contain a '
                                      'sub-directory for each subject.')
    parser_bimanual.add_argument('--tracked_objects_dir', type=str, required=True,
                                 help='Path to root directory containing all tracked 2D objects. This is the '
                                      '\'bimacs_derived_data_2d_objects_tracked\' dir.')
    parser_bimanual.add_argument('--tracked_hands_dir', type=str, required=True,
                                 help='Path to root directory containing all tracked hands. This is the '
                                      '\'bimacs_derived_data_hand_pose_tracked\' dir.')
    parser_bimanual.add_argument('--config_filepath', type=str, required=True,
                                 help='Path to \'faster_rcnn_R_101_C4_caffe.yaml\' file.')
    parser_bimanual.add_argument('--save_root', type=str,
                                 help='Path to directory to save the extracted visual features. A faster_rcnn.zarr '
                                      'folder is created inside save_root.')
    parser_bimanual.set_defaults(func=extract_bimanual_visual_features)
    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
