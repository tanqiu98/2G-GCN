from typing import Union, List

import numpy as np


def bounding_boxes_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Given a set of keypoints, extract their bounding boxes.

    Arguments:
        keypoints - Numpy array of shape (..., num_keypoints, 2) containing the x and y coordinates of the keypoints.
    Returns:
        Numpy array of shape (..., 4) containing the x_min, y_min, x_max, y_max values of the input keypoints.
    """
    min_vals, max_vals = np.nanmin(keypoints, axis=-2), np.nanmax(keypoints, axis=-2)
    bounding_boxes = np.concatenate([min_vals, max_vals], axis=-1)
    return bounding_boxes


def alter_bounding_boxes_size(bounding_boxes: np.ndarray, alter_percentage: Union[int, float]) -> np.ndarray:
    """Increase or decrease the sides of bounding boxes.

    If a bounding box has size 100 x 100 and alter_percentage is 120, the altered bounding box has size 140 x 140.

    Arguments:
        bounding_boxes - Numpy array of shape (..., 4) containing the x_min, y_min, x_max, y_max values of the
            bounding boxes.
        alter_percentage - Percentage to alter each side of the bounding boxes. For instance, if alter_percentage is
            120, increase each side of the bounding box by 20%; if alter_percentage is 70, decrease each side of the
            bounding box by 30%.
    Returns:
        A numpy array of shape (..., 4) containing the altered bounding boxes.
    """
    min_vals, max_vals = bounding_boxes[..., :2], bounding_boxes[..., 2:]
    alter_proportion = alter_percentage / 100
    scale_diff = abs(1.0 - alter_proportion) * (max_vals - min_vals)
    if alter_proportion < 1.0:
        min_vals += scale_diff
        max_vals -= scale_diff
    else:
        min_vals -= scale_diff
        max_vals += scale_diff
    bounding_boxes = np.concatenate([min_vals, max_vals], axis=-1)
    return bounding_boxes


def iou_between_bounding_boxes(many_bounding_boxes: List[np.ndarray]) -> np.ndarray:
    """Compute the area of the intersection over union of potentially many bounding boxes.

    Arguments:
        many_bounding_boxes - A list containing bounding boxes. Each element in the list is a numpy array of shape
            (..., 4) containing the x_min, y_min, x_max, y_max coordinates of each bounding box.
    Returns:
        A numpy array of shape (..., 1) containing the IoU of the bounding boxes.
    """
    bounding_boxes_union = many_bounding_boxes[0]
    bounding_boxes_intersection = many_bounding_boxes[0]
    for bounding_boxes in many_bounding_boxes[1:]:
        bounding_boxes_union = unionize_bounding_boxes(bounding_boxes_union, bounding_boxes)
        bounding_boxes_intersection = intersect_bounding_boxes(bounding_boxes_intersection, bounding_boxes)
    bounding_boxes_intersection_area = compute_bounding_boxes_area(bounding_boxes_intersection)
    bounding_boxes_intersection_area[np.isnan(bounding_boxes_intersection_area)] = 0.0
    bounding_boxes_union_area = compute_bounding_boxes_area(bounding_boxes_union)
    bounding_boxes_iou = bounding_boxes_intersection_area / bounding_boxes_union_area
    return bounding_boxes_iou


def intersect_bounding_boxes(bounding_boxes_a: np.ndarray, bounding_boxes_b: np.ndarray) -> np.ndarray:
    """Compute the intersection of bounding boxes.

    Returns NaN for bounding boxes that do not intersect.

    Arguments:
        bounding_boxes_a - Numpy array of shape (..., 4) containing the x_min, y_min, x_max, y_max of the
            bounding boxes.
        bounding_boxes_b - Same as bounding_boxes_a.
    Returns:
        A numpy array of shape (..., 4) containing the x_min, y_min, x_max, y_max of the intersecting bounding boxes.
    """
    min_vals = np.maximum(bounding_boxes_a[..., :2], bounding_boxes_b[..., :2])
    max_vals = np.minimum(bounding_boxes_a[..., 2:], bounding_boxes_b[..., 2:])
    is_consistent = (min_vals[..., :1] <= max_vals[..., :1]) & (min_vals[..., 1:2] <= max_vals[..., 1:2])
    is_consistent = np.repeat(is_consistent, repeats=2, axis=-1)
    min_vals = np.where(is_consistent, min_vals, np.nan)
    max_vals = np.where(is_consistent, max_vals, np.nan)
    intersected_bounding_boxes = np.concatenate([min_vals, max_vals], axis=-1)
    return intersected_bounding_boxes


def unionize_bounding_boxes(bounding_boxes_a: np.ndarray, bounding_boxes_b: np.ndarray) -> np.ndarray:
    """Compute the union of bounding boxes.

    Arguments:
        bounding_boxes_a - Numpy array of shape (..., 4) containing the x_min, y_min, x_max, y_max of the
            bounding boxes.
        bounding_boxes_b - Same as bounding_boxes_a.
    Returns:
        A numpy array of shape (..., 4) containing the x_min, y_min, x_max, y_max of the union of the bounding boxes.
    """
    min_vals = np.minimum(bounding_boxes_a[..., :2], bounding_boxes_b[..., :2])
    max_vals = np.maximum(bounding_boxes_a[..., 2:], bounding_boxes_b[..., 2:])
    unionized_bounding_boxes = np.concatenate([min_vals, max_vals], axis=-1)
    return unionized_bounding_boxes


def compute_bounding_boxes_area(bounding_boxes: np.ndarray) -> np.ndarray:
    """Compute the area of bounding boxes.

    Arguments:
        bounding_boxes - Numpy array of shape (..., 4) containing the x_min, y_min, x_max, y_max coordinates of the
            bounding boxes.
    Returns:
        A numpy array of shape (..., 1) containing the area of each input bounding box.
    """
    bounding_boxes_width = bounding_boxes[..., 2:3] - bounding_boxes[..., 0:1]
    bounding_boxes_height = bounding_boxes[..., 3:4] - bounding_boxes[..., 1:2]
    return bounding_boxes_width * bounding_boxes_height
