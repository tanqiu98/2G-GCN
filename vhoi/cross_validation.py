"""CAD-120 dataset functions to determine the cross-validation data split.

The functions in here craw through the annotations directory provided with the CAD-120 dataset to create a
mapping between each video id and its corresponding subject. In that way, we can correctly split the data into
four groups for cross-validation.
"""
from collections import defaultdict
import os


def generate_video_id_to_subject_mapping(path: str) -> dict:
    """Craw through the 'annotations' directory and generate video id to subject mapping."""
    subject_to_video_id = defaultdict(set)
    subject_dirs = os.listdir(path)
    for subject_dir in subject_dirs:
        subject_id = subject_dir.split(sep='_')[0]
        activity_dirs = os.listdir(os.path.join(path, subject_dir))
        for activity_dir in activity_dirs:
            filepath = os.path.join(path, subject_dir, activity_dir, 'labeling.txt')
            with open(filepath, mode='r') as f:
                for line in f:
                    video_id = line.strip().split(sep=',')[0]
                    subject_to_video_id[subject_id].add(video_id)
    video_id_to_subject = {}
    for subject_id, video_ids in subject_to_video_id.items():
        for video_id in video_ids:
            video_id_to_subject[video_id] = subject_id
    return video_id_to_subject
