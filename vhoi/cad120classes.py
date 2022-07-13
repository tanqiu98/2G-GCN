from collections import defaultdict


class CAD120Video:
    def __init__(self):
        # Initially represent it as a dictionary to simplify the collection of segments. Later, after all
        # segments have been collected, convert to a list of segments.
        self._video_segments = defaultdict(CAD120VideoSegment)

    def __getitem__(self, item):
        return self._video_segments[item]

    def __len__(self):
        return len(self._video_segments)

    def from_dict_to_list(self):
        self._video_segments = sorted(list(self._video_segments.items()))
        self._video_segments = [segment_features for segment_num, segment_features in self._video_segments]

    def update_next_labels(self):
        for video_segment, next_video_segment in zip(self._video_segments[:-1], self._video_segments[1:]):
            video_segment.next_subactivity = next_video_segment.subactivity
            video_segment.next_object_affordance = dict(next_video_segment.object_affordance)


class CAD120VideoSegment:
    def __init__(self):
        self.skeleton_features = None
        self.skeleton_object_features = {}  # Object ID -> features
        self.skeleton_temporal_features = None
        self.object_features = {}  # Object ID -> features
        self.object_object_features = {}  # (Object 1 ID, Object 2 ID) -> features
        self.object_temporal_features = {}  # Object ID -> features
        self.subactivity = None
        self.next_subactivity = None
        self.object_affordance = {}  # Object ID -> affordance
        self.next_object_affordance = {}  # Object ID -> affordance
        # Some Metadata
        self.subactivity_name = None
        self.object_affordance_name = {}  # Object ID -> affordance name
        self.object_type = {}  # Object ID -> type (e.g. mug, book)
        self.start_frame = None
        self.end_frame = None
