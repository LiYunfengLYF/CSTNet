import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class vtuavstDataset(BaseDataset):
    def __init__(self, env_num):
        super().__init__(env_num)
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)

        self.base_path = self.env_settings.vtuavst_path

        self.sequence_list = self._get_sequence_list(dataset_path=self.base_path)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/rgb.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=' ', dtype=np.float64)

        frames_path_i = '{}/{}/ir'.format(self.base_path, sequence_name)
        frames_path_v = '{}/{}/rgb'.format(self.base_path, sequence_name)
        frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".jpg")]

        frame_list_i.sort(key=lambda f: int(f[1:-4]))
        frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".jpg")]
        frame_list_v.sort(key=lambda f: int(f[1:-4]))
        frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        frames_list = [frames_list_v, frames_list_i]
        return Sequence(sequence_name, frames_list, 'vtuavst', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, dataset_path):
        sequence_list = os.listdir(dataset_path)
        return sequence_list
