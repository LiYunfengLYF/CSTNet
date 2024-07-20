import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0,
                threads=0, num_gpus=8, env_num=0):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name, env_num=env_num)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, env_num=env_num)]
    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus, env_num=env_num)


def main():
    debug = 0
    threads = 6
    num_gpus = 2
    env_num = 5

    tracker_list = [

        {'name': 'cstnet',
         'param': 'baseline'
         },

    ]

    dataset_list = [
        {'name': 'lasher'}
    ]

    for dataset_item in dataset_list:
        for tracker_item in tracker_list:
            trk_name, trk_param, data_name = tracker_item['name'], tracker_item['param'], dataset_item['name']
            run_tracker(trk_name, trk_param, None, data_name, None, debug, threads,
                        num_gpus=num_gpus, env_num=env_num)
            print(F'Tracker {trk_name} {trk_param} on {data_name} dataset is OK! \n\n')


if __name__ == '__main__':
    main()
