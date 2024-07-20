import argparse
import os
import sys
import time
#
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

'''

python tracking/test.py --script cstnet --config small --dataset vtuavst --threads 0 --num_gpus 1 --debug 1 --env_num 102


'''
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
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--script', type=str, default=None, help='test script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset', type=str, default='rgbt234', help='Sequence number or name.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    # parser.add_argument('--debug', type=int, default=1, help='Debug level.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    # parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--env_num', type=int, default=0, help='Use for multi environment developing, support: 0,1,2')
    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.script, args.config, args.runid, args.dataset, seq_name, args.debug, args.threads,
                num_gpus=args.num_gpus, env_num=args.env_num)
    print(F'Tracker {args.script} {args.config} on {args.dataset} dataset is OK! \n\n')


if __name__ == '__main__':
    main()
