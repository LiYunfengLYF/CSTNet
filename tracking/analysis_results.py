import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import matplotlib.pyplot as plt
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
import argparse

plt.rcParams['figure.figsize'] = [8, 8]

tracker_name = r'cstnet'

tracker_param = r'baseline'

dataset_name = r'lasher'

env_num = 102


trackers = []
trackers.extend(trackerlist(name=tracker_name, parameter_name=tracker_param, dataset_name=dataset_name,
                            run_ids=None, display_name=tracker_param, env_num=env_num))

dataset = get_dataset(dataset_name, env_num=env_num)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),
              env_num=env_num)
