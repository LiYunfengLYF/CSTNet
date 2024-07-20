import os
import numpy as np

tracker_name = 'CSTNet-small'
results_path = r'E:\code\CSTNet\output\test\tracking_results\cstnet\small\vtuavst'
save_path = r'E:\code\vtuav\BBresults'

for seq_name in os.listdir(results_path):
    bboxs = np.loadtxt(os.path.join(results_path, seq_name), delimiter='\t')

    np.savetxt(os.path.join(save_path, tracker_name+'_'+seq_name), bboxs, delimiter=' ')

# print(os.listdir(results_path))
