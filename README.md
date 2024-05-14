# CSTNet for RGB-T Tracking

Our manuscript is available at [arxiv](https://arxiv.org/abs/2405.03177)

## Environment Installation

prepare your environment as [TBSI](https://github.com/RyanHTR/TBSI).

Notice: Our use pytorch version is 1.13.0

## Project Paths Setup
You can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in `./data`. It should look like:
```
${PROJECT_ROOT}
  -- data
      -- lasher
          |-- trainingset
          |-- testingset
          |-- trainingsetList.txt
          |-- testingsetList.txt
          ...
```

## Training
Download [RGBT](https://pan.baidu.com/share/init?surl=8MYRT4jkunIPklD02daFXA&pwd=y2rz) (TBSI with SOT pretrained model) pretrained weights and put them under `$PROJECT_ROOT$/pretrained_models`.

```
python tracking/train.py --script cstnet --config baseline --save_dir ./output --mode multiple --env_num 5 --nproc_per_node 2 --use_wandb 0
```

env_num doesn't need to be considered, it can be set to any number. if you want to train in different devices, you can consider it.

if you want to use env_num, go to lib/train/admin/local.py and lib/test/evaluation/local.py to set different device's num


Our tensorboard is released tensorboard/

Our training log is released at cstnet-baseline.log. 
Although iou name of training log is 'giou', we use wiou loss function. 
See lib/train/train_script.py and  lib/train/actor/cstnet_actor.py 


## Evaluation

```
python tracking/test.py cstnet baseline --dataset_name lasher_test --threads 4 --num_gpus 1

python tracking/analysis_results.py --tracker_name cstnet --tracker_param baseline --dataset_name lasher
```


## Acknowledgments
Our project is developed upon [TBSI](https://github.com/RyanHTR/TBSI). Thanks for their contributions which help us to quickly implement our ideas.

