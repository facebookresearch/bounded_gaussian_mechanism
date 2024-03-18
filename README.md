# Privacy Amplification for the Gaussian Mechanism via Bounded Support

  

This repository contains code and experiments for the manuscript:

- Shengyuan Hu, Saeed Mahloujifar, Virginia Smith, Kamalika Chaudhuri, Chuan Guo. **[Privacy Amplification for the Gaussian Mechanism via Bounded Support]()**.

## General Setup
Run the following to fetch submodules and install updates
```
git submodule update --init
python install.py
```
  

## DP Experiments

  

The code for DP experiments can be found in ```/dp``` . The code is based off of the implementation of [TAN](https://github.com/facebookresearch/tan/) ([Sander et al.](https://arxiv.org/abs/2210.03403)) and [DP-RAFT](https://github.com/kiddyboots216/dp-custom) ([Panda et al.](https://arxiv.org/abs/2212.04486)). 

### Setup
Run the following to install packages 
```
conda create -n "dp_bounded" python=3.9 
conda activate dp_bounded
cd dp
pip install -r ./requirements.txt
```

As an example of features extracted from pretrained model, download all files from [DP-RAFT repo](https://github.com/kiddyboots216/dp-custom/tree/main/dp_finetuning/extracted_features/transfer/features/cifar10_beitv2_large_patch16_224_in22k) and store it under ```dp/utils/extracted_features/transfer/features/cifar10_beitv2_large_patch16_224_in22k```.

### Experiments

Here are a few examples to run experiments with a fixed set of hyperparameters.

Private linear probing for CIFAR-10 on pretrained BeitV2 feature extractor using vanilla Gaussian Mechanism with $L_{\infty}$ clipping:
```
python  bounded_gaussian_dp_raft.py  --dataset "cifar10" --lr  16  --arch  "beitv2_large_patch16_224_in22k"  --momentum  0.9  --max_physical_batch_size  2048  --max_per_sample_grad_norm 0.002 --batch_size  50000  --ref_nb_steps  20  --ref_B  50000  --ref_noise 2000 --data_root  "data/cifar10"  --master_port  $MASTER_PORT --pretrain  True
```
Private linear probing for CIFAR-10 on pretrained BeitV2 feature extractor using rectified Gaussian Mechanism with bounded support $\mathcal{B}=[-32,32]^d$:
```
python  bounded_gaussian_dp_raft.py  --dataset "cifar10" --rectification True --bound 32 --lr  16  --arch  "beitv2_large_patch16_224_in22k"  --momentum  0.9  --max_physical_batch_size  2048  --max_per_sample_grad_norm 0.002 --batch_size  50000  --ref_nb_steps  20  --ref_B  50000  --ref_noise 2000 --data_root  "data/cifar10"  --master_port  $MASTER_PORT --pretrain  True
```
Private linear probing for CIFAR-10 on pretrained BeitV2 feature extractor using truncated Gaussian Mechanism with bounded support $\mathcal{B}=[-32,32]^d$:
```
python  bounded_gaussian_dp_raft.py  --dataset "cifar10" --truncation True --bound 32 --lr  16  --arch  "beitv2_large_patch16_224_in22k"  --momentum  0.9  --max_physical_batch_size  2048  --max_per_sample_grad_norm 0.002 --batch_size  50000  --ref_nb_steps  20  --ref_B  50000  --ref_noise 2000 --data_root  "data/cifar10"  --master_port  $MASTER_PORT --pretrain  True
```
To perform experiments on CIFAR-100/OxfordIIITPet, set ```--dataset``` to be ```CIFAR100```/```OxfordIIITPet```. To perform experiments on other pretrained feature extractor, change ```--arch``` to be other models (e.g. ```vit_large_patch16_384``` used in this paper). Values of hyperparameters used in all experiments could be found in Appendix G of the paper.

## FIL Experiments
The code for FIL experiments can be found in ```/fil``` . The code is based off of the implementation of [Bounding Data Reconstruction](https://github.com/facebookresearch/bounding_data_reconstruction) ([Guo et al.](https://arxiv.org/abs/2201.12383)). 

### Setup
Run the following to install packages 
```
conda create -n "fil_bounded" python=3.9 
conda activate fil_bounded
cd fil
pip install -r ./requirements.txt
```

After doing that, install [jax](https://github.com/google/jax).

### Experiments

Here are a few examples to run experiments with a fixed set of hyperparameters.

Private linear probing for CIFAR-10 on pretrained BeitV2 feature extractor using vanilla Gaussian Mechanism with $L_{\infty}$ clipping:
```
python train_classifier_cifar_pretrain.py --model "linear" --data_path PATH_TO_FEATURES --sigma 2000  --batch_size  50000  --linf_clip  --num_epochs  20  --norm_clip 0.002 --step_size  16
```

Private linear probing for CIFAR-10 on pretrained BeitV2 feature extractor using rectified Gaussian Mechanism with bounded support $\mathcal{B}=[-32,32]^d$:
```
python train_classifier_cifar_pretrain.py --model "linear" --data_path PATH_TO_FEATURES --sigma 2000 --rectification  --batch_size  50000  --linf_clip  --num_epochs  20  --norm_clip 0.002 --step_size  16  --lb -32 --ub 32
```

Private linear probing for CIFAR-10 on pretrained BeitV2 feature extractor using truncated Gaussian Mechanism with bounded support $\mathcal{B}=[-32,32]^d$:
```
python train_classifier_cifar_pretrain.py --model "linear" --data_path PATH_TO_FEATURES --sigma 2000 --truncation  --batch_size  50000  --linf_clip  --num_epochs  20  --norm_clip 0.002 --step_size  16  --lb -32 --ub 32
```

An example of data path generated from the DP-RAFT experiments: ```PATH_TO_FEATURES="../dp/utils/extracted_features/transfer/features/cifar10_beitv2_large_patch16_224_in22k"```

Privately training a Wide ResNet using rectified Gaussian Mechanism with bounded support $\mathcal{B}=[-32,32]^d$:

```
python  train_classifier_cifar.py  --model "cnn_cifar" --sigma 2000 --batch_size 200  --rectification  --linf_clip  --num_epochs  150  --norm_clip  0.002  --step_size 0.03 --lb -32 --ub 32
```

For truncated Gaussian experiment, turn on ```--truncation``` flag as is done in the linear probing experiment.


## Code Acknowledgements

The majority of Bounded Gaussian Mechanism is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [Opacus](https://github.com/pytorch/opacus) and [DP-RAFT](https://github.com/kiddyboots216/dp-custom) are licensed under the Apache 2.0 license; and [TAN](https://github.com/facebookresearch/tan) is licensed under the BSD 3-Clause License.

All changes files are listed in ```list_of_files_changed.txt```.