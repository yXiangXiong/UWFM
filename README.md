# Comprehensive breast ultrasound intelligence with an uncertainty weighting foundation model

We developed a foundation model for breast ultrasound analysis. We validate the effectiveness of our framework using extensive and diverse breast ultrasound datasets, including 964,363 images involved in upstream learning and 44,649 images involved in 9 downstream tasks. Across these validations, the framework consistently outperformed state-of-the-art algorithms in learning effective representations for breast ultrasound images and demonstrated strong performance on comprehensive downstream tasks without any pre-training. The high-quality and harmonized representations learned by our framework have the potential to drive transformative improvements across all stages of breast cancer care, ranging from screening and diagnosis to neoadjuvant therapy response monitoring and molecular subtyping.


## Model system
<img src="https://github.com/yXiangXiong/UWFM/framework.png"/>

## [Datasets]
create a directory below and add your own datasets.
```
/home/data/nature-breast-ultrasound/pretrain
|─train
│      ALN-Ultra
│          001_.png
│          002_.png
│          ...
│          003_.png
│      Breast-Us-Video
│          001_.png
│          002_.png
│          ...
│          003_.png
│      BUI
│          ...
│      HiSBreast
│          ...
│      PKUTH
│          ...
│      SYSUCC
│          ...
│      TDSC-ABUS
│          ...
│      WHBUS
│          ...
│      ZJCH
│          ...
├─valid
│      BUSI
│        001_.png
│        002_.png
│        ...
│        003_.png
│      UDIAT
│        001_.png
│        002_.png
│        ...
│        003_.png
```

### Pretrain
```bash 
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 mae_pretrain.py \
--dataset_root /home/data/nature-breast-ultrasound \
--model_name mae_vit_large_patch16 --batch_size 512 --max_device_batch_size 128 \
--determinism --lam_rd 0.0015 --save_freq 500 --mask_ratio 0.65 --determinism
```

### downstream classification
```bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUSG-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUSG-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier
```
### downstream segmentaton
```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUSG \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUSG \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor
```
### downstream synthesis
```bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_enhancer.py \
--data_root /home/data/nature-breast-ultrasound/finetune-synthesis/USenhance-Breast \
--pretrained_dataset_name nature-breast-ultrasound \
--pretrained_model_name mae_vit_large_patch16 \
--batch_size 4 --num_epochs 100

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_enhancer.py \
--data_root /home/data/nature-breast-ultrasound/finetune-synthesis/USenhance-Breast \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_G_L2H
```
### downstream denoising
```bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.2 \
--pretrained_dataset_name nature-breast-ultrasound \
--pretrained_model_name mae_vit_large_patch16 \
--batch_size 4 --num_epochs 100

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_G_N2C
```
