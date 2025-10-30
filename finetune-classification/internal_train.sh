export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUSG-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUET-BUSD-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUID-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUS-BRA-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUS-UC-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUS-UCLM-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUS-UCLM-Lesion-CLS3 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUSI-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUSI-Lesion-CLS3 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/HZFPH-Lesion-CLS5 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/HZFPH-LN-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/TUS-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/UDIAT-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/YHD-Molecular-CLS4 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/YHD-pCR-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism
