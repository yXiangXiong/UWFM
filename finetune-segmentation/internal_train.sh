export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUSG \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUET-BUSD \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUID \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUS-BRA \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUS-UC \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUS-UCLM \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUSI \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/TUS \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/UDIAT \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32