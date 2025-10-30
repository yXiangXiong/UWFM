export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/nature-breast-ultrasound/finetune-segmentation/External-BUSI-WHU \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/nature-breast-ultrasound/finetune-segmentation/External-BUSIS \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/nature-breast-ultrasound/finetune-segmentation/External-ExpUNet \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_segmentor.py \
--data_root /home/nature-breast-ultrasound/finetune-segmentation/External-STU \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32