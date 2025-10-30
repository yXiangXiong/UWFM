export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.2 \
--pretrained_dataset_name nature-breast-ultrasound \
--pretrained_model_name mae_vit_large_patch16 \
--batch_size 4 --num_epochs 100

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.4 \
--pretrained_dataset_name nature-breast-ultrasound \
--pretrained_model_name mae_vit_large_patch16 \
--batch_size 4 --num_epochs 100

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.6 \
--pretrained_dataset_name nature-breast-ultrasound \
--pretrained_model_name mae_vit_large_patch16 \
--batch_size 4 --num_epochs 100

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.8 \
--pretrained_dataset_name nature-breast-ultrasound \
--pretrained_model_name mae_vit_large_patch16 \
--batch_size 4 --num_epochs 100