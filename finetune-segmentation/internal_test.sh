export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUSG \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUET-BUSD \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUID \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUS-BRA \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUS-UC \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUS-UCLM \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/BUSI \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/TUS \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/UDIAT \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor