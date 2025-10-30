export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/External-BUSI-WHU \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/External-BUSIS \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/External-ExpUNet \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_segmentor.py \
--data_root /home/data/nature-breast-ultrasound/finetune-segmentation/External-STU \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_segmentor