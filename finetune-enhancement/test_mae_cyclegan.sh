export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_enhancer.py \
--data_root /home/data/nature-breast-ultrasound/finetune-synthesis/USenhance-Breast \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_G_L2H