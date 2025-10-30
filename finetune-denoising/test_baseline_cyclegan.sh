export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.2 \
--finetuned_model_name cyclegan_G_N2C

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.4 \
--finetuned_model_name cyclegan_G_N2C

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.6 \
--finetuned_model_name cyclegan_G_N2C

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_denoise.py \
--data_root /home/data/nature-breast-ultrasound/finetune-denoise/BUSIS-0.8 \
--finetuned_model_name cyclegan_G_N2C
