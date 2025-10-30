export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 mae_pretrain.py \
--dataset_root /home/data/nature-breast-ultrasound \
--model_name mae_vit_large_patch16 --batch_size 512 --max_device_batch_size 128 \
--determinism --lam_rd 0.0015 --save_freq 500 --mask_ratio 0.65 --determinism