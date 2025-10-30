export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/External-BCMID-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/External-US3M-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/External-HZFPH-Lesion-CLS2 \
--pretrained_dataset_name nature-breast-ultrasound --pretrained_model_name mae_vit_large_patch16 \
--batch_size 64 --max_device_batch_size 32 --determinism