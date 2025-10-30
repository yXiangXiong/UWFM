export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/xiangyu/data/nature-breast-ultrasound/finetune-classification/External-BCMID-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/xiangyu/data/nature-breast-ultrasound/finetune-classification/External-US3M-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/xiangyu/data/nature-breast-ultrasound/finetune-classification/External-HZFPH-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier