export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUSG-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUET-BUSD-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUID-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUS-BRA-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUS-UC-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUS-UCLM-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUS-UCLM-Lesion-CLS3 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUSI-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/BUSI-Lesion-CLS3 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/HZFPH-Lesion-CLS5 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/HZFPH-LN-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/TUS-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/UDIAT-Lesion-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/YHD-Molecular-CLS4 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 test_classifier.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/YHD-pCR-CLS2 \
--finetuned_model_name nature-breast-ultrasound_vit_large_patch16_classifier