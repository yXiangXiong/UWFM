# evalutate HZFPH-Lesion-CLS5 performance across multiple classification conditions
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node=1 test_cls5_condition.py \
--data_root /home/data/nature-breast-ultrasound/finetune-classification/HZFPH-Lesion-CLS5 \
--finetuned_dataset_name HZFPH-Lesion-CLS5-BM \
--probability_excel_path /home/code/MAE-Breast-Ultrasound/finetune-classification/evaluations/HZFPH-Lesion-CLS5-BM/anonymous-groupAB_prob.xls \
--pretrained_model_name nature-breast-ultrasound_vit_large_patch16_classifier --num_class 5 --test_auc