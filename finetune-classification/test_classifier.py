import torch
import argparse
import os
import sys
import numpy as np
import torch.distributed as dist
import pandas as pd

from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append("..")
from sklearn.metrics import roc_auc_score, confusion_matrix, \
    precision_score, recall_score, f1_score, matthews_corrcoef
from plot import plot_confusion_matrix, plot_roc_auc_curve


def main(args):
    # get local_rank from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    # print(f"Process {local_rank} is using GPU {torch.cuda.current_device()}")

    # initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')

    # data preprocessing
    transform_valid = v2.Compose([
        v2.ToImage(),
        v2.Resize((args.input_size, args.input_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    valid_dataset = ImageFolder(root=args.data_root + '/test', transform=transform_valid)

    # distribute data using DistributedSampler
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.test_batchSize, sampler=valid_sampler, num_workers=8)

    # loading a finetuned model
    finetuned_model_path = os.path.join('checkpoints', os.path.basename(args.data_root), args.finetuned_model_name+'.pt')
    model = torch.load(finetuned_model_path, map_location='cpu', weights_only=False)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    # get ground truth labels
    labels = [] 
    for idx in range(len(valid_dataset)):
        labels.append(valid_dataset[idx][1])

    classes = valid_dataset.classes
    num_class = len(classes)
    
    # initialize the confusion matrix tensor
    pred_cm = torch.tensor([], dtype=float, device='cuda')
    pred_cm = pred_cm.to(local_rank)

    test_correct = 0
    prob_list = []
    sample_metrics = []  # List to store per-sample metrics

    # Initialize the list to store image paths
    image_paths = [os.path.basename(sample[0]) for sample in valid_dataset.samples]

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(valid_dataloader)):
            image, label = image.cuda(local_rank), label.cuda(local_rank)
            output = model(image)

            predict = torch.max(output, dim=1)[1]
            probability = torch.softmax(output, dim=1)
            prob_array = probability.data.cpu().float().numpy()

            # Collect per-sample metrics
            for i in range(len(label)):
                sample_idx = batch_idx * args.test_batchSize + i
                image_name = image_paths[sample_idx]  # Get the corresponding filename
                sample_probs = prob_array[i].tolist()
                sample_metrics.append({
                    'image_name': image_name,
                    'True_Label': label[i].item(),
                    'Predicted_Label': predict[i].item(),
                    **{f'Prob_{classes[j]}': round(prob_array[i][j], 4) for j in range(num_class)}
                })

            prob_list.extend(prob_array)
            test_correct += (predict == label).sum().item()
            pred_cm = torch.cat((pred_cm, predict), dim=0)

    ave_test_acc = test_correct/len(valid_dataloader.dataset)
    # print('Testing Accuracy: {:.2f}% ({}/{})'.format(ave_test_acc * 100, test_correct, len(valid_dataloader.dataset)))
    
    # print('\nThe Confusion Matrix is plotted and saved:')
    cMatrix = confusion_matrix(torch.tensor(valid_dataset.targets), pred_cm.cpu())
    # print(cMatrix)
    
    finetuned_dataset_name = args.data_root.split('/')[-1]
    evaluate_path = 'evaluations/{}/{}'.format(finetuned_dataset_name, args.finetuned_model_name)
    if not os.path.exists(evaluate_path): 
        os.makedirs(evaluate_path)
    plot_confusion_matrix(classes, cMatrix, evaluate_path)

    y_true = torch.tensor(valid_dataset.targets)
    y_pred = pred_cm.cpu() 

    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100

    # print('\nAdditional Evaluation Metrics:')
    # print('Precision (macro): {:.2f}%'.format(precision))
    # print('Recall (macro): {:.2f}%'.format(recall))
    # print('F1-Score (macro): {:.2f}%'.format(f1))
    # print('Matthews Correlation Coefficient: {:.2f}%'.format(mcc))

    # Save metrics to Excel
    metrics = {
        'Metric': ['Testing Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)', 'Matthews Correlation Coefficient'],
        'Value': [
            f'{round(ave_test_acc * 100, 2)} ({test_correct}/{len(valid_dataloader.dataset)})',
            round(precision, 2),
            round(recall, 2),
            round(f1, 2),
            round(mcc, 2)
        ]
    }

    print(finetuned_dataset_name)
    probs = np.array(prob_list)  # shape: (num_test_images, num_classes)
    if num_class == 2:
        p_score = []
        for idx in range(len(labels)):
            p_score.append(probs[idx, 1])
        auc_score = roc_auc_score(labels, p_score) * 100
        print('The AUC score of binary classification is: {:.2f}%\n'.format(auc_score))
        plot_roc_auc_curve(labels, p_score, args.finetuned_model_name, evaluate_path)
        metrics['Metric'].append('AUC (binary)')
        metrics['Value'].append(round(auc_score, 2))
    else:
        auc_ovr = roc_auc_score(labels, probs, multi_class='ovr', average='macro') * 100
        auc_ovo = roc_auc_score(labels, probs, multi_class='ovo', average='macro') * 100
        print('AUC (OvR): {:.2f}%'.format(auc_ovr))
        print('AUC (OvO): {:.2f}%'.format(auc_ovo))
        metrics['Metric'].extend(['AUC (OvR)', 'AUC (OvO)'])
        metrics['Value'].extend([round(auc_ovr, 2), round(auc_ovo, 2)])

    # Create DataFrame and save to Excel
    result_file = os.path.join(evaluate_path, 'evaluation_metrics.xlsx')
    df = pd.DataFrame(metrics)
    df.to_excel(result_file, index=False)

    # Save aggregate metrics to text file
    txt_result_file = os.path.join(evaluate_path, 'evaluation_metrics.txt')
    with open(txt_result_file, 'w') as f:
        for metric, value in zip(metrics['Metric'], metrics['Value']):
            f.write(f'{metric}: {value}\n')

    # Save per-sample metrics
    sample_df = pd.DataFrame(sample_metrics)
    sample_result_file = os.path.join(evaluate_path, 'individual_metrics.xlsx')
    sample_df.to_excel(sample_result_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--test_batchSize', default=1, type=int)
    parser.add_argument('--finetuned_model_name', type=str, default='')

    args = parser.parse_args()
    
    main(args)