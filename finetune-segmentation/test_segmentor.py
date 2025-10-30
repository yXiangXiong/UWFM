import torch
import argparse
import os
import sys
import numpy as np
import torch.distributed as dist
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image

sys.path.append("..")
from dataset import SegmentationDataset
from visualize import save_mask
from metric import calculate_dice, calculate_iou, \
    calculate_accuracy, calculate_sensitivity, calculate_hd95


def main(args):
    # get local_rank from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} is using GPU {torch.cuda.current_device()}")

    # initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')

    # data preprocessing
    valid_dataset = SegmentationDataset(args.data_root + '/test', args.input_size)
    num_classes = valid_dataset.num_class

    # distribute data using DistributedSampler
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.test_batchSize, sampler=valid_sampler, num_workers=8)

    # loading a finetuned model
    finetuned_model_path = os.path.join('checkpoints', args.data_root.split('/')[-1], args.finetuned_model_name+'.pt')
    model = torch.load(finetuned_model_path, map_location='cpu', weights_only=False)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    # obtaining finetuned_dataset string from data_root
    finetuned_dataset_name = args.data_root.split('/')[-1]

    # saving the calculated metrics and predicted masks
    evaluate_path = 'evaluations/{}/{}'.format(finetuned_dataset_name, args.finetuned_model_name)
    predicted_path = os.path.join(evaluate_path, 'predicted_masks')
    if not os.path.exists(predicted_path): 
        os.makedirs(predicted_path)

    # define the output file paths
    txt1_path = os.path.join(evaluate_path, 'individual_metrics.txt')
    txt2_path = os.path.join(evaluate_path, 'average_metrics.txt')
    xlsx1_path = os.path.join(evaluate_path, 'individual_metrics.xlsx')
    xlsx2_path = os.path.join(evaluate_path, 'average_metrics.xlsx')

    # image denormalization (restore the normalized image to its original range)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    dice_sum = 0.0
    iou_sum = 0.0
    acc_sum = 0.0
    sensitivity_sum = 0.0
    hd95_sum = 0.0
    valid_samples = 0
    individual_metrics = []

    # open the txt1 file to write single image metrics
    with open(txt1_path, 'w') as txt1_file:
        with torch.no_grad():
            for image, mask, mask_name in tqdm(valid_dataloader):
                image, mask = image.cuda(local_rank), mask.cuda(local_rank)
                output = model(image)
                predicted_mask = output.argmax(dim=1)

                for i in range(predicted_mask.size(0)):
                    pred_single = predicted_mask[i]
                    target_single = mask[i]

                    dice = calculate_dice(pred_single, target_single, num_classes)
                    iou = calculate_iou(pred_single, target_single, num_classes)
                    acc = calculate_accuracy(pred_single, target_single)
                    sensitivity = calculate_sensitivity(pred_single, target_single, num_classes)
                    hd95 = calculate_hd95(pred_single, target_single, num_classes)

                    dice_sum += dice
                    iou_sum += iou
                    acc_sum += acc
                    sensitivity_sum += sensitivity
                    if np.isfinite(hd95):
                        hd95_sum += hd95
                        valid_samples += 1

                    # Store individual metrics for Excel
                    individual_metrics.append({
                        'Image_Name': mask_name[i],
                        'Dice (%)': float("{:.2f}".format(dice.item() * 100)),
                        'IoU (%)': float("{:.2f}".format(iou.item() * 100)),
                        'Accuracy (%)': float("{:.2f}".format(acc.item() * 100)),
                        'Sensitivity (%)': float("{:.2f}".format(sensitivity.item() * 100)),
                        'HD95': float("{:.2f}".format(hd95))
                    })

                    # Convert to percentage (Dice, IoU, ACC, Sensitivity), HD95 retains the value
                    metric_str = f"{mask_name[i]} - Dice: {dice*100:.2f}%, IoU: {iou*100:.2f}%, ACC: {acc*100:.2f}%, Sensitivity: {sensitivity*100:.2f}%, HD95: {hd95:.2f}"
                    # print(metric_str)
                    txt1_file.write(metric_str + '\n')

                    file_base = os.path.splitext(mask_name[i])[0]
                    file_path = os.path.join(predicted_path, f"{file_base}_pred_mask.png")
                    save_mask(pred_single, file_path, num_classes)

                    orig_image_tensor = inv_normalize(image[i])
                    orig_image_array = orig_image_tensor.cpu().numpy().transpose(1, 2, 0)
                    orig_image_array = np.clip(orig_image_array * 255, 0, 255).astype(np.uint8)
                    orig_image = Image.fromarray(orig_image_array, mode='RGB')

                    orig_image.save(os.path.join(predicted_path, f"{file_base}_original.png"))
                    save_mask(target_single, os.path.join(predicted_path, f"{file_base}_gt_overlay.png"), num_classes, orig_image)
                    save_mask(pred_single, os.path.join(predicted_path, f"{file_base}_pred_overlay.png"), num_classes, orig_image)

                    combined_image = Image.new('RGB', (orig_image.width * 3, orig_image.height))
                    combined_image.paste(orig_image, (0, 0))
                    combined_image.paste(Image.open(os.path.join(predicted_path, f"{file_base}_gt_overlay.png")), (orig_image.width, 0))
                    combined_image.paste(Image.open(os.path.join(predicted_path, f"{file_base}_pred_overlay.png")), (orig_image.width * 2, 0))
                    combined_path = os.path.join(predicted_path, f"{file_base}_combined.png")
                    combined_image.save(combined_path)
            
    # Save individual metrics to Excel
    individual_df = pd.DataFrame(individual_metrics)
    individual_df.to_excel(xlsx1_path, index=False)

    # Calculate averages and standard deviations
    dice_values = [m['Dice (%)'] / 100 for m in individual_metrics]  # Convert back to [0,1] for std calculation
    iou_values = [m['IoU (%)'] / 100 for m in individual_metrics]
    acc_values = [m['Accuracy (%)'] / 100 for m in individual_metrics]
    sensitivity_values = [m['Sensitivity (%)'] / 100 for m in individual_metrics]
    hd95_values = [m['HD95'] for m in individual_metrics if np.isfinite(m['HD95'])]

    # Calculate the average
    average_dice = dice_sum / len(valid_dataset)
    average_iou = iou_sum / len(valid_dataset)
    average_acc = acc_sum / len(valid_dataset)
    average_sensitivity = sensitivity_sum / len(valid_dataset)
    average_hd95 = hd95_sum / valid_samples if valid_samples > 0 else np.inf

    # Calculate standard deviations
    std_dice = np.std(dice_values) * 100  # Convert to percentage
    std_iou = np.std(iou_values) * 100
    std_acc = np.std(acc_values) * 100
    std_sensitivity = np.std(sensitivity_values) * 100
    std_hd95 = np.std(hd95_values) if valid_samples > 0 else np.inf

    print('\n')
    # Prepare average metrics for text and Excel
    average_metrics = {
        'Metric': [
            'Number of Test Images',
            'Average Dice Score (%)',
            'Std Dice Score (%)',
            'Average IoU (%)',
            'Std IoU (%)',
            'Average Accuracy (%)',
            'Std Accuracy (%)',
            'Average Sensitivity (%)',
            'Std Sensitivity (%)',
            'Average HD95',
            'Std HD95'
        ],
        'Value': [
            len(valid_dataset),
            float("{:.2f}".format(average_dice.item() * 100)),
            float("{:.2f}".format(std_dice)),
            float("{:.2f}".format(average_iou.item() * 100)),
            float("{:.2f}".format(std_iou)),
            float("{:.2f}".format(average_acc.item() * 100)),
            float("{:.2f}".format(std_acc)),
            float("{:.2f}".format(average_sensitivity.item() * 100)),
            float("{:.2f}".format(std_sensitivity)),
            float("{:.2f}".format(average_hd95.item() if np.isfinite(average_hd95) else np.inf)),
            float("{:.2f}".format(std_hd95 if np.isfinite(std_hd95) else np.inf))
        ]
    }
    # Write to txt2 file and print
    with open(txt2_path, 'w') as txt2_file:
        for metric, value in zip(average_metrics['Metric'], average_metrics['Value']):
            line = f'{metric}: {value}'
            # print(line)
            txt2_file.write(line + '\n')

    # Save average metrics to Excel
    average_df = pd.DataFrame(average_metrics)
    average_df.to_excel(xlsx2_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--test_batchSize', default=4, type=int)
    parser.add_argument('--finetuned_model_name', type=str, default='')

    args = parser.parse_args()

    main(args)