import os
import torch
import argparse
import sys
import numpy as np
import pandas as pd
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

sys.path.append("..")
from dataset import DenoisingDataset
from metric import compute_metrics

def main(args):
    # get local_rank from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} is using GPU {torch.cuda.current_device()}")

    # initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')

    # data preprocessing
    valid_dataset = DenoisingDataset(os.path.join(args.data_root, 'test'), args.input_size)

    # distribute data using DistributedSampler
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.test_batchSize, shuffle=False, num_workers=8, pin_memory=True)

    # load the finetuned model
    finetuned_model_path = os.path.join('checkpoints', args.data_root.split('/')[-1], f"{args.finetuned_model_name}.pt")
    if not os.path.exists(finetuned_model_path):
        raise FileNotFoundError(f"Model file not found at {finetuned_model_path}")
    GeneratorN2C = torch.load(finetuned_model_path, map_location='cpu', weights_only=False)
    GeneratorN2C = GeneratorN2C.cuda(local_rank)
    GeneratorN2C = DDP(GeneratorN2C, device_ids=[local_rank])
    GeneratorN2C.eval()

    # set up output directories and files
    finetuned_dataset_name = args.data_root.split('/')[-1]
    evaluate_path = os.path.join('evaluations', finetuned_dataset_name, args.finetuned_model_name)
    predicted_path = os.path.join(evaluate_path, 'predicted_high_quality')
    os.makedirs(predicted_path, exist_ok=True)

    text1_path = os.path.join(evaluate_path, 'average_metrics.txt')
    text2_path = os.path.join(evaluate_path, 'individual_metrics.txt')
    excel1_path = os.path.join(evaluate_path, 'average_metrics.xlsx')
    excel2_path = os.path.join(evaluate_path, 'individual_metrics.xlsx')

    # Initialize lists to store metrics
    all_ssim, all_psnr, all_names = [], [], []

    # Inference and evaluation loop
    with torch.no_grad():
        loop = tqdm(valid_loader, leave=True, desc="Evaluating")
        for idx, (noised_image, clean_image, clean_name) in enumerate(loop):
            noised_image = noised_image.cuda(local_rank)
            clean_image = clean_image.cuda(local_rank)
            predicted_image = GeneratorN2C(noised_image)  # Shape: [batch_size, C, H, W]

            # Process each image in the batch
            for i in range(predicted_image.size(0)):
                pred_single = predicted_image[i]  # Shape: [C, H, W]
                clean_single = clean_image[i]       # Shape: [C, H, W]

                # Compute metrics
                ssim_val, psnr_val = compute_metrics(pred_single, clean_single)
                all_ssim.append(ssim_val)
                all_psnr.append(psnr_val)
                all_names.append(clean_name[i])

                # Save per-image metrics to text2.txt
                with open(text2_path, 'a') as f2:
                    f2.write(f"Image: {clean_name[i]}, SSIM: {ssim_val * 100:.2f}, PSNR: {psnr_val:.6f}\n")

                # Save predicted image
                pred_img = (pred_single + 1) / 2.0  # Rescale to [0, 1]
                pred_img = pred_img.clamp(0, 1)     # Ensure valid range
                pred_img = pred_img * 255           # Rescale to [0, 255]
                if pred_img.shape[0] == 1:          # Grayscale
                    pred_img = pred_img.squeeze(0)  # Shape: [H, W]
                elif pred_img.shape[0] == 3:        # RGB
                    pred_img = pred_img.permute(1, 2, 0)  # Shape: [H, W, C]
                pred_img = pred_img.cpu().numpy().astype('uint8')
                pred_pil = Image.fromarray(pred_img)
                pred_pil.save(os.path.join(predicted_path, f"{clean_name[i]}"))

            loop.set_postfix(batch_idx=idx)

    # Compute mean and Standard Deviation of metrics
    mean_ssim = np.mean(all_ssim)
    mean_psnr = np.mean(all_psnr)
    std_ssim = np.std(all_ssim)
    std_psnr = np.std(all_psnr)

    # Save mean and Standard Deviation metrics to text1.txt
    with open(text1_path, 'w') as f1:
        f1.write(f"Mean SSIM: {mean_ssim * 100:.2f}\n")
        f1.write(f"Mean PSNR: {mean_psnr:.2f}\n")
        f1.write(f"Standard Deviation SSIM: {std_ssim * 100:.2f}\n")
        f1.write(f"Standard Deviation PSNR: {std_psnr:.2f}\n")

    # Save mean and Standard Deviation metrics to excel1.xlsx
    mean_df = pd.DataFrame({
        'Metric': ['Mean SSIM', 'Mean PSNR', 'Standard Deviation SSIM', 'Standard Deviation PSNR'],
        'Value': [f"{mean_ssim * 100:.2f}%", f"{mean_psnr:.2f}", f"{std_ssim * 100:.6f}", f"{std_psnr:.6f}"]
    })
    mean_df.to_excel(excel1_path, index=False)

    # Save individual metrics to excel2.xlsx
    individual_df = pd.DataFrame({
        'Image': all_names,
        'SSIM': [f"{x * 100:.2f}%" for x in all_ssim],
        'PSNR': [f"{x:.2f}" for x in all_psnr]
    })
    individual_df.to_excel(excel2_path, index=False)

    print(f"Evaluation completed. Mean and Standard Deviation  metrics saved to {text1_path} and {excel1_path}")
    print(f"Per-image metrics saved to {text2_path} and {excel2_path}")
    print(f"Predicted images saved to {predicted_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN predictions with metrics")
    parser.add_argument('--data_root', type=str, default='', help='Root directory for dataset')
    parser.add_argument('--input_size', type=int, default=224, help='Size of input images')
    parser.add_argument('--test_batchSize', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--finetuned_model_name', type=str, default='', help='Name of the finetuned model file (without .pt)')

    args = parser.parse_args()

    # Validate arguments
    if not args.data_root:
        raise ValueError("Please specify --data_root")
    if not args.finetuned_model_name:
        raise ValueError("Please specify --finetuned_model_name")

    main(args)