import torch
import argparse
import os
import xlwt
import numpy as np
import sys
import torch.distributed as dist

from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from plot import plot_confusion_matrix, plot_roc_auc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append("..")
from utils import setup_seed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 自定义数据集类（返回路径）
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path


def test_model(model, valid_dataset, valid_loader, local_rank, args):
    labels = [] # ground_truth labels
    for idx in range(len(valid_dataset)):
        labels.append(valid_dataset[idx][1])
    # print('labels:', labels)

    classes = valid_dataset.classes
    print(valid_dataset.class_to_idx)

    pred_cm = torch.tensor([], dtype=float, device='cuda')
    pred_cm = pred_cm.to(local_rank)

    test_correct = 0
    prob_list = []

    # 保存每张图片的五分类的概率
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet('Sheet1')
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.bold = True
    style.font = font

    row = 0
    for data, label, path in tqdm(valid_loader): # image, label
        data, label = data.cuda(local_rank), label.cuda(local_rank)  # move to GPU
        output = model(data) # [batch_size, num_class]
        
        predict = torch.max(output, dim=1)[1]              # output predicted class (i.e., idx)
        probability = torch.softmax(output, dim=1)         # output probabilities

        prob_array = np.squeeze(probability.data.cpu().float().numpy())
        max_index = np.argmax(prob_array)

        # 概率保存
        img_name = path[0].split('/')[-1]
        worksheet.write(row, 0, label=img_name) # 第0列为图片名字
        col = 1
        for prob in prob_array:
            percentage = "{:.2%}".format(prob)
            if col == (max_index + 1):
                worksheet.write(row, col, label=percentage, style=style)
            else:
                worksheet.write(row, col, label=percentage)
            col += 1
        row += 1
        
        prob_list.append(prob_array)

        test_correct += (predict == label).sum().item()    # update validation correct numbers
        pred_cm = torch.cat((pred_cm, predict), dim=0)

    # workbook.save(args.probability_excel_path)

    ave_test_acc = test_correct/len(valid_loader.dataset)  # calculate average accuracy
    print('Testing Accuracy: {:.4f} ({}/{})'.format(ave_test_acc, test_correct, len(valid_loader.dataset)))
    

    if args.test_auc:
        probs = np.array(prob_list) # shape: (num_testImages, num_classes)

        if args.num_class == 2:
            p_malignant = []
            for idx in range(len(labels)):
                p_malignant.append(probs[idx, 1])
            auc_score = roc_auc_score(labels, p_malignant)
            print('The AUC score of binary classification is: {:.4f}\n'.format(auc_score))
            # plot_roc_auc_curve(labels, p_malignant, args.pretrained_model_name, matrix_path)

        if args.num_class == 5:
            auc_ovr = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            print('AUC (OvR): {:.4}'.format(auc_ovr))
            auc_ovo = roc_auc_score(labels, probs, multi_class='ovo', average='macro')
            print('AUC (OvO): {:.4}'.format(auc_ovo))

            # ----- evaluate the (benign vs malignant) mass under the five-class classification ----- #
            bm_prob = probs.copy()
            bm_prob[:, 0] = bm_prob[:, 0] / (bm_prob[:, 0] + bm_prob[:, 1]) # benign mass
            bm_prob[:, 1] = 1 - bm_prob[:, 0]                               # malignant mass

            label_mass = []
            p_mass = []
            image_names = []
            for idx in range(len(labels)):
                # select only benign mass and malignant mass
                if labels[idx] == 0 or labels[idx] == 1: 
                    label_mass.append(labels[idx])
                    p_mass.append(bm_prob[idx, :])
                    # Get image name from valid_dataset
                    img_path = valid_dataset.samples[idx][0]
                    img_name = img_path.split('/')[-1]
                    image_names.append(img_name)
            p_mass = np.array(p_mass)

            label_benign = [1 if label == 0 else 0 for label in label_mass]
            auc_score = roc_auc_score(label_benign, p_mass[:, 0])
            print('AUC score of benign mass: {:.4}'.format(auc_score))

            label_malignant = [1 if label == 1 else 0 for label in label_mass]
            auc_score = roc_auc_score(label_malignant, p_mass[:, 1])
            print('AUC score of malignant mass: {:.4}'.format(auc_score))

            # Save benign vs malignant (mass) probabilities to Excel (2025.09.25)
            malignant_prob_excel_path = os.path.join(
                'evaluations', args.finetuned_dataset_name, 
                f'{args.pretrained_model_name}_benign_vs_malignant(mass)_prob.xls'
            )
            workbook_mass = xlwt.Workbook(encoding='utf-8')
            worksheet_mass = workbook_mass.add_sheet('Benign_vs_Malignant(mass)')
            worksheet_mass.write(0, 0, label='Image_Name')  # Header for image names
            worksheet_mass.write(0, 1, label='Binary_Label')  # Header for binary labels (benign=0, malignant=1)
            worksheet_mass.write(0, 2, label='Malignant_Probability')  # Header for malignant probability

            # Write image names, binary labels, and malignant probabilities
            for idx, (img_name, binary_label, prob) in enumerate(zip(image_names, label_malignant, p_mass[:, 1])):
                worksheet_mass.write(idx + 1, 0, label=img_name)
                worksheet_mass.write(idx + 1, 1, label=str(binary_label))  # Binary label as 1 or 0
                worksheet_mass.write(idx + 1, 2, label=float(prob))  # Probability in decimal form

            workbook_mass.save(malignant_prob_excel_path)
            print(f'Benign vs malignant probabilities with binary labels saved to {malignant_prob_excel_path}')

            # ----- evaluate the (benign vs malignant) non_mass under the five-class classification ----- #
            bm_prob[:, 2] = bm_prob[:, 2] / (bm_prob[:, 2] + bm_prob[:, 3]) # benign non_mass
            bm_prob[:, 3] = 1 - bm_prob[:, 2]                               # malignant non_mass
            
            label_non_mass = []
            p_non_mass = []
            image_names = []
            for idx in range(len(labels)):
                # select only benign non_mass and malignant non_mass
                if labels[idx] == 2 or labels[idx] == 3:
                    label_non_mass.append(labels[idx])
                    p_non_mass.append(bm_prob[idx, :])
                    # Get image name from valid_dataset
                    img_path = valid_dataset.samples[idx][0]
                    img_name = img_path.split('/')[-1]
                    image_names.append(img_name)
            p_non_mass = np.array(p_non_mass)

            label_benign = [1 if label == 2 else 0 for label in label_non_mass]
            auc_score = roc_auc_score(label_benign, p_non_mass[:, 2])
            print('AUC score of benign non_mass: {:.4}'.format(auc_score))

            label_malignant = [1 if label == 3 else 0 for label in label_non_mass]
            auc_score = roc_auc_score(label_malignant, p_non_mass[:, 3])
            print('AUC score of malignant non_mass:{:.4}'.format(auc_score))

            # Save benign vs malignant (non-mass) probabilities to Excel (2025.09.25)
            malignant_prob_excel_path = os.path.join(
                'evaluations', args.finetuned_dataset_name, 
                f'{args.pretrained_model_name}_benign_vs_malignant(non-mass)_prob.xls'
            )
            workbook_mass = xlwt.Workbook(encoding='utf-8')
            worksheet_mass = workbook_mass.add_sheet('Benign_vs_Malignant(non-mass)')
            worksheet_mass.write(0, 0, label='Image_Name')  # Header for image names
            worksheet_mass.write(0, 1, label='Binary_Label')  # Header for binary labels (benign=0, malignant=1)
            worksheet_mass.write(0, 2, label='Malignant_Probability')  # Header for malignant probability

            # Write image names, binary labels, and malignant probabilities
            for idx, (img_name, binary_label, prob) in enumerate(zip(image_names, label_malignant, p_non_mass[:, 3])):
                worksheet_mass.write(idx + 1, 0, label=img_name)
                worksheet_mass.write(idx + 1, 1, label=str(binary_label))  # Binary label as 1 or 0
                worksheet_mass.write(idx + 1, 2, label=float(prob))  # Probability in decimal form

            workbook_mass.save(malignant_prob_excel_path)
            print(f'Benign vs malignant probabilities with binary labels saved to {malignant_prob_excel_path}')

            # ----- evaluate the (benign vs malignant) mass + non_mass under the five-class classification ----- #
            bm_prob = probs.copy()
            bm_prob[:, 0] = np.maximum(bm_prob[:, 0] , bm_prob[:, 2]) # all benign (mass + non_mass)
            bm_prob[:, 1] = np.maximum(bm_prob[:, 1] , bm_prob[:, 3]) # all malignant (mass + non_mass)
            bm_prob[:, 0] = bm_prob[:, 0] / (bm_prob[:, 0] + bm_prob[:, 1])
            bm_prob[:, 1] = 1 - bm_prob[:, 0]

            label_abnormal = []
            p_abnormal = []
            image_names = []
            for idx in range(len(labels)):
                if labels[idx] != 4: # not normal
                    label_abnormal.append(labels[idx])
                    p_abnormal.append(bm_prob[idx, :])
                    # Get image name from valid_dataset
                    img_path = valid_dataset.samples[idx][0]
                    img_name = img_path.split('/')[-1]
                    image_names.append(img_name)
            p_abnormal = np.array(p_abnormal)

            label_benign = [1 if (label == 0 or label == 2) else 0 for label in label_abnormal]
            auc_score = roc_auc_score(label_benign, p_abnormal[:, 0])
            print('AUC score of benign (mass and non_mass): {:.4}'.format(auc_score))

            label_malignant = [1 if (label == 1 or label == 3) else 0 for label in label_abnormal]
            auc_score = roc_auc_score(label_malignant, p_abnormal[:, 1])
            print('AUC score of malignant (mass and non_mass): {:.4}'.format(auc_score))

            # Save benign vs malignant (mass + non-mass) probabilities to Excel (2025.09.25)
            malignant_prob_excel_path = os.path.join(
                'evaluations', args.finetuned_dataset_name, 
                f'{args.pretrained_model_name}_benign_vs_malignant(mass+non-mass)_prob.xls'
            )
            workbook_mass = xlwt.Workbook(encoding='utf-8')
            worksheet_mass = workbook_mass.add_sheet('Benign_vs_Malignant(m_nm)')
            worksheet_mass.write(0, 0, label='Image_Name')  # Header for image names
            worksheet_mass.write(0, 1, label='Binary_Label')  # Header for binary labels (benign=0, malignant=1)
            worksheet_mass.write(0, 2, label='Malignant_Probability')  # Header for malignant probability

            # Write image names, binary labels, and malignant probabilities
            for idx, (img_name, binary_label, prob) in enumerate(zip(image_names, label_malignant, p_abnormal[:, 1])):
                worksheet_mass.write(idx + 1, 0, label=img_name)
                worksheet_mass.write(idx + 1, 1, label=str(binary_label))  # Binary label as 1 or 0
                worksheet_mass.write(idx + 1, 2, label=float(prob))  # Probability in decimal form

            workbook_mass.save(malignant_prob_excel_path)
            print(f'Benign vs malignant probabilities with binary labels saved to {malignant_prob_excel_path}')

            # ----- evaluate the (mass vs non-mass) all images without normal under the five-class classification ----- #
            mNm_prob = probs.copy()
            bm_prob[:, 0] = np.maximum(mNm_prob[:, 0] , mNm_prob[:, 1]) # all mass
            bm_prob[:, 1] = np.maximum(mNm_prob[:, 2] , mNm_prob[:, 3]) # all non_mass
            bm_prob[:, 0] = bm_prob[:, 0] / (bm_prob[:, 0] + bm_prob[:, 1])
            bm_prob[:, 1] = 1 - bm_prob[:, 0]

            label_abnormal = []
            p_abnormal = []
            for idx in range(len(labels)):
                if labels[idx] != 4:  # not normal
                    label_abnormal.append(labels[idx])
                    p_abnormal.append(bm_prob[idx, :])
                    # Get image name from valid_dataset
                    img_path = valid_dataset.samples[idx][0]
                    img_name = img_path.split('/')[-1]
                    image_names.append(img_name)
            p_abnormal = np.array(p_abnormal)

            label_mass = [1 if (label == 0 or label == 1) else 0 for label in label_abnormal]
            auc_score = roc_auc_score(label_mass, p_abnormal[:, 0])
            print('AUC score of mass (mass and non_mass): {:.4}'.format(auc_score))

            label_non_mass = [1 if (label == 2 or label == 3) else 0 for label in label_abnormal]
            auc_score = roc_auc_score(label_non_mass, p_abnormal[:, 1])
            print('AUC score of non_mass (mass and non_mass): {:.4}'.format(auc_score))

            # Save mass vs non-mass probabilities to Excel (2025.09.25)
            mass_prob_excel_path = os.path.join(
                'evaluations', args.finetuned_dataset_name, 
                f'{args.pretrained_model_name}_mass_vs_non-mass_prob.xls'
            )
            workbook_mass = xlwt.Workbook(encoding='utf-8')
            worksheet_mass = workbook_mass.add_sheet('Mass')
            worksheet_mass.write(0, 0, label='Image_Name')  # Header for image names
            worksheet_mass.write(0, 1, label='Binary_Label')  # Header for binary labels (non-mass=0, mass=1)
            worksheet_mass.write(0, 2, label='Mass_Probability')  # Header for mass probability

            # Write image names, binary labels, and non-mass probabilities
            for idx, (img_name, binary_label, prob) in enumerate(zip(image_names, label_mass, p_abnormal[:, 0])):
                worksheet_mass.write(idx + 1, 0, label=img_name)
                worksheet_mass.write(idx + 1, 1, label=str(binary_label))  # Binary label as 1 or 0
                worksheet_mass.write(idx + 1, 2, label=float(prob))  # Probability in decimal form

            workbook_mass.save(mass_prob_excel_path)
            print(f'Mass vs non-mass probabilities with binary labels saved to {mass_prob_excel_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--finetuned_dataset_name', type=str, default='')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_batchSize', default=1, type=int)
    parser.add_argument('--pretrained_model_name', type=str, default='')
    parser.add_argument('--test_auc', action='store_true', help='if true, test OvR AUC score')
    parser.add_argument('--probability_excel_path', type=str, default='')

    args = parser.parse_args()
    setup_seed(args.seed)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} is using GPU {torch.cuda.current_device()}")

    dist.init_process_group(backend='nccl', init_method='env://')
    setup_seed(args.seed)

    transform_valid = v2.Compose([
        v2.ToImage(),
        v2.Resize((args.input_size, args.input_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    valid_dataset = ImageFolderWithPaths(root=args.data_root + '/test', transform=transform_valid)
    
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.test_batchSize, sampler=valid_sampler, num_workers=8)

    pretrained_model_path = os.path.join('checkpoints', args.finetuned_dataset_name, args.pretrained_model_name+'.pt')
    print(pretrained_model_path)
    model = torch.load(pretrained_model_path, map_location='cpu')
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    model.eval()

    test_model(model, valid_dataset, valid_dataloader, local_rank, args)