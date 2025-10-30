import torch
import cv2
import numpy as np


def color_mapping(predicted_para):
    color_map = []
    for i in range(predicted_para.shape[0]):
        color_temp = predicted_para.data[i] 
        max_value = torch.max(color_temp)
        min_value = torch.min(color_temp)
        color_temp = ((color_temp - min_value) / (max_value - min_value)) * 255
        color_temp = color_temp.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        color_temp = cv2.applyColorMap(color_temp.astype(np.uint8), cv2.COLORMAP_JET)
        color_temp= cv2.cvtColor(color_temp, cv2.COLOR_BGR2RGB)
        color_temp = np.float32(color_temp) / 255
        color_temp = (color_temp - 0.5) / 0.5
        color_temp = torch.tensor(color_temp.transpose(2, 0, 1)).to(predicted_para.device)
        color_map.append(color_temp)
    color_map = torch.stack(color_map, dim=0)

    return color_map


def scaled_sigma(predicted_sigma):
    sigma_map = []
    for i in range(predicted_sigma.shape[0]):  
        sigma_temp = predicted_sigma.data[i]  
        max_value = torch.max(sigma_temp)
        min_value = torch.min(sigma_temp)
        sigma_temp = ((sigma_temp - min_value) / (max_value - min_value))        
        sigma_map.append(sigma_temp)
    sigma_map = torch.stack(sigma_map, dim=0)
    sigma_map = torch.exp(sigma_map / 0.75)

    return sigma_map


def calculate_variance(alpha, beta):
    a = 1/(alpha + 1e-5)
    a = torch.clip(a, min=1e-4, max=5)
    b = beta + 0.1
    b = torch.clip(b, min=0.1, max=5)
    sigma = a * (torch.exp(torch.lgamma(3/b))/torch.exp(torch.lgamma(1.0/b))) ** (1/2)

    return sigma