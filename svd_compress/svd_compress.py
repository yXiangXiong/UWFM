import os
import argparse
import numpy as np
from PIL import Image


def compress_color_image(image, k):
    compressed_channels = []
    for channel in range(3):
        U, S, Vt = np.linalg.svd(image[:, :, channel], full_matrices=False)
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        compressed_channel = U_k @ S_k @ Vt_k
        compressed_channels.append(compressed_channel)

    compressed_image = np.stack(compressed_channels, axis=2)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

    return compressed_image


def main(args):
    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)
    for mode_name in os.listdir(args.src_dir):
        print('Reading', mode_name, 'images from: ', args.src_dir)
        dst_mode = os.path.join(args.dst_dir, mode_name)
        if not os.path.exists(dst_mode):
            os.makedirs(dst_mode)
        mode_path = os.path.join(args.src_dir, mode_name)
        class_list = os.listdir(mode_path)
        for class_name in class_list:
            dst_class = os.path.join(dst_mode, class_name)
            if not os.path.exists(dst_class):
                os.makedirs(dst_class)
            class_path = os.path.join(mode_path, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)

                image = Image.open(file_path).convert('RGB')
                image = np.array(image)
                k = int(min(image.shape[0], image.shape[1]) * args.compression_ratio)
                image = compress_color_image(image, k)
                image = Image.fromarray(image)
                dst_path = os.path.join(dst_class, file_name)
                print(dst_path)
                image.save(dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Compression', add_help=False)
    parser.add_argument('--src_dir', default="", type=str,
                        help='source directory for images needed to be compressed.')
    parser.add_argument('--dst_dir', default="", type=str,
                        help='destination directory for compressed images to be saved.')
    parser.add_argument('--compression_ratio', default=0.1, type=float,
                        help='SVD image compression_ratio')
    args = parser.parse_args()

    main(args)