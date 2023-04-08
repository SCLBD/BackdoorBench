'''This script is to generate a black image with only a white square at the right corner, then convert it to a npy file'''

import numpy as np
import argparse
from PIL import Image

def generate_white_square_image(image_size, square_size, distance_to_right, distance_to_bottom):
    black_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    black_image[image_size - distance_to_bottom - square_size:image_size - distance_to_bottom, image_size - distance_to_right - square_size:image_size - distance_to_right, :] = 255
    return black_image

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--image_size', type=int, default=32)
    args.add_argument('--square_size', type=int, default=3)
    args.add_argument('--distance_to_right', type=int, default=0)
    args.add_argument('--distance_to_bottom', type=int, default=0)
    args.add_argument('--output_path', type=str, default='./trigger_image.png')
    args = args.parse_args()
    image = generate_white_square_image(
        args.image_size,
        args.square_size,
        args.distance_to_right,
        args.distance_to_bottom,
    )
    Image.fromarray(image).save(args.output_path)