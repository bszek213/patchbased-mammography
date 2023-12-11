import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import cv2
from pandas import read_csv
from sys import argv
from os.path import exists
from math import sqrt
from random import uniform#, sample
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows
# from seaborn import kdeplot
from matplotlib import patches
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SMALL_CUTOFF_AREA = 0.0015709008023612142 #0.0010151872005197758 #what bebis wants, 0.0015709008023612142 #what the average is
MEDIUM_CUTOFF_AREA = 0.004060748802079103 #what bebis wants, what the ae0.006415104487709461
LARGE_CUTOFF_AREA = 0.02431239780756779
NUM_IMAGES = 5

#Based on average size - converted to pixels
SMALL_PATCH = 80 #bebis wants this to be 60x60 or 80x80 was 124
MEDIUM_PATCH = 200 #bebsis wants to be 200x200 was 250
LARGE_PATCH = 490

def load_images(type='small'):
    #images
    train = os.path.join('data', f'{type}_images_train.npy')
    validate = os.path.join('data', f'{type}_images_validation.npy')
    test = os.path.join('data', f'{type}_images_test.npy')

    #annotations
    train_an = os.path.join('data', f'{type}_annotations_train.npy')
    validate_an = os.path.join('data', f'{type}_annotations_validation.npy')
    test_an = os.path.join('data', f'{type}_annotations_test.npy')

    train_data = np.load(train, allow_pickle=True)
    validation_data = np.load(validate, allow_pickle=True)
    test_data = np.load(test, allow_pickle=True)

    train_data_an = np.load(train_an, allow_pickle=True)
    validation_data_an = np.load(validate_an, allow_pickle=True)
    test_data_an = np.load(test_an, allow_pickle=True)

    return train_data, validation_data, test_data, train_data_an, validation_data_an, test_data_an

def create_patches(image,multiplication,type,annotations):
    if type == 'Small':
        patch_size = int(SMALL_PATCH * float(multiplication))
    elif type == 'Medium':
        patch_size = int(MEDIUM_PATCH * float(multiplication))
    elif type == 'Large':
        patch_size = int(LARGE_PATCH * float(multiplication))

    xmin, xmax, ymin, ymax = annotations[0], annotations[1], annotations[2], annotations[3]

    #Center
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    #randomly vary
    variation = 5 #pixel variability
    center_x = center_x + uniform(-variation, variation)
    center_y = center_y + uniform(-variation, variation)

    xmin_new, xmax_new = center_x - patch_size / 2, center_x + patch_size / 2
    ymin_new, ymax_new = center_y - patch_size / 2, center_y + patch_size / 2

    # mask = (xmin_new >= 0) & (xmax_new <= image.shape[1]) & (ymin_new >= 0) & (ymax_new <= image.shape[0])

    # Create an array of zeros with the size of the patch
    patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)

    # Calculate the overlap of the patch with the image
    overlap_x_start = max(0, -xmin_new)
    overlap_x_end = min(patch_size, image.shape[1] - int(xmin_new))
    overlap_y_start = max(0, -ymin_new)
    overlap_y_end = min(patch_size, image.shape[0] - int(ymin_new))

    # Copy the overlapping region from the image to the patch
    patch[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end] = \
        image[int(ymin_new) + overlap_y_start:int(ymin_new) + overlap_y_end,
              int(xmin_new) + overlap_x_start:int(xmin_new) + overlap_x_end]
    
    plt.imshow(patch, cmap='gray')
    plt.tight_layout()
    plt.title(f'small window')
    plt.show()
    
    # if not mask.all():
    #     xmin_new = max(0, xmin_new)
    #     xmax_new = min(image.shape[1], xmax_new)
    #     ymin_new = max(0, ymin_new)
    #     ymax_new = min(image.shape[0], ymax_new)
    #     patch = {
    #         'box': (xmin_new, ymin_new, xmax_new, ymax_new)
    #     }
    #     # Fill outside regions with zeros
    #     patched_image = np.zeros_like(image)
    #     patched_image[int(ymin_new):int(ymax_new), int(xmin_new):int(xmax_new)] = image[int(ymin_new):int(ymax_new), int(xmin_new):int(xmax_new)]
    #     box = patch['box']
    #     plt.imshow(patched_image, cmap='gray')
    #     # The lesion
    #     bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
    #                                      label='lesion', facecolor='none')
    #     plt.gca().add_patch(bounding_box)
    #     # The sampled patch
    #     rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='yellow',
    #                              facecolor='none')
    #     plt.gca().add_patch(rect)
    #     plt.tight_layout()
    #     plt.title(f'small window')
    #     plt.legend()
    #     plt.show()

def main():
    train_data, validation_data, test_data, train_data_an, validation_data_an, test_data_an = load_images('small')

    for train, valid, test, train_an, valid_an, test_an in zip(train_data, validation_data, test_data, train_data_an, validation_data_an, test_data_an):
        create_patches(train,1,'Small',train_an)

if __name__ == '__main__':
    main()