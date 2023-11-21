#CAM MAMMOGRAM
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
# from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense#, Input, Concatenate
# from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
# from keras.applications.vgg16 import VGG16
# from keras.layers import Dense#, Dropout, Flatten, BatchNormalization
# import tensorflow as tf
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import cv2
from pandas import read_csv
# from sklearn.feature_extraction.image import PatchExtractor
# from patchify import patchify
# from np_utils import to_categorical
from sklearn.model_selection import train_test_split
# from statistics import mode
# import pickle
from skimage.util import view_as_windows
from tensorflow.keras.models import save_model#, load_model
# from tensorflow.keras.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping
from os.path import exists
from math import sqrt
from seaborn import histplot
from random import uniform
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim
# from pandas import DataFrame
# from itertools import chain
"""
BIRADS Categories

Category 0: Incomplete assessment - Additional imaging is needed for a complete evaluation. This may be due to technical issues or incomplete imaging.

Category 1: Negative - No significant findings; the breast is entirely normal.

Category 2: Benign - There are findings indicative of benign (non-cancerous) conditions, such as cysts or benign tumors.

Category 3: Probably benign - The findings have a very low probability of being cancerous. Follow-up imaging may be recommended to monitor changes.

Category 4: Suspicious - There is a moderate to high suspicion of malignancy. Further evaluation, such as a biopsy, is usually recommended.

Category 5: Highly suggestive of malignancy - The findings are highly suspicious of being cancerous. Immediate action, like a biopsy, is usually recommended.

Category 6: Known biopsy-proven malignancy - The cancer has already been confirmed through a biopsy.
"""
WINDOW_SIZE = 222
def read_df(path="/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"):
    df = read_csv(os.path.join(path,"finding_annotations.csv"))
    df['breast_birads'] = df['breast_birads'].str.replace('BI-RADS ', '')
    df['breast_birads'] = df['breast_birads'].astype(int)
    return df[['study_id', 'image_id','view_position','breast_birads','xmin','xmax','ymin','ymax']]
def convert_dicom_to_png(dicom_file: str) -> np.ndarray:
    """
    dicom_file: path to the dicom fife

    return
        gray scale image with pixel intensity in the range [0,255]
        None if cannot convert

    """
    data = pydicom.read_file(dicom_file)
    if ('WindowCenter' not in data) or\
       ('WindowWidth' not in data) or\
       ('PhotometricInterpretation' not in data) or\
       ('RescaleSlope' not in data) or\
       ('PresentationIntentType' not in data) or\
       ('RescaleIntercept' not in data):

        print(f"{dicom_file} DICOM file does not have required fields")
        return

    intentType = data.data_element('PresentationIntentType').value
    if ( str(intentType).split(' ')[-1]=='PROCESSING' ):
        print(f"{dicom_file} got processing file")
        return


    c = data.data_element('WindowCenter').value # data[0x0028, 0x1050].value
    w = data.data_element('WindowWidth').value  # data[0x0028, 0x1051].value
    if type(c)==pydicom.multival.MultiValue:
        c = c[0]
        w = w[0]

    photometricInterpretation = data.data_element('PhotometricInterpretation').value

    try:
        a = data.pixel_array
    except:
        print(f'{dicom_file} Cannot get get pixel_array!')
        return

    slope = data.data_element('RescaleSlope').value
    intercept = data.data_element('RescaleIntercept').value
    a = a * slope + intercept

    try:
        pad_val = data.get('PixelPaddingValue')
        pad_limit = data.get('PixelPaddingRangeLimit', -99999)
        if pad_limit == -99999:
            mask_pad = (a==pad_val)
        else:
            if str(photometricInterpretation) == 'MONOCHROME2':
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    except:
        # Manually create padding mask
        # this is based on the assumption that padding values take majority of the histogram
        print(f'{dicom_file} has no PixelPaddingValue')
        a = a.astype(np.int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        try:
            # if the second most frequent value (if any) is significantly more frequent than the third then
            # it is also considered padding value
            if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
                mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
                print(f'{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}')
        except:
            print(f'{dicom_file} most frequent pixel value {sorted_pixels[0]}')

    # apply window
    mm = c - 0.5 - (w-1)/2
    MM = c - 0.5 + (w-1)/2
    a[a<mm] = 0
    a[a>MM] = 255
    mask = (a>=mm) & (a<=MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w-1) + 0.5) * 255

    if str( photometricInterpretation ) == 'MONOCHROME1':
        a = 255 - a

    a[mask_pad] = 0
    return a

def load_images():
    # Define the image size and number of channels
    # img_rows, img_cols = GLOBAL_X, GLOBAL_Y   
    num_channels = 1  # change to 1 for grayscale images
    # Define the paths to the image folders
    global_dir = "/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"
    save_all_dir = []
    for root, dirs, files in os.walk(global_dir):
        for dir in dirs:
            # Combine the directory with the main folder to get the complete path
            folder_path = os.path.join(root, dir)
            # Append the folder location to the list
            save_all_dir.append(folder_path)
    return save_all_dir

def clahe(image):
    A_cv2 = image.astype(np.uint8)
    tile_s0 = 8
    tile_s1 = 8
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(tile_s0,tile_s1))
    clahe = clahe.apply(A_cv2)
    clahe = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)
    return clahe

def get_lesion_size(list_mass,list_all_values):
    list_mass = [int(x) for x in list_mass]
    #Euclidean distance
    list_all_values.append(int(sqrt((list_mass[1] - list_mass[0])**2 + (list_mass[3] - list_mass[2])**2)))
    return list_all_values

def random_patch_abnormal(image,list_mass):
    """
    Small: 222.0
    Medium (Median): 344
    Large: 522.0
    """
    window_size = WINDOW_SIZE/2 
    list_mass = [int(x) for x in list_mass]
    xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
    #iterate until sampled adequately
    previous_patch = None
    max_attempts = 20   
    attempt = 0
    saved_patches = []
    lesion_im = image[list_mass[2]:list_mass[3], 
                        list_mass[0]:list_mass[1]]
    while attempt < max_attempts:
        x_low = uniform(xmin, xmax - window_size)
        x_high = x_low + window_size
        y_low = uniform(ymin, ymax - window_size)
        y_high = y_low + window_size
        # print(f'uniform estimation: {int(y_low),int(y_high), int(x_low), int(x_high)}')
        patch = image[int(y_low):int(y_high), int(x_low):int(x_high)]
        patch = {
            'box': (x_low, y_low, x_high, y_high),
            'data': image[int(y_low):int(y_high), int(x_low):int(x_high)]
        }
        if len(saved_patches) > 0:
            if not is_similar(patch, saved_patches):
                # Save the patch
                saved_patches.append(patch)
                # Plot the patch with the red bounding box
                # plt.imshow(lesion_im, cmap='gray')
                # rect = patches.Rectangle((0, 0), window_size, window_size, linewidth=2, edgecolor='red', facecolor='none')
                # plt.gca().add_patch(rect)
                # plt.show()
        else:
            saved_patches.append(patch)
        attempt += 1
    save_lesion = []
    for img in saved_patches:
        if img['data'].shape == (window_size, window_size, 3):
            save_lesion.append(img['data'])
    print(f'length of list {len(saved_patches)}')
    return np.array(save_lesion)

def is_similar(patch, existing_patches):
    for existing_patch in existing_patches:
        iou_score = calculate_iou(existing_patch['box'], patch['box'])
        if iou_score < 0.90:
            return True
    return False

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the intersection area
    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))

    # Calculate the area of each bounding box
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    # Calculate the Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def create_small_medium_large(list_all_values):
    q1 = int(np.percentile(list_all_values, 25))
    q2 = int(np.percentile(list_all_values, 50))
    q3 = int(np.percentile(list_all_values, 75))
    if q1 % 2 != 0:  # Check if it's odd
        q1 += 1
    if q2 % 2 != 0:  # Check if it's odd
        q2 += 1
    if q3 % 2 != 0:  # Check if it's odd
        q3 += 1
    print(f'number of lesions: {len(list_all_values)}')
    with open("quartiles.txt", "w") as file:
        file.write(f"Small: {q1}\n")
        file.write(f"Medium (Median): {q2}\n")
        file.write(f"Large: {q3}\n")

def main():
    glob_dir = '/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
    df = read_df()
    df.dropna(inplace=True)
    list_all_values = [] 
    combined_array_lesion = []
    combined_array = np.empty((0, WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    for iteration, (index, row) in enumerate(df.iterrows()):
            image_type = row['image_id'] + '.dicom'
            dicom_path = os.path.join(glob_dir,row['study_id'],image_type)
            if os.path.exists(dicom_path):
                png_file = convert_dicom_to_png(dicom_path)
                #abnormal images
                if row['breast_birads'] > 1:
                    clahe_image = clahe(png_file)
                    list_all_values = get_lesion_size([row['xmin'],row['xmax'],row['ymin'],row['ymax']],list_all_values)
                    lesion_images = random_patch_abnormal(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']])
                    print(np.concatenate(lesion_images).shape)
                    input()
                    # combined_array_lesion.append(lesion_images)
                    print(np.shape(combined_array))
    
    #THIS PART IS DONE
    # #create small medium and large patches
    # create_small_medium_large(list_all_values)
    # #plot hist
    # histplot(list_all_values, bins='auto', kde=True, color='tab:blue')
    # plt.xlabel('Euclidean Distance of the Bounding Box')
    # plt.ylabel('Count')
    # plt.savefig('hist_sizes_lesions.png',dpi=350)
    # plt.close()
    
if __name__ == "__main__":
    main()