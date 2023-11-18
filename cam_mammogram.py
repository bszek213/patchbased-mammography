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

GLOBAL_X = 250
GLOBAL_Y = 250
PATCH_SIZE = (GLOBAL_X, GLOBAL_Y)

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
    # cv2.imshow('Original Image', image)
    # cv2.imshow('CLAHE Enhanced Image', clahe_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def create_patch_model_dense(input_shape):
    """
    input shape = (250,250,3,12)
    """
    # input_layer = Input(shape=input_shape)
    # rgb_input = Concatenate()([input_layer, input_layer, input_layer])
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(GLOBAL_X, GLOBAL_Y, 3))
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(GLOBAL_X, GLOBAL_Y, 3))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    # x = GlobalAveragePooling2D()(base_model.output)
    # x = Dense(128, activation='relu')(x)
    # outputs = Dense(3, activation='softmax')(x)
    # x = GlobalAveragePooling2D()(base_model.output)
    # # x = Dense(128, activation='relu')(x)
    # outputs = Dense(3, activation='softmax')(x)
    # patch_model = Model(inputs=input_layer, outputs=outputs)
    # base_model = ResNet152V2(include_top=False, weights='imagenet', input_shape=input_shape)
    # x = GlobalAveragePooling2D()(base_model.output)
    # x = Dense(128, activation='relu')(x)
    # outputs = Dense(2, activation='softmax')(x)  # Two classes: benign and malignant

    # patch_model = Model(inputs=base_model.input, outputs=outputs)
    return model

def create_patch_model_res(input_shape):
    """
    input shape = (250,250,3,12)
    """
    # input_layer = Input(shape=input_shape)
    # rgb_input = Concatenate()([input_layer, input_layer, input_layer])
    base_model = ResNet152V2(include_top=False, weights='imagenet', input_shape=(GLOBAL_X, GLOBAL_Y, 3))
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(GLOBAL_X, GLOBAL_Y, 3))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation="softmax"))
    model.compile(loss=BinaryFocalCrossentropy(), optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    return model

# def create_global_model(num_rows,num_columns):
#     # Input shape for each patch (1 channel for grayscale)
#     patch_input_shape = (*PATCH_SIZE, 1)

#     # Create a list of patch models
#     patch_models = [create_patch_model(patch_input_shape) for _ in range(num_rows * num_columns)]

#     # Input shape for the global model
#     global_input_shape = (num_rows, num_columns, *patch_input_shape)

#     # Create a global model that takes a grid of patches as input
#     global_input = Input(shape=global_input_shape)
#     patch_inputs = tf.split(global_input, num_rows * num_columns, axis=(0, 1))

#     outputs = [patch_models[i](patch_inputs[i]) for i in range(num_rows * num_columns)]

#     # Flatten the patch results and classify globally
#     x = tf.concat(outputs, axis=1)
#     global_output = Dense(2, activation='softmax')(x)  # Two classes: benign and malignant

#     global_model = Model(inputs=global_input, outputs=global_output)
#     global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     return global_model

# def resNet152_baseline():
#     optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9,beta_2=0.999)

#     base_model = ResNet152V2(include_top=False, input_shape=(GLOBAL_X, GLOBAL_Y, 3))
#     for layer in base_model.layers:
#         layer.trainable = False
#     model = Sequential()
#     model.add(base_model)
#     model.add(Flatten())
#     model.add(Dense(3, activation="softmax"))
#     model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics="accuracy")
#     return model

def patch(image,list_mass,GLOBAL_LESION_COUNT):
    """
    Upper left corner 0,0 pixel in python
    """
    step_x = max(GLOBAL_X, GLOBAL_Y)
    step_y = max(GLOBAL_X, GLOBAL_Y)

    # Extract non-overlapping patches
    patches = view_as_windows(image, (GLOBAL_Y, GLOBAL_X, 3), step=(step_y, step_x, 3))
    # patches=patchify(image,(GLOBAL_X,GLOBAL_Y),step=GLOBAL_X)
    list_mass = [int(x) for x in list_mass]
    xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
    shape_patches = np.shape(patches)
    # black_images, tissue_images, lesion_images  = {}, {}, {}
    all_images = {}
    x_pix_curr, y_pix_curr = 0,0
    # print(list_mass)
    #FIND NONLESIONED SUBIMAGES
    for i in range(shape_patches[0]): #iterates over y
        y_pix_curr = i * step_y
        x_pix_curr = 0
        for j in range(shape_patches[1]): #iterates over x
            x_pix_curr = j * step_x

            patch_xmin = x_pix_curr
            patch_xmax = patch_xmin + GLOBAL_X
            patch_ymin = y_pix_curr
            patch_ymax = patch_ymin + GLOBAL_Y
            sample_patch = patches[i, j, :, :]
            sample_patch = sample_patch[0]

            #black images
            if np.all(sample_patch < 40):
                continue
                # all_images[i,j] = [sample_patch,0]
            #lesion images
            elif (
                # ((patch_xmin <= xmin <= patch_xmax) and 
                #   (patch_ymin <= ymin <= patch_ymax)) or 
                #   ((patch_xmin <= xmin <= patch_xmax) and 
                #   (patch_ymin <= ymax <= patch_ymax)) or 
                #   ((patch_xmin <= xmax <= patch_xmax) and 
                #   (patch_ymin <= ymin <= patch_ymax)) or 
                #   ((patch_xmin <= xmax <= patch_xmax) and 
                #   (patch_ymin <= ymax <= patch_ymax)) or 



                #   ((patch_xmin <= xmin <= patch_xmax) and 
                #    (ymin <= y_pix_curr <= ymax)) or 

                #   ((patch_xmin <= xmax <= patch_xmax) and 
                #    (ymin <= y_pix_curr <= ymax))
                (
                (patch_xmin <= xmin <= patch_xmax) and 
                  (patch_ymin <= ymin <= patch_ymax)) or 
                  ((patch_xmin <= xmin <= patch_xmax) and 
                  (patch_ymin <= ymax <= patch_ymax)) or 
                  ((patch_xmin <= xmax <= patch_xmax) and 
                  (patch_ymin <= ymin <= patch_ymax)) or 
                  ((patch_xmin <= xmax <= patch_xmax) and 
                  (patch_ymin <= ymax <= patch_ymax)) or 

                  ((patch_xmin <= xmin <= patch_xmax) and 
                   (ymin <= y_pix_curr <= ymax)) or 
                  ((patch_xmin <= xmax <= patch_xmax) and 
                   (ymin <= y_pix_curr <= ymax)) or 

                   ((patch_ymin <= ymin <= patch_ymax) and
                    (xmin <= x_pix_curr <= xmax)) or 
                    ((patch_ymin <= ymax <= patch_ymax) and
                    (xmin <= x_pix_curr <= xmax))


                  ): #patch_xmin >= xmin and patch_xmax <= xmax and patch_ymin >= ymin and patch_ymax <= ymax
                continue
                # all_images[i,j] = [sample_patch,1]
            #tissue images
            else:
                all_images[i,j] = [sample_patch,0]
    #FIND THE SUBIMAGES OF THE LESIONS
    center_x = (list_mass[0] + list_mass[1]) // 2
    center_y = (list_mass[2] + list_mass[3]) // 2

    if (list_mass[1] - list_mass[2] < GLOBAL_X) or (list_mass[3] - list_mass[2] < GLOBAL_Y):
        subimage_xmin = max(center_x - GLOBAL_X // 2, 0)
        subimage_xmax = min(center_x + GLOBAL_X // 2, image.shape[1])
        subimage_ymin = max(center_y - GLOBAL_Y // 2, 0)
        subimage_ymax = min(center_y + GLOBAL_Y // 2, image.shape[0])
    else:
        subimage_xmin = list_mass[0]
        subimage_xmax = list_mass[1]
        subimage_ymin = list_mass[2]
        subimage_ymax = list_mass[3]

    subimage = image[subimage_ymin:subimage_ymax, subimage_xmin:subimage_xmax]
    
    # subimage_height, subimage_width, _ = subimage.shape
    subimage_list, lesions_label = [], []
    # num_x_patches = (subimage_xmax - subimage_xmin) // GLOBAL_X
    # num_y_patches = (subimage_ymax - subimage_ymin) // GLOBAL_Y
    window_size = 75
    num_x_patches = (subimage_xmax - subimage_xmin + 1) // window_size
    num_y_patches = (subimage_ymax - subimage_ymin + 1) // window_size
    # fig, ax = plt.subplots()
    # ax.imshow(subimage, cmap='gray')
    combined_array_lesion = []
    for y in range(subimage_ymin - GLOBAL_Y // 2, subimage_ymax - GLOBAL_Y // 2 + 1, window_size):
        for x in range(subimage_xmin - GLOBAL_X // 2, subimage_xmax - GLOBAL_X // 2 + 1, window_size):
            patch_xmin = x
            patch_xmax = patch_xmin + GLOBAL_X
            patch_ymin = y
            patch_ymax = patch_ymin + GLOBAL_Y
            patch = image[patch_ymin:patch_ymax, patch_xmin:patch_xmax]
            if patch.shape == (250, 250, 3):
                combined_array_lesion.append(patch)
            # plt.imshow(patch,cmap='gray')
            # plt.pause(0.15)  # Pause for 500 milliseconds
            # plt.clf()
            # rect = plt.Rectangle((patch_xmin, patch_ymin), GLOBAL_X, GLOBAL_Y, linewidth=2, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)    
    lesion_image_array = np.array(combined_array_lesion)
    GLOBAL_LESION_COUNT += len(combined_array_lesion)
    # print(f'number of lesion images: {len(subimage_list)}')
    # print(f'total lesion images: {GLOBAL_LESION_COUNT}')
    # print(np.shape(subimage_list))
    # print(len(subimage_list))
    # print('==================')
    # plt.show()
    # # print(np.shape(subimage_list))
    # # print(len(subimage_list))
    # input()
    # fig, axes = plt.subplots(1, len(subimage_list), figsize=(12, 3))
    # for i, subimage_part in enumerate(subimage_list):
    #         if len(subimage_list) == 1:
    #             ax = axes
    #             print(subimage_part.shape)
    #             input()
    #             ax.imshow(subimage_part, cmap='gray')
    #             ax.axis('off')
    #             ax.set_title(f'Lesion {i+1}')
    #         else:
    #             ax = axes[i]
    #             ax.imshow(subimage_part, cmap='gray')
    #             ax.axis('off')
    #             ax.set_title(f'Lesion {i+1}')

    # plt.tight_layout()
    # plt.show()
    
    # label_colors = {0: 'green', 1: 'red', 2: 'blue'} # 0 = black, 1 = lesion, 2 = tissue
    # border_thickness = 2
    # num_rows = shape_patches[0]
    # num_cols = shape_patches[1]
    # plt.figure(figsize=(15, 10))
    # ix = 1
    # for i in range(shape_patches[0]): #iterates over y
    #     for j in range(shape_patches[1]): #iterates over x
    #         # Get the image and label for the current position
    #         if (i,j) in all_images:
    #             sample_patch, label = all_images[i, j]

    #             patch_ymin = i * step_x
    #             patch_ymax = patch_ymin + GLOBAL_Y
    #             patch_xmin = j * step_x
    #             patch_xmax = patch_xmin + GLOBAL_X

    #             # Specify subplot and turn off axis
    #             ax = plt.subplot(num_rows, num_cols, ix)
    #             ax.set_xticks([])
    #             ax.set_yticks([])

    #             # Plot the image with the colored border
    #             plt.imshow(sample_patch, cmap='gray')   
    #             ax.spines['top'].set_color('none')
    #             ax.spines['bottom'].set_color('none')
    #             ax.spines['left'].set_color('none')
    #             ax.spines['right'].set_color('none')
    #             ax.set_xticks([])
    #             ax.set_yticks([])

    #             # Add the colored border based on the label
    #             ax.spines['top'].set_color(label_colors[label])
    #             ax.spines['bottom'].set_color(label_colors[label])
    #             ax.spines['left'].set_color(label_colors[label])
    #             ax.spines['right'].set_color(label_colors[label])

    #             # Increase the border thickness
    #             ax.spines['top'].set_linewidth(border_thickness)
    #             ax.spines['bottom'].set_linewidth(border_thickness)
    #             ax.spines['left'].set_linewidth(border_thickness)
    #             ax.spines['right'].set_linewidth(border_thickness)
    #             title_text = f"{patch_xmin},{patch_xmax}\n" \
    #                     f"{patch_ymin},{patch_ymax}"
    #             ax.text(-0.7, 0.5, title_text, transform=ax.transAxes, fontsize=8, va='center', ha='center', rotation='horizontal')
    #             # plt.title(title_text,fontsize=10

    #             ix += 1
    # plt.savefig('labeled_patches.png',dpi=450)
    # plt.show()
    #NON LESION CLASSIFICATIONS
    feature_list, label_list = [], []
    for key , (matrix, integer) in all_images.items():
        feature_list.append(matrix)
        label_list.append(int(integer))
    #remove excess black images
    # feature_list, label_list = equalize_0_and_2(feature_list, label_list)
    #LESION CLASSIFICATIONS
    if len(combined_array_lesion) > 0:
        label_lesions_list = [1] * len(combined_array_lesion)
        # print(f'number of labels for lesions: {len(combined_array_lesion)}')
        return feature_list, lesion_image_array, label_list, label_lesions_list, GLOBAL_LESION_COUNT
    else:
        return feature_list, None, label_list, None, GLOBAL_LESION_COUNT
    # plt.savefig('patch_example.png',dpi=450)
    # plt.show()
    # pe = PatchExtractor(patch_size=(GLOBAL_X, GLOBAL_Y))
    # patches = pe.transform(image)
    # print(patches)
    
    # input()
def equalize_0_and_2(image_list, label_list):
    num_label_0 = np.sum(np.array(label_list) == 0)
    num_label_2 = np.sum(np.array(label_list) == 2)

    if num_label_0 > num_label_2:
        to_remove = num_label_0 - num_label_2
        i = 0
        while to_remove > 0 and i < len(label_list):
            if label_list[i] == 0:
                del label_list[i]
                del image_list[i]
                to_remove -= 1
            else:
                i += 1
    elif num_label_2 > num_label_0:
        to_remove = num_label_2 - num_label_0
        i = 0
        while to_remove > 0 and i < len(label_list):
            if label_list[i] == 2:
                del label_list[i]
                del image_list[i]
                to_remove -= 1
            else:
                i += 1

    return image_list, label_list

def randomly_select_half_data(data, labels):
    data_to_keep = []
    labels_to_keep = []
    data_with_label_1 = []

    for i, label_list in enumerate(labels):
        if 1 in label_list:
            data_with_label_1.append(i)
        else:
            data_to_keep.append(data[i])
            labels_to_keep.append(label_list)

    num_samples = len(data_to_keep)
    num_samples_to_select = num_samples // 2

    if num_samples_to_select > len(data_with_label_1):
        num_samples_to_select = len(data_with_label_1)

    selected_indices = np.random.choice(data_with_label_1, num_samples_to_select, replace=False)

    for i in selected_indices:
        data_to_keep.append(data[i])
        labels_to_keep.append(labels[i])
    return data_to_keep, labels_to_keep

def display_image(image):
    from PIL import Image
    # Downsample the image to 300x300
    # Convert the NumPy array to a Pillow image
    original_image = Image.fromarray(image, mode='L')  # 'L' mode represents grayscale

    # Downsample the image to 300x300
    downsampled_image = original_image.resize((300, 300), Image.ANTIALIAS)

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image in the first subplot
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')

    # Display the downsampled image in the second subplot
    ax2.imshow(np.array(downsampled_image), cmap='gray')
    ax2.set_title('Downsampled Image')

    # Hide axis labels
    ax1.axis('off')
    ax2.axis('off')

    # Show the Matplotlib figure
    plt.savefig('downsampleVsOG.png',dpi=400)
    plt.close()

def count_ones_in_sublists(sublist_list):
    count = 0
    for sublist in sublist_list:
        count += sublist.count(1)
    return count

def main():
    # all_dir = load_images()
    dict_save_benign = {}
    # dict_image_benign= {}
    dict_save_malig = {}
    # dict_image_malig = {}
    dict_image_malig, dict_image_benign = [], []
    label_benign, label_malignant = [], []
    row_list, col_list = [], []
    list_features_total, list_labels_total, list_images_lesion, list_label_images = [], [], [], []
    combined_array = np.empty((0, 250, 250, 3), dtype=np.uint8)
    combined_array_lesion = np.empty((0, 250, 250, 3), dtype=np.uint8)
    list_labels_total_np = np.array([])
    if not os.path.exists('X_train.npy'):
        glob_dir = '/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
        df = read_df()
        df.dropna(inplace=True)
        GLOBAL_LESION_COUNT = 0
        for iteration, (index, row) in enumerate(df.iterrows()):
            image_type = row['image_id'] + '.dicom'
            dicom_path = os.path.join(glob_dir,row['study_id'],image_type)
            list_of_dicts = []
            if os.path.exists(dicom_path):
                png_file = convert_dicom_to_png(dicom_path)
                if png_file is not None:
                    #BENIGNlist_images_lesion
                    if row['breast_birads'] > 1 and row['breast_birads'] < 4:
                        clahe_image = clahe(png_file)
                        dict_save_benign[index] = [row['view_position'],row['breast_birads']]   
                        # dict_image_benign[index] = patch(clahe_image)
                        feature_list, subimage_list, label_list, label_lesions_list, GLOBAL_LESION_COUNT  = patch(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']],GLOBAL_LESION_COUNT)
                        if iteration % 2 == 0:# and iteration != 0:
                            try:
                                combined_array = np.vstack((combined_array, feature_list))
                                list_labels_total_np = np.concatenate((list_labels_total_np, np.array(label_list)))
                                # list_labels_total.append(label_list)
                                # print(list_labels_total)
                            except:
                                    print('dimension issues. Just do not use that data')
                        # combined_array = np.vstack((combined_array, feature_list))
                        # list_features_total.append(np.array(feature_list))
                        # list_labels_total.append(label_list)
                        # if len(subimage_list) >= 0:
                        #     # list_images_lesion.append(np.array(subimage_list))
                            
                        if label_lesions_list != None:
                            list_label_images.append(label_lesions_list)
                            combined_array_lesion = np.vstack((combined_array_lesion, subimage_list))
                        print(np.shape(combined_array))
                        print(f'number of lesions: {GLOBAL_LESION_COUNT}')
                        # print(np.shape(combined_array_lesion))
                        # print(np.shape(list_features_total))
                        # print(count_ones_in_sublists(list_labels_total))
                        # row_list.append(np.shape(image_patch)[0])
                        # col_list.append(np.shape(image_patch)[1])
                        # dict_image_benign.append(image_patch)
                        # label_benign.append(0)
                        # print(f'length of benign features: {len(dict_image_benign)}')
                        # print(f'length of benign labels: {len(label_benign)}')
                    #MALIGNANT
                    elif row['breast_birads'] > 3:
                        clahe_image = clahe(png_file)
                        # display_image(clahe_image)
                        dict_save_malig[index] = [row['view_position'],row['breast_birads']]
                        feature_list, subimage_list, label_list, label_lesions_list, GLOBAL_LESION_COUNT = patch(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']],GLOBAL_LESION_COUNT)
                        if iteration % 2 == 0:# and iteration != 0:
                            try:
                                combined_array = np.vstack((combined_array, feature_list))
                                list_labels_total_np = np.concatenate((list_labels_total_np, np.array(label_list)))
                                # list_labels_total.append(label_list)
                                # print(list_labels_total)
                            except:
                                    print('dimension issues. Just do not use that data')
                        # list_features_total.append(np.array(feature_list))
                        # if len(subimage_list) >= 0:
                        #     # list_images_lesion.append(np.array(subimage_list))
                            
                        if label_lesions_list != None:
                            list_label_images.append(label_lesions_list)
                            combined_array_lesion = np.vstack((combined_array_lesion, subimage_list))
                        print(np.shape(combined_array))
                        print(f'number of lesions: {GLOBAL_LESION_COUNT}')
                        # print(np.shape(combined_array))
                        # print(count_ones_in_sublists(list_label_images))
                        # print(count_ones_in_sublists(label_list))
                        # row_list.append(np.shape(image_patch)[0])
                        # col_list.append(np.shape(image_patch)[1])
                        # dict_image_malig.append(image_patch)
                        # label_malignant.append(1)
                        # print(f'length of malignant features: {len(dict_image_malig)}')
                        # print(f'length of malignant labels: {len(label_malignant)}')
                        # print(np.shape(dict_image_malig)

        #train-valid split
    #     feature_data = dict_image_benign + dict_image_malig
    #     label_benign = np.array(label_benign, dtype='int32')
    #     label_malignant = np.array(label_malignant, dtype='int32')
    #     labels = np.concatenate((label_benign, label_malignant))
        # list_features_total_adj,list_labels_total_adj = randomly_select_half_data(list_features_total,list_labels_total)
        list_ones = [item for sublist in list_label_images for item in sublist]
        # list_other = [item for sublist in label_list for item in sublist]

        array_lesion = np.array(list_ones, dtype="int") #here is the problem
        # array_non_lesion = np.array(label_list,  dtype="int")
        labels = to_categorical(np.concatenate((list_labels_total_np, array_lesion)), 
                                               num_classes=2, dtype="int")
        
        # for sublist in list_features_total_adj:
        #     print(np.shape(sublist))
        features = np.concatenate((combined_array,combined_array_lesion), axis=0)
        # np_image_lesion = np.concatenate(list_images_lesion, axis=0)
        # np_image_non_lesion = np.vstack(list_features_total_adj)
        # np_image_lesion = np.vstack(list_images_lesion)
        # print(np.shape(np_image_non_lesion))
        # print(np.shape(np_image_lesion))
        # features = np.concatenate((np_image_non_lesion,np_image_lesion))
        print(np.shape(features))
        print(np.shape(labels))
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Save the data
        np.save('X_train.npy', X_train)
        np.save('X_validation.npy', X_validation)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_validation.npy', y_validation)
        np.save('y_test.npy', y_test)

    else:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        x_val = np.load('X_validation.npy')
        y_val = np.load('y_validation.npy')

    print(f'x_train size {np.shape(X_train)}')
    print(f'X_test size {np.shape(X_test)}')
    print(f'y_train size {np.shape(y_train)}')
    print(f'y_test size {np.shape(y_test)}')
    print(f'x_val size {np.shape(x_val)}')
    print(f'y_val size {np.shape(y_val)}')

    #label counts
    label_counts = np.sum(y_train, axis=0)
    total_labels = len(y_train)
    label_percentages = (label_counts / total_labels) * 100
    for label, percentage in enumerate(label_percentages):
        label_str = " ".join([str(int(val)) for val in y_train[label]])
        print(f"Label [{label_str}]: {percentage:.2f}%")
    # X_train = X_train.reshape(X_train.shape + (1,))
    # X_test = X_test.reshape(X_test.shape + (1,))
    # X_train = np.concatenate([X_train, X_train, X_train], axis=-1)
    # X_test = np.concatenate([X_test, X_test, X_test], axis=-1)
    #train model
    patch_architecture_dense = create_patch_model_dense((GLOBAL_X,GLOBAL_Y,3))
    plt.figure(figsize=(15, 8))
    previous_val_acc = 0
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print('Train denseNet')
    if not exists('patch_DenseNet.h5'):

        for i in range(1):
            history = patch_architecture_dense.fit(X_train, y_train, epochs=50, batch_size=64,callbacks=[early_stopping],
                                        validation_data=(x_val, y_val), verbose=1)

            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            plt.plot(history.history['accuracy'], label=f'Training Accuracy ({i}th iteration)')
            plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy ({i}th iteration)')
            #plt.plot(history.history['val_f1_score'], label=f'Validation F1 Score ({i}th iteration)')
            plt.title('DenseNet Baseline Model Accuracy History')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            plt.plot(history.history['loss'], label=f'Training Loss ({i} iteration)')
            plt.plot(history.history['val_loss'], label=f'Validation Loss ({i} iteration)')
            plt.title('Densenet Baseline Model Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            if history.history['val_accuracy'][-1] > previous_val_acc:
                save_patch_model = patch_architecture_dense
                print(f'({i} iteration) best model: {history.history["val_accuracy"][-1]}')

        plt.tight_layout()  # Adjust subplot spacing for better appearance
        plt.savefig('training_accuracy_loss_DenseNet.png', dpi=400)
        test_results = patch_architecture_dense.evaluate(X_test, y_test)
        print(f'Test Accuracy: {test_results[1]}')
        print(f'Test Loss: {test_results[0]}')
        with open('test_results_dense.txt', 'w') as file:
            file.write(f'Test Accuracy: {test_results[1]}\n')
            file.write(f'Test Loss: {test_results[0]}\n')
        save_model(save_patch_model,'patch_DenseNet121.h5')

    print('Train ResNet')
    previous_val_acc = 0
    patch_architecture_res = create_patch_model_res((GLOBAL_X,GLOBAL_Y,3))
    for i in range(1):
        history = patch_architecture_res.fit(X_train, y_train, epochs=50, batch_size=64,callbacks=[early_stopping],
                                        validation_data=(x_val, y_val), verbose=1)

        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.plot(history.history['accuracy'], label=f'Training Accuracy ({i}th iteration)')
        plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy ({i}th iteration)')
        #plt.plot(history.history['val_f1_score'], label=f'Validation F1 Score ({i}th iteration)')
        plt.title('ResNet152 Baseline Model Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.plot(history.history['loss'], label=f'Training Loss ({i} iteration)')
        plt.plot(history.history['val_loss'], label=f'Validation Loss ({i} iteration)')
        plt.title('ResNet152 Baseline Model Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if history.history['val_accuracy'][-1] > previous_val_acc:
            save_patch_model = patch_architecture_res
            print(f'({i} iteration) best model: {history.history["val_accuracy"][-1]}')

    plt.tight_layout()  # Adjust subplot spacing for better appearance
    plt.savefig('training_accuracy_loss_ResNet152.png', dpi=400)
    test_results = patch_architecture_res.evaluate(X_test, y_test)
    with open('test_results_res.txt', 'w') as file:
        file.write(f'Test Accuracy: {test_results[1]}\n')
        file.write(f'Test Loss: {test_results[0]}\n')
    save_model(save_patch_model,'patch_ResNet152.h5')

    # global_model = create_global_model(int(mode(row_list)),int(mode(col_list)))
    # history = global_model.fit(X_train, y_train, epochs=100, batch_size=2,
    #                                 validation_data=(X_test, y_test), verbose=1)

    # for sub_dir in all_dir:
    #     dicom_files = [os.path.join(sub_dir, filename) for filename in os.listdir(sub_dir) if filename.lower().endswith('.dicom')]
    #     for image in dicom_files:
    #         png_file = convert_dicom_to_png(image)
    #         # print(png_file)
    #         # print(type(png_file))
    #         # plt.imshow(png_file, cmap="gray")
    #         # plt.show()
    #         clahe_image = clahe(png_file)
    #         input()
if __name__ == "__main__":
    main()
