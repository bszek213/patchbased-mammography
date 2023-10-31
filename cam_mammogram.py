#CAM MAMMOGRAM
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
import tensorflow as tf
from keras.models import Sequential
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import cv2
from pandas import read_csv
# from sklearn.feature_extraction.image import PatchExtractor
from patchify import patchify


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

def read_df(path="/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"):
    df = read_csv(os.path.join(path,"breast-level_annotations.csv"))
    df['breast_birads'] = df['breast_birads'].str.replace('BI-RADS ', '')
    df['breast_birads'] = df['breast_birads'].astype(int)
    return df[['study_id', 'image_id','view_position','breast_birads']]

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
    return clahe.apply(A_cv2)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('CLAHE Enhanced Image', clahe_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def resNet152_baseline():
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9,beta_2=0.999)

    base_model = ResNet152V2(include_top=False, input_shape=(GLOBAL_X, GLOBAL_Y, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics="accuracy")
    return model

def patch(image):
    patches=patchify(image,(GLOBAL_X,GLOBAL_Y),step=GLOBAL_X)
    return patches
    # shape_patches = np.shape(patches)
    # plt.figure(figsize=(6, 6))
    # ix = 1
    # for i in range(shape_patches[0]):
    #     for j in range(shape_patches[1]):
    #         # specify subplot and turn of axis
    #         ax = plt.subplot(shape_patches[0], shape_patches[1], ix)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         # plot 
    #         plt.imshow(patches[i, j, :, :], cmap='gray')
    #         ix += 1
    # plt.savefig('patch_example.png',dpi=450)
    # plt.show()
    # pe = PatchExtractor(patch_size=(GLOBAL_X, GLOBAL_Y))
    # patches = pe.transform(image)
    # print(patches)
    # print(np.shape(patches))
    # input()
def main():
    # all_dir = load_images()
    glob_dir = '/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
    df = read_df()
    dict_save_benign = {}
    dict_image_benign= {}
    dict_save_malig = {}
    dict_image_malig = {}
    for iteration, (index, row) in enumerate(df.iterrows()):
        image_type = row['image_id'] + '.dicom'
        dicom_path = os.path.join(glob_dir,row['study_id'],image_type)
        if os.path.exists(dicom_path):
            png_file = convert_dicom_to_png(dicom_path)
            clahe_image = clahe(png_file)
            #BENIGN
            if row['breast_birads'] > 1 and row['breast_birads'] < 4:
                dict_save_benign[index] = [row['view_position'],row['breast_birads']]   
                dict_image_benign[index] = patch(clahe_image)
            #MALIGNANT
            elif row['breast_birads'] > 3:
                dict_save_malig[index] = [row['view_position'],row['breast_birads']]
                dict_image_malig[index] = patch(clahe_image)
                print(dict_save_malig)

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
