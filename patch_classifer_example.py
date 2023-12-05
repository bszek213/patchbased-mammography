from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from keras.applications.vgg16 import VGG16
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
# from patchify import patchify
# from np_utils import to_categorical
from sklearn.model_selection import train_test_split
from statistics import mode
# import pickle
from skimage.util import view_as_windows
from tensorflow.keras.models import save_model, load_model

GLOBAL_X = 250
GLOBAL_Y = 250
PATCH_SIZE = (GLOBAL_X, GLOBAL_Y)

def read_df():
    df = read_csv("finding_annotations.csv")
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

def patch(image,list_mass,GLOBAL_LESION_COUNT):

    step_x = max(GLOBAL_X, GLOBAL_Y)
    step_y = max(GLOBAL_X, GLOBAL_Y)

    patches = view_as_windows(image, (GLOBAL_Y, GLOBAL_X, 3), step=(step_y, step_x, 3))

    list_mass = [int(x) for x in list_mass]
    xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
    shape_patches = np.shape(patches)

    all_images = {}
    x_pix_curr, y_pix_curr = 0,0

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

            if np.all(sample_patch < 40):
                continue

            elif (
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
                    (xmin <= x_pix_curr <= xmax)) or
                    ((xmin <= patch_xmin <= xmax) and
                     (xmin <= patch_xmax <= xmax) and
                     (ymin <= patch_ymin <= ymax) and
                     (ymin <= patch_ymax <= ymax))
                  ): #patch_xmin >= xmin and patch_xmax <= xmax and patch_ymin >= ymin and patch_ymax <= ymax
                continue
                
            else:
                all_images[i,j] = [sample_patch,1]
    
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
    
    
    window_size = 75
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
            
    lesion_image_array = np.array(combined_array_lesion)
    GLOBAL_LESION_COUNT += len(combined_array_lesion)
    
    feature_list, label_list = [], []
    for key , (matrix, integer) in all_images.items():
        feature_list.append(matrix)
        label_list.append(int(integer))

    if len(combined_array_lesion) > 0:
        label_lesions_list = [0] * len(combined_array_lesion)
        return feature_list, lesion_image_array, label_list, label_lesions_list, GLOBAL_LESION_COUNT
    else:
        return feature_list, None, label_list, None, GLOBAL_LESION_COUNT
    
def display_image(image):
    from PIL import Image
    # Downsample the image to 300x300
    # Convert the NumPy array to a Pillow image
    original_image = Image.fromarray(image, mode='RGB')  # 'L' mode represents grayscale

    # Downsample the image to 300x300
    downsampled_image = original_image.resize((300, 300), Image.ANTIALIAS)

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image in the first subplot
    ax1.imshow(image)
    ax1.set_title('Original Image')

    # Display the downsampled image in the second subplot
    ax2.imshow(np.array(downsampled_image))
    ax2.set_title('Downsampled Image')

    # Hide axis labels
    ax1.axis('off')
    ax2.axis('off')

    # Show the Matplotlib figure
    plt.savefig('downsampleVsOG.png',dpi=400)
    plt.close()

def create_patch_model():
    # rgb_input = Concatenate()([input_layer, input_layer, input_layer])
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(GLOBAL_X, GLOBAL_Y, 3))

    model = Sequential()
    model.add(base_model)
    model.add(Conv2D(1024, (7, 7), activation="relu"))
    model.add(Conv2D(1024, (1, 1), activation="relu"))
    model.add(Conv2D(2, (1, 1), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    return model

def main():
    dict_save_benign = {}
    dict_save_malig = {}
    list_label_images = []
    combined_array = np.empty((0, 250, 250, 3), dtype=np.uint8)
    combined_array_lesion = np.empty((0, 250, 250, 3), dtype=np.uint8)
    list_labels_total_np = np.array([])
    if not os.path.exists('X_train.npy'):
        glob_dir = 'physionet.org/files/vindr-mammo/1.0.0/images/'
        df = read_df()
        df.dropna(inplace=True)
        GLOBAL_LESION_COUNT = 0
        for iteration, (index, row) in enumerate(df.iterrows()):
            image_type = row['image_id'] + '.dicom'
            dicom_path = os.path.join(glob_dir,row['study_id'],image_type)
            if os.path.exists(dicom_path):
                png_file = convert_dicom_to_png(dicom_path)
                if png_file is not None:
                    #BENIGNlist_images_lesion
                    if row['breast_birads'] > 1 and row['breast_birads'] < 4:
                        clahe_image = clahe(png_file)
                        dict_save_benign[index] = [row['view_position'],row['breast_birads']]   
                        # dict_image_benign[index] = patch(clahe_image)
                        feature_list, subimage_list, label_list, label_lesions_list, GLOBAL_LESION_COUNT  = patch(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']],GLOBAL_LESION_COUNT)
                        if iteration % 3 == 0:# and iteration != 0:
                            try:
                                combined_array = np.vstack((combined_array, feature_list))
                                list_labels_total_np = np.concatenate((list_labels_total_np, np.array(label_list)))
                            except:
                                    print('dimension issues. Just do not use that data')
                            
                        if label_lesions_list != None:
                            list_label_images.append(label_lesions_list)
                            combined_array_lesion = np.vstack((combined_array_lesion, subimage_list))

                    #MALIGNANT
                    elif row['breast_birads'] > 3:
                        clahe_image = clahe(png_file)
                        # display_image(clahe_image)
                        dict_save_malig[index] = [row['view_position'],row['breast_birads']]
                        feature_list, subimage_list, label_list, label_lesions_list, GLOBAL_LESION_COUNT = patch(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']],GLOBAL_LESION_COUNT)
                        if iteration % 3 == 0:# and iteration != 0:
                            try:
                                combined_array = np.vstack((combined_array, feature_list))
                                list_labels_total_np = np.concatenate((list_labels_total_np, np.array(label_list)))
                            except:
                                print('dimension issues. Just do not use that data')
                            
                        if label_lesions_list != None:
                            list_label_images.append(label_lesions_list)
                            combined_array_lesion = np.vstack((combined_array_lesion, subimage_list))
                        
        list_ones = [item for sublist in list_label_images for item in sublist]

        array_lesion = np.array(list_ones, dtype="int")

        labels = tf.keras.utils.to_categorical(np.concatenate((list_labels_total_np, array_lesion)), num_classes=2, dtype="int")
        
        features = np.concatenate((combined_array,combined_array_lesion), axis=0)

        X_train, X_val, y_train, Y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
        np.save('X_train.npy', X_train)
        np.save('X_val.npy', X_val)
        np.save('y_train.npy', y_train)
        np.save('Y_val.npy', Y_val)
    else:
        X_train = np.load('X_train.npy')
        X_val = np.load('X_val.npy')
        y_train = np.load('y_train.npy')
        Y_val = np.load('Y_val.npy')
    print(f'x_train size {np.shape(X_train)}')
    print(f'X_val size {np.shape(X_val)}')
    print(f'y_train size {np.shape(y_train)}')
    print(f'Y_val size {np.shape(Y_val)}')
    
    label_counts = np.sum(y_train, axis=0)
    total_labels = len(y_train)
    label_percentages = (label_counts / total_labels) * 100
    for label, percentage in enumerate(label_percentages):
        label_str = " ".join([str(int(val)) for val in y_train[label]])
        print(f"Label [{label_str}]: {percentage:.2f}%")
    
    patch_architecture = create_patch_model()
    plt.figure(figsize=(15, 8))
    history = patch_architecture.fit(X_train, np.expand_dims(np.expand_dims(y_train, 1), 1), epochs=2, batch_size=64,
                                        validation_data=(X_val, np.expand_dims(np.expand_dims(Y_val, 1), 1)), verbose=1)

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(history.history['accuracy'], label=f'Training Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy')
    plt.title('DenseNet Baseline Model Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(history.history['loss'], label=f'Training Loss')
    plt.plot(history.history['val_loss'], label=f'Validation Loss')
    plt.title('Densenet Baseline Model Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()  # Adjust subplot spacing for better appearance
    plt.savefig('training_accuracy_loss_DenseNet.png', dpi=400)
    save_model(patch_architecture,'patch_DenseNet121.h5')

if __name__ == "__main__":
    main()
