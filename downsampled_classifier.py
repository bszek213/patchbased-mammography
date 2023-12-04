from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
import tensorflow as tf
from keras.models import Sequential
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import cv2
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
from PIL import Image
from tqdm import tqdm

def read_df():
    df = read_csv("normal_training_data_ann.csv")
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

def downsample(image):
    original_image = Image.fromarray(image, mode='RGB')
    downsampled_image = original_image.resize((500, 500), Image.Resampling.LANCZOS)
    return np.array(downsampled_image)

def create_patch_model():
    # rgb_input = Concatenate()([input_layer, input_layer, input_layer])
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(GLOBAL_X, GLOBAL_Y, 3))

    model = Sequential()
    model.add(base_model)
    model.add(Conv2D(1024, (15, 15), activation="relu"))
    model.add(Conv2D(1024, (1, 1), activation="relu"))
    model.add(Conv2D(2, (1, 1), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    return model

def main():
    downsampled_images = []
    raw_label = []

    if not os.path.exists('X_train.npy'):
        glob_dir = 'physionet.org/files/vindr-mammo/1.0.0/images/'
        df = read_df().sample(frac=1)
        
        for iteration, (index, row) in tqdm(enumerate(df.iterrows())):
            image_type = row['image_id'] + '.dicom'
            dicom_path = os.path.join(glob_dir,row['study_id'],image_type)
            if os.path.exists(dicom_path):
                png_file = convert_dicom_to_png(dicom_path)
                if png_file is not None:
                    clahe_image = clahe(png_file)
                    downsampled_images.append(downsample(clahe_image))
                    if row['breast_birads'] <= 2:
                        raw_label.append(0)
                    else:
                        raw_label.append(1)

        labels = tf.keras.utils.to_categorical(raw_label, num_classes=2, dtype="int")
        features = np.array(downsampled_images, dtype=np.uint8)

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
    history = patch_architecture.fit(X_train, np.expand_dims(np.expand_dims(y_train, 1), 1), epochs=20, batch_size=64,
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