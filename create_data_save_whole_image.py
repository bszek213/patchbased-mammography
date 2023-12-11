import numpy as np
import os
from pandas import read_csv
import pydicom
import cv2

SMALL_CUTOFF_AREA = 0.0015709008023612142 #0.0010151872005197758 #what bebis wants, 0.0015709008023612142 #what the average is
MEDIUM_CUTOFF_AREA = 0.004060748802079103 #what bebis wants, what the ae0.006415104487709461
LARGE_CUTOFF_AREA = 0.02431239780756779
NUM_IMAGES = 5

def read_df(path="/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"):
    df = read_csv(os.path.join(path,"finding_annotations.csv"))
    df['breast_birads'] = df['breast_birads'].str.replace('BI-RADS ', '')
    df['breast_birads'] = df['breast_birads'].astype(int)
    return df[['study_id','finding_categories','image_id','view_position','breast_birads','xmin','xmax','ymin','ymax']]

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
    # Add white noise
    # noise = np.random.normal(0, 10, A_cv2.shape).astype(np.uint8)
    # noisy_image = cv2.add(A_cv2, noise)
    # noisy_image = np.clip(noisy_image, 0, 255)
    # #Apply Gblur
    # blurred_image = cv2.GaussianBlur(A_cv2, (5, 5), 0)
    #Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(tile_s0,tile_s1))
    clahe_image = clahe.apply(A_cv2)
    # clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(tile_s0,tile_s1))
    # clahe_blur = clahe.apply(blurred_image)
    #Convert
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)   
    # clahe_noise = cv2.cvtColor(clahe_noise, cv2.COLOR_GRAY2RGB)  
    # plt.figure()
    # plt.title('Original')
    # plt.imshow(A_cv2, cmap='gray')
    # plt.figure()
    # plt.title(f'Noise + CLAHE')
    # plt.imshow(clahe_noise,cmap='gray')
    # plt.figure()
    # plt.title('Blur + CLAHE')
    # plt.imshow(clahe_blur)
    # plt.show()
    return clahe_image

def get_lesion_size(list_mass,list_all_values,image_shape):
    list_mass = [int(x) for x in list_mass]
    x_dist_lesion = list_mass[1] - list_mass[0]
    y_dist_lesion = list_mass[3] - list_mass[2]
    #image area calculation 
    image_area = (x_dist_lesion * y_dist_lesion) / (image_shape[0] * image_shape[1])
    list_all_values.append(image_area)
    # #Euclidean distance
    # list_all_values.append(int(sqrt((list_mass[1] - list_mass[0])**2 + (list_mass[3] - list_mass[2])**2)))
    return list_all_values

def get_area(image_shape,list_mass):
    list_mass = [int(x) for x in list_mass]
    x_dist_lesion = list_mass[1] - list_mass[0]
    y_dist_lesion = list_mass[3] - list_mass[2]
    image_area = (x_dist_lesion * y_dist_lesion) / (image_shape[0] * image_shape[1])
    return image_area

glob_dir = '/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
df = read_df()
df.dropna(inplace=True)
save_small, save_medium, save_large = [], [], []
save_small_annotation, save_medium_annotation, save_large_annotation = [], [], []
for iteration, (index, row) in enumerate(df.iterrows()):
    image_type = row['image_id'] + '.dicom'
    dicom_path = os.path.join(glob_dir,row['study_id'],image_type)
    if os.path.exists(dicom_path):
        png_file = convert_dicom_to_png(dicom_path)
        #abnormal images
        annotations = row['finding_categories']
        if row['breast_birads'] > 1 and "Mass" in annotations:
            clahe_image = clahe(png_file)

            if get_area(clahe_image.shape,[row['xmin'],row['xmax'],row['ymin'],row['ymax']]) <= SMALL_CUTOFF_AREA:
                # get_small_patches(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']])
                save_small.append(clahe_image)
                save_small_annotation.append([row['xmin'],row['xmax'],row['ymin'],row['ymax']])
            if (get_area(clahe_image.shape,[row['xmin'],row['xmax'],row['ymin'],row['ymax']]) > SMALL_CUTOFF_AREA and 
                get_area(clahe_image.shape,[row['xmin'],row['xmax'],row['ymin'],row['ymax']]) < LARGE_CUTOFF_AREA
                ):
                save_medium.append(clahe_image)
                save_medium_annotation.append([row['xmin'],row['xmax'],row['ymin'],row['ymax']])
            if get_area(clahe_image.shape,[row['xmin'],row['xmax'],row['ymin'],row['ymax']]) >= LARGE_CUTOFF_AREA:
                save_large.append(clahe_image)
                save_large_annotation.append([row['xmin'],row['xmax'],row['ymin'],row['ymax']])
    print(f'percent finished: {(iteration / len(df))*100}')

save_np_small = np.array(save_small)
save_np_medium = np.array(save_medium)
save_np_large = np.array(save_large)

print(save_small_annotation)
save_small_annotation = np.array(save_small_annotation)
save_medium_annotation = np.array(save_medium_annotation)
save_large_annotation = np.array(save_large_annotation)
print(save_small_annotation)

if not os.path.exists('data'):
    os.mkdir('data')

np.save(os.path.join('data', 'whole_images_small.npy'), save_np_small)
np.save(os.path.join('data', 'whole_images_medium.npy'), save_np_medium)
np.save(os.path.join('data', 'whole_images_large.npy'), save_np_large)

np.save(os.path.join('data', 'whole_images_small_annotations.npy'), save_small_annotation)
np.save(os.path.join('data', 'whole_images_medium_annotations.npy'), save_medium_annotation)
np.save(os.path.join('data', 'whole_images_large_annotations.npy'), save_large_annotation)