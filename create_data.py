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
# if argv[1] == 'small':
#     WINDOW_SIZE = 136
#     NUM_IMAGES = 5
#     append_name = 'small'
# elif argv[1] == 'medium':
#     WINDOW_SIZE = 250
#     NUM_IMAGES = 5
#     append_name = 'medium'
# elif argv[1] == 'large':
#     WINDOW_SIZE = 488
#     NUM_IMAGES = 2
#     append_name = 'large'

SMALL_CUTOFF_AREA = 0.0015709008023612142 #0.0010151872005197758 #what bebis wants, 0.0015709008023612142 #what the average is
MEDIUM_CUTOFF_AREA = 0.004060748802079103 #what bebis wants, what the ae0.006415104487709461
LARGE_CUTOFF_AREA = 0.02431239780756779
NUM_IMAGES = 5

#Based on average size - converted to pixels
SMALL_PATCH = 80 #bebis wants this to be 60x60 or 80x80 was 124
MEDIUM_PATCH = 200 #bebsis wants to be 200x200 was 250
LARGE_PATCH = 490

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

def get_small_patches(image,list_mass):
    for i, col in zip([1,1.5,2,2.5,3],['y','b','g','pink','snow']):
        patch_size = int(SMALL_PATCH * float(i))
        list_mass = [int(x) for x in list_mass]
        xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
        #Center
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        #randomly vary
        variation = 5 #pixel variability
        center_x = center_x + uniform(-variation, variation)
        center_y = center_y + uniform(-variation, variation)

        xmin_new, xmax_new = center_x - patch_size, center_x + patch_size
        ymin_new, ymax_new = center_y - patch_size, center_y + patch_size
        patch = {
                'box': (xmin_new, ymin_new, xmax_new, ymax_new)
        }
        # plot to check
        if i == 1:
            plt.imshow(image, cmap='gray')
            #the lesion
            bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', label='lesion',facecolor='none')
            plt.gca().add_patch(bounding_box)
        xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
        box = patch['box']
        #the sampled patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=col, label=f'mean lesion size * {i}', facecolor='none')
        plt.gca().add_patch(rect)
        plt.tight_layout()
        plt.title(f'small window')
        plt.legend()
    plt.savefig('ranges_boundary_small_lesion.png',dpi=400)
    plt.show()
    plt.close()

def random_patch_abnormal(image,list_mass):
    """
    Small: 222.0
    Medium (Median): 344
    Large: 522.0
    """
    window_size = WINDOW_SIZE
    list_mass = [int(x) for x in list_mass]
    xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
    #iterate until sampled adequately
    max_attempts = 25
    attempt = 0
    saved_patches = []
    while attempt < max_attempts:
        #first attempt
        # x_low = uniform(xmin, xmax - window_size)
        # x_high = x_low + window_size
        # y_low = uniform(ymin, ymax - window_size)
        # y_high = y_low + window_size
        # # print(f'uniform estimation: {int(y_low),int(y_high), int(x_low), int(x_high)}')
        # patch = image[int(y_low):int(y_high), int(x_low):int(x_high)]
        # patch = {
        #     'box': (x_low, y_low, x_high, y_high),
        #     'data': image[int(y_low):int(y_high), int(x_low):int(x_high)]
        # }

        #second try
        # if uniform(0, 1) < 0.5:  # 50% chance to sample along x-axis
        #     x_low = uniform(xmin, xmax - window_size)
        #     x_high = x_low + window_size
        #     y_low = uniform(ymin, ymax)
        #     y_high = y_low + window_size
        # else:  # 50% chance to sample along y-axis
        #     x_low = uniform(xmin, xmax)
        #     x_high = x_low + window_size
        #     y_low = uniform(ymin, ymax - window_size)
        #     y_high = y_low + window_size
        # x_high = min(xmax, x_high)
        # y_high = min(ymax, y_high) 
        x_low = uniform(xmin, xmax)
        if uniform(0, 1) < 0.5:
            x_high = x_low - window_size
        else:
            x_high = x_low + window_size
        y_low = uniform(ymin, ymax)
        if uniform(0, 1) < 0.5:
            y_high = y_low + window_size
        else:
            y_high = y_low - window_size

        patch = {
            'box': (max(xmin, x_low), max(ymin, y_low), x_high, y_high),
            'data': image[int(max(ymin, y_low)):int(y_high), int(max(xmin, x_low)):int(x_high)]
        }

        #plot to check
        # plt.imshow(image, cmap='gray')
        # xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
        # #the lesion
        # bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', label='lesion',facecolor='none')
        # plt.gca().add_patch(bounding_box)
        # box = patch['box']
        # #the sampled patch
        # rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='y', label='patch', facecolor='none')
        # plt.gca().add_patch(rect)
        # plt.tight_layout()
        # plt.title(f'{argv[1]} window')
        # plt.legend()
        # plt.show()

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
    image_num = 0
    for img in saved_patches:
        if img['data'].shape == (window_size, window_size, 3):
            save_lesion.append(img['data'])
            image_num += 1
    print(f'length of lesions {image_num}')
    return np.array(save_lesion)

def random_patch_normal(image,list_mass):
    window_size = int(WINDOW_SIZE)
    patches = view_as_windows(image, (window_size, window_size, 3), step=(window_size, window_size, 3))
    shape_patches = np.shape(patches)
    xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
    valid_patches = []
    for i in range(shape_patches[0]): #iter250ates over 
        y_pix_curr = i * window_size
        x_pix_curr = 0
        for j in range(shape_patches[1]): #iterates over x
            x_pix_curr = j * window_size
            current_patch = patches[i, j]
            if (
                x_pix_curr < xmin or x_pix_curr + window_size > xmax or
                y_pix_curr < ymin or y_pix_curr + window_size > ymax
            ) and np.mean(current_patch[0] > 40):
                # print('------')
                # print(np.mean(current_patch[0]))
                # print('------')
                valid_patches.append(current_patch[0])
                # plt.imshow(current_patch[0])
                # rect = plt.Rectangle((0, 0), WINDOW_SIZE, WINDOW_SIZE, linewidth=15, edgecolor='g', facecolor='none')
                # plt.gca().add_patch(rect)
            else:
                continue
            #     plt.imshow(current_patch[0])
            # plt.show()
    return np.array(valid_patches)

def is_similar(patch, existing_patches):
    for existing_patch in existing_patches:
        iou_score = calculate_iou(existing_patch['box'], patch['box'])
        if iou_score > 0.80:
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
    list_all_values = np.array(list_all_values)
    q1 = np.percentile(list_all_values, 12)
    q2 = np.percentile(list_all_values, 25)
    q3 = np.percentile(list_all_values, 50)

    # if q1 % 2 != 0:  # Check if it's odd
    #     q1 += 1
    # if q2 % 2 != 0:  # Check if it's odd
    #     q2 += 1
    # if q3 % 2 != 0:  # Check if it's odd
    #     q3 += 1
    data_q1 = list_all_values[np.where(list_all_values < q1)[0]]
    data_between_q1_q2 = list_all_values[np.where((list_all_values >= q1) & (list_all_values < q3))[0]]
    data_q3 = list_all_values[np.where(list_all_values >= q3)[0]]

    mean_q1, std_q1 = np.mean(data_q1), np.std(data_q1)
    mean_between_q1_q2, std_between_q1_q2 = np.mean(data_between_q1_q2), np.std(data_between_q1_q2)
    mean_q3, std_q3 = np.mean(data_q3), np.std(data_q3)
    
    categories = ['Small', 'Medium', 'Large']
    means = [mean_q1, mean_between_q1_q2, mean_q3]
    stds = [std_q1, std_between_q1_q2, std_q3]

    with open("mean_std_lesions.txt", "w") as file:
        file.write(f"Small: {mean_q1} +/- {std_q1}\n")
        file.write(f"Medium (Median): {mean_between_q1_q2} +/- {std_between_q1_q2}\n")
        file.write(f"Large: {mean_q3} +/- {std_q3}\n")
        file.write(f'Small min and max: {np.min(data_q1)}, {np.max(data_q1)}')
        file.write(f'Medium min and max: {np.min(data_between_q1_q2)}, {np.max(data_between_q1_q2)}')
        file.write(f'Large min and max: {np.min(data_q3)}, {np.max(data_q3)}')
    plt.figure()
    plt.bar(categories, means, yerr=stds, capsize=10, color='blue', alpha=0.7)
    plt.xlabel('Percentile Ranges')
    plt.ylabel('Mean +/- Standard Deviation')
    plt.title('Mean and Standard Deviation for Different Percentile Ranges')
    plt.savefig('mean_std_percentiles.png',dpi=400)
    plt.close()
    with open("quartiles.txt", "w") as file:
        file.write(f"Small: {q1}\n")
        file.write(f"(Median): {q2}\n")
        file.write(f"Large: {q3}\n")
def main():
    datagen = ImageDataGenerator(
        rotation_range=45,  # 45 degree range for rotations - randomly
        horizontal_flip=True,  # Random horiz flips
        vertical_flip=True  # Random vert flips
    )
    # if not exists(f'X_train_{append_name}.npy'):
    glob_dir = '/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
    df = read_df()
    df.dropna(inplace=True)
    list_all_values = [] 
    selected_arrays = []
    selected_arrays_non_lesion = []
    all_images = 0
    # combined_array = np.empty((0, WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    save_small, save_medium, save_large = [], [], []
    for iteration, (index, row) in enumerate(df.iterrows()):
        image_type = row['image_id'] + '.dicom'
        dicom_path = os.path.join(glob_dir,row['study_id'],image_type)
        if os.path.exists(dicom_path):
            png_file = convert_dicom_to_png(dicom_path)
            #abnormal images
            annotations = row['finding_categories']
            if row['breast_birads'] > 1 and "Mass" in annotations:
                clahe_image = clahe(png_file)
                # list_all_values = get_lesion_size([row['xmin'],row['xmax'],row['ymin'],row['ymax']],
                #                                     list_all_values,clahe_image.shape)
                
                # all_images += 1
#             # for ind_image in clahe_image:
                # lesion_images = random_patch_abnormal(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']])
                #small lesions
                if get_area(clahe_image.shape,[row['xmin'],row['xmax'],row['ymin'],row['ymax']]) <= SMALL_CUTOFF_AREA:
                    # get_small_patches(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']])
                    save_small.append(clahe_image)
                if (get_area(clahe_image.shape,[row['xmin'],row['xmax'],row['ymin'],row['ymax']]) > SMALL_CUTOFF_AREA and 
                    get_area(clahe_image.shape,[row['xmin'],row['xmax'],row['ymin'],row['ymax']]) < LARGE_CUTOFF_AREA
                    ):
                    save_medium.append(clahe_image)
                if get_area(clahe_image.shape,[row['xmin'],row['xmax'],row['ymin'],row['ymax']]) >= LARGE_CUTOFF_AREA:
                    save_large.append(clahe_image)
        print(f'percent finished: {(iteration / len(df))*100}')

    #save data to .npy
    save_np_small = np.array(save_small)
    save_np_medium = np.array(save_medium)
    save_np_large = np.array(save_large)
    np.save(f'whole_small_images.npy', save_np_small)
    np.save(f'whole_medium_images.npy', save_np_medium)
    np.save(f'whole_large_images.npy', save_np_large)
    print(save_np_small.shape)
    print(save_np_medium.shape)
    print(save_np_large.shape)



                        # lesion_images = random_patch_abnormal(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']])
                        # non_lesion_images = random_patch_normal(clahe_image,[row['xmin'],row['xmax'],row['ymin'],row['ymax']])
                    # if lesion_images.shape[0] > 0:
                    #     print()



    #                     #augment_lesions
    #                     save_augmented = []
    #                     for i in range(lesion_images.shape[0]):
    #                         for m in range(NUM_IMAGES):  # Generate 5 augmented images
    #                             augmented_img = datagen.random_transform(lesion_images[i, :, :, :])
    #                             save_augmented.append(augmented_img)
    #                     save_np_augmented = np.array(save_augmented)

    #                     #create equal number of abnormal and normal
    #                     if lesion_images.shape[0] < non_lesion_images.shape[0]:
    #                         ran_select = np.random.choice(non_lesion_images.shape[0], size=lesion_images.shape[0], replace=False)
    #                         non_lesion_images = non_lesion_images[ran_select]

    #                     #augment_non_lesions
    #                     save_augmented_non = []
    #                     for i in range(non_lesion_images.shape[0]):
    #                         for m in range(NUM_IMAGES):  # Generate 5 augmented images
    #                             augmented_img = datagen.random_transform(non_lesion_images[i, :, :, :])
    #                             save_augmented_non.append(augmented_img)
    #                     save_np_augmented_non = np.array(save_augmented_non)

    #                     #Save data to lists
    #                     selected_arrays.append(save_np_augmented)
    #                     selected_arrays_non_lesion.append(save_np_augmented_non)
    #             print(f'percent finished: {(iteration / len(df))*100}')
    #             # if non_lesion_images.shape[0] > 0: 
    #             #     selected_arrays_non_lesion.append(non_lesion_images)
    # combined_array_lesion = np.concatenate(selected_arrays, axis=0)
    # combined_array_non_lesion = np.concatenate(selected_arrays_non_lesion, axis=0)
    # if combined_array_lesion.shape[0] < combined_array_non_lesion.shape[0]:
    #     selected_indices = np.random.choice(combined_array_non_lesion.shape[0], size=combined_array_lesion.shape[0], replace=False)
    #     combined_array_non_lesion_updated = combined_array_non_lesion[selected_indices]
    # else:
    #     combined_array_non_lesion_updated = combined_array_non_lesion
    # print(np.shape(combined_array_lesion))
    # print(np.shape(combined_array_non_lesion_updated))
    # non_lesion_labels = np.zeros(combined_array_non_lesion_updated.shape[0])
    # lesion_labels = np.ones(combined_array_lesion.shape[0])
    # labels = to_categorical(np.concatenate((non_lesion_labels, lesion_labels)), 
    #                                         num_classes=2, dtype="int")
    # features = np.concatenate((combined_array_non_lesion_updated,combined_array_lesion), axis=0)

    # X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
    # x_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # # Save the data
    # np.save(f'X_train_{append_name}.npy', X_train)
    # np.save(f'X_validation_{append_name}.npy', x_val)
    # np.save(f'X_test_{append_name}.npy', X_test)
    # np.save(f'y_train_{append_name}.npy', y_train)
    # np.save(f'y_validation_{append_name}.npy', y_val)
    # np.save(f'y_test_{append_name}.npy', y_test)




    # #plot hist
    # cutoff_small = np.percentile(list_all_values, 25)
    # cutoff_medium = np.percentile(list_all_values, 50)
    # cutoff_large = np.percentile(list_all_values, 75)
    # create_small_medium_large(list_all_values)
    # plt.figure()
    # hist, bins, _ = plt.hist(np.array(list_all_values).flatten(), bins='auto', color='tab:blue', density=True, alpha=1, label='Data')
    # # plt.fill_betweenx(y=[0, max_density], x1=0, x2=cutoff_small, color='tab:red', alpha=0.5, label='Small')
    # # plt.fill_betweenx(y=[0, max_density], x1=cutoff_small, x2=cutoff_large, color='tab:purple', alpha=0.5, label='Medium')
    # # plt.fill_betweenx(y=[0, max_density], x1=cutoff_large, x2=0.15, color='tab:green', alpha=0.5, label='Large')
    # plt.axvline(cutoff_small, color='yellow', linestyle='--', linewidth=2,label='Lower Cutoff')
    # plt.axvline(cutoff_large, color='tab:red', linestyle='--', linewidth=2,label='Higher Cutoff')
    # plt.xlabel('Proportion')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.xlim([0,0.15])
    # plt.savefig('hist_sizes_lesion_area.png',dpi=400)
    # plt.close()
    # with open("total_number_of_images.txt", "w") as file:
    #     file.write(f"total images: {all_images}\n")

if __name__ == "__main__":
    main()
                        # if list_all_values[-1] > 0.05:
                        #     plt.imshow(clahe_image, cmap='gray')
                        #     xmin, xmax, ymin, ymax = [row['xmin'],row['xmax'],row['ymin'],row['ymax']]
                        #     x_dist_lesion = xmax - xmin
                        #     y_dist_lesion = ymax - ymin
                        #     #the lesion
                        #     bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', label='lesion',facecolor='none')
                        #     plt.gca().add_patch(bounding_box)
                        #     # plt.tight_layout()
                        #     plt.title(f'Large Lesion: {int(y_dist_lesion)} x {int(x_dist_lesion)} pixels')
                        #     plt.legend()
                        #     plt.savefig('large_lesion.png',dpi=400)
                        #     plt.show()