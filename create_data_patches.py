# import matplotlib.pyplot as plt
# import pydicom
import numpy as np
import os
# import cv2
# from pandas import read_csv
from sys import argv
# from os.path import exists
# from math import sqrt
from random import uniform, choice#, sample
from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows
# from seaborn import kdeplot
# from matplotlib import patches
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SMALL_CUTOFF_AREA = 0.0019489851662238661 #0.0010151872005197758 #what bebis wants, 0.0015709008023612142 #what the average is
MEDIUM_CUTOFF_AREA = 0.006119264128657757 #what bebis wants, what the ae0.006415104487709461
LARGE_CUTOFF_AREA = 0.02067312075767155
NUM_IMAGES = 5

#Based on average size - converted to pixels
SMALL_PATCH = 138 #bebis wants this to be 60x60 or 80x80 was 124
MEDIUM_PATCH = 244 #bebsis wants to be 200x200 was 250
LARGE_PATCH = 450

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

def create_patches_abnormal(image,multiplication,type,annotations):
    if type == 'small':
        patch_size = int(SMALL_PATCH * float(multiplication))
    elif type == 'medium':
        patch_size = int(MEDIUM_PATCH * float(multiplication))
    elif type == 'large':
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

    #create patch
    patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)

    xmax_new = min(np.ceil(xmax_new), image.shape[1])
    ymax_new = min(np.ceil(ymax_new), image.shape[0])
    xmin_new = max(np.floor(xmin_new), 0)
    ymin_new = max(np.floor(ymin_new), 0)

    patch_height = int(np.ceil(ymax_new - ymin_new))
    patch_width = int(np.ceil(xmax_new - xmin_new))
    patch_height = min(patch_height, patch_size)
    patch_width = min(patch_width, patch_size)
    # if int(xmin_new) - int(xmax_new) != patch_size:
    #     xmax_new+=1
    # if int(ymax_new) - int(ymin_new) != patch_size:
    #     ymax_new+=1

    patch[:patch_height, :patch_width, :] = image[
        int(np.floor(ymin_new)):int(np.floor(ymin_new)) + patch_height,
        int(np.floor(xmin_new)):int(np.floor(xmin_new)) + patch_width,
        :
    ]
    # patch[0:int(ymax_new - ymin_new),0:int(xmax_new - xmin_new),] = image[int(ymin_new):int(ymax_new),int(xmin_new):int(xmax_new),:]
    
    return patch

def is_more_than_50_percent_black(image):
    grayscale_image = np.mean(image, axis=-1)
    black_pixel_count = np.sum(grayscale_image <= 40)
    total_pixels = np.prod(image.shape[:-1])  # Assumes the image shape is (height, width, channels)
    percentage_black = (black_pixel_count / total_pixels) * 100
    return percentage_black > 75

def random_patch_normal(image,multiplication,type,list_mass):
    if type == 'small':
        window_size = int(SMALL_PATCH * float(multiplication))
    elif type == 'medium':
        window_size = int(MEDIUM_PATCH * float(multiplication))
    elif type == 'large':
        window_size = int(LARGE_PATCH * float(multiplication))
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
            )  and (not is_more_than_50_percent_black(current_patch[0])): #and np.mean(current_patch[0] > 40)
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
    #randomly select valid patch
    random_index = choice(range(len(valid_patches)))
    selected_image = valid_patches[random_index]
    # print(selected_image.shape)
    # plt.imshow(selected_image, cmap='gray')
    # plt.tight_layout()
    # plt.title(f'small window')
    # plt.show()
    return selected_image
    
    # patch = {
    #     'box': (xmin_new, ymin_new, xmax_new, ymax_new)
    # }
    # # Fill outside regions with zeros
    # patched_image = np.zeros_like(image)
    # patched_image[int(ymin_new):int(ymax_new), int(xmin_new):int(xmax_new)] = image[int(ymin_new):int(ymax_new), int(xmin_new):int(xmax_new)]
    # box = patch['box']
    # plt.imshow(patched_image, cmap='gray')
    # # The lesion
    # bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
    #                                     label='lesion', facecolor='none')
    # plt.gca().add_patch(bounding_box)
    # # The sampled patch
    # rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='yellow',
    #                             facecolor='none')
    # plt.gca().add_patch(rect)
    # plt.tight_layout()
    # plt.title(f'small window')
    # plt.legend()
    # plt.show()

def data_augment(patch,datagen):
    return datagen.random_transform(patch)

def main():
    datagen = ImageDataGenerator(
        rotation_range=180,  # 90 degree range for rotations - randomly
        horizontal_flip=True,  # Random horiz flips
        vertical_flip=True,  # Random vert flips
        # width_shift_range=0.1,
        # height_shift_range=0.1
    )
    #argv
    type_patch = argv[1]#'small'
    multiply = argv[2] #1
    train_data, validation_data, test_data, train_data_an, validation_data_an, test_data_an = load_images(type_patch)

    #How many images do you want?
    im_aug_train = int(2000 / len(train_data)) #2000 abnormal patches
    patch_aug_train, patch_aug_valid, patch_aug_test = [], [], []
    patch_aug_train_label, patch_aug_valid_label, patch_aug_test_label = [], [], []

    #train
    print('create training...')
    for image, image_an in zip(train_data,train_data_an):
        patch = create_patches_abnormal(image,multiply,type_patch,image_an)
        patch_normal = random_patch_normal(image,multiply,type_patch,image_an)

        for i in range(im_aug_train):
            patch_aug_train.append(data_augment(patch,datagen))
            patch_aug_train_label.append(1)
            patch_aug_train.append(data_augment(patch_normal,datagen))
            patch_aug_train_label.append(0)

    #validate and get correct number of images
    print('create validation...')
    final_train_images = np.array(patch_aug_train)
    im_aug_valid = int((final_train_images.shape[0]*.15) / len(validation_data))
    for image, image_an in zip(validation_data,validation_data_an):
        patch = create_patches_abnormal(image,multiply,type_patch,image_an)
        patch_normal = random_patch_normal(image,multiply,type_patch,image_an)

        for i in range(im_aug_valid):
            patch_aug_valid.append(data_augment(patch,datagen))
            patch_aug_valid_label.append(1)
            patch_aug_valid.append(data_augment(patch_normal,datagen))
            patch_aug_valid_label.append(0)
    
    #test
    print('create testing...')
    im_aug_test = int((final_train_images.shape[0]*.15) / len(test_data))
    for image, image_an in zip(test_data,test_data_an):
        patch = create_patches_abnormal(image,multiply,type_patch,image_an)
        patch_normal = random_patch_normal(image,multiply,type_patch,image_an)

        for i in range(im_aug_test):
            patch_aug_test.append(data_augment(patch,datagen))
            patch_aug_test_label.append(1)
            patch_aug_test.append(data_augment(patch_normal,datagen))
            patch_aug_test_label.append(0)

    final_valid_images = np.array(patch_aug_valid)
    final_test_images = np.array(patch_aug_test)
    print(final_train_images.shape)
    print(final_valid_images.shape)
    print(final_test_images.shape)

    labels_train = to_categorical(np.array(patch_aug_train_label), 
                                            num_classes=2, dtype="int")
    labels_valid = to_categorical(np.array(patch_aug_valid_label), 
                                            num_classes=2, dtype="int")
    labels_test = to_categorical(np.array(patch_aug_test_label), 
                                            num_classes=2, dtype="int")
    if not os.path.exists('data_patch'):
        os.mkdir('data_patch')
    #save features
    multiply = str(multiply).replace(".", "-")
    np.save(os.path.join('data_patch', f'{type_patch}_patches_{str(multiply)}_train.npy'), final_train_images)
    np.save(os.path.join('data_patch', f'{type_patch}_patches_{str(multiply)}_valid.npy'), final_valid_images)
    np.save(os.path.join('data_patch', f'{type_patch}_patches_{str(multiply)}_test.npy'), final_test_images)

    #save_labels
    np.save(os.path.join('data_patch', f'{type_patch}_patches_train_{str(multiply)}_label.npy'), labels_train)
    np.save(os.path.join('data_patch', f'{type_patch}_patches_valid_{str(multiply)}_label.npy'), labels_valid)
    np.save(os.path.join('data_patch', f'{type_patch}_patches_test_{str(multiply)}_label.npy'), labels_test)
    # final_valid_images = np.array(patch_aug_valid)
    # print(final_valid_images.shape)

        #PLOT ALL AUGMENT AND ORIGINAL IMAGE
        # for i in range(final_train_images.shape[0]):
        #     plt.figure()
        #     plt.imshow(final_train_images[i,:,:,:], cmap='gray')
        #     plt.tight_layout()
        #     plt.title(f'augment')
        # plt.figure()
        # plt.imshow(patch_train, cmap='gray')
        # plt.title('original')
        # plt.show()

if __name__ == '__main__':
    main()