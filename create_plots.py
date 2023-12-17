import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from random import uniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SMALL_PATCH = 138 #bebis wants this to be 60x60 or 80x80 was 124
MEDIUM_PATCH = 244 #bebsis wants to be 200x200 was 250
LARGE_PATCH = 450

def show_small_medium_large_patches():
    if not os.path.exists('data_figures'):
        os.mkdir('data_figures')
    whole_small = np.load(os.path.join('data', 'whole_images_small.npy'),allow_pickle=True)
    whole_med = np.load(os.path.join('data', 'whole_images_medium.npy'),allow_pickle=True)
    whole_lage = np.load(os.path.join('data', 'whole_images_large.npy'),allow_pickle=True)

    whole_small_an = np.load(os.path.join('data', 'whole_images_small_annotations.npy'),allow_pickle=True)
    whole_med_an = np.load(os.path.join('data', 'whole_images_medium_annotations.npy'),allow_pickle=True)
    whole_large_an = np.load(os.path.join('data', 'whole_images_large_annotations.npy'),allow_pickle=True)

    small_patch_sizes(whole_small[0],whole_small_an[0])
    medium_patch_sizes(whole_med[0],whole_med_an[0])

    #small
    plt.figure()
    list_mass = whole_small_an[0]
    xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
    plt.imshow(whole_small[0], cmap='gray')
    bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                        label='lesion', facecolor='none')
    plt.gca().add_patch(bounding_box)
    plt.title('Small Lesion')
    plt.savefig(os.path.join('data_figures','whole_im_les_small.png'),dpi=400)

    #medium
    plt.figure()
    list_mass = whole_med_an[0]
    xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
    plt.imshow(whole_med[0], cmap='gray')
    bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                        label='lesion', facecolor='none')
    plt.gca().add_patch(bounding_box)
    plt.title('Medium Lesion')
    plt.savefig(os.path.join('data_figures','whole_im_les_med.png'),dpi=400)
    plt.close()
    #large
    plt.figure()
    list_mass = whole_large_an[1]
    xmin, xmax, ymin, ymax = list_mass[0], list_mass[1], list_mass[2], list_mass[3]
    plt.imshow(whole_lage[1], cmap='gray')
    bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                        label='lesion', facecolor='none')
    plt.gca().add_patch(bounding_box)
    plt.title('Large Lesion')
    plt.savefig(os.path.join('data_figures','whole_im_les_large.png'),dpi=400)
    plt.close()

def small_patch_sizes(image,annotations):
    mult_valus = [1,1.25,1.5,1.75,2,2.25]
    colors = ['yellow','white','blue','green','purple','orange']
    plt.figure(figsize=[10,10])
    for multiplication,color in zip(mult_valus,colors): 
        patch_size = int(SMALL_PATCH * float(multiplication))
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

        patch = {
        'box': (xmin_new, ymin_new, xmax_new, ymax_new)
        }
        # Fill outside regions with zeros
        box = patch['box']
        if multiplication == 1:
            plt.imshow(image, cmap='gray')
            # The lesion
            bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                                label='lesion', facecolor='none')
            plt.gca().add_patch(bounding_box)
        # The sampled patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=color,
                                    facecolor='none',label=str(multiplication))
        plt.gca().add_patch(rect)
        plt.tight_layout()
        plt.title(f'small window')
        plt.ylim([500,2250])
        plt.xlim([2000,image.shape[1]])
        plt.legend()
    plt.savefig(os.path.join('data_figures','patch_sizes_small.png'),dpi=400)
    plt.close()

    patch_height = int(np.ceil(ymax_new - ymin_new))
    patch_width = int(np.ceil(xmax_new - xmin_new))
    patch_height = min(patch_height, patch_size)
    patch_width = min(patch_width, patch_size)
    patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
    patch[:patch_height, :patch_width, :] = image[
        int(np.floor(ymin_new)):int(np.floor(ymin_new)) + patch_height,
        int(np.floor(xmin_new)):int(np.floor(xmin_new)) + patch_width,
        :
    ]
    plt.figure()
    datagen = ImageDataGenerator(
        rotation_range=180,  # 90 degree range for rotations - randomly
        horizontal_flip=True,  # Random horiz flips
        vertical_flip=True,  # Random vert flips
    )
    patch = datagen.random_transform(patch)
    plt.imshow(patch, cmap='gray')
    plt.savefig(os.path.join('data_figures','patch_augment_small.png'),dpi=400)
    plt.close()

def medium_patch_sizes(image,annotations):
    mult_valus = [1,1.25]
    colors = ['blue','orange']
    plt.figure(figsize=[10,10])
    for multiplication,color in zip(mult_valus,colors): 
        patch_size = int(MEDIUM_PATCH * float(multiplication))
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

        patch = {
        'box': (xmin_new, ymin_new, xmax_new, ymax_new)
        }
        # Fill outside regions with zeros
        box = patch['box']
        if multiplication == 1:
            plt.imshow(image, cmap='gray')
            # The lesion
            bounding_box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                                label='lesion', facecolor='none')
            plt.gca().add_patch(bounding_box)
        # The sampled patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=color,
                                    facecolor='none',label=str(multiplication))
        plt.gca().add_patch(rect)
        plt.tight_layout()
        plt.title(f'medium window')
        plt.ylim([1600,2800])
        plt.xlim([0,500])
        plt.legend()
    plt.savefig(os.path.join('data_figures','patch_sizes_medium.png'),dpi=400)
    plt.close()

    patch_height = int(np.ceil(ymax_new - ymin_new))
    patch_width = int(np.ceil(xmax_new - xmin_new))
    patch_height = min(patch_height, patch_size)
    patch_width = min(patch_width, patch_size)
    patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
    patch[:patch_height, :patch_width, :] = image[
        int(np.floor(ymin_new)):int(np.floor(ymin_new)) + patch_height,
        int(np.floor(xmin_new)):int(np.floor(xmin_new)) + patch_width,
        :
    ]
    plt.figure()
    datagen = ImageDataGenerator(
        rotation_range=180,  # 90 degree range for rotations - randomly
        horizontal_flip=True,  # Random horiz flips
        vertical_flip=True,  # Random vert flips
    )
    patch = datagen.random_transform(patch)
    plt.imshow(patch, cmap='gray')
    plt.savefig(os.path.join('data_figures','patch_augment_medium.png'),dpi=400)
    plt.close()

show_small_medium_large_patches()