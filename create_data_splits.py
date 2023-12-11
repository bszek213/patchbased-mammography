import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_data(images, annotations):
    # Split images and annotations
    indices = np.arange(len(images))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_images = images[train_indices]
    val_images = images[val_indices]
    test_images = images[test_indices]

    train_annotations = annotations[train_indices]
    val_annotations = annotations[val_indices]
    test_annotations = annotations[test_indices]

    return train_images, val_images, test_images, train_annotations, val_annotations, test_annotations

whole_small_images = np.load(os.path.join(os.getcwd(),'data', 'whole_images_small.npy'), allow_pickle=True)
whole_medium_images = np.load(os.path.join(os.getcwd(),'data', 'whole_images_medium.npy'), allow_pickle=True)
whole_large_images = np.load(os.path.join(os.getcwd(),'data', 'whole_images_large.npy'), allow_pickle=True)

whole_small_images_annotations = np.load(os.path.join(os.getcwd(),'data','whole_images_small_annotations.npy'), allow_pickle=True)
whole_medium_images_annotations = np.load(os.path.join(os.getcwd(),'data','whole_images_medium_annotations.npy'), allow_pickle=True)
whole_large_images_annotations = np.load(os.path.join(os.getcwd(),'data','whole_images_large_annotations.npy'), allow_pickle=True)

train_images_arr1, val_images_arr1, test_images_arr1, train_annotations_arr1, val_annotations_arr1, test_annotations_arr1 = split_data(whole_small_images, whole_small_images_annotations)
train_images_arr2, val_images_arr2, test_images_arr2, train_annotations_arr2, val_annotations_arr2, test_annotations_arr2 = split_data(whole_medium_images, whole_medium_images_annotations)
train_images_arr3, val_images_arr3, test_images_arr3, train_annotations_arr3, val_annotations_arr3, test_annotations_arr3 = split_data(whole_large_images, whole_large_images_annotations)

if not os.path.exists('data'):
    os.mkdir('data')
# Save small images and annotations
np.save(os.path.join('data', 'small_images_train.npy'), train_images_arr1)
np.save(os.path.join('data', 'small_images_validation.npy'), val_images_arr1)
np.save(os.path.join('data', 'small_images_test.npy'), test_images_arr1)
np.save(os.path.join('data', 'small_annotations_train.npy'), train_annotations_arr1)
np.save(os.path.join('data', 'small_annotations_validation.npy'), val_annotations_arr1)
np.save(os.path.join('data', 'small_annotations_test.npy'), test_annotations_arr1)

# Save medium images and annotations
np.save(os.path.join('data', 'medium_images_train.npy'), train_images_arr2)
np.save(os.path.join('data', 'medium_images_validation.npy'), val_images_arr2)
np.save(os.path.join('data', 'medium_images_test.npy'), test_images_arr2)
np.save(os.path.join('data', 'medium_annotations_train.npy'), train_annotations_arr2)
np.save(os.path.join('data', 'medium_annotations_validation.npy'), val_annotations_arr2)
np.save(os.path.join('data', 'medium_annotations_test.npy'), test_annotations_arr2)

# Save large images and annotations
np.save(os.path.join('data', 'large_images_train.npy'), train_images_arr3)
np.save(os.path.join('data', 'large_images_validation.npy'), val_images_arr3)
np.save(os.path.join('data', 'large_images_test.npy'), test_images_arr3)
np.save(os.path.join('data', 'large_annotations_train.npy'), train_annotations_arr3)
np.save(os.path.join('data', 'large_annotations_validation.npy'), val_annotations_arr3)
np.save(os.path.join('data', 'large_annotations_test.npy'), test_annotations_arr3)