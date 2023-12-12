import numpy as np
import os

print(np.load(os.path.join('data', 'whole_images_small.npy'),allow_pickle=True).shape)
print(np.load(os.path.join('data', 'whole_images_medium.npy'),allow_pickle=True).shape)
print(np.load(os.path.join('data', 'whole_images_large.npy'),allow_pickle=True).shape)

print('===========================')

print(np.load(os.path.join('data', 'small_images_train.npy'),allow_pickle=True).shape)
print(np.load(os.path.join('data', 'small_images_validation.npy'),allow_pickle=True).shape)
print(np.load(os.path.join('data', 'small_images_test.npy'),allow_pickle=True).shape)

