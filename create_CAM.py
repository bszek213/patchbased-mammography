import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import tensorflow as tf
from tensorflow.keras import backend as K
import os
from keras.models import Model

# tf.compat.v1.disable_eager_execution()

# Load the pre-trained model
model = load_model("patch_dense121_small_patches_1-25.h5")
input_shape = model.layers[0].output_shape[1:3]
# # Access the DenseNet121 base layer
# densenet_base = model.get_layer("densenet121").layers
# # Print the names of layers within the DenseNet121 base
# for layer in densenet_base.layers:
#     print(layer.name)
# Load an example image for which you want to generate the CAM
images = np.load(os.path.join(os.getcwd(), 'data_patch', 'small_patches_1-25_test.npy'))
img_array = images[0]
img_array = cv2.resize(img_array, (172, 172))  # Resize the image to match the model's expected input shape
img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
img_array = tf.keras.applications.densenet.preprocess_input(img_array)
img_array = tf.expand_dims(img_array, axis=0)

# Get the last convolutional layer and the model's prediction
densenet_base = model.get_layer("densenet121")
last_conv_layer = densenet_base.get_layer("conv5_block16_concat")
model2 = Model(inputs=model.input, outputs = last_conv_layer.output)
final_dense = model.get_layer('dense_2')
model2.summary()
W = final_dense.get_weights()[0]
# model_output = model.output
# class_index = tf.argmax(model_output[0])

# # Compute the gradient of the class output with respect to the last conv layer using TensorFlow's eager execution
# with tf.GradientTape() as tape:
#     tape.watch(img_array)
#     last_conv_output = densenet_base(img_array, training=False)
#     model_output = model(img_array)
#     class_index = tf.argmax(model_output[0])
#     model_output = model_output[:, class_index]

# # Get the gradient with respect to the last convolutional layer
# grads = tape.gradient(model_output, last_conv_output)

# # Compute the mean of the gradient over each feature map channel
# pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))


# # Multiply each channel in the feature map array by the "importance" (pooled grads)
# heatmap = tf.reduce_mean(intermediate_output * tf.expand_dims(pooled_grads, axis=(0, 1, 2)), axis=-1)

# # Normalize the heatmap between 0 and 1
# heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

# # Resize the heatmap to match the dimensions of the original image
# heatmap = tf.image.resize(heatmap, (input_shape[0], input_shape[1]))

# # Convert the heatmap to a NumPy array
# heatmap = heatmap.numpy()

# # Apply the heatmap to the original image
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + np.uint8(255 * img_array[0])

# # Save and display the results
# cv2.imwrite("heatmap.jpg", superimposed_img)
# plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
# plt.show()