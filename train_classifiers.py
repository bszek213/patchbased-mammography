import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout#, Input, Concatenate
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sys import argv
from keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.callbacks import LearningRateScheduler
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential

def create_patch_model_res(input_shape):
    """
    input shape = (250,250,3,12)
    """
    # input_layer = Input(shape=input_shape)
    # rgb_input = Concatenate()([input_layer, input_layer, input_layer])
    base_model = ResNet152V2(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(GLOBAL_X, GLOBAL_Y, 3))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss=BinaryFocalCrossentropy(), optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    return model

def create_patch_model_dense(input_shape):
    """
    input shape = (250,250,3,12)
    """
    # input_layer = Input(shape=input_shape)
    # rgb_input = Concatenate()([input_layer, input_layer, input_layer])
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(GLOBAL_X, GLOBAL_Y, 3))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    return model

def schedule(epoch, lr):
    if epoch < 5:
        return 0.001
    else:
        return 0.001 * np.exp(0.1 * (10 - epoch))
    
def main():
    type_patch_total = ['small']
    multiplication_values = ['1','1-25','1-50','1-75','2','2-25']

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(schedule)

    for type_patch in type_patch_total:
        for multiply in multiplication_values:
            X_train = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_{str(multiply)}_train.npy'))
            x_val = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_{str(multiply)}_valid.npy'))
            X_test = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_{str(multiply)}_test.npy'))

            y_train = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_train_{str(multiply)}_label.npy'))
            y_test = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_test_{str(multiply)}_label.npy'))
            y_val = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_valid_{str(multiply)}_label.npy'))

            print(f'x_train size {np.shape(X_train)}')
            print(f'X_test size {np.shape(X_test)}')
            print(f'y_train size {np.shape(y_train)}')
            print(f'y_test size {np.shape(y_test)}')
            print(f'x_val size {np.shape(x_val)}')
            print(f'y_val size {np.shape(y_val)}')
            
            #create model res
            patch_architecture_res = create_patch_model_res((X_train.shape[1],X_train.shape[1],3))
            history = patch_architecture_res.fit(X_train, y_train, epochs=100, batch_size=64,callbacks=[early_stopping,lr_scheduler],
                                                validation_data=(x_val, y_val), verbose=1)

            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            plt.plot(history.history['accuracy'], label=f'Training Accuracy')
            plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy')
            #plt.plot(history.history['val_f1_score'], label=f'Validation F1 Score ({i}th iteration)')
            plt.title(f'ResNet152 accuracy {type_patch} patches - {str(multiply)}xtimes patch size')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            plt.plot(history.history['loss'], label=f'Training Loss')
            plt.plot(history.history['val_loss'], label=f'Validation Loss')
            plt.title(f'ResNet152 loss {type_patch} patches - {str(multiply)}xtimes patch size')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()  # Adjust subplot spacing for better appearance
            plt.savefig(f'training_accuracy_loss_ResNet152_{type_patch}_patches_{str(multiply)}.png', dpi=400)
            plt.close()

            test_results = patch_architecture_res.evaluate(X_test, y_test)
            with open(f'test_results_res_{type_patch}_{str(multiply)}.txt', 'w') as file:
                file.write(f'Test Accuracy: {test_results[1]}\n')
                file.write(f'Test Loss: {test_results[0]}\n')
            patch_architecture_res.save(f'patch_ResNet152_{type_patch}_patches_{str(multiply)}.h5')

            #create model dense
            patch_architecture_dense = create_patch_model_dense((X_train.shape[1],X_train.shape[1],3))
            history = patch_architecture_dense.fit(X_train, y_train, epochs=100, batch_size=64,callbacks=[early_stopping,lr_scheduler],
                                                validation_data=(x_val, y_val), verbose=1)

            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            plt.plot(history.history['accuracy'], label=f'Training Accuracy')
            plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy')
            #plt.plot(history.history['val_f1_score'], label=f'Validation F1 Score ({i}th iteration)')
            plt.title(f'DenseNet121 accuracy {type_patch} patches - {str(multiply)}xtimes patch size')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            plt.plot(history.history['loss'], label=f'Training Loss')
            plt.plot(history.history['val_loss'], label=f'Validation Loss')
            plt.title(f'DenseNet121 loss {type_patch} patches - {str(multiply)}xtimes patch size')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()  # Adjust subplot spacing for better appearance
            plt.savefig(f'training_accuracy_loss_DenseNet121_{type_patch}_patches_{str(multiply)}.png', dpi=400)
            plt.close()

            test_results = patch_architecture_dense.evaluate(X_test, y_test)
            with open(f'test_results_dense_{type_patch}_{str(multiply)}.txt', 'w') as file:
                file.write(f'Test Accuracy: {test_results[1]}\n')
                file.write(f'Test Loss: {test_results[0]}\n')
            patch_architecture_dense.save(f'patch_dense121_{type_patch}_patches_{str(multiply)}.h5')

    #train medium and large
    type_patch_total = ['medium', 'large']
    multiplication_values = ['1']
    for type_patch in type_patch_total:
        for multiply in multiplication_values:
            X_train = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_{str(multiply)}_train.npy'))
            x_val = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_{str(multiply)}_valid.npy'))
            X_test = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_{str(multiply)}_test.npy'))

            y_train = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_train_{str(multiply)}_label.npy'))
            y_test = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_test_{str(multiply)}_label.npy'))
            y_val = np.load(os.path.join(os.getcwd(),'data_patch', f'{type_patch}_patches_valid_{str(multiply)}_label.npy'))

            print(f'x_train size {np.shape(X_train)}')
            print(f'X_test size {np.shape(X_test)}')
            print(f'y_train size {np.shape(y_train)}')
            print(f'y_test size {np.shape(y_test)}')
            print(f'x_val size {np.shape(x_val)}')
            print(f'y_val size {np.shape(y_val)}')
            
            #create model res
            patch_architecture_res = create_patch_model_res((X_train.shape[1],X_train.shape[1],3))
            history = patch_architecture_res.fit(X_train, y_train, epochs=100, batch_size=64,callbacks=[early_stopping,lr_scheduler],
                                                validation_data=(x_val, y_val), verbose=1)

            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            plt.plot(history.history['accuracy'], label=f'Training Accuracy')
            plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy')
            #plt.plot(history.history['val_f1_score'], label=f'Validation F1 Score ({i}th iteration)')
            plt.title(f'ResNet152 accuracy {type_patch} patches - {str(multiply)}xtimes patch size')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            plt.plot(history.history['loss'], label=f'Training Loss')
            plt.plot(history.history['val_loss'], label=f'Validation Loss')
            plt.title(f'ResNet152 loss {type_patch} patches - {str(multiply)}xtimes patch size')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()  # Adjust subplot spacing for better appearance
            plt.savefig(f'training_accuracy_loss_ResNet152_{type_patch}_patches_{str(multiply)}.png', dpi=400)
            plt.close()

            test_results = patch_architecture_res.evaluate(X_test, y_test)
            with open(f'test_results_res_{type_patch}_{str(multiply)}.txt', 'w') as file:
                file.write(f'Test Accuracy: {test_results[1]}\n')
                file.write(f'Test Loss: {test_results[0]}\n')
            patch_architecture_res.save(f'patch_ResNet152_{type_patch}_patches_{str(multiply)}.h5')

            #create model dense
            patch_architecture_dense = create_patch_model_dense((X_train.shape[1],X_train.shape[1],3))
            history = patch_architecture_dense.fit(X_train, y_train, epochs=100, batch_size=64,callbacks=[early_stopping,lr_scheduler],
                                                validation_data=(x_val, y_val), verbose=1)

            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            plt.plot(history.history['accuracy'], label=f'Training Accuracy')
            plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy')
            #plt.plot(history.history['val_f1_score'], label=f'Validation F1 Score ({i}th iteration)')
            plt.title(f'DenseNet121 accuracy {type_patch} patches - {str(multiply)}xtimes patch size')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            plt.plot(history.history['loss'], label=f'Training Loss')
            plt.plot(history.history['val_loss'], label=f'Validation Loss')
            plt.title(f'DenseNet121 loss {type_patch} patches - {str(multiply)}xtimes patch size')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()  # Adjust subplot spacing for better appearance
            plt.savefig(f'training_accuracy_loss_DenseNet121_{type_patch}_patches_{str(multiply)}.png', dpi=400)
            plt.close()

            test_results = patch_architecture_dense.evaluate(X_test, y_test)
            with open(f'test_results_dense_{type_patch}_{str(multiply)}.txt', 'w') as file:
                file.write(f'Test Accuracy: {test_results[1]}\n')
                file.write(f'Test Loss: {test_results[0]}\n')
            patch_architecture_dense.save(f'patch_dense121_{type_patch}_patches_{str(multiply)}.h5')
if __name__ == "__main__":
    main()