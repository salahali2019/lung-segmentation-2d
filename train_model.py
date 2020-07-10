from image_gen import ImageDataGenerator
from load_data import loadDataMontgomery, loadDataJSRT,loadTrainingData
from build_model import build_UNet2D_4L

import pandas as pd
# from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
import glob
from sklearn.model_selection import train_test_split
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __name__ == '__main__':

    import argparse


    parser = argparse.ArgumentParser(description='Training U net model')

    parser.add_argument(
        '--input',
       
        help='Path to input images'
        )
    parser.add_argument(
        '--mask',
        
        help='Path to the ground truth masks'
        )
    parser.add_argument(
        '--model_weight_dir',
        
        help='Path to the model weight'
        )
    parser.add_argument( "--epochs", type=int, default= 100,
                        help="number of epochs ")

    parser.add_argument( "--periods", type=int, default= 5,
                        help="periods of saving weights, i.e if epoch 100 , period 10. weights will be saved every 10 epochs ")
    args = parser.parse_args()
    input_dir = args.input
    mask_dir = args.mask
    model_weight_dir = args.model_weight_dir
    EPOCHS = args.epochs
    PERIODS = args.periods


    if not os.path.exists(model_weight_dir):
        os.mkdir(model_weight_dir)

    # mask_paths=glob.glob('training_dataset/output/*')
    # image_paths=glob.glob('training_dataset/input/*')

    image_paths=glob.glob(os.path.join(input_dir,'*'))[:100]
    mask_paths=glob.glob(os.path.join(mask_dir,'*'))[:100]

    print(len(image_paths))
    print(len(mask_paths))



    # Load training and validation data
    im_shape = (256, 256)
    X, y = loadTrainingData(image_paths, mask_paths, im_shape)
    X_train, X_val, y_train ,y_val = train_test_split( X, y, test_size=0.2, random_state=42)

    print(X_train.shape,y_train.shape)

    # Build model
    inp_shape = X_train[0].shape
    UNet = build_UNet2D_4L(inp_shape)
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Visualize model
    # plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
    model_file_format = os.path.join(model_weight_dir,'model.{epoch:03d}.hdf5')
    # print(model_file_format)
    checkpointer = ModelCheckpoint(model_file_format, period=PERIODS)

    # checkpointer = ModelCheckpoint(model_file_format, save_weights_only=True, save_best_only=True, mode='min')


    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

    test_gen = ImageDataGenerator(rescale=1.)

    batch_size = 8
    history = UNet.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                       epochs=EPOCHS,
                       callbacks=[checkpointer],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size)

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valdiation'], loc='upper left')
    plt.savefig('training_loss.png')

