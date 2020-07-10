from load_data import loadDataJSRT, loadDataMontgomery,loadTestingData

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, color, io, exposure
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.morphology import binary_closing, disk
import scipy.ndimage as nd
import glob
from skimage import transform, io, img_as_float, exposure
import os


import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage import filters #change to 'import filter' for Python>v2.7
from skimage import exposure
from keras import backend as K
import h5py


def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def Postprocessing(pred,img):
    pr = pred <0.03

    img=img.squeeze()
    im_shape=img.shape
    pr=remove_small_regions(pr, 0.06 * np.prod(im_shape))

    pr=morphology.closing(pr)


    strel = disk(9)
    pr = binary_closing(pr, strel)
    pr = nd.morphology.binary_fill_holes(pr)
    pr=img_as_ubyte(pr)
    
    return pr
def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

#Function to extract features from intermediate layers
def extra_feat(img_path,base_model):
        #Using a VGG19 as feature extractor
    # base_model = VGG19(weights='imagenet',include_top=False)
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    block1_pool_features=get_activations(base_model, 3, x)
    block2_pool_features=get_activations(base_model, 6, x)
    block3_pool_features=get_activations(base_model, 10, x)
    block4_pool_features=get_activations(base_model, 14, x)
    block5_pool_features=get_activations(base_model, 18, x)

    x1 = tf.image.resize_images(block1_pool_features[0],[112,112])
    x2 = tf.image.resize_images(block2_pool_features[0],[112,112])
    x3 = tf.image.resize_images(block3_pool_features[0],[112,112])
    x4 = tf.image.resize_images(block4_pool_features[0],[112,112])
    x5 = tf.image.resize_images(block5_pool_features[0],[112,112])
    
    F = tf.concat([x3,x2,x1,x4,x5],3) #Change to only x1, x1+x2,x1+x2+x3..so on, inorder to visualize features from diffetrrnt blocks
    return F
    
if __name__ == '__main__':

    import argparse


    parser = argparse.ArgumentParser(description='Create a verification report')

    parser.add_argument(
        '-i',
       
        help='Path to input images'
        )
    parser.add_argument(
        '-o',
        
        help='Path to the output directory'
        )
    parser.add_argument(
        '--model_weight_file',
        
        help='Path to the output directory'
        )
    args = parser.parse_args()
    input_dir = args.i
    output_Dir = args.o
    model_name=args.model_weight_file

    if not os.path.exists(output_Dir):
        os.mkdir(output_Dir)

    image_paths=glob.glob(os.path.join(input_dir,'*'))

    im_shape = (256, 256)
    X,original_shape= loadTestingData(image_paths,im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    # model_name = 'model.100.hdf5'

    # model_name = 'last_fluoro_model.1000.hdf5'
    UNet = load_model(model_name)
    base_model=UNet

    # For inference standard keras ImageGenerator is used.
    test_gen = ImageDataGenerator(rescale=1.)

    i = 0
    for i in range(n_test):    
        img=np.expand_dims(X[i], axis=(0))


        block1_pool_features=get_activations(base_model, 3, img)
        block2_pool_features=get_activations(base_model, 6, img)
        block3_pool_features=get_activations(base_model, 10, img)
        block4_pool_features=get_activations(base_model, 14, img)
        block5_pool_features=get_activations(base_model, 30, img)
        b1=np.array(block1_pool_features).squeeze()
        b2=np.array(block2_pool_features).squeeze()
        b3=np.array(block3_pool_features).squeeze()
        b4=np.array(block4_pool_features).squeeze()
        b5=np.array(block5_pool_features).squeeze()
        b1_out=np.sum(b1,axis=2)
        b2_out=np.sum(b2,axis=2)
        b3_out=np.sum(b3,axis=2)
        b4_out=np.sum(b4,axis=2)
        b5_out=np.sum(b5,axis=2)
        io.imsave('b1_out.tif',b1_out)
        io.imsave('b2_out.tif',b2_out)
        io.imsave('b3_out.tif',b3_out)
        io.imsave('b4_out.tif',b4_out)
        io.imsave('b5_out.tif',b5_out)


        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')


        # hf = h5py.File('data'+str(1)+'.h5', 'w')
        # hf.create_dataset('block1_pool_features', data=b1)
        # hf.create_dataset('block2_pool_features', data=b2)
        # hf.create_dataset('block3_pool_features', data=b3)
        # hf.create_dataset('block4_pool_features', data=b4)
        # hf.create_dataset('block5_pool_features', data=b5)



        # print(np.array(block1_pool_features).shape)
        # print(np.array(block2_pool_features).shape)
        # print(np.array(block3_pool_features).shape)
        # print(np.array(block4_pool_features).shape)
        # print(np.array(block5_pool_features).shape)


        # pred = UNet.predict(img)[..., 0].reshape(inp_shape[:2])
        # image_name=os.path.join(output_Dir,'Unet_'+image_paths[i].split('/')[-1])
        # pred=resize(pred,original_shape[i])
        # io.imsave(image_name,pred)
        
        # # Postprocessing
        # pr =Postprocessing(pred,img)
        # pr=img_as_ubyte(pr)

        # Processed_image_name=os.path.join(output_Dir,'Post_Unet_'+image_paths[i].split('/')[-1])
        # # print(Processed_image_name)
        # # print(pr.shape)
        # # print(pr.max())
        # # print(pr.min())

        # io.imsave(Processed_image_name,pr)

        i += 1
        if i == n_test:
            break

