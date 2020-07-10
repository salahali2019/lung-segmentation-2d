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

    # For inference standard keras ImageGenerator is used.
    test_gen = ImageDataGenerator(rescale=1.)

    i = 0
    for i in range(n_test):    
        img=np.expand_dims(X[i], axis=(0))
        pred = UNet.predict(img)[..., 0].reshape(inp_shape[:2])
        image_name=os.path.join(output_Dir,'Unet_'+image_paths[i].split('/')[-1])
        pred=resize(pred,original_shape[i])
        io.imsave(image_name,pred)
        
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
