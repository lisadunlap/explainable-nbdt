import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torchvision import models, transforms
from skimage import io
import cv2
import os
from datetime import datetime
import gc
import torch.nn as nn
from mpl_toolkits.axes_grid1 import ImageGrid

#techniques
from Grad_CAM.main_gcam import gen_gcam, gen_gcam_target, gen_bp, gen_gbp, gen_bp_target, gen_gbp_target
from utils import get_cam, get_model_info, get_imagenet_classes

def gen_grounding_gcam_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    '''
            Generate an explanation for an image using a certain technique.

            :param img: tensor of images
            :param technique: name of technique that you want to generate an
            explanation for. Currently supports LIME('lime'), RISE('rise'), Grad-CAM('gcam'),
            Backpropagation('bp'), GuidedBackprop('gbp'), Deconvolution('deconv')
            :param model: model or name of model(if pretrained) to evaluaute on
            :param show: display the image
            :param layer: (for gradcam) name of last convolutional layer of model
            (if using pretrained this will be filled in for you)
            :param save_path: path to results
            :param target_index: target class index if INDEX==True; otherwise the topk index
            :param save: save the explanations (both as imgs and np arrays of just the heatmaps)
            :param device: either cpu or gpu number
            :param index: predict with respect to a specific class number
            (default is predict topk class) TODO: get LIME and RISE working with specific indices
            :return explanation for image as np array:
    '''

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    if topk:
        masks = gen_gcam(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    else:
        masks = gen_gcam_target(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
    
    if show:
        #plot heatmaps
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         axes_pad=0.35,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cams[:4]):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
    if save:
        for i in range(len(imgs)):
            res_path = save_path + str(target_index[i].cpu().numpy()) + '/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            #print("saving explanation mask....\n")
            cv2.imwrite(res_path + 'original_img.png', get_displ_img(imgs[i]))

            if not cv2.imwrite(res_path +"gcam.png", np.uint8(cams[i]*255)):
                print('error saving explanation')
            np.save(res_path + "gcam_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks

def gen_grounding_bp_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    if topk:
        masks = gen_bp(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    else:
        masks = gen_bp_target(imgs, model, target_index = target_index, device=device, single=False, prep=False, classes=classes)
    
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
    
    if show:
        #plot heatmaps
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         axes_pad=0.35,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cams[:4]):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
    if save:
        for i in range(len(imgs)):
            res_path = save_path + str(target_index[i].cpu().numpy()) + '/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            #print("saving explanation mask....\n")
            cv2.imwrite(res_path + 'original_img.png', get_displ_img(imgs[i]))
            cv2.imwrite(res_path +"bp_mask.png", np.uint8(cams[i]*255))
            np.save(res_path +"bp_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks

def gen_grounding_gbp_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    if topk:
        masks = gen_gbp(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    else:
        masks = gen_gbp_target(imgs, model, target_index = target_index, device=device, single=False, prep=False, classes=classes)
        
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
            
    if save:
        for i in range(len(imgs)):
            res_path = save_path + str(target_index[i].cpu().numpy()) + '/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            #print("saving explanation mask....\n")
            cv2.imwrite(res_path + 'original_img.png', get_displ_img(imgs[i]))
            cv2.imwrite(res_path +"gbp_mask.png", np.uint8(cams[i]*255))
            np.save(res_path +"gbp_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks
