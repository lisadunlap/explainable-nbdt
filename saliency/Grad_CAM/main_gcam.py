from __future__ import print_function

# code originally from https://github.com/kazuto1011/grad-cam-pytorch

import copy
import os.path as osp

#import click
import cv2
import PIL
import matplotlib.cm as cm
import sys
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
from ..utils import get_imagenet_classes

from .gcam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
)


def get_device(cuda, device):
    cuda = cuda and torch.cuda.is_available()
    cuda_dev = "cuda:"+str(device)
    device = torch.device(cuda_dev if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def preprocess(raw_image, transf=None):
    is_tensor = type(raw_image) == torch.Tensor
    if is_tensor:
        orig = raw_image.clone()
        if transf:
            raw_image = transf(raw_image)
        raw_image = raw_image.unsqueeze(0)
        raw_image = torch.nn.functional.interpolate(raw_image, size=(224,224))
        raw_image = raw_image.squeeze(0).permute(1,2,0)
    elif type(raw_image) == PIL.Image.Image:
        raw_image = transforms.Resize(size=(224, 224))(raw_image)

    else:
        raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image.numpy() if is_tensor else raw_image.copy())
    if is_tensor:
        image = torch.Tensor(image)
        raw_image = orig
    return image, raw_image


def save_gradient(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    return gradient


def save_gradcam(gcam):
    gcam = gcam.cpu().numpy()
    return gcam

def gen_model_forward(imgs, model, device='cuda', prep=True, type='gcam', transf=None):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    #model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im, transf)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    
    images = torch.stack(images) #.to(device)

    if type == 'gcam':
        tech_model = GradCAM(model=model)
    elif type == 'bp':
        tech_model = BackPropagation(model=model)
    elif type == 'gbp':
        tech_model = GuidedBackPropagation(model=model)
    elif type == 'deconv':
        tech_model = Deconvnet(model=model)
    else:
        print('Invalid Type')
        sys.exit()

    probs, ids = tech_model.forward(images)
    return tech_model, probs, ids, images
    
def gen_gcam(imgs, model, target_layer='layer4', target_index=1, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Get model and forward pass   
    gcam, probs, ids, images = gen_model_forward(imgs, model, device=device, prep=prep, type='gcam')

    for i in range(target_index):
        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)
        masks = []
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Grad-CAM
            mask = save_gradcam(
                gcam=regions[j, 0]
            )
            masks += [mask]
    if len(masks) == 1:
        return masks[0]
    gcam.remove_hook()
    return masks

def gen_gcam_target(imgs, model, target_layer='layer4', target_index=None, classes=get_imagenet_classes(), device='cuda', prep=True, transf=None):
    """
    Visualize model responses given multiple images
    """
   
    # Get model and forward pass   
    gcam, probs, ids, images = gen_model_forward(imgs, model, device=device, prep=prep, type='gcam', transf=transf)
    
    ids_ = torch.LongTensor([[x] for x in target_index]).to(device)
    gcam.backward(ids=ids_)
    regions = gcam.generate(target_layer=target_layer)
    masks=[]
    for j in range(len(images)):
        mask = save_gradcam(
            gcam=regions[j, 0]
        )
        masks += [mask]

    if len(masks) == 1:
        return masks[0]
    return masks


def gen_bp(imgs, model, target_index=1, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """
    # Get model and forward pass   
    bp, probs, ids, images = gen_model_forward(imgs, model, device=device, prep=prep, type='bp')

    for i in range(target_index):
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()
        masks = []
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            mask = save_gradient(
                gradient=gradients[j],
            )
            mask /= np.max(mask)
            masks += [mask]
    if len(masks) == 1:
        return masks[0]
    bp.remove_hook()
    return masks

def gen_bp_target(imgs, model, target_index=None, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Get model and forward pass   
    bp, probs, ids, images = gen_model_forward(imgs, model, device=device, prep=prep, type='bp')
    
    ids_ = torch.LongTensor([[x] for x in target_index]).to(device)
    bp.backward(ids=ids_)
    gradients = bp.generate()
    masks = []
    for j in range(len(images)):
        mask = save_gradient(
                gradient=gradients[j],
            )
        mask /= np.max(mask)
        masks += [mask]
    if len(masks) == 1:
        return masks[0]
    return masks

def gen_gbp(imgs, model, target_index=1, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Get model and forward pass   
    gbp, probs, ids, images = gen_model_forward(imgs, model, device=device, prep=prep, type='gbp')

    for i in range(target_index):
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()
        masks = []
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            mask = save_gradient(
                gradient=gradients[j],
            )
            mask /= np.max(mask)
            masks += [mask]
    if len(masks) == 1:
        return masks[0]
    gbp.remove_hook()
    return masks

def gen_gbp_target(imgs, model, target_index=None, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Get model and forward pass   
    gbp, probs, ids, images = gen_model_forward(imgs, model, device=device, prep=prep, type='gbp')
    
    ids_ = torch.LongTensor([[x] for x in target_index]).to(device)
    gbp.backward(ids=ids_)
    gradients = gbp.generate()
    masks = []
    for j in range(len(images)):
        mask = save_gradient(
                gradient=gradients[j],
            )
        mask /= np.max(mask)
        masks += [mask]
    if len(masks) == 1:
        return masks[0]
    return masks


def gen_deconv(imgs, model, target_index=1, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Get model and forward pass   
    deconv, probs, ids, images = gen_model_forward(imgs, model, device=device, prep=prep, type='deconv')

    for i in range(target_index):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()
        masks = []
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            mask = save_gradient(
                gradient=gradients[j],
            )

            # For Display Purposes: only keep top 10% of values
            # this makes the gradient more targeted and less "static-y"
            mask /= np.max(mask)
            masks += [mask]
    if len(masks) == 1:
        return masks[0]
    deconv.remove_hook()
    return masks

def gen_deconv_target(imgs, model, target_index=None, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Get model and forward pass   
    deconv, probs, ids, images = gen_model_forward(imgs, model, device=device, prep=prep, type='deconv')

    ids_ = torch.LongTensor([[x] for x in target_index]).to(device)
    deconv.backward(ids=ids_)
    gradients = deconv.generate()
    masks = []
    for j in range(len(images)):
        mask = save_gradient(
                gradient=gradients[j],
            )
        mask /= np.max(mask)
        masks += [mask]
    if len(masks) == 1:
        return masks[0]
    return masks
