import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy as sp
import cv2
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.data_setup import get_model_info

# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.fromarray(x.astype('uint8'), 'RGB'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])

# for storing results; gets predicted class for given model and img
def get_top_prediction(model_name, img):
    if isinstance(model_name, str):
        model, classes, layer = get_model_info(model_name)
    else:
        model = model_name
    logits = model(img)
    probs = F.softmax(logits, dim=1)
    prediction = probs.topk(1)
    return classes[prediction[1][0].detach().cpu().numpy()[0]]

def get_model(model_name):
    return get_model_info(model_name)[0]

def get_imagenet_classes():
    classes = list()
    try:
        with open('/work/lisabdunlap/explain-eval/data/synset_words.txt') as lines:
            for line in lines:
                line = line.strip().split(' ', 1)[1]
                line = line.split(', ', 1)[0].replace(' ', '_')
                classes.append(line)
    except:
        with open('/work/lisabdunlap/explain-eval/data/synset_words.txt') as lines:
            for line in lines:
                line = line.strip().split(' ', 1)[1]
                line = line.split(', ', 1)[0].replace(' ', '_')
                classes.append(line)
    return classes

def get_model_layer(model_name):
    return get_model_info(model_name)[2]

#weight mask evenly so that the usm equals that of the heatmap
def weight_mask(heatmap, mask):
    heatmap_sum = np.sum(heatmap)
    lime_sum = float(np.sum(mask))
    mask = np.array(mask, dtype=float)
    mask[mask==1.0] = float(heatmap_sum/lime_sum)
    return heatmap, mask

def preprocess_groundings(map, sum_value=1, threshold=100, binary=False):
    m=np.array(map, dtype=float)
    if len(np.unique(m)) == 2:
        m = erode(map, threshold)
    else:
        m[m < np.percentile(m, 100-threshold)] = 0
        if binary:
            m[m!=0] = 1.0
            #print(np.sum(m))
    if not binary:
        norm = float(np.sum(m)/sum_value)
        m /= norm
    return m

def erode(mask, threshold):
    w,h = mask.shape
    #plt.imshow(mask)
    num_pix = (threshold/100)*(w*h)
    #print("threshold {0}".format(num_pix))
    eroded = mask
    i=1
    if threshold == 100:
        #print("return {0}".format(np.sum(mask)))
        return np.array(mask, dtype=float)
    while True:
        kernel = np.ones((5,5), np.uint8) 
        eroded = cv2.erode(mask, kernel, iterations=i)
        if np.sum(eroded) <= num_pix:
            #print("return {0}".format(np.sum(eroded)))
            #plt.imshow(eroded)
            return eroded
        else:
            i = i+1
    return np.array(eroded, dtype=float)

# calc iou where it is sum(values of interesction)/sum(values of union)
# if same=True, then we are comparing two maps from the same technique
def calc_iou(heatmap, mask, threshold=0, num_pixels=False):
    #if not same:
    #    heatmap, mask = weight_mask(heatmap, mask)
    if num_pixels:
        m1 = preprocess_groundings(heatmap, threshold=threshold, binary=True)
        m2 = preprocess_groundings(mask, threshold=threshold, binary=True)
    else:
        m1 = preprocess_groundings(heatmap, threshold=threshold)
        m2 = preprocess_groundings(mask, threshold=threshold)
    union = np.add(m1, m2)
    union[union == 2.0] = 1.0
    intersection = np.multiply(m1, m2)
    #print("intersesction {0}".format(np.sum(intersection)))
    #print("union {0}".format(np.sum(union)))
    iou =np.sum(intersection)/np.sum(union)
    return iou, intersection, union

# calc cosine similarity with mask and normalized heatmap
def cos_similarity(heatmap, mask, threshold=0):
    #norm_heatmap = heatmap/np.max(heatmap)
    m1 = preprocess_groundings(heatmap, threshold=threshold)
    m2 = preprocess_groundings(mask, threshold=threshold)
    A = m1
    B = m2

    Aflat = np.hstack(A)
    Bflat = np.hstack(B)

    dist = distance.cosine(Aflat, Bflat)
    return dist

def jsd(map1, map2, base=np.e, threshold=0):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    m1 = preprocess_groundings(map1, threshold=threshold)
    m2 = preprocess_groundings(map2, threshold=threshold)
    #m = 1./2*(m1 + m2)
    #return sp.stats.entropy(m1,m, base=base)/2. +  sp.stats.entropy(m2, m, base=base)/2.
    return distance.jensenshannon(m1, m2)

def jensenshannon(p, q, base=None, threshold=100, mask=False):
    if mask:
        p = preprocess_groundings(p, threshold=threshold)
    q = preprocess_groundings(q, threshold=threshold)
    p = p.flatten()
    q = q.flatten()
    m = (p + q) / 2.0
    left = sp.special.rel_entr(p, m)
    right = sp.special.rel_entr(q, m)
    js = np.sum(left, axis=0) + np.sum(right, axis=0)
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)

def tvd(map1, map2, threshold):
    # Assumes a, b are numpy arrays
    m1 = preprocess_groundings(map1, threshold=threshold)
    m2 = preprocess_groundings(map2, threshold=threshold)
    m1 = m1.flatten()
    m2 = m2.flatten()
    return sum(abs(m1-m2))/2

def get_stats(map1, map2, threshold=0):
    #iou, intersection, union = calc_iou(map1, map2, threshold=threshold)
    #print("IoU: {0}".format(iou))
    iou_pix, intersection_pix, union_pix = calc_iou(map1, map2, threshold=threshold, num_pixels=True)
    print("pixel count IoU: {0}".format(iou_pix))
    cos_dist = cos_similarity(map1, map2, threshold=threshold)
    print("cos similarity: {0}".format(cos_dist))
    #flatten maps to get Jensen Shannon Distance
    js_dist = jensenshannon(map1, map2, threshold=threshold)
    print("Jenson Shannon dist: {0}".format(js_dist))
    tv_dist = tvd(map1, map2, threshold=threshold)
    print("total variation distance: {0}".format(tv_dist))
    return iou_pix, cos_dist, js_dist, tv_dist

"""from matplotlib import pylab as P
def showGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
        P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)"""

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def pointing_game(expl, mask):
    max_saliency = expl
    #print(max_saliency[max_saliency == np.max(max_saliency)])
    max_saliency[max_saliency == np.max(max_saliency)] = 1
    #mask = get_img_mask(expl, location, show=False)
    if len(mask[max_saliency == 1]) == 1:
        return [0.0,1.0]
    else:
        return [1.0, 0.0]

def get_img_mask(img, location, show=True):
    try:
        l,w,_ = img.shape
    except:
        l,w = img.size
    overlay = np.zeros((l,w))
    overlay[int(location[0]):int(location[2]), int(location[1]):int(location[3])].fill(1)
    if show:
        plt.imshow(overlay)
    return overlay

def get_displ_img(img):
    try:
        img = img.cpu().numpy().transpose((1, 2, 0))
    except:
        img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    displ_img = std * img + mean
    displ_img = np.clip(displ_img, 0, 1)
    displ_img /= np.max(displ_img)
    displ_img = displ_img
    return np.uint8(displ_img*255)

'''
Displayes mask as heatmap on image
'''
def get_cam(img, mask):
    img = get_displ_img(img)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0), cv2.COLORMAP_JET),
                               cv2.COLOR_BGR2RGB)
    alpha = .4
    cam = heatmap*alpha + np.float32(img)*(1-alpha)
    cam /= np.max(cam)
    return cam