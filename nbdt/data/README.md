# NBDT Few-Shot Datasets

## Available Datasets
So far, we have integrated the following normal datasets:
* CIFAR10
* CIFAR100
* TinyImagenet
* ImageNet

And the following Zero-Shot/Meta Learning datasets:
* Animals with Attribtes 2
* CalTech Birds (CUB) Dataset (I think...)
* MiniImagenet

For these zeroshot datasets, there is a `--zeroshot` flag which adds in the predetermined zeroshot classes

## Custom

For the non-zeroshot datasets, we have implemented a few datasets which allow you to remove classes from your dataset 
* IncludeLabels/ExcludeLabels: either includes or excludes certain labels (indices) from a dataset. If `drop_classes=False`, then it keeps the indices of the original dataset but doesnt provide any samples for the excluded indices.
* IncludeClasses/ExcludeClasses: same as above but takes in a string ("cat") instead of a label (3)