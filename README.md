# Neural-Backed Decision Trees for Few-Shot Learning

Run a decision tree that achieves accuracy within 1% of a recently state-of-the-art (WideResNet) neural network's accuracy on CIFAR10 and CIFAR100 and within 1.5% on TinyImagenet200.

![pipeline](https://user-images.githubusercontent.com/2068077/76384774-1ffb8480-631d-11ea-973f-7cac2a60bb10.jpg)

Per the pipeline illustration above, we (1) [generate the hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#1-Hierarchies) and (2) train the neural network [with a tree supervision loss](https://github.com/alvinwan/neural-backed-decision-trees#2-Tree-Supervision-Loss). Then, we (3) [run inference](https://github.com/alvinwan/neural-backed-decision-trees#3-Inference) by featurizing images using the network backbone and running embedded decision rules.

# Getting Started
The pipeline for few shot learning is thus: 
1. Train model: ```python main.py --model ResNet12 --dataset CIFAR10ExcludeClasses --exclude-classes cat --checkpoint-fname ckpt-CIFAR10-exclude-cat```
2. Induce Hierarchy" ```python generate_hierarchy.py --method induced --dataset CIFAR10 --ignore-labels 3 --induced-checkpoint ./checkpoint/ckpt-CIFAR10-exclude-cat.pth```
3. Finetune with NBDT (`--freeze-conv` is optional): ```python main.py --model ResNet12 --dataset CIFAR10ExcludeClasses --exclude-classes cat --resume --path-resume ./checkpoint/ckpt-CIFAR10-exclude-cat.pth --loss SoftTreeSupLoss --path-graph [JSON FILE OF HIERARCHY] --freeze-conv --analysis SoftEmbeddedDecisionRules```
4. Add zershot vector (use `--replace` if replacing a row): ```python add_zeroshot_vec.py --model ResNet12 --resume --path-resume ./checkpoint/ckpt-CIFAR10-exclude-cat.pth --new-classes cat --num-samples 5 --checkpoint-fname ckpt-CIFAR10-exclude-cat-weighted```
5. Reform Hierarchy: ```python generate_hierarchy.py --method induced --dataset CIFAR10 --ignore-labels 3 --induced-checkpoint ./checkpoint/ckpt-CIFAR10-exclude-cat-weighted.pth```
6. Evaluate: ```python main.py --model ResNet12 --dataset CIFAR10 --resume --eval --path-resume ./checkpoint/ckpt-CIFAR10-exclude-cat-weighted.pth --path-graph [JSON FILE OF HIERARCHY] --analysis SoftEmbeddedDecisionRules```


The bash scripts above are explained in more detail in [Induced Hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#Induced-Hierarchy), [Soft Tree Supervision Loss](https://github.com/alvinwan/neural-backed-decision-trees#Tree-Supervision-Loss), and [Soft Inference](https://github.com/alvinwan/neural-backed-decision-trees#Soft-Inference).

# 1. Hierarchies

## Induced Hierarchy

Run the following to generate and test induced hierarchies for CIFAR10, CIFAR100, based off of the WideResNet model. The script also downloads pretrained WideResNet models.

```
bash scripts/generate_hierarchies_induced_wrn.sh
```

![induced_structure](https://user-images.githubusercontent.com/2068077/76388304-0e6aaa80-6326-11ea-8c9b-6d08cb89fafe.jpg)


The below just explains the above `generate_hierarches_induced.sh`, using CIFAR10. You do not need to run the following after running the above bash script. Note that the following commands can be rerun with different checkpoints from different architectures, to produce different hierarchies.

```
# Step A. Download and evaluate pre-trained weights for WideResNet on CIFAR10.
python main.py --eval --pretrained --model=wrn28_10_cifar10 --dataset=CIFAR10

# Step B through D. Generate induced hierarchies, using the pretrained checkpoints
python generate_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --dataset=CIFAR10

# Test hierarchy
  python test_generated_graph.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --dataset=CIFAR100
```

## Wordnet Hierarchy

Run the following to generate and test Wordnet hierarchies for CIFAR10, CIFAR100, and TinyImagenet200. The script also downloads the NLTK Wordnet corpus.

```
bash scripts/generate_hierarchies_wordnet.sh
```

The below just explains the above `generate_hierarchies_wordnet.sh`, using CIFAR10. You do not need to run the following after running the above bash script.

```
# Generate mapping from classes to WNID. This is required for CIFAR10 and CIFAR100.
python generate_wnids.py --single-path --dataset=CIFAR10

# Generate hierarchy, using the WNIDs. This is required for all datasets: CIFAR10, CIFAR100, TinyImagenet200
python generate_hierarchy.py --single-path --dataset=CIFAR10

# Test hierarchy. This is optional but supported for all datasets. Make sure that your output ends with `==> All checks pass!`.
python test_generated_graph.py --single-path --dataset=CIFAR10
```

## Random Hierarchy

Use `--method=random` to randomly generate a binary-ish hierarchy. Additionally,
random trees feature two more flags:

- `--seed` to generate random leaf orderings. Use `seed=-1` to *not* randomly shuffle leaves.
- `--branching-factor` to generate trees with different branching factors. Setting branching factor to the number of classes is a nice sanity check. We used this for debugging, ourselves.

For example, to generate a sanity check hierarchy for CIFAR10, use

```
python generate_hierarchy.py --seed=-1 --branching-factor=10 --single-path --dataset=CIFAR10
```

## Visualize Hierarchy

Run the visualization generation script to obtain both the JSON representing
the hierarchy and the HTML file containing a d3 visualization.

```
python generate_vis.py --json-path [PATH TO JSON FILE]
```

The above script will output the following.

```
==> Reading from ./data/CIFAR10/graph-wordnet.json
==> Found just 1 root.
==> Wrote HTML to out/wordnet-tree.html
==> Wrote HTML to out/wordnet-graph.html
```

There are two visualizations. Open up `out/wordnet-tree.html` in your browser
to view the d3 tree visualization.

<img width="1436" alt="Screen Shot 2020-02-22 at 1 52 51 AM" src="https://user-images.githubusercontent.com/2068077/75101893-ca8f4b80-5598-11ea-9b47-7adcc3fc3027.png">

Open up `out/wordnet-graph.html` in your browser to view the d3 graph
visualization.

# 2. Tree Supervision Loss

In the below training commands, we uniformly use `--path-resume=<path/to/checkpoint> --lr=0.01` to fine-tune instead of training from scratch. Our results using a recently state-of-the-art pretrained checkpoint (WideResNet) were fine-tuned.

Run the following bash script to fine-tune WideResNet with both hard and soft tree supervision loss on CIFAR10, CIFAR100.

```
bash scripts/train_wrn.sh
```

As before, the below just explains the above `train_wrn.sh`. You do not need to run the following after running the previous bash script.

```
# fine-tune the wrn pretrained checkpoint on CIFAR10 with hard tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-resume=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10 --loss=HardTreeSupLoss

# fine-tune the wrn pretrained checkpoint on CIFAR10 with soft tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-resume=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10 --loss=SoftTreeSupLoss

# fine-tune the wrn pretrained checkpoint on CIFAR10 with hard tree supervision loss with a tree that has multiple paths to a leaf
python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-resume=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10 --loss=HardTreeSupLossMultiPath
```

To train from scratch, use `--lr=0.1` and do not pass the `--path-resume` flag.

# 3. Inference

![inference_modes](https://user-images.githubusercontent.com/2068077/76388544-9f418600-6326-11ea-9214-17356c71a066.jpg)

Like with the tree supervision loss variants, there are two inference variants: one is hard and one is soft. The best results in our paper, oddly enough, were obtained by running hard and soft inference *both* on the neural network supervised by a soft tree supervision loss.

Run the following bash script to obtain these numbers.

```
bash scripts/eval_wrn.sh
```

As before, the below just explains the above `eval_wrn.sh`. You do not need to run the following after running the previous bash script. Note the following commands are nearly identical to the corresponding train commands -- we drop the `lr`, `path-resume` flags and add `resume`, `eval`, and the `analysis` type (hard or soft inference).

```
# running soft inference on soft-supervised model
python main.py --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --tree-supervision-weight=10 --loss=SoftTreeSupLoss --eval --resume --analysis=SoftEmbeddedDecisionRules

# running hard inference on soft-supervised model
python main.py --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --tree-supervision-weight=10 --loss=SoftTreeSupLoss --eval --resume --analysis=HardEmbeddedDecisionRules
```

# Configuration

## Architecture

As a sample, we've included copies of all the above bash scripts but for ResNet10 and ResNet18. Simply add new model names or new dataset names to these bash scripts to test our method with more models or datasets.

```
bash scripts/generate_hierarchies_induced_resnet.sh  # this will train the network on the provided datasets if no checkpoints are found
bash scripts/train_resnet.sh
bash scripts/eval_resnet.sh
```

## Importing Other Models (`torchvision`, `pytorchcv`)

To add new models present in [`pytorchcv`](https://github.com/osmr/imgclsmob/tree/master/pytorch),
just add a new line to `models/__init__.py` importing the models you want. For
example, we added `from pytorchcv.models.wrn_cifar import *` for CIFAR wideresnet
models.

To add new models present in [`torchvision`](https://pytorch.org/docs/stable/torchvision/models.html), likewise just add a new line to `models/__init__.py`. For example, to import all, use `from torchvision.models import *`.

You can immediately start using these models with any of our utilities
above, including the custom tree supervision losses and extracted decision trees.

```
python main.py --model=wrn28_10_cifar10 --eval
python main.py --model=wrn28_10_cifar10 --eval --pretrained  # loads pretrained model
python main.py --model=wrn28_10_cifar10 --eval --pretrained --analysis=HardEmbeddedDecisionRules  # run the extracted hard decision tree
python main.py --model=wrn28_10_cifar10 --loss=HardTreeSupLoss --batch-size=256  # train with tree supervision loss
```

To download a pretrained checkpoint for a `pytorchcv` model, simply add the
`--pretrained` flag.

```
python main.py --model=wrn28_10_cifar10 --pretrained
```

If a pretrained checkpoint is already downloaded to disk, pass the path
using `--path-checkpoint`

```
python main.py --model=wrn28_10_cifar10 --path-checkpoint=...
```

## Inference on a Single Image

To run inference on a single image, run `explain_single.py`, passing in the model, the dataset it was trained on, and its graph

```
python explain_single.py --dataset=CIFAR10 --model=wrn28_10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --resume
```

## Modify Graph

You can modify a given graph in 2 ways: either you can cluster leaves with no siblings or add paths given parent nodes and wnids


To cluster:
```
python modify_tree.py --json-path=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --dataset CIFAR10 --method clustered
```

To add custom paths:
```
python modify_tree.py --json-path=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --dataset CIFAR10 --parents f00000010 --children n02958343
```

To get rid of those weird one off leaves (helps balance the tree)
```
python modify_tree.py --json-path=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --dataset CIFAR10 --mode prettify
```

## Weights and Biases Logging

To log results on weights and biases (which is strongly recommended), you must first create an account and run your command (either training, inference, or analysis), with the `--wandb` flag. The first time you run it you will be prompted to enter your api key.

## Generating Word2Vec Embeddings for Zeroshot/Fewshot training

To use word2vec embeddings, call `generate_word2vec.py` to generate and store Word2vec embeddings for your dataset before training. Since the model is pretrained, run `pip install gensim` if it is not installed

Example:
```
python generate_word2vec.py --dataset CIFAR10
```

## Zero Shot CIFAR10

To run zero shot experiments on cifar10 with the word2vec apprach or the image feature approach:
1. Train model on seen classes (add in `--word2vec` if using word2vec apprach)
```
python main.py --dataset=CIFAR10IncludeClasses --include-classes airplane automobile bird deer dog frog horse ship --model=ResNet10 --checkpoint-fname ckpt-CIFAR10-w2v-zsl-cat-truck --word2vec
```
2. Add in zero shot fc layers 
```
python add_zeroshot_vec.py --dataset=CIFAR10 --model=ResNet10 --new-classes cat truck --path-resume=./checkpoint/ckpt-CIFAR10-w2v-zsl-cat-truck.pth  --resume --checkpoint-fname ckpt-CIFAR10-zeroshot-cat-truck-full --word2vec
```
3. Evaluate
```
python main.py --dataset=CIFAR10 --model=ResNet10 --path-resume ./checkpoint/ckpt-CIFAR10-zeroshot-cat-truck-full.pth  --resume --eval --analysis ConfusionMatrix
```

## Tree Inference Ignoring Zero Shot Classes

Directly adding in feature vectors to the FC results in most samples from the original classes being classified
as the newly inserted classes. To combat this, we can run inference through an induced tree that ignores the
newly inserted weights until they are direct children of whatever node our path currently leads to. Assuming
the weight vectors have already been inserted, to do this:
1. Induce the tree by ignoring the new classes (`--ignore-labels`. This will induce a tree on all the other classes, then attach the
new classes to an existing node with the closest weight.
```
python generate_hierarchy.py --method=induced --induced-checkpoint=./checkpoint/ckpt-CIFAR10-Exclude-0-1-full.pth --ignore-labels 0 1
```
2. Run analysis by passing the same `--ignore-labels` flag.
```
python main.py --dataset=CIFAR10 --model=ResNet10 --induced-checkpoint=./checkpoint/ckpt-CIFAR10-Exclude-0-1-full.pth --ignore-labels 0 1 --resume --eval --analysis=HardEmbeddedDecisionRules
```

## Integrating a secondary OOD Dataset
Currently, this process is slightly annoying because you will have to create a new `wnids.txt` file for each set of OOD classes you want to test (e.g. by using the `SoftFullTreeOODPrior` class with the OOD parameters). For example, the following code trains a model on a dataset with images from `CIFAR10 - {cat, train}`, then feeds in images of `cat` and `train` as the OOD dataset.
*Note:* When listing out classes in parameters below, if you run into any errors, try alphabetizing the class names you pass in (this issue seems to come up every now and then).
```
# Train model
python main.py \
  --dataset=CIFAR10IncludeClasses --include-classes "airplane" "automobile" "bird" "deer" "dog" "frog" "horse" "ship" # list all classes in training set \
  --model=ResNet10 --checkpoint-fname=... 

# Generate hierarchy
python generate_hierarchy.py --method=induced --induced-checkpoint=... \
  --dataset=CIFAR10IncludeClasses --include-classes "airplane" "automobile" "bird" "deer" "dog" "frog" "horse" "ship"

# Analysis on OOD dataset
python main.py \
  --dataset=CIFAR10IncludeClasses --include-classes "airplane" "automobile" "bird" "deer" "dog" "frog" "horse" "ship" # list all classes in training set \
  --model=ResNet10 --path-resume=... --eval --resume --wandb \
  --path-graph=... --path-graph-analysis=... --path-wnids=... # use generated files above \
  --analysis=SoftFullTreeOODPrior --ood-dataset=CIFAR10IncludeClasses --ood-classes "cat" "truck" # include OOD classes here
  --ood-path-wnids=... # path to wnids.txt for the OOD classes listed in line above \
```

Here, the `wnids.txt` file for the OOD file, which should contain `wnid`s for `cat` and `truck`, looks like:
```
n02121620
n04490091
```
The format is exactly the same as the `wnids.txt` for the original dataset, except you only keep wnids for the remaining OOD classes. No need to change the `wnids.txt` for the original dataset.

## Comparing ZS embeddings to Seen embeddings
This will compare the image embeddings of the seen classes to the image embeddings of the ZS classes

Example with AwA2(same args as `add_zeroshot_vec.py` but you dont need the `--zeroshot-dataset` flag):
```
python get_image_centers.py --model resnet50 --pretrained --dataset AnimalsWithAttributes2 --path-resume ./checkpoint/ckpt-awa2-resnet50-freeze.pth --resume --new-labels 5 13 14 17 23 24 33 38 41 47 --replace
```
