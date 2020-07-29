# NBDT



## Analysis
There are many analysis classes that are quite useful for evaluation:
* Noop: normal evaluation
* SoftEmbeddedDecisionRules/HardEmbeddedDecisionRules: NBDT tree traversal analysis
* SoftFullTreePrior/HardFullTreePrior: Tracks the path of each class down the tree and displays them in the html formal
* SoftFullTreeOODPrior: same as above but takes in classes out of distribution
* SingleRISE/SingleGradCAM: Computes GradCAM/RISE saliency maps of a single image given a model and hierarchy
