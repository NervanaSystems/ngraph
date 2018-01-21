.. model-phases:


Phases 
======

With the optimizations built into Intel nGraph library core, you can 
train a model and quickly iterate upon (or with) learnings from your 
original dataset. Once the model's data has been trained with the nGraph 
library, it is essentially "freed" from the original framework that you 
wrangled it into, and you can apply different kinds of operations and 
tests to further refine to the goals of your data science.  

For example, let's say that you notice the `MNIST` MLP dataset running
with MXNet on nGraph trains itself to  0.997345 or 1.00000 accuracy after 
only 10 Epochs. The original model was written to train the dataset for 
20 Epochs. This means that there are potentially 10 wasted cycles of 
compute power that can be used elsewhere.  

