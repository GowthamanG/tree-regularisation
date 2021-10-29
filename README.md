# tree-regularisation
This repository contains the code base of my Master's Thesis "Tree Regularization of Deep Networks." Based on the publication
<a>[1]</a>, the aim is to replicate the tree regularization method for feed-forward deep networks, which can be 
then approximated by simple decision trees. The idea is, to better simulate the network's decision-making via decision trees. 
Tree-regularized models belong to *model-specific* interpretable models.

## Conda Environment
We strongly recommend to work and run the code within a conda environment. Use the environment.yml file, to create the 
environment, which includes the Python interpreter version 3.8.8, PyTorch, scikit-learn and some other dependencies.

First create the environment.
```
conda env create -f environment.yml
````

Everytime you work with this code, activate the environment.

```
conda activate tree-regularisation
````

## Data sets
This code base contains implementation to generate synthetic, two-dimensional data sets. The "Parabola data set" would have
data instances in the 2D space [0, 1.5] x [0, 1.5], and uses a parabola function as decision function to separate the data
into classes. The "Cosine data set", uses the cosine function in the space [-6, 6] x [-2, 2]. Run the following command to
generate the data sets.

```
python datasets.py --sample <data set> --sample_size <size> --path <path>
````

For `--sample` you can either enter `parabola` or `cosine`, for `--size` any sample size you want, we recommend a 
large number for that, and for `--path` use the directory path `dataset\parabola` or `dataset\cosine`. This repository already
contains samples, the parabola set with 20'000 samples, and the cosine set with 35'000 samples.

If you run the script it will open two plots, a scatter plot with data points, and a plot with the error zone, where a noise
was added. These plots shows how densely the data points are distributed.

## Training
To execute the script for training, run the following command:

```
python train.py --label <label> --lambda_init <initial lambda> --lambda_target <target lambda> --ep <total number of epochs>
--min_samples_leaf <minimal number of sample per leaf> --batch <batch size>
````

The parameters already contain default values, run the script with parameter `-h` to see the description:

```
(tree-regularisation) D:\Gowthaman\Projects\Python\tree-regularisation>python train.py -h
usage: train.py [-h] [--label LABEL] [--lambda_init LAMBDA_INIT] [--lambda_target LAMBDA_TARGET] [--ep EP] [--min_samples_leaf MIN_SAMPLES_LEAF] [--batch BATCH]

optional arguments:
  -h, --help            show this help message and exit
  --label LABEL         Additional label as postfix to the directory path name to indicate this run
  --lambda_init LAMBDA_INIT
                        Initial lambda value as regularisation term
  --lambda_target LAMBDA_TARGET
                        Target lambda value as regularisation term
  --ep EP               Total number of epochs, default 1000 (300 warm up + 700 regularisation)
  --min_samples_leaf MIN_SAMPLES_LEAF
                        Minimum samples leaf for pre-pruning, default 5
  --batch BATCH         Batch size, default 1024

````

By default, the training is provided with the parabola data set. To work with the cosine data set, change the variable 
`fun = parabola`, `fun_name = 'parabola'` and `space = [[0,1.5],[0,1.5]]` to `fun = cos`, `fun_name = 'cos` and 
`space = [[-6,6],[-2,2]]`, in the `__main__` module (at the bottom of the script).

## Tensorboard
The training outcomes like the loss, accuracy and some plots, can be visualized in the tensorboard. Run the following command
in a separate terminal:

``
tensorboard --logdir=runs
``

To learn more about tensorboard, checkout https://pytorch.org/docs/stable/tensorboard.html.

## References
<a id="1">[1]</a>
M. Wu, M. Hughes, S. Parbhoo, M. Zazzi, V. Roth, and F. Doshi-Velez, “Beyond sparsity: Tree regularization of deep models 
for interpretability,” inAAAI, 2018.

<a id="1">[2]</a>
M. Wu, S. Parbhoo, M. C. Hughes, V. Roth, and F. Doshi-Velez, “Optimizing for Interpretability in DeepNeural Networks with 
Tree Regularization,”arXiv e-prints, p. arXiv:1908.05254, Aug. 2019.

<a id="1">[3]</a>
M. Wu, S. Parbhoo, M. Hughes, R. Kindle, L. Celi, M. Zazzi, V. Roth, and F. Doshi-Velez, “Regional Tree Regularization for 
Interpretability in Black Box Models,”arXiv e-prints, p. arXiv:1908.04494, Aug.2019.


