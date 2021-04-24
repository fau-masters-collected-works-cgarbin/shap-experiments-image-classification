# Experiments with SHAP and image classification


This repository explores how to interpret predictions of an image classification neural network using [SHAP](https://arxiv.org/abs/1705.07874).

The goals of the experiments are to:

1. Explore how SHAP explain the predictions. This experiment uses a (fairly) accurate network to understand how SHAP attributes the predictions.
1. Explore how SHAP behaves with innacurate predictions. This experiment uses a network with lower accuracy and prediction probabilities that are less robust (more spread among the classes) to understand how SHAP behaves when the predicitons are not reliable (a hat tip to [Dr. Rudin's work](https://arxiv.org/abs/1811.10154)).


SHAP has multiple explainers. The code uses the DeepExplainer explainer because it is the one used in the [image classification SHAP sample code](https://shap.readthedocs.io/en/latest/image_examples.html).

The code is based on the [SHAP MNIST example](https://shap.readthedocs.io/en/stable/example_notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.html), available as notebook [on GitHub](https://github.com/slundberg/shap/blob/master/notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.ipynb). This notebook uses the PyTorch sample code because at this time (April 2021), SHAP does not support TensorFlow 2.0. [This GitHub issue](https://github.com/slundberg/shap/issues/850) tracks the work to support TensorFlow 2.0 in SHAP.

The code for the experiments is [on this Jupyter notebook](https://github.com/fau-masters-collected-works-cgarbin/shap-experiments-image-classification/blob/master/shap-experiments-image-classification.ipynb).

## Running the code

To run the code, configure the Python environment as described below, then open the Jupyter notebook.

### Setting up the Python environment

#### Installing Python 3

The project uses Python 3.

Verify that you have Python 3.x installed: `python --version` should print `Python 3.x.y`. If
it prints `Python 2.x.y`, try `python3 --version`. If that still doesn't work, please install
Python 3.x before proceeding. The official Python download site is
[here](https://www.python.org/downloads/).

From this point on, the instructions assume that **Python 3 is installed as `python3`**.

#### Cloning the repository

```bash
git clone https://github.com/fau-masters-collected-works-cgarbin/shap-experiments-image-classification.git
```

The repository is now in the directory `shap-experiments-image-classification`.

#### Creating a Python virtual environment

Execute these commands to create and activate a [virtual environment]((https://docs.python.org/3/tutorial/venv.html)) for the project:

```bash
#  switch to the directory where the cloned repository is
cd machine-learning-but-not-understanding

python3 -m venv env
source env/bin/activate
# or in Windows: env\Scripts\activate.bat
```

#### Installing the dependencies

It's important to update `pip` first. Older `pip` versions may fail to install some components.

```bash
pip install --upgrade pip
pip install -r requirements.txt`
```

#### Running the notebook

`jupyter lab shap-experiments-image-classification.ipynb`
