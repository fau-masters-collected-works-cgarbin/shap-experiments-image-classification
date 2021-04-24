# Running the code

To run the code, configure the Python environment as described below, then open the Jupyter notebook.

## Installing Python 3

The project uses Python 3.

Verify that you have Python 3.x installed: `python --version` should print `Python 3.x.y`. If
it prints `Python 2.x.y`, try `python3 --version`. If that still doesn't work, please install
Python 3.x before proceeding. The official Python download site is
[here](https://www.python.org/downloads/).

From this point on, the instructions assume that **Python 3 is installed as `python3`**.

## Cloning the repository

```bash
git clone https://github.com/fau-masters-collected-works-cgarbin/shap-experiments-image-classification.git
```

The repository is now in the directory `shap-experiments-image-classification`.

## Creating a Python virtual environment

Execute these commands to create and activate a [virtual environment]((https://docs.python.org/3/tutorial/venv.html)) for the project:

```bash
#  switch to the directory where the cloned repository is
cd machine-learning-but-not-understanding

python3 -m venv env
source env/bin/activate
# or in Windows: env\Scripts\activate.bat
```

## Installing the dependencies

It's important to update `pip` first. Older `pip` versions may fail to install some components.

```bash
pip install --upgrade pip
pip install -r requirements.txt`
```

## Running the notebook

`jupyter lab shap-experiments-image-classification.ipynb`
