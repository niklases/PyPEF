This repository contains supplementary information to the paper

Niklas E. Siedhoff<sup>*§*</sup>, Alexander-Maurice Illig<sup>*§*</sup>, Ulrich Schwaneberg, Mehdi D. Davari, PyPEF – An Integrated Framework for Data-driven Protein Engineering, 2020 

<sup>*§*</sup>equal contribution

# PyPEF: Pythonic Protein Engineering Framework

a framework written in Python 3 for performing sequence-based machine learning-assisted protein engineering.

Protein engineering by rational or random approaches generates data
that can aid the construction of self-learned sequence-function
landscapes to predict beneficial variants by using probabilistic methods that can screen the unexplored sequence space with uncertainty *in silico* .
Such predictive methods can be applied for increasing the success/effectivity of an
engineering campaign while partly offering the prospect to reveal hidden patterns or
complex networks of epistatic effects. Here we present an engineering framework termed
PyPEF for assisting the tuning and validation of models
for combination of identified substitutions using machine learning algorithms (partial least squares (PLS) regression)
from the [scikit-learn](https://github.com/scikit-learn/scikit-learn) package.
As training input, the developed software tool requires the sequence and 
the corresponding screening results (fitness labels) of the
identified variants as .csv (or .fasta datasets). Using PLS regression, PyPEF trains
on the given learning data and validates model performances on left-out data.
Finally, the selected or best model for validation can be
used to perform directed evolution walks *in silico* (see [Church-lab implementation](https://github.com/churchlab/UniRep) or the [reimplementation](https://github.com/ivanjayapurna/low-n-protein-engineering)) or to predict natural diverse or recombinant sequences that
subsequently are to be designed and validated in the wet-lab.


Detailed information is given in the following publication, PyPEF - an Integrated Framework for Data-driven Protein Engineering, and the
workflow procedure is explained in the [Jupyter notebook](/workflow/Workflow_PyPEF.ipynb) (.ipynb) protocol (see
./workflow directory).  

 

Before starting running the tutorial, it is a good idea to set-up a new Python environment using Anaconda, https://www.anaconda.com/, e.g. using [Anaconda](https://www.anaconda.com/products/individual) ([sh installer](https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh)) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Change to the download directory and run the installation, e.g. in Linux:

`bash Anaconda3-2020.07-Linux-x86_64.sh`  .

After accepting all steps, the conda setup should also be written to your `~/.bashrc`file, so that you can call anaconda typing `conda`.
To setup a new environment you just need to specify the name of the environment and the Python version, e.g.

`conda create --name pypef python=3.7`  .

To activate the envirionment you can define

`conda activate pypef`  .

After activating the environment you can install required packages after changing the directory to this PyPEF directory and install required
packages with pip (or conda itself:

`python3 -m pip install -r requirements.txt`

and optionally

`python3 -m pip install -r requirements_parallelization.txt`  .

Next, install Jupyter Notebook:

`python3 -m pip install notebook`  .

Now you should be able to directly start with the tuorial. Change directory to ./workflow and run the .ipynb file:

`jupyter-notebook Workflow_PyPEF.ipynb`  .

Good luck and have fun!
