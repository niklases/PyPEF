Niklas E. Siedhoff, Alexander-Maurice Illig, Ulrich Schwaneberg, Mehdi D. Davari, PyPEF – An Integrated Framework for Data-driven Protein Engineering, 2020  

# PyPEF
Pythonic Protein Engineering Framework (PyPEF);
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
/workflow directory).  

