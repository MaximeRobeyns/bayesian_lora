Bayesian LoRA
=============

This repository contains:

- an implementation of K-FAC
- Bayesian LoRA for language models.

.. .. todo:: Add a diagram of the components
.. .. figure:: _static/ml_overview.png

Installation Guide
------------------

The simplest way to use the library is to simply pip install it::

    pip install bayesian-lora

Editable Installation
^^^^^^^^^^^^^^^^^^^^^

If you would like to modify the library or build upon it, while keeping it as a
separate library, then you can clone the repo and run an editable installation::

    git clone https://github.com/MaximeRobeyns/bayesian_lora
    cd bayesian_lora
    pip install -e .

Hackable Installation
^^^^^^^^^^^^^^^^^^^^^

The library is currently very small, and has three core dependencies, ``torch``
``tqdm``, and ``jaxtyping``; and two main files.

To this end, feel free to directly copy the file you need into your own project
and start hacking on it.

Installation with Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^

There are some examples included with the GitHub repository. Before running the
code in these files, you must install some additional dependencies which are
omitted from the main library to keep it small. To do this, after cloning the
repo, from the root simply run::

    pip install -e ".[examples]"

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

If you plan on developing on this library, you may wish to install some
development-related packages with ``pip install -e ".[dev]"``. To write
documentation, install the requirements with ``pip install -e ".[docs]"``.

For simplicity, you can also just run::

    pip install -e ".[all]"

Jupyter Notebooks
`````````````````

To test functions from a Jupyter notebook, make sure that you have installed the
project with the ``dev`` dependencies. You then need to run the following
command once to set up the iPython kernel::

    make kernel

You only need to do this once. After you do so, you will see a new
``bayesian_lora`` kernel inside jupyterlab. To launch jupyterlab, we include a
convenience target::

    make lab

Contents
--------

.. toctree::
   :maxdepth: 2
   :glob:
   :caption: Contents:

   kfac
   bayesian_lora
   example_usage

..
    Indices and tables
    ------------------

    * :ref:`genindex`
    * :ref:`modindex`
