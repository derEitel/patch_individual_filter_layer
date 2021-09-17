##########
contribute
##########


******
coding
******

numpy coding style!
useful links:

* https://numpydoc.readthedocs.io/en/latest/format.html

* https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

our standard:

* init functions will be explained in CLASS docstring!

* private functions begin with "_". Their docstring should not be included in Documentation

* private attributes also begin wih "_". Their docstring should not be included in Documentation

* be strict !!!

*************
documentation
*************

documentation via sphinx:

* :code:`pip install sphinx`

* :code:`pip install sphinxcontrib-napoleon`

* :code:`pip install sphinx-rtd-theme`

* :code:`cd /pathToNitorch/docs`

* :code:`sphinx-apidoc -f -o source/ /pathToNitorch/`

configuration:

* set in source/conf.py: extensions = ['sphinxcontrib.napoleon', "sphinx_rtd_theme" ]

* set in source/conf.py: html_theme = "sphinx_rtd_theme"

useful links:

* https://www.sphinx-doc.org/en/1.5/markup/toctree.html#id3

* http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html

* https://sphinx-rtd-theme.readthedocs.io/en/latest/

****
PLAN
****

documentation should be hosted (once finished) on readthedocs:

https://docs.readthedocs.io/en/stable/index.html



