Implementation of the tag value assessment described on:

http://dx.doi.org/10.1109/SocialCom.2010.69

(this document is initially a stub. It will be incremented together with the code)


Dependencies
============

* numpy
* scipy
* cython
* igraph (optional)
* networkx (optional for some graph experiments)
* pytables (optional: depends on cython, pyrex and numexpr)
* pymongo (optional)
* nosetests (used to run unit tests)

only numpy, scipy and cython are needed to use the probability estimators and value calculators.
the other dependencies are mostly used by scripts, but I suggest installing pytables and networkx.

pymongo and pytables are used to store tags in databases. You can ignore this and store them in files if you wish.

How to install dependencies (ubuntu)
====================================

The part of the guide will assume you use ubuntu or debian. This is only needed to install virtualenv and python dev
packages. Both can be installed from source or use respective packages for your OS. 

Another option is to use the very good Enthought Canopy python distribution (free for academics). 
It is a pre-built python with numpy, scipy and cython (plus lots of other goodies) out of the box 
(cython may be outdated, but you can updat just it). You can probably just install Canopy, activate it and
do a $ pip install Cython and be ready!
See https://www.enthought.com/downloads/

* Make sure you have python 2.6+ (but not Python 3+).
* Install the dependencies. I prefer to use virutalenv for this

::

$ apt-get install python-virtualenv

* You will also need the python-dev package to build numpy

::

$ apt-get install python2.7-dev

* Create your virtual env. From here everythun will be local.

::

$ virtualenv --no-site-packages ~/virtualenvs/tagassess/ #or any folder you wish

* Activate it

::

$ source source ~/virtualenvs/tagassess/bin/activate

* Install deps in the virtualenv

::

$ pip install numpy
$ pip install scipy
$ pip install cython

* Install nose to run unit tests more easily

::

$ pip install nose

* I also suggest having networkx and pytables

::

$ pip install networkx
$ pip install pyrex
$ pip install numexpr
$ pip install pytables

How to install dependencies using Canopy
========================================

Another option is to use the very good Enthought Canopy python distribution (free for academics). 
It is a pre-built python with numpy and scipy (plus lots of other goodies) out of the box. 

* Download and install canopy. See https://www.enthought.com/
* After installing Canopy activate the virtualenv (or you can at install choose for it to be activated always)

::

$ source ~/Enthought/Canopy_64bit/System/bin/activate

* PROTIP: The cython in canopy package manager is old. I recommend using  easy_install or pip to install new versions.

* Instal cython on canopy. pip does not play nice with Canopy, so use easy_install. Basically the same thing.

::

$ easy_install Cython

* I also suggest having networkx and pytables. Now using enpkg which is how Canopy manages packages. You can also 
  use the GUI. Or easy_install them for newest versions (not required)

::

$ enpkg networkx
$ enpkg tables


How to install the package
==========================

* Clone the repo

::

$ git clone https://github.com/flaviovdf/tag_assess.git

* Make sure you are in you previously configured python enviroment
* Install and be happy. Go to code folder and o

:: 

$ python setup.py install

* If you don't want to instal you can just build using make. Just run make in the folder
* From here you can run unittests

:: 

$ nosetests

* If you don't install pytables, networkx or pymongo (with mongodb) some tests will fail. I need to ignore
  them still if the enviroment is not good for them. These tests can be ignored. They are the one in the 
  tagassess.dao and tagassess.graph packages. *Other tests should not fail!!!*

* And use the pyrun.sh script to use the package withou installing. Useful when developint

:: 

$ ./pyrun.sh
