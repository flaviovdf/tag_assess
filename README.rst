Implementation of the tag value assessment described on:

http://dx.doi.org/10.1109/SocialCom.2010.69

(this document is initially a stub. It will be incremented together with the code)


Dependencies
============

* numpy
* scipy
* cython
* pytables (depends on cython, pyrex and numexpr)
* networkx (optional for some graph experiments)
* pymongo (optional, alternative data storage to pytables)
* nosetests (used to run unit tests)

only numpy, scipy and cython are needed to use the probability estimators and value calculators.
the other dependencies are mostly used by scripts, but I suggest installing pytables.

How to install dependencies (ubuntu)
====================================

The part of the guide will assume you use ubuntu or debian. This is only needed to install virtualenv and python dev
packages. Both can be installed from source or use respective packages for your OS. 

* Make sure you have python 2.6+ (but not Python 3+).
* Install the required packages

::

$ apt-get install python-virtualenv

* You will also need the python-dev package to build numpy

::

$ apt-get install python2.7-dev

* Create your virtual env. From here root will no longer be required.

::

$ virtualenv --no-site-packages ~/virtualenvs/tagassess/ #or any folder you wish

* Activate it

::

$ source ~/virtualenvs/tagassess/bin/activate

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
$ pip install Pyrex
$ pip install numexpr
$ pip install pytables

How to install dependencies using Canopy
========================================

Another option is to use the very good Enthought Canopy python distribution (free for academics). 
It is a pre-built python with numpy and scipy (plus lots of other goodies) out of the box. 

* Download and install Canopy. See https://www.enthought.com/
* After installing Canopy activate the virtualenv (at install time you can choose for it to be activated always)

::

$ source ~/Enthought/Canopy_64bit/System/bin/activate

* PROTIP: The cython in Canopy package manager is old. I recommend using  easy_install or pip to install new versions.
  Since pip does not play well with Canopy, use easy_install.

* Install cython on Canopy

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

* Make sure you are in you previously configured python environment
* If you don't want to install you can just build using make. Just run make in the folder
* From here you can run unittests

:: 

$ nosetests

* And use the pyrun.sh script to use the package withou installing. Useful when developing

:: 

$ ./pyrun.sh

* If you want to install and be happy. Go to code folder and run

:: 

$ python setup.py install

* If you don't install pytables, networkx or pymongo (with mongodb) some tests will fail. I need to add ignore
  flags on them if the packages are not installed. These failures can be ignored, the affected packages are 
  tagassess.dao and tagassess.graph. *Other tests should not fail!!!*
  
How to create databases
=======================

* Use the ParseFile.py script in the src/scripts/ folder. We use the py_run.sh script, but if you installed
  the package you can just run python.

::

$ ./py_run.sh src/scripts/ParseFile.py ~/LT/UI2.txt ~/db/db.h5 ~/db lt

* The line above will parser the UI2.txt file (from the LibraryThing trace) and create a pytables database
  on the ~/db folder. The last argument lt is just to indicate that this file is in LibraryThing format.

* A database with the name lt will be created on the h5 file. Multiple databases can exist on the same file

* You can run the script with no options to see other ways to parse data

* The LibraryThing dataset is available here http://www.macle.nl/tud/LT/

How to separate data in train, validation and test
==================================================

* Use the CreateTrainValTest.py script in the src/scripts/ folder. We use the py_run.sh script, but if you 
  installed the package you can just run python.

::

$ ./py_run.sh src/scripts/CreateTrainValTest.py ~/db/db.h5 lt ~/db/lt-cv/

* The first argument is the database file, the second is the database name and the last the output folder

* This will create files needed to identify train, validation and test sets on the folder ~/db/lt-cv. The 
  folder must exist and be empty
  
How to perform a grid search for best estimator parameters
==========================================================

* Use the GridSearch.py script to achieve this. We use the py_run.sh script, but if you installed
  the package you can just run python.

:: 

$ ./py_run.sh src/scripts/GridSearch.py ~/db/db.h5 lt ~/db/lt-cv/ ~/db/lt-cv/smooth/ -est-name smooth

or 

:: 

$ ./py_run.sh src/scripts/GridSearch.py ~/db/db.h5 lt ~/db/lt-cv/ ~/db/lt-cv/lda/ -est-name lda

* NOTE!! This script creates a lot of files as output. Make sure you have enough space. We are still
  looking into changing them for better storage options

* The script will compute probability values for different parameters of the estimator and save them
  in a folder. For example, the first line will use the bayes smooth estimator and save output on the
  ~/db/lt-cv/smmoth/ folder
