#!/usr/bin/env python
# -*- coding: utf-8

'''Setup script'''

import glob
import os
import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

SOURCE = 'src/'
os.chdir(SOURCE)

if sys.version_info[:2] < (2, 7):
    print('Requires Python version 2.7 or later (%d.%d detected).' %
          sys.version_info[:2])
    sys.exit(-1)

def get_packages():
    '''Appends all packages (based on recursive sub dirs)'''

    packages  = ['tagassess',
                 'cy_tagassess']

    for package in packages:
        base = os.path.join(package, '**/')
        sub_dirs = glob.glob(base)
        while len(sub_dirs) != 0:
            for sub_dir in sub_dirs:
                package_name = sub_dir.replace('/', '.')
                if package_name.endswith('.'):
                    package_name = package_name[:-1]

                packages.append(package_name)
        
            base = os.path.join(base, '**/')
            sub_dirs = glob.glob(base)

    return packages

def get_extensions():
    '''Get's all .pyx and.pxd files'''
    
    base = 'cy_tagassess'
    extensions = []

    pyx_files = glob.glob(os.path.join(base, '*.pyx'))
    for pyx in pyx_files:
        
        pxd = pyx.replace('pyx', 'pxd')
        module = pyx.replace('.pyx', '').replace('/', '.')
        
        if os.path.exists(pxd):
            extensions.append(Extension(module, [pyx, pxd]))
        else:
            extensions.append(Extension(module, [pyx]))
    
    return extensions

if __name__ == "__main__":
    packages = get_packages()
    extensions = get_extensions()
    
    setup(
        cmdclass = {'build_ext': build_ext},
        name             = 'tagassess',
        packages         = packages,
        ext_modules      = extensions
      )
