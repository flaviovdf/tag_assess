'''
This module is a reflection of some of the tagasses code in Cython. This
will be useful for speed-ups.

The code should resemble regular python code, but adding static types. 
The notes bellow can help with some special cases.

Notes: 
 (1) cpdef makes the function a C function with the ability of it being called in
 python. 
 (2) cdef is a C function which can only be called by others (defined by cdef or cpdef).
 (3) normal functions are defined as def.
 (4) Py_ssize_t must be used when accessing arrays: Py_ssize_t i = 0; array[i]. This is the platform
     independent type which determines valid array indexes.
 (5) The decorators @cython.boundscheck(False) and @cython.wraparound(False) speed up
     methods dealing with arrays by disabling out-of-bounds and negative indices checks.
 (6) .pxd files are analogous to .h files.
'''