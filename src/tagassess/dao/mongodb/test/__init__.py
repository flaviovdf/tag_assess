'''Tests for the mongodb package'''

from pymongo import Connection
from pymongo.errors import PyMongoError

import os
import random
import shutil
import string
import subprocess
import tempfile

class MongoManagerException(Exception):
    '''Signals errors when dealing with mongo process'''
    pass

class MongoManager(object):
    '''Class for starting and stopping mongodb'''
    
    def __new__(cls, *args, **kwargs):
        if not '_the_instance' in cls.__dict__:
            cls._the_instance = object.__new__(cls, *args, **kwargs)
        return cls._the_instance
    
    def __init__(self, exe='mongod'):
        self.exe = exe
        self.mongo_proc = None
        self.connection = None
        self.tmp_dir = None
        self.out_file = None
        self.err_file = None
        self.col_names = set()

    def started(self):
        '''Returns `True` if started'''
        return self.connection is not None

    def start_mongo(self):
        '''Starts the mongo daemon'''
        if not self.mongo_proc:
            self.tmp_dir = tempfile.mkdtemp()
            self.out_file = open(os.path.join(self.tmp_dir, 'out'), 'w')
            self.err_file = open(os.path.join(self.tmp_dir, 'err'), 'w')
            self.mongo_proc = subprocess.Popen([self.exe, 
                                                '--dbpath', 
                                                self.tmp_dir],
                                                stdout=self.out_file,
                                                stderr=self.err_file)
            
            while self.connection is None:
                try:
                    #TODO: Maybe change this to be configurable
                    self.connection = Connection()
                except PyMongoError:
                    self.connection = None
        else:
            raise MongoManagerException('Already running')
    
    def stop_mongo(self):
        '''Stops the mongo daemon'''
        if self.mongo_proc:
            self.connection.disconnect()
            self.mongo_proc.terminate()
            self.mongo_proc.wait()
            self.mongo_proc = None
            self.connection = None
            self.col_names = set()
            
            self.out_file.close()
            self.err_file.close()
            shutil.rmtree(self.tmp_dir)
            
            self.out_file = None
            self.err_file = None
            self.tmp_dir = None
        else:
            raise MongoManagerException('Not running')
    
    def __del__(self):
        self.stop_mongo()