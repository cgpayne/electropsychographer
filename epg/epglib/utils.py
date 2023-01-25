#!/usr/bin/env python
'''
utils.py = holds utility functions for electropsychographer
  import epglib.utils as ut
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  these utility functions are shared throughout the electropsychographer code
NOTES
  [none]
RESOURCES
  [none]
CONVENTIONS
  [none]
KNOWN BUGS
  [none]
DESIRED FEATURES
  [none]
'''

# externel imports
import sys
import time
import datetime
import os


def eprint(*args, **kwargs):
    '''
    FUNCTION: eprint = print to stderr
          IN: << same as print >>
         OUT: << print to stderr >>
    '''
    sys.stdout.flush()  # so stederr doesn't redirect to top of file with 2>&1
    print(*args, file=sys.stderr, **kwargs)


def time_stamp(t_del: float, t_zero: float, stamp: str) -> None:
    '''
    FUNCTION: time_stamp = create a time stamp within the code
          IN: t_del = the time of the last time point
              t_zero = the 0th time point
              stamp = a string identifier for the stamp itself
         OUT: << print out a time stamp to screen >>
    '''
    t_now = time.time()
    t_abs = t_now - t_zero
    t_rel = t_now - t_del
    
    time_front = f"\n---< --<< -<<< TIME STAMP: {stamp} -<>- relative: {str(datetime.timedelta(seconds=t_rel))[:-3]}"
    time_back = f"-<>- absolute: {str(datetime.timedelta(seconds=t_abs))[:-3]} >>>- >>-- >---\n"
    print(time_front + " " + time_back)
    
    return t_now


def make_dir(path_to_dir: str) -> None:
    '''
    FUNCTION: make_dir = make a directory (recursively) if it doesn't already exist
    '''
    if not os.path.isdir(path_to_dir):
        os.makedirs(path_to_dir)
