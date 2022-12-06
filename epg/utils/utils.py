#!/usr/bin/env python
#  utils.py = holds utility functions for electropsychographer
#  import utils.utils as ut
#  By: Charlie Payne
#  Licence: n/a
# DESCRIPTION
#  these utility functions are shared throughout the electropsychographer code
# NOTES
#  [none]
# CONVENTIONS
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

import time
import datetime


def time_stamp(t_del, t_zero, stamp):
    t_now = time.time()
    t_abs = t_now - t_zero
    t_rel = t_now - t_del
    
    time_front = f"\n---< --<< -<<< TIME STAMP: {stamp} -<>- relative: {str(datetime.timedelta(seconds=t_rel))[:-3]}"
    time_back = f"-<>- absolute: {str(datetime.timedelta(seconds=t_abs))[:-3]} >>>- >>-- >---\n"
    print(time_front + " " + time_back)
    
    return t_now
