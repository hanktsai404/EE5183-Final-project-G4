#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:46:54 2020
@author: austinhsu
"""

import time
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from contextlib import contextmanager

class SimpleProfiler:
    """Profiler code from PyTorchLightning"""
    def __init__(self, output_filename: str = None):
        self.output_filename = output_filename
        self.output_file = open(output_filename, 'w') if output_filename else None
        self.write_streams = [self.output_file.write] if self.output_file is not None else [print]
        self.current_action = {}
        self.recorded_durations = defaultdict(list)
    
    def start(self, action_name: str):
        if action_name in self.current_action:
            raise ValueError(f"Action name {action_name} is already started.")
        self.current_action[action_name] = time.monotonic()
        
    def stop(self, action_name: str):
        end_time = time.monotonic()
        if action_name not in self.current_action:
            raise ValueError(f"Action name {action_name} was never started.")
        start_time = self.current_action.pop(action_name)
        duration = end_time-start_time
        self.recorded_durations[action_name].append(duration)
        
    @contextmanager
    def profile(self, action_name: str):
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)
        
    def profile_iterable(self, iterable, action_name: str):
        iterable = iter(iterable)
        while True:
            try:
                self.start(action_name)
                value = next(iterable)
                self.stop(action_name)
                yield value
            except StopIteration:
                self.stop(action_name)
                break
    
    def summary(self):
        output_table = OrderedDict({"Action": [], "Mean Duration (s)": [], "Total Time (s)": []})
        for actions, durations in self.recorded_durations.items():
            output_table["Action"].append(actions)
            output_table["Mean Duration (s)"].append(f"{np.mean(durations):<.8f}")
            output_table["Total Time (s)"].append(f"{np.sum(durations):<.8f}")
        output_table = pd.DataFrame(output_table)
        return str(output_table)
        
    def describe(self):
        for write in self.write_streams:
            write(self.summary())
        if self.output_file is not None:
            self.output_file.flush()