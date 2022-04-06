import numpy as np
import os
import sys

class DataLoader:
    def __init__(self, path) -> None:
        
        self.path = path
        self.data = []
        self.labels = []


    def load_data(self):
        