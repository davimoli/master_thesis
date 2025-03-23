# M. David Synthetic data generation March 2025
import pybamm
import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time for measuring execution time
import re  # Import re for regular expressions
import pickle
with open('1C_298K.pkl', 'rb') as f:
    data = pickle.load(f)
# Record the start time for the entire script
start_time_total = time.perf_counter()

# Print PyBaMM version
print(f"PyBaMM version: {pybamm.__version__}")


ma_sim = pb.load('/Users/david/PycharmProjects/memoire/Synthetic_data/1C_298K.pkl')