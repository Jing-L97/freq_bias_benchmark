"""explore checkpoints"""

import os
import pandas as pd
from tqdm import tqdm

import os

# Directory containing the files
directory = "/path/to/directory"

# Dictionary to store results
result_dict = {}

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pt"):
        # Split filename into parts using "_" as separator
        parts = filename.split("_")
        if len(parts) >= 3:
            # Extract number1 and number2 from the filename
            number1 = int(parts[1])
            number2 = int(parts[2].split(".")[0])  # Remove ".pt" extension

            # Update dictionary
            if number1 in result_dict:
                result_dict[number1][0] = max(result_dict[number1][0], number2)
                result_dict[number1][1] += 1
            else:
                result_dict[number1] = [number2, 1]

print(result_dict)




