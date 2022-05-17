import os
import csv
import json

ROOT_DIR = 'MIDV-2020-Text'

SUB_DIRS = os.listdir(ROOT_DIR)

data = []

# Read the JSON files
for sd in SUB_DIRS:
    sd_path = os.path.join(ROOT_DIR, sd)
    files = os.listdir(sd_path)
    for file in files:
        # Read the current JSON file
        f = open(os.path.join(sd_path, file), encoding='utf-8')
        d = json.load(f)

        # Add the key-value pairs to a list
        data += list(d.items())

        # Close the JSON file
        f.close()

# Combine list of key-value pairs to one CSV file
with open(os.path.join(ROOT_DIR, ROOT_DIR + '.csv'), 'w', encoding='utf-8', newline='') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['category', 'text'])
    csv_out.writerows(data)
