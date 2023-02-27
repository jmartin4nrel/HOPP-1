import csv
import os
from pathlib import Path

# Stick this in the same directory as the "_all.csv" files to split in to resource and generation files
# WARNING: NEEDS ~10 GB OF DISK SPACE!

cd = Path(__file__).parent.absolute()
files = os.listdir(cd)

for fn in files:
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        gen_fn = fn[:-8]+'_gen.csv'
        with open(gen_fn, 'w', newline='') as gen_file:
            gen_writer = csv.writer(gen_file)
            res_fn = fn[:-8]+'_res.csv'
            with open(res_fn, 'w', newline='') as res_file:
                res_writer = csv.writer(res_file)
                for line in reader:
                    gen_writer.writerow(line[:96])
                    res_writer.writerow(line[96:])