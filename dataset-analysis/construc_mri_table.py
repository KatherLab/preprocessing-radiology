#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import pandas as pd
clini_table = '/mnt/sda1/swarm-learning/radiology-dataset/Clinical_and_Other_Features.xlsx'
df = pd.read_excel(clini_table, header=[0, 1, 2])
df = df[df[df.columns[38]] != "NC"] # use unilateral and bilateral tumors
df = df[[df.columns[0], df.columns[36], df.columns[38]]]  # Only pick relevant columns: Patient ID, Tumor Side
df.columns = ['PatientID', 'Location', 'Bilateral Information']  # Simplify columns as: Patient ID, Tumor Side
dfs = []
for side in ["left", 'right']:
    dfs.append(pd.DataFrame({
        'PatientID': df["PatientID"] + f"_{side}",
        # if df[df.columns[38]] is 1 (bilateral tumor), then Malign = 1, else Malign = 0
        #'Malign': df["Location"].apply(
        #    lambda x: 1 if ((x == side[0].upper())) else 0),
        'Malign': df["Location"].apply(lambda x: 1 if ((x == side[0].upper())) else 0) | df["Bilateral Information"].apply(
            lambda x: 1 if x==1 else 0),

        }))
df = pd.concat(dfs, ignore_index=True).set_index('PatientID', drop=True)

df.to_csv('clinical_table.csv')