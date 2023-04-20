#%%
# package install
import pandas as pd
import numpy as np
import matplotlib as mlt
import seaborn as sns

# %%
# Load Stroke Preciction Dataset
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke.head()
# %%
# I am focusing on the variables age, heart disease, and married

