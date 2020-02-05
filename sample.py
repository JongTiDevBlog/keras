import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_data = pd.read_csv('samsung1.csv', index_col=0, encoding='ANSI')
df_data.head()