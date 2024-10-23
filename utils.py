import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def patched_read_parquet(*args, **kwargs):
    kwargs['engine'] = 'pyarrow'
    return _read_parquet(*args, **kwargs)


def viz_distro(df, col, label):
    sns.histplot(df[df[col] > 0][col], kde=True, label = label)