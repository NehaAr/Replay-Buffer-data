from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp

def standardize(data):
  data.columns=data.columns.str.strip()
  return data

def moments(data,data_syn):
  print(data.shape,data_syn.shape)
  data=standardize(data)
  data_syn=standardize(data_syn)
  numeric_data_o = data.select_dtypes(include=[np.number])
  numeric_data = data_syn.select_dtypes(include=[np.number])
  original=stats.kurtosis(numeric_data_o,fisher=True)
  synthetic=stats.kurtosis(numeric_data,fisher=True)
  original_mean = np.nanmean(original)
  synthetic_mean = np.nanmean(synthetic)

  print(f"Original kurtosis  : {original_mean:.3f}")
  print(f"Synthetic kurtosis : {synthetic_mean:.3f}")
  print(f"Δ kurtosis         : {synthetic_mean - original_mean:.3f}")

  mean_orig = numeric_data_o.mean()
  mean_syn  = numeric_data.mean()

  mean_diff = (mean_syn - mean_orig).abs() / mean_orig.abs()
  mean_preserved = mean_diff.median()
  print(f"Mean preserved     : {mean_preserved:.3f}")

  std_orig = numeric_data_o.std()
  std_syn  = numeric_data.std()

  std_diff = (std_syn - std_orig).abs() / std_orig.abs()
  std_preserved = std_diff.median()
  print(f"Std preserved      : {std_preserved:.3f}")
  wd_vals=[]
  for col in numeric_data_o.columns:
    print(data.columns,data_syn.columns)
    wd=wasserstein_distance(data[col],data_syn[col])
    wd_vals.append(wd)
  print(wd_vals)



moments(data,data_syn)
moments(data1,data_syn1)
moments(data2,data_syn2)
moments(data3,data_syn3)
moments(data4,data_syn4)
moments(data5,data_syn5)
moments(data6,data_syn6)
moments(data7,data_syn7)
