# -----------------------------------------------------------------------------
# Copyright (c) 2026 Neha
#
# Licensed under the GPL License. See LICENSE file for details.
# -----------------------------------------------------------------------------
from os import remove
#Dataset Cleaning
import faiss
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import time
import gc
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from annoy import AnnoyIndex
import gower
import hdbscan
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
mask_columns_list=[]
#important functions

def standardize_data(data):
    data.columns=data.columns.str.strip()
    return data

def check_len(data):

   if len(data)==0:
     return 0
   else:
     return len(data)


def check_dtype(data):
 for i in data.columns:
   if data[i].apply(lambda x: isinstance(x, float)).all():
        min_value = data[i].min()
        max_value = data[i].max()
        if np.all(np.isfinite(data[i])):
          if np.abs(min_value) < np.finfo(np.float32).max and np.abs(max_value) < np.finfo(np.float32).max:
            data[i]=data[i].astype('float32')
          else:
            data[i]=data[i].astype('float64')
   elif data[i].apply(lambda x: isinstance(x, int)).all():
        # Check if the integer column exceeds the int32 range
        if data[i].min() < -2**31 or data[i].max() > 2**31-1:
            data[i] = data[i].astype('int64')
        else:
            data[i] = data[i].astype('int32')
   elif data[i].apply(lambda x: isinstance(x, str)).all():
      if data[i].nunique()<=0.5*len(data):
        data[i]=data[i].astype('category')
      else:
        data[i]=data[i].astype('String')
   else:
        data[i]=data[i].astype('object')

 return data

def check_zero(data):
    target_col=data['Label']
    data_new=data.drop(columns=['Label'])
    mask_columns=((data_new==0).all()) | (data_new.nunique()==1)
    zero_columns=mask_columns[mask_columns].index.to_list()
    mask_columns_list.extend(zero_columns)


def remove_zero(data):
    data=data.drop(columns=mask_columns_list,errors='ignore')
    mask_rows=(data==0).all(axis=1)
    data=data.loc[~mask_rows,:]
    return data

def remove_outliers(data, threshold=3, type=2):      #IQR hadles NaNs
  if type==1:
   print("remove outliers",data.columns)
   start_time=time.time()
   q1=data[numerical_cols].quantile(0.25)
   q3=data[numerical_cols].quantile(0.75)
   iqr=q3-q1
   lower_bound=q1-1.5*iqr
   upper_bound=q3+1.5*iqr
   mask = ((data[numerical_cols] < lower_bound) | (data[numerical_cols] > upper_bound))
   outlier_content=mask.sum(axis=1)
    # Keep rows that do NOT have any outliers
   data = data[outlier_content<=threshold]
   print("after outlier removal",data.columns)
   end_time=time.time()
   print("time taken after removal of outliers",end_time-start_time)
   print("Data Summary",data.describe())

  elif type==2:
    print("remove outliers",data.columns)
    start_time=time.time()
    transformer=ColumnTransformer(transformers=[('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder())]),categorical_cols),('num',Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler',StandardScaler())]),numerical_cols)],remainder='passthrough')
    model=IsolationForest(contamination=0.1, random_state=42)
    pipeline=Pipeline(steps=([('t',transformer),('m',model)]))
    pipeline.fit(data)
    predictions=pipeline.predict(data)
    mask=predictions==1
    data=data[mask]
    print("after outlier removal",data.columns)
    end_time=time.time()
    print("time taken after removal of outliers",end_time-start_time)

    print("Data Summary",data.describe())

  elif type==3:
    print("remove outliers",data.columns)
    start_time=time.time()
    transformer=ColumnTransformer(transformers=[('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder())]),categorical_cols),('num',Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler',StandardScaler())]),numerical_cols)],remainder='passthrough')
    model=DBSCAN(eps=0.5,min_samples=5,metric='euclidean',algorithm="ball_tree",n_jobs=-1)                   #kd-tree for smaller dimensions and ball_tree for higher dimensions
    pipeline=Pipeline(steps=([('t',transformer),('m',model)]))
    predictions=pipeline.fit_predict(data)
    mask=predictions!=-1
    data=data[mask]
    end_time=time.time()
    print("after outlier removal",data.columns)
    print("time taken after removal of outliers",end_time-start_time)
    print("Data Summary",data.describe())

  elif type==4:
    print("remove outliers",data.columns)
    start_time=time.time()
   # 1. Preprocess pipeline (for other models, not DBSCAN)
    transformer = ColumnTransformer(
      transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder())
        ]), categorical_cols),

        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
     ],
     remainder='passthrough'
       )

    pipeline = Pipeline([
     ('prep', transformer)
     ])
    for col in data.select_dtypes(include='category').columns:
      data[col] = data[col].astype(object) 
    gower_dist_matrix = gower.gower_matrix(data)

   # 3. Fit DBSCAN
    db = DBSCAN(
     eps=0.45,
     min_samples=5,
     metric="precomputed",
     n_jobs=-1
    )

    pred = db.fit_predict(gower_dist_matrix)

    # 4. Filter original DataFrame
    mask = pred != -1
    data = data[mask]
    end_time=time.time()
    print("time taken after removal of outliers",end_time-start_time)
    print("Data Summary",data.describe())
  elif type==5:
     print("remove outliers",data.columns)
     start_time=time.time()
     nlist=10
     print("IS GPU Avalaible", torch.cuda.is_available())
     if torch.cuda.is_available():
       res=faiss.StandardGpuResources()
       index=faiss.IndexFlatL2(data[numerical_cols].shape[1])
       gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
       index_ivf=faiss.IndexIVFFlat(gpu_index,data[numerical_cols].shape[1],nlist,faiss.METRIC_L2)
       index_ivf.train(data[numerical_cols].values)
       index_ivf.add(data[numerical_cols].values)
     # Perform nearest neighbor search
       D, I = index_ivf.search(data.astype(np.float32), k=5)
       distance_matrix=np.zeros((len(data),len(data)))
       for i in range(len(data)):
        for j, idx in enumerate(I[i]):
          distance_matrix[i,idx]=D[i,j]
     else:
       index=faiss.IndexFlatL2(data[numerical_cols].shape[1])
       index_ivf=faiss.IndexIVFFlat(index,data[numerical_cols].shape[1],nlist,faiss.METRIC_L2)
       index_ivf.train(data[numerical_cols].values)
       index_ivf.add(data[numerical_cols].values)
     # Perform nearest neighbor search
       D, I = index_ivf.search(data.astype(np.float32), k=5)
       distance_matrix=np.zeros((len(data),len(data)))
       for i in range(len(data)):
        for j, idx in enumerate(I[i]):
          distance_matrix[i,idx]=D[i,j]

     clusterer=hdbscan.HDBSCAN(metric="precomputed",min_cluster_size=10,min_samples=5)
     pred=clusterer.fit_predict(distance_matrix)
     mask=pred!=-1
     data=data[mask]
     end_time=time.time()
     print("after outlier removal",data.columns)
     print("time taken after removal of outliers",end_time-start_time)
     print("Data Summary",data.describe())
  elif type==6:
     print("remove outliers",data.columns)
     start_time=time.time()
     neighbors=NearestNeighbors(n_neighbors=5,algorithm='ball_tree',n_jobs=-1)
     neighbors.fit(data[numerical_cols])
     distances,indices=neighbors.kneighbors(data[numerical_cols])
     distance_matrix = np.zeros((len(data), len(data)))

      # Step 4: Fill the distance matrix with the nearest neighbor distances
     for i in range(len(data)):
      for j, idx in enumerate(indices[i]):
        distance_matrix[i, idx] = distances[i, j]
     dbscan=DBSCAN(eps=0.5,min_samples=5,metric='precomputed')
     pred=dbscan.fit_predict(distance_matrix)
     mask=pred!=-1
     data=data[mask]
     end_time=time.time()
     print("after outlier removal",data.columns)
     print("time taken after removal of outliers",end_time-start_time)
     print("Data Summary",data.describe())
  elif type==7:
    print("remove outliers",data.columns)
    start_time=time.time()
    annoy_index=AnnoyIndex(data[numerical_cols].shape[1],metric='angular')
    for i in range(len(data)):
       annoy_index.add_item(i,data[numerical_cols].iloc[i,:])
    annoy_index.build(10)
    k=5
    neighbors=[]
    distances=[]
    for i in range(len(data)):
      neighbor_indices,neighbor_distances=annoy_index.get_nns_by_item(i,k,include_distances=True)
      neighbors.append(neighbor_indices)
      distances.append(neighbor_distances)
    neighbors=np.array(neighbors)
    distances=np.array(distances)
    distance_matrix=np.zeros((len(data),len(data)))
    for i in range(len(data)):
     for j,idx in enumerate(neighbors[i]):
       distance_matrix[i,idx]=distances[i,j]
    dbscan=DBSCAN(eps=0.5,min_samples=5,metric='precomputed')
    pred=dbscan.fit_predict(distance_matrix)
    mask=pred!=-1
    data=data[mask]
    end_time=time.time()
    print("after outlier removal",data.columns)
    print("time taken after removal of outliers",end_time-start_time)
    print("Data Summary",data.describe())

  else:
   print("remove_outliers", data.columns)
   start_time=time.time()
   gower_dist = gower.gower_matrix(data)
   gower_dist=gower_dist.astype(np.float64)
   clusterer=hdbscan.HDBSCAN(metric="euclidean",min_cluster_size=10,min_samples=5)
   pred=clusterer.fit_predict(data[numerical_cols])
   mask=pred!=-1
   data=data[mask]
   end_time=time.time()
   print("after outlier removal",data.columns)
   print("time taken after removal of outliers",end_time-start_time)
   print("Data Summary",data.describe())
  return data


def data_imputer(data):
   for i in numerical_cols:
       data[i]=data[i].fillna(data[i].mean())
   for i in categorical_cols:
       data[i]=data[i].fillna(data[i].mode()[0])
   return data

#Random Matrix Theory(Eigen Value Clipping)
#ledoit Wolf(features>>samples),hard and soft thrshlding

def clean_corrmat(data,n_surr=20,random_state=None,method=1):
  if method==1:
   print("In method1")
   start_time=time.time()
   rng=np.random.default_rng(random_state)
   N,M=data.shape
   eig_surr=np.zeros((n_surr,M))
   for i in range(n_surr):
     data_surr=data.copy()
     for col in data_surr.columns:
        data_surr[col] = np.random.permutation(data_surr[col].values)
     corr_surr=np.corrcoef(data_surr,rowvar=False)
     corr_surr=np.nan_to_num(corr_surr,nan=0, posinf=0.0, neginf=0.0)
     print(corr_surr)
     eigs=np.linalg.eigvals(corr_surr)
     eig_surr[i,:]=np.sort(eigs)[::-1]
   all_surr_eigs = eig_surr.ravel()
   lam_empirical = np.percentile(all_surr_eigs, 97.5)
   C_real=np.corrcoef(data,rowvar=False)
   evals_real, evecs_real = np.linalg.eigh(C_real)
   evals_real_desc = evals_real[::-1]
   evecs_real_desc = evecs_real[:, ::-1]
   signal_mask = evals_real_desc > lam_empirical
   corr_clean=evals_real_desc.copy()
   corr_clean[~signal_mask]=0
   corr_clean=evecs_real_desc@np.diag(corr_clean)@evecs_real_desc.T
   corr_clean=(corr_clean+corr_clean.T)/2
   D=np.sqrt(np.diag(corr_clean))
   D=np.nan_to_num(D,nan=0, posinf=0.0, neginf=0.0)
   corr_clean=np.nan_to_num(corr_clean,nan=0, posinf=0.0, neginf=0.0)
   corr_clean=corr_clean/np.outer(D,D)
   np.fill_diagonal(corr_clean,1)
   end_time=time.time()
   print("time taken for cleaning correlation",end_time-start_time)
   print("method1")
   return corr_clean

  elif method==2:
    print("In method2")
    start_time=time.time()
    lw = LedoitWolf()
    lw.fit(data)
    cov_clean = lw.covariance_
    corr_lw=cov_clean/np.sqrt(np.outer(np.diag(cov_clean),np.diag(cov_clean)))
    np.fill_diagonal(corr_lw,1)
    corr_clean=corr_lw
    end_time=time.time()
    print("time taken for cleaning correlation",end_time-start_time)
    print("method2")
    return corr_clean

  else:  # laplacian smoothening
    print("In method3")
    start_time=time.time()
    adcacency=np.abs(data.corr())
    daigonal=np.diag(adcacency.sum(Axis=1))
    laplacian=daigonal-adcacency
    alpha=0.1
    corr_clean=np.linalg.inv(np.eye(len(data.corr))+alpha)@data.corr
    end_time=time.time()
    print("time taken for cleaning correlation",end_time-start_time)
    print("method3")
    return corr_clean

def remove_correlated(data):
  target_col=data['Label']
  data_new=data.drop(columns=['Label'])
  columns=data_new.columns
  for i in range(1,4):
    correlation_matrix=clean_corrmat(data_new,random_state=42,method=i)
    correlated_features=set()
    for i in range(correlation_matrix.shape[0]):
       for j in range(i):
         if abs(correlation_matrix[i,j])>=0.7 and i!=j:
          colname=columns[i]
          correlated_features.add(colname)
    if len(correlated_features)==0:
      print("No correlated features")
      return data
    else:
      label_encoder = LabelEncoder()
      target_col_numeric = label_encoder.fit_transform(target_col)

       # Now you can compute correlation
      target_col_numeric = pd.Series(target_col_numeric)
      target_corr=[]
      for i in correlated_features:
        check_corr=target_col_numeric.corr(data_new[i])
        if abs(check_corr)>=0.5:
          target_corr.append(i)
      if len(target_corr)!=0:
        data_new=data_new[target_corr]
        X=data_new
        y=target_col
        rf=RandomForestClassifier()
        rf.fit(X,y)
        feature_importance=rf.feature_importances_
        feature_importance_df=pd.DataFrame({'feature':X.columns,'importance':feature_importance})
        feature_importance_df=feature_importance_df.sort_values(by='importance',ascending=False)
        feature_importance_df=feature_importance_df[feature_importance_df['importance']>0.05]
        for i in feature_importance_df['feature'].tolist():
          correlated_features.remove(i)
        data=data.drop(columns=correlated_features)
      else:
        data=data.drop(columns=correlated_features)
      return data


data = pd.read_csv('/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
data1 = pd.read_csv('/content/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
data2 = pd.read_csv('/content/Friday-WorkingHours-Morning.pcap_ISCX.csv')
data3 = pd.read_csv('/content/Monday-WorkingHours.pcap_ISCX.csv')
data4 = pd.read_csv('/content/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
data5 = pd.read_csv('/content/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
data6 = pd.read_csv('/content/Tuesday-WorkingHours.pcap_ISCX.csv')
data7 = pd.read_csv('/content/Wednesday-workingHours.pcap_ISCX.csv')

#standardize the dataset columns

data=standardize_data(data)
data1=standardize_data(data1)
data2=standardize_data(data2)
data3=standardize_data(data3)
data4=standardize_data(data4)
data5=standardize_data(data5)
data6=standardize_data(data6)
data7=standardize_data(data7)

print("Data shape at first stage")
print(data.shape)
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data5.shape)
print(data6.shape)
print(data7.shape)
#Replaces Infinity with np.nan

check_len(data)
check_len(data1)
check_len(data2)
check_len(data3)
check_len(data4)
check_len(data5)
check_len(data6)
check_len(data7)

data=data.replace([np.inf,-np.inf],np.nan)
data1=data1.replace([np.inf,-np.inf],np.nan)
data2=data2.replace([np.inf,-np.inf],np.nan)
data3=data3.replace([np.inf,-np.inf],np.nan)
data4=data4.replace([np.inf,-np.inf],np.nan)
data5=data5.replace([np.inf,-np.inf],np.nan)
data6=data6.replace([np.inf,-np.inf],np.nan)
data7=data7.replace([np.inf,-np.inf],np.nan)

data.drop_duplicates(inplace=True)
data1.drop_duplicates(inplace=True)
data2.drop_duplicates(inplace=True)
data3.drop_duplicates(inplace=True)
data4.drop_duplicates(inplace=True)
data5.drop_duplicates(inplace=True)
data6.drop_duplicates(inplace=True)
data7.drop_duplicates(inplace=True)

#Removes columns with all Na values
data.dropna(axis=1,how='all',inplace=True)
data1.dropna(axis=1,how='all',inplace=True)
data2.dropna(axis=1,how='all',inplace=True)
data3.dropna(axis=1,how='all',inplace=True)
data4.dropna(axis=1,how='all',inplace=True)
data5.dropna(axis=1,how='all',inplace=True)
data6.dropna(axis=1,how='all',inplace=True)
data7.dropna(axis=1,how='all',inplace=True)
check_len(data)
check_len(data1)
check_len(data2)
check_len(data3)
check_len(data4)
check_len(data5)
check_len(data6)
check_len(data7)

print("Data shape at second stage after removal of duplicates and nan columns")
print(data.shape)
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data5.shape)
print(data6.shape)
print(data7.shape)
#Removes rows with all null values
data.dropna(how='all',inplace=True)
data1.dropna(how='all',inplace=True)
data2.dropna(how='all',inplace=True)
data3.dropna(how='all',inplace=True)
data4.dropna(how='all',inplace=True)
data5.dropna(how='all',inplace=True)
data6.dropna(how='all',inplace=True)
data7.dropna(how='all',inplace=True)
check_len(data)
check_len(data1)
check_len(data2)
check_len(data3)
check_len(data4)
check_len(data5)
check_len(data6)
check_len(data7)

print("Data shape at 3rd stage after removal of nan rows")
print(data.shape)
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data5.shape)
print(data6.shape)
print(data7.shape)
#Removes rows with 50 percent missing values
data.dropna(thresh=data.shape[1]/2,inplace=True)
data1.dropna(thresh=data1.shape[1]/2,inplace=True)
data2.dropna(thresh=data2.shape[1]/2,inplace=True)
data3.dropna(thresh=data3.shape[1]/2,inplace=True)
data4.dropna(thresh=data4.shape[1]/2,inplace=True)
data5.dropna(thresh=data5.shape[1]/2,inplace=True)
data6.dropna(thresh=data6.shape[1]/2,inplace=True)
data7.dropna(thresh=data7.shape[1]/2,inplace=True)
check_len(data)
check_len(data1)
check_len(data2)
check_len(data3)
check_len(data4)
check_len(data5)
check_len(data6)
check_len(data7)

print("Data shape at 4th stage after removal of rows with 50 percent nan values")
print(data.shape)
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data5.shape)
print(data6.shape)
print(data7.shape)
# Sets the  target column
data=check_dtype(data)
data1=check_dtype(data1)
data2=check_dtype(data2)
data3=check_dtype(data3)
data4=check_dtype(data4)
data5=check_dtype(data5)
data6=check_dtype(data6)
data7=check_dtype(data7)

check_zero(data)
check_zero(data1)
check_zero(data2)
check_zero(data3)
check_zero(data4)
check_zero(data5)
check_zero(data6)
check_zero(data7)
mask_columns_list=list(set(mask_columns_list))
data=remove_zero(data)
data1=remove_zero(data1)
data2=remove_zero(data2)
data3=remove_zero(data3)
data4=remove_zero(data4)
data5=remove_zero(data5)
data6=remove_zero(data6)
data7=remove_zero(data7)

print("Data Shape at fifth stage after removal of constant and 0 columns")
print(data.shape)
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data5.shape)
print(data6.shape)
print(data7.shape)
print(data.columns)
print(data1.columns)
print(data2.columns)
print(data3.columns)
print(data4.columns)
print(data5.columns)
print(data6.columns)
print(data7.columns)

print("###################################################################")
numerical_cols=data.select_dtypes(include=['number','int64','int32','float64','float32']).columns
categorical_cols=data.select_dtypes(exclude=['number','int64','int32','float64','float32']).columns
print("###################################################################")

data=data_imputer(data)
data1=data_imputer(data1)
data2=data_imputer(data2)
data3=data_imputer(data3)
data4=data_imputer(data4)
data5=data_imputer(data5)
data6=data_imputer(data6)
data7=data_imputer(data7)
data_new=data.copy()
data1_new=data1.copy()
data2_new=data2.copy()
data3_new=data3.copy()
data4_new=data4.copy()
data5_new=data5.copy()
data6_new=data6.copy()
data7_new=data7.copy()
for i in range(1,8):
   data=remove_outliers(data,type=i)
   data1=remove_outliers(data1,type=i)
   data2=remove_outliers(data2,type=i)
   data3=remove_outliers(data3,type=i)
   data4=remove_outliers(data4,type=i)
   data5=remove_outliers(data5,type=i)
   data6=remove_outliers(data6,type=i)
   data7=remove_outliers(data7,type=i)

   print("Data Shape at sixth stage after imputation and removal of outliers")
   print(data.shape)
   print(data1.shape)
   print(data2.shape)
   print(data3.shape)
   print(data4.shape)
   print(data5.shape)
   print(data6.shape)
   print(data7.shape)
   l=check_len(data)
   l1=check_len(data1)
   l2=check_len(data2)
   l3=check_len(data3)
   l4=check_len(data4)
   l5=check_len(data5)
   l6=check_len(data6)
   l7=check_len(data7)
   #Cleaning correlation matrix leads to addition of nan values
   data=data_imputer(data)
   data1=data_imputer(data1)
   data2=data_imputer(data2)
   data3=data_imputer(data3)
   data4=data_imputer(data4)
   data5=data_imputer(data5)
   data6=data_imputer(data6)
   data7=data_imputer(data7)
   print(data.shape)
   print(data1.shape)
   print(data2.shape)
   print(data3.shape)
   print(data4.shape)
   print(data5.shape)
   print(data6.shape)
   print(data7.shape)

   print("Data Shape at seventh stage after imputation to avoid nan's due to correlation matrix cleaning")

   datasets=[data,data1,data2,data3,data4,data5,data6,data7]
   concat_data=pd.concat(datasets,axis=0)
   del datasets,data,data1,data2,data3,data4,data5,data6,data7
   gc.collect()
   concat_data_new=remove_correlated(concat_data)
   print(concat_data_new.shape)
   print("Data Shape at eiigth stage after correlated columns removal")
   split_indices =[0,l,l+l1,l+l1+l2,l+l1+l2+l3,l+l1+l2+l3+l4,l+l1+l2+l3+l4+l5,l+l1+l2+l3+l4+l5+l6,l+l1+l2+l3+l4+l5+l6+l7]
   cleaned_datasets=[concat_data.iloc[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices)-1)]
   del split_indices
   gc.collect()

   data=cleaned_datasets[0]
   data1=cleaned_datasets[1]
   data2=cleaned_datasets[2]
   data3=cleaned_datasets[3]
   data4=cleaned_datasets[4]
   data5=cleaned_datasets[5]
   data6=cleaned_datasets[6]
   data7=cleaned_datasets[7]
   print(data.shape)
   print(data1.shape)
   print(data2.shape)
   print(data3.shape)
   print(data4.shape)
   print(data5.shape)
   print(data6.shape)
   print(data7.shape)
   data=data_new
   data1=data1_new
   data2=data2_new
   data3=data3_new
   data4=data4_new
   data5=data5_new
   data6=data6_new
   data7=data7_new

