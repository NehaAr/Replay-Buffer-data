import pandas as pd
from numbers import Rational
import random

def add_data(data,syn_data, ratio=0.5):
   total_len=len(data)+len(syn_data)
   cap,remaining_cap=0,0
   if total_len>=buff. capacity:
     print(total_len)
     print(buff.capacity)

     cap=int(buff.capacity*ratio)
     remaining_cap=buff.capacity-cap
     print(cap,remaining_cap)
   else:
     cap=int((total_len)*ratio)
     remaining_cap=int(total_len*(1-ratio))
   n_clean=data.sample(cap,random_state=42)
   n_syn=syn_data.sample(remaining_cap,random_state=42)
   for sample in n_clean.iterrows():
        buff.add(sample)
   for sample in n_syn.iterrows():
        buff.add(sample)
    

