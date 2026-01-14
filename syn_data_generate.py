#Gaussian Copula assumes data is symmetric. It is used to model linear dependencies in data
#clayton is for assymetric dependence when one variable tend to have extreme effect on another variable
#Gumbel Copula is also for assymetric dependence when both the variables co vary in extreme. Variables that have high tedency to be together
#Skewness values close to zero indicate symmetry in the distribution. If skewness is far from zero, the data may be asymmetric.

#Kurtosis greater than 3 indicates that the data has heavy tails (potentially important for tail dependence modeling).
#skewness and kurtosis of GaussianCopulaSynthesizer does not match
!pip install sdv
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import Metadata
import numpy as np
from sdv.metadata import Metadata
import numpy as np
#checking skewness kurtosis only for numeric columns
def check_symmetry(data):
   numeric_cols=data.select_dtypes(include=[np.number]).columns
   data=data[numeric_cols]
   #correlation=data.corr()
   skewness=data.skew()
   kurtosis=data.kurtosis()
   return skewness,kurtosis

skewness,kurtosis=check_symmetry(data)
print("Skewness")
print(skewness)
print("Kurtosis")
print(kurtosis)

skewness,kurtosis=check_symmetry(data1)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)

skewness,kurtosis=check_symmetry(data2)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)

skewness,kurtosis=check_symmetry(data3)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)

skewness,kurtosis=check_symmetry(data4)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)

skewness,kurtosis=check_symmetry(data5)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)

skewness,kurtosis=check_symmetry(data6)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)

skewness,kurtosis=check_symmetry(data7)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)




metadata = Metadata.detect_from_dataframe(data)
metadata1 = Metadata.detect_from_dataframe(data1)
metadata2 = Metadata.detect_from_dataframe(data2)
metadata3 = Metadata.detect_from_dataframe(data3)
metadata4 = Metadata.detect_from_dataframe(data4)
metadata5 = Metadata.detect_from_dataframe(data5)
metadata6 = Metadata.detect_from_dataframe(data6)
metadata7 = Metadata.detect_from_dataframe(data7)
print(data3.shape)
synthesizer = CopulaGANSynthesizer( metadata=metadata3,epochs=5)
synthesizer.fit(data3)
synthetic_data = synthesizer.sample(num_rows=1000)
skewness,kurtosis=check_symmetry(synthetic_data)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)
synthetic_data.to_csv("data3.csv")
synthesizer = CopulaGANSynthesizer( metadata=metadata4,epochs=10)
synthesizer.fit(data4)
synthetic_data = synthesizer.sample(num_rows=1000)
skewness,kurtosis=check_symmetry(synthetic_data)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)
synthetic_data.to_csv("data4.csv")
synthesizer = CopulaGANSynthesizer( metadata=metadata5,epochs=10)
synthesizer.fit(data5)
synthetic_data = synthesizer.sample(num_rows=1000)
skewness,kurtosis=check_symmetry(synthetic_data)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)
synthetic_data.to_csv("data5.csv")
synthesizer = CopulaGANSynthesizer(metadata=metadata6,epochs=10)

synthesizer.fit(data6)
synthetic_data = synthesizer.sample(num_rows=1000)
skewness,kurtosis=check_symmetry(synthetic_data)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)
synthetic_data.to_csv("data6.csv")
synthesizer = CopulaGANSynthesizer( metadata=metadata7,epochs=10)
synthesizer.fit(data7)
synthetic_data = synthesizer.sample(num_rows=1000)
skewness,kurtosis=check_symmetry(synthetic_data)
print("skewness")
print(skewness)
print("kurtosis")
print(kurtosis)
synthetic_data.to_csv("data7.csv")
