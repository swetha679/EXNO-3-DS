## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ```
 REG NO:212224040345
NAME:B R SWETHA NIVASINI
```

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data (2).csv")
df
```
![Screenshot 2025-04-19 181928](https://github.com/user-attachments/assets/55b0cd6f-26bb-4956-9668-7e0251fdc575)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![Screenshot 2025-04-19 182028](https://github.com/user-attachments/assets/24e68796-9de6-472a-a9bf-23662dd951c5)

```
df
```
![Screenshot 2025-04-19 182103](https://github.com/user-attachments/assets/0e154334-80ca-4815-ac0c-218ce6c9e783)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![Screenshot 2025-04-19 182155](https://github.com/user-attachments/assets/a43e2eef-2a0f-4878-a48d-9d825093bd5d)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![Screenshot 2025-04-19 182255](https://github.com/user-attachments/assets/bc11f3ba-1777-4dc7-8e04-735305c65770)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-19 182317](https://github.com/user-attachments/assets/ee705485-84e4-4bc5-9e31-f98deb2f21b2)

```
pip install --upgrade category_encoders
```

![Screenshot 2025-04-19 182403](https://github.com/user-attachments/assets/3c94f7bb-a331-48e6-9e01-0ce1fe2ddd15)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data (2).csv")
df
```

![Screenshot 2025-04-19 182501](https://github.com/user-attachments/assets/d5063275-11de-452e-ad8a-2843770ffbdc)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

![Screenshot 2025-04-19 182535](https://github.com/user-attachments/assets/63cccc0c-b2c3-475f-8fce-647897d153fa)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```


![Screenshot 2025-04-19 182635](https://github.com/user-attachments/assets/3959057f-25b9-4f74-8d63-4c17610adf41)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```


![Screenshot 2025-04-19 182739](https://github.com/user-attachments/assets/7b02b117-458e-4067-b4cf-3531769cc51d)

```
df.skew()
```

![Screenshot 2025-04-19 182826](https://github.com/user-attachments/assets/8aefda8c-00f0-457f-8b6c-b4d0aa68b02f)

```
np.log(df["Highly Positive Skew"])
```


![Screenshot 2025-04-19 182912](https://github.com/user-attachments/assets/383c6304-ef49-4c5a-a616-16fb053c1929)

```
np.reciprocal(df["Moderate Positive Skew"])
```


![Screenshot 2025-04-19 183028](https://github.com/user-attachments/assets/7acd1dd8-6418-4d7a-9cc6-eb45b6297ea1)

```
np.sqrt(df["Highly Positive Skew"])
```

![Screenshot 2025-04-19 183110](https://github.com/user-attachments/assets/af885ccf-e105-4434-b026-8ed3d1874b5e)

```
np.square(df["Highly Positive Skew"])
```


![Screenshot 2025-04-19 183154](https://github.com/user-attachments/assets/1c261dcd-25e5-4a8e-9e18-c4dadb4eb9f5)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![Screenshot 2025-04-19 183223](https://github.com/user-attachments/assets/a760c71e-72b9-4dfd-a0c0-e4896f738f03)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```

![Screenshot 2025-04-19 183305](https://github.com/user-attachments/assets/deb225bf-5bb2-490a-9eaa-bd337b421046)

```
df["Highly Negative Skew_yeojohson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```


![Screenshot 2025-04-19 183338](https://github.com/user-attachments/assets/843e9af6-86c5-4e6c-a670-bef50442070e)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```


![Screenshot 2025-04-19 183432](https://github.com/user-attachments/assets/3870f7a6-978b-4441-be6a-79971719c43a)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2025-04-19 183509](https://github.com/user-attachments/assets/b4c34d34-d7c1-4f51-baf7-ab5799e60619)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![Screenshot 2025-04-19 183548](https://github.com/user-attachments/assets/cdb2978b-a283-47aa-901e-be5babca0f92)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2025-04-19 183617](https://github.com/user-attachments/assets/53830cf9-4d23-40f8-93ad-38eedaad6eb7)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![Screenshot 2025-04-19 183658](https://github.com/user-attachments/assets/c4195cfa-d873-4e19-b9ef-eb8f270c1c74)

```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```

![Screenshot 2025-04-19 183729](https://github.com/user-attachments/assets/3661cbb6-59ba-4591-9db3-4a5ce48e4a93)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```


![Screenshot 2025-04-19 183814](https://github.com/user-attachments/assets/63485ce3-0121-4693-998d-40149cad771c)



















































































































































































































































































































# RESULT:
       # INCLUDE YOUR RESULT HERE

       
