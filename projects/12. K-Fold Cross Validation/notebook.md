---
layout: default
title: "K-Fold Cross Validation - Notebook"
permalink: projects/k-fold-cross-validation/notebook
sidebar: sidebar
---

# K-Fold Cross Validation - Jupyter Notebook

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

## LOAD THE DATASETS AND LIBRARIES


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')
```


```python
from sklearn.datasets import load_digits
data = load_digits()

dir(data)
```




    ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']




```python
df = pd.DataFrame(data.data)
df['target'] = data.target

df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>478</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>927</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>13.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>224</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>288</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>313</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>345</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>209</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>820</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 65 columns</p>
</div>




```python
X = df.drop(columns = ['target'])
y = df['target']

X.shape, y.shape
```




    ((1797, 64), (1797,))




```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

X_train.shape, y_train.shape
```




    ((1257, 64), (1257,))




```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
```




    0.9648148148148148




```python
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
```




    0.987037037037037




```python
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
```




    0.9722222222222222



This method of checking which model is better is too naive, and it is not a best practical method to know it. Because the dataset is not split correctly


```python
from sklearn.model_selection import KFold

kf = KFold(n_splits = 3)
kf
```




    KFold(n_splits=3, random_state=None, shuffle=False)




```python
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

# this is how kfold works, where you can see the whole dataset is split in a balanced way, and given all the dataset in train and test split
```

    [3 4 5 6 7 8] [0 1 2]
    [0 1 2 6 7 8] [3 4 5]
    [0 1 2 3 4 5] [6 7 8]
    


```python
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# This function makes it easy to return the scores of all the comparing models
```


```python
get_score(lr, X_train, X_test, y_train, y_test)
```




    0.9648148148148148




```python
get_score(svm, X_train, X_test, y_train, y_test)
```




    0.987037037037037




```python
get_score(rf, X_train, X_test, y_train, y_test)
```




    0.975925925925926




```python
# The stratified version of KFold, this devides the target column in uniform way to get the balaced data in all the folds

from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits = 3)
```


```python
scores_l = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(get_score(lr, X_train, X_test, y_train, y_test))
    print(get_score(svm, X_train, X_test, y_train, y_test))
    print(get_score(rf, X_train, X_test, y_train, y_test))
    
```

    0.9215358931552587
    0.9649415692821369
    0.9432387312186978
    0.9415692821368948
    0.9799666110183639
    0.9532554257095158
    0.9165275459098498
    0.9649415692821369
    0.9332220367278798
    


```python
scores_l = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scores_l.append(get_score(lr, X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(svm, X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(rf, X_train, X_test, y_train, y_test))
    
```


```python
scores_l
np.average(scores_l)
```




    0.9265442404006677




```python
scores_svm
np.average(scores_svm)

# This is the best model that we got
```




    0.9699499165275459




```python
scores_rf
np.average(scores_rf)

# lets take the average and get which model is the best
```




    0.9393433500278241




```python
# Instead of creating a different funcion as above, we have a built in library

from sklearn.model_selection import cross_val_score

cross_val_score(LogisticRegression(), X, y)
```




    array([0.92222222, 0.86944444, 0.94150418, 0.93871866, 0.89693593])




```python
cross_val_score(svm, X, y)
```




    array([0.96111111, 0.94444444, 0.98328691, 0.98885794, 0.93871866])




```python
cross_val_score(rf, X, y)
```




    array([0.93055556, 0.90277778, 0.95821727, 0.95543175, 0.93036212])




```python
# we can do directly like this also

scores1 = cross_val_score(LogisticRegression(), X, y, cv = 10)
np.average(scores1)
```




    0.928193668528864




```python
scores2 = cross_val_score(svm, X, y, cv = 10)
np.average(scores2)

# The best model
```




    0.9699503414028554




```python
scores3 = cross_val_score(rf, X, y, cv = 10)
np.average(scores3)
```




    0.9487988826815641


