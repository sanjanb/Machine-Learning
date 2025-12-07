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

    /kaggle/input/carprices/carprices.csv
    /kaggle/input/encoding/homeprices (2).csv
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv("/kaggle/input/encoding/homeprices (2).csv")
df.head()
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
      <th>town</th>
      <th>area</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>monroe township</td>
      <td>2600</td>
      <td>550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>monroe township</td>
      <td>3000</td>
      <td>565000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>monroe township</td>
      <td>3200</td>
      <td>610000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>monroe township</td>
      <td>3600</td>
      <td>680000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>monroe township</td>
      <td>4000</td>
      <td>725000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (13, 3)




```python
df.town.value_counts()
```




    town
    monroe township    5
    west windsor       4
    robinsville        4
    Name: count, dtype: int64



## GET DUMMIES METHOD FROM PANDAS


```python
dummies = pd.get_dummies(df.town).astype(int)
dummies.head(10)
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
      <th>monroe township</th>
      <th>robinsville</th>
      <th>west windsor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged = pd.concat([ df, dummies], axis = 'columns')
merged.head()
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
      <th>town</th>
      <th>area</th>
      <th>price</th>
      <th>monroe township</th>
      <th>robinsville</th>
      <th>west windsor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>monroe township</td>
      <td>2600</td>
      <td>550000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>monroe township</td>
      <td>3000</td>
      <td>565000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>monroe township</td>
      <td>3200</td>
      <td>610000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>monroe township</td>
      <td>3600</td>
      <td>680000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>monroe township</td>
      <td>4000</td>
      <td>725000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Lets drop town column

final = merged.drop(['town', 'west windsor'], axis = 1)
final
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
      <th>area</th>
      <th>price</th>
      <th>monroe township</th>
      <th>robinsville</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>550000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>565000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>610000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>680000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
      <td>725000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2600</td>
      <td>585000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2800</td>
      <td>615000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3300</td>
      <td>650000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3600</td>
      <td>710000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2600</td>
      <td>575000</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2900</td>
      <td>600000</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3100</td>
      <td>620000</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3600</td>
      <td>695000</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(final)
```




    [<matplotlib.lines.Line2D at 0x7d972e82fb10>,
     <matplotlib.lines.Line2D at 0x7d972e6dae90>,
     <matplotlib.lines.Line2D at 0x7d972e6db190>,
     <matplotlib.lines.Line2D at 0x7d972e6db510>]




    
![png](notebook_files/notebook_9_1.png)
    



```python
from sklearn import linear_model

model = linear_model.LinearRegression()
```


```python
X = final.drop(columns = ['price'])
y = final['price']
X.shape, y.shape
```




    ((13, 3), (13,))




```python
model.fit(X, y)
```




<style>#sk-container-id-14 {color: black;background-color: white;}#sk-container-id-14 pre{padding: 0;}#sk-container-id-14 div.sk-toggleable {background-color: white;}#sk-container-id-14 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-14 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-14 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-14 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-14 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-14 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-14 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-14 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-14 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-14 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-14 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-14 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-14 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-14 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-14 div.sk-item {position: relative;z-index: 1;}#sk-container-id-14 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-14 div.sk-item::before, #sk-container-id-14 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-14 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-14 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-14 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-14 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-14 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-14 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-14 div.sk-label-container {text-align: center;}#sk-container-id-14 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-14 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-14" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" checked><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
model.coef_
```




    array([   126.89744141, -40013.97548914, -14327.56396474])




```python
model.intercept_
```




    249790.36766292527




```python
import pickle

with open('model_pickle', 'wb') as f:
    pickle.dump(model, f)
```


```python
with open('model_pickle', 'rb') as f:
    reg = pickle.load(f)
```


```python
reg.coef_
```




    array([   126.89744141, -40013.97548914, -14327.56396474])




```python
reg.predict([[2800, 0, 1]])
```

    /usr/local/lib/python3.11/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    




    array([590775.63964739])




```python
# to calculate the model

reg.score(X, y) * 100
```




    95.73929037221873



## ONE HOT ENCODING METHOD FROM SKLEARN


```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
```


```python
df_le = df
df_le
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
      <th>town</th>
      <th>area</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>monroe township</td>
      <td>2600</td>
      <td>550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>monroe township</td>
      <td>3000</td>
      <td>565000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>monroe township</td>
      <td>3200</td>
      <td>610000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>monroe township</td>
      <td>3600</td>
      <td>680000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>monroe township</td>
      <td>4000</td>
      <td>725000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>west windsor</td>
      <td>2600</td>
      <td>585000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>west windsor</td>
      <td>2800</td>
      <td>615000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>west windsor</td>
      <td>3300</td>
      <td>650000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>west windsor</td>
      <td>3600</td>
      <td>710000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>robinsville</td>
      <td>2600</td>
      <td>575000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>robinsville</td>
      <td>2900</td>
      <td>600000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>robinsville</td>
      <td>3100</td>
      <td>620000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>robinsville</td>
      <td>3600</td>
      <td>695000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_le.town = le.fit_transform(df_le.town)
df_le.town
```




    0     0
    1     0
    2     0
    3     0
    4     0
    5     2
    6     2
    7     2
    8     2
    9     1
    10    1
    11    1
    12    1
    Name: town, dtype: int64




```python
df_le
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
      <th>town</th>
      <th>area</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2600</td>
      <td>550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3000</td>
      <td>565000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3200</td>
      <td>610000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3600</td>
      <td>680000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4000</td>
      <td>725000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>2600</td>
      <td>585000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>2800</td>
      <td>615000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>3300</td>
      <td>650000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>3600</td>
      <td>710000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2600</td>
      <td>575000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2900</td>
      <td>600000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>3100</td>
      <td>620000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>3600</td>
      <td>695000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df_le.drop(columns = ['price'])
y = df_le['price']

X.shape, y.shape
```




    ((13, 2), (13,))




```python
reg.coef_
```




    array([   126.89744141, -40013.97548914, -14327.56396474])




```python
model.coef_
```




    array([   126.89744141, -40013.97548914, -14327.56396474])




```python
model.fit(X, y)
```




<style>#sk-container-id-15 {color: black;background-color: white;}#sk-container-id-15 pre{padding: 0;}#sk-container-id-15 div.sk-toggleable {background-color: white;}#sk-container-id-15 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-15 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-15 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-15 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-15 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-15 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-15 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-15 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-15 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-15 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-15 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-15 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-15 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-15 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-15 div.sk-item {position: relative;z-index: 1;}#sk-container-id-15 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-15 div.sk-item::before, #sk-container-id-15 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-15 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-15 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-15 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-15 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-15 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-15 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-15 div.sk-label-container {text-align: center;}#sk-container-id-15 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-15 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-15" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" checked><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
model.coef_
```




    array([20112.74367181,   126.05469887])




```python
model.predict([[2, 3300]])
```

    /usr/local/lib/python3.11/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    




    array([670283.67762584])




```python
X = df_le[['town', 'area']].values
X
```




    array([[   0, 2600],
           [   0, 3000],
           [   0, 3200],
           [   0, 3600],
           [   0, 4000],
           [   2, 2600],
           [   2, 2800],
           [   2, 3300],
           [   2, 3600],
           [   1, 2600],
           [   1, 2900],
           [   1, 3100],
           [   1, 3600]])




```python
y = df_le.price
```


```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
```


```python
X = ohe.fit_transform(X[:, 0:1]).toarray()
X
```




    array([[1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 1., 0.]])




```python
X = X[:, 1:]
X
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.]])




```python
model.fit(X, y)
```




<style>#sk-container-id-16 {color: black;background-color: white;}#sk-container-id-16 pre{padding: 0;}#sk-container-id-16 div.sk-toggleable {background-color: white;}#sk-container-id-16 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-16 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-16 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-16 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-16 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-16 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-16 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-16 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-16 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-16 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-16 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-16 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-16 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-16 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-16 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-16 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-16 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-16 div.sk-item {position: relative;z-index: 1;}#sk-container-id-16 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-16 div.sk-item::before, #sk-container-id-16 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-16 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-16 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-16 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-16 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-16 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-16 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-16 div.sk-label-container {text-align: center;}#sk-container-id-16 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-16 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-16" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" checked><label for="sk-estimator-id-16" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
model.coef_
```




    array([-3500., 14000.])



## EXERCISE


```python
df = pd.read_csv("/kaggle/input/carprices/carprices.csv")
df.head()
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
      <th>Car Model</th>
      <th>Mileage</th>
      <th>Sell Price($)</th>
      <th>Age(yrs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW X5</td>
      <td>69000</td>
      <td>18000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW X5</td>
      <td>35000</td>
      <td>34000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW X5</td>
      <td>57000</td>
      <td>26100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW X5</td>
      <td>22500</td>
      <td>40000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW X5</td>
      <td>46000</td>
      <td>31500</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Car Model'].value_counts()
```




    Car Model
    BMW X5                   5
    Audi A5                  4
    Mercedez Benz C class    4
    Name: count, dtype: int64



## GET DUMMIES FUNCTION


```python
dummies = pd.get_dummies(df['Car Model']).astype(int)
dummies
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
      <th>Audi A5</th>
      <th>BMW X5</th>
      <th>Mercedez Benz C class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(columns = ['Car Model'])

final = pd.concat([df, dummies], axis = 1)
final.head(4)
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
      <th>Mileage</th>
      <th>Sell Price($)</th>
      <th>Age(yrs)</th>
      <th>Audi A5</th>
      <th>BMW X5</th>
      <th>Mercedez Benz C class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69000</td>
      <td>18000</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35000</td>
      <td>34000</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57000</td>
      <td>26100</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22500</td>
      <td>40000</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = final.drop(columns = ['Sell Price($)'])
y = final['Sell Price($)']

X.shape, y.shape
```




    ((13, 5), (13,))




```python
y
```




    0     18000
    1     34000
    2     26100
    3     40000
    4     31500
    5     29400
    6     32000
    7     19300
    8     12000
    9     22000
    10    20000
    11    21000
    12    33000
    Name: Sell Price($), dtype: int64




```python
X
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
      <th>Mileage</th>
      <th>Age(yrs)</th>
      <th>Audi A5</th>
      <th>BMW X5</th>
      <th>Mercedez Benz C class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69000</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35000</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57000</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22500</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46000</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>59000</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>52000</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>72000</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>91000</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>67000</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>83000</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>79000</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>59000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
```




<style>#sk-container-id-17 {color: black;background-color: white;}#sk-container-id-17 pre{padding: 0;}#sk-container-id-17 div.sk-toggleable {background-color: white;}#sk-container-id-17 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-17 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-17 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-17 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-17 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-17 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-17 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-17 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-17 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-17 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-17 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-17 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-17 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-17 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-17 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-17 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-17 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-17 div.sk-item {position: relative;z-index: 1;}#sk-container-id-17 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-17 div.sk-item::before, #sk-container-id-17 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-17 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-17 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-17 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-17 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-17 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-17 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-17 div.sk-label-container {text-align: center;}#sk-container-id-17 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-17 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-17" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" checked><label for="sk-estimator-id-17" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
model.coef_, model.intercept_
```




    (array([-3.70122094e-01, -1.33245363e+03,  6.10375284e+02, -3.67429130e+03,
             3.06391602e+03]),
     55912.70994756205)




```python
model.predict([[6900, 6, 0,1 ,0]])
```

    /usr/local/lib/python3.11/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    




    array([41689.85442592])




```python
model.predict([[7900, 7, 0, 0 ,1]])
```

    /usr/local/lib/python3.11/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    




    array([46725.48602958])




```python
model.score(X, y)
```




    0.9417050937281082




```python
import pickle

with open('model_name_for_pickle', 'wb') as f:
    pickle.dump(model, f)
```


```python
with open('model_name_for_pickle', 'rb') as f:
    reg = pickle.load(f)
```

## ONE-HOT ENCODING


```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
```


```python
df = pd.read_csv("/kaggle/input/carprices/carprices.csv")
df.head()
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
      <th>Car Model</th>
      <th>Mileage</th>
      <th>Sell Price($)</th>
      <th>Age(yrs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW X5</td>
      <td>69000</td>
      <td>18000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW X5</td>
      <td>35000</td>
      <td>34000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW X5</td>
      <td>57000</td>
      <td>26100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW X5</td>
      <td>22500</td>
      <td>40000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW X5</td>
      <td>46000</td>
      <td>31500</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Car Model'] = le.fit_transform(df['Car Model'])

df['Car Model']
```




    0     1
    1     1
    2     1
    3     1
    4     1
    5     0
    6     0
    7     0
    8     0
    9     2
    10    2
    11    2
    12    2
    Name: Car Model, dtype: int64




```python
df
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
      <th>Car Model</th>
      <th>Mileage</th>
      <th>Sell Price($)</th>
      <th>Age(yrs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>69000</td>
      <td>18000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>35000</td>
      <td>34000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>57000</td>
      <td>26100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>22500</td>
      <td>40000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>46000</td>
      <td>31500</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>59000</td>
      <td>29400</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>52000</td>
      <td>32000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>72000</td>
      <td>19300</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>91000</td>
      <td>12000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>67000</td>
      <td>22000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>83000</td>
      <td>20000</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>79000</td>
      <td>21000</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>59000</td>
      <td>33000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df[['Car Model','Mileage', 'Age(yrs)']].values
X
```




    array([[    1, 69000,     6],
           [    1, 35000,     3],
           [    1, 57000,     5],
           [    1, 22500,     2],
           [    1, 46000,     4],
           [    0, 59000,     5],
           [    0, 52000,     5],
           [    0, 72000,     6],
           [    0, 91000,     8],
           [    2, 67000,     6],
           [    2, 83000,     7],
           [    2, 79000,     7],
           [    2, 59000,     5]])




```python
y = df['Sell Price($)']
y
```




    0     18000
    1     34000
    2     26100
    3     40000
    4     31500
    5     29400
    6     32000
    7     19300
    8     12000
    9     22000
    10    20000
    11    21000
    12    33000
    Name: Sell Price($), dtype: int64




```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
X = ohe.fit_transform(X)
X
```




    <Compressed Sparse Row sparse matrix of dtype 'float64'
    	with 39 stored elements and shape (13, 22)>




```python
model.coef_
```




    array([-3.70122094e-01, -1.33245363e+03,  6.10375284e+02, -3.67429130e+03,
            3.06391602e+03])




```python
model.fit(X, y)
```




<style>#sk-container-id-18 {color: black;background-color: white;}#sk-container-id-18 pre{padding: 0;}#sk-container-id-18 div.sk-toggleable {background-color: white;}#sk-container-id-18 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-18 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-18 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-18 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-18 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-18 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-18 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-18 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-18 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-18 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-18 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-18 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-18 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-18 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-18 div.sk-item {position: relative;z-index: 1;}#sk-container-id-18 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-18 div.sk-item::before, #sk-container-id-18 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-18 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-18 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-18 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-18 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-18 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-18 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-18 div.sk-label-container {text-align: center;}#sk-container-id-18 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-18 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-18" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" checked><label for="sk-estimator-id-18" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
model.score(X, y)
```




    1.0


