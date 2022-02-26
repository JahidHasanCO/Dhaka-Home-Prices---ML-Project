# Dhaka-Home-Prices---ML-Project

---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.7
  nbformat: 4
  nbformat_minor: 5
---

::: {.cell .code execution_count="3"}
``` {.python}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
:::

::: {.cell .code execution_count="4"}
``` {.python}
df = pd.read_csv('dhaka_homeprices.csv')
```
:::

::: {.cell .code execution_count="5"}
``` {.python}
df
```

::: {.output .execute_result execution_count="5"}
```{=html}
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>56500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>61000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>68000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5000</td>
      <td>71000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2500</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2700</td>
      <td>38000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1200</td>
      <td>17000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5000</td>
      <td>100000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="6"}
``` {.python}
df.head()
```

::: {.output .execute_result execution_count="6"}
```{=html}
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>56500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>61000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>68000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
      <td>72000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="8"}
``` {.python}
df.shape
```

::: {.output .execute_result execution_count="8"}
    (10, 2)
:::
:::

::: {.cell .code execution_count="11"}
``` {.python}
df.isnull().any()
```

::: {.output .execute_result execution_count="11"}
    area     False
    price    False
    dtype: bool
:::
:::

::: {.cell .code execution_count="12"}
``` {.python}
df.isnull().sum()
```

::: {.output .execute_result execution_count="12"}
    area     0
    price    0
    dtype: int64
:::
:::

::: {.cell .code execution_count="16"}
``` {.python}
x = df[['area']]
y = df['price']
```
:::

::: {.cell .code execution_count="19"}
``` {.python}
plt.scatter(df['area'],df['price'])
plt.xlabel("Area in square ft")
plt.ylabel("Price in TK")
plt.title("Home Prices in Dhaka")
```

::: {.output .execute_result execution_count="19"}
    Text(0.5, 1.0, 'Home Prices in Dhaka')
:::

::: {.output .display_data}
![](vertopal_509bc42536fd4da1984e3c892e8a6cfa/ef93c0f3311824ee65657a37076fecf614661145.png)
:::
:::

::: {.cell .code execution_count="14"}
``` {.python}
x
```

::: {.output .execute_result execution_count="14"}
```{=html}
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2700</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1200</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="15"}
``` {.python}
y
```

::: {.output .execute_result execution_count="15"}
    0     55000
    1     56500
    2     61000
    3     68000
    4     72000
    5     71000
    6     40000
    7     38000
    8     17000
    9    100000
    Name: price, dtype: int64
:::
:::

::: {.cell .code execution_count="20"}
``` {.python}
from sklearn.model_selection import train_test_split
```
:::

::: {.cell .code execution_count="21"}
``` {.python}
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = .30, random_state=1)
```
:::

::: {.cell .code execution_count="22"}
``` {.python}
xtrain
```

::: {.output .execute_result execution_count="22"}
```{=html}
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2700</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1200</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="23"}
``` {.python}
ytrain
```

::: {.output .execute_result execution_count="23"}
    4    72000
    0    55000
    3    68000
    1    56500
    7    38000
    8    17000
    5    71000
    Name: price, dtype: int64
:::
:::

::: {.cell .code execution_count="24"}
``` {.python}
xtest
```

::: {.output .execute_result execution_count="24"}
```{=html}
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>3200</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2500</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="25"}
``` {.python}
ytest
```

::: {.output .execute_result execution_count="25"}
    2     61000
    9    100000
    6     40000
    Name: price, dtype: int64
:::
:::

::: {.cell .code execution_count="26"}
``` {.python}
from sklearn.linear_model import LinearRegression
```
:::

::: {.cell .code execution_count="27"}
``` {.python}
reg = LinearRegression()
```
:::

::: {.cell .code execution_count="28"}
``` {.python}
reg.fit(xtrain,ytrain)
```

::: {.output .execute_result execution_count="28"}
    LinearRegression()
:::
:::

::: {.cell .code execution_count="29"}
``` {.python}
reg.predict(xtest)
```

::: {.output .execute_result execution_count="29"}
    array([54577.95521897, 81852.07441554, 43971.35330919])
:::
:::

::: {.cell .code execution_count="31"}
``` {.python}
plt.scatter(df['area'],df['price'])
plt.xlabel("Area in square ft")
plt.ylabel("Price in TK")
plt.title("Home Prices in Dhaka")
plt.plot(df.area,reg.predict(df[['area']]))
```

::: {.output .execute_result execution_count="31"}
    [<matplotlib.lines.Line2D at 0x27e0d3dc7f0>]
:::

::: {.output .display_data}
![](vertopal_509bc42536fd4da1984e3c892e8a6cfa/557576f0f606e889c90bbe12fe78ce81664cce67.png)
:::
:::

::: {.cell .code execution_count="32"}
``` {.python}
reg.predict([[3500]])
```

::: {.output .execute_result execution_count="32"}
    array([59123.64175173])
:::
:::

::: {.cell .code execution_count="33"}
``` {.python}
reg.coef_
```

::: {.output .execute_result execution_count="33"}
    array([15.15228844])
:::
:::

::: {.cell .code execution_count="34"}
``` {.python}
reg.intercept_
```

::: {.output .execute_result execution_count="34"}
    6090.63220283173
:::
:::

::: {.cell .code}
``` {.python}
```
:::
