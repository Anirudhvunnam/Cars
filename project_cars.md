```python
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```


```python
#Reading the data from the csv file
cars_data = pd.read_csv('CarPrice_Assignment.csv')
cars_data
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>alfa-romero giulia</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>alfa-romero stelvio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>alfa-romero Quadrifoglio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>audi 100 ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>audi 100ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>200</th>
      <td>201</td>
      <td>-1</td>
      <td>volvo 145e (sw)</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>23</td>
      <td>28</td>
      <td>16845.0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>202</td>
      <td>-1</td>
      <td>volvo 144ea</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>8.7</td>
      <td>160</td>
      <td>5300</td>
      <td>19</td>
      <td>25</td>
      <td>19045.0</td>
    </tr>
    <tr>
      <th>202</th>
      <td>203</td>
      <td>-1</td>
      <td>volvo 244dl</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>173</td>
      <td>mpfi</td>
      <td>3.58</td>
      <td>2.87</td>
      <td>8.8</td>
      <td>134</td>
      <td>5500</td>
      <td>18</td>
      <td>23</td>
      <td>21485.0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>204</td>
      <td>-1</td>
      <td>volvo 246</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>145</td>
      <td>idi</td>
      <td>3.01</td>
      <td>3.40</td>
      <td>23.0</td>
      <td>106</td>
      <td>4800</td>
      <td>26</td>
      <td>27</td>
      <td>22470.0</td>
    </tr>
    <tr>
      <th>204</th>
      <td>205</td>
      <td>-1</td>
      <td>volvo 264gl</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>19</td>
      <td>25</td>
      <td>22625.0</td>
    </tr>
  </tbody>
</table>
<p>205 rows × 26 columns</p>
</div>




```python
#Finding the shape of the data
cars_data.shape
```




    (205, 26)




```python
#Describing the data
cars_data.describe()
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>wheelbase</th>
      <th>carlength</th>
      <th>carwidth</th>
      <th>carheight</th>
      <th>curbweight</th>
      <th>enginesize</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>103.000000</td>
      <td>0.834146</td>
      <td>98.756585</td>
      <td>174.049268</td>
      <td>65.907805</td>
      <td>53.724878</td>
      <td>2555.565854</td>
      <td>126.907317</td>
      <td>3.329756</td>
      <td>3.255415</td>
      <td>10.142537</td>
      <td>104.117073</td>
      <td>5125.121951</td>
      <td>25.219512</td>
      <td>30.751220</td>
      <td>13276.710571</td>
    </tr>
    <tr>
      <th>std</th>
      <td>59.322565</td>
      <td>1.245307</td>
      <td>6.021776</td>
      <td>12.337289</td>
      <td>2.145204</td>
      <td>2.443522</td>
      <td>520.680204</td>
      <td>41.642693</td>
      <td>0.270844</td>
      <td>0.313597</td>
      <td>3.972040</td>
      <td>39.544167</td>
      <td>476.985643</td>
      <td>6.542142</td>
      <td>6.886443</td>
      <td>7988.852332</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-2.000000</td>
      <td>86.600000</td>
      <td>141.100000</td>
      <td>60.300000</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>2.540000</td>
      <td>2.070000</td>
      <td>7.000000</td>
      <td>48.000000</td>
      <td>4150.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>5118.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>94.500000</td>
      <td>166.300000</td>
      <td>64.100000</td>
      <td>52.000000</td>
      <td>2145.000000</td>
      <td>97.000000</td>
      <td>3.150000</td>
      <td>3.110000</td>
      <td>8.600000</td>
      <td>70.000000</td>
      <td>4800.000000</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>7788.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>103.000000</td>
      <td>1.000000</td>
      <td>97.000000</td>
      <td>173.200000</td>
      <td>65.500000</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>3.310000</td>
      <td>3.290000</td>
      <td>9.000000</td>
      <td>95.000000</td>
      <td>5200.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>10295.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>154.000000</td>
      <td>2.000000</td>
      <td>102.400000</td>
      <td>183.100000</td>
      <td>66.900000</td>
      <td>55.500000</td>
      <td>2935.000000</td>
      <td>141.000000</td>
      <td>3.580000</td>
      <td>3.410000</td>
      <td>9.400000</td>
      <td>116.000000</td>
      <td>5500.000000</td>
      <td>30.000000</td>
      <td>34.000000</td>
      <td>16503.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>205.000000</td>
      <td>3.000000</td>
      <td>120.900000</td>
      <td>208.100000</td>
      <td>72.300000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>3.940000</td>
      <td>4.170000</td>
      <td>23.000000</td>
      <td>288.000000</td>
      <td>6600.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>45400.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Finding the information of the data
cars_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 26 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   car_ID            205 non-null    int64  
     1   symboling         205 non-null    int64  
     2   CarName           205 non-null    object 
     3   fueltype          205 non-null    object 
     4   aspiration        205 non-null    object 
     5   doornumber        205 non-null    object 
     6   carbody           205 non-null    object 
     7   drivewheel        205 non-null    object 
     8   enginelocation    205 non-null    object 
     9   wheelbase         205 non-null    float64
     10  carlength         205 non-null    float64
     11  carwidth          205 non-null    float64
     12  carheight         205 non-null    float64
     13  curbweight        205 non-null    int64  
     14  enginetype        205 non-null    object 
     15  cylindernumber    205 non-null    object 
     16  enginesize        205 non-null    int64  
     17  fuelsystem        205 non-null    object 
     18  boreratio         205 non-null    float64
     19  stroke            205 non-null    float64
     20  compressionratio  205 non-null    float64
     21  horsepower        205 non-null    int64  
     22  peakrpm           205 non-null    int64  
     23  citympg           205 non-null    int64  
     24  highwaympg        205 non-null    int64  
     25  price             205 non-null    float64
    dtypes: float64(8), int64(8), object(10)
    memory usage: 41.8+ KB
    


```python
#Finding the null values in the dataset
cars_data.isnull().sum()
```




    car_ID              0
    symboling           0
    CarName             0
    fueltype            0
    aspiration          0
    doornumber          0
    carbody             0
    drivewheel          0
    enginelocation      0
    wheelbase           0
    carlength           0
    carwidth            0
    carheight           0
    curbweight          0
    enginetype          0
    cylindernumber      0
    enginesize          0
    fuelsystem          0
    boreratio           0
    stroke              0
    compressionratio    0
    horsepower          0
    peakrpm             0
    citympg             0
    highwaympg          0
    price               0
    dtype: int64




```python
#Finding the duplicate rows in the dataset
cars_data.duplicated(subset = ['car_ID']).sum()
```




    0




```python
#Removing the id column
cars_data = cars_data.drop(['car_ID'], axis = 1)
cars_data.head()
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
      <th>symboling</th>
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>carlength</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>alfa-romero giulia</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>alfa-romero stelvio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>alfa-romero Quadrifoglio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>audi 100 ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>audi 100ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
#Counting the cars Company wise
cars_data['CarName'].value_counts()
```




    toyota corona         6
    peugeot 504           6
    toyota corolla        6
    subaru dl             4
    honda civic           3
                         ..
    nissan dayz           1
    honda civic (auto)    1
    mazda glc custom l    1
    renault 5 gtl         1
    dodge rampage         1
    Name: CarName, Length: 147, dtype: int64




```python
#Removing the Car names and keeping only the company names
cars_data['CarsCompany'] = cars_data['CarName'].apply(lambda x:x.strip().split(' ')[0])
cars_data = cars_data.drop(['CarName'], axis = 1)
cars_data.head()
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
      <th>symboling</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>carlength</th>
      <th>carwidth</th>
      <th>...</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
      <th>CarsCompany</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>...</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
      <td>alfa-romero</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>...</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
      <td>alfa-romero</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>...</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
      <td>alfa-romero</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>...</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>...</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
      <td>audi</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
cars_data['CarsCompany'].value_counts()
```




    toyota         31
    nissan         17
    mazda          15
    mitsubishi     13
    honda          13
    subaru         12
    peugeot        11
    volvo          11
    volkswagen      9
    dodge           9
    buick           8
    bmw             8
    audi            7
    plymouth        7
    saab            6
    isuzu           4
    porsche         4
    alfa-romero     3
    jaguar          3
    chevrolet       3
    renault         2
    vw              2
    maxda           2
    mercury         1
    vokswagen       1
    Nissan          1
    porcshce        1
    toyouta         1
    Name: CarsCompany, dtype: int64




```python
#Correcting the spellings of the company names
cars_data['CarsCompany'].replace('toyouta', 'toyota', inplace = True)
cars_data['CarsCompany'].replace('Nissan', 'nissan', inplace = True)
cars_data['CarsCompany'].replace('maxda', 'mazda', inplace = True)
cars_data['CarsCompany'].replace('vokswagen', 'volkswagen', inplace = True)
cars_data['CarsCompany'].replace('vw', 'volkswagen', inplace = True)
cars_data['CarsCompany'].replace('porcshce', 'porsche', inplace = True)
#Counting the number of cars of each company
cars_data['CarsCompany'].value_counts()
```




    toyota         32
    nissan         18
    mazda          17
    honda          13
    mitsubishi     13
    subaru         12
    volkswagen     12
    volvo          11
    peugeot        11
    dodge           9
    buick           8
    bmw             8
    plymouth        7
    audi            7
    saab            6
    porsche         5
    isuzu           4
    jaguar          3
    chevrolet       3
    alfa-romero     3
    renault         2
    mercury         1
    Name: CarsCompany, dtype: int64




```python
#Plotting vaious properties of the cars
plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
sns.distplot(cars_data['symboling'])
plt.subplot(2,2,2)
sns.distplot(cars_data['wheelbase'])
plt.subplot(2,2,3)
sns.distplot(cars_data['carlength'])
plt.subplot(2,2,4)
sns.distplot(cars_data['carwidth'])
```




    <AxesSubplot:xlabel='carwidth', ylabel='Density'>




    
![png](output_12_1.png)
    



```python
#Plotting the Heatmap to find the correlations
plt.figure(figsize = (20,20))
sns.heatmap(cars_data.corr(), annot = True)
plt.show()
```


    
![png](output_13_0.png)
    



```python
#Cars prices based on the Cars Companies
plt.figure(figsize = (20,20))
sns.boxplot(x = 'CarsCompany', y = 'price', data = cars_data)
```




    <AxesSubplot:xlabel='CarsCompany', ylabel='price'>




    
![png](output_14_1.png)
    



```python
cars_data = cars_data.drop(['doornumber', 'cylindernumber'], axis = 1)
cars_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 23 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   symboling         205 non-null    int64  
     1   fueltype          205 non-null    object 
     2   aspiration        205 non-null    object 
     3   carbody           205 non-null    object 
     4   drivewheel        205 non-null    object 
     5   enginelocation    205 non-null    object 
     6   wheelbase         205 non-null    float64
     7   carlength         205 non-null    float64
     8   carwidth          205 non-null    float64
     9   carheight         205 non-null    float64
     10  curbweight        205 non-null    int64  
     11  enginetype        205 non-null    object 
     12  enginesize        205 non-null    int64  
     13  fuelsystem        205 non-null    object 
     14  boreratio         205 non-null    float64
     15  stroke            205 non-null    float64
     16  compressionratio  205 non-null    float64
     17  horsepower        205 non-null    int64  
     18  peakrpm           205 non-null    int64  
     19  citympg           205 non-null    int64  
     20  highwaympg        205 non-null    int64  
     21  price             205 non-null    float64
     22  CarsCompany       205 non-null    object 
    dtypes: float64(8), int64(7), object(8)
    memory usage: 37.0+ KB
    


```python
categories = cars_data.select_dtypes(include = ['object'])
categories.head()
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
      <th>fueltype</th>
      <th>aspiration</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>enginetype</th>
      <th>fuelsystem</th>
      <th>CarsCompany</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>gas</td>
      <td>std</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>dohc</td>
      <td>mpfi</td>
      <td>alfa-romero</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gas</td>
      <td>std</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>dohc</td>
      <td>mpfi</td>
      <td>alfa-romero</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gas</td>
      <td>std</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>ohcv</td>
      <td>mpfi</td>
      <td>alfa-romero</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gas</td>
      <td>std</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>ohc</td>
      <td>mpfi</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gas</td>
      <td>std</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>ohc</td>
      <td>mpfi</td>
      <td>audi</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Creating the dummy variables
dummies = pd.get_dummies(categories, drop_first = True)
cars_df = pd.concat([cars_data, dummies], axis = 1)
cars_df = cars_df.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation','enginetype', 'fuelsystem', 'CarsCompany'], axis = 1)
cars_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 58 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   symboling               205 non-null    int64  
     1   wheelbase               205 non-null    float64
     2   carlength               205 non-null    float64
     3   carwidth                205 non-null    float64
     4   carheight               205 non-null    float64
     5   curbweight              205 non-null    int64  
     6   enginesize              205 non-null    int64  
     7   boreratio               205 non-null    float64
     8   stroke                  205 non-null    float64
     9   compressionratio        205 non-null    float64
     10  horsepower              205 non-null    int64  
     11  peakrpm                 205 non-null    int64  
     12  citympg                 205 non-null    int64  
     13  highwaympg              205 non-null    int64  
     14  price                   205 non-null    float64
     15  fueltype_gas            205 non-null    uint8  
     16  aspiration_turbo        205 non-null    uint8  
     17  carbody_hardtop         205 non-null    uint8  
     18  carbody_hatchback       205 non-null    uint8  
     19  carbody_sedan           205 non-null    uint8  
     20  carbody_wagon           205 non-null    uint8  
     21  drivewheel_fwd          205 non-null    uint8  
     22  drivewheel_rwd          205 non-null    uint8  
     23  enginelocation_rear     205 non-null    uint8  
     24  enginetype_dohcv        205 non-null    uint8  
     25  enginetype_l            205 non-null    uint8  
     26  enginetype_ohc          205 non-null    uint8  
     27  enginetype_ohcf         205 non-null    uint8  
     28  enginetype_ohcv         205 non-null    uint8  
     29  enginetype_rotor        205 non-null    uint8  
     30  fuelsystem_2bbl         205 non-null    uint8  
     31  fuelsystem_4bbl         205 non-null    uint8  
     32  fuelsystem_idi          205 non-null    uint8  
     33  fuelsystem_mfi          205 non-null    uint8  
     34  fuelsystem_mpfi         205 non-null    uint8  
     35  fuelsystem_spdi         205 non-null    uint8  
     36  fuelsystem_spfi         205 non-null    uint8  
     37  CarsCompany_audi        205 non-null    uint8  
     38  CarsCompany_bmw         205 non-null    uint8  
     39  CarsCompany_buick       205 non-null    uint8  
     40  CarsCompany_chevrolet   205 non-null    uint8  
     41  CarsCompany_dodge       205 non-null    uint8  
     42  CarsCompany_honda       205 non-null    uint8  
     43  CarsCompany_isuzu       205 non-null    uint8  
     44  CarsCompany_jaguar      205 non-null    uint8  
     45  CarsCompany_mazda       205 non-null    uint8  
     46  CarsCompany_mercury     205 non-null    uint8  
     47  CarsCompany_mitsubishi  205 non-null    uint8  
     48  CarsCompany_nissan      205 non-null    uint8  
     49  CarsCompany_peugeot     205 non-null    uint8  
     50  CarsCompany_plymouth    205 non-null    uint8  
     51  CarsCompany_porsche     205 non-null    uint8  
     52  CarsCompany_renault     205 non-null    uint8  
     53  CarsCompany_saab        205 non-null    uint8  
     54  CarsCompany_subaru      205 non-null    uint8  
     55  CarsCompany_toyota      205 non-null    uint8  
     56  CarsCompany_volkswagen  205 non-null    uint8  
     57  CarsCompany_volvo       205 non-null    uint8  
    dtypes: float64(8), int64(7), uint8(43)
    memory usage: 32.8 KB
    


```python
#Making the input set
X = cars_df.drop(['price'], axis = 1)
#Creating the output set
y = cars_df['price']
```


```python
# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
```


```python
# instantiate
lm = LinearRegression()
# fit
lm.fit(X_train, y_train)
# predict 
y_pred = lm.predict(X_test)
# metrics
from sklearn.metrics import r2_score
print(r2_score(y_true=y_test, y_pred=y_pred))
```

    0.9188472898532467
    


```python
# RFE with 15 features
from sklearn.feature_selection import RFE

# RFE with 15 features
lm = LinearRegression()
rfe_15 = RFE(lm, 15)

# fit with 15 features
rfe_15.fit(X_train, y_train)

# Printing the boolean results
print(rfe_15.support_)           
print(rfe_15.ranking_)  
```

    [False False False False False False False False False False False False
     False False False False  True False False False  True False  True False
      True False  True False  True False  True False False False False False
     False  True  True False False False  True  True False False False False
      True False  True False  True  True False False False]
    [34 29 31  6 28 39 30 21 27 13 37 40 38 32 11  8  1  2  5  3  1 35  1 36
      1 33  1  7  1 24  1 12 42 23 26 41 18  1  1 22 17 16  1  1 10 43  9 19
      1 20  1  4  1  1 15 14 25]
    


```python
# making predictions using rfe model
y_pred = rfe_15.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))
```

    0.7642408874669062
    


```python
# RFE with 6 features
from sklearn.feature_selection import RFE

# RFE with 6 features
lm = LinearRegression()
rfe_6 = RFE(lm, 6)

# fit with 6 features
rfe_6.fit(X_train, y_train)

# predict
y_pred = rfe_6.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))
```

    0.7049098734402879
    


```python
# import statsmodels
import statsmodels.api as sm  

# subset the features selected by rfe_15
col_15 = X_train.columns[rfe_15.support_]

# subsetting training data for 15 selected columns
X_train_rfe_15 = X_train[col_15]

# add a constant to the model
X_train_rfe_15 = sm.add_constant(X_train_rfe_15)
X_train_rfe_15.head()
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
      <th>const</th>
      <th>carbody_hardtop</th>
      <th>drivewheel_fwd</th>
      <th>enginelocation_rear</th>
      <th>enginetype_l</th>
      <th>enginetype_ohcf</th>
      <th>enginetype_rotor</th>
      <th>fuelsystem_4bbl</th>
      <th>CarsCompany_bmw</th>
      <th>CarsCompany_buick</th>
      <th>CarsCompany_isuzu</th>
      <th>CarsCompany_jaguar</th>
      <th>CarsCompany_peugeot</th>
      <th>CarsCompany_porsche</th>
      <th>CarsCompany_saab</th>
      <th>CarsCompany_subaru</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>122</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>166</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>199</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fitting the model with 15 variables
lm_15 = sm.OLS(y_train, X_train_rfe_15).fit()   
print(lm_15.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.776
    Model:                            OLS   Adj. R-squared:                  0.752
    Method:                 Least Squares   F-statistic:                     31.72
    Date:                Wed, 08 Jun 2022   Prob (F-statistic):           7.13e-35
    Time:                        23:51:44   Log-Likelihood:                -1377.0
    No. Observations:                 143   AIC:                             2784.
    Df Residuals:                     128   BIC:                             2829.
    Df Model:                          14                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    const                1.505e+04    755.176     19.933      0.000    1.36e+04    1.65e+04
    carbody_hardtop     -2360.3726   2787.824     -0.847      0.399   -7876.560    3155.814
    drivewheel_fwd      -5937.6608    847.065     -7.010      0.000   -7613.723   -4261.598
    enginelocation_rear  7335.4638   3713.881      1.975      0.050     -13.084    1.47e+04
    enginetype_l        -3964.5422   3915.661     -1.012      0.313   -1.17e+04    3783.261
    enginetype_ohcf      2343.6588   1875.149      1.250      0.214   -1366.644    6053.962
    enginetype_rotor      591.7970   3962.170      0.149      0.882   -7248.033    8431.627
    fuelsystem_4bbl     -3500.0000   4491.251     -0.779      0.437   -1.24e+04    5386.707
    CarsCompany_bmw      1.182e+04   1758.325      6.725      0.000    8345.150    1.53e+04
    CarsCompany_buick    1.587e+04   1896.310      8.371      0.000    1.21e+04    1.96e+04
    CarsCompany_isuzu   -2888.7625   2282.414     -1.266      0.208   -7404.909    1627.384
    CarsCompany_jaguar   1.955e+04   2369.203      8.250      0.000    1.49e+04    2.42e+04
    CarsCompany_peugeot  4450.0892   4211.541      1.057      0.293   -3883.165    1.28e+04
    CarsCompany_porsche  1.166e+04   2852.112      4.087      0.000    6012.657    1.73e+04
    CarsCompany_saab     6211.1244   2290.575      2.712      0.008    1678.831    1.07e+04
    CarsCompany_subaru  -4991.8050   2038.008     -2.449      0.016   -9024.353    -959.257
    ==============================================================================
    Omnibus:                       30.291   Durbin-Watson:                   2.084
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               62.310
    Skew:                           0.916   Prob(JB):                     2.95e-14
    Kurtosis:                       5.664   Cond. No.                     3.81e+16
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.42e-31. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    


```python
# making predictions using rfe_15 sm model
X_test_rfe_15 = X_test[col_15]


# # Adding a constant variable 
X_test_rfe_15 = sm.add_constant(X_test_rfe_15, has_constant='add')
X_test_rfe_15.info()


# # Making predictions
y_pred = lm_15.predict(X_test_rfe_15)

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 62 entries, 160 to 128
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   const                62 non-null     float64
     1   carbody_hardtop      62 non-null     uint8  
     2   drivewheel_fwd       62 non-null     uint8  
     3   enginelocation_rear  62 non-null     uint8  
     4   enginetype_l         62 non-null     uint8  
     5   enginetype_ohcf      62 non-null     uint8  
     6   enginetype_rotor     62 non-null     uint8  
     7   fuelsystem_4bbl      62 non-null     uint8  
     8   CarsCompany_bmw      62 non-null     uint8  
     9   CarsCompany_buick    62 non-null     uint8  
     10  CarsCompany_isuzu    62 non-null     uint8  
     11  CarsCompany_jaguar   62 non-null     uint8  
     12  CarsCompany_peugeot  62 non-null     uint8  
     13  CarsCompany_porsche  62 non-null     uint8  
     14  CarsCompany_saab     62 non-null     uint8  
     15  CarsCompany_subaru   62 non-null     uint8  
    dtypes: float64(1), uint8(15)
    memory usage: 1.9 KB
    


```python
# r-squared
r2_score(y_test, y_pred)
```




    0.7642408874669054




```python
# subset the features selected by rfe_6
col_6 = X_train.columns[rfe_6.support_]

# subsetting training data for 6 selected columns
X_train_rfe_6 = X_train[col_6]

# add a constant to the model
X_train_rfe_6 = sm.add_constant(X_train_rfe_6)


# fitting the model with 6 variables
lm_6 = sm.OLS(y_train, X_train_rfe_6).fit()   
print(lm_6.summary())


# making predictions using rfe_6 sm model
X_test_rfe_6 = X_test[col_6]


# Adding a constant  
X_test_rfe_6 = sm.add_constant(X_test_rfe_6, has_constant='add')
X_test_rfe_6.info()


# # Making predictions
y_pred = lm_6.predict(X_test_rfe_6)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.654
    Model:                            OLS   Adj. R-squared:                  0.638
    Method:                 Least Squares   F-statistic:                     42.77
    Date:                Wed, 08 Jun 2022   Prob (F-statistic):           5.15e-29
    Time:                        23:53:25   Log-Likelihood:                -1408.3
    No. Observations:                 143   AIC:                             2831.
    Df Residuals:                     136   BIC:                             2851.
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    const                1.068e+04    423.349     25.233      0.000    9845.129    1.15e+04
    enginelocation_rear  7318.7500   5750.378      1.273      0.205   -4052.973    1.87e+04
    CarsCompany_bmw       1.62e+04   1962.987      8.250      0.000    1.23e+04    2.01e+04
    CarsCompany_buick    2.025e+04   2141.994      9.452      0.000     1.6e+04    2.45e+04
    CarsCompany_jaguar   2.392e+04   2743.613      8.718      0.000    1.85e+04    2.93e+04
    CarsCompany_porsche  1.603e+04   3346.865      4.789      0.000    9408.294    2.26e+04
    CarsCompany_saab     4644.3401   2743.613      1.693      0.093    -781.322    1.01e+04
    ==============================================================================
    Omnibus:                       17.704   Durbin-Watson:                   2.313
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               20.414
    Skew:                           0.907   Prob(JB):                     3.69e-05
    Kurtosis:                       3.368   Cond. No.                         15.7
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 62 entries, 160 to 128
    Data columns (total 7 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   const                62 non-null     float64
     1   enginelocation_rear  62 non-null     uint8  
     2   CarsCompany_bmw      62 non-null     uint8  
     3   CarsCompany_buick    62 non-null     uint8  
     4   CarsCompany_jaguar   62 non-null     uint8  
     5   CarsCompany_porsche  62 non-null     uint8  
     6   CarsCompany_saab     62 non-null     uint8  
    dtypes: float64(1), uint8(6)
    memory usage: 1.3 KB
    


```python
# r2_score for 6 variables
r2_score(y_test, y_pred)
```




    0.7049098734402888




```python
n_features_list = list(range(4, 20))
adjusted_r2 = []
r2 = []
test_r2 = []

for n_features in range(4, 20):

    # RFE with n features
    lm = LinearRegression()

    # specify number of features
    rfe_n = RFE(lm, n_features)

    # fit with n features
    rfe_n.fit(X_train, y_train)

    # subset the features selected by rfe_6
    col_n = X_train.columns[rfe_n.support_]

    # subsetting training data for 6 selected columns
    X_train_rfe_n = X_train[col_n]

    # add a constant to the model
    X_train_rfe_n = sm.add_constant(X_train_rfe_n)


    # fitting the model with 6 variables
    lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
    adjusted_r2.append(lm_n.rsquared_adj)
    r2.append(lm_n.rsquared)
    
    
    # making predictions using rfe_15 sm model
    X_test_rfe_n = X_test[col_n]


    # # Adding a constant variable 
    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')



    # # Making predictions
    y_pred = lm_n.predict(X_test_rfe_n)
    
    test_r2.append(r2_score(y_test, y_pred))
```


```python
# plotting adjusted_r2 against n_features
plt.figure(figsize=(10, 8))
plt.plot(n_features_list, adjusted_r2, label="adjusted_r2")
plt.plot(n_features_list, r2, label="train_r2")
plt.plot(n_features_list, test_r2, label="test_r2")
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_31_0.png)
    



```python
# RFE with n features
lm = LinearRegression()

n_features = 6

# specify number of features
rfe_n = RFE(lm, n_features)

# fit with n features
rfe_n.fit(X_train, y_train)

# subset the features selected by rfe_6
col_n = X_train.columns[rfe_n.support_]

# subsetting training data for 6 selected columns
X_train_rfe_n = X_train[col_n]

# add a constant to the model
X_train_rfe_n = sm.add_constant(X_train_rfe_n)


# fitting the model with 6 variables
lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
adjusted_r2.append(lm_n.rsquared_adj)
r2.append(lm_n.rsquared)


# making predictions using rfe_15 sm model
X_test_rfe_n = X_test[col_n]


# # Adding a constant variable 
X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')



# # Making predictions
y_pred = lm_n.predict(X_test_rfe_n)

test_r2.append(r2_score(y_test, y_pred))
```


```python
# summary
lm_n.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.654</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.638</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   42.77</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 08 Jun 2022</td> <th>  Prob (F-statistic):</th> <td>5.15e-29</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:57:49</td>     <th>  Log-Likelihood:    </th> <td> -1408.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   143</td>      <th>  AIC:               </th> <td>   2831.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   136</td>      <th>  BIC:               </th> <td>   2851.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>               <td> 1.068e+04</td> <td>  423.349</td> <td>   25.233</td> <td> 0.000</td> <td> 9845.129</td> <td> 1.15e+04</td>
</tr>
<tr>
  <th>enginelocation_rear</th> <td> 7318.7500</td> <td> 5750.378</td> <td>    1.273</td> <td> 0.205</td> <td>-4052.973</td> <td> 1.87e+04</td>
</tr>
<tr>
  <th>CarsCompany_bmw</th>     <td>  1.62e+04</td> <td> 1962.987</td> <td>    8.250</td> <td> 0.000</td> <td> 1.23e+04</td> <td> 2.01e+04</td>
</tr>
<tr>
  <th>CarsCompany_buick</th>   <td> 2.025e+04</td> <td> 2141.994</td> <td>    9.452</td> <td> 0.000</td> <td>  1.6e+04</td> <td> 2.45e+04</td>
</tr>
<tr>
  <th>CarsCompany_jaguar</th>  <td> 2.392e+04</td> <td> 2743.613</td> <td>    8.718</td> <td> 0.000</td> <td> 1.85e+04</td> <td> 2.93e+04</td>
</tr>
<tr>
  <th>CarsCompany_porsche</th> <td> 1.603e+04</td> <td> 3346.865</td> <td>    4.789</td> <td> 0.000</td> <td> 9408.294</td> <td> 2.26e+04</td>
</tr>
<tr>
  <th>CarsCompany_saab</th>    <td> 4644.3401</td> <td> 2743.613</td> <td>    1.693</td> <td> 0.093</td> <td> -781.322</td> <td> 1.01e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>17.704</td> <th>  Durbin-Watson:     </th> <td>   2.313</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  20.414</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.907</td> <th>  Prob(JB):          </th> <td>3.69e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.368</td> <th>  Cond. No.          </th> <td>    15.7</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# results 
r2_score(y_test, y_pred)

```




    0.7049098734402888


