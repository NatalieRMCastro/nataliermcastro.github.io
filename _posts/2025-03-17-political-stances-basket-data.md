---
layout: post
title: "Political Stances - Creating Basket Data"
categories: projects
published: true
in_feed: False
---

# 2. Creating Basket Data + DTM Data

The purpose of this notebook is to generate basket data, which is good for clusering and association roles mining. The desired shape of the table has each row is a document, and the columns are the individual words. Finally, the output will be in a csv format.

The data used in this notebook are from the Count Vectorizer. This is used so that the values in the CountVectorizer can be converted to binary, and then the row can be cleaned.

## 1. Environment Creation

### 1.1 Library Import


```python
import pandas as pd
import os
from tqdm.notebook import tqdm
import csv
```

### 1.2 Data Import


```python
bill_information = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\Bills Lemmed- Count Vectorizer.csv")
```


```python
bill_information.head(2)
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
      <th>Bill Type</th>
      <th>Sponser Affiliation</th>
      <th>Sponser State</th>
      <th>Committees</th>
      <th>aa</th>
      <th>aaa</th>
      <th>aarhu</th>
      <th>ab</th>
      <th>abandon</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>hr</td>
      <td>D</td>
      <td>HI</td>
      <td>House - Natural Resources, Agriculture | Senat...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>hr</td>
      <td>R</td>
      <td>NY</td>
      <td>House - Agriculture</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 15494 columns</p>
</div>




```python
news = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\News Articles Lemmed- Count Vectorizer.csv")
```


```python
news.head(2)
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
      <th>Party</th>
      <th>publisher</th>
      <th>aapi</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abc</th>
      <th>ability</th>
      <th>able</th>
      <th>abolish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Republican</td>
      <td>The Verge</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Republican</td>
      <td>Gizmodo.com</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 2361 columns</p>
</div>




```python
party_platform = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\Party Platform Lemmed- Count Vectorizer.csv")
```


```python
party_platform.head(2)
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
      <th>Unnamed: 0</th>
      <th>Party</th>
      <th>ability</th>
      <th>able</th>
      <th>abortion</th>
      <th>access</th>
      <th>accessible</th>
      <th>according</th>
      <th>accountability</th>
      <th>accountable</th>
      <th>...</th>
      <th>won</th>
      <th>word</th>
      <th>work</th>
      <th>worker</th>
      <th>working</th>
      <th>world</th>
      <th>worship</th>
      <th>worst</th>
      <th>year</th>
      <th>young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Republican</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>15</td>
      <td>2</td>
      <td>15</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Democrat</td>
      <td>7</td>
      <td>13</td>
      <td>13</td>
      <td>72</td>
      <td>15</td>
      <td>1</td>
      <td>6</td>
      <td>14</td>
      <td>...</td>
      <td>10</td>
      <td>1</td>
      <td>101</td>
      <td>79</td>
      <td>81</td>
      <td>69</td>
      <td>4</td>
      <td>4</td>
      <td>149</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 894 columns</p>
</div>



### 1.2.2 Light Data Cleaning

The data utilized in this notebook has already undergone extensive cleaning processes, which may be viewed on [my project page](https://nataliermcastro.github.io/projects/2025/01/14/political-stances-data.html). In this section, the labels will be removed and the unnamed column will be dropped.


```python
bill_information.drop(columns=['Unnamed: 0','Sponser State','Bill Type','Sponser Affiliation','Committees'],inplace=True)
```


```python
bill_information.drop(columns=['helvetica','noto','sego','neue','vh','html','webkit','emoji','blinkmacsystemfont','arial','roboto','ui','serif',
                              'column','font','pad','width','auto','left','height'],inplace=True)
```


```python
news.drop(columns=['Unnamed: 0','Party','publisher'],inplace=True)
```


```python
party_platform.drop(columns=['Unnamed: 0','Party'],inplace=True)
```

## 2. Creating Basket Data

### 2.1 Assembling the Transactions


```python
def transaction_creator(index,basket):
    transcation_dictionary = {key:val for key, val in basket[index].items() if val != 0.0}
    items = list(transcation_dictionary.keys())
    return (items)
```


```python
news_basket = news.to_dict(orient='records')
bills_basket = bill_information.to_dict(orient='records')
party_basket = party_platform.to_dict(orient='records')
```


```python
baskets = [news_basket,bills_basket,party_basket]
```


```python
transactions = []

for current_basket in baskets:
    for index in tqdm(range(0,len(current_basket)),desc='🛒🐛... | inching through the store'):
        transactions.append(transaction_creator(index,current_basket))
        
```


    🛒🐛... | inching through the store:   0%|          | 0/820 [00:00<?, ?it/s]



    🛒🐛... | inching through the store:   0%|          | 0/3261 [00:00<?, ?it/s]



    🛒🐛... | inching through the store:   0%|          | 0/2 [00:00<?, ?it/s]



```python
## And checking to make sure it worked...
transactions[0]
```




    ['ai',
     'backing',
     'billionaire',
     'day',
     'donald',
     'electric',
     'ha',
     'impacting',
     'industry',
     'news',
     'office',
     'policy',
     'president',
     'taking',
     'tech',
     'tiktok',
     'time',
     'trump',
     'vehicle']



### 2.2 Saving the Basket Data


```python
''' WRITING TO A CSV '''

with open('Basket Data.csv', 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(transactions)
```

## 3. Creating Document Term Matrix Data


```python
dtm_df = pd.concat([bill_information,news, party_platform])
```


```python
dtm_df.head()
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
      <th>aa</th>
      <th>aaa</th>
      <th>aarhu</th>
      <th>ab</th>
      <th>abandon</th>
      <th>abandonth</th>
      <th>abat</th>
      <th>abbrevi</th>
      <th>abercrombi</th>
      <th>abey</th>
      <th>...</th>
      <th>universal</th>
      <th>unjustly</th>
      <th>unlawful</th>
      <th>unnecessary</th>
      <th>upholding</th>
      <th>various</th>
      <th>vigorously</th>
      <th>violate</th>
      <th>weakened</th>
      <th>whistleblower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 17198 columns</p>
</div>




```python
dtm_df = dtm_df.fillna(0)
```


```python
dtm_df.reset_index(inplace=True,drop=True)
```


```python
dtm_df.head(1)
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
      <th>aa</th>
      <th>aaa</th>
      <th>aarhu</th>
      <th>ab</th>
      <th>abandon</th>
      <th>abandonth</th>
      <th>abat</th>
      <th>abbrevi</th>
      <th>abercrombi</th>
      <th>abey</th>
      <th>...</th>
      <th>universal</th>
      <th>unjustly</th>
      <th>unlawful</th>
      <th>unnecessary</th>
      <th>upholding</th>
      <th>various</th>
      <th>vigorously</th>
      <th>violate</th>
      <th>weakened</th>
      <th>whistleblower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 17198 columns</p>
</div>




```python
dtm_df.to_csv("Document Term Matrix.csv")
```
