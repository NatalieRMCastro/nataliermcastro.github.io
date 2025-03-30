---
layout: post
title: "Political Stances: Latent Dirichlet Allocation Code"
categories: projects
published: true
in_feed: false
---

# 5. Latent Dirichlet Allocation

## 1. Environment Creation:

### 1.1 Library Import


```python
''' DATA MANAGEMENT '''
import pandas as pd
import regex as re
import numpy as np
import os
from gensim import corpora
from gensim.corpora import Dictionary

''' TEXT PROCESSING '''
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


''' DATA VISUALIZATION '''
import seaborn as sb
from wordcloud import WordCloud
import matplotlib.pyplot as plt

''' LDA VIS'''
import pyLDAvis

''' SANITY '''
from tqdm import tqdm
```

    C:\Users\natal\miniconda3\lib\site-packages\pandas\core\arrays\masked.py:60: UserWarning: Pandas requires version '1.3.6' or newer of 'bottleneck' (version '1.3.5' currently installed).
      from pandas.core import (
    

### 1.2 Data Import


```python
news_data = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\News Articles Lemmed- Count Vectorizer.csv")
```


```python
news_data
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
      <th>publisher</th>
      <th>aapi</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abc</th>
      <th>ability</th>
      <th>able</th>
      <th>abolish</th>
      <th>...</th>
      <th>yes</th>
      <th>york</th>
      <th>young</th>
      <th>youth</th>
      <th>zealot</th>
      <th>zeldin</th>
      <th>zero</th>
      <th>zers</th>
      <th>zone</th>
      <th>zuckerberg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
    <tr>
      <th>1</th>
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
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Republican</td>
      <td>BBC News</td>
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
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Republican</td>
      <td>BBC News</td>
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
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Republican</td>
      <td>BBC News</td>
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
      <th>815</th>
      <td>816</td>
      <td>Democrat</td>
      <td>PBS</td>
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
    <tr>
      <th>816</th>
      <td>817</td>
      <td>Democrat</td>
      <td>PBS</td>
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
    <tr>
      <th>817</th>
      <td>818</td>
      <td>Democrat</td>
      <td>The Times of India</td>
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
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>818</th>
      <td>819</td>
      <td>Democrat</td>
      <td>The Times of India</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>819</th>
      <td>699</td>
      <td>NaN</td>
      <td>NaN</td>
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
<p>820 rows × 2361 columns</p>
</div>




```python
labels_news_party = news_data['Party'].to_list()
labels_news_publisher = news_data['publisher'].to_list()
```


```python
news_data.drop(columns=['Unnamed: 0','Party','publisher'],inplace=True)
```


```python
news_data.fillna(0,inplace=True)
```


```python
news_data.head()
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
      <th>aapi</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abc</th>
      <th>ability</th>
      <th>able</th>
      <th>abolish</th>
      <th>abortion</th>
      <th>absolutely</th>
      <th>abuse</th>
      <th>...</th>
      <th>yes</th>
      <th>york</th>
      <th>young</th>
      <th>youth</th>
      <th>zealot</th>
      <th>zeldin</th>
      <th>zero</th>
      <th>zers</th>
      <th>zone</th>
      <th>zuckerberg</th>
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
<p>5 rows × 2358 columns</p>
</div>



#### Climate Bill Data


```python
bills_data = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\Bills Lemmed- Count Vectorizer.csv")
```


```python
labels_bills_billtype = bills_data['Bill Type']
labels_bills_sponser_affiliation = bills_data['Sponser Affiliation']
labels_bills_sponser_state = bills_data['Sponser State']
labels_bills_committees = bills_data['Committees']
```


```python
bills_data.drop(columns=['Unnamed: 0','Bill Type','Sponser Affiliation','Sponser State','Committees'],inplace=True)
```


```python
bills_data.fillna(0,inplace=True)
```


```python
bills_data.head()
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
      <th>zoe</th>
      <th>zone</th>
      <th>zonea</th>
      <th>zonesnotwithstand</th>
      <th>zoneth</th>
      <th>zoo</th>
      <th>zoolog</th>
      <th>zoonot</th>
      <th>zooplankton</th>
      <th>zquez</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>...</td>
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
      <td>...</td>
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
      <th>2</th>
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
      <td>...</td>
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
      <th>3</th>
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
      <td>...</td>
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
      <th>4</th>
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
      <td>...</td>
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
<p>5 rows × 15489 columns</p>
</div>



#### Party Platform Data


```python
party_data = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\Party Platform Lemmed- Count Vectorizer.csv")
```


```python
labels_party_party = party_data['Party']
```


```python
party_data.drop(columns=["Unnamed: 0","Party"],inplace=True)
```


```python
party_data.fillna(0,inplace=True)
```


```python
party_data.head()
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
      <th>ability</th>
      <th>able</th>
      <th>abortion</th>
      <th>access</th>
      <th>accessible</th>
      <th>according</th>
      <th>accountability</th>
      <th>accountable</th>
      <th>achieved</th>
      <th>act</th>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
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
      <td>7</td>
      <td>13</td>
      <td>13</td>
      <td>72</td>
      <td>15</td>
      <td>1</td>
      <td>6</td>
      <td>14</td>
      <td>1</td>
      <td>88</td>
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
<p>2 rows × 892 columns</p>
</div>



### 1.3 Re- Count Vectorizing the Data


```python

## Recreate CountVectorizer with the original vocabulary
vectorizer_news = CountVectorizer(vocabulary=news_data.columns)
vectorizer_bills = CountVectorizer(vocabulary=bills_data.columns)
vectorizer_party = CountVectorizer(vocabulary=party_data.columns)

```

### 2. LDA Topic Modeling


```python
''' TESTING TOPIC MODELS '''

## First, creating a storage container to test numbers of topics:

topic_ns = []
for num in range (2,20,2):
    topic_ns.append(num)
    
## Creating a storage container for the topics in the list:
def save_topics(model, vectorizer, top_n=10):
    topic_contents = {}
    
    ## Iterating through the topics in the model components
    for idx, topic in enumerate(model.components_):
        
        ## Extracting the top words:
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_n - 1:-1]]
        
        ## Storing them: 
        topic_contents[idx] = top_words
        
    return (topic_contents)

## Creating something to visualize the topics with:
def topic_visualizer(lda_model, dataset, topic_number,plot_title, fontsize=10):
    
    word_topic = np.array(lda_model.components_).transpose()
    
    vocab_array = np.asarray(dataset.columns.to_list())
    
    fontsize_base = fontsize
    plt.rcParams["font.family"] = "Times New Roman"  
    
    fig, axes = plt.subplots(1, topic_number, figsize=(3 * topic_number, 6),dpi=1000)  # Adjust figure size
    plt.suptitle(plot_title, fontsize=16, fontname="Times New Roman", fontweight="bold")  # Main title

    
    ## Iterating and plotting the topics
    for t in range(topic_number):
        ax = plt.subplot(1, topic_number, t + 1)  # Create subplot
        ax.set_ylim(0, 15 + 0.5)  # Stretch the y-axis to accommodate the words
        ax.set_xticks([])  # Remove x-axis markings ('ticks')
        ax.set_yticks([])  # Remove y-axis markings ('ticks')
        ax.set_title(f'Topic #{t}', fontname="Times New Roman")  # Set title font
        
        # Change border (spine) colors to blue
        for spine in ax.spines.values():
            spine.set_edgecolor("blue")  # Set border color
            spine.set_linewidth(0.5)  # Make the border thicker

        top_words_idx = np.argsort(word_topic[:, t])[::-1]  # Descending order
        top_words_idx = top_words_idx[:15]
        top_words = vocab_array[top_words_idx]
        top_words_shares = word_topic[top_words_idx, t]

        for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
            ax.text(0.3, 15 - i - 0.5, word, fontsize=fontsize_base, fontname="Times New Roman")


    plt.tight_layout()
    plt.show()

## And now creating a recursive tester
def lda_tester(topic_number,dataset,vectorizer,top_n,dataset_name):
    ## Instantiating a model:
    
    lda_model = LatentDirichletAllocation(n_components=topic_number,max_iter=50, learning_method='online')
    
    ## Fitting the model:
    fit_model = lda_model.fit_transform(dataset)
    
    ## Storing the contents of the topic model:
    topic_contents = save_topics(lda_model, vectorizer,top_n)

    plot_title = f"LDA for {topic_number} Clusters - {dataset_name}"
    #topic_visualizer(lda_model, dataset,topic_number,plot_title,fontsize=10)
    
    return (lda_model,topic_contents)

## And finally wrapping it into a loop
def lda_modeler(start_topics, end_topics,dataset,vectorizer,top_n,dataset_name):
    
    dataset_topics = {}
    
    for topic_set in tqdm(range (start_topics,end_topics,2),desc='🐜🐛... inching through data',leave=False,):
        lda_model, lda_topics = lda_tester(topic_set,dataset,vectorizer,top_n,dataset_name)
        
        dataset_topics[topic_set] = {"LDA MODEL":lda_model,"LDA TOPICS": lda_topics}
        
    return (dataset_topics)
```


```python
''' TESTING NEWS DATA'''
news_model  = lda_modeler(6,20,news_data, vectorizer_news,15,'Climate News')
```

                                                                                                                           


```python
''' TESTING CLIMATE DATA '''
bills_model  = lda_modeler(6,20,bills_data, vectorizer_bills,15,'Proposed Climate Bills')
```

                                                                                                                           


```python
''' TESTING PARTY DATA '''
party_model  = lda_modeler(6,20,party_data, vectorizer_party,15,'2024 Party Platforms')
```

                                                                                                                           

### examining the outputs


```python
news_model
```




    {6: {'LDA MODEL': LatentDirichletAllocation(learning_method='online', max_iter=50, n_components=6),
      'LDA TOPICS': {0: ['trump',
        'biden',
        'new',
        'energy',
        'president',
        'republican',
        'wa',
        'gas',
        'house',
        'people',
        'oil',
        'vote',
        'fuel',
        'drilling',
        'time'],
       1: ['wildfire',
        'california',
        'los',
        'angeles',
        'newsom',
        'la',
        'city',
        'gavin',
        'bass',
        'mayor',
        'state',
        'democrat',
        'home',
        'post',
        'ha'],
... (truncated) ...}



```python
bills_model
```

    {6: {'LDA MODEL': LatentDirichletAllocation(learning_method='online', max_iter=50, n_components=6),
      'LDA TOPICS': {0: ['substanc',
        'chemic',
        'wast',
        'cover',
        'facil',
        'recycl',
        'manufactur',
        'materi',
        'site',
        'plastic',
        'dispos',
        'notic',
        'mixtur',
        'violat',
        'discharg'],
... (truncated) ...}

```python
party_model
```
    {6: {'LDA MODEL': LatentDirichletAllocation(learning_method='online', max_iter=50, n_components=6),
      'LDA TOPICS': {0: ['republican',
        'american',
        'policy',
        'border',
        'restore',
        'america',
        'great',
        'country',
        'protect',
        'education',
        'support',
        'right',
        'common',
        'people',
        'government'],
... (truncated) ...}
 



### visualizing the best topic models


```python
bills_model_best = bills_model[18]['LDA MODEL']
```


```python
topic_visualizer(bills_model_best,bills_data,18,"LDA Topic Modeling with 18 Clusters - Introduced Climate Bills",fontsize=10)
```


    
![png](output_35_0.png)
    



```python
import numpy as np
import matplotlib.pyplot as plt

red_words = {"republican", "trump",'donald','musk','elon'}
blue_words = {"democrat", "biden",'joe','newsom'}

def topic_visualizer(lda_model, vectorizer, topic_number, plot_title, fontsize=10):
    word_topic = np.array(lda_model.components_).transpose()
    vocab_array = np.array(vectorizer.get_feature_names_out())  # Get vocab directly

    fontsize_base = fontsize
    plt.rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(1, topic_number, figsize=(3 * topic_number, 6), dpi=1000)
    plt.suptitle(plot_title, fontsize=16, fontname="Times New Roman", fontweight="bold")

    for t in range(topic_number):
        ax = plt.subplot(1, topic_number, t + 1)
        ax.set_ylim(0, 15 + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Topic #{t}', fontname="Times New Roman")

        for spine in ax.spines.values():
            spine.set_edgecolor("blue")
            spine.set_linewidth(0.5)

        # Fix: Ensure indices are within vocab bounds
        valid_indices = np.argsort(word_topic[:, t])[::-1]  # Sort by importance
        valid_indices = [idx for idx in valid_indices if idx < len(vocab_array)][:15]  # Keep valid indices
        top_words = vocab_array[valid_indices]

        for i, word in enumerate(top_words):
            color = "black"
            if word.lower() in red_words:
                color = "red"
            elif word.lower() in blue_words:
                color = "blue"

            ax.text(0.3, 15 - i - 0.5, word, fontsize=fontsize_base, fontname="Times New Roman", color=color)

    plt.tight_layout()
    plt.show()

```


```python
party_model_best = party_model[8]['LDA MODEL']
```


```python
topic_visualizer(party_model_best,vectorizer_party,8,"LDA Topic Modeling with 8 Clusters - 2024 Partisan Platforms",fontsize=10)
```


    
![png](output_38_0.png)
    



```python
news_model_best = news_model[8]['LDA MODEL']
```


```python
topic_visualizer(news_model_best, vectorizer_news, 8,"LDA Topic Modeling with 8 Clusters - Climate Related News", fontsize=10)
```


    
![png](output_40_0.png)
    


#### creating a special visualizer for the longer topic models


```python
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def topic_visualizer_scrollable(lda_model, dataset, topic_number, plot_title, fontsize=10, output_file="topic_visualizer.html"):
    
    word_topic = np.array(lda_model.components_).transpose()
    vocab_array = np.asarray(dataset.columns.to_list())

    fontsize_base = fontsize
    plt.rcParams["font.family"] = "Times New Roman"  

    fig, axes = plt.subplots(1, topic_number, figsize=(2 * topic_number, 6), dpi=1000)
    
    for t in range(topic_number):
        ax = plt.subplot(1, topic_number, t + 1)
        ax.set_ylim(0, 15 + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Topic #{t}', fontname="Times New Roman")

        for spine in ax.spines.values():
            spine.set_edgecolor("blue")
            spine.set_linewidth(0.5)

        top_words_idx = np.argsort(word_topic[:, t])[::-1][:min(15, len(vocab_array))]
        top_words = vocab_array[top_words_idx]

        for i, word in enumerate(top_words):
            ax.text(0.3, 15 - i - 0.5, word, fontsize=fontsize_base, fontname="Times New Roman")

    plt.tight_layout()

    # Save figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Convert image to Base64
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_html = f'<img src="data:image/png;base64,{encoded}" style="max-height:600px;">'

    # HTML template with scrollable container
    html_template = f"""
    <html>
    <head>
        <style>
            .scroll-container {{
                width: 100%;
                overflow-x: auto;  /* Enable horizontal scrolling */
                white-space: nowrap;
                border: 1px solid #ddd;
                padding: 10px;
            }}
        </style>
    </head>
    <body>
        <h2 style="font-family: 'Times New Roman'; text-align: center;">{plot_title}</h2>
        <div class="scroll-container">
            {img_html}
        </div>
    </body>
    </html>
    """

    # Save HTML file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"HTML file saved: {output_file}")


```


```python
topic_visualizer_scrollable(bills_model_best,bills_data,18,'LDA Topic Modeling with 18 Clusters - Introduced Climate Policy',10,'lda_climate_bills.html')
```

    HTML file saved: lda_climate_bills.html
    

### 3. Creating Intertopic Distance Maps


```python
def lda_visualizer(data,model,vectorizer,filename):
    
    ''' PREPARARING THE DATA TO FIT '''
    doc_lengths_sparse = data.sum(axis=1)
    
    ## converting to an array
    doc_lengths = np.asarray(doc_lengths_sparse).flatten()
    
    ## normalizing the distributions from the model
    topic_term_distributions = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    
    ## extracting the documnet - topic distributions
    document_topic_distributions = model.transform(data)
    
    ## extracting the vocabulary from the vectorizer
    vocabulary = vectorizer.get_feature_names_out()
    
    ## extracting the term frequencies
    term_frequencies = np.asarray(data.sum(axis=0)).flatten()
    
    ''' GENERATING THE VISUALIZATION'''
    visualization = pyLDAvis.prepare(topic_term_dists=topic_term_distributions,
                                    doc_topic_dists=document_topic_distributions,
                                    doc_lengths=doc_lengths,
                                    vocab=vocabulary,
                                    term_frequency=term_frequencies)
    
    ''' SAVING THE VISUALIZATION'''
    
    saving_filename = filename+".html"
    pyLDAvis.save_html(visualization, saving_filename)

    
```


```python
lda_visualizer(bills_data,bills_model_best,vectorizer_bills,'LDA Interactive Topics - Bills')
```


```python
lda_visualizer(party_data,party_model_best,vectorizer_party,'LDA Interactive Topics - Party')
```


```python
lda_visualizer(news_data,news_model_best,vectorizer_news,'LDA Interactive Topics - News')
```
