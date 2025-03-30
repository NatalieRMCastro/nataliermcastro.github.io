---
layout: post
title: "Political Stances - Data Cleaning"
categories: projects
published: true
in_feed: False
---
<section>
    <div class="row">
        <div class="col-6 col-12-small">
            <ul class="actions" style="display: flex; gap: 10px; list-style: none; padding: 0;">
                <li><a href="https://nataliermcastro.github.io/projects/2025/01/14/political-stances.html" class="button fit small">Navigate to Project Page</a></li>
                <li><a href="https://nataliermcastro.github.io/projects/2025/01/14/political-stances-data.html" class="button fit small">Navigate to Data Page</a></li>
            </ul>
        </div>
    </div> 
</section> 

# Data Cleaning

The purpose of this notebook is to clean the data generated in notebook "0. Data Collection". I will clean each source and generate multiple dataframes to later model language. 

## 1. Environment Creation

### 1.1 Library Import


```python
''' OS MANAGEMENT '''
import os

''' DATA MANAGEMENT '''
import pandas as pd
import regex as re

''' DATA STRUCTURING '''
import ast 

''' TEXT PROCESSING '''
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

from nltk.stem import WordNetLemmatizer 
lem = WordNetLemmatizer() 



''' DATA VISUALIZATION '''
import seaborn as sb
from wordcloud import WordCloud
import matplotlib.pyplot as plt
```

### 1.2 Data Import


```python
''' NEWS ARTICLE IMPORT'''
republican_news = pd.read_csv("NEWSAPI - republican climate articles raw.csv")
democrat_news = pd.read_csv("NEWSAPI - democrat climate articles raw.csv")

republican_news.drop(columns='Unnamed: 0',inplace=True)
democrat_news.drop(columns='Unnamed: 0',inplace=True)

''' EPA BILL IMPORT '''
bills = pd.read_csv(r"Bill Information Full FINAL.csv")
bills.drop(columns='Unnamed: 0',inplace=True)

''' PARTY PLATFORM IMPORT '''
with open("democrat_party_platform.txt", "r") as file:
    democrat_pdf = file.read()
    
    
with open("republican_party_platform.txt",'r') as f:
    republican_pdf = f.read()
```

### 1.3 Function Definition


```python
''' 🫧🧼 | now lets create a cleaning function '''

def text_cleaner(text):
    try:
        scrubbed_text1 = re.sub('\d',' ',text)
        scrubbed_text2 = re.findall('\w+',scrubbed_text1)
        scrubbed_text3 = ' '.join(scrubbed_text2)
        scrubbed_text = scrubbed_text3.lower()
        clean_text = scrubbed_text.strip(" ")
        
        return(clean_text)
    
    except:
        return(text)
```


```python
''' Code Source: Gates Bolton Analytics'''

def stemmer(string):
    try:
        words = re.sub(r"[^A-Za-z\-]", " ", string).lower().split()
        words = [ps.stem(word) for word in words]
        return words
    except:
        return ("")

def lemmer(string):
    try:
        words = re.sub(r"[^A-Za-z\-]", " ", string).lower().split()
        words = [lem.lemmatize(word) for word in words]
        return words
    except:
        return ("")
```


```python
def count_vectorizer_creation(max_features,content,labels,label_colname):
    ''' COUNT VECTORIZER INSTANTIATION'''

    ## Instantiating the model, the filename parameter will take the list of file names
    ## filter for English stopwords, and take a max feature of 50
    count_vec = CountVectorizer(input='content',  stop_words='english', max_features=max_features, min_df=2,max_df=700)

    ## Fitting the model to the corpus
    model = count_vec.fit_transform(content)
    
    ''' EXTRACTING FEATURES'''

    ## Using get feature names out to name the columns
    columns = count_vec.get_feature_names_out()

    ## Creating the dataframe 
    vect_dataframe = pd.DataFrame(model.toarray(),columns=columns)
    
    ''' ADDING LABELS '''
    if type(labels) == list:
        vect_dataframe.insert(0,label_colname,labels)
        
    else:
        vect_dataframe = pd.concat([labels,vect_dataframe],axis=1)
        
    return (vect_dataframe)
```


```python
def tfidf_vectorizer_creation(max_features,content,labels,label_colname):
    ''' INSTANTIATING THE MODEL '''

    ## Input is set to content here because I will be passing in a list of the descriptions from the CSV File
    tfidf_vec = TfidfVectorizer(input='content',stop_words='english',max_features=max_features,min_df=2,max_df=700)

    ''' TRAINING THE MODEL '''
    tfidf_model = tfidf_vec.fit_transform(content)

    ''' STRUCTURING THE DATAFRAME '''
    tfidf_columns = tfidf_vec.get_feature_names_out()
    tfidf_df =pd.DataFrame(tfidf_model.toarray(),columns=tfidf_columns)
        
    ''' ADDING LABELS '''
    if type(labels) == list:
        tfidf_df.insert(0,label_colname,labels)
        
    else:
        tfidf_df = pd.concat([labels,tfidf_df],axis=1)
        
    return (tfidf_df)
```

## 2. News Article Cleaning


```python
republican_news.head(1)
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
      <th>source</th>
      <th>author</th>
      <th>title</th>
      <th>description</th>
      <th>url</th>
      <th>urlToImage</th>
      <th>publishedAt</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'id': 'the-verge', 'name': 'The Verge'}</td>
      <td>Nilay Patel</td>
      <td>Trump’s first 100 days: all the news impacting...</td>
      <td>President Donald Trump is taking on TikTok, el...</td>
      <td>https://www.theverge.com/24348851/donald-trump...</td>
      <td>https://cdn.vox-cdn.com/thumbor/Nwo4_i4giY8lRM...</td>
      <td>2025-01-22T14:30:00Z</td>
      <td>Filed under:\r\nByLauren Feiner, a senior poli...</td>
    </tr>
  </tbody>
</table>
</div>




```python
republican_label = []
for i in range(0,len(republican_news)):
    republican_label.append('Republican')
    
democrat_label = []
for i in range(0,len(democrat_news)):
    democrat_label.append('Democrat')
    
republican_news['Party'] = republican_label
democrat_news['Party'] = democrat_label
```


```python
news_data = pd.concat([republican_news,democrat_news])
news_data.reset_index(inplace=True)
news_data = news_data[['Party','source','title','description',]]
```


```python
## Now fixing the source
def source_fixer(source):
    source_dict = ast.literal_eval(source)
    source_name = source_dict['name']
    return (source_name)
```


```python
publisher = []
for article in range(0,len(news_data)):
    source_raw = news_data.at[article,'source']
    source_name = source_fixer(source_raw)
    publisher.append(source_name)
    
## Appending the column to track the source
news_data['publisher'] = publisher
news_data.drop(columns='source',inplace=True)
```


```python
''' CLEANING THE TITLES '''
clean_titles = []
for article in range(0,len(news_data)):
    title_raw = news_data.at[article,'title']
    title = text_cleaner(title_raw)
    clean_titles.append(title)
    
## Appending the column to track the source
news_data['clean title'] = clean_titles

''' CLEANING THE DESCRIPTIONS '''
clean_descriptions = []
for article in range(0,len(news_data)):
    description_raw = news_data.at[article,'description']
    description = text_cleaner(description_raw)
    clean_descriptions.append(description)
    
## Appending the column to track the source
news_data['clean description'] = clean_descriptions

```


```python
''' APPENDING THE COLUMNS FOR LARGER CONTEXT '''
news_data['title + description'] = news_data['clean title'] +" " + news_data['clean description']
```


```python
news_data.head(3)
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
      <th>title</th>
      <th>description</th>
      <th>publisher</th>
      <th>clean title</th>
      <th>clean description</th>
      <th>title + description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republican</td>
      <td>Trump’s first 100 days: all the news impacting...</td>
      <td>President Donald Trump is taking on TikTok, el...</td>
      <td>The Verge</td>
      <td>trump s first days all the news impacting the ...</td>
      <td>president donald trump is taking on tiktok ele...</td>
      <td>trump s first days all the news impacting the ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Republican</td>
      <td>The Quiet Death of Biden’s Climate Corps—and W...</td>
      <td>Biden's green jobs program was never what it s...</td>
      <td>Gizmodo.com</td>
      <td>the quiet death of biden s climate corps and w...</td>
      <td>biden s green jobs program was never what it s...</td>
      <td>the quiet death of biden s climate corps and w...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Republican</td>
      <td>The peanut farmer who rose to US president and...</td>
      <td>The US president struggled in the White House ...</td>
      <td>BBC News</td>
      <td>the peanut farmer who rose to us president and...</td>
      <td>the us president struggled in the white house ...</td>
      <td>the peanut farmer who rose to us president and...</td>
    </tr>
  </tbody>
</table>
</div>




```python
news_data.dropna(inplace=True)
news_data.reset_index(inplace=True)
```


```python
news_data.to_csv("News Data Cleaned.csv")
```


```python
print (news_data.at[169,'title'])
print (news_data.at[169,'description'])

print ("\nCleaned and Combined Title and Description----------------")
print (news_data.at[169,'title + description'])
```

    The Trump-Newsom Fight Over an Alleged 'Water Restoration Declaration,' Explained
    Trump claimed Newsom's refusal to sign the document led to a water shortage during the Los Angeles fires. But there's more to the story.
    
    Cleaned and Combined Title and Description----------------
    the trump newsom fight over an alleged water restoration declaration explained trump claimed newsom s refusal to sign the document led to a water shortage during the los angeles fires but there s more to the story
    


```python
publisher_counts = Counter(news_data['publisher'].to_list())
```


```python
count = pd.DataFrame.from_dict(publisher_counts,orient='index',columns=['Count'])
```


```python
print (f"There are {len(count)} unique sources in the dataset.")
```

    There are 166 unique sources in the dataset.
    


```python
labels = news_data[['Party','publisher']]
```

#### 2.2.1 Stemming


```python
stemmed_texts = []
for article in range(0,len(news_data)):
    text = news_data.at[article,'title + description']
    stemmed = stemmer(text)
    st = ' '.join(stemmed)
    stemmed_texts.append(st)
```


```python
stemmed_texts[0]
```




    'trump s first day all the news impact the tech industri presid donald trump is take on tiktok electr vehicl polici and ai in hi first day in offic thi time around he ha the back of mani tech billionair'




```python
news_vec_stemmed = count_vectorizer_creation(100000,stemmed_texts,labels,['Party','publisher'])
```


```python
news_vec_stemmed
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
      <th>abc</th>
      <th>abil</th>
      <th>abl</th>
      <th>abolish</th>
      <th>abort</th>
      <th>abov</th>
      <th>...</th>
      <th>yekel</th>
      <th>york</th>
      <th>young</th>
      <th>youth</th>
      <th>zealot</th>
      <th>zeldin</th>
      <th>zer</th>
      <th>zero</th>
      <th>zone</th>
      <th>zuckerberg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republican</td>
      <td>The Verge</td>
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
      <td>Republican</td>
      <td>Gizmodo.com</td>
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
      <td>Republican</td>
      <td>BBC News</td>
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
      <td>Republican</td>
      <td>BBC News</td>
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
      <td>Republican</td>
      <td>BBC News</td>
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
      <th>816</th>
      <td>Democrat</td>
      <td>PBS</td>
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
      <th>817</th>
      <td>Democrat</td>
      <td>PBS</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <th>818</th>
      <td>Democrat</td>
      <td>The Times of India</td>
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
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Democrat</td>
      <td>The Times of India</td>
      <td>NaN</td>
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
      <th>699</th>
      <td>NaN</td>
      <td>NaN</td>
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
<p>820 rows × 2122 columns</p>
</div>




```python
news_vec_stemmed.to_csv("News Articles Stemmed- Count Vectorizer.csv")
```


```python
news_tfidf_stemmed = tfidf_vectorizer_creation(100000,stemmed_texts,labels,['Party','publisher'])
```


```python
news_tfidf_stemmed
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
      <th>abc</th>
      <th>abil</th>
      <th>abl</th>
      <th>abolish</th>
      <th>abort</th>
      <th>abov</th>
      <th>...</th>
      <th>yekel</th>
      <th>york</th>
      <th>young</th>
      <th>youth</th>
      <th>zealot</th>
      <th>zeldin</th>
      <th>zer</th>
      <th>zero</th>
      <th>zone</th>
      <th>zuckerberg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republican</td>
      <td>The Verge</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Republican</td>
      <td>Gizmodo.com</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Republican</td>
      <td>BBC News</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Republican</td>
      <td>BBC News</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Republican</td>
      <td>BBC News</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
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
      <th>816</th>
      <td>Democrat</td>
      <td>PBS</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>817</th>
      <td>Democrat</td>
      <td>PBS</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.244297</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>818</th>
      <td>Democrat</td>
      <td>The Times of India</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.20093</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Democrat</td>
      <td>The Times of India</td>
      <td>NaN</td>
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
      <th>699</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>820 rows × 2122 columns</p>
</div>




```python
news_tfidf_stemmed.to_csv("News Articles Stemmed- TFIDF.csv")
```

#### 2.2.2 Lemmatization


```python
lemmed_texts = []
for article in range(0,len(news_data)):
    text = news_data.at[article,'title + description']
    lemmed = lemmer(text)
    lt = ' '.join(lemmed)
    lemmed_texts.append(lt)
```


```python
news_vec_lemmed = count_vectorizer_creation(10000,lemmed_texts,labels,['Party','publisher'])
```


```python
news_vec_lemmed
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
      <th>abortion</th>
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
      <td>Republican</td>
      <td>The Verge</td>
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
      <td>Republican</td>
      <td>Gizmodo.com</td>
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
      <td>Republican</td>
      <td>BBC News</td>
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
      <td>Republican</td>
      <td>BBC News</td>
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
      <td>Republican</td>
      <td>BBC News</td>
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
      <th>816</th>
      <td>Democrat</td>
      <td>PBS</td>
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
      <th>817</th>
      <td>Democrat</td>
      <td>PBS</td>
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
      <th>818</th>
      <td>Democrat</td>
      <td>The Times of India</td>
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
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Democrat</td>
      <td>The Times of India</td>
      <td>NaN</td>
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
      <th>699</th>
      <td>NaN</td>
      <td>NaN</td>
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
<p>820 rows × 2360 columns</p>
</div>




```python
news_vec_lemmed.to_csv("News Articles Lemmed- Count Vectorizer.csv")
```


```python
news_tfidf_lemmed = tfidf_vectorizer_creation(10000,lemmed_texts,labels,['Party','publisher'])
```


```python
news_tfidf_lemmed
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
      <th>abortion</th>
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
      <td>Republican</td>
      <td>The Verge</td>
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
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Republican</td>
      <td>Gizmodo.com</td>
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
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Republican</td>
      <td>BBC News</td>
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
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Republican</td>
      <td>BBC News</td>
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
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Republican</td>
      <td>BBC News</td>
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
      <td>0.000000</td>
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
      <th>816</th>
      <td>Democrat</td>
      <td>PBS</td>
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
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>817</th>
      <td>Democrat</td>
      <td>PBS</td>
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
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>818</th>
      <td>Democrat</td>
      <td>The Times of India</td>
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
      <td>0.201006</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Democrat</td>
      <td>The Times of India</td>
      <td>NaN</td>
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
      <th>699</th>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>820 rows × 2360 columns</p>
</div>




```python
news_tfidf_lemmed.to_csv("News Articles Lemmed- TFIDF.csv")
```

#### 2.2.3 CountVectorizer


```python
texts = news_data['title + description'].to_list()
```


```python
news_vec = count_vectorizer_creation(10000,texts,labels,['Party','publisher'])
```


```python
news_vec.to_csv("News Articles - Count Vectorizer.csv")
```

#### 2.2.4 TF-IDF Vectorizer


```python
news_tfidf = tfidf_vectorizer_creation(10000,texts,labels,['Party','publisher'])
```


```python
news_tfidf.to_csv("News Articles - TF-IDF.csv")
```

## 3. EPA Bill Cleaning


```python
bills.head(1)
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
      <th>API URL</th>
      <th>Congress Number</th>
      <th>Bill Type</th>
      <th>Bill Number</th>
      <th>Legislation Number</th>
      <th>URL_y</th>
      <th>Congress</th>
      <th>Title</th>
      <th>Sponsor</th>
      <th>Date of Introduction</th>
      <th>...</th>
      <th>Number of Cosponsors</th>
      <th>Amends Bill</th>
      <th>Date Offered</th>
      <th>Date Submitted</th>
      <th>Date Proposed</th>
      <th>Amends Amendment</th>
      <th>Sponser Affiliation</th>
      <th>Sponser State</th>
      <th>Bill Title (XML)</th>
      <th>Bill Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.congress.gov/119/bills/hr375/BILLS...</td>
      <td>119</td>
      <td>hr</td>
      <td>375</td>
      <td>H.R. 375</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>Continued Rapid Ohia Death Response Act of 2025</td>
      <td>Tokuda, Jill N. [Rep.-D-HI-2] (Introduced 01/1...</td>
      <td>1/13/2025</td>
      <td>...</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>D</td>
      <td>HI</td>
      <td>&lt;dc:title&gt;119 HR 375 : Continued Rapid Ohia De...</td>
      <td>\n\n119 HR 375 : Continued Rapid Ohia Death Re...</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 23 columns</p>
</div>




```python
''' CLEANING THE BILLS '''
clean_bills= []
for bill in range(0,len(bills)):
    bill_text_raw = bills.at[bill,'Bill Text']
    clean_bill = text_cleaner(bill_text_raw)
    clean_bills.append(clean_bill)
    
## Appending the column to track the source
bills['Bill Text Clean'] = clean_bills
```


```python
texts = clean_bills
```


```python
labels = bills[['Bill Type','Sponser Affiliation','Sponser State','Committees']]
```

#### 3.2.1 Stemming


```python
stemmed_texts = []
for article in range(0,len(bills)):
    text = bills.at[article,'Bill Text Clean']
    stemmed = stemmer(text)
    st = ' '.join(stemmed)
    stemmed_texts.append(st)
```


```python
stemmed_texts[0]
```




    'hr continu rapid ohia death respons act of u s hous of repres text xml en pursuant to titl section of the unit state code thi file is not subject to copyright protect and is in the public domain iib th congress st sessionh r in the senat of the unit statesjanuari receiv read twice and refer to the committe on agricultur nutrit and forestryan actto requir the secretari of the interior to partner and collabor with the secretari of agricultur and the state of hawaii to address rapid ohia death and for other purpos short titlethi act may be cite as the continu rapid ohia death respons act of definitionsin thi act rapid ohia deathth term rapid ohia death mean the diseas caus by the fungal pathogen known as ceratocysti fimbriata that affect the tree of the speci metrosidero polymorpha stateth term state mean the state of hawaii collaborationth secretari of the interior shall partner and collabor with the secretari of agricultur and the state to address rapid ohia death sustain effort a transmissionth secretari of the interior act through the director of the unit state geolog survey and the chief of the forest servic act through the forest servic institut of pacif island forestri shall continu to conduct research on rapid ohia death vector and transmiss b ungul managementth secretari of the interior act through the director of the unit state fish and wildlif servic shall continu to partner with the secretari of agricultur the state and with local stakehold to manag ungul in rapid ohia death control area on feder state and privat land with the consent of privat landown c restor and researchth secretari of agricultur act through the chief of the forest servic shall continu to provid financi assist includ through agreement with the secretari of the interior a to prevent the spread of rapid ohia death and b to restor the nativ forest of the state and staff and necessari infrastructur fund to the institut of pacif island forestri to conduct research on rapid ohia death pass the hous of repres januari kevin f mccumber clerk'




```python
bill_vec_stemmed = count_vectorizer_creation(100000,stemmed_texts,labels,['Bill Type','Sponser Affiliation','Sponser State','Committees'])
```


```python
bill_vec_stemmed
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
      <th>abandonth</th>
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
      <td>hr</td>
      <td>D</td>
      <td>HI</td>
      <td>House - Natural Resources, Agriculture | Senat...</td>
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
      <td>hr</td>
      <td>R</td>
      <td>NY</td>
      <td>House - Agriculture</td>
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
      <td>hr</td>
      <td>R</td>
      <td>TX</td>
      <td>House - Energy and Commerce</td>
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
      <td>hr</td>
      <td>R</td>
      <td>NY</td>
      <td>House - Transportation and Infrastructure, Nat...</td>
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
      <td>hr</td>
      <td>R</td>
      <td>OH</td>
      <td>House - Transportation and Infrastructure</td>
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
      <th>3256</th>
      <td>hr</td>
      <td>D</td>
      <td>CA</td>
      <td>House - Transportation and Infrastructure</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
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
      <th>3257</th>
      <td>hr</td>
      <td>R</td>
      <td>CO</td>
      <td>House - Resources</td>
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
      <th>3258</th>
      <td>hr</td>
      <td>D</td>
      <td>MI</td>
      <td>House - Energy and Commerce</td>
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
      <th>3259</th>
      <td>s</td>
      <td>D</td>
      <td>NJ</td>
      <td>Senate - Environment and Public Works</td>
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
      <th>3260</th>
      <td>hr</td>
      <td>R</td>
      <td>TX</td>
      <td>House - Energy and Commerce</td>
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
<p>3261 rows × 15493 columns</p>
</div>




```python
bill_vec_stemmed.to_csv("Bills Stemmed- Count Vectorizer.csv")
```


```python
bill_tfidf_stemmed = tfidf_vectorizer_creation(100000,stemmed_texts,labels,['Bill Type','Sponser Affiliation','Sponser State','Committees'])
```


```python
bill_tfidf_stemmed
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
      <th>abandonth</th>
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
      <td>hr</td>
      <td>D</td>
      <td>HI</td>
      <td>House - Natural Resources, Agriculture | Senat...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>hr</td>
      <td>R</td>
      <td>NY</td>
      <td>House - Agriculture</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>hr</td>
      <td>R</td>
      <td>TX</td>
      <td>House - Energy and Commerce</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>hr</td>
      <td>R</td>
      <td>NY</td>
      <td>House - Transportation and Infrastructure, Nat...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>hr</td>
      <td>R</td>
      <td>OH</td>
      <td>House - Transportation and Infrastructure</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <th>3256</th>
      <td>hr</td>
      <td>D</td>
      <td>CA</td>
      <td>House - Transportation and Infrastructure</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.022374</td>
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
      <th>3257</th>
      <td>hr</td>
      <td>R</td>
      <td>CO</td>
      <td>House - Resources</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <th>3258</th>
      <td>hr</td>
      <td>D</td>
      <td>MI</td>
      <td>House - Energy and Commerce</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <th>3259</th>
      <td>s</td>
      <td>D</td>
      <td>NJ</td>
      <td>Senate - Environment and Public Works</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <th>3260</th>
      <td>hr</td>
      <td>R</td>
      <td>TX</td>
      <td>House - Energy and Commerce</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
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
<p>3261 rows × 15493 columns</p>
</div>




```python
bill_tfidf_stemmed.to_csv("Bills Stemmed- TFIDF.csv")
```

#### 3.2.2 Lemmatization


```python
lemmed_texts = []
for article in range(0,len(bills)):
    text = bills.at[article,'Bill Text Clean']
    lemmed = lemmer(text)
    lt = ' '.join(lemmed)
    lemmed_texts.append(lt)
```


```python
bill_vec_lemmed = count_vectorizer_creation(100000,stemmed_texts,labels,['Bill Type','Sponser Affiliation','Sponser State','Committees'])
```


```python
bill_vec_lemmed.to_csv("Bills Lemmed- Count Vectorizer.csv")
```


```python
bill_tfidf_lemmed = tfidf_vectorizer_creation(10000,lemmed_texts,labels,['Party','publisher'])
```


```python
bill_tfidf_lemmed.to_csv("Bills Lemmed- TFIDF.csv")
```

#### 3.2.3 CountVectorizer


```python
bill_vec = count_vectorizer_creation(100000,texts,labels,['Bill Type','Sponser Affiliation','Sponser State','Committees'])
```


```python
bill_vec.to_csv("Bills - Count Vectorizer.csv")
```

#### 3.2.4 TF-IDF Vectorizer


```python
bill_tfidf = tfidf_vectorizer_creation(100000,texts,labels,['Bill Type','Sponser Affiliation','Sponser State','Committees'])
```


```python
bill_tfidf.to_csv("Bills - TF-IDF.csv")
```

### 3.3 Visualizing Before and After Bill Cleaning


```python
raw_example = bills['Bill Text'].to_list()
raw_example_text = raw_example[200]
```


```python
raw_example_string = ' '.join(raw_example)
```


```python
clean_example_string = ' '.join(texts)
```


```python
raw_example_text
```




    '\n\n\n HR 2950 ENR: Coastal Habitat Conservation Act of 2023\nU.S. House of Representatives\n\ntext/xml\nEN\nPursuant to Title 17 Section 105 of the United States Code, this file is not subject to copyright protection and is in the public domain.\n\n\n\nIB\nOne Hundred Eighteenth Congress of the United States of AmericaAt the Second SessionBegun and held at the City of Washington on Wednesday, the third day of January, two thousand and twenty-four\nH. R. 2950\n\nAN ACT\nTo authorize the Secretary of the Interior, through the Coastal Program of the United States Fish and Wildlife Service, to work with willing partners and provide support to efforts to assess, protect, restore, and enhance important coastal landscapes that provide fish and wildlife habitat on which certain Federal trust species depend, and for other purposes.\n\n\n1.Short titleThis Act may be cited as the Coastal Habitat Conservation Act of 2023. 2.PurposeThe purpose of this Act is to legislatively authorize the Coastal Program of the Service in effect as of the date of the enactment of this Act to conduct collaborative landscape-level planning and on-the-ground coastal habitat assessment, coastal habitat protection, coastal habitat restoration, and coastal habitat enhancement projects in priority coastal landscapes to conserve and recover Federal trust species.\n3.DefinitionsIn this Act: (1)Coastal ecosystemThe term coastal ecosystem means a biological community of organisms interacting with each other and their habitats in a coastal landscape.\n(2)Coastal habitat assessmentThe term coastal habitat assessment means the process of evaluating the physical, chemical, and biological function of a coastal site to determine the value of the site to fish and wildlife. (3)Coastal habitat enhancementThe term coastal habitat enhancement means the manipulation of the physical, chemical, or biological characteristics of a coastal ecosystem to increase or decrease specific biological functions that make the ecosystem valuable to fish and wildlife.\n(4)Coastal habitat planningThe term coastal habitat planning means the process of developing a comprehensive plan that— (A)characterizes a coastal ecosystem;\n(B)sets protection, restoration, or enhancement goals and identifies the priorities of those goals; (C)describes conservation strategies and methodologies;\n(D)establishes a timetable for implementation of the plan; and (E)identifies roles of participants and stakeholders.\n(5)Coastal habitat protection\n(A)In generalThe term coastal habitat protection means a long-term action to safeguard habitat of value to fish and wildlife in a coastal ecosystem. (B)InclusionThe term coastal habitat protection includes activities to support establishment of a conservation easement or fee title acquisition by Federal and non-Federal partners.\n(6)Coastal habitat restorationThe term coastal habitat restoration means the manipulation of the physical, chemical, or biological characteristics of a coastal ecosystem with the goal of returning, to the maximum extent practicable, the full natural biological functions to lost or degraded native habitat. (7)Coastal landscapeThe term coastal landscape means a portion of a coastal ecosystem within or adjacent to a coastal State that contains various habitat types, including—\n(A)a fresh or saltwater wetland in a coastal watershed; (B)a coastal river, stream, or waterway;\n(C)a coastal bay or estuary; (D)a seagrass bed, reef, or other nearshore marine habitat;\n(E)a beach or dune system; (F)a mangrove forest; and\n(G)an associated coastal upland.  (8)Coastal StateThe term coastal State means—\n(A)a State in, or bordering on, the Atlantic, Pacific, or Arctic Ocean, the Gulf of Mexico, the Long Island Sound, or 1 or more of the Great Lakes; (B)the District of Columbia;\n(C)the Commonwealth of Puerto Rico; (D)Guam;\n(E)American Samoa; (F)the Commonwealth of the Northern Mariana Islands;\n(G)the Federated States of Micronesia; (H)the Republic of the Marshall Islands;\n(I)the Republic of Palau; and (J)the United States Virgin Islands.\n(9)Federal trust speciesThe term Federal trust species means migratory birds, threatened species or endangered species listed under the Endangered Species Act of 1973 (16 U.S.C. 1531 et seq.), interjurisdictional fish, and marine mammals for which the Secretary has management authority. (10)Financial assistanceThe term financial assistance means Federal funding provided to Federal, State, local, or Tribal governments, nongovernmental institutions, nonprofit organizations, and private individuals and entities through a grant or cooperative agreement.\n(11)SecretaryThe term Secretary means the Secretary of the Interior. (12)ServiceThe term Service means the United States Fish and Wildlife Service.\n(13)Technical assistanceThe term technical assistance means a collaboration, facilitation, or consulting action relating to a coastal habitat planning, coastal habitat assessment, coastal habitat protection, coastal habitat restoration, or coastal habitat enhancement project or initiative in which the Service contributes scientific knowledge, skills, and expertise to the project or initiative. 4.Coastal programThe Secretary shall carry out the Coastal Program within the Service to—\n(1)identify the leading threats to priority coastal landscapes and conservation actions to address those threats in partnership with Federal, State, local, and Tribal governments, nongovernmental institutions, nonprofit organizations, and private individuals and entities;  (2)provide technical assistance and financial assistance through partnerships with Federal, State, local, and Tribal governments, nongovernmental institutions, nonprofit organizations, and private individuals and entities to conduct voluntary coastal habitat planning, coastal habitat assessment, coastal habitat protection, coastal habitat restoration, and coastal habitat enhancement projects on public land or private land;\n(3)ensure the health and resilience of coastal ecosystems through adaptive management procedures based on the best available science; (4)build the capacity of Federal, State, local, and Tribal governments, nongovernmental institutions, nonprofit organizations, and private individuals and entities to carry out environmental conservation and stewardship measures;\n(5)assist in the development and implementation of monitoring protocols to ensure the success of coastal ecosystem restoration and coastal ecosystem enhancement measures; and (6)collaborate and share information with partners and the public relating to best management practices for the conservation, restoration, and enhancement of coastal ecosystems.\n5.Reports\n(a)In generalNot later than 1 year after the date of the enactment of this Act, and annually thereafter, the Secretary, acting through the Director of the Service, shall submit to the Committees on Appropriations and Natural Resources of the House of Representatives and the Committees on Appropriations and Environment and Public Works of the Senate, and make available to the public on the website of the Service, a report on the Coastal Program carried out under this Act. (b)RequirementsEach report submitted under subsection (a) shall assess on regional and nationwide bases—\n(1)Coastal Program work on coastal ecosystems; (2)progress made by the Coastal Program toward identifying the leading threats to priority coastal landscapes and conservation actions to address those threats; and\n(3)prospects for, and success of, protecting, restoring, and enhancing coastal ecosystems. (c)InclusionsEach report submitted under subsection (a) shall include—\n(1)quantitative information on coastal landscapes protected, restored, or enhanced; (2)funds appropriated to the Coastal Program that have been expended or leveraged;\n(3)a description of adaptive management practices implemented; and (4)a description of emerging challenges or data gaps that hinder the ability of the Coastal Program to achieve the purpose of this Act.\n6.Authorization of appropriationsThere is authorized to be appropriated to carry out this Act $16,957,000 for each of fiscal years 2024 through 2028.  Speaker of the House of Representatives.Vice President of the United States and President of the Senate. '




```python
title = bills.at[200,'Title']
```


```python
clean_example_text = texts[200]
```


```python
sb.set_style("white")
sb.set(font='Times New Roman', font_scale=1.2)

wc = WordCloud(width=600,height=300, background_color='white',colormap='gist_earth_r',max_words=1000)
wc.generate_from_text(raw_example_string)

## Plotting the cloud

plt.figure(figsize=(9,6),dpi=750)
plt.imshow(wc)
plt.axis('off')
plt.title("Raw Bill Text")
plt.savefig("Bill - Raw Text.png",dpi=1000);
```


    
![png](output_82_0.png)
    



```python
sb.set_style("white")
sb.set(font='Times New Roman', font_scale=1.2)

wc = WordCloud(width=600,height=300, background_color='white',colormap='gist_earth_r',max_words=1000)
wc.generate_from_text(clean_example_string)

## Plotting the cloud

plt.figure(figsize=(9,6),dpi=750)
plt.imshow(wc)
plt.axis('off')
plt.title("Clean Bill Text")
plt.savefig("Bill - Clean Text.png",dpi=1000);
```


    
![png](output_83_0.png)
    



```python

```

## 4. Party Platform Cleaning


```python
democrat_pdf[0:100]
```




    ' Democratic\nNational\nConvention \nLand\nAcknowledgement\nThe\nDemocratic\nNational\nCommittee\nwishes\nto\nac'




```python
republican_pdf[0:100]
```




    '4343RDRD REPUBLICAN NATIONAL CONVENTION REPUBLICAN NATIONAL CONVENTION\nPLATFORMTHE 2024 REPUBLICAN\nM'



### 4.1 Visualizing Raw Data


```python
sb.set_style("white")
sb.set(font='Times New Roman', font_scale=1.2)

wc = WordCloud(width=600,height=300, background_color='white',colormap='Blues',max_words=250)
wc.generate_from_text(democrat_pdf)

## Plotting the cloud

plt.figure(figsize=(9,6),dpi=750)
plt.imshow(wc)
plt.axis('off')
plt.title("DNC Party Platform - 2024\nRaw PDF Text")
plt.savefig("DNC Party Platform - Raw Text",dpi=1000);
```


    
![png](output_89_0.png)
    



```python
sb.set_style("white")
sb.set(font='Times New Roman', font_scale=1.2)

wc = WordCloud(width=600,height=300, background_color='white',colormap='OrRd',max_words=250)
wc.generate_from_text(republican_pdf)

## Plotting the cloud

plt.figure(figsize=(9,6),dpi=750)
plt.imshow(wc)
plt.axis('off')
plt.title("GOP Party Platform - 2024\nRaw PDF Text")
plt.savefig("GOP Party Platform - Raw Text",dpi=1000);
```


    
![png](output_90_0.png)
    


### 4.2 DF Creation


```python
''' CLEANING THE TEXT -- DEMOCRAT '''

## First, doing some basic cleaning for the text
democrat_text = re.sub('\W',' ',democrat_pdf)

## And Now Stripping Any White Space
democrat_string = democrat_text.strip(" ")

## And Now Splitting!
democrat_text_list = democrat_string.split(" ")
```


```python
democrat_text_list[0:10]
```




    ['Democratic',
     'National',
     'Convention',
     '',
     'Land',
     'Acknowledgement',
     'The',
     'Democratic',
     'National',
     'Committee']




```python
## It looks like there is a few empty characters in here, so now lets clean that up as well and lower the full text,
## and removing numbers
dem_text_clean = []

for word in democrat_text_list:
    if len(word) > 1:
        if len(re.findall("\d",word)) < 1:
            dem_text_clean.append(word.lower())
```


```python
dem_text_clean[0:10]
```




    ['democratic',
     'national',
     'convention',
     'land',
     'acknowledgement',
     'the',
     'democratic',
     'national',
     'committee',
     'wishes']




```python
''' CLEANING THE TEXT -- REPUBLICAN '''

## First, doing some basic cleaning for the text
republican_text = re.sub('\W',' ',republican_pdf)

## And Now Stripping Any White Space
republican_string = republican_text.strip(" ")

## And Now Splitting!
republican_text_list = republican_string.split(" ")
```


```python
republican_text_list[0:10]
```




    ['4343RDRD',
     'REPUBLICAN',
     'NATIONAL',
     'CONVENTION',
     'REPUBLICAN',
     'NATIONAL',
     'CONVENTION',
     'PLATFORMTHE',
     '2024',
     'REPUBLICAN']




```python
## It looks like there is a few empty characters in here, so now lets clean that up as well and lower the full text,
## and removing numbers
rep_text_clean = []

for word in republican_text_list:
    if len(word) > 1:
        if len(re.findall("\d",word)) < 1:
            rep_text_clean.append(word.lower())
```


```python
rep_text_clean[0:10]
```




    ['republican',
     'national',
     'convention',
     'republican',
     'national',
     'convention',
     'platformthe',
     'republican',
     'make',
     'america']




```python
''' CREATING A DATAFRAME '''
rep_text_final = ' '.join(rep_text_clean)
dem_text_final = ' '.join(dem_text_clean)

party_platforms = pd.DataFrame(columns=['Party','Text'])
```


```python
party_platforms['Party'] = ['Republican','Democrat']
party_platforms['Text'] = [rep_text_final,dem_text_final]
```


```python
party_platforms
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
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republican</td>
      <td>republican national convention republican nati...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Democrat</td>
      <td>democratic national convention land acknowledg...</td>
    </tr>
  </tbody>
</table>
</div>



#### 4.2.1 Stemming


```python
rep_stemmed = stemmer(rep_text_final)
dem_stemmed = stemmer(dem_text_final)

rep_stem_text = ' '.join(rep_stemmed)
dem_stem_text = ' '.join(rep_stemmed)
```


```python
platform_vec_stemmed = count_vectorizer_creation(10000,[rep_stem_text,dem_stem_text],['Republican','Democrat'],'Party')
```


```python
platform_vec_stemmed
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
      <th>abernathi</th>
      <th>abil</th>
      <th>abl</th>
      <th>abort</th>
      <th>absolut</th>
      <th>abund</th>
      <th>access</th>
      <th>accomplish</th>
      <th>accord</th>
      <th>...</th>
      <th>worst</th>
      <th>wrongdoer</th>
      <th>www</th>
      <th>wyom</th>
      <th>year</th>
      <th>yob</th>
      <th>york</th>
      <th>young</th>
      <th>zack</th>
      <th>zoraida</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republican</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Democrat</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1379 columns</p>
</div>




```python
platform_vec_stemmed.to_csv("Party Platform Stemmed- Count Vectorizer.csv")
```


```python
platform_tfidf_stemmed = tfidf_vectorizer_creation(10000,[rep_stem_text,dem_stem_text],['Republican','Democrat'],'Party')
```


```python
platform_tfidf_stemmed
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
      <th>abernathi</th>
      <th>abil</th>
      <th>abl</th>
      <th>abort</th>
      <th>absolut</th>
      <th>abund</th>
      <th>access</th>
      <th>accomplish</th>
      <th>accord</th>
      <th>...</th>
      <th>worst</th>
      <th>wrongdoer</th>
      <th>www</th>
      <th>wyom</th>
      <th>year</th>
      <th>yob</th>
      <th>york</th>
      <th>young</th>
      <th>zack</th>
      <th>zoraida</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republican</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.014261</td>
      <td>0.023769</td>
      <td>0.004754</td>
      <td>0.009508</td>
      <td>...</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.033277</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.033277</td>
      <td>0.004754</td>
      <td>0.004754</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Democrat</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.014261</td>
      <td>0.023769</td>
      <td>0.004754</td>
      <td>0.009508</td>
      <td>...</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.033277</td>
      <td>0.004754</td>
      <td>0.004754</td>
      <td>0.033277</td>
      <td>0.004754</td>
      <td>0.004754</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1379 columns</p>
</div>




```python
platform_tfidf_stemmed.to_csv("Party Platform Stemmed- TFIDF.csv")
```

#### 4.2.2 Lemmatization


```python
rep_lemmed = lemmer(rep_text_final)
dem_lemmed = lemmer(dem_text_final)

rep_lem_text = ' '.join(rep_lemmed)
dem_lem_text = ' '.join(dem_lemmed)
```


```python
platform_vec_lemmed = count_vectorizer_creation(10000,[rep_lem_text,dem_lem_text],['Republican','Democrat'],'Party')
```


```python
platform_vec_lemmed
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
      <th>ability</th>
      <th>able</th>
      <th>abortion</th>
      <th>access</th>
      <th>accessible</th>
      <th>according</th>
      <th>accountability</th>
      <th>accountable</th>
      <th>achieved</th>
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
      <td>Republican</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
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
      <td>Democrat</td>
      <td>7</td>
      <td>13</td>
      <td>13</td>
      <td>72</td>
      <td>15</td>
      <td>1</td>
      <td>6</td>
      <td>14</td>
      <td>1</td>
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
<p>2 rows × 893 columns</p>
</div>




```python
platform_vec_lemmed.to_csv("Party Platform Lemmed- Count Vectorizer.csv")
```


```python
platform_tfidf_lemmed = tfidf_vectorizer_creation(10000,[rep_lem_text,dem_lem_text],['Republican','Democrat'],'Party')
```


```python
platform_tfidf_lemmed
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
      <th>ability</th>
      <th>able</th>
      <th>abortion</th>
      <th>access</th>
      <th>accessible</th>
      <th>according</th>
      <th>accountability</th>
      <th>accountable</th>
      <th>achieved</th>
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
      <td>Republican</td>
      <td>0.005157</td>
      <td>0.005157</td>
      <td>0.005157</td>
      <td>0.020627</td>
      <td>0.005157</td>
      <td>0.005157</td>
      <td>0.005157</td>
      <td>0.020627</td>
      <td>0.005157</td>
      <td>...</td>
      <td>0.005157</td>
      <td>0.005157</td>
      <td>0.020627</td>
      <td>0.077351</td>
      <td>0.010314</td>
      <td>0.077351</td>
      <td>0.010314</td>
      <td>0.005157</td>
      <td>0.036097</td>
      <td>0.036097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Democrat</td>
      <td>0.006634</td>
      <td>0.012320</td>
      <td>0.012320</td>
      <td>0.068233</td>
      <td>0.014215</td>
      <td>0.000948</td>
      <td>0.005686</td>
      <td>0.013268</td>
      <td>0.000948</td>
      <td>...</td>
      <td>0.009477</td>
      <td>0.000948</td>
      <td>0.095716</td>
      <td>0.074867</td>
      <td>0.076762</td>
      <td>0.065390</td>
      <td>0.003791</td>
      <td>0.003791</td>
      <td>0.141205</td>
      <td>0.009477</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 893 columns</p>
</div>




```python
platform_tfidf_lemmed.to_csv("Party Platform Lemmed- TFIDF.csv")
```

#### 4.2.3 CountVectorizer


```python
platform_vec = count_vectorizer_creation(10000,[rep_text_final,dem_text_final],['Republican','Democrat'],'Party')
```


```python
platform_vec
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
      <th>ability</th>
      <th>able</th>
      <th>abortion</th>
      <th>access</th>
      <th>accessible</th>
      <th>according</th>
      <th>accountability</th>
      <th>accountable</th>
      <th>achieved</th>
      <th>...</th>
      <th>work</th>
      <th>worker</th>
      <th>workers</th>
      <th>working</th>
      <th>world</th>
      <th>worship</th>
      <th>worst</th>
      <th>year</th>
      <th>years</th>
      <th>young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republican</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>14</td>
      <td>2</td>
      <td>15</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Democrat</td>
      <td>7</td>
      <td>13</td>
      <td>13</td>
      <td>72</td>
      <td>15</td>
      <td>1</td>
      <td>6</td>
      <td>14</td>
      <td>1</td>
      <td>...</td>
      <td>100</td>
      <td>4</td>
      <td>75</td>
      <td>81</td>
      <td>69</td>
      <td>4</td>
      <td>4</td>
      <td>80</td>
      <td>69</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 946 columns</p>
</div>




```python
platform_vec.to_csv("Party Platform - Count Vectorizer.csv")
```

#### 4.2.4 TF-IDF Vectorizer


```python
platform_tfidf = tfidf_vectorizer_creation(10000,[rep_text_final,dem_text_final],['Republican','Democrat'],'Party')
```


```python
platform_tfidf
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
      <th>ability</th>
      <th>able</th>
      <th>abortion</th>
      <th>access</th>
      <th>accessible</th>
      <th>according</th>
      <th>accountability</th>
      <th>accountable</th>
      <th>achieved</th>
      <th>...</th>
      <th>work</th>
      <th>worker</th>
      <th>workers</th>
      <th>working</th>
      <th>world</th>
      <th>worship</th>
      <th>worst</th>
      <th>year</th>
      <th>years</th>
      <th>young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republican</td>
      <td>0.005705</td>
      <td>0.005705</td>
      <td>0.005705</td>
      <td>0.022819</td>
      <td>0.005705</td>
      <td>0.005705</td>
      <td>0.005705</td>
      <td>0.022819</td>
      <td>0.005705</td>
      <td>...</td>
      <td>0.022819</td>
      <td>0.005705</td>
      <td>0.079866</td>
      <td>0.011409</td>
      <td>0.085571</td>
      <td>0.011409</td>
      <td>0.005705</td>
      <td>0.011409</td>
      <td>0.028524</td>
      <td>0.039933</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Democrat</td>
      <td>0.007566</td>
      <td>0.014051</td>
      <td>0.014051</td>
      <td>0.077823</td>
      <td>0.016213</td>
      <td>0.001081</td>
      <td>0.006485</td>
      <td>0.015132</td>
      <td>0.001081</td>
      <td>...</td>
      <td>0.108088</td>
      <td>0.004324</td>
      <td>0.081066</td>
      <td>0.087551</td>
      <td>0.074581</td>
      <td>0.004324</td>
      <td>0.004324</td>
      <td>0.086470</td>
      <td>0.074581</td>
      <td>0.010809</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 946 columns</p>
</div>




```python
platform_tfidf.to_csv("Party Platform - TF-IDF.csv")
```

### 4.3 Visualizing the Clean Party Platforms


```python
republican_platform_clean = platform_vec_lemmed[platform_vec_lemmed['Party'] == 'Republican']
```


```python
freq = republican_platform_clean.transpose().to_dict()
```


```python
freq = freq[0]
del freq['Party']
```


```python
sb.set_style("white")
sb.set(font='Times New Roman', font_scale=1.2)

wc = WordCloud(width=600,height=300, background_color='white',colormap='OrRd',max_words=250)
wc.generate_from_frequencies(freq)

## Plotting the cloud

plt.figure(figsize=(9,6),dpi=750)
plt.imshow(wc)
plt.axis('off')
plt.title("GOP Party Platform - 2024\nClean PDF Text")
plt.savefig("GOP Party Platform - Clean Text",dpi=1000);
```


    
![png](output_131_0.png)
    



```python
democrat_platform_clean = platform_vec_lemmed[platform_vec_lemmed['Party'] == 'Democrat']
```


```python
freq = democrat_platform_clean.transpose().to_dict()
```


```python
freq = freq[1]
del freq['Party']
```


```python
sb.set_style("white")
sb.set(font='Times New Roman', font_scale=1.2)

wc = WordCloud(width=600,height=300, background_color='white',colormap='Blues',max_words=250)
wc.generate_from_frequencies(freq)

## Plotting the cloud

plt.figure(figsize=(9,6),dpi=750)
plt.imshow(wc)
plt.axis('off')
plt.title("DNC Party Platform - 2024\nClean PDF Text")
plt.savefig("DNC Party Platform - Clean Text",dpi=1000);
```


    
![png](output_135_0.png)
    



```python

```
