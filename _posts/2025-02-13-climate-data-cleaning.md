---
layout: post
title: "Political Stances - Data Cleaning"
categories: projects
published: true
in_feed: False
---
# 0. Data Collection
---
Author: Natalie Castro
Date: 1/15/2025

The purpose of this notebook is to collect varying forms of data from different sources on the Internet to answer the research question:

    What are characteristics of self-identified political parities expressed in 2025 in climate change?


## 🌐1. Environment Creation

### 1.1 Library Import


```python
''' DATA QUERYING '''
from bs4 import BeautifulSoup
import json
import requests
from time import sleep
import pypdf

''' DATA MANAGEMENT '''
import pandas as pd
import regex as re
```

    C:\Users\natal\miniconda3\lib\site-packages\pypdf\_crypt_providers\_cryptography.py:32: CryptographyDeprecationWarning: ARC4 has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.ARC4 and will be removed from this module in 48.0.0.
      from cryptography.hazmat.primitives.ciphers.algorithms import AES, ARC4
    

### 1.2 Secret Storage


```python
''' NEWSAPI KEY'''
api_key = 'xxxxxxxxxxxxxxxxxxxx'
```

## **📡2. API Requests**

The keywords used in this analysis will be "democrat+climate" and "republican+climate"

### **2.1 NewsAPI**

For NewsAPI, three types of data will be collected. First, using the 'everything' endpoint which collects every article in the past five years from their corpora using a keyword. Next, will be the top headlines on January 20th, President Donald Trump's inaguration day. The final endpoint used will be the 'sources' to better undestand media production for each key word during this date.

#### 2.1.1 NewsAPI: Everything Endpoint

The code shown below is how the I got the API up and running, this was then iterated upon with the parameter 'page' to extract the entire set of results.


```python
''' BUIDLING THE URL '''
base_url = "https://newsapi.org/v2/everything?"

url_post = {'apiKey':api_key,
            'source':'everything',
            'q':'democrat+climate', ## this iteration will be looking at democrat referencing articles
            'language':'en', ## selecting English as the data
            'sortBy':'popularity', ## used to generate a popularity label
}

url_post2 = {'apiKey':api_key,
            'source':'everything',
            'q':'republican+climate', ## this iteration will be looking at democrat referencing articles
            'language':'en', ## selecting English as the data
            'sortBy':'popularity', ## used to generate a popularity label
    
}
```


```python
''' MAKING THE REQUEST '''

response = requests.get(base_url,url_post)
```


```python
''' CHECKING OUT THE RESPONSE: DEMOCRAT '''
text = response.json()
```


```python
dem_text['articles'][0]
```




    {'source': {'id': None, 'name': 'CNET'},
     'author': 'Katie Collins',
     'title': 'For Progress on Climate and Energy in 2025, Think Local',
     'description': "As Trump and his anti-science agenda head for the White House, look to America's city and state leaders to drive climate action and prioritize clean energy.",
     'url': 'https://www.cnet.com/home/energy-and-utilities/for-progress-on-climate-and-energy-in-2025-think-local/',
     'urlToImage': 'https://www.cnet.com/a/img/resize/5fa89cffd3d573f39b4cf70398e5bb4b3038a2d7/hub/2024/12/31/540a7c2e-63f9-445f-a311-5744bcce16a2/us-map-localized-energy-progress.jpg?auto=webp&fit=crop&height=675&width=1200',
     'publishedAt': '2025-01-03T13:00:00Z',
     'content': 'With its sprawling canopy of magnolia, dogwood, southern pine and oak trees, Atlanta is known as the city in the forest. The lush vegetation helps offset the pollution from the commuter traffic as pe… [+16751 chars]'}




```python
rep_text = republican_response.json()
```


```python
rep_text.keys()
```




    dict_keys(['status', 'totalResults', 'articles'])




```python
rep_text['totalResults']
```




    1559




```python
rep_text['articles'][0]
```




    {'source': {'id': 'the-verge', 'name': 'The Verge'},
     'author': 'Nilay Patel',
     'title': 'Trump’s first 100 days: all the news impacting the tech industry',
     'description': 'President Donald Trump is taking on TikTok, electric vehicle policy, and AI in his first 100 days in office. This time around, he has the backing of many tech billionaires.',
     'url': 'https://www.theverge.com/24348851/donald-trump-presidency-tech-science-news',
     'urlToImage': 'https://cdn.vox-cdn.com/thumbor/Nwo4_i4giY8lRM0Rtzih1IHTSLU=/0x0:2040x1360/1200x628/filters:focal(1020x680:1021x681)/cdn.vox-cdn.com/uploads/chorus_asset/file/25531809/STK175_DONALD_TRUMP_CVIRGINIA_C.jpg',
     'publishedAt': '2025-01-22T14:30:00Z',
     'content': 'Filed under:\r\nByLauren Feiner, a senior policy reporter at The Verge, covering the intersection of Silicon Valley and Capitol Hill. She spent 5 years covering tech policy at CNBC, writing about antit… [+7943 chars]'}




The structure of the response is a nested dictionary, with each list entry in the response a dictionary for the respective news article.

**🫏Iterating for Democrat Articles**


```python
''' PAGE TURNER 

    INPUT: the desired page for the API call, the keyword used to build the URL
    OUTPUT: a list of the articles from the page
    
    The function page_turner is used to collect the entire corpus from the NEWSAPI
    for a particular keyword. This function is used wrapped into a for loop so 
    it can build a new url for each distinct page.

'''

base_url = "https://newsapi.org/v2/everything?"


def page_turner(page_number,keyword):
    sleep(2)
    ## Building the post URL for every page in the iteration
    url_post = {'apiKey':api_key,
            'source':'everything',
            'q':keyword, ## this iteration will be looking at democrat referencing articles
            'language':'en', ## selecting English as the data
            'sortBy':'popularity', ## used to generate a popularity label
            'page':page_number}
    
    
    response = requests.get(base_url,url_post)
    json_ = response.json()
    #print (json_.keys())
    return(json_['articles'])
```


```python
nested_responses_democrat = []

for page in range(1,7):
    page_contents = page_turner(page,'democrat+climate')
    nested_responses_democrat.append(page_contents)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[20], line 4
          1 nested_responses_democrat = []
          3 for page in range(1,7):
    ----> 4     page_contents = page_turner(page,'democrat+climate')
          5     nested_responses_democrat.append(page_contents)
    

    Cell In[13], line 29, in page_turner(page_number, keyword)
         27 json_ = response.json()
         28 #print (json_.keys())
    ---> 29 return(json_['articles'])
    

    KeyError: 'articles'



```python
nested_responses_republican = []

for page in range(1,17):
    page_contents = page_turner(page,'republican+climate')
    nested_responses_republican.append(page_contents)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[31], line 4
          1 nested_responses_republican = []
          3 for page in range(1,17):
    ----> 4     page_contents = page_turner(page,'republican+climate')
          5     nested_responses_republican.append(page_contents)
    

    Cell In[13], line 29, in page_turner(page_number, keyword)
         27 json_ = response.json()
         28 #print (json_.keys())
    ---> 29 return(json_['articles'])
    

    KeyError: 'articles'



```python
len(nested_responses_republican)
```




    5



### 🦅 2.2 Congress.Gov API 

The purpose of collecting data from this API is to understand how different political sides have represented + instutionalized their view about climate change.

[GitHub Documentation](https://github.com/LibraryOfCongress/api.congress.gov/)  
[Using Congress Data Offsite](https://www.congress.gov/help/using-data-offsite)  
[Congress API Endpoints](https://gpo.congress.gov/#/)  
[Python Code Examples](https://github.com/LibraryOfCongress/api.congress.gov/tree/main/api_client/python)


```python
api_key = 'XXX'
```


```python
base_url = 'https://api.congress.gov/v3/bill?api_key=te7ilzFKEeAOrjfEalH5mrtFU0Dw35E6B70Nfhnn'

url_post = {
            'format':'json', # specifying the response format
            'offset':0, ## specifying the start of the records returned,
            'limit':10 ## specifying the number of records returned
            }

## For government APIs, it's generally good practice to provide some sort of user agent!
user_agent = {'user-agent': 'University of Colorado at Boulder, natalie.castro@colorado.edu'}
```


```python
''' TEST 2'''
test2_response = requests.get(base_url,url_post,headers=user_agent)
```


```python
test2_response
```




    <Response [200]>



### 2.2.1 Collecting Bill Numbers

To do so, I will be changing the URL for a few different parameters and scraping the congress site. The filters generated are: legislation any status of legislation, and environmental protection policy area.

The URL for the bill search (as of 1/28/2025) is:https://www.congress.gov/search?q=%7B%22congress%22%3A%22all%22%2C%22source%22%3A%22all%22%2C%22bill-status%22%3A%22all%22%2C%22subject%22%3A%22Environmental+Protection%22%7D

A CSV was downloaded with the bill numbers with a total of 8,056 bills. The original downloaded CSV comes with three 'metadata' lines, these were consisted of the date collected, and the URL, which I have listed here. The three lines were deleted to read them in using Pandas.


```python
bill_information1 = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\epa_bills_119_113.csv",encoding='utf-8')
bill_information2 = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\epa_bills_112_103.csv",encoding='utf-8')
bill_information3 = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\epa_bills_102_95.csv",encoding='utf-8')
bill_information4 = pd.read_csv(r"C:\Users\natal\OneDrive\university\info 5653\data\epa_bills_94_93.csv",encoding='utf-8')

bill_information = pd.concat([bill_information1,bill_information2,bill_information3,bill_information4])
```


```python
bill_information.head()
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
      <th>Legislation Number</th>
      <th>URL</th>
      <th>Congress</th>
      <th>Title</th>
      <th>Sponsor</th>
      <th>Date of Introduction</th>
      <th>Committees</th>
      <th>Latest Action</th>
      <th>Latest Action Date</th>
      <th>Number of Cosponsors</th>
      <th>Amends Bill</th>
      <th>Date Offered</th>
      <th>Date Submitted</th>
      <th>Date Proposed</th>
      <th>Amends Amendment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>H.R. 375</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>Continued Rapid Ohia Death Response Act of 2025</td>
      <td>Tokuda, Jill N. [Rep.-D-HI-2] (Introduced 01/1...</td>
      <td>1/13/2025</td>
      <td>House - Natural Resources, Agriculture | Senat...</td>
      <td>Received in the Senate and Read twice and refe...</td>
      <td>1/24/2025</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>H.R. 349</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>Goldie’s Act</td>
      <td>Malliotakis, Nicole [Rep.-R-NY-11] (Introduced...</td>
      <td>1/13/2025</td>
      <td>House - Agriculture</td>
      <td>Referred to the House Committee on Agriculture.</td>
      <td>1/13/2025</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>H.R. 313</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>Natural Gas Tax Repeal Act</td>
      <td>Pfluger, August [Rep.-R-TX-11] (Introduced 01/...</td>
      <td>1/9/2025</td>
      <td>House - Energy and Commerce</td>
      <td>Referred to the House Committee on Energy and ...</td>
      <td>1/9/2025</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>H.R. 288</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>Long Island Sound Restoration and Stewardship ...</td>
      <td>LaLota, Nick [Rep.-R-NY-1] (Introduced 01/09/2...</td>
      <td>1/9/2025</td>
      <td>House - Transportation and Infrastructure, Nat...</td>
      <td>Referred to the Subcommittee on Water Resource...</td>
      <td>1/10/2025</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>H.R. 284</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>GLRI Act of 2025</td>
      <td>Joyce, David P. [Rep.-R-OH-14] (Introduced 01/...</td>
      <td>1/9/2025</td>
      <td>House - Transportation and Infrastructure</td>
      <td>Referred to the Subcommittee on Water Resource...</td>
      <td>1/10/2025</td>
      <td>28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
''' CREATING CONGRESS TITLES '''
def congress_finder(current_congress): 
    ## looking for, but not including the th or st
    regex_pattern = '[0-9]{2,3}(?!=[a-z]{2})'
    congress_match = re.findall(regex_pattern,current_congress)
    
    congress_num = int(congress_match[0])
    
    return(congress_num)
```


```python
bill_information['Congress Number'] = bill_information['Congress'].apply(lambda x: congress_finder(x))
```


```python
''' CREATING A COLUMN FOR BILL TYPE:

The structure of all of the legislation numbers is [BILL TYPE]_[BILL NUMBER]. I will be using
Regex to extract and separate these columns
'''
```


```python
''' BILL TYPE CLEANER: This function is used in a lambda apply down the rows to create the 
                        bill types needed for the congress.gov API

'''

bt_pattern = r'[A-Za-z]+\.*'
def bill_type_cleaner(bt):
    matches = re.findall(bt_pattern,bt)
    type_dirty = ''.join(matches)
    type_text = re.sub("\.","",type_dirty)
    type_clean = type_text.lower()
    return (type_clean)
```


```python
bill_information['Bill Type'] = bill_information['Legislation Number'].apply(lambda x: bill_type_cleaner(x))
```


```python
''' CREATING A COLUMN FOR BILL NUMBER '''
bt_num_pattern = r'[^A-Za-z\.]'
def bill_num_cleaner(bt):
    matches = re.findall(bt_num_pattern,bt)
    type_dirty = ''.join(matches)
    type_clean = type_dirty.lower().strip()
    return (int(type_clean)) ## The API asks for an integer
```


```python
bill_information['Bill Number'] = bill_information['Legislation Number'].apply(lambda x: bill_num_cleaner(x))
```


```python
''' CREATING A COLUMN FOR SPONSOR AFFILIATION & CREATING A COLUMN FOR SPONSOR STATE ''' '''
```


```python
affiliation_pattern = r'-[DRI]'
state_pattern = r'-[A-Z]{2}'
def affiliation_finder(sponsor):
    ## For party affiliation
    match = re.findall(affiliation_pattern,sponsor)
    clean_affiliation = re.sub("-","",match[0])
    
    return (clean_affiliation)

def state_finder(sponsor):
    
    ## For State affiliation
    state_match = re.findall(state_pattern,sponsor)
    clean_state = re.sub("-",'',state_match[0])
    return (clean_state)
```


```python
bill_information['Sponser Affiliation'] = bill_information['Sponsor'].apply(lambda x: affiliation_finder(x))
bill_information['Sponser State'] = bill_information['Sponsor'].apply(lambda x: state_finder(x))
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
      <th>Legislation Number</th>
      <th>URL</th>
      <th>Congress</th>
      <th>Title</th>
      <th>Sponsor</th>
      <th>Date of Introduction</th>
      <th>Committees</th>
      <th>Latest Action</th>
      <th>Latest Action Date</th>
      <th>Number of Cosponsors</th>
      <th>Amends Bill</th>
      <th>Date Offered</th>
      <th>Date Submitted</th>
      <th>Date Proposed</th>
      <th>Amends Amendment</th>
      <th>Congress Number</th>
      <th>Bill Type</th>
      <th>Bill Number</th>
      <th>Sponser Affiliation</th>
      <th>Sponser State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>H.R. 375</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>Continued Rapid Ohia Death Response Act of 2025</td>
      <td>Tokuda, Jill N. [Rep.-D-HI-2] (Introduced 01/1...</td>
      <td>1/13/2025</td>
      <td>House - Natural Resources, Agriculture | Senat...</td>
      <td>Received in the Senate and Read twice and refe...</td>
      <td>1/24/2025</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>119</td>
      <td>hr</td>
      <td>375</td>
      <td>D</td>
      <td>HI</td>
    </tr>
    <tr>
      <th>1</th>
      <td>H.R. 349</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>Goldie’s Act</td>
      <td>Malliotakis, Nicole [Rep.-R-NY-11] (Introduced...</td>
      <td>1/13/2025</td>
      <td>House - Agriculture</td>
      <td>Referred to the House Committee on Agriculture.</td>
      <td>1/13/2025</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>119</td>
      <td>hr</td>
      <td>349</td>
      <td>R</td>
      <td>NY</td>
    </tr>
  </tbody>
</table>
</div>




```python
bill_information.reset_index(inplace=True)
bill_information.drop(columns='index',inplace=True)
```

### 2.2. Collecting Bill Text URLS to Scrape from the Congress API


```python
''' BUILDING A FUNCTION FOR THE URLS '''
def url_builder(congress,bill_type,bill_number):
    base_url = "https://api.congress.gov/v3/bill/"
    
    request_url = ""
    request_url = request_url + base_url + str(congress) + "/" + bill_type + "/" + str(bill_number) + '/text?api_key=te7ilzFKEeAOrjfEalH5mrtFU0Dw35E6B70Nfhnn'
    
    return (request_url)
```


```python
user_agent = {'user-agent': 'University of Colorado at Boulder, natalie.castro@colorado.edu'}

''' BUILDING A FUNCTION FOR THE API REQUESTS TO GATHER THE DATA '''
def xml_link_collector(url):
    ## Making a response with our URL
    response = requests.get(url,headers=user_agent)
    
    ## Making sure the response was valid
    try:
        out = response.json()
        collected_url = out['textVersions'][0]['formats'][2]['url'] ## This was determined through parsing the output for a test example
        return (collected_url)
    except:
        print (f"⚠️ uh oh! there was an error when using the API with this url:{url}\n ")
        return ("NO URL FOUND")
```


```python
url = url_builder(119,'hjres',30)
```


```python
xml_link_collector(url)
```




    'https://www.congress.gov/119/bills/hjres30/BILLS-119hjres30ih.xml'



### Collecting the Bill Information:

This will take place in two parts, because the rate limit for the congress API is 5000 requests per hour


```python
''' SCRAPING XMLS USING THE API -- [PART 1]'''

url_list = []

for row in range(0,len(bill_information[0:4998])):
    congress = bill_information.at[row,'Congress Number']
    bill_type = bill_information.at[row,'Bill Type']
    bill_number = bill_information.at[row,'Bill Number']
    
    current_url = url_builder(congress,bill_type,bill_number)
    
    ## Making sure the API connection is well rested (ie - avoiding the rate limit)
    if row % 100 == 0:
        sleep (5)
    
    xml_found = xml_link_collector(current_url)
    
    url_list.append(xml_found)

```

The otuput from the above cell was removed because there was a lot of errors, and when converting to HTML I did not want to bog down the page! Here is what some of the errors looked like
![image.png](image.png)


```python
print (len(url_list))
```

    4998
    


```python
url_list[3249]
```




    'https://www.congress.gov/109/bills/s2920/BILLS-109s2920is.xml'




```python
url_list = url_list[:3250]
```


```python
url_list[3249:3250]
```




    ['https://www.congress.gov/109/bills/s2920/BILLS-109s2920is.xml']




```python
saving_urls = pd.DataFrame(url_list)
saving_urls.to_csv("Found URLs for Bills 0 - 3250.csv")
```


```python
''' SCRAPING XMLS USING THE API -- [PART 2] '''

for row in range(3250,len(bill_information[3250:])):
    congress = bill_information.at[row,'Congress Number']
    bill_type = bill_information.at[row,'Bill Type']
    bill_number = bill_information.at[row,'Bill Number']
    
    current_url = url_builder(congress,bill_type,bill_number)
    
    ## Making sure the API connection is well rested (ie - avoiding the rate limit)
    if row % 100 == 0:
        sleep (5)
    
    xml_found = xml_link_collector(current_url)
    
    url_list.append(xml_found)
```


```python
## Testing that all went well...

print (f"The expected length of the URL list should be {len(bill_information)}.\nThe actual length of the URL list is {len(url_list)}.")
```

    The expected length of the URL list should be 8056.
    The actual length of the URL list is 4806.
    

#### Uh-Oh! I think a lot of the older bills do not have the ability to be called for API access

To fix this, I am going to deconstruct the URLs and then only get the information on those Bills.

At this stage, I am filtering to look for what Bills do have available URLs


```python
urls_raw = pd.DataFrame(url_list)
urls_raw.rename(columns={0:"URL"},inplace=True)
```


```python
congress_pattern = '(?<=/)[0-9]{2,3}'
bt_pattern = '(?<=s/)[h,c,r,o,n,e,s,j,r]{1,7}'
bill_num_pattern = '(?<=[a-z])\d{1,4}(?!=\/BILL)'
```


```python
url_congress = []
url_bt = []
url_num = []

for row in range(0,len(urls_raw)):
    curr_url = urls_raw.at[row,'URL']
    
    congress = re.findall(congress_pattern,curr_url)
    bill_type = re.findall(bt_pattern,curr_url)
    bill_num = re.findall(bill_num_pattern,curr_url)
    
    if len(congress) > 0:
        if len(bill_type) > 0:
            if len(bill_num) > 0:
                url_congress.append(int(congress[0]))
                url_bt.append(bill_type[0])
                url_num.append(int(bill_num[0]))
                
            else:
                url_num.append('DROP')
        else:
            url_bt.append('DROP')
    else:
        url_congress.append('DROP')
        url_bt.append('DROP')
        url_num.append('DROP')
    
    
```


```python
urls_raw['Congress Number'] = url_congress
urls_raw['Bill Type'] = url_bt
urls_raw['Bill Number'] = url_num
```


```python
## Cleaning the URLs
urls_raw.drop_duplicates(inplace=True)
```


```python
## And dropping if there was any error in the regex find all
congress_condition = urls_raw['Congress Number'] != 'DROP'
bt_condition = urls_raw['Bill Type'] != 'DROP'
bn_condition = urls_raw['Bill Number'] != 'DROP'

urls_raw1 = urls_raw[congress_condition]
urls_raw2 = urls_raw1[bt_condition]
urls_clean = urls_raw2[bn_condition]
```

    C:\Users\natal\AppData\Local\Temp\ipykernel_13272\878941524.py:7: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      urls_raw2 = urls_raw1[bt_condition]
    C:\Users\natal\AppData\Local\Temp\ipykernel_13272\878941524.py:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      urls_clean = urls_raw2[bn_condition]
    


```python
## Now mergine back from the Bill Information

bill_information_final = pd.merge(left=urls_clean,right=bill_information, on=['Congress Number','Bill Type','Bill Number'],validate='1:1')
```


```python
len(bill_information_final)
```




    3262




```python
## Now Saving the Supplemented DataFrame 
bill_information_final.to_csv("Bill Information Supplemented.csv")
```


```python
url_list = bill_information_final['API URL'].to_list()
```


```python
bill_information_final.head(2)
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
      <th>URL</th>
      <th>Congress</th>
      <th>Title</th>
      <th>Sponsor</th>
      <th>Date of Introduction</th>
      <th>...</th>
      <th>Latest Action</th>
      <th>Latest Action Date</th>
      <th>Number of Cosponsors</th>
      <th>Amends Bill</th>
      <th>Date Offered</th>
      <th>Date Submitted</th>
      <th>Date Proposed</th>
      <th>Amends Amendment</th>
      <th>Sponser Affiliation</th>
      <th>Sponser State</th>
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
      <td>Received in the Senate and Read twice and refe...</td>
      <td>1/24/2025</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>D</td>
      <td>HI</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.congress.gov/119/bills/hr349/BILLS...</td>
      <td>119</td>
      <td>hr</td>
      <td>349</td>
      <td>H.R. 349</td>
      <td>https://www.congress.gov/bill/119th-congress/h...</td>
      <td>119th Congress (2025-2026)</td>
      <td>Goldie’s Act</td>
      <td>Malliotakis, Nicole [Rep.-R-NY-11] (Introduced...</td>
      <td>1/13/2025</td>
      <td>...</td>
      <td>Referred to the House Committee on Agriculture.</td>
      <td>1/13/2025</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>R</td>
      <td>NY</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>



### 2.2.2 🕸️Scraping XML URLs

To get the text for each bill, the XML URL will be scraped and appended to the dataframe we are working with.


```python
''' XML-PARSER:

In the XML output, I am interested in the Bill Title and Text, although the title is already in the Bill Information,
I just want to make sure everything is correct! This parser will take an input of an XML url, make the request, parse 
in the soup, and return the output as a title and text!

No preprocessing will occur at this stage, and the raw text will just be appended as a column to the Bill Information. 
Later on, this will make it easier to generate the labels

'''

def xml_searcher(xml_url):
    xml_output = requests.get(xml_url)
    raw_xml = xml_output.text
    
    ## Using Beautiful Soup as a parser
    xml_soup = BeautifulSoup(raw_xml,'xml')
    
    ## Parsing for the title
    current_title = xml_soup.title
    
    ## Parsing for the text
    current_text = xml_soup.text
    
    return (current_title,current_text)
```


```python
test_title ,test_text = xml_searcher(url_list[400])
```


```python
test_title
```




    <dc:title>118 S2959 RS: Brownfields Reauthorization Act of 2023</dc:title>




```python
test_text
```




    '\n\n118 S2959 RS: Brownfields Reauthorization Act of 2023\nU.S. Senate\n2023-09-27\ntext/xml\nEN\nPursuant to Title 17 Section 105 of the United States Code, this file is not subject to copyright protection and is in the public domain.\n\n\n\nIICalendar No. 214118th CONGRESS1st SessionS. 2959IN THE SENATE OF THE UNITED STATESSeptember 27 (legislative day, September 22), 2023Mr. Carper, from the Committee on Environment and Public Works, reported the following original bill; which was read twice and placed on the calendarA BILLTo amend the Comprehensive Environmental Response, Compensation, and Liability Act of 1980 to reauthorize brownfields revitalization funding, and for other purposes.1.Short titleThis Act may be cited as the Brownfields Reauthorization Act of 2023.2.Improving small and disadvantaged community access to grant opportunitiesSection 104(k) of the Comprehensive Environmental Response, Compensation, and Liability Act of 1980 (42 U.S.C. 9604(k)) is amended—(1)in paragraph (1)(I), by inserting or 501(c)(6) after section 501(c)(3);(2)in paragraph (5), by striking subparagraph (E);(3)in paragraph (6)(C), by striking clause (ix) and inserting the following:(ix)The extent to which the applicant has a plan—(I)to engage a diverse set of local groups and organizations that effectively represent the views of the local community that will be directly affected by the proposed brownfield project; and(II)to meaningfully involve the local community described in subclause (I) in making decisions relating to the proposed brownfield project.;(4)in paragraph (10)(B)(iii)—(A)by striking 20 percent and inserting 10 percent;(B)by inserting the eligible entity is located in a small community or disadvantaged area (as those terms are defined in section 128(a)(1)(B)(iv)) or after unless; and(C)by inserting , in which case the Administrator shall waive the matching share requirement under this clause before ; and; and(5)in paragraph (13), by striking 2019 through 2023 and inserting 2024 through 2029.3.Increasing grant amountsSection 104(k)(3)(A)(ii) of the Comprehensive Environmental Response, Compensation, and Liability Act of 1980 (42 U.S.C. 9604(k)(3)(A)(ii)) is amended by striking $500,000 and all that follows through the period at the end and inserting $1,000,000 for each site to be remediated..4.State response programsSection 128(a) of the Comprehensive Environmental Response, Compensation, and Liability Act of 1980 (42 U.S.C. 9628(a)) is amended—(1)in paragraph (1)(B)(i), by striking or enhance and inserting , enhance, or implement; and(2)by striking paragraph (3) and inserting the following:(3)Authorization of appropriationsThere are authorized to be appropriated to carry out this subsection—(A)$50,000,000 for fiscal year 2024;(B)$55,000,000 for fiscal year 2025;(C)$60,000,000 for fiscal year 2026;(D)$65,000,000 for fiscal year 2027;(E)$70,000,000 for fiscal year 2028; and(F)$75,000,000 for fiscal year 2029..5.Report to identify opportunities to streamline application process; updating guidance(a)ReportNot later than 1 year after the date of enactment of this Act, the Administrator of the Environmental Protection Agency (referred to in this section as the Administrator) shall submit to Congress a report that evaluates the application ranking criteria and approval process for grants and loans under section 104(k) of the Comprehensive Environmental Response, Compensation, and Liability Act of 1980 (42 U.S.C. 9604(k)), which shall include, with respect to those grants and loans—(1)an evaluation of the shortcomings in the existing application requirements that are a recurring source of confusion for potential recipients of those grants or loans;(2)an identification of the most common sources of point deductions on application reviews;(3)strategies to incentivize the submission of applications from small communities and disadvantaged areas (as those terms are defined in section 128(a)(1)(B)(iv) of that Act (42 U.S.C. 9628(a)(1)(B)(iv)); and(4)recommendations, if any, to Congress on suggested legislative changes to the ranking criteria that would achieve the goal of streamlining the application process for small communities and disadvantaged areas (as so defined).(b)Updating guidanceNot later than 1 year after the date of enactment of this Act, the Administrator shall update the guidance relating to the application ranking criteria and approval process for grants and loans under section 104(k) of the Comprehensive Environmental Response, Compensation, and Liability Act of 1980 (42 U.S.C. 9604(k)) to reduce the complexity of the application process while ensuring competitive integrity.6.Brownfield revitalization funding for Alaska Native tribesSection 104(k)(1) of the Comprehensive Environmental Response, Compensation, and Liability Act of 1980 (42 U.S.C. 9604(k)(1)) is amended—(1)in subparagraph (G), by striking other than in Alaska; and(2)by striking subparagraph (H) and inserting the following:(H)a Regional Corporation or a Village Corporation (as those terms are defined in section 3 of the Alaska Native Claims Settlement Act (43 U.S.C. 1602));.September 27 (legislative day, September 22), 2023Read twice and placed on the calendar'




```python
''' APPLYING THE XML PARSER '''
title_list = []
text_list = []

for url in url_list:
    try:
        title ,text = xml_searcher( url)

        title_list.append(title)
        text_list.append(text)
    except:
        title_list.append("ERROR")
        text_list.append("ERROR")
```


```python
## Checking that everything went smoothly
print (len(title_list))
print (len(text_list))
```

    3262
    3262
    


```python
bill_information_final['Bill Title (XML)'] = title_list
bill_information_final['Bill Text'] = text_list
```


```python
## Saving just in case :)
bill_information.to_csv("Bill Information Supplemented.csv")
```

## 👷🏻‍♀️ 3. Data Structuring

At this stage, the data will not be cleaned and will be done so in a subsequent notebook. The data is structured using the provided query from the API or webscrape and is not altered.

### 3.1 News API

#### 3.1.1 NewsAPI: Everything


```python
''' DATAFRAME GENERATION - DEMOCRAT '''
concatenate_democrats = []
for dataset in nested_responses_democrat:
    current_dataframe = pd.DataFrame(dataset)
    concatenate_democrats.append(current_dataframe)
```


```python
democrat_articles = pd.concat(concatenate_democrats)
```


```python
print (len(democrat_articles))
```

    444
    


```python
democrat_articles.head(2)
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
      <td>{'id': None, 'name': 'CNET'}</td>
      <td>Katie Collins</td>
      <td>For Progress on Climate and Energy in 2025, Th...</td>
      <td>As Trump and his anti-science agenda head for ...</td>
      <td>https://www.cnet.com/home/energy-and-utilities...</td>
      <td>https://www.cnet.com/a/img/resize/5fa89cffd3d5...</td>
      <td>2025-01-03T13:00:00Z</td>
      <td>With its sprawling canopy of magnolia, dogwood...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'id': 'time', 'name': 'Time'}</td>
      <td>Will Weissert and Chris Megerian / AP</td>
      <td>Trump to Visit Disaster-Stricken California an...</td>
      <td>President Trump is heading to hurricane-batter...</td>
      <td>https://time.com/7209700/trump-los-angeles-wil...</td>
      <td>https://api.time.com/wp-content/uploads/2025/0...</td>
      <td>2025-01-24T06:30:00Z</td>
      <td>WASHINGTON President Donald Trump is heading t...</td>
    </tr>
  </tbody>
</table>
</div>




```python
democrat_articles.describe()
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
      <th>count</th>
      <td>444</td>
      <td>402</td>
      <td>444</td>
      <td>443</td>
      <td>444</td>
      <td>319</td>
      <td>444</td>
      <td>444</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>119</td>
      <td>285</td>
      <td>421</td>
      <td>429</td>
      <td>444</td>
      <td>315</td>
      <td>424</td>
      <td>441</td>
    </tr>
    <tr>
      <th>top</th>
      <td>{'id': None, 'name': 'Freerepublic.com'}</td>
      <td>Breitbart</td>
      <td>Democrat Sen. Markey: L.A. Fires Are ‘Climate ...</td>
      <td>Democrat Massachusetts Sen. Ed Markey has clai...</td>
      <td>https://www.cnet.com/home/energy-and-utilities...</td>
      <td>https://static.dw.com/image/71400751_6.jpg</td>
      <td>2025-01-25T05:00:00Z</td>
      <td>In a confirmation hearing on Thursday, Democra...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>119</td>
      <td>14</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
''' DATAFRAME GENERATION - REPUBLICAN '''
concatenate_republicans = []
for dataset in nested_responses_republican:
    current_dataframe = pd.DataFrame(dataset)
    concatenate_republicans.append(current_dataframe)
```


```python
republican_articles = pd.concat(concatenate_republicans)
```


```python
republican_articles.head(2)
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
    <tr>
      <th>1</th>
      <td>{'id': None, 'name': 'Gizmodo.com'}</td>
      <td>Kate Yoder, Grist</td>
      <td>The Quiet Death of Biden’s Climate Corps—and W...</td>
      <td>Biden's green jobs program was never what it s...</td>
      <td>https://gizmodo.com/the-quiet-death-of-bidens-...</td>
      <td>https://gizmodo.com/app/uploads/2025/01/Americ...</td>
      <td>2025-01-18T15:00:26Z</td>
      <td>Giorgio Zampaglione loved his two-hour commute...</td>
    </tr>
  </tbody>
</table>
</div>




```python
republican_articles.describe()
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
      <th>count</th>
      <td>376</td>
      <td>350</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>373</td>
      <td>376</td>
      <td>376</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>93</td>
      <td>267</td>
      <td>375</td>
      <td>365</td>
      <td>376</td>
      <td>369</td>
      <td>371</td>
      <td>370</td>
    </tr>
    <tr>
      <th>top</th>
      <td>{'id': None, 'name': 'Forbes'}</td>
      <td>ABC News</td>
      <td>US to withdraw from Paris agreement, expand dr...</td>
      <td>Organizations including Walmart, Lowe’s and Me...</td>
      <td>https://www.theverge.com/24348851/donald-trump...</td>
      <td>https://imageio.forbes.com/specials-images/ima...</td>
      <td>2025-01-16T10:00:00Z</td>
      <td>&lt;ul&gt;&lt;li&gt;Trump endorses House Speaker Mike John...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>29</td>
      <td>18</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
''' DATA STORAGE

This is saving all of the raw data (minimal structuring) to their respective CSV files.

'''

## Everything endpoint
republican_articles.to_csv("NEWSAPI - republican climate articles raw.csv")
democrat_articles.to_csv("NEWSAPI - democrat climate articles raw.csv")
```

#### 3.1.2 NewsAPI sources

The sources for each type will be removed from each headline with their authors. This is to better understand who is representing and providing narrative to each party.


```python
democrat_sources = democrat_articles[['source','author']]
republican_sources = republican_articles[['source','author']]
```


```python
len(democrat_sources)
```




    444




```python
democrat_sources.head(2)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'id': None, 'name': 'CNET'}</td>
      <td>Katie Collins</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'id': 'time', 'name': 'Time'}</td>
      <td>Will Weissert and Chris Megerian / AP</td>
    </tr>
  </tbody>
</table>
</div>




```python
republican_sources.head(2)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'id': 'the-verge', 'name': 'The Verge'}</td>
      <td>Nilay Patel</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'id': None, 'name': 'Gizmodo.com'}</td>
      <td>Kate Yoder, Grist</td>
    </tr>
  </tbody>
</table>
</div>




```python
''' SOURCE CLEANER:

    INPUT: a list of sources in a dictionary structure with the key 'name'
    OUTPUT: two lists: the first contains the entire list of sources 
            cleaned, and the second is only the unique sources
            
    The purpose of this function is to clean the results from the everything
    endpoint from the NewsAPI. The result will be a list (can be appened)
    to a new dataframe to match the authors of clean sources. The second
    return of the function is a unique list of sources

'''


def source_cleaner(source_list):
    ## Creating a storage container for the cleaned sources
    cleaned_sources = []
    
    ## Iterating through each source in the provided list
    for source in source_list:
        ## Obtaining the name + storing it
        current_source = source['name']
        
        cleaned_sources.append(current_source)
    
    ## Finding the Unique Sources from the list
    unique_sources = list(set(cleaned_sources))
    
    return (cleaned_sources,unique_sources)
```


```python
dem_sources_full = democrat_sources['source'].to_list().copy()
rep_sources_full = republican_sources['source'].to_list().copy()
```


```python
dem_cleaned_sources,dem_unique_sources = source_cleaner(dem_sources_full)
rep_cleaned_sources,rep_unique_sources = source_cleaner(rep_sources_full)
```


```python
democrat_sources['source'] = dem_cleaned_sources
republican_sources['source'] = rep_cleaned_sources
```

    C:\Users\natal\AppData\Local\Temp\ipykernel_22124\2483140677.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      democrat_sources['source'] = dem_cleaned_sources
    C:\Users\natal\AppData\Local\Temp\ipykernel_22124\2483140677.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      republican_sources['source'] = rep_cleaned_sources
    


```python
dem_list = []
for i in range(0,len(democrat_sources)):
    dem_list.append("Democrat")
    
democrat_sources['Party'] = dem_list

rep_list = []
for i in range(0,len(republican_sources)):
    rep_list.append("Republican")
    
republican_sources['Party'] = rep_list
```

    C:\Users\natal\AppData\Local\Temp\ipykernel_22124\654442435.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      democrat_sources['Party'] = dem_list
    C:\Users\natal\AppData\Local\Temp\ipykernel_22124\654442435.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      republican_sources['Party'] = rep_list
    


```python
democrat_sources.fillna('No Author',inplace=True)
republican_sources.fillna('No Author',inplace=True)
```

    C:\Users\natal\AppData\Local\Temp\ipykernel_22124\1565605381.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      democrat_sources.fillna('No Author',inplace=True)
    C:\Users\natal\AppData\Local\Temp\ipykernel_22124\1565605381.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      republican_sources.fillna('No Author',inplace=True)
    


```python
democrat_sources
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
      <th>Party</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CNET</td>
      <td>Katie Collins</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time</td>
      <td>Will Weissert and Chris Megerian / AP</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Politicopro.com</td>
      <td>Blanca Begert, Camille von Kaenel, Thomas Fran...</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Scientific American</td>
      <td>Tanya Lewis</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Vox</td>
      <td>Benji Jones</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>PBS</td>
      <td>Will Weissert, Associated Press, Amelia Thomso...</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>96</th>
      <td>PBS</td>
      <td>Michelle L. Price, Associated Press</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>97</th>
      <td>PBS</td>
      <td>Bernard McGhee, Associated Press</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>98</th>
      <td>The Times of India</td>
      <td>Navtej Sarna</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>99</th>
      <td>The Times of India</td>
      <td>AP</td>
      <td>Democrat</td>
    </tr>
  </tbody>
</table>
<p>444 rows × 3 columns</p>
</div>




```python
republican_sources
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
      <th>Party</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Verge</td>
      <td>Nilay Patel</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gizmodo.com</td>
      <td>Kate Yoder, Grist</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBC News</td>
      <td>No Author</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBC News</td>
      <td>No Author</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBC News</td>
      <td>No Author</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>75</th>
      <td>MSNBC</td>
      <td>Jasen Castillo, John Schuessler, Miranda Priebe</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>76</th>
      <td>MSNBC</td>
      <td>Jen Psaki</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Themorningnews.org</td>
      <td>The Morning News</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Finextra</td>
      <td>Editorial Team</td>
      <td>Republican</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Japan Today</td>
      <td>No Author</td>
      <td>Republican</td>
    </tr>
  </tbody>
</table>
<p>376 rows × 3 columns</p>
</div>




```python
all_sources = pd.concat([democrat_sources,republican_sources])
```

## 4. Supplementary Media

The supplementary media collected in this section is two PDF files from each of the respective parties. The party platform is the document produced by each party, to proclaim their goals and resolutions if they take office.


```python
dem_pdf= pypdf.PdfReader(r"C:\Users\natal\OneDrive\university\info 5653\data\2024_democratic_party_platform.pdf",strict=True)
```


```python
rep_pdf = pypdf.PdfReader(r"C:\Users\natal\OneDrive\university\info 5653\data\2024_republican_party_platform.pdf")
```


```python
''' EXPLORING THE METADATA'''
dem_pdf.metadata
```




    {'/Title': 'FINAL MASTER PLATFORM',
     '/Producer': 'Skia/PDF m129 Google Docs Renderer'}




```python
dem_pdf.
```




    <bound method PdfReader.decode_permissions of <pypdf._reader.PdfReader object at 0x00000206AB5438B0>>




```python
len(dem_pdf.pages)
```




    92




```python
rep_pdf.metadata
```




    {'/CreationDate': "D:20240710083033-05'00'",
     '/Creator': 'Adobe InDesign 19.4 (Macintosh)',
     '/ModDate': "D:20240710083036-05'00'",
     '/Producer': 'Adobe PDF Library 17.0',
     '/Trapped': '/False'}




```python
len(rep_pdf.pages)
```




    28



#### 4.3.1 Extracting Text


```python
''' A SIMPLE EXTRACTION TEXT'''
page1 = rep_pdf.pages[0]
```


```python
print(page1.extract_text())
```

    4343RDRD REPUBLICAN NATIONAL CONVENTION REPUBLICAN NATIONAL CONVENTION
    PLATFORMTHE 2024 REPUBLICAN
    MAKE AMERICA GREAT AGAIN!
    


```python
''' EXTRACTING TEXT FROM DEMOCRAT DOCUMENT'''
democrat_party_platform = []

for page in range(0,len(dem_pdf.pages)):
    current_page = dem_pdf.pages[page]
    current_text = current_page.extract_text()
    democrat_party_platform.append(current_text)
```


```python
''' EXTRACTING TEXT FROM REPUBLICAN DOCUMENT'''
republican_party_platform = []

for page in range(0,len(rep_pdf.pages)):
    current_page = rep_pdf.pages[page]
    current_text = current_page.extract_text()
    republican_party_platform.append(current_text)
```


```python
''' COMBINING INTO ONE TEXT PER PARTY '''
democrat_party_platform_all = ' '.join(democrat_party_platform)
republican_party_platform_all = ' '.join(republican_party_platform)
```


```python
''' SAVING THE RAW TEXTS AS .TXTS'''
with open("democrat_party_platform.txt", "w",errors='replace') as file:
    file.write(democrat_party_platform_all)
```


```python
with open("republican_party_platform.txt", "w",errors='replace') as file:
    file.write(republican_party_platform_all)
```


```python

```
