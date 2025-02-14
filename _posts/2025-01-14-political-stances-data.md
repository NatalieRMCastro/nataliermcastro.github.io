---
layout: post
title: "Political Stances: Data"
categories: projects
published: true
in_feed: false
---

Data came from four sources. The NewsAPI, to collect information about public media exposure to partisan views of climate change. A combination of the Congress API, to generate metadata about all proposed Federal climate change bill information since the 93rd congress (1973), and then web scraping the Library of Congress to collect the bill text. Finally, auxiliary information from both Democrat and Republican Party Platforms about language used in the most recent presidential election. If you are interested in seeing exactly how I collected the data, you are welcome to reference [this page, which links to the MarkDown](https://nataliermcastro.github.io/projects/2025/02/13/climate-data-cleaning.html) version of my notebook or my [GitHub repository](https://github.com/NatalieRMCastro/climate-policy/blob/main/0.%20Data%20Collection%20-%20for%20website.ipynb) where you can download the IPYNB file.

### NewsAPI
The [NewsAPI] (https://newsapi.org/) is an easy to access API that provides key word queries to search for data in their repository of millions of articles. To understand partisan polarization in regard to climate change this is especially valuable, as many studies have begun to quantify the partisan lean of news websites. The keywords “climate change” and either “republican” or “democrat” were used to search NewsAPI. The URL query used to connect to the NewsAPI was created through a base url and url post system. The posted url is:

``` python
url_post = {'apiKey':api_key,
            'source':'everything',
            'q':'democrat+climate', ## this iteration will be looking at democrat referencing articles
            'language':'en', ## selecting English as the data
            'sortBy':'popularity', ## used to generate a popularity label
}
```

Another version of the POST URL was generated for Republican and Climate News sources. Returned from the query was the source, author, title of the article, description, url, and published date. A total of 1,559 news articles were found that have the keywords “Republican” and “climate change” and a total of 671 news articles were found for “Democrat” – this totals to 2,230 total articles collected. 


### Congess.gov API + Web Scraping 
The[ Congress.Gov API]( https://github.com/LibraryOfCongress/api.congress.gov) is a publicly available tool that can assist anyone who is interested in collecting government data. [Congress.gov]( https://www.congress.gov/), serves as a repository for the government to archive any Bill or Policy proceedings since the 1970s. To best iterative capture policy about climate change, I first used the Congress.gov front end to identify information about the bills and download my search results ([stored here]( https://github.com/NatalieRMCastro/climate-policy/tree/main/data/raw)). However, this data is not comprehensive and only provides information about the congress it was passed under, its latest action, the action note, bill number, origin chamber, title, and URL. I created functions to parse through the information to generate labels for Sponsor Affiliation (Republican or Democrat), Sponsor State,  and then a built URL to pass into the congress API. 

```python
''' BUILDING A FUNCTION FOR THE URLS '''
def url_builder(congress,bill_type,bill_number):
    base_url = "https://api.congress.gov/v3/bill/"
    
    request_url = ""
    request_url = request_url + base_url + str(congress) + "/" + bill_type + "/" + str(bill_number) + '/text?api_key=xxx'
    
    return (request_url)

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
Using my _url_builder_ function, it was able to iterate through the DataFrame, build a URL and then using  *xml_link_collector* the built URL would be passed in, called to the Congress.Gov API, and the XML URL (generated from parsing through the API response) would be the final output. My interest in collecting the text from the policy is for two primary reasons. First, to explore what language is used and potential similarities between Republican and Democrat Sponsored policies. Second, I am interested in understanding the similarities or differences between policy and news headlines on an intensely politicized issue. 

To collect the bill text, I then made an individual web-scraping call to the XML URL found. Throughout this process, it became clear that many of the older bills did not have a digitized version of the bill available, thus making the sample size smaller. A total of 3,262 policy documents were collected using the Congress.Gov API, or 40% of the entire climate related bills introduced at the federal level.



### Party Platform Declarations 

<section>
	<p><span class="image fit"><img src="/assets/images/DNC Party Platform - Raw Text.png" alt=""  /></span> </p>
	<p><span class="image fit"><img src="/assets/images/GOP Party Platform - Raw Text.png" alt="" /></span> </p>
</section>


<section>
---
Bibliography

</section>
