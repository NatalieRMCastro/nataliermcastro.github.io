---
layout: post
title: "Political Stances: Data"
categories: projects
published: true
in_feed: false
---
<section>
    <div class="row">
        <div class="col-6 col-12-small">
            <ul class="actions" style="display: flex; gap: 10px; list-style: none; padding: 0;">
                <li><a href="https://nataliermcastro.github.io/projects/2025/01/14/political-stances.html" class="button fit small">Navigate to Project Page</a></li>
            </ul>
        </div>
    </div> 
</section> 

Data for this project originated from four sources. The NewsAPI, to collect information about public media exposure to partisan views of climate change. A combination of the Congress API, to generate metadata about all proposed Federal climate change bill information since the 93rd congress (1973), and then web scraping the Library of Congress to collect the bill text. Finally, auxiliary information from both Democrat and Republican Party Platforms about language used in the most recent presidential election. If you are interested in seeing exactly how the data was collected, you are welcome to reference [this page, which links to the MarkDown](https://nataliermcastro.github.io/projects/2025/02/13/climate-data-cleaning.html) version of the notebook or the [GitHub repository](https://github.com/NatalieRMCastro/climate-policy/blob/main/0.%20Data%20Collection%20-%20for%20website.ipynb) where you can download the IPYNB file. To reference how the data was cleaned, the HTML version of [the notebook is provided here](https://nataliermcastro.github.io/projects/2025/02/14/political-stances-cleaning.html) and a [downloadable version here](https://github.com/NatalieRMCastro/climate-policy/blob/main/1.%20Data%20Cleaning.ipynb). 

The clean data is stored in this [Hugging Face Collection](https://huggingface.co/collections/nataliecastro/climate-policy-bills-67afd0eaa0c3f328d4b00136), which was created for this project. The core method used to generate the DataFrames were both Count Vectorizer or TF-IDF vectorizer. These methods were then duplicated for a Porter Stemmed version of the text, a Lemmatized version of the text, and the cleaned but multiple word forms version. This resulted in a total of 6 different dataframes for each source. The labels used in the data vary based on the source. In combination the labels are 'source type' (news, media, or policy), 'political party', 'publisher' (either Sponsor State or News Publisher), 'bill type' (where the policy originated from), and 'commitee'. 

**Table of Contents**
- [NewsAPI](#NewsAPI)
- [Congress API + Web Scraping](#CongressAPI)
- [Party Platform Declarations](#PPD)
  
---

 <a id="NewsAPI"></a>
### NewsAPI
The [NewsAPI](https://newsapi.org/) is an easy to access API that provides key word queries to search for data in their repository of millions of articles. To understand partisan polarization in regard to climate change this is especially valuable, as many studies have begun to quantify the partisan lean of news websites. The keywords “climate change” and either “republican” or “democrat” were used to search NewsAPI. The URL query used to connect to the NewsAPI was created through a base url and url post system. The posted url is:

``` python
url_post = {'apiKey':api_key,
            'source':'everything',
            'q':'democrat+climate', ## this iteration will be looking at democrat referencing articles
            'language':'en', ## selecting English as the data
            'sortBy':'popularity', ## used to generate a popularity label
}
```

Another version of the POST URL was generated for Republican and Climate News sources. Returned from the query was the source, author, title of the article, description, url, and published date. A total of 1,559 news articles were found that have the keywords “Republican” and “climate change” and a total of 671 news articles were found for “Democrat” – this totals to 2,230 total articles collected. 

 <a id="NA_Raw_Data"></a>
#### Raw Data
The raw data from NewsAPI is relatively clean – it had minimal aggregation from the NewsAPI curators, making this task relatively smooth! An example of a raw title and description are 
> “The Trump-Newsom Fight Over an Alleged 'Water Restoration Declaration,' Explained – Trump claimed Newsom's refusal to sign the document led to a water shortage during the Los Angeles fires. But there's more to the story”

The cleaned version is 
> “the trump newsom fight over an alleged water restoration declaration explained trump claimed newsom s refusal to sign the document led to a water shortage during the los angeles fires but there s more to the story”.

While this text may look more challenging for the human eye to read, it becomes much easier for the computer to read. The title and description were concatenated to capture more meaning from such short phrases. Considering both the title and description to better understand partisan views are important. Many people often skim the results page or headlines, but never really dive into an article at the same rate. This results in more biased language used in these headlines because they are attempting to get readers to engage. 

<a id="NA_Clean_Data"></a>
#### Clean Data
The final cleaned version from the article titles and descriptions resulted in a total vocabulary of about 2,000 words. This can be expected because the descriptions of the texts are rather short, and in addition, the topics are foucsed. Reported in Table 1 are the different vocabulary sizes for each cleaning technique. 

| <span style="color:black; background-color:transparent; font-size:16px;">__Cleaning Technique__</span> | <span style="color:black; background-color:transparent; font-size:16px;">__Vectorizer Type__</span> | <span style="color:black; background-color:transparent; font-size:16px;">__Vocabulary Size__</span> |
| --- | --- | --- |
|Cleaned Text, No Processing|Count Vectorizer|2,504|
|Porter Stem|Count Vectorizer|2,357|
|Lemmatization|Count Vectorizer|2,119|
|Cleaned Text, No Processing|TF-IDF Vectorizer|2,504|
|Porter Stem|TF-IDF Vectorizer|2,357|
|Lemmatization|TF-IDF Vectorizer|2,119|

Again, the source of the News Headlines and Descriptions came from an API, so it can be excpected that there is not a lot of 'messy' data which comes from this process. The main source of variety in vocabulary size originates from the techniques used to Stem and Lemmatize the words, as they are looking for different types of wordforms from the available morphemes. 


 <a id="CongressAPI"></a>
### Congess.gov API + Web Scraping 
The [Congress.Gov API]( https://github.com/LibraryOfCongress/api.congress.gov) is a publicly available tool that can assist anyone who is interested in collecting government data. [Congress.gov]( https://www.congress.gov/), serves as a repository for the government to archive any Bill or Policy proceedings since the 1970s. To best iterative capture policy about climate change, I first used the Congress.gov front end to identify information about the bills and download my search results ([stored here]( https://github.com/NatalieRMCastro/climate-policy/tree/main/data/raw)). However, this data is not comprehensive and only provides information about the congress it was passed under, its latest action, the action note, bill number, origin chamber, title, and URL. Functions were created to parse through the information to generate labels for Sponsor Affiliation (Republican or Democrat), Sponsor State,  and then a built URL to pass into the congress API. 

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
Using the _url_builder_ function, it was able to iterate through the DataFrame, build a URL and then using  *xml_link_collector* the built URL would be passed in, called to the Congress.Gov API, and the XML URL (generated from parsing through the API response) would be the final output.The motivation in collecting the text from the policy is for two primary reasons. First, to explore what language is used and potential similarities between Republican and Democrat Sponsored policies. Second, to better understand the similarities or differences between policy and news headlines on an intensely politicized issue. 

To collect the bill text, an individual web-scraping call was made to the XML URL found with the *xml_link_collector*. Throughout this process, it became clear that many of the older bills did not have a digitized version of the bill available, thus making the sample size smaller. A total of 3,262 policy documents were collected using the Congress.Gov API, or 40% of the entire climate related bills introduced at the federal level.

 <a id="C_Raw_Data"></a>
#### Raw Text
As noted above, the raw texts comes from [Congress's XML archive](https://www.congress.gov/119/bills/hr375/BILLS-119hr375rfs.xml) of freshly presented and historical bills about climate change. This archive is structured in an expected and consistent way so retrieving data from it can be easily iterated on. Text was collected from the 'body' tag, and then stored alongside of its other features such as Commitee, Sponsor State, or Congress Number. Policy documents represent concern from political entities about something (as noted earlier, this may or may not be about climate change). Thus, the archived text uses similar language and a similar tone because if is written as a formal policy document. 

 <section>
	 <p><span class="image left"><img src="/assets/images/xml page.png" alt="" /></span> The image to the left is a screen capture from the API URL (possible to be viewed through the Congress XML link above), showing the individual webpages that were scrpaed. As illustrated, the text is also relatively clean. There are no advertisements nor inconsistencies in the layouts from one policy to the next. </p>
		
</section>
The text from each webpage was collected and cleaned using the *text cleaner* function, described below. After applying a cleaning iteratively to the documents the texts are transformed into something that is machine readable and easy to model in subsequent analysis. The *text_cleaner* function is built on [RegEx](https://en.wikipedia.org/wiki/Regular_expression), which can identify digits (filtered out first), and alphabetical characters (kept). String properties in Python allow for lowering of the text and stripping the unneccessary white space characters generated during RegEx cleaning. The cleaned bills were stored in the same dataframe for consistency.

 <section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/raw bill text.png" alt=""  /></span> 
			</div>
		</div>
	</div>
</section>

The raw XML is pictured above. This was captured by just looking at the text output from the *requests.text()* feature in the Pyton Library. The raw text from the website preserved the hierarchial bulleting system that is traditional in bills. There are not a lot of coherent sentences present in the text because of this. However, the jargon used to define and operationalize climate change by different political parties is most pertient to the reserach task at hand, making policy documents an important addition to the data.

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/Bill - Raw Text.png " alt=""  /></span> 
			</div>
		</div>
	</div>
</section>

<a id="C_Clean_Data"></a>
#### Clean Data
Cleaning the text will not remove any of the meaningful words, and in fact, TF-IDF allows for vectorization to uplift the significance of words that carry important semantic meaning. Across all texts the *text_cleaner* was used. It filters for different kinds of characters and outputs a string with the freshly cleaned text. 

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

This processing function resulted in some, but minimal changes. As noted with the NewsAPI, the structure of the text documents in their XML form is relatively clean, thus the majority of cleaning is similarizing word stems and ensuring there are no stopwords.

| <span style="color:black; background-color:transparent; font-size:16px;">__Cleaning Technique__</span> | <span style="color:black; background-color:transparent; font-size:16px;">__Vectorizer Type__</span> | <span style="color:black; background-color:transparent; font-size:16px;">__Vocabulary Size__</span> |
| --- | --- | --- |
|Cleaned Text, No Processing|Count Vectorizer|21,700|
|Porter Stem|Count Vectorizer|15,490|
|Lemmatization|Count Vectorizer|15,490|
|Cleaned Text, No Processing|TF-IDF Vectorizer|21,700|
|Porter Stem|TF-IDF Vectorizer|10,001|
|Lemmatization|TF-IDF Vectorizer|15,490|

The raw bill text vocabulary had the most prevalent words of "United State", "Administrator", and "section". The other words refer to the composition of the documents, like amend or inserting. A similar distribution of words is seen in the clean bill text, however "section", "paragraph", and "subsection" seem to be the most prevalent. This could be due to the unigram model employed to tokenize the data. The total clean vocabulary, of single words, is 21,700 words. The TF-IDF vectorizer with Porter Stemming reduced this over half to a total of 10,001 words. This may be a result of multiple iterations of the same action, such as viewed, view, or viewing. This data had the largest shift in vocabulary after text pre-processing techniques were applied.

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/Bill - Clean Text.png " alt=""  /></span> 
			</div>
		</div>
	</div>
</section>




<a id="PPD"></a>
### Party Platform Declarations 
Next, two forms of supplementary media were collected – the GOP and DNC Party Platform for the 2024 election. This will serve as an anchor for the analysis to understand the kinds of public facing language each party uses. PDFs were downloaded the PDFs from the party’s respective websites, but will not be linking them here due to copyright concerns. Using the Python Library *pypdf*, the documents were converted into a text file and ‘read’ through each page using the following code:

```python
''' EXTRACTING TEXT FROM REPUBLICAN DOCUMENT'''
republican_party_platform = []

for page in range(0,len(rep_pdf.pages)):
    current_page = rep_pdf.pages[page]
    current_text = current_page.extract_text()
    republican_party_platform.append(current_text)
```
The text was then joined together and coerced into a new text file with the basic file write capabilities provided by Python. It should be noted that the Democrat Party Platform was much longer than that of the Republican Party at 92 pages (in comparison to 28). It is interesting, but not relevant to note that the Democrat PDF was rendered in Google Docs, but the Republican’s was rendered on a Mac Computer Adobe Acrobat Version, last updated in July. 

<a id="PPD_Raw_Data"></a>
#### Raw Data

<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/DNC Party Platform - Raw Text.png" class="image fit thumb"><img src="/assets/images/DNC Party Platform - Raw Text.png" alt="" /></a>
			<h3>DNC Party Platform - Raw Text</h3>
			<p>This Word Cloud shows the raw DNC Party Platform document for the 2024 election cycle. The largest words are "President Biden", "Federal", and "American"</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/GOP Party Platform - Raw Text.png" class="image fit thumb"><img src="/assets/images/GOP Party Platform - Raw Text.png" alt="" /></a>
			<h3>GOP Party Platform - Raw Text</h3>
			<p>This Word CLoud shows the raw GOP Party Platform document for the 2024 election cycle. The largest words are "will", "American", and "Republican".</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/DNC Party Platform - Clean Text.png" class="image fit thumb"><img src="/assets/images/DNC Party Platform - Clean Text.png" alt="" /></a>
			<h3>DNC Party Platform - Clean Text</h3>
			<p>Now, the largest words are "american", "president", and "biden".</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/GOP Party Platform - Clean Text.png" class="image fit thumb"><img src="/assets/images/GOP Party Platform - Clean Text.png" alt="" /></a>
			<h3>GOP Party Platform - Clean Text</h3>
			<p>After cleaning, the text looks a little different with the largest words as "republican","american", and "border".</p>
		</article>
	</div>
</section>

 #### Clean Data 
 
Table 3 demonstrates the shift in vocabulary for the different types of Stemming, Lemming, and Vecotorizing Techniques. The Party Platforms have the smallest vocabulary out of the other two sources, and thus the lowest variation between the types. Between the Porter Stem and the Lemmatization, regardless of vecotorizer was a difference of about 400 words. 

| <span style="color:black; background-color:transparent; font-size:16px;">__Cleaning Technique__</span> | <span style="color:black; background-color:transparent; font-size:16px;">__Vectorizer Type__</span> | <span style="color:black; background-color:transparent; font-size:16px;">__Vocabulary Size__</span> |
| --- | --- | --- |
|Cleaned Text, No Processing|Count Vectorizer|943|
|Porter Stem|Count Vectorizer|890|
|Lemmatization|Count Vectorizer|1,376|
|Cleaned Text, No Processing|TF-IDF Vectorizer|943|
|Porter Stem|TF-IDF Vectorizer|890|
|Lemmatization|TF-IDF Vectorizer|1,376|


---  
### Bibliography:

Bird, S., Loper, E., & Kafe, E. (n.d.). Natural Language Toolkit: WordNet stemmer interface. Retrieved February 14, 2025, from https://www.nltk.org/_modules/nltk/stem/wordnet.html#WordNetLemmatizer

Porter, M. F. (2001, October). Snowball: A language for stemming algorithms. Snowball.Tartarus.Org. http://snowball.tartarus.org/texts/introduction.html

scikit-learn. (2024a). scikit-learn/sklearn/feature_extraction/text.py—Count vectorizer [Computer software]. https://github.com/scikit-learn/scikit-learn/blob/6a0838c416c7c2a6ee7fe4562cd34ae133674b2e/sklearn/feature_extraction/text.py

scikit-learn. (2024b). scikit-learn/sklearn/feature_extraction/text.py—Tf-idf vectorizer [Computer software]. https://github.com/scikit-learn/scikit-learn/blob/6a0838c41/sklearn/feature_extraction/text.py#L1734
