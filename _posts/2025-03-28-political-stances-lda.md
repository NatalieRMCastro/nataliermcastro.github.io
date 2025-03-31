---
layout: post
title: "Political Stances: Latent Dirichlet Allocation"
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

Latent Dirichlet Allocation (LDA) is a form of topic modeling. It is a "generative probabilistic model for collections of discrete data such as text corpora" [1] which sorts documents. A topic is developed over the distirbution of topic probabilities to consider multiple organizations of the data. LDA supports an exploration of the topics in multiple different representations of the text. 

Topic Modeling is used in this instance because of the size of the corpus and its span over time. In addition, it can provide insight into different groupings of data that may not have been previsouly considered. Topic modeling is applied to this data provide an overall understanding of the contents in the corpus. 

**Table of Contents**
- [Data](#data)
- [Method: LDA](#method_lda)
- [Cluster Optimization](#allocation_optimization)
- [LDA Findings](#findings)
- [LDA Clusters: Interactive Analysis](#interactive)

---
 <a id="data"></a>
### Data Preparation: Unlabed Document Term Matricies

Three different LDA models will be generated during this analysis, one for news data, one for proposed climate bills, and one for the Party Platform. To do so, each datasource requires an unlabeled document term matrix. This means that every row represents a document in the corpus, and then the columns are the cleaned and meaningful words which consist the texts. More information about how the data was cleaned and collected may be found [linked here](https://nataliermcastro.github.io/projects/2025/01/14/political-stances-data.html). 

The Count Vectorizer was used here because of its ability to show the feature names after the fact of clustering and are able to show larger distances between the data. After reading in the data from the CSV files, the label columns are stored and any NA values are filled with zeros. Labels are removed from this data because the LDA model only takes numerical data, so the prevalence of the label would cause the model not to work. Illustrated below is the lemmatized and CountVectorizer processed version of the Party Platform. This data may be downloaded, alongside others used in this analysis, at [my HuggingFace Collection](https://huggingface.co/datasets/nataliecastro/climate-news-countvectorizer-dtm). 

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/lda data example.png" alt="A Document Term Matrix with the words in the party platform. The values in the cells are whole numbers."  /></span> 
			</div>
		</div>
	</div>
</section>



 <a id="method_lda"></a>
### Method: Latent Dirichlet Allocation 
<section>
    <div class="row">
        <div class="col-6 col-12-small">
            <ul class="actions" style="display: flex; gap: 10px; list-style: none; padding: 0;">
                <li><a href="https://nataliermcastro.github.io/projects/2025/03/30/political-stances-lda-code.html" class="button fit small">View Code</a></li>
		<li><a href="https://github.com/NatalieRMCastro/climate-policy/blob/main/5.%20Latent%20Dirichlet%20Allocation.ipynb" class="button fit small">Visit GitHub Repository</a></li>
            </ul>
        </div>
    </div> 
</section> 

To generate LDA clusters, the data was imported and then processed as described above. Once a topic model is instantiated, it may be customized for the specifc parameters. 

```python
 lda_model = LatentDirichletAllocation(n_components=topic_number,max_iter=50, learning_method='online')

fit_model = lda_model.fit_transform(dataset)
```

This fitting process will be completed for each of the datasets, so one for the News Corpus, one for the Bills Corpus, and a final for the party platform. Each of these will then be explored and visualized in depth. The instantiation process is fairly straightforward, however, to optimize the use of the LDA algorithim, recursive testing must be applied to each dataset.

 <a id="#allocation_optimization"></a>
### Allocation Optimization

A function was developed to systematically test different allocation numbers, another function was developed to save the iterations of the topic model across testing and recall it later. 

The *lda_tester* takes a number of topics, so four or eight, the dataset that is in Document Term Matrix form, the vectorizer, number of words for visualization, and a name of the dataset. This function takes a lot, so it will be explained through the steps of the function. First, it instantiates the model with an online learning method (the batch calculations are different) and passes in the desired number of topics. 

The LDA model is then fit with the provided dataset. Using the *save_topics* function (described next) takes the LDA model, vectorizer, and top words to vectorize. These are then both returned for subsequent analysis. 

```python
def lda_tester(topic_number,dataset,vectorizer,top_n,dataset_name):
    ## Instantiating a model:
    
    lda_model = LatentDirichletAllocation(n_components=topic_number,max_iter=50, learning_method='online')
    
    ## Fitting the model:
    fit_model = lda_model.fit_transform(dataset)
    
    ## Storing the contents of the topic model:
    topic_contents = save_topics(lda_model, vectorizer,top_n)
    return (lda_model,topic_contents)
```
The *save_topics* then takes the LDA model and the vectorizer for the dataset. The model component are then iterated through and the top words are extracted and saved. This is then stored in a dictionary format, with the the IDX value, or index, and then a list of the top words for the particular topic.

```python
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
```

To then wrap it all together, a recursive function was developed to just pass in a range of the dataset, the data, its vectorizer, and then a string of the dataset name. The function iterates through the entire range and applies the *lda_tester* and passess the model and the topics into another dictionary format. 

For subsequent analysis, this was particularly helpful because the topic model was able to be called through the dictionary structure. Different amounts of the allocation were able to be visualized and explored because of this. 

```python
def lda_modeler(start_topics, end_topics,dataset,vectorizer,top_n,dataset_name):
    
    dataset_topics = {}
    
    for topic_set in tqdm(range (start_topics,end_topics,2),desc='🐜🐛... inching through data',leave=False,):
        lda_model, lda_topics = lda_tester(topic_set,dataset,vectorizer,top_n,dataset_name)
        
        dataset_topics[topic_set] = {"LDA MODEL":lda_model,"LDA TOPICS": lda_topics}
        
    return (dataset_topics)
```

 <a id="#findings"></a>
### LDA Findings

<div style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
  <iframe src="/assets/images/lda_climate_bills.html" width="100%" height="650px" allowfullscreen></iframe>
</div>

<section>
	<br>
</section>
For both Party Platforms and News Clustering partisan affiliated words are colored in accoradance to the traditional colors associated with political parties.

<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/LDA  - climate news.png" class="image fit thumb"><img src="/assets/images/LDA  - climate news.png" alt="" /></a>
			<h3>
				LDA Clustering for Climate News</h3>
			<p>figure explanation. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/LDA - party platform.png" class="image fit thumb"><img src="/assets/images/LDA - party platform.png" alt="" /></a>
			<h3>LDA Clustering for Climate News</h3>
			<p>figure explanation</p>
		</article>
	</div>
</section>

 <a id="#interactive"></a>
### LDA Clustering: Interactive Analysis

##### LDA Clustering: Climate Related News Topics
<iframe src="/assets/images/LDA Interactive Topics - News.html" width="100%" height="600px" title="News Data" scrolling="no" style="border: none; overflow: hidden;"></iframe>

##### LDA Clustering: Proposed Climate Bills
<iframe src="/assets/images/LDA Interactive Topics - Bills.html" width="100%" height="600px" title="Bills Data" scrolling="no" style="border: none; overflow: hidden;"></iframe>


##### LDA Clustering: 2024 Party Platforms
<iframe src="/assets/images/LDA Interactive Topics - Party.html" width="100%" height="600px" title="Party Data" scrolling="no" style="border: none; overflow: hidden;"></iframe>

<iframe src="/assets/images/LDA Interactive Topics - Party.html" width="100%" height="600px" style="border: none; overflow: hidden; max-width: 100%; max-height: 100%;" id="ldavis_el768431687202567204275348056"></iframe>

---
## Bibliography:
[1] Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. J. Mach. Learn. Res., 3(null), 993–1022.


