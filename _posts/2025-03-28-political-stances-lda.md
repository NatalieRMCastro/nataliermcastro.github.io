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

introduction text to lda

**Table of Contents**
- [Data](#data)
- [Method: LDA](#method_lda)
- [Cluster Optimization](#cluster_optimization)
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

```python
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
```

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

 <a id="#cluster_optimization"></a>
### Cluster Optimization

 <a id="#findings"></a>
### LDA Findings

<div style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
  <iframe src="/assets/images/lda_climate_bills.html" width="100%" height="700px" style="border: 2px solid gray;" allowfullscreen></iframe>
</div>

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

<iframe src="/assets/images/LDA Interactive Topics - Party.html" width="100%" height="600px" style="border: none; overflow: hidden; object-fit: contain;" id="ldavis_el768431687202567204275348056"></iframe>

