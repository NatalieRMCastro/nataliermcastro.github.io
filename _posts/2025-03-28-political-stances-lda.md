---
layout: post
title: "Political Stances: Latent Dirichlet Allocation"
categories: projects
published: true
in_feed: false
---

introduction text to lda

**Table of Contents**
- [Data](#data)
- [Method: LDA](#method_lda)
- [Cluster Optimization](#cluster_optimization)
- [LDA Findings](#findings)

---
 <a id="data"></a>
### Data Preparation: Unlabed Document Term Matricies

Three different LDA models will be generated during this analysis, one for news data, one for proposed climate bills, and one for the Party Platform. To do so, each datasource requires an unlabeled document term matrix. This means that every row represents a document in the corpus, and then the columns are the cleaned and meaningful words which consist the texts. More information about how the data was cleaned and collected may be found linked here.

 <a id="method_lda"></a>
### Method: Latent Dirichlet Allocation 

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
