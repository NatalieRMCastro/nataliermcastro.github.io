---
layout: post
title: "Political Stances: Clustering Partisian Climate Data"
categories: projects
published: true
in_feed: false
---

**Table of Contents**
- [Method](#Method)
- [Cluster Optimization](#Cluster Optimization)
- [Findings](#Finding1)
- [x](#Finding2)
  
---

 <a id="Method"></a>
### Method

To cluster any form of natural language data, the documents must be vectorized. Doing so allows for the computer to ‘read’ the language, but instead of with eyes it is computation. Document-Term Matrices were created for each data form. The first, is of Climate Bill data. This was collected from the United States Library of Congress and contains all bills with the topic of ‘climate’ that was introduced in either chamber. The total vocabulary after removing for stop words, numerical characters, and labels was over 10,000 different words. This data was collected using XML web scraping and the API provided by the Library of Congress. Next, news headlines were collected about ‘climate change’ and then labeled in accordance with the Partisan party mentioned in the news headline. This includes the news headline descriptions, but not the entire article. This data was collected using the NewsData API. Finally, each party platform for the 2024 United States Election was utilized. To read more about the data collection process, reference [Data Collection]([LINK HERE](https://nataliermcastro.github.io/projects/2025/01/14/political-stances-data.html))

Specifically for K-Means clustering, the data was normalized before by using TF-IDF. TF-IDF was developed by Luhn and Sparack Jones and is used for information retrieval [1, 2]. It informs what terms are uniquely relevant to providing meaning through the documents. This method calculates the term frequency and then the inverse document frequency. This provides a relative amount of how important the word is to the entire corpus and in relation to the document. For example, the word ‘climate’ can be assumed to be highly frequent across all documents, this term will be down weighted. This normalization process helps to situate the documents within a contextual space that is bounded in frequency.

Creating Document Term Matrices allow for the vectorization. Each document counts as a vector, and each row counts as an index in the vector. This allows for vector similarity to be calculated and to then cluster the data in the high dimensional space. K-Means clustering takes unlabeled data. This means that the words are clustered based on their vectors alone, not based on prior metadata. Using Lloyd’s Algorithm the vectors dimensionality is reduced to then be able to situate and cluster the indices in high dimensional space. Further dimensionality reduction is used conducted using t – Stochastic Neighbor Embedding, t-SNE, to the visualize the clusters in something we are familiar with: two dimensions. 

To generate the clusters the Python library Sci-Kit Learn was utilized. The parameters afforded through the library are the number of iterations, number of clusters, and random state. The same random state was used across the model to set the model in a similar ‘mode’ when it is clustering the visualizations. Other than setting a defined seed, the random state does not have much of an effect on how the clusters are generated or represented in two-dimensional space. For all tests and modeling the random state of 811 was selected. No particular reason, I just thought it would be a nice number. 

 <a id="Cluster Optimization"></a>
### Cluster Optimization
To identify the most appropriate clusters for the data, a recursive algorithim was created to test clustering methods at different step sizes. The purpose of this function was to systematically identify the best fitting cluster for each dataset on the same random seed and parameters. 


``` python
'''SILHOUETTE CLUSTERING'''
## Defining the function to take the data to cluster, a consistent random state to use across testing, the desired number of clusters, and the number of iterations
def shadow_reporter(data,random_state,num_clusters,num_iterations):

    ## Instantiating the Model with the desired parameters
    km = KMeans()
    km.set_params(random_state=random_state)
    km.set_params(n_clusters = num_clusters)
    km.set_params(n_init = num_iterations)
    ## Fitting the data
    km.fit(data)
    ## Assessing the model for each iteration on performance metrics.
    shadow = metrics.silhouette_score(data, km.labels_,sample_size = len(data))
    return (shadow)


''' SHADOW TESTER '''

## This function takes a range and applies the Shadow Reporter based on the current parameters for the iteration. 
def shadow_tester(data_name,data,random_state,num_iterations, step_size,cluster_start,cluster_end):
    shadow_scores = []

    ## Iterating through the parameters with a certain step size
    for clust in tqdm(range(cluster_start,cluster_end,step_size),desc='🧺🐜... clustering',leave=True):
        clust_score = shadow_reporter(data,random_state,clust,num_iterations)
        shadow_scores.append({'Clusters':clust,"Silhouette":clust_score})
    title = f"Silhouette Clustering Testing:\n{data_name}\n\tIterations: {num_iterations}\n\tCluster Range: {cluster_start} - {cluster_end}\n\tStep Size: {step_size}"
    df = pd.DataFrame(shadow_scores)
    plot = sb.lineplot(data=df,x='Clusters',y='Silhouette')

    ## Printing the output to judge best clustering fit.
    print (title)
    print ("-------------------------------------------------------------------------------")
    plt.show()
    print ("-------------------------------------------------------------------------------")
    print (df)
    return (shadow_scores)
```

<div style="display: flex; gap: 20px; align-items: flex-start;">

  <!-- Left Column: Image Section -->
  <section style="flex: 1;">
    <div class="box alt">
      <div class="row gtr-50 gtr-uniform">
        <div class="col-12">
          <span class="image fit">
            <img src="/assets/images/Systemic Clustering Testing.png" alt="An example output from the Cluster Testing Function" />
          </span>
          <figcaption>Example Function Output from Systematic Testing</figcaption>
        </div>
      </div>
    </div>
  </section>

  <!-- Right Column: Table -->
  <div style="flex: 1;">
    <table style="width: 100%;">
      <thead>
        <tr>
          <th style="color:black; background-color:transparent; font-size:16px;">Cleaning Technique</th>
          <th style="color:black; background-color:transparent; font-size:16px;">Vectorizer Type</th>
          <th style="color:black; background-color:transparent; font-size:16px;">Vocabulary Size</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Cleaned Text, No Processing</td>
          <td>Count Vectorizer</td>
          <td>2,504</td>
        </tr>
        <tr>
          <td>Porter Stem</td>
          <td>Count Vectorizer</td>
          <td>2,357</td>
        </tr>
        <tr>
          <td>Lemmatization</td>
          <td>Count Vectorizer</td>
          <td>2,119</td>
        </tr>
        <tr>
          <td>Cleaned Text, No Processing</td>
          <td>TF-IDF Vectorizer</td>
          <td>2,504</td>
        </tr>
        <tr>
          <td>Porter Stem</td>
          <td>TF-IDF Vectorizer</td>
          <td>2,357</td>
        </tr>
        <tr>
          <td>Lemmatization</td>
          <td>TF-IDF Vectorizer</td>
          <td>2,119</td>
        </tr>
      </tbody>
    </table>
  </div>

</div>


<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/Systemic Clustering Testing.png" alt="An example output from the Cluster Testing Function"  /></span> 
        <figcaption>Example Function Output from Systematic Testing</figcaption>
			</div>
		</div>
	</div>
</section>

| <span style="color:black; background-color:transparent; font-size:16px;">__Cleaning Technique__</span> | <span style="color:black; background-color:transparent; font-size:16px;">__Vectorizer Type__</span> | <span style="color:black; background-color:transparent; font-size:16px;">__Vocabulary Size__</span> |
| --- | --- | --- |
|Cleaned Text, No Processing|Count Vectorizer|2,504|
|Porter Stem|Count Vectorizer|2,357|
|Lemmatization|Count Vectorizer|2,119|
|Cleaned Text, No Processing|TF-IDF Vectorizer|2,504|
|Porter Stem|TF-IDF Vectorizer|2,357|
|Lemmatization|TF-IDF Vectorizer|2,119|

This function was applied to each type of data, the News Corpus, the Climate Bills, and the Party Platform. Systematically testing the number of clusters with the best fit will generate more accurate models. Reported below are the parameters set for each model. 

 <a id="Finding1"></a>
### Findings

