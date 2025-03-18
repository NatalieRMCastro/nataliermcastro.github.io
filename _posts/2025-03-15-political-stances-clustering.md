---
layout: post
title: "Political Stances: Clustering Partisian Climate Data"
categories: projects
published: true
in_feed: false
---
Clustering is a text mining method which uses the frequencies within a document to assign meaning. To answer the overarching question of ideological differences divided among party lines clustering methods will be used to both identify differences or subsequent similaries from the differences in the available data. Methods used in this exploration are K-Means Clustering (completed in Python), *hclust* informed by Cosine Similarity for hierarchical clustering (completed in R), and Principal Component Analysis for 3-D spatial representation (completed in Python). This combination of methods are applied to provide a holistic understanding of the data provided in addition to different embedding measures to interpret the data differently. 

To answer the research questions, the data will be clustered separately. One model will be used to inform the differences in partisian conceptializations of climate policy and the next will be used to inform the differences in media coverage about contentious partisian differences. In this analysis, the Party Platform will not be clustered because there are only two items in that dataset. 

**Table of Contents**
- [Method](#Method)
- [Cluster Optimization](#Cluster Optimization)
- [K-Means Findings](#kmeans)
- [HClust and Cosine Similarity](#hclust)
- [PCA Findings](#pca)
- [x](#Finding2)
  
---

 <a id="Method"></a>
### Method

To cluster any form of natural language data, the documents must be vectorized. Doing so allows for the computer to ‘read’ the language, but instead of with eyes it is computation. Document-Term Matrices were created for each data form. The first, is of Climate Bill data. This was collected from the United States Library of Congress and contains all bills with the topic of ‘climate’ that was introduced in either chamber. The total vocabulary after removing for stop words, numerical characters, and labels was over 10,000 different words. This data was collected using XML web scraping and the API provided by the Library of Congress. Next, news headlines were collected about ‘climate change’ and then labeled in accordance with the Partisan party mentioned in the news headline. This includes the news headline descriptions, but not the entire article. This data was collected using the NewsData API. Finally, each party platform for the 2024 United States Election was utilized. To read more about the data collection process, reference [Data Collection](https://nataliermcastro.github.io/projects/2025/01/14/political-stances-data.html)

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

This function was applied to each type of data, the News Corpus, the Climate Bills, and the Party Platform. Systematically testing the number of clusters with the best fit will generate more accurate models. Reported below are the parameters set for each model. As noted previously, the random state instantiates a shared vector, so the results may be replicated if so desired.


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
	  K-Means Clustering Parameters for Corpora
    <table style="width: 100%;">
      <thead>
        <tr>
          <th style="color:black; background-color:transparent; font-size:12px;">Data Source</th>
          <th style="color:black; background-color:transparent; font-size:12px;">Number of Clusters</th>
          <th style="color:black; background-color:transparent; font-size:12px;">Number of Iterations</th>
	<th style="color:black; background-color:transparent; font-size:12px;">Silhouette Score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>News Corpus</td>
          <td>4</td>
          <td>1,000</td>
	<td>0.011</td>
        </tr>
	<tr>
          <td>News Corpus</td>
          <td>20</td>
          <td>1,000</td>
	<td>0.027</td>
        </tr>
	<tr>
          <td>Climate Policy</td>
          <td>100</td>
          <td>0.24</td>
        </tr> 
      </tbody>
    </table>
  </div>

</div>

 <a id="kmeans"></a>
### K-Means: Topic Representations of Introduced Climate Bills and Media Concerns


<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans Climate News.png" class="image fit thumb"><img src="/assets/images/kmeans Climate News.png" alt="" /></a>
			<h3>Distribution of Climate News Clustered Topics</h3>
			<p>text</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans Climate Clustering Scatter.png" class="image fit thumb"><img src="/assets/images/kmeans Climate Clustering Scatter.png" alt="" /></a>
			<h3>Climate News: K-Means Clustering</h3>
			<p>text</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans Climate Cluster Distribution.png" class="image fit thumb"><img src="/assets/images/kmeans Climate Cluster Distribution.png" alt="" /></a>
			<h3>Distribution of Introduced Cliamte Policy Clustered Topics</h3>
			<p>text</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans Climate Clustering Small Scatter.png" class="image fit thumb"><img src="/assets/images/kmeans Climate Clustering Small Scatter.png" alt="" /></a>
			<h3>Introduced Climate Policy: K-Means Clustering</h3>
			<p>text</p>
		</article>
	</div>
</section>

<a id="hclust"></a>
### H-Clust and Cosine Similarity Findings: 


<a id="pca"></a>
### PCA Findings: 

