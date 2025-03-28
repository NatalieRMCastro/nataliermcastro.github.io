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
- [Method: K-Means](#method_kmeans)
- [Cluster Optimization](#Cluster Optimization)
- [Method: H-Clust](#method_hclust)
- [Method: PCA](#method_pca)
- [K-Means Findings](#kmeans)
- [HClust and Cosine Similarity Findings](#hclust)
- [PCA Findings](#pca)
  
---

 <a id="method_kmeans"></a>
### Method: K-Means

To cluster any form of natural language data, the documents must be vectorized. Doing so allows for the computer to ‘read’ the language, but instead of with eyes it is computation. Document-Term Matrices were created for each data form. The first, is of Climate Bill data. This was collected from the United States Library of Congress and contains all bills with the topic of ‘climate’ that was introduced in either chamber. The total vocabulary after removing for stop words, numerical characters, and labels was over 10,000 different words. This data was collected using XML web scraping and the API provided by the Library of Congress. Next, news headlines were collected about ‘climate change’ and then labeled in accordance with the Partisan party mentioned in the news headline. This includes the news headline descriptions, but not the entire article. This data was collected using the NewsData API. Finally, each party platform for the 2024 United States Election was utilized. To read more about the data collection process, reference [Data Collection](https://nataliermcastro.github.io/projects/2025/01/14/political-stances-data.html)

#### Data Preparation

Specifically for K-Means clustering, the data was normalized before by using TF-IDF. TF-IDF was developed by Luhn and Sparack Jones and is used for information retrieval [1, 2]. It informs what terms are uniquely relevant to providing meaning through the documents. This method calculates the term frequency and then the inverse document frequency. This provides a relative amount of how important the word is to the entire corpus and in relation to the document. For example, the word ‘climate’ can be assumed to be highly frequent across all documents, this term will be down weighted. This normalization process helps to situate the documents within a contextual space that is bounded in frequency.

Creating Document Term Matrices allow for the vectorization. Each document counts as a vector, and each row counts as an index in the vector. This allows for vector similarity to be calculated and to then cluster the data in the high dimensional space. K-Means clustering takes unlabeled data. This means that the words are clustered based on their vectors alone, not based on prior metadata. Using Lloyd’s Algorithm the vectors dimensionality is reduced to then be able to situate and cluster the indices in high dimensional space. Further dimensionality reduction is used conducted using t – Stochastic Neighbor Embedding, t-SNE, to the visualize the clusters in something we are familiar with: two dimensions. 

<section class="gallery">
	<div class="row">
 <article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans rawdata.png" class="image fit thumb"><img src="/assets/images/kmeans rawdata.png" alt="" /></a>
			<h3>Raw News Data Example</h3>
			<p>This was the raw data from news sources. It was processed using Lemming and TF-IDF, as illustrated in the News Data TF-IDF example.</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans dataprep.png" class="image fit thumb"><img src="/assets/images/kmeans dataprep.png" alt="" /></a>
			<h3>
				News Data TF-IDF Example</h3>
			<p>The Lemmed and TF-IDF version of the data was utilized in order to bound the frequency. Lemmatized data was then used to combine multiple word senses. </p>
		</article>	
	</div>
</section>

The data is available at [this Hugging Face repository](https://huggingface.co/datasets/nataliecastro/climate-news-tfidf-lemmed/blob/main/News%20Articles%20Lemmed-%20TFIDF.csv)

#### Generating K-Means Clusters

To generate the clusters the Python library Sci-Kit Learn was utilized. The parameters afforded through the library are the number of iterations, number of clusters, and random state. The same random state was used across the model to set the model in a similar ‘mode’ when it is clustering the visualizations. Other than setting a defined seed, the random state does not have much of an effect on how the clusters are generated or represented in two-dimensional space. For all tests and modeling the random state of 811 was selected. No particular reason, I just thought it would be a nice number. 

 <a id="Cluster Optimization"></a>
#### K-Means Cluster Optimization
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

 <a id="method_hclust"></a>
### Method: H-Clust


 <a id="method_pca"></a>
### Method: Principle Compontent Analysis

```python
def pca_tester(scaled_data, raw_data, components,title):
    ''' PCA FITTING '''
    ## Fitting the PCA Model to the number of components desired
    pca_model = PCA(n_components=components)
    result = pca_model.fit_transform(scaled_data)

    ## Extracting the Values to Identify Variance
    eigenvalues = pca_model.explained_variance_
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    ## Creating and Index for the Numbers of Principle Components Generated
    component_indices = np.arange(1, components + 1)

    ## Fitting them to a dataframe to plot
    data = {
        'Component': component_indices,
        'Eigen Values': list(eigenvalues),
        'Explained Variance Ratio': list(explained_variance_ratio),
        'Cumulative Variance': list(cumulative_variance)
    }

    plotting_data = pd.DataFrame.from_dict(data)

    ## Identfiying the largest Eigen Value drop
    eigenvalue_diffs = np.diff(eigenvalues) 
    max_drop_index = np.argmax(np.abs(eigenvalue_diffs))  
    max_drop_component = component_indices[max_drop_index]  

    ''' PLOTTING '''
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Eigenvalues", "Variance"])

    ## Plotting EigenValues on the left hand side
    fig.add_trace(
        go.Scatter(x=plotting_data["Component"], y=plotting_data["Eigen Values"], mode="lines+markers", name="Eigenvalues"),
        row=1, col=1
    )

    ## Highlight the largest drop
    fig.add_trace(
        go.Scatter(
            x=[max_drop_component, max_drop_component + 1],
            y=[eigenvalues[max_drop_index], eigenvalues[max_drop_index + 1]],
            mode="markers+lines",
            marker=dict(color='red', size=10),
            name="Largest Drop"
        ),
        row=1, col=1
    )

    ## Plotting Variance on the right hand side
    fig.add_trace(
        go.Scatter(x=plotting_data["Component"], y=plotting_data["Explained Variance Ratio"], mode="lines+markers", name="Explained Variance Ratio"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=plotting_data["Component"], y=plotting_data["Cumulative Variance"], mode="lines+markers", name="Cumulative Variance"),
        row=1, col=2
    )

    ## Situating Plotting Labels
    fig.update_layout(title_text=title, showlegend=True, xaxis_title="Principal Component", yaxis_title="Values")

    fig.show()
```
 

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/pca fitting.png" alt="A line graph demonstrating the elbow curve used to identify how many PCs are best."  /></span> 
			</div>
		</div>
	</div>
</section>


 <a id="kmeans"></a>
### K-Means: Topic Representations of Introduced Climate Bills and Media Concerns
K-Means represents the different clustering methods using the Euclidean distnance of the vectorized text. The more similar a document, it will be clustered closer together in the high dimensional space. First let's address how climate change represented in climate policy introduced at the federal level. Using K-Means, it was illustrated the different trends or types of climate policy. The complexity and length of the documents resulted in a generous amount of topics. During optimization, the silhouette score continued to increase as the number of clusters increased - even into the upper 200s for *k* clusters. 100 clusters was selected to visualize because it is still detailed enough to obtain a silhouette score of 0.23, however, not too detailed in an attempt to preserve interpretability. A smaller K-Means clustering set of 20 was also established for the Climate Policy Data in order to compare a similar amount of clusters for the news data.

The distribution of the clusters may be observed in *Figure: Distribution of Introduced Cliamte Policy Clustered Topics* is fairly even, with the largest cluster being about the National Environmental Protection Act (NEPA) and wildfires. This would explain the far-reaching clustering demonstrated in *Figure: Introduced Climate Policy: K-Means Clustering*. As NEPA is an organization which has ties to multiple facets of climate change and other concerns. 

<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans Climate Cluster Distribution.png" class="image fit thumb"><img src="/assets/images/kmeans Climate Cluster Distribution.png" alt="" /></a>
			<h3>
				Distribution of Introduced Cliamte Policy Clustered Topics</h3>
			<p>The largest cluster is in relation to NEPA, with over 900 bills fitting into this topic. Subsequently, the distribution of the clusters are normal, and there are a relatively even distribution amongst clusters. A few of the largest clusters are: (19) eligible, treatment, stormwater, wastewater, loans (2) climate, adopotion, global, coastal, resilience (8) permit, discharge, vessel, permitting, specification. Remaining clustes are primarily concerned with water, waste or pollution, and inequtiable impact. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans Climate Clustering Small Scatter.png" class="image fit thumb"><img src="/assets/images/kmeans Climate Clustering Small Scatter.png" alt="" /></a>
			<h3>Introduced Climate Policy: K-Means Clustering</h3>
			<p>As expected, the NEPA cluster is far reaching. It is not siloed within a particular topic such as water safety or ground pollution, thus it is similar to multiple aspects in the embedding space. Similar language used in climate bills about different topics may be identified. For example, cluster 5 is about species protection which is semantically closest to cluster 4 focusing on lake resotration. The cluster with terms about disapproval (13) is closet to NEPA and mercurcy terms.</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans Climate News.png" class="image fit thumb"><img src="/assets/images/kmeans Climate News.png" alt="" /></a>
			<h3>
				Distribution of Climate News Clustered Topics</h3>
			<p>The largest clusters are concerned with the new Trump Adminsitration. Three out of the four largest clusters focused on Donald Trump. The only exception being a cluster concerned with the Los Angeles Wildfires, something which topped news charts for days. Climate topics are not as clear in this distribution of topics. Illustrated from this iteration of K-Means clustering is the decision to withdrawl from the Paris Cliamte Agreement, the L.A. Fires, and the hurricane in North Carolina. This illustrates that the media is more concerned with disastorous and largely consequential events instead of more passive changes in regards to climate change.</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/kmeans Climate Clustering Scatter.png" class="image fit thumb"><img src="/assets/images/kmeans Climate Clustering Scatter.png" alt="" /></a>
			<h3>Climate News: K-Means Clustering</h3>
			<p>text</p>
		</article>
		
	</div>
</section>

In comparison to the introduced climate bills, the media coverage about climate change from a paritisan lens is less dense and topically interconnected. This can be inferred from *Figure Climate News: K-Means Clustering*. Unlike the NEPA cluster, there are not as many distincy forms of clutering between the topics. Interrelated topics may be identified when in reference to a particular entity like the transition between the Biden Administration and the Trump Administration (Clusters 18 and 4). The largest two clusters from the media dataset are focused on the incoming Trump Adminsitration (Clusters 6 and 9). Subsequently is a cluster about the Los Angeles Wildfires, which was extremly pertinent at the time of data collection. Spatially, this cluster is closest to Clusters 7 and 9, both of which are about Donald Trump and then a senate hearing. This suggests that there are not other media forms or current topics which were close to the magnitude of the L.A. fires. In the K-Means clustering of introduced climate bills, wildfires is not one of the 20 topics identified. 

<a id="hclust"></a>
### H-Clust and Cosine Similarity Findings: 
<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/HClust Before Word Cloud.jpeg" alt="Wordccloud with before HClust Methods - The largest words are 'cover', 'recycle', and 'list'."  /></span> 
			</div>
		</div>
	</div>
</section>

<div style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
  <iframe src="/assets/images/hclust radial small.html" width="100%" height="500px" style="border: none;" allowfullscreen></iframe>
</div>
*Radial Network of Most Similar Words*

<div style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
  <iframe src="/assets/images/hclust random parition large.html" width="100%" height="250px" style="border: 2px solid blue;" allowfullscreen></iframe>
</div>
*Dendrogram of Most Similar Words*

<section>

 	<div class="row">
		<div class="col-6 col-12-small">
			<ul class="actions stacked">
				<li><a href="https://nataliermcastro.github.io/projects/2025/03/28/hclust-vis.html" class="button fit small">View Dendrogram in Separate Page</a></li>
				<li><a href="https://nataliermcastro.github.io/projects/2025/03/28/political-stances-hclust.html" class="button fit small">View Code</a></li>
			</ul>
		</div>
<section/>

The size and scope of the data used in this anaylsis is not appropriate for HClust. Illustrated here is a small parition of the data, and is not representative of the entire vector space.  

<section>

 
</section>
<a id="pca"></a>
### PCA Findings: 

