---
layout: post
title: "Political Stances: Clustering Partisian Climate Data"
categories: projects
published: true
in_feed: false
---

**Method**
To cluster any form of natural language data, the documents must be vectorized. Doing so allows for the computer to ‘read’ the language, but instead of with eyes it is computation. Document-Term Matrices were created for each data form. The first, is of Climate Bill data. This was collected from the United States Library of Congress and contains all bills with the topic of ‘climate’ that was introduced in either chamber. The total vocabulary after removing for stop words, numerical characters, and labels was over 10,000 different words. This data was collected using XML web scraping and the API provided by the Library of Congress. Next, news headlines were collected about ‘climate change’ and then labeled in accordance with the Partisan party mentioned in the news headline. This includes the news headline descriptions, but not the entire article. This data was collected using the NewsData API. Finally, each party platform for the 2024 United States Election was utilized. To read more about the data collection process, reference [Data Collection]([LINK HERE](https://nataliermcastro.github.io/projects/2025/01/14/political-stances-data.html))

Specifically for K-Means clustering, the data was normalized before by using TF-IDF. TF-IDF was developed by Luhn and Sparack Jones and is used for information retrieval [1, 2]. It informs what terms are uniquely relevant to providing meaning through the documents. This method calculates the term frequency and then the inverse document frequency. This provides a relative amount of how important the word is to the entire corpus and in relation to the document. For example, the word ‘climate’ can be assumed to be highly frequent across all documents, this term will be down weighted. This normalization process helps to situate the documents within a contextual space that is bounded in frequency.

Creating Document Term Matrices allow for the vectorization. Each document counts as a vector, and each row counts as an index in the vector. This allows for vector similarity to be calculated and to then cluster the data in the high dimensional space. K-Means clustering takes unlabeled data. This means that the words are clustered based on their vectors alone, not based on prior metadata. Using Lloyd’s Algorithm the vectors dimensionality is reduced to then be able to situate and cluster the indices in high dimensional space. Further dimensionality reduction is used conducted using t – Stochastic Neighbor Embedding, t-SNE, to the visualize the clusters in something we are familiar with: two dimensions. 

To generate the clusters the Python library Sci-Kit Learn was utilized. The parameters afforded through the library are the number of iterations, number of clusters, and random state. The same random state was used across the model to set the model in a similar ‘mode’ when it is clustering the visualizations. Other than setting a defined seed, the random state does not have much of an effect on how the clusters are generated or represented in two-dimensional space. For all tests and modeling the random state of 811 was selected. No particular reason, I just thought it would be a nice number. 

