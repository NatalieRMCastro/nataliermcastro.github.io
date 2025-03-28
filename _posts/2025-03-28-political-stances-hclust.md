---
layout: post
title: "Political Stances: R Code to Preform HClust"
categories: projects
published: true
in_feed: false
---

```r
##########################
# 1. ENVIRONMENT CREATION --------------------------
##########################
install.packages("tm")
install.packages("Snowball")
install.packages("slam")
install.packages("quanteda")
install.packages('proxy')
install.packages('stylo')
install.packages("philentropy")
install.packages("ggdendro")

library(tm)
library (stringr)
library(wordcloud)
library (slam)
library(quanteda)
library(SnowballC)
library(arules)
library (proxy)
library (cluster)
library (stringr)
library(Matrix)
library(tidytext)
library(plyr)
library(ggplot2)
library(factoextra)
library(mclust)
library(textstem)
library(amap)
library(networkD3)
library(NbClust)
library(cluster)
library(mclust)
library(factoextra) ## for cluster vis, silhouette, etc.
library(purrr)
library(stylo)
library(philentropy)
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)

## Setting the working directory
setwd("C:\\Users\\natal\\OneDrive\\university\\info 5653")

## Loading in the DTM
raw_csv <- read.csv("Document Term Matrix.csv",check.names = FALSE, row.names=1)

## Converting to a Matrix 
raw_matrix <- as.matrix(raw_csv)

## Converting to a Document Term Matrix

dtm_data <- as.DocumentTermMatrix(as.matrix(raw_matrix),weighting=weightTf,stopwords=TRUE)

## Creating a Matrix
dtm_matrix <- as.matrix(dtm_data)
dtm_matrix[1:13,1:10]

labels_vector <- colnames(dtm_matrix)

##########################
# 2. DATA PROCESSING       --------------------------
##########################

## Dividing each element in the row by the sum of elements in that row
## Normalizing the Data
dtm_normalized <- as.matrix(dtm_data)

dtm_normalized <- apply(dtm_normalized,1,function(i) round(i/sum(i),3))
## And transposing to return to a DTM
dtm_normalized <- t(dtm_normalized)

## CREATING A DATAFRAME ##
dtm_df <- as.data.frame(as.matrix(dtm_data))

##########################
# 3. EXPLORATORY DATA ANALYSIS       --------------------------
##########################

## Counting word frequencies 
word_frequences <- colSums(dtm_matrix)
word_frequences <- sort(word_frequences,decreasing=TRUE)

## Visualizing with the wordcloud library
wordcloud(names(word_frequences),word_frequences,max.words=3000)

##########################
# 4. DISTANCE MEASURES       --------------------------
##########################

## Calculating the cosine distance with the normalized data
cosine_distance <- dist(dtm_normalized,method='cosine')

## Filling any NA values
cosine_distance[is.na(cosine_distance)] <- 0 # Replacing the NA values

## Creating a matrix for subsequent filtering
cosine_distance_matrix <- as.matrix(cosine_distance)

##########################
# 5. CLUSTERING            --------------------------
##########################

# 5.1 USING H CLUST WITH COSINE SIMILARITY           --------------------------
length(cosine_distance) #8333403

## After some EDA it looks like the cosine distances is a rather large matrix, and can't be visualzed fully
## using any of the dendrograms, radial or other visualizations

## A random index was taken to filter for cosine similarity 
order_indices <- order(-cosine_distance_matrix[2500, ])  

## Using the order indicies, the most similar words from the provided row are identified
sorted_cosine_matrix <- cosine_distance_matrix[order_indices, order_indices]

## The labels vector is using the words from the original document term matrix
labels_vector <- colnames(dtm_matrix)  

## The labels are then sorted using the identified order in the cosine distance matrix
sorted_labels <- labels_vector[order_indices]

## The labels are then shortened to be the length of the sorted_cosine_matrix
smaller_labels <- sorted_labels[1:nrow(sorted_cosine_matrix)]



## Start and end values are defined here to identify the first 500 most similar words
start <- 1
end <- 500

## Subseting the data with the above start and end values and generating a distance object
smaller_data <- sorted_cosine_matrix[start:end, start:end]
smaller_data_dist <- as.dist(smaller_data)  

## Labels are now pruned to have the start and end value provided from the sorted cosine matrix
labels <- smaller_labels[start:end]

## Preforming the hierarchical clustering using Hclust
cosine_hclusters <- hclust(smaller_data_dist, method = "ward.D")

## Assinging the sorted and mapped labels
cosine_hclusters$labels <- labels  


# 5.2 CREATING A DENDROGRAM           --------------------------
dendrog <- dendroNetwork(cosine_hclusters, fontSize = 8, height = 5000, width = 500,zoom=TRUE) 
saveNetwork(dendrog, "hclust random parition large.html", selfcontained = TRUE )

# 5.3 CREATING A RADIAL NETWORK           --------------------------
## The display of the radial network will need less data provided than the Dendrogram, so this portion is repeat
## from the above section
start <- 1
end <- 100

##Creating an even smaller subset
smaller_data <- sorted_cosine_matrix[start:end, start:end]  
smaller_data_dist <- as.dist(smaller_data)


## Fixing the labels and reclustering
labels <- smaller_labels[start:end]
cosine_hclusters <- hclust(smaller_data_dist, method = "ward.D")
cosine_hclusters$labels <- labels 

## Convert to a dendrogram and plot
hcd_pruned <- as.dendrogram(cosine_hclusters)

## Plotting and saving the radial network
radial <- radialNetwork(as.radialNetwork(hcd_pruned), fontSize=10)
saveNetwork(radial, "hclust radial small.html", selfcontained = TRUE )
```

