---
layout: post
title: "Political Stances: Naive Bayes"
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

### Overview
Naïve Bayes is a machine learning model which uses probabilistic classification to assign a label to a text. Using probability the multinomial model may consider the number of tokens and the probability of the token falling into its respective label. Its naïvety in part due to the few parameters needed to train but also the assumption that the model may then not capture as much nuance within the data as a result of this. (Grimmer et al: 20222). For this project, its simplicity is a strongsuit the linear classifications which are not prone to overfitting.

The questions posed in the [introduction](https://nataliermcastro.github.io/projects/2025/01/14/political-stances-introduction.html) are a well suited task for this method. First, considering how climate change is represented requires an understanding of the way in which the text composes the document. Being able to classify the documents into the labels, which were already embedded into the creation of the document, may inform how climate change is represented at the federal level. Next, when considering news polarization with respect to climate change, a strength in classifier performance specifically when delineating descriptions about Republican or Democrat headlines, may support the idea of increased news polarization.

**Table of Contents**
- [Data Preparation](#data-prep)
- [Method](#method)
- [A Naive Reading of the News](#results-news-data)
- [Interpreting Features about Climate Bills Naively](#results-bills-data)
- [Conclusions](#conc)
  
---

 <a id="data-prep"></a>
### Data Preparation
All supervised learning models (ie, they take label data) require a few things. The first, is a train test parition. This is important because to test the models efficacy it requires an unseen dataset - it not the model is trained incorrectly and is often times [overfit](https://en.wikipedia.org/wiki/Overfitting). Next, the supervised model requires labels. The [data](https://nataliermcastro.github.io/projects/2025/01/14/political-stances-data.html) utilized in this project is a combination of [News Headlines](https://huggingface.co/datasets/nataliecastro/climate-news-countvectorizer-dtm/viewer) and [Proposed Bills](https://huggingface.co/datasets/nataliecastro/climate-bills-lemmed-count-vectorizer/tree/main) tagged as climate at the federal level. The data may be found either at the in-text links just provided, or at the [HuggingFace Collection](https://huggingface.co/collections/nataliecastro/climate-policy-bills-67afd0eaa0c3f328d4b00136) developed for this project. The prior labels generated for news data are the mentioned partisian affiliation, and the publisher of the news source. For the proposed bills the labels included are the bill sponsor's state, partisian affiliation, [bill type](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-naive-bayes.html#bill-type), and hearing committee. 

For the purposes of the following machine learning methods, a 'combined' metadata label was generated. For all combined labels the token '\|' was utilized to split the different labels. This token was selected because of its rarity in the other labels, and a clean way to later visualize the combined labels. An example of news metadata label may be 'Republican \| The Verge', and for the proposed cliamte bills 'hr \| D \| hi'. The combined label for the bills data first represents the bill type, or what chamber it originated from, then the bill sponsor's partisian affiliation and state. This code is consistent across the entire dataset. The two figures below illustrate the headed versions of the data frames, with their labels preserved. It should be noted that the labels were removed for the classification by the model, or else it wouldn't work!

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/NB - bills data.png" alt="Labeled Proposed Climate Bill Data Headed Dataframe"  /></span> 
			</div>
		</div>
	</div>
</section>

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/NB - news data.png" alt="Labeled News Headline Data Headed Dataframe"  /></span> 
			</div>
		</div>
	</div>
</section>

To generate a Train Test Split consistently across all of the superivsed models (Naive Bayes, [Decision Trees](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-decision-trees.html), and [Support Vector Machines](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-svm.html)), a custom function was developed. This function would take a Pandas DataFrame and then the label column. This is used to generate both lists of labels and to then visualizae the distribution of the particular label with respect to the train test split. 

The function uses [sklearn's train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) with a parition of 30% reserved for testing. The size of the dataset allows for this large of a parition to be taken. The custom function then generates a list of the training labels, and the testing labels. The option to remove the label column is preserved in the function but seldomly used. Finally, the function returns the training data, testing data, and their respective labels stored in separate lists. 

``` python
def train_test_splitter(data, label_column):
    
    data_train, data_test = train_test_split(data, test_size = 0.3,)
    labels_train = data_train[label_column]
    labels_test = data_test[label_column]
    
    #data_train.drop(columns='LABEL', inplace=True)
    #data_test.drop(columns='LABEL', inplace=True)
    
    
    return (data_train, data_test, labels_train, labels_test)

```


Each label had a custom testing and training split, as the model can only handle one label at once (but this was in an attempt to be mitigated by generating the combined 'metadata' labels. It is illustrated in the *Train and Testing Data Parition Table* that regardless of the label, the split generated is the same size every time. In total eight dataframes were generated for each type of label. To then include the testing and training splits, a total of sixteen dataframes were generated.

**Train and Testing Data Parition**
<table>
<thead>
<tr><th>                                                   </th><th style="text-align: center;">  Training Data </th><th style="text-align: center;"> Testing Data </th></tr>
</thead>
<tbody>
<tr><td>News Headline: Partisian Affiliation               </td><td style="text-align: center;">      573       </td><td style="text-align: center;">     246      </td></tr>
<tr><td>News Headlines: Publisher                          </td><td style="text-align: center;">      573       </td><td style="text-align: center;">     246      </td></tr>
<tr><td>News Headlines: Publisher and Partisian Affiliation</td><td style="text-align: center;">      573       </td><td style="text-align: center;">     246      </td></tr>
<tr><td>Climate Bills: Sponsor Affiliation                 </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Sponsor State                       </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Metadata                            </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Bill Type                           </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Hearing Committee                   </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
</tbody>
</table>

The distribution of the Partisian Labels are illustrated below. It is important when developing these models to consider the distribution of the labels within the training and testing sets. If the training sets are disparate this may lead to the model overfitting on a particular label and not 'learning' what comprises of the other labels.  Provided below are the distributions, specifically for the testing data of the Partisian Labels. For the News Headlines, the distribution is relatively split equally between Republican and Democrat. In comparison to the Proposed Cliamte Bills, the distribution is not as equal - and includes the Indepent Partisian Affiliation. These differences in distribution are not something to necessarily correct - as the goal of machine learning is not to have the model discern half of the data automatically as one label - but to instead be able to learn the semantic characteristics which generate text to be a particular label.


<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/NB - Testing Data - Party Data Partisan Labels.png" class="image fit thumb"><img src="/assets/images/NB - Testing Data - Party Data Partisan Labels.png" alt="" /></a>
			<h3>Testing Data Paritisan Label Paritions of Proposed Climate Bills</h3>
			<p> The Democrat party has 538 testing instances, which is more than the Republican party with 422 testing instances. The Independent party has seven testing instances. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/NB - Testing Data - News Data Partisan Labels.png" class="image fit thumb"><img src="/assets/images/NB - Testing Data - News Data Partisan Labels.png" alt="" /></a>
			<h3>Testing Data Partisian Label Paritions of News Headlines</h3>
			<p> The distribution of testing data between Republican and Democrat is relatively even. The Republican label has 124 testing instances, and the Democrat label has 122 instances. </p>
		</article>
	</div>
</section>


<a id="method"></a>
### Method
<section>
    <div class="row">
        <div class="col-6 col-12-small">
            <ul class="actions" style="display: flex; gap: 10px; list-style: none; padding: 0;">
                <li><a href="https://nataliermcastro.github.io/projects/2025/04/21/political-stances-naive-bayes-code.html" class="button fit small">View Code</a></li>
		<li><a href="https://github.com/NatalieRMCastro/climate-policy/blob/main/6.%20Naive%20Bayes.ipynb" class="button fit small">Visit GitHub Repository</a></li>
            </ul>
        </div>
    </div> 
</section> 

The method section is split into two parts, first an explanation of how the MNB models were fit. As noted earlier, there is eight different dataframes which MNB will be trained on, so internal consistency is important - especially when evaluating the performance of different labels tandemly. Next, the evaluation function is described and its application. 

#### Systemicatically Fitting the Multinomial Naïve Bayes Models

```python
def mnb_modeler(data_train, labels_train, data_test, labels_test, label_column_name,  graph_title, labels_name, file_name,filter_top_n = False, N=10 ,fig_x = 6, fig_y = 4):
    data_train = data_train.drop(columns = label_column_name).copy()
    data_test = data_test.drop(columns = label_column_name).copy()
    
    mnb_model = MultinomialNB()

    ## Fitting the data
    mnb_full = mnb_model.fit(data_train, labels_train)
    
    ## Creating predictions
    predictions = mnb_full.predict(data_test)
    
    ## Assessing the models abilitiy
    accuracy, precision, recall = model_verification(labels_test, predictions)
    
    ## Filtering for Clean Visualizations
    if filter_top_n == True:
        labels_test, predictions = filter_top_n_labels(labels_test, predictions, N)

    ## Generating a confusion matrix
    
    matrix_ = confusion_matrix(labels_test, predictions)
    visual_confusion_matrix(matrix_, labels_test, predictions, graph_title, labels_name, file_name, fig_x, fig_y)
    
    return (accuracy, precision, recall)
```

#### Devoping a System for Model Evaluation
<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/Confusion Matrix.png" alt="A Confusion Matrix with the equations for recall, precision, and accuracy"  /></span> 
			</div>
		</div>
	</div>
</section>
*Image Source: Jurafsky and Martin, Speech and Language Processing, page 67*

```python
def model_verification(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division = 0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division = 0)
    
    return accuracy, precision, recall
```

### Results
The results section is split into two parts, the first is an evaluation of the Multinomial Naïve Bayes Models' preformance. Next, a discussion will be presented about the nuance behind the confusion matricies presented. Finally, this page will close with the potential implications of the resutls presented here with respect to the research questions outlined in the introduction.

<a id="results-model-evaluation"></a>
### Assessing the Validity of the Multinomial Naïve Bayes
<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/NB - Model Evaluation.png" alt="An overview of the model validity. Everything illustrated is present in the above table."  /></span> 
			</div>
		</div>
	</div>
</section>

<table>
<thead>
<tr><th>                                                   </th><th style="text-align: center;">  Accuracy </th><th style="text-align: center;"> Precision </th><th style="text-align: center;"> Recall </th></tr>
</thead>
<tbody>
<tr><td>News Headlines: Partisian Affiliation              </td><td style="text-align: center;">   0.577   </td><td style="text-align: center;">   0.578   </td><td style="text-align: center;"> 0.577  </td></tr>
<tr><td>News Headlines: Publisher                          </td><td style="text-align: center;">   0.276   </td><td style="text-align: center;">   0.16    </td><td style="text-align: center;"> 0.145  </td></tr>
<tr><td>News Headlines: Publisher and Partisian Affiliation</td><td style="text-align: center;">   0.167   </td><td style="text-align: center;">   0.035   </td><td style="text-align: center;"> 0.038  </td></tr>
<tr><td>Climate Bills: Sponsor Affiliation                 </td><td style="text-align: center;">   0.759   </td><td style="text-align: center;">   0.521   </td><td style="text-align: center;"> 0.518  </td></tr>
<tr><td>Climate Bills: Sponsor State                       </td><td style="text-align: center;">   0.255   </td><td style="text-align: center;">   0.23    </td><td style="text-align: center;"> 0.171  </td></tr>
<tr><td>Climate Bills: Metadata                            </td><td style="text-align: center;">   0.206   </td><td style="text-align: center;">   0.126   </td><td style="text-align: center;"> 0.117  </td></tr>
<tr><td>Climate Bills: Bill Type                           </td><td style="text-align: center;">   0.711   </td><td style="text-align: center;">   0.449   </td><td style="text-align: center;"> 0.463  </td></tr>
<tr><td>Climate Bills: Hearing Committee                   </td><td style="text-align: center;">   0.478   </td><td style="text-align: center;">   0.111   </td><td style="text-align: center;"> 0.109  </td></tr>
</tbody>
</table>


<a id="results-news-data"></a>
### A Naive Reading of the News Headlines


<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/mnb cm - news publisher.png" class="image fit thumb"><img src="/assets/images/mnb cm - news publisher.png" alt="" /></a>
			<h3> Publisher </h3>
			<p> TEXT </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/mnb cm - news partisian affiliation.png" class="image fit thumb"><img src="/assets/images/mnb cm - news partisian affiliation.png" alt="" /></a>
			<h3> Named Political Party </h3>
			<p> TEXT </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/mnb cm - news combined label.png" class="image fit thumb"><img src="/assets/images/mnb cm - news combined label.png" alt="" /></a>
			<h3>Combined Headline Metadata</h3>
			<p> TEXT .</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/NB - News Data Publisher Labels.png" class="image fit thumb"><img src="/assets/images/NB - News Data Publisher Labels.png" alt="Freerepublic has the most news sources that are meaningful according to Naive Bayes with 40. The remained all have below ten." /></a>
			<h3>Hearing Committee</h3>
			<p> TEXT </p>
		</article>
	</div>
</section>



<a id="results-news-data"></a>
### Interpreting Features About Climate Bills Naively

<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/mnb cm - sponser state label truncated.png" class="image fit thumb"><img src="/assets/images/mnb cm - sponser state label truncated.png" alt="" /></a>
			<h3>Bill Sponsor State</h3>
			<p> TEXT </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/mnb cm - sponsor affiliation label.png" class="image fit thumb"><img src="/assets/images/mnb cm - sponsor affiliation label.png" alt="" /></a>
			<h3> Sponsor Affiliation </h3>
			<p> TEXT </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/mnb cm - bill metadata label.png" class="image fit thumb"><img src="/assets/images/mnb cm - bill metadata label.png" alt="" /></a>
			<h3>Combined Bill Metadata</h3>
			<p> TEXT .</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/mnb cm - bill committee label.png" class="image fit thumb"><img src="/assets/images/mnb cm - bill committee label.png" alt="" /></a>
			<h3>Hearing Committee</h3>
			<p> TEXT </p>
		</article>
	</div>
</section>

 <a id="bill-type"></a>
<div style="display: flex; gap: 20px; align-items: flex-start;">

  <!-- Left Column: Image Section -->
  <section style="flex: 1;">
    <div class="box alt">
      <div class="row gtr-50 gtr-uniform">
        <div class="col-12">
          <span class="image fit">
            <img src="/assets/images/mnb cm - bill type label.png" alt="" />
          </span>
          <figcaption>Confusion Matrix for MNB Classification of Bill Type</figcaption>
        </div>
      </div>
    </div>
  </section>

  <!-- Right Column: Table -->
  <div style="flex: 1;">
	 Abbreviations for Federal Bill Types
    <table>
<thead>
<tr><th>Abbreviation  </th><th> Bill Type                                                     </th></tr>
</thead>
<tbody>
<tr><td>hconres       </td><td>Concurrent Resolution Originating From House of Representatives</td></tr>
<tr><td>hjres         </td><td>Joint Resolution Originating from House of Representatives     </td></tr>
<tr><td>hr            </td><td>House of Representatives                                       </td></tr>
<tr><td>hres          </td><td>Resolution From House of Representatives                       </td></tr>
<tr><td>s             </td><td>Senate                                                         </td></tr>
<tr><td>sconres       </td><td>Concurrent Resolution Originating From Senate                  </td></tr>
<tr><td>sjres         </td><td>Joint Resolution Originating from Senate                       </td></tr>
<tr><td>sres          </td><td>Resolution from Senate                                         </td></tr>
</tbody>
</table>
  </div>

</div>

https://www.house.gov/the-house-explained/the-legislative-process/bills-resolutions#:~:text=Concurrent%20Resolutions&text=A%20concurrent%20resolution%20originating%20in,the%20Secretary%20of%20the%20Senate.


### Conclusions

---
### Bibliography

Grimmer, J., Roberts, M. E., & Stewart, B. M. (2022). Text as data: A new framework for machine learning and the social sciences. Princeton University Press.

Jurafsky, D., & Martin, J. (2025). Speech and Language Processing (3rd ed.). https://web.stanford.edu/~jurafsky/slp3/ed3book_Jan25.pdf

