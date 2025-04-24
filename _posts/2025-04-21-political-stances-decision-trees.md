---
layout: post
title: "Political Stances: Decision Trees"
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
A decision tree is used to identify which features are important in a label. The salient features are those which are discriminating factors to determine what is important to a particular label. For example, a decision tree can provide important information about what is important for 'Republican', or a particular news media company. 

The discrimate word, or the root node, is used as the original split for the classification path. The features included in a decision tree subsqeuntly then show 'lef nodes' which are further paritions of the root node. The tree will then generate until the paritions are no longer fit (either decided through parameters or the data is exahusted). 

The nodes generate a parallel leaf which illustrate the change of a particular item, with respect to the other options that are available to it. For this example the chance is demonstarted in the node on the right. Until the leaf reaches its final node, the chance is not yet illustrated clearly through a special model type, samples, and class nodes.

<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/decision tree example.png" class="image fit thumb"><img src="/assets/images/decision tree example.png" alt="" /></a>
			<h3>Decision Tree Structure</h3>
   <p>Image Citation: <a href="https://www.geeksforgeeks.org/decision-tree/" target="_blank">Geeks for Geeks Decision Tree</a></p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/decision tree node example.png" class="image fit thumb"><img src="/assets/images/decision tree node example.png" alt="" /></a>
			<h3>Decision Tree Nodes</h3>
			<p>Image Citation: <a href="https://medium.com/analytics-vidhya/understanding-decision-tree-3591922690a6" target="_blank">Abhigyan "Understanding Decision Tree!!"</a></p>

		</article>
	</div>
</section>



**Table of Contents**
- [Data Preparation](#data-prep)
- [Method](#method)
- [Evaluating The Gini and Entropy Decision Trees](#results-model-evaluation)
- [A Naive Reading of the News](#results-news-data)
- [Interpreting Features about Climate Bills Naively](#results-bills-data)
- [Conclusions](#conc)
  
---

 <a id="data-prep"></a>
### Data Preparation
Data was prepared using the same functions outlined in great detail in the Naive Bayes model. It can be reviewed [here](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-naive-bayes.html#data-prep), or in any of the linked notebooks. The training and testing split is similar to that of the Naive Bayes model as well. A headed version of the dataframes are provided here alongside with the distribution of the train, test, split.

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

#### **Train Test Split Distribution**
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

The same naming scheme and train and test scheme were applied across all machine learning methods applied here. This is in order to be able to compare the methods and discuss the original nuance between them. For all three methods, the exact same data preparation was applied - which is why it is not repeated here. 

### Method
<section>
    <div class="row">
        <div class="col-6 col-12-small">
            <ul class="actions" style="display: flex; gap: 10px; list-style: none; padding: 0;">
                <li><a href="https://nataliermcastro.github.io/projects/2025/04/21/political-stances-decision-tree-code.html" class="button fit small">View Code</a></li>
		<li><a href="https://github.com/NatalieRMCastro/climate-policy/blob/main/7.%20Decision%20Trees.ipynb" class="button fit small">Visit GitHub Repository</a></li>
            </ul>
        </div>
    </div> 
</section> 

A Decision Tree Modeler was developed similar to that of the systematic testing developed in [Naive Bayes Modeling](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-naive-bayes.html). The Decision Tree used to classify the data was from [SciKit Learn's DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier). Both the Entropy and Gini models were tested. The splitter was set to 'best' , in opposition to the 'random' node split. The max depth was then limited to 7 for Decision Tree readability. The minimum samples split and leaf were inflated to ten. This is because of the scope and number of features in the dataset. The data was also limited to 500 maximum features in order to extract the most frequent commonalities between the data. The max leaf nodes and the minimum impurity decrease were set at the defaults from SciKit Learn. Finally, the class weight was balanced in order to normalize the differences in label distribution. In order to preserve systematic testing, a random state was set to thirty. The random state was able to preserve the essence of 'randomness' across the nuancce within the data.

```python
## Decision Tree Modeler:
def tree_modeler(data_train, labels_train, data_test, labels_test, label_column_name, graph_title, labels_name, file_name,filter_top_n = False, N=10 ,fig_x = 6, fig_y = 4):
    data_train = data_train.drop(columns = label_column_name).copy()
    data_test = data_test.drop(columns = label_column_name).copy()
    
    feature_names_train = list(data_train.columns)
    feature_names_test = list(data_test.columns)
    
    decision_tree = DecisionTreeClassifier(criterion='entropy', ## This is changed based on the model test that was used
                                      splitter = 'best',
                                      max_depth = 7,
                                      min_samples_split = 10,
                                      min_samples_leaf = 10,
                                      max_features = 500,
                                      random_state = 30,
                                      max_leaf_nodes = None,
                                      min_impurity_decrease = 0.0,
                                      class_weight = 'balanced')

    ## Fitting the data
    decision_tree_model = decision_tree.fit(data_train, labels_train)
    tree.plot_tree(decision_tree,feature_names = feature_names_train)
    plt.savefig(f"Decision Tree - {file_name}.png")
    
    ## Plotting the tree
    dot_data = tree.export_graphviz(decision_tree_model, out_file=None,
                                    feature_names=feature_names_train,
                                    class_names = [str(cls) for cls in decision_tree_model.classes_],
                                    filled=True,
                                    rounded=True,
                                    special_characters=True, 
                                    label='all', proportion = True
                                   )

    cleaned_dot_data = re.sub(r'value = \[[^\]]*\]&lt;br/&gt;|value = \[[^\]]*\]<br/?>', '', dot_data)
    
    graph = graphviz.Source(cleaned_dot_data)
    graph.render(f"Decision Tree - {graph_title}",cleanup=True)
    
    
    ## Creating predictions
    predictions = decision_tree_model.predict(data_test)
    
    

    ## Assessing the models abilitiy
    accuracy, precision, recall = model_verification(labels_test, predictions)
    
    ## Filtering for Clean Visualizations
        ## Filtering for Clean Visualizations
    if filter_top_n:
        # Call filter_top_n_labels to get filtered labels and predictions
        labels_test_filtered, predictions_filtered = filter_top_n_labels(labels_test, predictions, N)

        # If data remains after filtering, create the filtered confusion matrix
        if len(labels_test_filtered) > 0 and len(predictions_filtered) > 0:
            visual_confusion_matrix(labels_test_filtered, predictions_filtered,
                                    f"{graph_title} (Top {N})", labels_name,
                                    f"filtered_{file_name}", fig_x, fig_y)
        else:
            print(f"[Warning] No data left after filtering top {N} labels — skipping confusion matrix.")
    else:
        # If no filtering is needed, generate confusion matrix with all data
        visual_confusion_matrix(labels_test, predictions, graph_title, labels_name, file_name, fig_x, fig_y)
    
    return (accuracy, precision, recall)
```

 <a id="results-model-evaluation"></a>
### Evaluating Gini and Entropy Decision Trees
The precision, recall, and accuracy measures will be reported and compared. By an initial look at the bar charts, it is evident that the model only learned features about paritisan affiliation in news, bill sponsor affiliation and bill type. The preformance of all of the other models illustrate a failure of learning for the more complicated labels such as sponsor state or bill hearing committee.

In this section,first, the evaluation metrics will be reported for the entire set, and then for the three models which had non-zero (some learning) preformance.


#### Gini Model Evaluation

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/Decision Trees Gini - Model Evaluation.png" alt="An overview of the model validity. Everything illustrated is present in the below table."  /></span> 
			</div>
		</div>
	</div>
</section>

**Average Full Model Preformance**
- Accuracy: 10.18%
- Recall: 18.12%
- Precision: 17.8%

**Average Learned Label Preformance**
- Accuracy: 37.33%
- Recall: 46.33%
- Precision: 47%

The overall preformance of the model is poor. The evaluation metrics, specicially that of precision and recall, demonstrate that the model did not learn in many facets. It preformed the best on partisan affiliation for both cliamte bills and news headlines. The Gini model preformed the same on News Headlines paritisan affiliation as [Naive Bayes](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-naive-bayes.html#results-model-evaluation) (58%) but had a lower accuracy on climate bills (compared to 76%). The Gini Model preformed stronger than the SVM model on News Headlines (54%) but lower on Bill Sponsor evaluation (compared to 80%). 

This demonstrates that in comparison to the other models utilized, and for this data, that the Gini Model would not be appropriate to draw accurate conclusions on.

#### Entropy Model Evaluation
<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/Decision Trees Entropy - Model Evaluation.png" alt="An overview of the model validity. Everything illustrated is present in the below table."  /></span> 
			</div>
		</div>
	</div>
</section>

**Average Full Model Preformance**
- Accuracy: 19.12%
- Recall: 18.5%
- Precision: 17.37%

**Average Learned Label Preformance**
- Accuracy: 49%
- Recall: 45.33%
- Precision: 44.33%

The Entropy model had stronger preformance than the Gini model. It specically increased the ability to classify the Bill Type imprving the Gini accuracy from 9% to 46%. However, 'better' does not necessarily mean the model would be able to reliably classify, as while the accuracy is higher the precision and recall averages are both lower. In comparison to the aforementioned comparison between Naive Bayes and the SVM models, the Entropy Model mirrors the preformance of the Gini. The difference lies in the ability for Bill Type Prediction. The Entropy model preforms at an evaluation of 31% (P), 46% (A), and 29% (R). When comparing to the Gini, the models have similar accuracy and recall (35% and 30% respectively), but do not preform as well as the Naive Bayes Model (45% P, 71% A, 46% R) or the SVM Model (76% P, 79% A, 69% R). 

The classifications observed by the evaluation metrics for both Gini and Entropy illustrate that the data may not be a good fit for the Decision Tree model.

#### Label Classification: Consufion Matrix Comparison
<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/decision tree gini cm- bills partisian affiliation.png" class="image fit thumb"><img src="/assets/images/decision tree gini cm- bills partisian affiliation.png" alt="" /></a>
			<h3> Gini Partisian Affiliation Labeling for Climate Bills </h3>
			<p> TEXT </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/decision tree entropy cm- bills partisian affiliation.png" class="image fit thumb"><img src="/assets/images/decision tree entropy cm- bills partisian affiliation.png" alt="" /></a>
			<h3> Entropy Partisian Affiliation Labeling for Climate Bills </h3>
			<p> TEXT </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/decision tree gini cm- news partisian affiliation.png" class="image fit thumb"><img src="/assets/images/decision tree gini cm- news partisian affiliation.png" alt="" /></a>
			<h3>Gini Partisian Affiliation Labeling for News Headlines</h3>
			<p> TEXT .</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/decision tree entropy cm- news partisian affiliation.png" class="image fit thumb"><img src="/assets/images/decision tree entropy cm- news partisian affiliation.png" alt=" " /></a>
			<h3>Entropy Partisian Affiliation Labeling for News Headlines</h3>
			<p> TEXT </p>
		</article>
  <article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/decision tree gini cm- bill type label.png" class="image fit thumb"><img src="/assets/images/decision tree gini cm- bill type label.png" alt="" /></a>
			<h3>Entropy Proposed Bill Type Labeling</h3>
			<p> TEXT .</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/decision tree entropy cm- bill type label.png" class="image fit thumb"><img src="/assets/images/decision tree entropy cm- bill type label.png" alt=" " /></a>
			<h3>Entropy Proposed Bill Type Labeling</h3>
			<p> TEXT </p>
		</article>
	</div>
</section>

<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="3" style="text-align: center;"> **Gini** </th>
      <th colspan="3" style="text-align: center;"> **Entropy** </th>
    </tr>
    <tr>
      <th style="text-align: center;">Accuracy</th>
      <th style="text-align: center;">Precision</th>
      <th style="text-align: center;">Recall</th>
      <th style="text-align: center;">Accuracy</th>
      <th style="text-align: center;">Precision</th>
      <th style="text-align: center;">Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>News Headlines: Partisian Affiliation</td>
      <td style="text-align: center;">0.585</td><td style="text-align: center;">0.621</td><td style="text-align: center;">0.607</td>
      <td style="text-align: center;">0.585</td><td style="text-align: center;">0.621</td><td style="text-align: center;">0.607</td>
    </tr>
    <tr>
      <td>News Headlines: Publisher</td>
      <td style="text-align: center;">0.008</td><td style="text-align: center;">0</td><td style="text-align: center;">0.011</td>
      <td style="text-align: center;">0.004</td><td style="text-align: center;">0.002</td><td style="text-align: center;">0.01</td>
    </tr>
    <tr>
      <td>News Headlines: Publisher and Partisian Affiliation</td>
      <td style="text-align: center;">0.008</td><td style="text-align: center;">0</td><td style="text-align: center;">0.008</td>
      <td style="text-align: center;">0.004</td><td style="text-align: center;">0</td><td style="text-align: center;">0.008</td>
    </tr>
    <tr>
      <td>Climate Bills: Sponsor Affiliation</td>
      <td style="text-align: center;">0.454</td><td style="text-align: center;">0.44</td><td style="text-align: center;">0.478</td>
      <td style="text-align: center;">0.434</td><td style="text-align: center;">0.396</td><td style="text-align: center;">0.457</td>
    </tr>
    <tr>
      <td>Climate Bills: Sponsor State</td>
      <td style="text-align: center;">0.033</td><td style="text-align: center;">0.019</td><td style="text-align: center;">0.019</td>
      <td style="text-align: center;">0.034</td><td style="text-align: center;">0.035</td><td style="text-align: center;">0.049</td>
    </tr>
    <tr>
      <td>Climate Bills: Metadata</td>
      <td style="text-align: center;">0.029</td><td style="text-align: center;">0</td><td style="text-align: center;">0.005</td>
      <td style="text-align: center;">0.017</td><td style="text-align: center;">0.009</td><td style="text-align: center;">0.036</td>
    </tr>
    <tr>
      <td>Climate Bills: Bill Type</td>
      <td style="text-align: center;">0.09</td><td style="text-align: center;">0.353</td><td style="text-align: center;">0.305</td>
      <td style="text-align: center;">0.461</td><td style="text-align: center;">0.309</td><td style="text-align: center;">0.293</td>
    </tr>
    <tr>
      <td>Climate Bills: Hearing Committee</td>
      <td style="text-align: center;">0.022</td><td style="text-align: center;">0.002</td><td style="text-align: center;">0.014</td>
      <td style="text-align: center;">0.009</td><td style="text-align: center;">0.012</td><td style="text-align: center;">0.047</td>
    </tr>
  </tbody>
</table>




### Decision Tree Visualization

Due to the lower results of the ability to classify across many labels, the decision trees illustrated here are the only models that may be assumed were not direct guesses, and illustrative of the models ability to learn. To generate these visualizations, the values (or impurity) scores were removed, and the class name was added to the node in order to identify more clearly the detail within the tree.

### Decision Tree: Bill Sponsor Affiliation
<iframe
  src="https://mozilla.github.io/pdf.js/web/viewer.html?file=https://raw.githubusercontent.com/NatalieRMCastro/nataliermcastro.github.io/89c8ed91d5d8d2a8f1387f0795ca1abcab36f197/assets/images/Decision%20Tree%20-%20Climate%20Bills%20Sponsor%20Affiliation%20Entropy.pdf"
  width="100%"
  height="600px"
  style="border: none;">
</iframe>

### Decision Tree: Bill Type
<iframe
  src="https://mozilla.github.io/pdf.js/web/viewer.html?file=https://raw.githubusercontent.com/NatalieRMCastro/nataliermcastro.github.io/3543eb536c77a40d7192689f63f7738ae0a98136/assets/images/Decision%20Tree%20-%20Climate%20Bill%20Type%20Entropy.pdf"
  width="100%"
  height="600px"
  style="border: none;">
</iframe>

### Decision Tree: News Headlines 
<iframe
  src="https://mozilla.github.io/pdf.js/web/viewer.html?file=https://raw.githubusercontent.com/NatalieRMCastro/nataliermcastro.github.io/89c8ed91d5d8d2a8f1387f0795ca1abcab36f197/assets/images/Decision%20Tree%20-%20News%20Headlines%20Partisan%20Affiliation%20Entropy.pdf"
  width="100%"
  height="600px"
  style="border: none;">
</iframe>

### Conclusions
