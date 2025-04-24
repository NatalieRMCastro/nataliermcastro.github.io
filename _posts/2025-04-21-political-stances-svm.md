---
layout: post
title: "Political Stances: Support Vector Machines"
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
The Support Vector Machine (SVM) preforms linear classification (among with other kernels) to project the data into classifications, based on their long vector inputs. Applying this model to text data may allow for classification of unlabeled text. This classification is based in [Vapnik's statistical learning theory ](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) which has provided insight into the classification of text. These methods inform the ability to understand nuance and future action of a particular text. For exampale, the query '[using support vector machines to predict the future](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C6&q=using+support+vector+machines+to+predict+the+future&btnG=)' generates over one million citations on Google Scholar. Specifically in the context of bills, it may be insightful to understand the characteristics and similarities to understand how bills change and evolve both spatially and temporally. Applying SVM to news data may provide insights into media bias as well. How are certain features in the news cycle represented differently by different sources and within different contexts? Applyinng SVM in for this case will provide insight into differences of language when reporting on partisian issues and how this polarization (may or may not) be present in proposed climate oriented bills. 

**Table of Contents**
- [Sentiment Analysis](#sentiment-analysis)
- [Data Preparation](#data-prep)
- [Method](#method)
- [Model Evaluation](#results-model-evaluation)
- [Assessing the Preformance of SVM Models](#results-model-evaluation)
- [Assessing the Classifications of the Polynomial Cost 1000 SVM](#polynomial-classifications)
- [Conclusions](#conclusions)
  
---

 <a id="sentiment-analysis"></a>
### Sentiment Analaysis
The sentiment was calculated for all news articles and climate bills in an attempt to generate a classifier which preforms well on sentiment. This decision was motivated as it may be interesting to train a classifier - in addition to the tools already provided - in an attempt to identify nuance within sentiment. To generate a sentiment label to compliment the eight labels in the data already, NLTK was utilized. The [Seniment Intensity Analyzer from the Vader attribute](https://www.nltk.org/api/nltk.sentiment.vader.html#nltk.sentiment.vader.SentimentIntensityAnalyzer) was utilized to generate a numerical score referred to as the 'compound'. It illustrates how strong a particular emotion is throughout the text. 

The analyzer was instantiated through a non-paramaratized call and then wrapped into a function:

```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def word_scorer(x):
    sentiment = sia.polarity_scores(text=x)
    return (sentiment['compound'])

def sentiment_translator(x):
    if x >= 0.7:
        return ("Positive")

    else:
        return ("Negative")
```
This function utilized *x* as an input, which would then be lambda applied to an alphabatized text string (as the data was reconstructed from the Count Vectorizer counts). This generated a sentiment value column, which would then be converted using the *sentiment_translator* into a categorical label. The threshold of 0.7 was selected after an exploratory analysis of both the data and the distribution of sentiment.

#### Sentiment Distribution for the Testing Dataset
<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Sentiment Distribution - Climate Bills.png" class="image fit thumb"><img src="/assets/images/Sentiment Distribution - Climate Bills.png" alt="" /></a>
			<h3>Sentiment Score Distribution: Proposed Climate Bills</h3>
			<p> The distribution matches the news headlines, but has an overwhelming about of positive samples. There is a long skewed tail beginning at 0.7 valence. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Sentiment Distribution - News.png" class="image fit thumb"><img src="/assets/images/Sentiment Distribution - News.png" alt="" /></a>
			<h3>Sentiment Score Distribution: News Headlines</h3>
			<p> The scores are a negatively skewed (right leaning not sentiment negative!) modal distribution. The majority of the sample have an strong positive average. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Sentiment Labels Distribution - Climate Bills.png" class="image fit thumb"><img src="/assets/images/Sentiment Labels Distribution - Climate Bills.png" alt="" /></a>
			<h3>Sentiment Label Distribution: Proposed Climate Bills</h3>
			<p>The distribution is uneven with 135 negative instances and 3,088 positive instances. This may cause later classification issues.</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Sentiment Labels Distribution - News Headlines.png" class="image fit thumb"><img src="/assets/images/Sentiment Labels Distribution - News Headlines.png" alt="" /></a>
			<h3>Sentiment Label Distribution: News Headlines</h3>
			<p> The distribution is mor equal than the climate bills with 492 positive instances and 327 negative instances</p>
		</article>
	</div>
</section>

 <a id="data-prep"></a>
### Data Preparation

<table>
<thead>
<tr><th>                                                   </th><th style="text-align: center;">  Training Data </th><th style="text-align: center;"> Testing Data </th></tr>
</thead>
<tbody>
<tr><td>News Headline: Partisian Affiliation               </td><td style="text-align: center;">      573       </td><td style="text-align: center;">     246      </td></tr>
<tr><td>News Headlines: Publisher                          </td><td style="text-align: center;">      573       </td><td style="text-align: center;">     246      </td></tr>
<tr><td>News Headlines: Publisher and Partisian Affiliation</td><td style="text-align: center;">      573       </td><td style="text-align: center;">     246      </td></tr>
<tr><td>News Headlines: Sentiment                          </td><td style="text-align: center;">      573       </td><td style="text-align: center;">     246      </td></tr>
<tr><td>Climate Bills: Sponsor Affiliation                 </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Sponsor State                       </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Metadata                            </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Bill Type                           </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Hearing Committee                   </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
<tr><td>Climate Bills: Sentiment                           </td><td style="text-align: center;">      2256      </td><td style="text-align: center;">     967      </td></tr>
</tbody>
</table>

 <a id="method"></a>
### Method
<section>
    <div class="row">
        <div class="col-6 col-12-small">
            <ul class="actions" style="display: flex; gap: 10px; list-style: none; padding: 0;">
                <li><a href="https://nataliermcastro.github.io/projects/2025/04/21/political-stances-svm-code.html" class="button fit small">View Code</a></li>
		<li><a href="https://github.com/NatalieRMCastro/climate-policy/blob/main/8.%20Support%20Vector%20Machines.ipynb" class="button fit small">Visit GitHub Repository</a></li>
            </ul>
        </div>
    </div> 
</section> 



<div style="display: flex; justify-content: space-around; text-align: center;">

  <div>
    <img src="/assets/images/svm linear split.png" alt="" width="200">
    <p>Linear Kernel</p>
  </div>

  <div>
    <img src="/assets/images/svm polynomial split.png" alt="" width="200">
    <p>Polynomial Kernel</p>
  </div>

  <div>
    <img src="/assets/images/svm rbf split.png" alt="" width="200">
    <p>Radial Basis Function Kernel</p>
  </div>

</div>
*Image Source: [Plot Classification Boundaries with Different SVM Kernels](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html)*

Multiple Kernels and costs were tested to exhaust the possibilities of the model. The linear kernel 

 <a id="results-model-evaluation"></a>
### Assessing the Preformance of Multiple Iterations of SVM
Across all tests, the relative performance of different labels is similar. The Sentiment classification has the highest accuracy with a range between 93% and 94% for Climate Bills. The precision and recall mirror these patterns with the same rate of prediction. The News headlines have a larger range between 89%  and 90%. Precison tends to be above the accuracy, ranging between 82% - 94%. The recall is much lower, yet comparable to the scores of the other 'well' preforming labels in the news data. It ranges from 58% to 66%. The model had similar preformance trends to that of the Naive Bayes model and the same labels.


<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/SVM Linear Cost 100 - Model Evaluation.png" class="image fit thumb"><img src="/assets/images/SVM Linear Cost 100 - Model Evaluation.png" alt="" /></a>
			<h3>Linear Kernel with Cost 100</h3>
			<p> The average accuracy between the Linear Models is the same at 56%. The average precision is 43.4% and the average recall is 43.2%.</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/SVM Linear Cost 1000 - Model Evaluation.png" class="image fit thumb"><img src="/assets/images/SVM Linear Cost 1000 - Model Evaluation.png" alt="" /></a>
			<h3> Linear Kernel with Cost 1000 </h3>
			<p> There is no observed difference between the varying costs in any evaluation metric.</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/SVM Poly - cost 100 - Model Evaluation.png" class="image fit thumb"><img src="/assets/images/SVM Poly - cost 100 - Model Evaluation.png" alt="" /></a>
			<h3>Polynomial Kernel with Cost 100</h3>
			<p>The average accuracy is 54.5%, average precision is 45.9%, and average recall is  40.8%.</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/SVM Poly - cost 1000 - Model Evaluation.png" class="image fit thumb"><img src="/assets/images/SVM Poly - cost 1000 - Model Evaluation.png" alt="" /></a>
			<h3>Polynomial Kernel with Cost 1000</h3>
			<p>The average accuracy is 56.1%, average precision is 46.1%, and average recall is  42.7%. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/SVM Poly - cost 1000 - Model Evaluation.png" class="image fit thumb"><img src="/assets/images/SVM Poly - cost 1000 - Model Evaluation.png" alt="" /></a>
			<h3>RBF Kernel with Cost 100</h3>
			<p>The average accuracy is 49.4%, average precision is 48.2%, and average recall is  40.8%. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/SVM RBF - cost 1000 - Model Evaluation.png" class="image fit thumb"><img src="/assets/images/SVM RBF - cost 1000 - Model Evaluation.png" alt="" /></a>
			<h3>RBF Kernel with Cost 1000</h3>
			<p>The average accuracy is 49.5%, average precision is 43.2%, and average recall is  42.6%. </p>
		</article>
	</div>
</section>

So... what is the best model? The polynomial kernel with cost 1000. The data was paritioned using a "U" like concave into the other classification. This works particulary well for this type of data because in some cases, the labels are binary - in addition, the few testing instances that are negative may make their features *more salient* because they are more unique in this way. To keep the discussion concisce, only the results from this particular model will be reported. If desired, the outputs from the models are all listed in the [GitHub version of the code notebook](https://github.com/NatalieRMCastro/climate-policy/blob/main/8.%20Support%20Vector%20Machines.ipynb).

 <a id="polynomial-classifications"></a>
### Assessing the Classifications of the Polynomial Cost 1000 SVM
<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/filtered_svm - poly - cost 1000 - news - metadata.png" class="image fit thumb"><img src="/assets/images/filtered_svm - poly - cost 1000 - news - metadata.png" alt="" /></a>
			<h3>News: Publisher & Partisianship</h3>
			<p>  </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/svm - poly - cost 1000 - news - party.png" class="image fit thumb"><img src="/assets/images/svm - poly - cost 1000 - news - party.png" alt="" /></a>
			<h3>News: Partisian Affiliation</h3>
			<p> </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/svm - poly - cost 1000 - news - sentiment.png" class="image fit thumb"><img src="/assets/images/svm - poly - cost 1000 - news - sentiment.png" alt="" /></a>
			<h3>News: Sentiment</h3>
			<p></p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/svm - poly - cost 1000 - bills - sentiment.png" class="image fit thumb"><img src="/assets/images/svm - poly - cost 1000 - bills - sentiment.png" alt="" /></a>
			<h3>Climate Bills: Sentiment</h3>
			<p></p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/svm - poly - cost 1000 - bills - sponsor party.png" class="image fit thumb"><img src="/assets/images/svm - poly - cost 1000 - bills - sponsor party.png" alt="" /></a>
			<h3>Climate Bills: Sponsor Affiliation</h3>
			<p></p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/filtered_svm - poly - cost 1000 - bills - sponsor state.png" class="image fit thumb"><img src="/assets/images/filtered_svm - poly - cost 1000 - bills - sponsor state.png" alt="" /></a>
			<h3>Climate Bills: Sponsor State</h3>
			<p> </p>
		</article>
	</div>
</section>

 <a id="conclusions"></a>
### Conclusions
