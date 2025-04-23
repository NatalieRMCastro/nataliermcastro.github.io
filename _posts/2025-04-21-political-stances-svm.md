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

### Sentimenet Analaysis
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

### Results

### Conclusions
