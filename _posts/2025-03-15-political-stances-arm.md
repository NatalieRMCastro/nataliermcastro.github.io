---
layout: post
title: "Political Stances: Association Rule Mining"
categories: projects
published: true
in_feed: false
---

Introduction Text

**Table of Contents**
- [Method](#Method)
- [Party Platform Declarations](#PPD)
  
---

 <a id="Method"></a>
### Method
Association Rule Mining (ARM) is the task of understanding interactions through transaction data. Transaction data represents a collection of documents where each row is a document. The columns then illustrate the words which compose each row, or transaction. Take the sentence ‘The ocean is blue and the ocean is big’, the row would compose of the words ‘ocean’, ‘blue’, ‘big’. Words like stop words are removed from the data because they are frequent, and likely to co-occur with most of the words because of their syntactical functions. In addition to filtering out the stopwords, pre-processing data for ARM does not rely on frequencies, just the presence of the words. In short, ARM takes transaction data, which is composed of the unique meaningful words in each document.

To preform ARM, R was utilized. The full code used to describe the visuals provided below are presented in [ARM - CODE](https://nataliermcastro.github.io/projects/2025/03/15/political-stances-arm-code.html). The R file and workspace may also be downlaoaded at the provided link.

<a id="ARM_Viz"></a>
#### Associaiton Rules for Climate, Government, Republican, and Democrat
The figures illustrated below are generated using lemmatized data. This allows for a merge of multiple word forms with the same word senses to be illustrated visually. If you see words that are incomplete like 'extren' this may be representing "externally", or "external"
<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Climate ARM.png" class="image fit thumb"><img src="/assets/images/Climate ARM alt="" /></a>
			<h3>Climate Associations</h3>
			<p>The highgest supported accosciations when climate is present are the words "weather","change", and "president"</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Government Association Plot.png" class="image fit thumb"><img src="/assets/images/Government Association Plot.png" alt="" /></a>
			<h3>Government Associations</h3>
			<p>Some associations with multiple co-occurances are "government, contribution, cause", "governmnet, prepare, maintain" and "government, contract, science".</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Republican Associations.png" class="image fit thumb"><img src="/assets/images/Republican Associations.png" alt="" /></a>
			<h3>Republican Associations</h3>
			<p>Republican ARM rules highlight more concerns about immigration and climate change.</p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Democrat Association Rules.png" class="image fit thumb"><img src="/assets/images/Democrat Association Rules.png" alt="" /></a>
			<h3>GOP Party Platform - Clean Text</h3>
			<p>The Democrat ARM rules foucs more on words describing concen, like 'disproportion', 'fair', and 'threat'.</p>
		</article>
	</div>
</section>
