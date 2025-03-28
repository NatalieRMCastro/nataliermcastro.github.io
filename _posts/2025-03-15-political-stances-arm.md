---
layout: post
title: "Political Stances: Association Rule Mining"
categories: projects
published: true
in_feed: false
---

There are multiple associtations within natural language. Texts we generate are embedded with nuance and word senses and further analysis may provide insight into the nuance of word co-occurance. For example, consider the word "climate" and words which may trail after it. "Change" may be a suitable word to follow "climate". But what are other possibilities of words that may follow, and in a specific text corpus like news and introduced climate bills at the federal level. Using Association Rule Mining, a systematic analysis may be conducted about what words are most likely to follow in suite from the next. 

**Table of Contents**
- [Data Preparation](#Method)
- [Association Rules for Climate, Government, Republican, and Democrat](#ARM_Viz)
- [15 Most Meaningful Rules for Support, Lift, and Confidence](#top_15)
---

 <a id="Method"></a>
### Data Preparation
Association Rule Mining (ARM) is the task of understanding interactions through transaction data. Transaction data represents a collection of documents where each row is a document. The columns then illustrate the words which compose each row, or transaction. Take the sentence ‘The ocean is blue and the ocean is big’, the row would compose of the words ‘ocean’, ‘blue’, ‘big’. Words like stop words are removed from the data because they are frequent, and likely to co-occur with most of the words because of their syntactical functions. In addition to filtering out the stopwords, pre-processing data for ARM does not rely on frequencies, just the presence of the words. In short, ARM takes transaction data, which is composed of the unique meaningful words in each document. To view how basket data was generated, reference the code provided [here](https://nataliermcastro.github.io/projects/2025/03/17/political-stances-basket-data.html), the data is stored at this [HuggingFace Repository](https://huggingface.co/datasets/nataliecastro/climate-corpus-basket-data/blob/main/Basket%20Data.csv). A snippet of the transaction data is illustrated below. This continues on with however many documents compose the total corpus. The transactions illustrated in the screenshot are short because they are news headlines, however, further down in the corpus are the climate bills which extend beyond just a few words.

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/transaction data.png" alt="An example of thhe CSV basket data. Each row is a comma separated list of the meaningful words."  /></span> 
			</div>
		</div>
	</div>
</section>


To preform ARM, R was utilized. The full code used to describe the visuals provided below are presented in [ARM - CODE](https://nataliermcastro.github.io/projects/2025/03/15/political-stances-arm-code.html) or provided at my [GitHub](https://github.com/NatalieRMCastro/climate-policy/blob/main/3.%20Association%20Rule%20Mining.R). The R file and workspace may also be downlaoaded at the provided link.

<a id="ARM_Viz"></a>
### Association Rules for Climate, Government, Republican, and Democrat
Association role mining was identified based on words which co-occur. The large amount of data originally presented to the model generated over 11,000 rules. These rules alone were far reaching, and after filtering for lift validity did not provide much shape to the data. To mediate this, words were selected it order to identify what was the most meaningfully co-occurring in certain contexts. To do so the words “climate”, “Democrat”, “Republican”, “Science”, “Tribal”, “Government”. The most confident rules were then selected to be visualized. First, the rules for climate and government are described to illustrate the embedding space. Then, the partisian representations for “Democrat” and “Republican” are both discussed.

The figures illustrated below are generated using lemmatized data. This allows for a merge of multiple word forms with the same word senses to be illustrated visually. If you see words that are incomplete like 'extren' this may be representing "externally", or "external"
<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Climate ARM.png" class="image fit thumb"><img src="/assets/images/Climate ARM.png" alt="" /></a>
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

Co-occurrence rules generated from the word “tribal” are illustrative of the specific environmental concerns which originate from Indigenous land. The term ‘tribal’ was selected here in opposition to ‘indigenous’ or ‘First Nation’ because ‘tribal’ is the language utilized by the federal government, thus would be more representative of the climate policies introduced on behalf or facilitated by Indigenous peoples. The words which are illustrative of the specific concerns on territories are “fish”, “basin”, “wildfire”, and “industri-“. Further illustrated are the particular entities which co-occur with tribal: “member”, “indian”, “scientific”, “district”, “private”, “non-profit”, and ”head”. Due to the language constraints illustrated above, it may be expected that the word “tribal” may not be as frequent in media coverage as it is in federal bills. For this reason, the entities described here may be illustrative of the actors most involved in Indigenous concern about policy. 

Strong action oriented phrases are also utilized which illustrate the kinds of mitigation measures discussed in Indigenous climate policy. The most prevalent rules are: “restore”, “converse”, “promote”, “create”, “strategy”, “enforce”, “prevent”, “perform, “and “mitigate”. These words are more specific than that illustrated in either “science” or “government.” This may be a result of the language utilized, however, an overwhelming proportion of the data distribution belongs to Republican and Democrat focused climate bills, and strong action words may be more illustrative of the *kinds* of efforts or bills that each side is proposing for law. 

For the term ‘Republican” the most confident rules which mention climate related activities (tangentially) are: “wildfire” and  “land”. In comparison to the rules generated from Democrat co-occurrences words “climate”, “los” (also referring to wildfires), “green”, “climat”. It should be noted that while both presidents are illustrated in the climate embedding space, only ‘Republican” is present. The Republican associations are more concerned with fiscal resources, policy shifts and governing entities, or immigration. This may be illustrated through rules like “immigration”, “anti”, or “vote”.

Rules generated from ‘Democrat’ co-occurrence illustrate more emotion that illustrate concern. These rules may be illustrated with “injustice” , “justice”, “safe” or “benefit”. These words, or words with similar word senses were not illustrated in the Republican affiliated rules with the most confidence. This may demonstrate the kinds of language used to construct media and policy about the Democrats are inherently more emotional and concerned with justice. In addition, the co-occurrence with words that *may* be related to climate change is higher. Words that are used to illustrate this are: “green”, “waste”, “disaster”, “hazard”, and “extreme”. These words also co-occur alongside of marginalized groups such as “gender”, “territories”, “indigenous”, and “color”. The combination of the rules presented show a fundamentally different media and policy space. 

The language used to construct narratives about climate change and climate related concerns are politically siloed. Co-occureances most likely in the Republican group are more aligned the words that co-occurred with the word “government”. Neither of the co-occurrence spaces are closely aligned with that of the “scientific” or “tribal” associations.

<a id="top_15"></a>
### 15 Most Meaningful Rules for Support, Lift, and Confidence


