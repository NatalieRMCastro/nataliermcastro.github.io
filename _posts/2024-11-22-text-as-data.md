---
layout: post
title: "Text As Data: Method Primer"
categories: projects
published: true
in_feed: true
---
This repository holds the Gists and the Notebook Tutorials for a variety of NLP methods. I created these tutorials and Gists for my MS independnet study based on the textbook Text As Data by Justin Grimmer, Margaret Roberts, and Brandon Stewart.

<section>
	<p>
	<ul class="actions">
		<li><a href="https://natalie-castro.notion.site/text-as-data?v=14156d64b28980ac913f000c7c247f5e" class="button fit small">Navigate to Notion Page</a></li>
	</ul>
  </p>
  
</section>

<section>
	<p>
	<ul class="actions">
		<li><a href="https://github.com/NatalieRMCastro/Text-As-Data/tree/main" class="button fit small">Navigate to GitHub</a></li>
	</ul>
  </p>
  
</section>


Natural Language Processing allows for text data to be parsed by a computer. It supports the inference process about a particular text or corpora to facilitate new understandings in research. The four principals outlined by Grimmer et al in Text As Data for “text as data” are: discovery, measurement, prediction, and causal inference.

In this page, I explore four different word counting methods. Word counting acts as discovery, it can indicate at what points a researchers should closely consider or what silences are originally appearing in the dataset. Discovery supports the exploration of a research question, and can also inform grounded theory analysis. K-Means, or word clustering helps to facilitate measurement by providing common co-occurrence probabilities. The large scope made possible by K-Means are important here because it can help to inform the descriptive models which come later on. Multinomial Language Models will generate trained classifiers based on prediction, which iteratively update based on new data. And finally, topic models can support causal inference to individuals in the digital humanities fields because of how the data can be engaged.

In combination, these computational tools provide broader context to the text and augment the human’s ability to conduct a distance reading. In the same breath, I must state the importance of hand validation and careful interpretation of the meanings that can be attributed to text data. Best worded by Grimmer et al, there are no true agnostic ways to interpret text. Both the researcher and the tools are encoded with bias. Key documentation must be included alongside with the ways of understanding the findings from text models.

I hope that these tutorials are helpful, for each method a tutorial is provided that walks through the basic steps of using that model and some visualizations. The Notion page provides information on the side along with the notebooks. For the Gist’s they are able to be taken as is, and just input your data. These are intended to help anyone engage with the method, regardless of prior Python experience! Data can be passed in using an Excel or CSV file through the Pandas DataFrame.

In all, text as data is a powerful tool that can support and facilitate a new kind of reading. The insights provided through these processes must be validated and shared with confidence that the quantification of the text is representative of its meaning. These methods provide an exploratory start to inductive data analysis, or can support a deductive theory at a larger scale. The ability provided through text as data methods also supports visualizing the texts in new wats, that the physical realm may have constrained. These applications are continuously evolving and something that can be applied into the future.
