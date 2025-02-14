---
layout: post
title: "Political Stances: Data"
categories: projects
published: true
in_feed: false
---

Data came from four sources. The NewsAPI, to collect information about public media exposure to partisan views of climate change. A combination of the Congress API, to generate metadata about all proposed Federal climate change bill information since the 93rd congress (1973), and then web scraping the Library of Congress to collect the bill text. Finally, auxiliary information from both Democrat and Republican Party Platforms about language used in the most recent presidential election. If you are interested in seeing exactly how I collected the data, you are welcome to reference [this page, which links to the MarkDown]('https://nataliermcastro.github.io/projects/2025/02/13/climate-data-cleaning.html') version of my notebook or my [GitHub repository]('https://github.com/NatalieRMCastro/climate-policy/blob/main/0.%20Data%20Collection%20-%20for%20website.ipynb') where you can download the IPYNB file.

### NewsAPI
The [NewsAPI](https://newsapi.org/) is an easy to access API that provides key word queries to search for data in their repository of millions of articles. To understand partisan polarization in regard to climate change this is especially valuable, as many studies have begun to quantify the partisan lean of news websites. The keywords “climate change” and either “republican” or “democrat” were used to search NewsAPI. Returned from the query was the source, author, title of the article, description, url, and published date. A total of 1,559 news articles were found that have the keywords “Republican” and “climate change” and a total of 671 news articles were found for “Democrat” – this totals to 2,230 total articles collected.

### Congess.gov API



### Party Platform Declarations 

<section>
	<p><span class="image fit"><img src="/assets/images/DNC Party Platform - Raw Text.png" alt=""  /></span> </p>
	<p><span class="image fit"><img src="/assets/images/GOP Party Platform - Raw Text.png" alt="" /></span> </p>
</section>


<section>
---
Bibliography

</section>
