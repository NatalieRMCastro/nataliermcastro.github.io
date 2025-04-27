---
layout: post
title: "Political Stances: Conclusion"
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

Climate change is an increasingly salient issue, it is ubiqutious in its impact, however highly contentious. Those who reside in the Global South are dispropotionaley impacted by the policy and response to climate change from large companies who reside in the United States - especially those who outsouce their labor. For many, climate change is a daily concern, it results in housing inequality, shift in agriculture, and other daily aspects of well being. Across the world, 2024 was the hottest year on record. 

While tclimate change issues have developed over time, its severity and polarization has increased. After 2009, a clear partisian divide was idenitfied in response to the shift in President Barack Obama's climate policies - and the Republican Media campagain against it. Something similar may be observed in the current news cycle. While President Joe Biden recently granted the largest amount of monies in support of climate preservation policies, President Donald Trump, leading a Republican Trifecta has now pledged to repeal the Green Deal, and more recently impact the funding of many scientific funded agencies.

Posed at the start of this project were two central research questions:
1. How is climate change represented in climate policy passed as the federal level?
2. What ways does news coverage reflect polarization in regards to climate change?

These questions were explored through the lens of over 3,000 proposed climate bills and resolutions at the United States federal level in addition to over 800 news articles tagged with paritisan affiliations. Three different categories of text mining methods were applied to generate the following conclusions. 

### Key Findings

**Text Clustering**  
Clustering illustsrated the prevalence of the Environmental Protection Agency across all bills. The semantic distance between the language used for climate bills were demonstrated to be close to multiple topics suhc as 'greenhouse gas and carbon dioxide', 'lead and contaminated drinking water', or 'threatened wildlife and engaged species'. This prevalence demonstrates that the EPA is synonymolus with different facets of climate policy. Specific partisian differences were not identified through K-Means. This method demonstated different nuance within climate policy. The most prevalent topics helped to illustrate trends within climate policy over time in the United Stated. This trend was identified in both K-Means clustering and Latent Dirichlet Allocation. The consistency across methods supports the accuracy of this finding. Through twenty different topics, it is clear that different concerns such as land, chemical regulations, and environmental factors illustrate the multiciplicty and different complexities of climate regulation. 

This trend is not illustrated as clearly for news sources. The largest clusters focused on President Donald Trump, with one cluster about the Los Angeles wildfire. Between the clusters, there was not as much overlap  between pnamed political entities and their partisian affiliation. For example, topics either focused soley on Donald Trump or Joe Biden. There was one exception to this trend with the topic name 'joe biden, job, trump, did'. An additional cluster generated through a different method illustrated paritisan overlap in two topics about the California wildfires and about offshore drilling. This was one of the smaller clusters, so it is a small exception to this. 

Association Rule Mining was used to identify was co-occurance between particular words. The words most associate with climate are to be expected, in addition, both political affiliations were illustrated in words that are associated. Government was not illustrated to be aligned with any of the named political parties. For Republican, the words associated highlight more concerns about immigration and climate change with resepect to the administration. The Democrat Associations were illustrated to be aligned with words about concern for welfare and other aspects about well-being.

**Machine Learning**  
Classification methods may illustrate how clear, or learnable, the language is used to delineate partisian ideology. A variety of methods (Support Vector Machines, Naive Bayes, Decision Trees, and Neural Networks) were used to identify how clear these language differences were and to understand the consistency across categories. Between methods, the accuracy was assessed using the same scale. This demonstrated both the strength of some methods in relation to others and the potential 'confusion' which the models may have learned.

Partisian Affiliations were often the most correct label predicted across all methods. In comparison, proposed climate bills were predicted more consistently than news articles. This finding may be in result to how the climate bills are structured. They include the bill type, bill sponsor state, bill sponsor affiliation, and hearing committee in the metadata. Using a classifier in this sense then in the future may be able to predict alignment of text in the future to a corpora of historic bills. A classifier was able to identify with relatively high accuracy the Bill Sponsor's state. Another method struggled to do so consistently. This may suggest that some methods were not the most appropriate to identify ideological diversions while others may 'learn' the features of the text better. Across all models, it was challenging to learn features about the Independnet Partisian affiliation because of the few amount of training features used to teach the model.

In addition to the labels generated using the text metadata, sentiment was derived to generate positive or negative labels. The overall distribution of language usage was positive to neutral. This suggests that what is assumed to be partisian affiliation is constructed through netural to positive language. Polarization is a nuanced linguistic and social construction, and this feature of it introduces further complexity in identifying polarization through text mining methods. Sentiment was predicted more accurately for the climate bills than that of the news headlines. This suggests that the language used in climate bills are definitive with regards to the strength of the emotion utilized. 

### Study Limitations

The findings generated here must be hedged with the natural limitations of these methods. Text mining methods are not actually reading the text, nor are they understanding the contextual or historical meaning embedded in the language. In addition, the language used in political text is illustrative of problematized norms and understandings about what is approporiate for bills and resolutions. For these reasons, the language used may be convulated and more challenging to read. This embedded text complicity and nuanced partisian language may not be identified through methods which read only the words which are present in the text. A second limitation may be the labeling strategy used to generate partisian affiliations from News Sources. The labeling scheme was generated using a query based technique provided by NewsAPI. A more appropriate scheme in the future may use word matching in order to labeling the data in response to the true content provided. 

<section>
		<p><span class="image right"><img src="/assets/images/TOMAYTO.png" alt="Two people in characicture drawings yelling at a can of tomato soup. One has the pronunciation 'tomayto' the other 'tomahto'."  /></span> Party Polartization feels nearly akin to screaming different things at the same topic (cliamte change). Political polarization dictates views, and concerns often about similar things. Presented in this project is a closer understanding of what tangible polarization may look like. </p>
	</section>
*Note: This image is AI Generated by the Canva AI for the purposes of this illustration.*

### Final Conclusions
This project illustrated how political polarization may be understood through proposed climate bills and news articles. Identified was a clear delineation between language used to speak to similar issues about climate change. In news headlines political parties were often not named together, suggestiong an opposing dialogue instead of a conjoined one. Clustering a large corpora of bills identified a temporal understanding of climate change bills with specific named language. This may be useful in the future to understand the placement and direction of political shifts in administration into the future. 
<section> <span class="image fit"><img src="/assets/images/climate-protest.jpg" alt="People holding up protest signs. One has a large print of the NOAA logo and says 'we save lives'."  /></span>  </section>
 
*Note: This image from a local Colorado news source. The faces of the people in the photo have been blurred and the image citation will not be provided in an attempt to protect the anonymity of the protestors.*
