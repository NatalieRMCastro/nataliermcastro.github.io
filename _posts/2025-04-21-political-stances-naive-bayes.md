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

<a id="results-model-evaluation"></a>
### Assessing the Validity of the Multinomial Naïve Bayes

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


<a id="results-news-data"></a>
### Interpreting Features About Climate Bills Naively
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

https://www.house.gov/the-house-explained/the-legislative-process/bills-resolutions#:~:text=Concurrent%20Resolutions&text=A%20concurrent%20resolution%20originating%20in,the%20Secretary%20of%20the%20Senate.


### Conclusions

---
### Bibliography
Grimmer, J., Roberts, M. E., & Stewart, B. M. (2022). Text as data: A new framework for machine learning and the social sciences. Princeton University Press.
