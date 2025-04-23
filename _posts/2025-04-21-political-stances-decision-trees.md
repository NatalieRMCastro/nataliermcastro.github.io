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

### Data Preparation

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

### Method

### Results

#### Gini Model Evaluation
<table>
<thead>
<tr><th>                                                   </th><th style="text-align: center;">  Accuracy </th><th style="text-align: center;"> Precision </th><th style="text-align: center;"> Recall </th></tr>
</thead>
<tbody>
<tr><td>News Headlines: Partisian Affiliation              </td><td style="text-align: center;">   0.585   </td><td style="text-align: center;">   0.621   </td><td style="text-align: center;"> 0.607  </td></tr>
<tr><td>News Headlines: Publisher                          </td><td style="text-align: center;">   0.008   </td><td style="text-align: center;">     0     </td><td style="text-align: center;"> 0.011  </td></tr>
<tr><td>News Headlines: Publisher and Partisian Affiliation</td><td style="text-align: center;">   0.008   </td><td style="text-align: center;">     0     </td><td style="text-align: center;"> 0.008  </td></tr>
<tr><td>Climate Bills: Sponsor Affiliation                 </td><td style="text-align: center;">   0.454   </td><td style="text-align: center;">   0.44    </td><td style="text-align: center;"> 0.478  </td></tr>
<tr><td>Climate Bills: Sponsor State                       </td><td style="text-align: center;">   0.033   </td><td style="text-align: center;">   0.019   </td><td style="text-align: center;"> 0.019  </td></tr>
<tr><td>Climate Bills: Metadata                            </td><td style="text-align: center;">   0.029   </td><td style="text-align: center;">     0     </td><td style="text-align: center;"> 0.005  </td></tr>
<tr><td>Climate Bills: Bill Type                           </td><td style="text-align: center;">   0.09    </td><td style="text-align: center;">   0.353   </td><td style="text-align: center;"> 0.305  </td></tr>
<tr><td>Climate Bills: Hearing Committee                   </td><td style="text-align: center;">   0.022   </td><td style="text-align: center;">   0.002   </td><td style="text-align: center;"> 0.014  </td></tr>
</tbody>
</table>

#### Entropy Model Evaluation
<table>
<thead>
<tr><th>                                                   </th><th style="text-align: center;">  Accuracy </th><th style="text-align: center;"> Precision </th><th style="text-align: center;"> Recall </th></tr>
</thead>
<tbody>
<tr><td>News Headlines: Partisian Affiliation              </td><td style="text-align: center;">   0.585   </td><td style="text-align: center;">   0.621   </td><td style="text-align: center;"> 0.607  </td></tr>
<tr><td>News Headlines: Publisher                          </td><td style="text-align: center;">   0.004   </td><td style="text-align: center;">   0.002   </td><td style="text-align: center;">  0.01  </td></tr>
<tr><td>News Headlines: Publisher and Partisian Affiliation</td><td style="text-align: center;">   0.004   </td><td style="text-align: center;">     0     </td><td style="text-align: center;"> 0.008  </td></tr>
<tr><td>Climate Bills: Sponsor Affiliation                 </td><td style="text-align: center;">   0.434   </td><td style="text-align: center;">   0.396   </td><td style="text-align: center;"> 0.457  </td></tr>
<tr><td>Climate Bills: Sponsor State                       </td><td style="text-align: center;">   0.034   </td><td style="text-align: center;">   0.035   </td><td style="text-align: center;"> 0.049  </td></tr>
<tr><td>Climate Bills: Metadata                            </td><td style="text-align: center;">   0.017   </td><td style="text-align: center;">   0.009   </td><td style="text-align: center;"> 0.036  </td></tr>
<tr><td>Climate Bills: Bill Type                           </td><td style="text-align: center;">   0.461   </td><td style="text-align: center;">   0.309   </td><td style="text-align: center;"> 0.293  </td></tr>
<tr><td>Climate Bills: Hearing Committee                   </td><td style="text-align: center;">   0.009   </td><td style="text-align: center;">   0.012   </td><td style="text-align: center;"> 0.047  </td></tr>
</tbody>
</table>


### Conclusions
