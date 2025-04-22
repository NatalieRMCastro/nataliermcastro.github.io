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
<tr><td>News Headlines: Partisian Affiliation              </td><td style="text-align: center;">   0.553   </td><td style="text-align: center;">   0.604   </td><td style="text-align: center;"> 0.569  </td></tr>
<tr><td>News Headlines: Publisher                          </td><td style="text-align: center;">   0.138   </td><td style="text-align: center;">   0.011   </td><td style="text-align: center;"> 0.015  </td></tr>
<tr><td>News Headlines: Publisher and Partisian Affiliation</td><td style="text-align: center;">   0.179   </td><td style="text-align: center;">   0.002   </td><td style="text-align: center;"> 0.008  </td></tr>
<tr><td>Climate Bills: Sponsor Affiliation                 </td><td style="text-align: center;">   0.56    </td><td style="text-align: center;">   0.457   </td><td style="text-align: center;"> 0.457  </td></tr>
<tr><td>Climate Bills: Sponsor State                       </td><td style="text-align: center;">   0.02    </td><td style="text-align: center;">   0.086   </td><td style="text-align: center;"> 0.057  </td></tr>
<tr><td>Climate Bills: Metadata                            </td><td style="text-align: center;">   0.016   </td><td style="text-align: center;">     0     </td><td style="text-align: center;"> 0.005  </td></tr>
<tr><td>Climate Bills: Bill Type                           </td><td style="text-align: center;">   0.535   </td><td style="text-align: center;">   0.468   </td><td style="text-align: center;"> 0.457  </td></tr>
<tr><td>Climate Bills: Hearing Committee                   </td><td style="text-align: center;">   0.194   </td><td style="text-align: center;">   0.001   </td><td style="text-align: center;"> 0.007  </td></tr>
</tbody>
</table>

#### Entropy Model Evaluation
<table>
<thead>
<tr><th>                                                   </th><th style="text-align: center;">  Accuracy </th><th style="text-align: center;"> Precision </th><th style="text-align: center;"> Recall </th></tr>
</thead>
<tbody>
<tr><td>News Headlines: Partisian Affiliation              </td><td style="text-align: center;">   0.545   </td><td style="text-align: center;">   0.595   </td><td style="text-align: center;"> 0.562  </td></tr>
<tr><td>News Headlines: Publisher                          </td><td style="text-align: center;">   0.024   </td><td style="text-align: center;">   0.024   </td><td style="text-align: center;"> 0.019  </td></tr>
<tr><td>News Headlines: Publisher and Partisian Affiliation</td><td style="text-align: center;">   0.008   </td><td style="text-align: center;">   0.004   </td><td style="text-align: center;"> 0.007  </td></tr>
<tr><td>Climate Bills: Sponsor Affiliation                 </td><td style="text-align: center;">   0.56    </td><td style="text-align: center;">   0.398   </td><td style="text-align: center;"> 0.389  </td></tr>
<tr><td>Climate Bills: Sponsor State                       </td><td style="text-align: center;">   0.039   </td><td style="text-align: center;">   0.091   </td><td style="text-align: center;"> 0.082  </td></tr>
<tr><td>Climate Bills: Metadata                            </td><td style="text-align: center;">   0.034   </td><td style="text-align: center;">   0.058   </td><td style="text-align: center;"> 0.053  </td></tr>
<tr><td>Climate Bills: Bill Type                           </td><td style="text-align: center;">   0.69    </td><td style="text-align: center;">   0.505   </td><td style="text-align: center;">  0.52  </td></tr>
<tr><td>Climate Bills: Hearing Committee                   </td><td style="text-align: center;">   0.035   </td><td style="text-align: center;">   0.043   </td><td style="text-align: center;"> 0.064  </td></tr>
</tbody>
</table>


### Conclusions
