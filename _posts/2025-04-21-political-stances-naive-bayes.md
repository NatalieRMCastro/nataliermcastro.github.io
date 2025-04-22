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

### Method

### Results
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

### Conclusions
