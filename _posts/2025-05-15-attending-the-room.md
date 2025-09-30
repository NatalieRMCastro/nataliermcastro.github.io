---
layout: post
title: "Attending to the Room: Sentiment Classification with Masked Language Modeling"
categories: projects
published: true
in_feed: true
---

Sentiment Classification is a common NLP task with multiple applications. Its use for bias detection and content moderation makes illustrates the importance of developing robust tools to contribute to the safety of online narratives. Social media is a common recipient of these tools, and thus, the cental focus of the work presented. Our final project for a Natural Language Class at CU Boulder was the creation of sentiment classification with masked language modeling. I collaborated with [Mukun Mahesan](https://www.linkedin.com/in/mukund-mahesan/) to complete this project.

Method: Using the Sentiment 140 dataset a masked transformer was developed with the task of positive, negative, or neutral classification. Over 1,500,000 tweets com- pose this dataset and were used to train the masked transformer. The architecture of the transformer utilized causal attention with masked inputs over a variety of training parameters and epochs. A classifier head was then applied to complete the task. 

Result: Our best performing model achieved an F1 score of 0.81 and a validation accuracy of 0.81, which shows us that even a compact transformer architecture with causal attention can effectively capture sentiment patterns in short social media text. 

Conclusion: Our findings support the viabil- ity of lightweight transformer models for sentiment analysis tasks and also highlight their potential as efficient tools for scalable content moderation and bias detection in online environments.

*Note:* The below manuscript is NOT peer reviewed, it was our final write up and proof of concept for our sentiment classification architecture. The GitHub repository can be viewed in our shared repository [csci-5832](https://github.com/NatalieRMCastro/csci-5832). 

Our report from this work can be found at this page: [Attending to the Room: Sentiment Classification with Masked Language Modeling](https://drive.google.com/file/d/10qF-KFVRDyMRDti7UDVA645ZutW_wePJ/view?usp=sharing), or by clicking the teaser image below!

<section>
  <a href="https://drive.google.com/file/d/10qF-KFVRDyMRDti7UDVA645ZutW_wePJ/view?usp=sharing" target="https://drive.google.com/file/d/10qF-KFVRDyMRDti7UDVA645ZutW_wePJ/view?usp=sharing">
    <div class="box alt">
      <div class="row gtr-50 gtr-uniform">
        <div class="col-12">
          <span class="image fit">
            <img src="/assets/images/attending to the room teaser.png" alt="A screenshot of the first page of our PDF document." />
          </span>
          <figcaption>click to read full PDF.</figcaption>
        </div>
      </div>
    </div>
  </a>
</section>





