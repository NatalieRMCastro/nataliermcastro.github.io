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
