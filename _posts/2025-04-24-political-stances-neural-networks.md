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

A [Neural Network](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)), is inspired (loosely) by the neurological structure of the neurons in our brains. It mirrors the dendritic branches through the vectorized input layer of the text. The branches then pass in to the nucleas, where the input layer is transformed into an output signal, in the case of a neural network this is represented as *y*, but in the case of the brain it may be represented as an itch, yell, or laugh. The structure of a neural network unit is roughly as follows: the network takes an input layer (a [sentence](https://en.wikipedia.org/wiki/Sentence_(linguistics))), converts it to numbers so it is computer readable ([vectorization](https://en.wikipedia.org/wiki/Word_embedding)), then generates [weights](https://en.wikipedia.org/wiki/Weighting) (learned through backpropoagation of the neural network), preforms a transformation to interpret the input ([sigmoid](https://en.wikipedia.org/wiki/Activation_function) or some other activation function), and then normalizes the output into something bounded (softmax). This is then repeated multipls times to train multi-layered network, and has formed the later intution for more complicated techniques like transformers.

What is the advantage of using a neural network to classify climate bills and news headlines? It uses a more holisitic approach to 'read' the text, and in some cases (when paramaratized correctly) may outpreform traditional machine learning methods. I continue to pose the same questions in this section focusing on political polarizaiton through the lens of climate.

<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Blausen_0657_MultipolarNeuron.png" class="image fit thumb"><img src="/assets/images/Blausen_0657_MultipolarNeuron.png" alt="" /></a>
			<h3>McCulloch-Pitts neuron (circa 1943)</h3>
			<p></p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/Neural Network Unit - Jurafsky and Martin.png" class="image fit thumb"><img src="/assets/images/Neural Network Unit - Jurafsky and Martin.png" alt="" /></a>
			<h3>Neural Network Unit</h3>
			<p></p>
		</article>
	</div>
</section>
(Neuron Citaton: [BruceBlaus](https://commons.wikimedia.org/w/index.php?curid=28761830), Neural Network Citation and Comparison Structure: [Jurafsky and Martin (2025)](https://web.stanford.edu/~jurafsky/slp3/7.pdf))

**Table of Contents**
- [Data Preparation](#data-prep)
- [Method](#method)
- [Evaluating the Neural Network](#evaluation)
- [Conclusions](#conclusion)
  
---
<a id="data-prep"></a>
### Data Preparation

#### Labels



<a id="method"></a>
### Method

<a id="evaluation"></a>
### Evaluating the Neural Network

<a id="conclusion"></a>
### Conclusions
