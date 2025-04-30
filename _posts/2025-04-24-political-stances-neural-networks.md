---
layout: post
title: "Political Stances: Neural Networks"
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

What is the advantage of using a neural network to classify climate bills and news headlines? It uses a more holisitic approach to 'read' the text, and in some cases (when paramaratized correctly) may outpreform traditional machine learning methods. I continue to pose the same questions in this section focusing on political polarizaiton through the lens of climate. In this notebook, I will be focusing on the partisian affiliation of both news and climate bills in addition to the sponsor state label.

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

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/neural net - start data.png" alt="Labeled Proposed Climate Bill Data Headed Dataframe"  /></span> 
			</div>
		</div>
	</div>
</section>
Neural network needs text data and does not rely on counts from CountVectorizer and TF-IDF. To generate this, the original data was used that is not split and preserves the original text. This data may be found at the [HuggingFace Repository](https://huggingface.co/datasets/nataliecastro/climate-news/blob/main/News%20Data%20Cleaned.csv) for this project. While the neural network takes text data it needs to still be processed in order to have a clean training environment that is tokenized. To do so, a function was utilized to preprocess the raw text data. It is used to separate the text, lemmatize the words, and then remove any special tokens. This returns a string of the lemmatized and clean texts. 



```python
def preprocess(text : str) -> list:
    ## first, converting the string into a list format
    text_ = text.split()
    
    text_storage = []
    ## using the wordnet lemmatizer to lemmatize the words
    for word in text_:
        lemma = lemmatizer.lemmatize(word)
        text_storage.append(lemma)
        
    ## removing all of the punctuation, special characters, digits, and trailing spaces using RegEx
    text_for_cleaning = ' '.join(text_storage)
    clean_text = re.sub('[!@#$%^&*()_+\'",.?*-+:;<>~`0-9]',' ',text_for_cleaning)
    stripped_text = clean_text.strip()
    
    ##splitting the string back into a list 
    preprocessed_text = stripped_text.split()
    
    ## returning the the final processed text
    return (preprocessed_text)
```

<section>
	<div class="box alt">
		<div class="row gtr-50 gtr-uniform">
			<div class="col-12"><span class="image fit"><img src="/assets/images/neural net - preprocessed documents.png" alt="Labeled News Headline Data Headed Dataframe"  /></span> 
			</div>
		</div>
	</div>
</section>

After processing both the bills and the news data, the sequences have to be embedded. This changes the word tokens into numerical tokens based on the order of the text. Each word in the vocabulary is assigned to a number, which is then organized into vectors which are representative of the sentence. For example the sentence "Trump's first day in office" would then become the vector [4, 19, 32, 21, 7]. To generate a token to index vector a dictionary was used to mangage the conversions. The index zero was reserved for a '[PAD]' token which is used to help truncate the input sequences. In order to preform matrix multiplication and shape the The news tokens were set to have a truncated value of 200 and an input label of 512. 

<section>
		<p><span class="image left"><img src="/assets/images/neural net -tokenized data.png" alt="A tokenized and padded sequence."  /></span> The sequences were padded and tokenized into numbers in order to be accurately passed to the Tensors. This workflow is common when training a neural network and is an accepted practice to pad the input sequences. </p>
	</section>



The data was split into train, test, and validation paritions. The threshold set for training was 80% with 10% reserved for testing and validation (each). The train_test_split from SciKit Learn ws utilized to generate these paritions. Using [PyTorch's Tensor](https://pytorch.org/docs/stable/tensors.html) Object, the vectorized data was transformed for better training. [Tensor Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) was also utilized in tandem with the DataLoader. 


#### Labels
The Label Encoder from [SciKit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) was utilized to encode the labels. This function also allows for reverse encoding, so the development of visualizations are rooted in text labels - not numerical labels. This is particularly important non-binary labels, such as the Sponsor State, which may have multiple categories.

```python
y_label_train_news_party = label_encoder.fit_transform(y_label_train_news)
```

<a id="method"></a>
### Method
The Neural Network was hand coded. As introduced earlier, the network requires a few different layers. The data preparation discussed above prepared the original input layer. 


<a id="evaluation"></a>
### Evaluating the Neural Network

To evaluate the model, it is asked to predict (with no altered gradient - or it would result to training on the test set!) the testing and validation sets. The predictions are then stored and the evaluation metrics are computer. The evaluation metrics are discussed at length 

``` python
def evaluation(model, test_loader):
    for batch in test_loader:
        test_inputs, test_targets = batch
        with torch.no_grad():
            test_outputs = model(test_inputs)
        
        predictions = test_outputs.view(-1)
        predictions = torch.tensor([1 if x >= 0.5 else 0 for x in predictions])
    
        accuracy = accuracy_score(test_targets, predictions)
        precision = precision_score(test_targets, predictions)
        recall = recall_score(test_targets, predictions)
        f1 = f1_score(test_targets, predictions)
        
        print(f'accuracy: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'f1: {f1}')
        

    return (predictions, test_targets, accuracy, precision, recall, f1 )
```

<a id="conclusion"></a>
### Conclusions
