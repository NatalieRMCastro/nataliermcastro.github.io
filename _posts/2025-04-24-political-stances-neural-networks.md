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
- [Classifying News Headline Partisian Affiliations](#evaluation-news-headlines)
- [Classifying Climate Bill Sponsor Partisian Affiliations](#evaluation-bills-party)
- [Classifying Climate Bill Sponsor State](#evaluation-bills-state)
- [Results](#results)
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

The data was split into train, test, and validation paritions. The threshold set for training was 80% with 10% reserved for testing and validation (each). The train_test_split from SciKit Learn ws utilized to generate these paritions. Using [PyTorch's Tensor](https://pytorch.org/docs/stable/tensors.html) Object, the vectorized data was transformed for better training. [Tensor Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) was also utilized in tandem with the DataLoader. To generate a training and testing set the code utilized the Label Encoded data (explained below) and an array of Tensor data. This reserved validation set will be held out until a model has been selected for good preformance on the train and testing sets. In addition, the labels for the news data are binary, however, this is not true for the climate bills. Each type of label for the climate bill (sponsor affiliation and sponsor state) were used to generate an individual training, testing, and validation split.

``` python
''' NEWS'''
X_train_news, X_test_and_val_news, y_label_train_news, y_label_test_and_val_news = train_test_split(truncated_sequences_int_news, labels_full_news_party, test_size=0.2, random_state=123)
X_test_news, X_val_news, y_test_label_news, y_val_label_news = train_test_split(X_test_and_val_news, y_label_test_and_val_news, test_size=0.5, random_state=42)
```
Keeping the training, testing, and validation sets completely separate are important. If this is not maintained, the model may be overfit, incorrect results are reported, or the application of the model on truly held out data will be incorrect. In addition, the training data, testing data, and later the validation data must originate from similar instances or the model will not preform as expected. For example, if a model is trained on niche academic representations of partisian affiliations, if the model was tested on Reddit slang and discourse about partisian affiliations, because the language used is fundamentally different - and with a different goal, it will not preform accurately.

#### Labels
The Label Encoder from [SciKit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) was utilized to encode the labels. This function also allows for reverse encoding, so the development of visualizations are rooted in text labels - not numerical labels. This is particularly important non-binary labels, such as the Sponsor State, which may have multiple categories. 

```python
''' NEWS LABELS '''
labels_full_news_party = news_data_raw['Party'].to_list()
label_encoder = LabelEncoder()
labels_full_news_party = label_encoder.fit_transform(labels_full_news_party)
```

As noted above, the Climate Bills have two labels of interest - partisian affiliation and sponsor state, and for these reasons, the labels were generated separately - one for each. These are kept separate and stored with their respective training, testing, or validation split, in order to assess the models preformance. 

<a id="method"></a>
### Method
<section>
    <div class="row">
        <div class="col-6 col-12-small">
            <ul class="actions" style="display: flex; gap: 10px; list-style: none; padding: 0;">
                <li><a href="https://nataliermcastro.github.io/projects/2025/04/21/political-stances-neural-networks-code.html" class="button fit small">View Code</a></li>
		<li><a href="https://github.com/NatalieRMCastro/climate-policy/blob/main/6.%20Naive%20Bayes.ipynb" class="button fit small">Visit GitHub Repository</a></li>
            </ul>
        </div>
    </div> 
</section> 
The Neural Network was hand coded. As introduced earlier, the network requires a few different layers. The data preparation discussed above prepared the original input layer or the embedding layer. The padding_idx token was noted earlier as 0, and the number of embeddings are the total vocabulary. 

```python
''' EMBEDDING LAYER: '''
 self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
```

The subsequnet layers are the first linear layer, then an activation, the second linear layer, and finally the sigmoid:

```python
''' MODEL ARCHITECTURE '''
self.linear1 = nn.Linear(input_size, hidden_size)
self.activation = torch.nn.ReLU()
self.linear2 = torch.nn.Linear(hidden_size, 1)
self.sigmoid = torch.nn.Sigmoid()
```

These layes are instantiated as functions, howver during the forward pass, where the model is 'learning' the current *x* or the token is passed into the calculations.
```python
''' FORWARD PASS '''
    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1) 
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        
        return (x)
```

An early stop function was also implemented in the case that the model was overfit on the data and during the validation of the model the loss was actually increasing. Finally, a training loop was developed in order to assess the validity of the model and compare between paramaterized models. To address the above note of non-binary labels for the Climate Bills, the training loop is able to alter the instantiation of the model in order to take multiple options for the labels.

The model had a 

<a id="evaluation"></a>
### Evaluating the Neural Network

To evaluate the model, it is asked to predict (with no altered gradient - or it would result to training on the test set!) the testing and validation sets. The predictions are then stored and the evaluation metrics are computer. The evaluation metrics are discussed at length in the [Naive Bayes](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-naive-bayes.html#method) section of the project page. In addition, a similar confusion matrix was genereated using the same code (also in Naive Bayes) to illustrate differences between the models.

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


<a id="evaluation-news-headlines"></a>
#### Classifying News Headline Partisian Affiliations:

The parameters and evaluation metrics are reported in the table below. For each different iteration of the parameters, they were recorded in order to select the most effective model for subsequent discussion. In addition, a lineplot was generated to identify the training versus validation loss per epoch. This type of metric provides insight into what model is the most effective. 

| Test Number | D  | H  | Batch Size | Epochs | Epochs Completed | Learning Rate | F1    | Accuracy | Precision | Recall |
|--------|----|----|-------------|--------|------------------|----------------|-------|----------|-----------|--------|
| 1 | 50 | 50 | 8 | 500 |  3 | 0.2 | 0.64 |  0.47  | 0.47  | 1  |
| 2 | 50 | 50 | 4 | 100 | 23 | 0.005 | 0.49 | 0.59 | 0.61 | 0.41 |
| 3 | 500 | 500 | 4 | 100 | 100 | 0.005 | 0.50 | 0.57 | 0.56 | 0.46 |
| 4 | 500 | 500 | 4 | 100 | 18 | 0.0005 | 0.53 | 0.59 | 0.59 | 0.48 |
| 5 | 500 | 500 | 8 | 100 | 100 | 0.0005 | 0.45 | 0.56 | 0.55 | 0.38 |
| 6 | 700 | 700 | 4 | 100 | 60 | 0.0001 | 0.48 | 0.58 | 0.59 | 0.41 |
| 7 | 500 | 500 | 8 | 500 | 255 | 0.0001 | 0.57 | 0.67 | 0.75 | 0.46 |
| 8 | 500 | 500 | 4 | 500 | 500 | 0.0001 | 0.47 | 0.62 | 0.70 | 0.35 |
| 9 | 500 | 500 | 4 | 1000 | 586 | 0.0001 | 0.50 | 0.62 | 0.66 | 0.41 |
| 10 | 500 | 500 | 16 | 1000 | 1000 | 0.0001 | 0.52 | 0.59 | 0.60 | 0.46 |

The model which preforemd the best was determined using F1, as it is a holisitc understanding of multiple different measures, this will also be determined using a look at the other evaluation metrics as well. Model 7 had the highest F1 score and the highest supporting evaluation scores. 

<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/neural net - news - best model loss.png" class="image fit thumb"><img src="/assets/images/neural net - news - best model loss.png" alt="" /></a>
			<h3>Test 7 Model Loss</h3>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/neural net - news - confusion matrix.png" class="image fit thumb"><img src="/assets/images/neural net - news - confusion matrix.png" alt="" /></a>
			<h3>Test 7 Model Loss</h3>
		</article>
	</div>
</section>

The few epochs completed by the model indicates that the early stop was utilized in order to prevent overfitting. However, such an early stop shows that the model is deeply struggling to learn on the text that it is provided with. After 250 epochs, the training loss was 0.55 and the validation loss was around 0.70. This was then validated using a visual confusion matrix. The model was more accurately able to predict the Democrat Labels, however, it had a strong tendancy to only predict Democrat Labels. The balance of the testing corpora was 43 Democrat labels and 39 Republican labels. This implies that the Neural Network was able to accurately classify news headlines and labels 57% of the time, with higher accuracy for Democrat mentions.

Implications of this accuracy suggests that there is some differences in language used to define partisian affiliations through news headlines, however, these distinctions are not always clear. This is a finding that was evidenced through [Naive Bayes](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-naive-bayes.html#results-news-data) and [Support Vector Machines](https://nataliermcastro.github.io/projects/2025/04/21/political-stances-svm.html#polynomial-classifications) as well. Naive Bayes had observed a similar misclassification pattern, predicting Democrat when the label was actually Republican. This is interesting because it is consisteny across models, and suggests that instead of one model preforming poorly there is a trend throughout different training and pre-procoessing techniques. The Neural Network achieved a higher accuracy rate of 67% (compared to 57%), higher precision 75% (NB: 57%), but lower recall of 46% (NB: 57%). In comparison to the SVM the Neural Network was able to preform better on all metrics except for recall. Further explorations of accuracy will be discussed using the helf out testing set (different than the validation and training) in the next section.

<a id="evaluation-bills-party"></a>
#### Classifying Climate Bill Sponsor Partisian Affiliations:

A similar iteration process to identify a potential best model was utilized for the climate bill sponsor partisian affiliation as well. It should be noted that for this data the labels are not binary as the climate bill sponsor may be independent as well. In addition, due to the truncation of the otherwise lengthy data, this may result in a more challenging prediction task. This is not so clearly evidenced in the evaluation table presented below. As demonstrated through the epochs completed, this data had a problem with overfitting - despite rather small learning rates. This caused the early stopper function to trigger, thus the training was cut short as soon as the function detected overfitting. It should be reiterated that the train test split for each model used the same data and the same random seed. This means that the data used to test the models were consistent across iterations.

While the models were able to achieve relatively high F1 scores (in addition to high evaluation scores all around), however the issue is that the models struggled to learn features of Independent affiliated sponsored bills. This presents a challenge, as other models such as Naive Bayes, were able to identify this distinction. Only one model (2) was able to identify the Independent label, but it did not predict correctly.

| Test Number | D   | H   | Batch Size | Epochs | Epochs Completed | Learning Rate | F1       | Accuracy | Precision | Recall |
|-------------|-----|-----|------------|--------|------------------|---------------|----------|----------|-----------|--------|
| 1           | 50  | 50  | 16         | 5      | 4                | 0.2           | 0.37     | 0.59     | 0.29      | 0.5    |
| 2           | 500 | 500 | 16         | 500    | 4                | 0.002         | 0.53     | 0.81     | 0.54      | 0.53   |
| 3           | 500 | 500 | 8          | 500    | 4                | 0.002         | 0.81     | 0.82     | 0.81      | 0.80   |
| 4           | 500 | 500 | 16         | 500    | 6                | 0.0009        | 0.82     | 0.83     | 0.82      | 0.82   |
| 5           | 250 | 250 | 16         | 500    | 5                | 0.002         | 0.82     | 0.83     | 0.82      | 0.82   |
| 6           | 500 | 500 | 32         | 500    | 29               | 0.0001        | 0.54     | 0.81     | 0.54      | 0.53   |
| 7           | 250 | 250 | 16         | 500    | 6                | 0.001         | 0.80     | 0.81     | 0.80      | 0.80   |
| 8           | 300 | 300 | 16         | 500    | 4                | 0.002         | 0.81     | 0.82     | 0.82      | 0.80   |
| 9           | 500 | 500 | 16         | 500    | 24               | 0.0001        | 0.79     | 0.80     | 0.79      | 0.79   |
| 10          | 499 | 499 | 31         | 500    | 21               | 0.00015       | 0.81     | 0.82     | 0.81      | 0.81   |


<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/neural net - bills party - best model loss.png" class="image fit thumb"><img src="/assets/images/neural net - bills party - best model loss.png" alt="" /></a>
			<h3>Test 10 Model Loss</h3>
			<p> This model was able to completed 20 epochs, but experienced overfitting around Epoch 15. The training loss decreased, with a clear elbow at about the fourth epoch, which was also when the validation departed from the continual decrease with the training losses. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/neural net - bills party - test 2 model loss.png" class="image fit thumb"><img src="/assets/images/neural net - bills party - test 2 model loss.png" alt="" /></a>
			<h3>Test 2 Model Loss</h3>
			<p> This model only completed five epochs. At epoch three, the model began to overfit as the validation loss began to increase and continue to do so. </p>
		</article>
	</div>
</section>


<section class="gallery">
	<div class="row">
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/neural net - bills party - confusion matrix.png" class="image fit thumb"><img src="/assets/images/neural net - bills party - confusion matrix.png" alt="" /></a>
			<h3>Test 10 Confusion Matrix</h3>
			<p> This model predicted a relatively accurate split between the Democrat and Republican affiliated bills - however, it was not able to learn the features of an Independent bill sponsor. Similar to other models, the model was more likely to predict Democrat than Republican. </p>
		</article>
		<article class="col-6 col-12-xsmall gallery-item">
			<a href="/assets/images/neural net - bills party - test 2 confusion matrix.png" class="image fit thumb"><img src="/assets/images/neural net - bills party - test 2 confusion matrix.png" alt="" /></a>
			<h3>Test 2 Confusion Matrix</h3>
			<p> This model preformed nearly identical to that of Model 10, however, it was able to learn about the Independent party. While it did misclassfy it during its prediction, it still understood that it was a viable label. </p>
		</article>
	</div>
</section>

These results make it challenging to compare and select a best model. Model 10 recieved a much higher F1 score (nearly 30% improved) and an increase in both precision and recall. However, it is clear that the model did not learn all of the features of the data. Yet, due to its evaluation metrics overall the 10th model was selected to be utilized in further comparisons.

<a id="evaluation-bills-state"></a>
#### Classifying Climate Bill Sponsor State:



| Test Number | D   | H   | Batch Size | Epochs | Epochs Completed | Learning Rate | F1       | Accuracy | Precision | Recall |
|-------------|-----|-----|------------|--------|------------------|---------------|----------|----------|-----------|--------|
| 1           | 50  | 50  | 16         | 5      | 4                | 0.2           | 0.004    | 0.113    | 0.002     | 0.020  |
| 2           | 500 | 500 | 32         | 500    | 68               | 0.0001        | 0.015    | 0.043    | 0.019     | 0.016  |
| 3           | 750 | 750 | 16         | 500    |  4               | 0.005         | 0.016    | 0.055    | 0.018     | 0.018  |
| 4           | 50  | 50  | 200        | 500    | 7                | 0.005         | 0.016    | 0.040    | 0.021     | 0.015  |
| 5           | 50  | 50  | 200        | 500    | 402              | 0.0005        | 0.025    | 0.071    | 0.024     | 0.029  |
| 6           | 100 | 100 | 250        | 500    | 331               | 0.00005       | 0.018    | 0.052    | 0.019     | 0.020  |
| 7           | 250 | 250 | 250        | 1500   |  99               | 0.00005       | 0.023    | 0.064    | 0.027     | 0.023  |



<a id="conclusion"></a>
### Conclusions
