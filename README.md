# Sentiment-Analysis-using-BERT
Sentiment Analysis model using DistilBERT and ML classification model.

In this notebook, we will use a pre-trained deep learning model to process some text. We will then use the output of that model to classify the text. The text is a list of 1000 amazon reviews from 2014 specifically on books and then we will classify each sentence as either speaking "positively" about its subject or "negatively".
Under the hood, the black-box model is made up of two models.

Firstly, DistilBERT processes the sentence and passes along some information extracted from it on to the next model.
The next model, a basic Logistic Regression model from scikit learn will take in the result of DistilBERT’s processing, and classify the sentence as either positive or negative (1 or 0, respectively). Before we can hand our sentences to BERT, we need to do some minimal processing to put them in the format it requires. Our first step is to tokenize the sentences break them up into words and subwords in the format accepted by DistillBERT.. We then want BERT to process our examples all at once (as one batch). It's just faster that way. For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-D array, rather than a list of lists (of different lengths). If we directly send padded to BERT, that would slightly confuse it. We need to create another variable to tell it to ignore (mask) the padding we've added when it's processing its input. 

The model() function runs our sentences through BERT. The results of the processing will be returned into last_hidden_states.
The way BERT does sentence classification is that it adds a token called [CLS] (for classification) at the beginning of every sentence. The output corresponding to that token can be thought of as an embedding for the entire sentence. So, we slice only the [CLS] part of the output.

Then fianlly, we split our dataset into a training set and testing set and evaluate how well does our model does in classifying sentences.


# Reference: 
http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
