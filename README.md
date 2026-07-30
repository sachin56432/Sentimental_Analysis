# Sentimental_Analysis

<p>Sentiment analysis is the computational study of the sentiments, opinions, attitudes, and 
emotions of people expressed in written language. It is one of the most active research areas 
in natural language processing and data mining in recent years. Its popularity is mainly due 
to two reasons. First, it has a wide range of applications because opinions are central to 
almost all human activities and are key influencers of our behavior. Whenever we need to 
decide, we tend to hear other’s opinion. Second, it presents many challenging research 
problems, which never had been attempted before the year 2000. One of the major reasons 
for the lack of study earlier is that there was way too little opinionated text in digital form. 
Hence, it is not really a surprise that the inception and the rapid growth of the field coincide 
with those of the social media on the web. In fact, the research has also spread out of 
computer science to management sciences due to its importance to business and a society. In 
this talk, we will start with the discussion of the mainstream sentiment analysis research and 
then move on to describe some recent work on modelling comments, discussions and 
debates, which give an overview of a whole different kind of analysis of sentiments and 
opinions</p>

ML pipeline: data preprocessing → feature engineering → handling imbalance → model training → evaluation → interpretation.

NLTK provides

Tokenization
Stopword removal
Lemmatization
POS tagging
SentiWordNet

What is TF-IDF?

TF-IDF converts text into numbers.
TF = frequency of a word in one document.
IDF = importance of that word across all documents.
Common words get lower weight.
Rare but informative words get higher weight.

Why use both unigrams and bigrams?
To capture both individual words and short phrases, improving context understanding.

Why was class imbalance a problem?

Example

Positive
18000 reviews
Negative
4000 reviews
A model could predict everything as positive and still achieve high accuracy, so imbalance must be addressed.
