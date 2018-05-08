# public-understanding-of-science

The main idea of this project is to assess the public understanding of research by considering the social media discussion about the research

--Data Collection: Data is collected from Altmetrics (https://www.altmetric.com/) where data is in the form of json files. Each json file will have the information like the details about the article and social media discussion about that article.

--Data Cleaning: Here the major concern of data munging was that data was in string format. Data cleaning is done using python.

--Building the model

  --Lexicon: A 1xN vector with all the unique words in the documents. Common words like "is, a, the, it, etc" were ignored while building the lexiconAll the words in the lexicon were in lemmatized form ie words like run, running, runs were considered as run.

--understanding.py: Here the final model is built using gensim library and results are calculated.

--projectbestresult.PNG: This file contains the result of this project. 
  
 
