
<p align="center">
  <img src="https://user-images.githubusercontent.com/28828162/66569500-9d9b4880-eb63-11e9-90f9-077eb0f7dadd.png" width="400"/> 
</p>


<b>Polarity.tn</b> is a web platform that detects the <b>language</b> and <b>sentiment polarity</b> of a text written in <b>arabic characters</b>. It differenciates arabic and tunisian dialect texts <b>using machine learning techniques</b>. 

### :speech_balloon: Language identification 

   These steps give an overview on the language identification pipeline of our script: 

1. Text Cleaning 
2. Construct a language classifier using supervised learning by training our arab and tunisian corpuses
3. Converting the documents to feature vectors using the BOW-tfidf method with character n-grams : max_df= 0.85, min_df=0.25, ngram_range = (1,4)
4. Training and testing our MultinomialNB model ( Parameters of BOW were chosen according to accuracy value & confusion matrix results (F1) after multiple tests) 


### :chart_with_upwards_trend: Sentiment analysis 

These steps give an overview on the sentiment analysis pipeline of our script: 

  1. Text Cleaning
  2. Normalization & tokenization
  3. Remove stop words
  4. Stemming
  5. Document representation using BOW
  6. Learning Clasiffication model: we tested Naive Bayes Classifier, SVM and LP to finally choose the NB classifier because it gave us the best accuracy and confusion matrix compared to  LR and SVM .
  7. Construct the final model using the entire corpus.
 

<b> Realized by [Ibtihel Sidhom](https://github.com/IbtihelSidhom), [Molka Zaouali](https://github.com/aklom) and [Taysir Ben Hamed](https://github.com/TaysirBenHamed) in December 2018 :computer: </b>


<br/>
<br/>


## :gear: Configuration 

### Set up your Python environment
Run this command under the root directory of this repository:

```shell
$ pipenv install
```

To create a virtual environment you just execute the `$ pipenv shell` command.

<br/>

## :open_book: User Manual

### Language Identification script

To run the language identification script on the existing corpus files, you can execute this command:

```shell
$ python Generating-models/language-identification.py 
```
You can also test it locally by uncommenting the last lines of the script and typing your input text in the script. Comment the dumping part to make the script run faster.

### Sentiment Analysis script

To run the sentiment analysis script on the existing corpus files, you can execute this command:

```shell
$ python Generating-models/sentiment-analysis.py 
```

You can also test it locally by uncommenting the last lines of the script and typing your input text in the script. Comment the dumping part to make the script run faster.

### Web application

To start the web application, you can execute this command: 

```shell
$ python Web-application/app.py
```

####  Entering a text message... 

<p align="center">
  <img src="https://user-images.githubusercontent.com/28828162/66261544-34b07b00-e7c7-11e9-9b0f-37fd5a17fb5b.png" width="700"/> 
   <img src="https://user-images.githubusercontent.com/28828162/66261554-7c370700-e7c7-11e9-9c2d-1c46b6f56014.png" width="700"/> 
</p>


#### or Uploading a file ! 

<p align="center">
  <img src="https://user-images.githubusercontent.com/28828162/66261541-3417e480-e7c7-11e9-8f1e-e344b45bfac7.png" width="700"/> 
   <img src="https://user-images.githubusercontent.com/28828162/66261543-34b07b00-e7c7-11e9-9258-237f96e511ea.png" width="700"/> 
</p>


#### Reviewing the prediction results :sparkles:	

In order to enlarge our data, when you get the results of a text message, you are asked for feedback on the predicted results by answering the given small form.

Based on this evaluation, this data will be stored in a file to be added to the corpus in the future.

<p align="center">
  <img src="https://user-images.githubusercontent.com/28828162/66261555-7c370700-e7c7-11e9-9301-89b0837f6985.png" width="700"/> 
</p>

