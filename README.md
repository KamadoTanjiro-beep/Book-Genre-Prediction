# New Document# Book-Genre-Prediction-using-Book-Summary
### The aim of this project is to apply the principles of text mining on a piece of literary text, and categorize it into the genre into which it best fits.





- The dataset that have been used here is a subset of the CMU Book Summary Dataset available at [CMU Book Summary Dataset](http://www.cs.cmu.edu/~dbamman/booksummaries.html "CMU Book Summary Dataset"). Dataset was unevenly distributed at the beginning. After days of data wrangling we could attain some uniformity in the dataset. The processed dataset has been uploaded as BookDataSet.csv , which has 6 main genres - "Crime Fiction", "Fanstasy", "Historical novel", "Horror", "Science Fiction", "Thriller". 
 ![Snap of dataset](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/dataset_snap.png)
 
 - The highest accuracy was obtained during a 85-15% split while using a svm with rbf as the kernel function. The highest accuracy is 79.55 %
 
 ##### Data pre-processing

- Preprocessing the "summary" column and making it ready for prediction.

- At first the the "summary" is traversed and only the alphabets are kept while filtering out everything else and then the alphabets are converted into lowercase.

- The most common words also known as stop words are removed.

  - Lemmatization is performed.

  - Stemming is performed.

- Filtering out any character which is not an alphabet and then converting each character into lowercase - 
```python
   def clean(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text
```

- Removing stop words.
  - Another part of data cleaning is the removal of stop words – that is, common words like “the”, “a”, “an”. They are assumed to have no consequence over the classification process.
 ```python
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  stop_words = set(stopwords.words('english'))

  #function to remove stopwords
  def removestopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

  books['summary'] = books['summary'].apply(lambda x: removestopwords(x))
 ```
- Lemmatization
   - Greater the degree of randomness, greater would be the computation time, and lesser would be the        efficiency of the detection of patterns in the textual corpora. We introduced the module which          performs lemmatization on words, that is, it groups different versions of the same word into one –      for example, “do/doing/does/did”, “go/going/goes/went” – so as to not let the algorithm treat          similar words as different, and hence, make the analysis stronger.
   ```python
      nltk.download('wordnet')
      from nltk.stem import WordNetLemmatizer

      lemma=WordNetLemmatizer()

      def lematizing(sentence):
          stemSentence = ""
          for word in sentence.split():
               stem = lemma.lemmatize(word)
               stemSentence += stem
               stemSentence += " "
          stemSentence = stemSentence.strip()
          return stemSentence
          
      books['summary'] = books['summary'].apply(lambda x: lematizing(x))
   ```
 -  Stemming
    - Stemming is the process of producing morphological variants of a root/base word. Stemming programs are commonly referred to as stemming algorithms or stemmers. A stemming algorithm reduces the words “chocolates”, “chocolatey”, “choco” to the root word, “chocolate” and “retrieval”, “retrieved”, “retrieves” reduce to the stem “retrieve”. 
    ```python
    
		from nltk.stem import PorterStemmer
		stemmer = PorterStemmer()
		def stemming(sentence):
    		stemSentence = ""
    		for word in sentence.split():
        		stem = stemmer.stem(word)
        		stemSentence += stem
        		stemSentence += " "
    		stemSentence = stemSentence.strip()
    		return stemSentence

		books['summary'] = books['summary'].apply(lambda x: stemming(x))
    ```
 - Visualizing the words in book's summary after successfully pre-processing.

![Snap of Frequency](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/word_freq.png)

##### Models
**1. At first a 80-20% split was performed on the dataset**
  - Logistic Regression is used when the dependent variable( target ) is categorical.

    - logistic regressions are only binary classifiers, meaning they cannot handle target vectors with more than two classes. However, there are clever extensions to logistic regression to do just that. In one-vs-rest logistic regression (OVR) a separate model is trained for each class predicted whether an observation is that class or not (thus making it a binary classification problem). It assumes that each classification problem (e.g. class 0 or not) is independent.
    ![Snap of LR 80-20](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/LR80_20.png)
   - SVMs try to find a separating line(or hyperplane) between data of two classes. SVM is an algorithm that takes the data as an input and outputs a line that separates those classes if possible.

     - According to the SVM algorithm we find the points closest to the line from both the classes.These points are called support vectors. Now, we compute the distance between the line and the support vectors. This distance is called the margin. Our goal is to maximize the margin. The hyperplane for which the margin is maximum is the optimal hyperplane.Thus SVM tries to make a decision boundary in such a way that the separation between the two classes(that street) is as wide as possible.The linear, polynomial and RBF or Gaussian kernel are simply different in case of making the hyperplane decision boundary between the classes. The kernel functions are used to map the original dataset (linear/nonlinear ) into a higher dimensional space with view to making it linear dataset.Usually linear and polynomial kernels are less time consuming and provides less accuracy than the rbf or Gaussian kernels.

**SVM - Linear**

![Snap of SVM L 80-20](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/svm80_20.png)

**SVM - RBF**

![Snap of SVM RBF 80-20](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/svm80_20.png)

**2. Next a 85-15% split was performed on the dataset**

**Logistic Regression**

![Snap of LR 85-15](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/LR85_15.png)

**SVM - Linear**

![Snap of SVM L 85-15](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/svm85_15.png)

**SVM - RBF**

![Snap of SVM RBF 85-15](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/svm85_15.png)

- Thus we conclude that **highest accuracy** was obtained during a **85-15%** split while using a **svm with rbf as the kernel function**.

	- The highest **accuracy is 79.55 %**

	- Now, we build an **inference function for predicting the genres of new books in the future.**

	- Our book genre prediction system should be able to take a book's summary in raw form as input and generate its genre tag.

	- To achieve this the steps to be considered during building the function are -

		- Clean the texts.
		- Remove stopwords from the cleaned texts.
		- Perform Lemmatization.
		- Perform Stemming.
		- Extract features from the text.
		- Make predictions using svm with rbf as the kernel function.
		- Return the predicted book genre tag.
    - The inference function on SVM(kernel=rbf) to predict future unknown genre: 
    ```python
    	def infertag(q):
    		q = clean(q)
    		q = removestopwords(q)
    		q = lematizing(q)
    		q = stemming(q)
    		q_vec = tfidf_vectorizer.transform([q])
    		q_pred = svc.predict(q_vec)
    		return LE.inverse_transform(q_pred)[0]
    		#return q_pred[0]


		for i in range(50): 
  			k = xval.sample(1).index[0] 
  
  			print("\nBook: ", books['book_name'][k], )
  			print("\nPredicted genre: ", infertag(xval[k]))
  			print("\nActual genre: ",books['genre'][k], "\n")
 			print("-------------------------------")
    ```
    
 - Sample output:
 
 ![Snap of Final output](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/sample_op.png)
   
Team Members - 
- [Aniket Patra](https://www.linkedin.com/in/aniket-patra/)
- [Arnanta Chatterjee](https://www.linkedin.com/in/arnanta-chatterjee-a60684179/)
- [Sirshendu Ganguly](https://www.linkedin.com/in/sirshendu-ganguly/)


