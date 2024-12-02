# Book Genre Prediction using Book Summary
### The aim of the project is to apply the principles of text mining on a piece of literary text, and categorize it into the genre into which it best fits.


IPython Notebook has been used for this project.
   
   
   - The dataset that have been used here is a subset of the CMU Book Summary Dataset available at [CMU Book Summary Dataset](http://www.cs.cmu.edu/~dbamman/booksummaries.html "CMU Book Summary Dataset"). Dataset was unevenly distributed at the beginning. After days of data wrangling we could attain some uniformity in the dataset. The processed dataset has been uploaded as BookDataSet.csv , which has 6 main genres - "Crime Fiction", "Fanstasy", "Historical novel", "Horror", "Science Fiction", "Thriller". 
   
 ![Snap of dataset](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/dataset_snap.png)
 
   
   - The goal of this repository is to provide an example for performing Text Classification using the techniques of Natural Language Processing.
   
   - The highest accuracy was obtained during a 85-15% split while using a SVM with Radial Basis Function as the kernel function. The highest accuracy is 79.55%.
   
   - Required Libraries
       - [Pandas](https://pandas.pydata.org/)
       
       - [NumPy](https://numpy.org/")

       - [Scikit-Learn](https://scikit-learn.org/stable/)

       - [Matplotlib](https://matplotlib.org/)

       - [Seaborn](https://seaborn.pydata.org/)
       
       - [NLTK](https://www.nltk.org/)
       

### For viewing the whole code - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/BookGenrePrediction.ipynb)
 
 ##### Data pre-processing

- Let us have a look at the "summary" column.

<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/summary.png" />

- Preprocessing the "summary" column and making it ready for prediction.

- At first the the "summary" is traversed and only the alphabets are kept while filtering out everything else and then the alphabets are converted into lowercase.

<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/summaryfiltering.png" />

- Visualizing words and their frequency in the "summary".

<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/summaryvis.png" />

- Removing stop words.

  Another part of data cleaning is the removal of stop words – that is, common words like “the”, “a”, “an”. They are assumed to have no consequence over the classification process.
 
 <img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/summarystop.png" />
 
 
- Lemmatization

  In lemmatization, the different versions of the same word are put into one – for example, “do/doing/does/did”, “go/going/goes/went” – so as to not let the algorithm treat similar words as different, and hence, make the analysis stronger.
  
 <img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/lem.png" />
 
  
 -  Stemming
 
   Stemming is the process of producing morphological variants of the root/base word. Stemming reduces the words “chocolates”, “chocolatey”, “choco” to the root word, “chocolate” and “retrieval”, “retrieved”, “retrieves” reduce to the stem “retrieve”.
   
 <img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/stem.png" />
 
 - Visualizing the words in book's summary after successfully pre-processing.
 
  <img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/visa.png" />
  
 - Finally tf-idf is performed on the data set. This numerical statistic reflects the fact that how a word is important in a document. 
 
 	Tf_Idf(t,d,D) = tf(t,d).idf(t,D)
	
	Where,
		Tf(t,d) = ft,d
		Idf(t,D) = log(N/{d∈D : t ∈ d})
	With,
	
 		ft,d: Frequency of the term t in document d.
		N: Total number of documents.
		{d∈D : t ∈ d}: Number of documents where the term t appears.
 

##### Models

**1. At first a 80-20% split was performed on the dataset**
  - Logistic Regression is used when the dependent variable( target ) is categorical.Logistic regressions are only binary classifiers, meaning they cannot handle target vectors with more than two classes. However, there are clever extensions to logistic regression to do just that. In one-vs-rest logistic regression (OVR) a separate model is trained for each class predicted whether an observation is that class or not (thus making it a binary classification problem). It assumes that each classification problem (e.g. class 0 or not) is independent.
    
<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/LR80_20.png" width="50%" height="60%" />
    
   - SVMs try to find a separating line(or hyperplane) between data of two classes. SVM is an algorithm that takes the data as an input and outputs a line that separates those classes if possible.According to the SVM algorithm we find the points closest to the line from both the classes.These points are called support vectors. Now, we compute the distance between the line and the support vectors. This distance is called the margin. Our goal is to maximize the margin. The hyperplane for which the margin is maximum is the optimal hyperplane.Thus SVM tries to make a decision boundary in such a way that the separation between the two classes(that street) is as wide as possible.The linear, polynomial and RBF or Gaussian kernel are simply different in case of making the hyperplane decision boundary between the classes. The kernel functions are used to map the original dataset (linear/nonlinear ) into a higher dimensional space with view to making it linear dataset.Usually linear and polynomial kernels are less time consuming and provides less accuracy than the rbf or Gaussian kernels.

**SVM - Linear**


<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/svm80_20.png" width="50%" height="60%" />

**SVM - RBF**

<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/svm_rbf80_20.png" width="50%" height="60%" />

**2. Next a 85-15% split was performed on the dataset**

**Logistic Regression**

<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/LR85_15.png" width="50%" height="60%" />

**SVM - Linear**

<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/svm85_15.png" width="50%" height="60%" />

**SVM - RBF**

<img src="https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/svm_rbf85_15.png" width="50%" height="60%" />

- Thus we conclude that **highest accuracy** was obtained during a **85-15%** split while using a **svm with rbf as the kernel function**.

	- The highest **accuracy is 79.55 %**

	- Now, we build an **inference function for predicting the genres of new books in the future.**

	- Our book genre prediction system should be able to take a book's summary in raw form as input and generate its genre tag.

	
   
 - Sample output :
 
 ![Snap of Final output](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/sample_op.png)
   
Team Members - 
- Aniket Patra [Github](https://github.com/KamadoTanjiro-beep/)
- [Sirshendu Ganguly](https://www.linkedin.com/in/sirshendu-ganguly/)  [![Github](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/github.png)](https://github.com/Sirsho1997)
- [Arnanta Chatterjee](https://www.linkedin.com/in/arnanta-chatterjee-a60684179/)  [![Github](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/image/github.png)](https://github.com/arnanta)



