# Alcohol-Flag-
Classifying free-text fields as alcohol-related or not using a Support Vector Machine.




## About this project

Owing to the inaccuracies of the current measure of alcohol-related crime in police data, this project intends to create a new measure of alcohol-related crime in police data, with the aim to improve estimates of alcohol-related crimes in the police setting. 


## How can we improve estimates of alcohol-related crime in police data?

Police data contains a field which provides a worded summary of the details of the crime. We call this the 'free-text field' hereon. 

The free-text field may contain alcohol-related keywords and violence-related terminology - which can be associated with alcohol-related crimes. 

As humans we can read a free-text field and with subjective judgement determine whether it is alcohol-related or not. 
It is also possible for algorithms to make this distinction too. For example, if we label a free-text field as alcohol-related (or not), the algorithm can learn which text is associated with an alcohol-related label, and which text is not.

As we are dealing with text data, this involves Natural Language Processing to convert the text data into a structured format which our algorithm can understand. 

Once we have done that, we can feed the data into an algorithm, namely a Support Vector Machine (SVM) which works well with text classification, for it to learn, classify and predict whether a free-text field is alcohol-related. 




## Guide to the R script: steps to the analysis and methods

Prior to analysis, we take our dataset and manually review each free-text field in it, assigning a label based on whether we think it is alcohol-related or not.

We split the data into 80:20 ratio: 80% is our training data and 20% is our testing data

1. We convert our labelled training data into a Document Term Matrix (DTM). A DTM is a mathemiatical matrix that describes the frequency of terms that occur in a collection of documents (rows, or free-text fields). We then place it in a container.

2. The SVM learns from the labelled training data in the container.

3. We predict the SVM on the 20% test data. The data is labelled (for later validation) however the SVM does not see the label. 

4. Save the predictions in a separate file and validate them. To validate, we compare the predictions with the actual value (the labels we created).

5. Once validated, create a confusion matrix of the predictions against the actual values. This will produce some nice stats so we can measure the performance of the SVM. 

6. Good performance? Predict the SVM on a larger dataset and compare the SVM's prediction of alcohol-related crime with the current police measure of alcohol-related crime! 
