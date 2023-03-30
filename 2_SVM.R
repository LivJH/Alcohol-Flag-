
'''-------------------------------------------------------------------------
AIM: The aim of this code is to create an SVM Model to classify whether a 
free-text field (in police data) is alcohol-related or not. The methods are 
outlined below:

We initally create a dataset of crimes and their free-text fields (crime descriptions)
and manually review each free-text field, assigning it a label based on whether we think 
it is alcohol-related or not.

We split the data into 80:20 ratio: 80% is our training data and 20% is our testing data

1. We convert the free-text fields in our labelled training data into a Document Term Matrix (DTM). 
   A DTM is a mathemiatical matrix that describes the frequency of terms that occur in a collection of documents (rows, or free-text fields). 
   We then place it in a container.

2. The SVM is trained to learn from the labelled training data in the container.

3. We predict the SVM on the 20% test data. 
   The data is labelled (for later validation) however the SVM does not see the label.

4. Save the predictions in a separate file and validate them. 
   To validate, we compare the predictions with the actual value (the labels we created).

5. We can further assess the SVM model peformance by creating a confusion matrix of the predictions against the actual values. 
   This will produce some nice stats so we can measure the performance of the SVM.

6. Good performance? Predict the SVM on a larger dataset and compare the SVMs prediction of alcohol-related crime 
   with the current police measure of alcohol-related crime!

-------------------------------------------------------------------------------------------------------------------------------------------------'''


#-----------------------------------------------------------------------------------------
# Add libraries and read in data
#----------------------------------------------------------------------------------------
install.packages("e1071")
library(RTextTools)
library(caret)
library(kernlab)
library(dplyr)
library(tidyverse)
library(e1071)
library(lubridate)


trainingdata <- read_csv("1_Data/TrainingData.csv") 
# 80% of the original data. The free-text fields in this data have been reviewed and are
# allocated a label called 'my_label' which states whether the crime is alcohol-related. 

#--------

testdata <- read_csv("1_Data/TestData.csv") 
# 20% of the original data (50% are non alcohol-related, 50% are alcohol-related). 

#---------

testdata_50000 <- read_csv("1_Data/TestData_50000.csv")
# this data contains 50,000 crimes which the model will predict for later analysis.
# We couldn't predict on more than 50,000 crimes, as it is a computationally expensive task.
# The 50,000 crimes were chosen using a random subset in R from another larger crime dataset.





'-----------------------------------------------------------------------------------------
1. Convert to DTM
----------------------------------------------------------------------------------------

We need to convert the free-text fields in our labelled training data into a Document Term Matrix (DTM)

A DTM is a mathemiatical matrix that describes the frequency of terms that occur
in a collection of documents (rows, or free-text fields)
----------------------------------------------------------------------------------------'


# 1.1. create the DTM from the free-text field
dtMatrix <- create_matrix(trainingdata["FREETEXTFIELD"])



'-----------------------------------------------------------------------------------------
 2. Create and train the SVM Model
----------------------------------------------------------------------------------------

We need to put the DTM inside a container. 
In the containers configuration, we indicate that the whole dataset will be the training set
----------------------------------------------------------------------------------------'

# 2.1. Configure the training data - the redacted data. 'my_label' is the label to indicate
# whether the crime is actually alcohol-related or not.
container <- create_container(dtMatrix, trainingdata$`my_label`, trainSize = 1:500, virgin = FALSE) 
# train size = length of data 
# virgin=FALSE argument =  is here to tell RTextTools not to save an analytics_virgin-class object 
# inside the container. This parameter does not interest us now but is required by the function.

# 2.2. train a SVM model
model <- train_model(container, "SVM", kernel = "linear", cost = 1)





'----------------------------------------------------------------------------------------
3. Test on 20% data
----------------------------------------------------------------------------------------

Now that our model is trained, we can test to see if it works.

In the test data, we have a label to say whether the crimes are alcohol-related or not, 
these will be compared with the SVM predictions to validate the model
------------------------------------------------------------------------------------------'

# 3.1. Extract the free-text field from the test data
testdata <- testdata[, "FREETEXTFIELD"]

# 3.2. convert rows in data frame to a list
PredictionData <- split(testdata, seq(nrow(testdata)))

# 3.3. create a DTM for free-text field
predMatrix <- create_matrix(PredictionData, originalMatrix = dtMatrix)
# here, we provide the original matrix as a parameter. This is because we want 
# the new matrix to use the same vocabulary as the training matrix

# 3.4. create the container for the free-text field
predSize = length(PredictionData)
predictionContainer <- create_container(predMatrix, labels=rep(0, predSize),
                                        testSize = 1:predSize, virgin = FALSE)
# two things are different here:
# we use 0 vector for labels, because we want to predict them
# we specified testSize instead of trainingSize so that the data will be used for testing


# 3.5. Predict/test the SVM model on the test data
results <- classify_model(predictionContainer, model)
results 




'------------------------------------------------------------------------------------------
4. Save the predictions 
------------------------------------------------------------------------------------------ 
Re-read in the original test data.
Bind the SVM results to the original test data.
Save this to a separate file and validate them. 
To validate, we compare the predictions with the actual value (the labels we created).
------------------------------------------------------------------------------------------'

# 4.1. Re-read in the original test data (this contains 'my_label' - the actual label to indicate
# whether a crime is alcohol-related or not
testdata <- read_csv("1_Data/TestData.csv") 

# 4.2. Attach the results of the SVM model to the test data to validate against 'my_label'
testdata_SVMresults <- cbind(testdata, results)

# 4.3. write these results in a csv
write.csv(testdata_SVMresults, "testdata_SVMresults.csv")

# now open the file you created and validate them, do they look right?






'------------------------------------------------------------------------------------------
5. Model performance 
------------------------------------------------------------------------------------------ 

For further validation, we can assess the SVM model peformance by creating a confusion matrix 
of the predictions against the actual values. 

This will produce some nice stats so we can measure the performance of the SVM.

To understand these stats, head to:

------ https://datatricks.co.uk/confusion-matrix-in-r-two-simple-methods
OR
------ understanding stats: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

-----------------------------------------------------------------------------------------'

# 5.1. Read in test data again
testdata_SVMresults <- read_csv("testdata_SVMresults.csv")

# 5.2. Create a Confusion Matrix of the SVM label and the actual label ('my_label')
cm <- table(testdata_SVMresults$`SVM_LABEL`, testdata_SVMresults$`my_label`, dnn = c("Prediction", "Actual"))
# NOTE: the SVM_LABEL might be called something else, so change as appropriate


# 5.3. Produce Confusion Matrix results/statistics 
confusionMatrix(cm)



'If the accuracy of the model isnt great, maybe increase the size of the training
data, or re-assess how you allocated the labels in the first place'







'-----------------------------------------------------------------------------------------
6. Predict the SVM on a larger dataset 
------------------------------------------------------------------------------------------

Now that our model is trained and performing well, we can apply it to a larger dataset
to make new predictions.

We will predict the model on 50,000 crimes as this is the maximum capacity that my pc can 
computate.

These 50,000 crimes data were randomly subset in R from a larger dataset of crimes

----------------------------------------------------------------------------------------'


# 6.1. Extract the free text field from the data
testdata_50000 <- testdata_50000[, "FREETEXTFIELD"]

# 6.2. Split rows in data frame to a list
PredictionData <- split(testdata_50000, seq(nrow(testdata_50000)))

# 6.3. Create a DTM for the free-text field in the data
predMatrix <- create_matrix(PredictionData, originalMatrix = dtMatrix)
# here, we provide the original matrix as a parameter. This is because we want 
# the new matrix to use the same vocabulary as the training matrix

# 6.4. Create the container for the free-text field
predSize = length(PredictionData)
predictionContainer <- create_container(predMatrix, labels=rep(0, predSize),
                                        testSize = 1:predSize, virgin = FALSE)
# two things are different here:
# we use 0 vector for labels, because we want to predict them
# we specified testSize instead of trainingSize so that the data will be used for testing


# 6.5. Predict!
results <- classify_model(predictionContainer, model)
results


# 6.6. Re-read in the large test data
testdata_50000 <- read_csv("1_Data/TestData.csv") 

# 6.7. Attach the results of the SVM model to the test data
testdata50000_SVMresults <- cbind(testdata_50000, results)

# 6.8. write these results in a csv
write.csv(testdata50000_SVMresults, "testdata50000_SVMresults.csv")

# we don't create a confusion matrix like above because we do not have ACTUAL
# values (my_label) - as it's pretty difficult to review 50,000 crimes!! 


'Now, compare the SVMs predictions with the current police measure for alcohol-related crime!'

