
'''-------------------------------------------------------------------------
AIM: The aim of this code is to create an SVM Model

1. 

1. It will be tested on 20% of the original data to determine its performance.

2. The goal is to predict the models on 50,000 crimes, and to have these
contained in a dataset
 
--------------------------------------------------------------------------'''


#-----------------------------------------------------------------------------------------
# 1. Add libraries and read in data
#----------------------------------------------------------------------------------------
install.packages("e1071")
library(RTextTools)
library(caret)
library(kernlab)
library(dplyr)
library(tidyverse)
library(e1071)
library(lubridate)


trainingdata <- read_csv("1_Data/TrainingData.csv") # This is the dataset which is a subset of 500 crimes
# from the original data. The crime notes in this data have been reviewed by myself and are
# allocated a 'true alcohol' flag which states whether the crime is alcohol-related. 
# parse date format to datetime fromat to work with lubridate



#--------

testdata_100 <- read_csv("1_Data/TestData_100.csv") 
# this is a dataset i created of 100 randomly selected crime notes from the 
# original data (1-50 are non AR, 51-100 are AR). it is used to test how well the
# model performs, before predicting the model on larger data.

# crimes in the training data cannot occur in the test data as the same crime
# cannot be trained and then tested on as it will affect the performance of the model
# this test data of 100 crimes does therefore not include any crimes in the training
# data.



testdata_125 <- read_csv("1_Data/TestData_125.csv")
# this is a dataset i created of 125 randomly selected crime notes from the 
# original data (62 are non AR, 63 are AR). it is used to test how well the
# model performs, before predicting the model on larger data.

'we test on 125 crimes instead of 100 as 125 calculates as 20% of our total data, 
if our training data (500 crimes) = 80%'

# crimes in the training data cannot occur in the test data as the same crime
# cannot be trained and then tested on as it will affect the performance of the model
# this test data of 100 crimes does therefore not include any crimes in the training
# data.



testdata_50000 <- read_csv("1_Data/TestData_50000.csv")
# this data contains 50,000 crimes which the model will predict for analysis.
# These 50,000 crimes data were randomly subset in R from the original data
# and validated (to see more on how this was created look at the bottom document 
# under 'CODE NOT IN USE' and find 'SUBSETTING DATA TO 50,000'
# parse date format to datetime format to work with lubridate
months <- testdata_50000 %>%
  group_by(Month_Yr) %>%
  summarise(number_months = n()) %>%
  drop_na()
#---------




'''ALWAYS READ IN BELOW LINES WHEN READING IN THIS DATA'''
original_data <- read_csv("1_Data/OriginalData.csv") # original data
# Clean data  - remove crimes before April 2021 and after September 2022
#               (September because there are not enough crimes in October 2022).
#-------------------------------------------------------------------------------
# parse date format to datetime format to work with lubridate
original_data$Month_Yr <- dmy_hm(original_data$`DATE EARLIEST COMMITTED`)
# pull out months 
original_data$Month_Yr <- format(as.Date(original_data$Month_Yr), "%Y-%m")
# keep only months between April 2021 and September 2022
original_data <- original_data[original_data$Month_Yr >= "2021-04" & original_data$Month_Yr <= "2022-09", ]






'''--------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
SVM based on crime notes

Support Vector Machine tutorial 1 
----------------------------------------------------------------
  (from https://www.svm-tutorial.com/2014/11/svm-classify-text-r/)

-----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------'


#-----------------------------------------------------------------------------------------
# 1. Convert to DTM
#--------------------

# we need to convert the crime notes text into a Document Term Matrix (DTM)

# A DTM is a mathemiatical matrix that describes the frequency of terms that occur
# in a collection of documents (rows, or crimes)
#----------------------------------------------------------------------------------------


# create the DTM
dtMatrix <- create_matrix(trainingdata["CRIME NOTES"])






#-----------------------------------------------------------------------------------------
# 2. Create and train the SVM Model
#-----------------------------------

# We need to put the DTM inside a container. 

# In the container's configuration, we indicate that the whole dataset will be the training set
#----------------------------------------------------------------------------------------

# Configure the training data - the redacted data
container <- create_container(dtMatrix, trainingdata$`TRUE ALCOHOL`, trainSize = 1:500, virgin = FALSE) 
# the virgin=FALSE argument is here to tell RTextTools not to save an analytics_virgin-class object 
#inside the container. This parameter does not interest us now but is required by the function.

# train a SVM model
model <- train_model(container, "SVM", kernel = "linear", cost = 1)





#-----------------------------------------------------------------------------------------
# 3. Test/Predict on 100 and 125 crimes
#--------------------------------------------

# Now that our model is trained, we can use it to make new predictions.
# The test data is produced by taking 100 crimes at random from the original data


# In the test data, we have a 'true alcohol' flag which i have created to say whether
# the crimes are alcohol-related or not, these will be compared with the SVM predictions to 
# validate the model
#------------------------------------------------------------------------------------------

'change code depending on whether test = 100 or 125'
# test data - 100 or 125 crimes
testdata_100 <- testdata_100[, "CRIME NOTES"]

# rows in data frame to a list
PredictionData <- split(testdata_100, seq(nrow(testdata_100)))

# create a DTM for the test data
predMatrix <- create_matrix(PredictionData, originalMatrix = dtMatrix)
# here, we provide the original matrix as a parameter. This is because we want 
#the new matrix to use the same vocabulary as the training matrix

# create the container
predSize = length(PredictionData)
predictionContainer <- create_container(predMatrix, labels=rep(0, predSize),
                                        testSize = 1:predSize, virgin = FALSE)
# two things are different here:
# we use 0 vector for labels, because we want to predict them
# we specified testSize instead of trainingSize so that the data will be used for testing


# predict
results <- classify_model(predictionContainer, model)
results 


# append the results to the corresponding data 
testdata_100 <- read_csv("TestData_100.csv") # read in original test data (100 crimes)
testdata_125 <- read_csv("TestData_125.csv") # read in original test data (125 crimes - 20% test data)
test_data_SVMflag <- cbind(testdata_100, results) #  attach the SVM flag to test data



# write these results in a csv for 100 crimes
write.csv(test_data_SVMflag, "SVM_flag_100.csv")
# write these results in a csv for 125 crimes
write.csv(test_data_SVMflag, "SVM_flag_125.csv")





#-----------------------------------------------------------------------------------------
# 4. How well did the SVM do?
#-----------------------------------------------------------------------------------------
#------ https://datatricks.co.uk/confusion-matrix-in-r-two-simple-methods
#------ understanding stats: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/


# read in data
SVM_flag_100 <- read_csv("4_Outputs/SVM_flag_100.csv") # 100 crimes
SVM_flag_125 <- read_csv("4_Outputs/SVM_flag_125.csv") # 125 crimes (20% test data)

# Confusion matrix 
cm <- table(SVM_flag_125$`SVM_LABEL`, SVM_flag_125$`True Alcohol`, dnn = c("Prediction", "Actual"))

# Produce statistics 
confusionMatrix(cm)
#------------------- 




' The reason we dont compare the SVM with the police flags accuracy is
because the police flag is nearly 99% accurate when compared with the 
true alcohol flag.

This is because true alcohol-related crimes (true alcohol flag) were based on
the police alcohol flag - as sometimes, although the crime notes dont
appear to be alcohol-related (e.g. they contain violent keywords but no
alcohol keywords), if the police have flagged them as so, our
model needs to understand that that crime will be alcohol-related - it was
important for the model to understand police language.'



#-----------------------------------------------------------------------------------------
# 5. Predict on 50,000 crimes from the original data
#---------------------------------------------------

# Now that our model is trained, we can use it to make new predictions.
# We will predict the model on 50,000 crimes as this is the maximum capacityof the model

# These 50,000 crimes data were randomly subset in R from the original data
# and validated (to see more on how this was created look at the bottom document 
# under #CODE NOT IN USE# and find #SUBSETTING DATA TO 50,000#
#----------------------------------------------------------------------------------------



data <- testdata_50000 # save a copy of the 50,000 crimes and all the columns as 
# as this data will get changed when we clean it later, and will be needed for later, 


'''we do not create a confusion matrix like above because we do not have ACTUAL
values - as these are based on the TRUE ALCOHOL FLAG and i cant do that on 50,000 crimes. 

The point of this section is to apply the model to 50,000 crimes to predict whether
they are alcohol-related or not (SVM flag) and compare them with the police flag and
the keyword flag).'''


# 5.a. Prepare the test data (50,000 crimes - random subset) -------------------------------------
#-------------------------------------------------------------------------

testdata_50000 <- testdata_50000[, "CRIME NOTES"]
# rows in data frame to a list
PredictionData <- split(testdata_50000, seq(nrow(testdata_50000)))

# create a DTM for the test data
predMatrix <- create_matrix(PredictionData, originalMatrix = dtMatrix)
# here, we provide the original matrix as a parameter. This is because we want 
#the new matrix to use the same vocabulary as the training matrix

# create the container
predSize = length(PredictionData)
predictionContainer <- create_container(predMatrix, labels=rep(0, predSize),
                                        testSize = 1:predSize, virgin = FALSE)
# two things are different here:
# we use 0 vector for labels, because we want to predict them
# we specified testSize instead of trainingSize so that the data will be used for testing


# predict
results <- classify_model(predictionContainer, model)
results

# append the results to the corresponding data 
test_data_SVMflag <- cbind(data, results) #  attach the SVM flag to test data


# write these results in a csv
write.csv(test_data_SVMflag, "SVM_flag_50000.csv")
















''' ----------------------------------------------------------------------------

CODE NOT IN USE

-----------------------------------------------------------------------------'''



################################################################################
#                             SUBSETTING DATA TO 50,000
################################################################################



# 5.a. Subset the test data to 50,000-------------------------------------
#-------------------------------------------------------------------------

#---Before running the below, read in training data ----------------------

#crimes in the training data cannot occur in the test data as the same crime
#cannot be trained and then tested on. To remove crimes which occur in the test data 
#from the training data, run the following (we can't do this manually like in 
#the test data of 100 crimes because we have a larger dataset):

# for rows that occur in original data (as we are going to subset it to 50,000) 
# that are in the training data, remove from the original data
originaldata_minus_trainingdata <- original_data[!(original_data$`Crime Ref` %in% trainingdata$`Crime Ref`), ]

# we need to make sure this subset occurs uniformly across time. If we simply 
# subset the data to 50,000 (e.g. data[50000, ]) then we're only taking the first 
# 50,000 which would be the first few months. we therefore need a random subset 
# of 50,000 crimes:
testdata_50000 <- originaldata_minus_trainingdata[sample(nrow(originaldata_minus_trainingdata), size = 50000), ]

# save 
write.csv(testdata_50000, "TestData_50000.csv")


# 5.a.a Are these 50,000 crimes uniformly distributed across time in the -------
#same way as the original data?-------------------------------------------------
#-------------------------------------------------------------------------------

## ALL DATA (no subset)
#------------------------------------------------------------------------------

# Month  
#--------------
# parse date format to datetime format to work with lubridate
testdata_50000$Month_Yr <- dmy_hm(testdata_50000$`DATE EARLIEST COMMITTED`)
# pull out months 
testdata_50000$Month_Yr <- format(as.Date(testdata_50000$Month_Yr), "%Y-%m")
# keep only months between April 2021 and September 2022
testdata_50000 <- testdata_50000[testdata_50000$Month_Yr >= "2021-04" & testdata_50000$Month_Yr <= "2022-09", ]

# aggregate number of rows (crimes) to month
months <- testdata_50000 %>%
  group_by(Month_Yr) %>%
  summarise(number_months = n()) %>%
  drop_na()
# plot months
ggplot(data = months, aes(x = Month_Yr, y = number_months)) +
  geom_line(group = 1) +
  ylab("crimes (count)") +
  xlab("Year-Month") +
  theme(axis.text.x = element_text(size = 8, angle = 45, vjust = 1, hjust=1))



# Weekday
#--------------
# aggregate number of rows (crimes) to wkday
wkday <- testdata_50000 %>%
  group_by(`DATE EARLIEST COMMITTED WKDAY`) %>%
  summarise(number_wkday = n()) %>%
  drop_na()

# reorder from alphabetical to mon-fri for graph
wkday$`DATE EARLIEST COMMITTED WKDAY` <- factor(wkday$`DATE EARLIEST COMMITTED WKDAY`, 
                                                levels = c("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"))

# Plot 
ggplot(data = wkday, aes(x = `DATE EARLIEST COMMITTED WKDAY`, y = number_wkday)) +
  geom_line(group = 1) +
  scale_x_discrete() +
  ylab("crimes (count)") +
  xlab("Week day") +
  theme(axis.text.x = element_text(size = 8, angle = 45, vjust = 1, hjust=1))


# time of day
#----------------
# aggregate number of rows (crimes) to wkday
hourday <- testdata_50000 %>%
  group_by(`DATE EARLIEST COMMITTED HOUR`) %>%
  summarise(number_hour = n()) %>%
  drop_na()
# Plot 
ggplot(data = hourday, aes(x = `DATE EARLIEST COMMITTED HOUR`, y = number_hour)) +
  geom_line(group = 1) +
  scale_x_discrete() +
  ylab("crimes (count)") +
  xlab("time of day (24 hour)") +
  theme(axis.text.x = element_text(size = 7, vjust = 1, hjust=1))



# Crime type
#----------------
crimetype <- testdata_50000 %>%
  group_by(`OFFENCE RECORDED GROUP`) %>%
  summarise(number = n())
# Plot 
ggplot(data = crimetype, aes(x = reorder(`OFFENCE RECORDED GROUP`, -number), y = number)) +
  geom_bar(stat = "identity") +
  ylab("crimes (count)") +
  xlab("crime type") +
  theme(axis.text.x = element_text(size = 8, angle = 45, vjust = 1, hjust=1))

#----------------


# ALCOHOL DATA (subset alcohol-related crimes)
#-----------------------------------------------------------------------------

alcohol_testdata_50000 <- testdata_50000 %>%
  filter(`IND ALCOHOL` == "Y")


# Month  
#--------------
# parse date format to datetime fromat to work with lubridate
alcohol_testdata_50000$month <- dmy_hm(alcohol_testdata_50000$`DATE EARLIEST COMMITTED`)
# pull out months 
alcohol_testdata_50000$month <- month(alcohol_testdata_50000$month)

# aggregate number of rows (crimes) to month
months <- alcohol_testdata_50000 %>%
  group_by(Month_Yr) %>%
  summarise(number_months = n()) %>%
  drop_na()
# plot months
ggplot(data = months, aes(x = Month_Yr, y = number_months)) +
  geom_line(group = 1) +
  ylab("crimes (count)") +
  theme(axis.text.x = element_text(size = 8, angle = 45, vjust = 1, hjust=1))




# Weekday
#--------------
# aggregate number of rows (crimes) to wkday
wkday <- alcohol_testdata_50000 %>%
  group_by(`DATE EARLIEST COMMITTED WKDAY`) %>%
  summarise(number_wkday = n()) %>%
  drop_na()
# Plot 
ggplot(data = wkday, aes(x = `DATE EARLIEST COMMITTED WKDAY`, y = number_wkday)) +
  geom_line(group = 1) +
  scale_x_discrete() +
  ylab("crimes (count)") +
  xlab("Week day") +
  theme(axis.text.x = element_text(size = 8, angle = 45, vjust = 1, hjust=1))



# time of day
#----------------
# aggregate number of rows (crimes) to wkday
hourday <- alcohol_testdata_50000 %>%
  group_by(`DATE EARLIEST COMMITTED HOUR`) %>%
  summarise(number_hour = n()) %>%
  drop_na()
# Plot 
ggplot(data = hourday, aes(x = `DATE EARLIEST COMMITTED HOUR`, y = number_hour)) +
  geom_line(group = 1) +
  scale_x_discrete() +
  ylab("crimes (count)") +
  xlab("time of day (24 hour)") +
  theme(axis.text.x = element_text(size = 7, vjust = 1, hjust=1))


# Crime type
#----------------
crimetype <- alcohol_testdata_50000 %>%
  group_by(`OFFENCE RECORDED GROUP`) %>%
  summarise(number = n())
# Plot 
ggplot(data = crime_type, aes(x = `OFFENCE RECORDED GROUP`, y = number)) +
  geom_bar(stat = "identity") +
  ylab("crimes (count)") +
  xlab("crime type") +
  theme(axis.text.x = element_text(size = 8, angle = 45, vjust = 1, hjust=1))




'''The testdata_50000 appears to be similarly distributed as the original data
so we will use that subset for the algorithm to predict

it is then written as a csv file (which we read in at the top)'''





'''---------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
SVM MODEL 2 - based on time of day, day of week, crime type'''

# accessed code from: https://www.projectpro.io/recipes/use-svm-classifier-r 

# also here:
# tutorial: https://odsc.medium.com/build-a-multi-class-support-vector-machine-in-r-abcdd4b7dab6

'-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------'



#-----------------------------------------------------------------------------------------
# 1. Cleaning data
#--------------------


# training data ----------------------------------
train <- trainingdata[, c("TRUE ALCOHOL", "DATE EARLIEST COMMITTED HOUR", "DATE EARLIEST COMMITTED WKDAY", "IND DOMESTIC")]

'''clean data - currently text and categorical data - change to numeric'''

# convert outcome variable to factor
train$`TRUE ALCOHOL` <- as.factor(train$`TRUE ALCOHOL`)
#----------------
# recode weekdays and turn to dummy variables for modelling - this is because they are nominal variable types
# recode friday - sun = weekend
train$`DATE EARLIEST COMMITTED WKDAY` <- recode(train$`DATE EARLIEST COMMITTED WKDAY`, FRIDAY = "weekend", SATURDAY = "weekend", SUNDAY = "weekend")
# recode mon - thurs = weekday
train$`DATE EARLIEST COMMITTED WKDAY` <- recode(train$`DATE EARLIEST COMMITTED WKDAY`, MONDAY = "weekday", TUESDAY = "weekday", WEDNESDAY = "weekday", THURSDAY = "weekday")
# turn weekend and weekday into dummy variables
train$weekenddummy <- ifelse(train$`DATE EARLIEST COMMITTED WKDAY` == 'weekend', 1, 0) 
train$weekdaydummy <- ifelse(train$`DATE EARLIEST COMMITTED WKDAY` == 'weekday', 1, 0) 
#-----------------
# recode hour of day to dummy variables for modelling - this is because they are nominal variable types
# recode 12:00 (noon) - 18:00 = nonalcoholhours...according to Flatley (2015) these hours are where there are less alcohol-related offences
train$`DATE EARLIEST COMMITTED HOUR` <- recode(train$`DATE EARLIEST COMMITTED HOUR`, 
                                               `12` = "nonalcoholhour", `13`= "nonalcoholhour", 
                                               `14`= "nonalcoholhour", `15`= "nonalcoholhour", 
                                               `16`= "nonalcoholhour", `17`= "nonalcoholhour", `18`= "nonalcoholhour")
# recode 19:00 - 05:00 = alcohol hours...according to Flatley (2015) these hours are where there are more alcohol-related offences
train$`DATE EARLIEST COMMITTED HOUR` <- recode(train$`DATE EARLIEST COMMITTED HOUR`, 
                                               `19`= "alcoholhour", `20`= "alcoholhour", `21`= "alcoholhour", 
                                               `22`= "alcoholhour", `23`= "alcoholhour", `0`= "alcoholhour", `1`= "alcoholhour", `2`= "alcoholhour",
                                               `3`= "alcoholhour", `4`= "alcoholhour",`5`= "alcoholhour", `6`= "alcoholhour",`7`= "alcoholhour", `8`= "alcoholhour",
                                               `9`= "alcoholhour", `10`= "alcoholhour",`11`= "alcoholhour")
# turn alcoholhour and nonalcoholhour into dummy variable
train$nonalcoholhourdummy <- ifelse(train$`DATE EARLIEST COMMITTED HOUR` == 'nonalcoholhour', 1, 0) 
train$alcoholhourdummy <- ifelse(train$`DATE EARLIEST COMMITTED HOUR` == 'alcoholhour', 1, 0) 
#--------------------
# recode DV flag
train$DVFlag <- recode(train$`IND DOMESTIC`, N = "0", Y = "1")
train$DVFlag <- as.factor(train$DVFlag) # change to factor
#--------------------
# OPTION 1 - subset train data to recoded variables
train <- train[, c("TRUE ALCOHOL", "weekenddummy", "weekdaydummy", "nonalcoholhourdummy", "alcoholhourdummy", "DVFlag")]
# OPTION 2 - subset train data to recoded variables - remove reference variable
train <- train[, c("TRUE ALCOHOL", "weekdaydummy", "alcoholhourdummy", "DVFlag")]










# test data-----------------------------------------------

# choose which test data to use:
testdata_100 <- read_csv("TestData_100.csv") 
testdata_125 <- read_csv("TestData_125.csv")
'change below depending on which test data using'
test <- testdata_125


test <- test[, c("DATE EARLIEST COMMITTED HOUR", "DATE EARLIEST COMMITTED WKDAY", "IND DOMESTIC")]

# recode weekdays and turn to dummy variables for modelling
# recode friday - sun = weekend
test$`DATE EARLIEST COMMITTED WKDAY` <- recode(test$`DATE EARLIEST COMMITTED WKDAY`, FRIDAY = "weekend", SATURDAY = "weekend", SUNDAY = "weekend")
# recode mon - thurs = weekday
test$`DATE EARLIEST COMMITTED WKDAY` <- recode(test$`DATE EARLIEST COMMITTED WKDAY`, MONDAY = "weekday", TUESDAY = "weekday", WEDNESDAY = "weekday", THURSDAY = "weekday")
# turn weekend and weekday into dummy variable
test$weekenddummy <- ifelse(test$`DATE EARLIEST COMMITTED WKDAY` == 'weekend', 1, 0) 
test$weekdaydummy <- ifelse(test$`DATE EARLIEST COMMITTED WKDAY` == 'weekday', 1, 0) 

# recode hour of day to dummy variables for modelling - this is because they are nominal variable types
# recode 12:00 (noon) - 18:00 = nonalcoholhours...according to Flatley (2015) these hours are where there are less alcohol-related offences
test$`DATE EARLIEST COMMITTED HOUR` <- recode(test$`DATE EARLIEST COMMITTED HOUR`, 
                                              `12` = "nonalcoholhour", `13`= "nonalcoholhour", 
                                              `14`= "nonalcoholhour", `15`= "nonalcoholhour", 
                                              `16`= "nonalcoholhour", `17`= "nonalcoholhour", `18`= "nonalcoholhour")
# recode 19:00 - 05:00 = alcohol hours...according to Flatley (2015) these hours are where there are more alcohol-related offences
test$`DATE EARLIEST COMMITTED HOUR` <- recode(test$`DATE EARLIEST COMMITTED HOUR`, 
                                              `19`= "alcoholhour", `20`= "alcoholhour", `21`= "alcoholhour", 
                                              `22`= "alcoholhour", `23`= "alcoholhour", `0`= "alcoholhour", `1`= "alcoholhour", `2`= "alcoholhour",
                                              `3`= "alcoholhour", `4`= "alcoholhour",`5`= "alcoholhour", `6`= "alcoholhour",`7`= "alcoholhour", `8`= "alcoholhour",
                                              `9`= "alcoholhour", `10`= "alcoholhour",`11`= "alcoholhour")
# turn alcoholhour and nonalcoholhour into dummy variable
test$nonalcoholhourdummy <- ifelse(test$`DATE EARLIEST COMMITTED HOUR` == 'nonalcoholhour', 1, 0) 
test$alcoholhourdummy <- ifelse(test$`DATE EARLIEST COMMITTED HOUR` == 'alcoholhour', 1, 0) 
#--------------------
# recode DV flag
test$DVFlag <- recode(test$`IND DOMESTIC`, N = "0", Y = "1")
test$DVFlag <- as.factor(test$DVFlag) # change to factor
#--------------------
# OPTION 1: subset test data to recoded variables
test <- test[, c("weekenddummy", "weekdaydummy", "nonalcoholhourdummy", "alcoholhourdummy", "DVFlag")]
# OPTION 2: subset test data to recoded variables - remove reference variable
test <- test[, c("weekdaydummy", "alcoholhourdummy", "DVFlag")]





#-----------------------------------------------------------------------------------------
# 2. Train the SVM Model
#-----------------------------------

# specifying the CV technique which will be passed into the train() function later and
# number parameter is the "k" in K-fold cross validation
train_control <- trainControl(method = "cv", number = 5)

set.seed(50)



# training a regression model while tuning parmeters (Method = "rpart)
model <- train(`TRUE ALCOHOL` ~ weekdaydummy + alcoholhourdummy + DVFlag,
               data = train, method = "svmLinear", trControl = train_control)





#-----------------------------------------------------------------------------------------
# 3. Predict on 100 or 125 crimes
#--------------------------------------------

# Now that our model is trained, we can use it to make new predictions.
# The test data is produced by taking 100 crimes at random from the original data


# In the test data, we have a 'true alcohol' flag which i have created to say whether
# the crimes are alcohol-related or not, these will be compared with the SVM predictions to 
# validate the model
#------------------------------------------------------------------------------------------


''' at the moment i cant find a way to append the results in the right order to the test 
dataset 
- need to check it has also been appended correctly for the first model too'''

results <- predict(model, test)
results <- as.data.frame(results)

# append the results to the corresponding data 
testdata_100 <- read_csv("TestData_100.csv") # read in original test data
testdata_125 <- read_csv("TestData_125.csv") # read in original test data
SVM2_flag_125 <- cbind(testdata_125, results) #  attach the SVM flag to test data

write.csv(SVM2_flag_125, "SVM2_flag_125.csv")





#-----------------------------------------------------------------------------------------
# 4. How well did the SVM do?
#-----------------------------------------------------------------------------------------
#------ https://datatricks.co.uk/confusion-matrix-in-r-two-simple-methods
#------ understanding stats: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/


# Confusion matrix 
cm <- table(SVM2_flag_125$results, SVM2_flag_125$`True Alcohol`, dnn = c("Prediction", "Actual"))

# Produce statistics 
confusionMatrix(cm)







#-----------------------------------------------------------------------------------------
# 5. Predict on 50,000 crimes from the original data
#---------------------------------------------------

# Now that our model is trained, we can use it to make new predictions.
# We will predict the model on 50,000 crimes as this is the maximum capacity
# of the model

# These 50,000 crimes data were randomly subset in R from the original data
# and validated (to see more on how this was created look at the bottom document 
# under #CODE NOT IN USE# and find #SUBSETTING DATA TO 50,000#
#----------------------------------------------------------------------------------------

'''we do not create a confusion matrix like above because we do not have ACTUAL
values - as these are based on the TRUE ALCOHOL FLAG and i cant do that on 50,000 crimes. 

The point of this section is to apply the model to 50,000 crimes to predict whether
they are alcohol-related or not (SVM flag) and compare them with the police flag and
the keyword flag).'''


data <- testdata_50000 # save a copy of the 50,000 crimes and all the columns as 
# as this data will get changed when we clean it later, and will be needed for later, 

test <- testdata_50000



# 5.a. Prepare the test data (50,000 crimes - random subset) -------------------------------------
#-------------------------------------------------------------------------

test <- test[, c("DATE EARLIEST COMMITTED HOUR", "DATE EARLIEST COMMITTED WKDAY", "IND DOMESTIC")]

# recode weekdays and turn to dummy variables for modelling
# recode friday - sun = weekend
test$`DATE EARLIEST COMMITTED WKDAY` <- recode(test$`DATE EARLIEST COMMITTED WKDAY`, FRIDAY = "weekend", SATURDAY = "weekend", SUNDAY = "weekend")
# recode mon - thurs = weekday
test$`DATE EARLIEST COMMITTED WKDAY` <- recode(test$`DATE EARLIEST COMMITTED WKDAY`, MONDAY = "weekday", TUESDAY = "weekday", WEDNESDAY = "weekday", THURSDAY = "weekday")
# turn weekend and weekday into dummy variable
test$weekenddummy <- ifelse(test$`DATE EARLIEST COMMITTED WKDAY` == 'weekend', 1, 0) 
test$weekdaydummy <- ifelse(test$`DATE EARLIEST COMMITTED WKDAY` == 'weekday', 1, 0) 

# recode hour of day to dummy variables for modelling - this is because they are nominal variable types
# recode 12:00 (noon) - 18:00 = nonalcoholhours...according to Flatley (2015) these hours are where there are less alcohol-related offences
test$`DATE EARLIEST COMMITTED HOUR` <- recode(test$`DATE EARLIEST COMMITTED HOUR`, 
                                              `12` = "nonalcoholhour", `13`= "nonalcoholhour", 
                                              `14`= "nonalcoholhour", `15`= "nonalcoholhour", 
                                              `16`= "nonalcoholhour", `17`= "nonalcoholhour", `18`= "nonalcoholhour")
# recode 19:00 - 05:00 = alcohol hours...according to Flatley (2015) these hours are where there are more alcohol-related offences
test$`DATE EARLIEST COMMITTED HOUR` <- recode(test$`DATE EARLIEST COMMITTED HOUR`, 
                                              `19`= "alcoholhour", `20`= "alcoholhour", `21`= "alcoholhour", 
                                              `22`= "alcoholhour", `23`= "alcoholhour", `0`= "alcoholhour", `1`= "alcoholhour", `2`= "alcoholhour",
                                              `3`= "alcoholhour", `4`= "alcoholhour",`5`= "alcoholhour", `6`= "alcoholhour",`7`= "alcoholhour", `8`= "alcoholhour",
                                              `9`= "alcoholhour", `10`= "alcoholhour",`11`= "alcoholhour")
# turn alcoholhour and nonalcoholhour into dummy variable
test$nonalcoholhourdummy <- ifelse(test$`DATE EARLIEST COMMITTED HOUR` == 'nonalcoholhour', 1, 0) 
test$alcoholhourdummy <- ifelse(test$`DATE EARLIEST COMMITTED HOUR` == 'alcoholhour', 1, 0) 
#--------------------
# recode DV flag
test$DVFlag <- recode(test$`IND DOMESTIC`, N = "0", Y = "1")
test$DVFlag <- as.factor(test$DVFlag) # change to factor
#--------------------
# OPTION 1: subset train data to recoded variables
test <- test[, c("weekenddummy", "weekdaydummy", "nonalcoholhourdummy", "alcoholhourdummy", "DVFlag")]
# OPTION 2: subset test data to recoded variables - remove reference variable
test <- test[, c("weekdaydummy", "alcoholhourdummy", "DVFlag")]



# 5.b. Predict----------------------------------------------------------
#-----------------------------------------------------------------------

# predict
results <- predict(model, test)
results <- as.data.frame(results)

# append the results to the corresponding data
data <- read_csv("testdata_50000.csv") # read in original test data
SVM2_flag_50000 <- cbind(data, results) #  attach the SVM flag to test data

write.csv(SVM2_flag_50000, "SVM2_flag_50000_VERSION2.csv")







#-----------------------------------------------------------------------------------------
# 6. Append the data

# Add all the predictions of 50,000 crimes from SVM model 1 and 2 
# into one dataset
#-----------------------------------------------------------------------------------------


SVM1 <- read_csv("SVM_flag_50000.csv")
SVM2 <- read_csv("SVM2_flag_50000.csv")



# clean data -------------------------------

# rename columns for easier reading before merging
SVM1 <- SVM1 %>%
  rename(SVM1_LABEL = SVM_LABEL) %>%
  rename(SVM1_PROB = SVM_PROB)

SVM2 <- SVM2 %>%
  rename(SVM2_LABEL = results)


SVM2 <- SVM2[,20] # keep the prediction only



# merge together ----------------------------
predictions <- cbind(SVM1, SVM2)
write.csv(predictions, "Predictions.csv")
