setwd("D:/Analytics/Insight/Blog/Logistic Regresion")
data <- read.csv("bank.csv")
head(data)
str(data)
# Formatting the variables as per requirement
data$id <- as.character(data$id)
data$age <- as.double(data$age)
data$sex <- factor(data$sex)
data$region <- factor(data$region)
data$income <- as.double(data$income)
data$married <- factor(data$married)
data$children <- as.integer(data$children)
data$car <- factor(data$car)
data$saving_ac <- factor(data$saving_ac)
data$current_ac <- factor(data$current_ac)
data$cross_sell <- as.character(data$cross_sell)
data[which(data$cross_sell == "YES"), ]$cross_sell <- 1
data[which(data$cross_sell == "NO"), ]$cross_sell <- 0
data$cross_sell <- as.integer(data$cross_sell)

str(data)
sum(is.na(data))    # No Missing values are there

dim(data)
# Checking class bias for the dependent variable i.e. cross_sell
table(data$cross_sell)

# Creating the Training sample ---

input_one  <- data[which(data$cross_sell == 1), ]  # all 1
input_zero <- data[which(data$cross_sell == 0), ]  # all 0
set.seed(100)                                        # for repeatability of samples
input_one_training_rows <- sample(1:nrow(input_one), 0.8*nrow(input_one))  # 1's for training
input_zero_training_rows <- sample(1:nrow(input_zero), 0.8*nrow(input_zero))  # 0's for training. Pick as many 0's as 1's
training_one <- input_one[input_one_training_rows, ]  
training_zero <- input_zero[input_zero_training_rows, ]
trainingData <- rbind(training_one, training_zero)  # row bind the 1's and 0's
table(trainingData$cross_sell)

# Creating Test sample ---
test_one <- input_one[-input_one_training_rows, ]
test_zero <- input_zero[-input_zero_training_rows, ]
testData <- rbind(test_one, test_zero)  # row bind the 1's and 0's 
table(testData$cross_sell)

# Building the Logit Models and Predict
options(scipen=99999)  # For fixing decimal points in scientific notation
logitMod <- glm(cross_sell ~ age + sex + region + income + married + children + car + saving_ac + current_ac, data=trainingData, family=binomial(link="logit"))
# Model Diagnostics
summary(logitMod)

# Modelling with significant parameters
logitMod <- glm(cross_sell ~ income + married, data=trainingData, family=binomial(link="logit"))

# Predicting with the model on the test data
predicted_training <- predict(logitMod, trainingData, type="response")
predicted_test <- predict(logitMod, testData, type="response")

# Decide on optimal prediction probability cutoff for the model
#library(InformationValue)
optCutOff <- optimalCutoff(testData$cross_sell, predicted_test)[1] 
optCutOff

# Misclassification Error
misClassError(trainingData$cross_sell, predicted_training, threshold = optCutOff)
misClassError(trainingData$cross_sell, predicted_training, threshold = 0.5)
misClassError(testData$cross_sell, predicted_test, threshold = 0.5)
misClassError(testData$cross_sell, predicted_test, threshold = optCutOff)


###ROC: Receiver Operating Characteristics Curve: 
## mark more of actual 1's as positives and lesser of actual 0's as 1's. 
##So for a good model, the curve should rise steeply, 
##indicating that the TPR (Y-Axis) increases faster than the FPR (X-Axis) as the cutoff score decreases. 
##Greater the area under the ROC curve, better the predictive ability of the model.
plotROC(trainingData$cross_sell, predicted_training)
plotROC(testData$cross_sell, predicted_test)

###Specificity and Sensitivity

##Sensitivity=(# Actual 1's and Predicted as 1's)/(# of Actual 1's)
sensitivity(testData$default, predicted.training, threshold = 0.5)
sensitivity(testData$default, predicted.training, threshold = optCutOff)
sensitivity(testData$default, predicted.test, threshold = 0.5)
sensitivity(testData$default, predicted.test, threshold = optCutOff)

##Specificity=(# Actual 0's and Predicted as 0's)/(# of Actual 0's)
specificity(trainingData$default, predicted.training, threshold = 0.5)
specificity(trainingData$default, predicted.training, threshold = optCutOff)
specificity(testData$default, predicted.test, threshold = 0.5)
specificity(testData$default, predicted.test, threshold = optCutOff)


##Confusion Matrix
confusionMatrix(trainingData$default, predicted.training, threshold = 0.5)
confusionMatrix(trainingData$default, predicted.training, threshold = optCutOff)
confusionMatrix(testData$default, predicted.test, threshold = 0.5)
confusionMatrix(testData$default, predicted.test, threshold = optCutOff)
