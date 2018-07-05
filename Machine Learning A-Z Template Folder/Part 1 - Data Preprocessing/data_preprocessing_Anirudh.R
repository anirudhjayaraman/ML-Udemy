# Importing the dataset
dataset <- read.csv('Data.csv')

# taking care of missing data
dataset$Age <- ifelse(is.na(dataset$Age),
                      mean(dataset$Age, na.rm = T),dataset$Age)

dataset$Salary <- ifelse(is.na(dataset$Salary),
                               mean(dataset$Salary, na.rm = T), dataset$Salary)

# encoding categorical data
dataset$Country <- factor(dataset$Country,
                          levels = c('France','Spain','Germany'),
                          labels = c(0,1,2))

dataset$Purchased <- factor(dataset$Purchased,
                            levels = c('Yes', 'No'),
                            labels= c(1,0))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[,c('Age','Salary')] <- scale(training_set[,c('Age','Salary')])
test_set[,c('Age','Salary')] <- scale(test_set[,c('Age','Salary')])
