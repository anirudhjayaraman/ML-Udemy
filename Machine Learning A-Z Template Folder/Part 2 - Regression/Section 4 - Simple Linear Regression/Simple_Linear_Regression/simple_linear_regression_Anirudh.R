setwd("F:/ML_Udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression")

set.seed(123)

dat <- read.csv('Salary_Data.csv')

library(caTools)
spl <- sample.split(dat$Salary, SplitRatio = 2/3)
train <- subset(dat, spl == T)
test <- subset(dat, spl == F)

linear_model <- lm(Salary ~ YearsExperience, data = train)
summary(linear_model)
salary_test_preds <- predict(linear_model, newdata = test)

library(ggplot2)

# Visualizing training set results
ggplot() + 
  geom_point(aes(x = train$YearsExperience, y = train$Salary), colour = 'red') +
  geom_line(aes(x = train$YearsExperience, y = predict(linear_model)), colour = 'blue') +
  ggtitle(label = 'Linear Model fit to Data') +
  xlab('Years of Experience') + ylab('Salary')

# Visualizing test set results
ggplot() + 
  geom_point(aes(x = test$YearsExperience, y = test$Salary), col = 'red') + 
  geom_line(aes(x = test$YearsExperience, y = salary_test_preds), col = 'blue') + 
  ggtitle('Linear Model Predictions on Test Set vs Actual') +
  xlab('Years of Experience') + ylab('Salary')