setwd("F:/ML_Udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression")

dat <- read.csv('50_Startups.csv')

# Encode the categorical Variable
dat$State <- factor(dat$State,
                    levels = c('California','Florida','New York'),
                    labels = c(3,2,1))

library(caTools)

set.seed(123)
spl <- sample.split(dat$Profit, SplitRatio = 0.8)
dat_train <- subset(dat, spl == T)
dat_test <- subset(dat, spl == F)

regressor <- lm(Profit ~ ., data = dat_train)
summary(regressor)

# predict test set results
test_preds <- predict(regressor, newdata = dat_test)

# building optimal model using backwarad elimination
# Step 1
regressor <- lm(Profit ~ ., data = dat_train)
summary(regressor)

# Step 2
regressor <- lm(Profit ~ . - State, data = dat_train)
summary(regressor)

# Step 3
regressor <- lm(Profit ~ . - State - Administration, data = dat_train)
summary(regressor)

# Step 4
regressor <- lm(Profit ~ . - State - Administration - Marketing.Spend, data = dat_train)
summary(regressor)

# predict test set results
test_preds <- predict(regressor, newdata = dat_test)
