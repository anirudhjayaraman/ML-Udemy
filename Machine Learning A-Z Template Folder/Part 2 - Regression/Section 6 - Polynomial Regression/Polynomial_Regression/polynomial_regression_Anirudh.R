# Polynomial Regression

# set working directory
setwd("F:/ML-Udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression")

datset <- read.csv('Position_Salaries.csv')

# look at variable types
sapply(datset, class)

# only include relevant variables
datset <- datset[,2:3]

# first fit a linear model
linearModel <- lm(Salary ~ Level, data = datset)
# summarize the model
summary(linearModel)


# now fit a polynomial model
datset$Level2 <- datset$Level^2
datset$Level3 <- datset$Level^3

polyRegModel <- lm(Salary ~ Level + Level2 + Level3, data = datset)
summary(polyRegModel)

# Visualizing and Comparing Model Results
library(ggplot2)
# Linear Model
ggplot() + 
  geom_point(aes(x = datset$Level, y = datset$Salary), color = 'red') + 
  geom_line(aes(x = datset$Level, y = predict(linearModel)), color = 'blue') +
  ggtitle('Linear Model Salary Predictions vs Acutal Salaries') +
  xlab(label = 'Level') +
  ylab(label = 'Salary')

# Polynomial Model
ggplot() + 
  geom_point(aes(x = datset$Level, y = datset$Salary), col = 'red') +
  geom_line(aes(x = datset$Level, y = predict(polyRegModel)), col = 'blue') + 
  ggtitle('Polynomial Regression Model Predictions vs Actual') + 
  xlab('Level') + 
  ylab('Salary')

# For smoother predictions with the polynomial model:

new_levels = seq(1,10,length.out = 100)

smoothPolyRegPreds <- predict(polyRegModel,
                              newdata = data.frame(Level = new_levels,
                                                   Level2 = new_levels^2,
                                                   Level3 = new_levels^3))

ggplot() +
  geom_point(aes(x = datset$Level, y = datset$Salary), col = 'red') +
  geom_line(aes(x = new_levels, y = smoothPolyRegPreds), col = 'blue') + 
  ggtitle('Polynomial Regression Model Predictions vs Actual') +
  xlab('Level') + 
  ylab('Salary')

  


