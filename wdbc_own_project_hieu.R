#######################################################
# Name:     Hieu Nguyen                               #
# Course:   Data Science Capstone - Own Project       #
# Project:  Wisconsin Diagnostic Breast Cancer (WDBC) #
# Date:     Oct 2021                                  #
# Email:    hieunguyenthanh@gmail.com                 #
#######################################################

# Check and install libraries if not exist
if(!require(tidyverse)) install.packages("tidyverse",
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret",
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table",
                                          repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats",
                                           repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2",
                                       repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2",
                                        repos = "http://cran.us.r-project.org")
if(!require(psych)) install.packages("psych",
                                     repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(matrixStats)
library(ggplot2)
library(reshape2)
library(psych)


# Download Breast Cancel Dataset from Machine Learning Repository, 
# UCI:  http://archive.ics.uci.edu/ml/index.php
dl <- tempfile()
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases"
file <- "breast-cancer-wisconsin/wdbc.data"
download.file(paste(url, file, sep = "/"),dl)


# Dataset is in CSV format without header
data_raw <- read.csv(dl, header = FALSE)

# Assign columns name
features_list <- c("radius", "texture", "perimeter", "area", "smoothness",
                   "compactness", "concavity", "concave_points", "symmetry",
                   "fractal_dimension")

colnames(data_raw) <- c("ID", "diagnostic", 
                    paste(features_list,"mean", sep = "_"), 
                    paste(features_list,"se", sep = "_"),
                    paste(features_list, "worst", sep = "_"))

# Remove ID column
wdbc <- data_raw[-1]

# Convert diagnostic value to 0/1: B (Benign) = 0 and M (Malignant) = 1
wdbc[which(wdbc[,1] == "B"),1] <- 0
wdbc[which(wdbc[,1] == "M"),1] <- 1
wdbc[,1] <- strtoi(wdbc[,1])
wdbc <- as.matrix(wdbc)

rm(dl, features_list)

# #####################
#  Exploring Data     #
#######################

str(data_raw)

# Number of Benign and Malignant
data_raw %>% group_by(diagnostic) %>% 
  summarize(count = n(), count_percent = round(count/nrow(data_raw)*100)) %>%
  knitr::kable()

# Plot PIE chart for % of B and M
sl <- c(sum(data_raw$diagnostic == "B"), sum(data_raw$diagnostic == "M"))
lbs <- c("Benign", "Malignant")
sl_pct <- round(sl / sum(sl) * 100)
lbs <- paste(lbs, sl, sep = " - ")
lbs <- paste(lbs, sl_pct, sep = " - ")
lbs <- paste(lbs, "%", sep ="")
pie(sl, labels = lbs, col = rainbow(length(sl)), main = "B and M %")


# Plot HeatMap of correlation among diagnostic and all features
wdbc_cor_melted <- melt(cor(wdbc))

ggheatmap <- ggplot(data = wdbc_cor_melted, aes(Var2, Var1, fill = value)) + 
  geom_tile(color = "white") + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Cor.") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 90, hjust =1)) +
  coord_fixed()

ggheatmap + geom_text(aes(Var2, Var1, label=round(value,2)), color="black", size=2) +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank())



# Visualize correlation among features via Pairs Plot

# Pairs Plot of Diagnostic and all mean features
pairs.panels(wdbc[,c(1:11)], pch = 21 + as.numeric(wdbc[,1]), 
             bg = c("green", "red")[wdbc[,1] + 1], ellipses = FALSE, 
             cex.cor = 1.5)

# Pairs Plot of Diagnostic and all se features
pairs.panels(wdbc[,c(1, 12:21)], pch = 21 + as.numeric(wdbc[,1]), 
             bg = c("green", "red")[wdbc[,1] + 1], ellipses = FALSE, 
             cex.cor = 1.5)

# Pairs Plot of Diagnostic and all worst features
pairs.panels(wdbc[,c(1, 22:31)], pch = 21 + as.numeric(wdbc[,1]), 
             bg = c("green", "red")[wdbc[,1] + 1], ellipses = FALSE, 
             cex.cor = 1.5)

# Pairs Plot of mean and worst features: first 5 pairs features
pairs.panels(wdbc[,c(2:6, 22:26)], pch = 21 + as.numeric(wdbc[,1]), 
             bg = c("green", "red")[wdbc[,1] + 1], ellipses = FALSE, 
             cex.cor = 1.5)

# Pairs Plot of mean and worst features: last 5 pairs features
pairs.panels(wdbc[,c(7:11, 27:31)], pch = 21 + as.numeric(wdbc[,1]), 
             bg = c("green", "red")[wdbc[,1] + 1], ellipses = FALSE, 
             cex.cor = 1.5)


# From plot, we identify highly correlated features:
# - radius_ highly correlated with perimeter_ and area_ --> remove perimeter_, area_
# - concavity_mean highly correlated with concave_points_mean --> remove concave_points_mean
# - All _worst features except symmetry_worst and fractal_dimension_worst are 
#   highly correlated with _mean features respectively --> remove those _worst features

# Plot HeatMap of correlation among diagnostic and features after removing
wdbc_cor_dropped_melted <- melt(cor(wdbc[,c(-4,-5,-8,-14,-15,-22:-29)]))

ggheatmap_dropped <- ggplot(data = wdbc_cor_dropped_melted, 
                            aes(Var2, Var1, fill = value)) + 
  geom_tile(color = "white") + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Cor.") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 90, hjust =1)) +
  coord_fixed()

ggheatmap_dropped + geom_text(aes(Var2, Var1, label = round(value,2)), 
                              color = "black", size = 2) +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank())


#########################################################################
# Create train and test dataset                                         #
# and calculate the Accuracy of different ML models: glm, lda, qda,     #
# loess, knn, rf on original dataset and dropped features dataset       #
#########################################################################

# Scaling WDBC data to standard distribution
wdbc_scaled <- sweep(wdbc[,-1], 2, colMeans(wdbc[,-1]))
wdbc_scaled <- sweep(wdbc_scaled, 2, colSds(wdbc[,-1]), FUN = "/")


# Creat training and testing data 
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = as.factor(wdbc[,1]), 
                                  times = 1, p = 0.25, list = FALSE)

train_x <- wdbc_scaled[-test_index,]
train_x_dropped <- train_x[,c(-3,-4,-7,-13,-14,-21:-28)]
train_y <- as.factor(wdbc[-test_index, 1])

test_x <- wdbc_scaled[test_index,]
test_x_dropped <- test_x[,c(-3,-4,-7,-13,-14,-21:-28)]
test_y <- as.factor(wdbc[test_index, 1])

# % of train and test dataset
data.frame(total = nrow(data_raw),
           train = nrow(train_x),
           train_percent = round(nrow(train_x)/nrow(data_raw)*100),
           test = nrow(test_x),
           test_percent = round(nrow(test_x)/nrow(data_raw)*100),
           row.names = "Number of Observation"
) %>% knitr::kable()

# Cross check ratio of B and M in train and test set to ensure they are similar
data.frame(
  diagnostic = c("B", "M"),
  total = c(sum(data_raw$diagnostic == "B"), sum(data_raw$diagnostic == "M")),
  total_percent = c(round(sum(data_raw$diagnostic == "B")/nrow(data_raw)*100),
                    round(sum(data_raw$diagnostic == "M")/nrow(data_raw)*100)),
  train_y = c(sum(train_y == 0), sum(train_y == 1)),
  train_y_per = c(round(sum(train_y == 0)/length(train_y)*100),
                  round(sum(train_y == 1)/length(train_y)*100)),
  test_y = c(sum(test_y == 0), sum(test_y == 1)),
  test_y_per = c(round(sum(test_y == 0)/length(test_y)*100),
                 round(sum(test_y == 1)/length(test_y)*100))
) %>% knitr::kable()


# Running ML models on full features and features dropped dataset

# Logistic regression model
set.seed(1, sample.kind = "Rounding") 
train_glm <- train(train_x, train_y, method = "glm")
glm_preds <- predict(train_glm, test_x)

set.seed(1, sample.kind = "Rounding") 
train_glm_dropped <- train(train_x_dropped, train_y, method = "glm")
glm_preds_dropped <- predict(train_glm_dropped, test_x_dropped)

# Create Accuracy Table of all ML model
accuracy_table <- data.frame(
  glm = c(mean(glm_preds == test_y), mean(glm_preds_dropped == test_y)), 
  row.names = c("Full Features","Features-Dropped"))


# LDA model
set.seed(1, sample.kind = "Rounding") 
train_lda <- train(train_x, train_y, method = "lda")
lda_preds <- predict(train_lda, test_x)

set.seed(1, sample.kind = "Rounding") 
train_lda_dropped <- train(train_x_dropped, train_y, method = "lda")
lda_preds_dropped <- predict(train_lda_dropped, test_x_dropped)

accuracy_table <- accuracy_table %>% mutate(lda = c(mean(lda_preds == test_y),
                                              mean(lda_preds_dropped == test_y)))

# QDA model
set.seed(1, sample.kind = "Rounding")
train_qda <- train(train_x, train_y, method = "qda")
qda_preds <- predict(train_qda, test_x)

set.seed(1, sample.kind = "Rounding")
train_qda_dropped <- train(train_x_dropped, train_y, method = "qda")
qda_preds_dropped <- predict(train_qda_dropped, test_x_dropped)

accuracy_table <- accuracy_table %>% mutate(qda = c(mean(qda_preds == test_y),
                                              mean(qda_preds_dropped == test_y)))

# Loess model
set.seed(1, sample.kind = "Rounding")
train_loess <- train(train_x, train_y, method = "gamLoess")
loess_preds <- predict(train_loess, test_x)

set.seed(1, sample.kind = "Rounding")
train_loess_dropped <- train(train_x_dropped, train_y, method = "gamLoess")
loess_preds_dropped <- predict(train_loess_dropped, test_x_dropped)

accuracy_table <- accuracy_table %>% mutate(loess = c(mean(loess_preds == test_y),
                                              mean(loess_preds_dropped == test_y)))


# K-nearest neighbors model
set.seed(1, sample.kind = "Rounding")
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(3, 21, 2)))
knn_preds <- predict(train_knn, test_x)

set.seed(1, sample.kind = "Rounding")
train_knn_dropped <- train(train_x_dropped, train_y,
                           method = "knn", 
                           tuneGrid = data.frame(k = seq(3, 21, 2)))
knn_preds_dropped <- predict(train_knn_dropped, test_x_dropped)

accuracy_table <- accuracy_table %>% mutate(knn = c(mean(knn_preds == test_y),
                                              mean(knn_preds_dropped == test_y)))

# Random forest model
set.seed(1, sample.kind = "Rounding")
train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = data.frame(mtry = c(3, 5, 7, 9)),
                  importance = TRUE)
rf_preds <- predict(train_rf, test_x)

set.seed(1, sample.kind = "Rounding")
train_rf_dropped <- train(train_x_dropped, train_y,
                          method = "rf",
                          tuneGrid = data.frame(mtry = c(3, 5, 7, 9)),
                          importance = TRUE)
rf_preds_dropped <- predict(train_rf_dropped, test_x_dropped)

accuracy_table <- accuracy_table %>% mutate(rf = c(mean(rf_preds == test_y),
                                              mean(rf_preds_dropped == test_y)))

# Ensemble Model: combining results from 6 models 
# (glm, lda, qda, loess, knn, rf) to predict output
ensemble <- cbind(glm = glm_preds == 0, lda = lda_preds == 0,
                  qda = qda_preds == 0, loess = loess_preds == 0, 
                  knn = knn_preds == 0, rf = rf_preds == 0)

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, 0, 1)

ensemble_dropped <- cbind(glm_dropped = glm_preds_dropped == 0, 
                  lda_dropped = lda_preds_dropped == 0,
                  qda_dropped = qda_preds_dropped == 0, 
                  loess_dropped = loess_preds_dropped == 0, 
                  knn_dropped = knn_preds_dropped == 0, 
                  rf_dropped = rf_preds_dropped == 0)

ensemble_preds_dropped <- ifelse(rowMeans(ensemble_dropped) > 0.5, 0, 1)

accuracy_table <- accuracy_table %>% mutate(ensemble = c(mean(ensemble_preds == test_y),
                                              mean(ensemble_preds_dropped == test_y)))


# Create Ensemble Mixed model by:
#   - selecting highest prediction accuracy between two datasets, then
#   - combining best results from 6 models (glm, lda, qda, loess, knn, rf) 
#     to predict output

if (accuracy_table[1,1] > accuracy_table[2,1]) {
  glm_mixed <- glm_preds == 0} else {glm_mixed <- glm_preds_dropped == 0}

if (accuracy_table[1,2] > accuracy_table[2,2]) {
  lda_mixed <- lda_preds == 0} else {lda_mixed <- lda_preds_dropped == 0}

if (accuracy_table[1,3] > accuracy_table[2,3]) {
  qda_mixed <- qda_preds == 0} else {qda_mixed <- qda_preds_dropped == 0}

if (accuracy_table[1,4] > accuracy_table[2,4]) {
  loess_mixed <- loess_preds == 0} else {loess_mixed <- loess_preds_dropped == 0}

if (accuracy_table[1,5] > accuracy_table[2,5]) {
  knn_mixed <- knn_preds == 0} else {knn_mixed <- knn_preds_dropped == 0}

if (accuracy_table[1,6] > accuracy_table[2,6]) {
  rf_mixed <- rf_preds == 0} else {rf_mixed <- rf_preds_dropped == 0}

ensemble_mixed <- cbind(glm_mixed = glm_mixed, lda_mixed = lda_mixed,
                        qda_mixed = qda_mixed, loess_mixed = loess_mixed, 
                        knn_mixed = knn_mixed, rf_mixed = rf_mixed)

ensemble_preds_mixed <- ifelse(rowMeans(ensemble_mixed) > 0.5, 0, 1)

accuracy_ensemble_mixed <- mean(ensemble_preds_mixed == test_y)

# Prediction Accuracy Results:
round(accuracy_table*100,3) %>% knitr::kable()

print(paste("Prediction Accuracy of Ensemble Mixed model is:",
            round(accuracy_ensemble_mixed*100,3)))


# END OF SCRIPT
