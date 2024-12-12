# ? Source - https://www.kaggle.com/competitions/playground-series-s4e12

# Loading libraries ####
library(dplyr)
library(tidyr)
library(stringr)
library(janitor) # Column names cleaning
library(missMDA) # Missing value imputation
library(caret) # Data splitting
# library(randomForest) # Random Forest model
library(ranger) # Random Forest model
library(doParallel) # Parallel processing

# Reading data ####
df_train <- read.csv("data/train.csv", header = TRUE)
df_test <- read.csv("data/test.csv", header = TRUE)

# Preprocessing ####
df_train <- clean_names(df_train)
df_test <- clean_names(df_test)

# Data Cleaning ####
## Outlier Detection and Removal ####

# Remove outliers
outliers <- boxplot.stats(df_train$`premium_amount`)$out
df_train <- df_train[-which(df_train$`premium_amount` %in% outliers), ]

## Missing Value Imputation ####
# Drop NA rows
df_train <- df_train %>% drop_na()

# Convert categorical variables to factors ####
df_train[sapply(df_train, is.character)] <-
  lapply(df_train[sapply(df_train, is.character)], as.factor)

df_test[sapply(df_test, is.character)] <-
  lapply(df_test[sapply(df_test, is.character)], as.factor)

# Remove factor columns ####
df_train <- df_train %>%
  select(-where(is.factor))

# Splitting data into train and test sets ####
# Set seed for reproducibility
set.seed(7991)

# Split the data
train_indices <- createDataPartition(df_train$premium_amount,
                                     p = 0.7, list = FALSE)

# Create train and test sets
train_set <- df_train[train_indices, ]
test_set <- df_train[-train_indices, ]

# Check dimensions
cat("Training set size:", nrow(train_set), "\n")
cat("Test set size:", nrow(test_set), "\n")

# Set up parallel processing ####
num_cores <- detectCores() - 2  # Use one less core than available
cl <- makeCluster(num_cores)
registerDoParallel(cl)


# Random Forest Model ####
## Define training control ####
train_control <- trainControl(
  method = "cv",                # Cross-validation
  number = 5,                   # Number of folds
  search = "grid",              # Grid search for hyperparameter tuning
  allowParallel = TRUE          # Enable parallel processing
)

## Define hyperparameter grid ####
grid <- expand.grid(
  mtry = c(5, 10, 15),         # Number of features considered at each split
  splitrule = c("gini", "extratrees"),  # Splitting rules
  min.node.size = c(5, 10, 20) # Minimum node size
)

## Train the model ####
train_set <- train_set %>% select(-id)

rf_model <- train(
  premium_amount ~ .,
  data = train_set,
  method = "ranger",
  trControl = train_control,
  tuneGrid = grid,
  metric = "RMSE"
)

## Stop the parallel cluster ####
stopCluster(cl)
registerDoSEQ()

# Print best model
print(rf_model)

# Predict on test data
predictions <- predict(rf_model, newdata = test_set)

# Confusion Matrix
confusion_matrix <- table(Predicted = predictions,
                          Actual = test_set$premium_amount)
print(confusion_matrix)

# Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 2)))