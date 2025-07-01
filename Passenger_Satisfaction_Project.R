library(rpart)                    # For decision tree modeling
library(rpart.plot)               # For visualizing decision trees
library(caret)                    # For evaluation (e.g., confusion Matrix)

# A confusion matrix is a table used to evaluate the performance of a classification model. It shows the 
# number of correct and incorrect predictions made by the model compared to the actual values in the data.

# Training Dataset - To teach the model how to make predictions.It contains 
# Input features (like gender, age, flight distance, service ratings, etc.)
# Target/output variable (like satisfaction: satisfied / not satisfied)

# Test Dataset - To evaluate how well the trained model performs on unseen data.

#Fetching the data
passenger_train.df <- read.csv("C:/Users/sachi/Downloads/airline_passenger_train.csv", header = TRUE)
passenger_test.df <- read.csv("C:/Users/sachi/Downloads/airline_passenger_test.csv")

head(passenger_train.df)
head(passenger_test.df)

# Working with the training dataset
# Summary Statistics is used to see if there are any missing values, unusual distributions,etc..
summary(passenger_train.df)

# Loop over categorical variables to check NA/empty strings in categorical columns.because Missing data can bias the model.
for (col in colnames(passenger_train.df)) {
  # Check if the column is categorical (factor or character)
  if (is.factor(passenger_train.df[[col]]) || is.character(passenger_train.df[[col]])) {
    na_count <- sum(is.na(passenger_train.df[[col]]))
    empty_count <- sum(passenger_train.df[[col]] == "", na.rm = TRUE)
    cat(
      "Column:", col, 
      "- NA values:", na_count, 
      "- Empty strings:", empty_count, "\n"
    )
  }
}

#Remove na values
passenger_train.df <- na.omit(passenger_train.df)

#EDA analysis

# 1) Pie charts for categorical variables
satisfaction_loyalty <- table(passenger_train.df$satisfaction, passenger_train.df$Customer.Type)
satisfaction_loyalty
par(mfrow = c(1, 2), pin = c(2,2))  # Set the plot area to have 1 row and 2 columns

# Pie chart for satisfied customers, Counts of satisfaction by loyalty status, gender, class, travel type
# the piechart implies A large majority (≈90%) of satisfied customers are loyal & few are disloyal
satisfied_counts <- satisfaction_loyalty["satisfied", ]
pie(satisfied_counts, main = "Satisfied Customers", 
    col = c("lightblue", "lightgreen"), 
    labels = paste(names(satisfied_counts), "\n", satisfied_counts))

# Pie chart for non-satisfied customers
# Among unsatisfied passengers, disloyal customers form a much larger proportion (≈25%) 
# compared to their share in the "satisfied" group (≈10%).
non_satisfied_counts <- satisfaction_loyalty["neutral or dissatisfied", ]
pie(non_satisfied_counts, main = "neutral or dissatisfied", 
    col = c("salmon", "lightpink"), 
    labels = paste(names(non_satisfied_counts), "\n", non_satisfied_counts))

# Customer loyalty is a major predictor of satisfaction.
# While most customers are loyal, disloyal passengers are 2.5× more likely to be dissatisfied than satisfied.
# This suggests that building loyalty programs could significantly improve overall passenger satisfaction.

#-----Gender-------
Gender_satisfaction <- table(passenger_train.df$satisfaction, passenger_train.df$Gender)
par(mfrow = c(1, 2), pin = c(2,2))  # Set the plot area to have 1 row and 2 columns

# Pie chart for satisfied customers
# This suggests that gender does not have a major impact on satisfaction in this dataset.
satisfied_counts <- Gender_satisfaction["satisfied", ]
pie(satisfied_counts, main = "Satisfied Customers", 
    col = c("lightblue", "lightgreen"), 
    labels = paste(names(satisfied_counts), "\n", satisfied_counts))

# Pie chart for non-satisfied customers
non_satisfied_counts <- Gender_satisfaction["neutral or dissatisfied", ]
pie(non_satisfied_counts, main = "neutral or dissatisfied", 
    col = c("salmon", "lightpink"), 
    labels = paste(names(non_satisfied_counts), "\n", non_satisfied_counts))

#------Class------------
Class_satisfaction <- table(passenger_train.df$satisfaction, passenger_train.df$Class)
par(mfrow = c(1, 2), pin = c(2,2))  # Set the plot area to have 1 row and 2 columns

# Pie chart for satisfied customers
satisfied_counts <- Class_satisfaction["satisfied", ]
pie(satisfied_counts, main = "Satisfied Customers", 
    col = c("lightblue", "lightgreen","yellow"), 
    labels = paste(names(satisfied_counts), "\n", satisfied_counts))

# Pie chart for non-satisfied customers
non_satisfied_counts <- Class_satisfaction["neutral or dissatisfied", ]
pie(non_satisfied_counts, main = "neutral or dissatisfied", 
    col = c("salmon", "lightpink","yellow"), 
    labels = paste(names(non_satisfied_counts), "\n", non_satisfied_counts))

#-------Type of travel----------
travelType_satisfaction <- table(passenger_train.df$satisfaction, passenger_train.df$Type.of.Travel)
par(mfrow = c(1, 2), pin = c(2,2))  # Set the plot area to have 1 row and 2 columns

# Pie chart for satisfied customers
satisfied_counts <- travelType_satisfaction["satisfied", ]
pie(satisfied_counts, main = "Satisfied Customers", 
    col = c("lightblue", "lightgreen"), 
    labels = paste(names(satisfied_counts), "\n", satisfied_counts))

# Pie chart for non-satisfied customers
non_satisfied_counts <- travelType_satisfaction["neutral or dissatisfied", ]
pie(non_satisfied_counts, main = "neutral or dissatisfied", 
    col = c("salmon", "lightpink"), 
    labels = paste(names(non_satisfied_counts), "\n", non_satisfied_counts))

#------Bar graphs---------------
df<-passenger_train.df
colnames(df)

satisfied_customers <- df[df$satisfaction == "satisfied", ]
unsatisfied_customers <- df[df$satisfaction == "neutral or dissatisfied", ]

service_columns <- c("Inflight.service", "Ease.of.Online.booking", "Gate.location", 
                     "Food.and.drink", "Online.boarding", "Seat.comfort", 
                     "Inflight.entertainment", "Leg.room.service", "On.board.service", 
                     "Baggage.handling", "Inflight.wifi.service", "Cleanliness")

# Mean ratings for satisfied customers
# For satisfied customers only, this computes the average rating for each service listed.
# na.rm = TRUE: ensures any missing values (NA) don’t interfere with the average.
mean_satisfied <- colMeans(satisfied_customers[service_columns], na.rm = TRUE)

# Mean ratings for unsatisfied customers
mean_unsatisfied <- colMeans(unsatisfied_customers[service_columns], na.rm = TRUE)

# Step 3: Set up the bar plots
par(mfrow = c(1, 2))  # Create a layout for two plots side by side

# Bar plot for satisfied customers
barplot(mean_satisfied, 
        main = "Mean Ratings for Satisfied Customers",
        xlab = "Service Aspects", 
        ylab = "Mean Rating", 
        col = "lightblue", 
        las = 2,  # Make the axis labels perpendicular
        ylim = c(0, 5))  # Adjust the y-axis limit based on your rating scale

# Bar plot for unsatisfied customers
barplot(mean_unsatisfied, 
        main = "Mean Ratings for Unsatisfied Customers", 
        xlab = "Service Aspects", 
        ylab = "Mean Rating", 
        col = "salmon", 
        las = 2,  # Make the axis labels perpendicular
        ylim = c(0, 5))  # Adjust the y-axis limit based on your rating scale


#heatmap
hmap <- cor(passenger_train.df[,c("Age","Arrival.Delay.in.Minutes","Departure.Delay.in.Minutes","Flight.Distance")])
heatmap(cor(passenger_train.df[,c("Age","Arrival.Delay.in.Minutes","Departure.Delay.in.Minutes","Flight.Distance")]), Rowv = NA, Colv = NA)

# flights delayed at departure are often delayed at arrival. --- Strong red square → high positive correlation
# Age ↔ Others -- Age doesn’t meaningfully correlate with delays or distance. --- weak correlation with all variables
# Flight.Distance ↔ Age -- Slightly darker → small positive correlation

#Boxplot
#------satisfaction vs flight distance
# Ensure Satisfaction is a factor
df$satisfaction <- as.factor(df$satisfaction)

# Boxplot of Flight Distance by Satisfaction Level
boxplot(Flight.Distance ~ satisfaction, data = df,
        main = "Flight Distance by Satisfaction Level",
        xlab = "Satisfaction Level",
        ylab = "Flight Distance (miles)",
        col = c("lightblue", "salmon"),
        notch = TRUE)

# Satisfied passengers tend to fly longer distances, while unsatisfied passengers are more common on shorter flights.
# This suggests that longer flights may offer better or more premium services, contributing to higher satisfaction.

#satisfcation vs age
boxplot(Age ~ satisfaction, data = df,
        main = "Satisfaction Level by Age",
        xlab = "Satisfaction Level",
        ylab = "Age",
        col = c("lightblue", "salmon"),
        notch = TRUE)

# Preparing data for analysis
# Since the columns Departure.Delay.in.Minutes and Arrival.Delay.in.Minutes are highly correlated we are dropping the column Arrival.Delay.in.Minutes from the ML model

summary(passenger_train.df)
head(passenger_train.df)

passenger_train.df <- passenger_train.df[, !colnames(passenger_train.df) %in% c("X", "Arrival.Delay.in.Minutes")]
rownames(passenger_train.df) <- passenger_train.df$id
passenger_train.df <- passenger_train.df[,-1]

# Decision tree Analysis
# A Decision Tree is a Supervide ML Algo. used for classification and regression tasks
# it works like a flowchart splits data into branches based on questions about features,that leads to a decision (prediction) at the end of each path.
# 1) default
passenger.ct <- rpart(satisfaction ~ ., data =  passenger_train.df ,method = "class")
passenger.ct
# plot tree
prp(passenger.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = 10)
# count number of leaves
length(passenger.ct$frame$var[passenger.ct$frame$var == "<leaf>"])

train_pred <- predict(passenger.ct, passenger_train.df, type = "class")
confusionMatrix(train_pred, as.factor(passenger_train.df$satisfaction))

# The decision tree shows that online boarding experience, inflight wifi service, and type of travel are the top three predictors of passenger satisfaction.
# Most passengers who rated online boarding below 4 were predicted as unsatisfied, whereas personal travelers with higher ratings for boarding and wifi were generally satisfied.

# 2) Cross validation and optimized tree
# Cross-validation to assess performance and Cost-complexity pruning (cp) to simplify the tree
#  Pruning the decision tree using cross-validation helped reduce overfitting and simplified the model.
# The optimal tree was chosen at cp = 0.00593, where the validation error stopped improving significantly.

cv.ct <- rpart(satisfaction ~ ., data =  passenger_train.df ,method = "class", minsplit = 1, xval = 5)  
printcp(cv.ct)  

cv.ct <- rpart(satisfaction ~ ., data = passenger_train.df, method = "class", cp = 0.00001, minsplit = 1, xval = 5)  # xval is number K of folds in a K-fold cross-validation.
printcp(cv.ct)  # Print out the cp table of cross-validation errors. 
# Based on the cp table after 12 splits the xerror or relative decreases very slowly so choosing that value of cp for the pruned tree
pruned.ct <- prune(cv.ct, cp = 5.9300e-03)

train_pred <- predict(pruned.ct, passenger_train.df, type = "class")
confusionMatrix(train_pred, as.factor(passenger_train.df$satisfaction))

# 3) Classifying the validation dataset based on the pruned tree
passenger_test.df <- passenger_test.df[, !colnames(passenger_test.df) %in% c("X", "Arrival.Delay.in.Minutes")]
rownames(passenger_test.df) <- passenger_test.df$id
passenger_test.df <- passenger_test.df[,-1]
head(passenger_test.df)

test_pred <- predict(pruned.ct, passenger_test.df, type = "class")
confusionMatrix(test_pred, as.factor(passenger_test.df$satisfaction))




####NEURAL NETWORK----

if (!require(neuralnet)) install.packages("neuralnet")
library(neuralnet)
library(caret)

# Load and inspect the data
train_data <- passenger_train.df
test_data <- passenger_test.df

# Normalize numerical features for better neural network performance
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization to numerical columns (assuming satisfaction is the target variable)
numerical_cols <- sapply(train_data, is.numeric)
train_data[numerical_cols] <- lapply(train_data[numerical_cols], normalize)
test_data[numerical_cols] <- lapply(test_data[numerical_cols], normalize)

# Define the formula for neural network (assuming 'satisfaction' is the target)
target <- "satisfaction"
predictors <- setdiff(names(train_data), target)
formula <- as.formula(paste(target, "~", paste(predictors, collapse = "+")))

# Convert all non-numeric variables to numeric/dummies
library(dplyr)

# Ensure target is numeric (e.g., binary classification: 0/1)
train_data[[target]] <- as.numeric(as.factor(train_data[[target]])) - 1
test_data[[target]] <- as.numeric(as.factor(test_data[[target]])) - 1

# Convert all non-numeric predictors to numeric/dummy variables
train_data <- train_data %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), ~ as.numeric(.)))

test_data <- test_data %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), ~ as.numeric(.)))

# Check the structure of the dataset to ensure numeric predictors
str(train_data)

# Redefine the formula for training
formula <- as.formula(paste(target, "~", paste(setdiff(names(train_data), target), collapse = "+")))

# Train the neural network
set.seed(123)  # Ensure reproducibility
nn_model <- neuralnet(
  formula,
  data = train_data,
  hidden = c(5, 3),  # Two hidden layers with 5 and 3 neurons respectively
  linear.output = FALSE
)

# Visualize the neural network
plot(nn_model)

# Train the neural network
set.seed(123)  # Ensure reproducibility
nn_model <- neuralnet(
  formula,
  data = train_data,
  hidden = c(5, 3),  # Two hidden layers with 5 and 3 neurons respectively
  linear.output = FALSE
)

# Visualize the neural network
plot(nn_model)

# Predict on test data
nn_predictions <- compute(nn_model, test_data[predictors])$net.result

# Convert predictions to binary (assuming binary classification)
predicted_classes <- ifelse(nn_predictions > 0.5, 1, 0)

# Evaluate the model
confusionMatrix(as.factor(predicted_classes), as.factor(test_data[[target]]))
