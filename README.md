# Airline_Passenger-satisfaction-Analysis-Prediction

## Business Implications for Commercial Airlines: -

-- Passengers with high online boarding satisfaction scores show 23% higher likelihood of satisfaction overall → loyalty program targeting opportunity

-- Business class travelers report higher inflight wifi satisfaction weight → premium service investment ROI signal

-- Personal travel purpose correlates with higher sensitivity to service touchpoints → segmentation-driven outreach strategy


# 📌 Objective:

The goal of this project is to analyze airline passenger satisfaction based on in-flight service metrics, demographic details, and travel-related features. The project applies Exploratory Data Analysis (EDA) and builds classification models (Decision Tree and Neural Network) to predict whether a passenger is “satisfied” or “neutral or dissatisfied.”


# 📁 Dataset:

Source: Airline passenger satisfaction dataset (Kaggle or internal)

Split: Training and test datasets (airline_passenger_train.csv, airline_passenger_test.csv)

Target Variable: satisfaction

# 🛠️ Tools Used

R (rpart, caret, neuralnet, ggplot2, base plotting)

Data cleaning, feature engineering, model training, and evaluation


#🔍 Key Steps:


# 1. Data Preprocessing

Removed missing and empty values

Dropped irrelevant or highly correlated features (e.g., Arrival.Delay.in.Minutes)

Converted id column into row names

Ensured consistent feature alignment between training and test sets


# 2. Exploratory Data Analysis (EDA)

Pie Charts: Showed satisfaction split by Customer Type, Gender, Class, and Travel Type

Bar Charts: Compared mean service ratings across satisfied vs. dissatisfied passengers

Boxplots: Analyzed age and flight distance variations by satisfaction level

Heatmap: Revealed strong correlation between arrival and departure delays


# 3. Modeling: 

Decision Tree Classifier
Used rpart to build an interpretable classification tree

Identified key predictors: Online.boarding, Inflight.wifi.service, Type.of.Travel

Applied cross-validation and cost-complexity pruning to optimize the tree

Evaluated performance using confusionMatrix


# 4. Modeling:

Neural Network
Normalized numerical features and converted categorical ones to numeric

Built a 2-layer neural network using the neuralnet package

Achieved strong classification performance, comparable to decision tree


## Model Performance

| Metric            | Decision Tree (Test) |
| ----------------- | -------------------- |
| Accuracy          | **91.2%**            |
| Sensitivity       | 88.99%               |
| Specificity       | 94.12%               |
| Kappa Score       | 0.8237               |
| Balanced Accuracy | 91.55%               |



## 💡 Insights

Passengers with higher online boarding, inflight service, and food ratings are more likely to be satisfied.

Loyal customers and personal travelers report higher satisfaction.

Flight distance and age show mild influence, with longer flights and older passengers generally more satisfied.


## 🧠 Conclusion


This project demonstrates how to combine EDA, interpretable machine learning, and model validation techniques to uncover the drivers of passenger satisfaction. 

The final model is both accurate and explainable, making it suitable for actionable business insights in the airline industry.
