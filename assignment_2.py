import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.inspection import PartialDependenceDisplay

# Assignment No and Group details
from sklearn.utils import estimator_html_repr
import streamlit.components.v1 as components

st.subheader('Assignment 2(TIMG 5301 - Group 9)- Lamya Raisa, Muhammed Haris, Manimalavan Dilipkumar, Shivani Sharma')
st.markdown(
    f'<h1 style="color:#00008B;font-size:32px;">{""}</h1>',
    unsafe_allow_html=True)

# Step 1: Importing the dataset

df=pd.read_csv("C:\Desktop Files\crunchbase.csv")
print(df)

st.title('Crunchbase Data')
num_comp = df.count()[0]

# Step 2. Exploratory Data Analysis

# Counting the number of companies

if st.sidebar.button("Preprocessed Data"):
    st.subheader("Preprocessed Data")
    st.write(df)
    st.write('Number of companies initially provided:', num_comp)

# Creating a new target variable (success) with two values: 1 for success and 0 for failure.
# Using the definition of startup success provided to determine the value of the target variable.
# We have considered the startup to be failure if all the three conditions on IPO,
# is_acquired and is_closed is mentioned as False

def success(row):
    if row['ipo']==True or row['is_acquired']==True and row['is_closed']==False:
        return 1
    elif row['ipo']== False and row['is_acquired']==False and row['is_closed']==False:
        return 0
    else:
        return 0
df['success']=df.apply(success, axis=1)

df = df.drop('ipo', axis=1)
df = df.drop('is_acquired', axis=1)
df = df.drop('is_closed', axis=1)

# Missing Values

df['mba_degree'] = df['mba_degree'].fillna(0)
df['phd_degree'] = df['phd_degree'].fillna(0)
df['ms_degree'] = df['ms_degree'].fillna(0)
df['other_degree'] = df['other_degree'].fillna(0)

# Combining the features related to the education levels of the founders (mba_degree, phd_degree, ms_degree, other_degree)
# into a new feature for the total number of degrees obtained by the founders (number_degrees).

df['number_degrees'] = ""
column_names = ['mba_degree', 'phd_degree', 'ms_degree', 'other_degree']
df['number_degrees'] = df[column_names].sum(axis=1)

# dropping the education level columns of founders after combining these features into number_degrees attribute.

df = df.drop('mba_degree', axis=1)
df = df.drop('phd_degree', axis=1)
df = df.drop('ms_degree', axis=1)
df = df.drop('other_degree', axis=1)

# Identifying the numerical features in the dataset and showing their correlations with one another and the target in a heatmap.

numerical_features = ['average_funded', 'total_rounds', 'average_participants', 'products_number', 'acquired_companies',
                      'offices', 'age']
numerical_features_and_target = numerical_features + ['success']

# Creating the heatmap using corelation matrix.

if st.sidebar.button("Show correlation matrix"):
    st.subheader("Correlation matrix")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(df[numerical_features_and_target].corr(), annot=True, fmt=".2f", ax=ax,cmap=plt.cm.Blues)
    ax.set_title("Correlations of numerical features and Success")
    st.pyplot(fig)


# Identifying the categorical features in the dataset

categorical_features = ['category_code', 'country_code', 'state_code']


#  Computation of the missing values ratio for all features

def missing_values_ratios(df):
    return df.isna().sum() / len(df)


if st.sidebar.button("Show missing values ratios"):
    st.subheader("Missing values ratios")
    st.write(missing_values_ratios(df))

# Handling Duplicates

num_duplicate_rows = df.duplicated().sum()

if st.sidebar.button("Duplicate rows"):
    st.subheader('Number of duplicate rows')
    st.write("%d rows are duplicates" % (num_duplicate_rows))

if st.sidebar.button("Processed Data"):
    st.subheader("Processed Data")
    st.write(df)

# Checking for Class Imbalance

# Display the option in the sidebar
# Checking for Class Imbalance
class_counts = df["success"].value_counts()
total_samples = len(df["success"])

# Calculate class percentages
class_percentages = class_counts / total_samples * 100

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.8))
class_counts.plot(kind="bar", ax=ax, color=['red', 'green'])

# Annotate bars with percentages
for i, count in enumerate(class_counts):
    percentage = class_percentages[i]
    ax.text(i, count + 0.1, f"{percentage:.2f}%", ha='center', va='bottom')

ax.set_title("Class Imbalance")
ax.set_xlabel("success")
ax.set_ylabel("Count")

if st.sidebar.button("Show Class Imbalance"):
    st.subheader("Class Imbalance")
    st.pyplot(fig)

# Display the imbalance chart in the main area if the checkbox is selected


# Step 3. Modelling

# Pipeline 1 for pre-processing numerical and categorical features to counter missing values

def pre_processor(numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                            ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=
                                     [('num', numerical_transformer, numerical_features),
                                      ('cat', categorical_transformer, categorical_features)])
    return preprocessor


# Pipeline 2 for Modelling

preprocessor = pre_processor(numerical_features, categorical_features)
type_of_classifier = st.sidebar.radio("Select type of classifier", ("Random Forest", "Logistic Regression"))

if type_of_classifier == "Random Forest":
    classifier = RandomForestClassifier(random_state=1, max_depth=14, min_samples_split=2, min_samples_leaf=1,
                                        class_weight='balanced', )
elif type_of_classifier == "Logistic Regression":
    classifier = LogisticRegression(max_iter=2000, penalty='l2',
                                    class_weight='balanced')
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

# features X and target y (y will not be part of input)

X = df.drop('success', axis=1)
y = df['success']

# Choosing train and test datasets
# Split the data into 70% training and 30% testing, we use the stratified sampling in shuffling because our sample is not random

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)

# Step 4: Evaluation

#Removing imbalance using k-fold cross-validation and evaluating the model

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_train, y_train, scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'], cv=cv,
                        n_jobs=-1)
if st.sidebar.button("Cross validation Performance"):
    st.subheader('Cross Validation Performance')
    # Performance scores for each fold
    st.write("Scores for each fold (only positive class):")
    df_scores = pd.DataFrame(scores).transpose()
    df_scores['mean'] = df_scores.mean(axis=1)
    st.dataframe(df_scores)

# Evaluate the model against the test dataset
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

if st.sidebar.button('Model Performance'):
    st.subheader('Model performance')
    st.write(f"The table shows the performance of the {type_of_classifier} classifier on the test data.")
    # Scores from applying the model to the test dataset
    score = metrics.classification_report(y_test, y_pred, output_dict=True)
    # Add AUC score to the report
    score['auc'] = metrics.roc_auc_score(y_test, y_pred)
    score = pd.DataFrame(score).transpose()
    st.write("Scores from applying the model to the test dataset:")
    st.write(score)

if st.sidebar.button('Confusion matrix'):
    st.subheader('Confusion matrix')
    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap=plt.cm.Purples)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes'])
    ax.yaxis.set_ticklabels(['No', 'Yes'])
    st.pyplot(fig)