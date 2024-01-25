#!/usr/bin/env python
# coding: utf-8

# In[31]:


##Hospital Re-Admission
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
import dtale
## importing the data ##
df=pd.read_csv("C:\\Users\\DELL\\Downloads\\Final Projects\\hospital_readmissions.csv")
#dtale.show(df)


# In[32]:


df.columns


# In[3]:


## Data Preprocessing
# Step 1
#Data Cleaning:
#Handle missing data: Decide whether to impute missing values or remove instances with missing values.
##Info on data
df.info()


# In[4]:


#Decription of Data
df.describe()


# In[5]:


## Handling missing data
df.isnull().sum()


# In[6]:


## filling the numnerical missing values in mean and categprical missing values in mode.
df['A1C_Result']=df['A1C_Result'].fillna(df['A1C_Result'].mode()[0])

##checking again the columns having any missing values
df.isnull().sum()*100/len(df)


# In[7]:


# Check for duplicates in the entire DataFrame
duplicates = df.duplicated()

# Display rows with duplicate values
if any(duplicates):
    print("Duplicate rows:")
    print(df[duplicates])
else:
    print("No duplicate rows.")


# In[8]:


#Boxplot
plt.boxplot(df[['Age','Num_Lab_Procedures','Num_Medications',
                'Num_Outpatient_Visits','Num_Inpatient_Visits','Num_Emergency_Visits','Num_Diagnoses']])
plt.show()


# In[9]:


## Data Exploration and Analysis
#Understand the distribution of the data through visualizations.

# Numerical Features: Histogram
#This helps to understand the range, central tendency, and spread of each variable.

numerical_columns = ['Age','Num_Lab_Procedures','Num_Medications',
                'Num_Outpatient_Visits','Num_Inpatient_Visits','Num_Emergency_Visits','Num_Diagnoses']

# Create subplots for numerical features
import seaborn as sn
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(4, 4, i)
    sn.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[10]:


df.columns


# In[11]:


# Categorical Features: Bar Plots
#It provides insights into the frequency of each category.

categorical_columns = ['Gender', 'Admission_Type', 'Diagnosis','A1C_Result','Readmitted']

# Creating subplots for categorical features
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, 5, i)
    sn.countplot(x=col, data=df, palette='pastel')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


# In[12]:


## Heat map
#This heatmap will help you visually identify the strength and direction of linear relationships between the numerical variables.
# High absolute correlation values (close to 1 or -1) may indicate a strong linear relationship

# Numerical Features for Correlation Analysis
numerical_columns = ['Age','Num_Lab_Procedures','Num_Medications',
                'Num_Outpatient_Visits','Num_Inpatient_Visits','Num_Emergency_Visits','Num_Diagnoses']

# Compute the correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Plot a heatmap for better visualization
plt.figure(figsize=(10, 8))
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[13]:


## Scatter Plot
sn.relplot(x='Num_Medications',y='Num_Outpatient_Visits',hue='Readmitted',data=df)


# In[14]:


#Multivariate Analysis:To see realtionships b/w all numeric variables
##Pair Plot
df=df.drop('Patient_ID',axis=1)
sn.pairplot(df)


# In[15]:


##Target Variable Analysis:
df['Readmitted'].value_counts()


# In[16]:


df.info()


# In[17]:


##Encoding Variables:
#In Label Encoding, each category in a categorical variable is assigned a unique integer label.
#--This is typically done in alphabetical order or based on the order of appearance.
#-- Label Encoding is suitable for ordinal data, where there is an inherent order among the categories.
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['Admission_Type']=label.fit_transform(df['Admission_Type'])
df['Diagnosis']=label.fit_transform(df['Diagnosis'])
df['A1C_Result']=label.fit_transform(df['A1C_Result'])

#Mapping
df['Gender']=df['Gender'].map({'Male':1,'Female':0,'Other':2}).astype('int')
df['Readmitted']=df['Readmitted'].map({'Yes':1,'No':0}).astype('int')


# In[18]:


df.head(2)


# In[19]:


df_1=df[['Age','Gender','Admission_Type','Diagnosis','Num_Lab_Procedures','Num_Medications','A1C_Result','Readmitted']]


# In[20]:


## Feature Scaling -- It helps in finding the distance b/w the data.
#If not , the feature with higher value range starts dominating while calculating the distance
col=['Age','Num_Lab_Procedures','Num_Medications']
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
df_1[col]=scaled_data=st.fit_transform(df_1[col])


# In[21]:


df_1.head(2)


# In[22]:


cleaned_data=df_1.drop('Readmitted',axis=1)
output=df_1['Readmitted']

x_train,x_test,y_train,y_test=train_test_split(cleaned_data,output,test_size=0.3,random_state=34)

## Model Building
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score

Models = {'Logistic Regression': (LogisticRegression(solver='lbfgs'), {'C': [0.1,0.008,0.0045], 'penalty': ['l2']}),
         'Decision Tree': (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [4, 5, 2]}),
         'SVM':(svm.SVC(),{'C': [0.0012,0.005,0.09], 'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.2,0.5,0.008]}),
         'Random Forest':(RandomForestClassifier(),{'n_estimators': [10,500,400,20,70], 'max_depth': [3,2,5]}),
         'k-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [1,2], 'weights': ['distance']}),
         'Gradient Boosting':(GradientBoostingClassifier(),{'n_estimators': [500,200],'learning_rate': [0.7,0.04], 'max_depth': [1,2,4]}),
         'XG Boost':(XGBClassifier(),{'n_estimators': [500,300],'learning_rate': [0.78,0.45],'gamma': [0.09,0.8],'reg_alpha': [0.78,0.23],'reg_lambda': [0.56,0.43]}),
         'Naive Bayes':(GaussianNB(),{})}


for model_name,(model,param_grid) in Models.items():
    grid=GridSearchCV(model,param_grid,cv=5)
    grid.fit(x_train,y_train)
    print(f"Best Parameters for {model_name}:{grid.best_params_}")
    print(f"Best Accuracy for {model_name}:{grid.best_score_}\n")


# In[23]:


## Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.activations import relu,sigmoid,tanh
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2


model_relu=Sequential([Dense(64,activation=relu,input_shape=(7,),kernel_initializer=GlorotUniform(seed=42),
                            kernel_regularizer=l2(0.01)),
                  Dropout(0.25),
                  Dense(100,activation=relu,kernel_initializer=GlorotUniform(seed=42)),
                  Dense(150,activation=relu,kernel_initializer=GlorotUniform(seed=42)),
                  Dense(200,activation=relu,kernel_initializer=GlorotUniform(seed=42)),
                  Dense(200,activation=relu,kernel_initializer=GlorotUniform(seed=42)),
                  Dense(250,activation=relu,kernel_initializer=GlorotUniform(seed=42)),
                  Dense(200,activation=relu,kernel_initializer=GlorotUniform(seed=42)),
                  Dense(1,activation=sigmoid,kernel_initializer=GlorotUniform(seed=42))
                 ])
model_relu.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model_relu.fit(x_train,y_train,epochs=50,batch_size=32,validation_data=(x_test,y_test),validation_batch_size=32)


# In[24]:


ran=RandomForestClassifier(max_depth= 5, n_estimators= 500)
ran.fit(x_train,y_train)
y_pred=ran.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[25]:


print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(f"F1 Score: {f1}")
print(classification_report(y_test, y_pred))


# In[26]:


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[27]:


# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = {:.2f})'.format(pr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.show()


# In[28]:


import pickle

# now you can save it to a file
file = 'C:\\Users\\DELL\\Downloads\\Model for Hospital Readmissions\\ran_Model.pkl'
with open(file, 'wb') as f:
    pickle.dump(ran,f)
    
with open(file, 'rb') as f:
    k = pickle.load(f)


# In[29]:


df_1.columns


# In[30]:


import streamlit as st
from PIL import Image
import numpy as np
import pickle
import base64

# Load the model
with open('C:\\Users\\DELL\\Downloads\\Model for Hospital Readmissions\\ran_Model.pkl', 'rb') as file:
    model = pickle.load(file)

def run():
    img = Image.open('C:\\Users\\DELL\\Downloads\\Hospital_logo.jpg')
    img = img.resize((150, 150))
    st.image(img, use_column_width=False)
    
    def add_bg_from_local(image_file):
        
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
    add_bg_from_local('C:/Users/DELL/Downloads/Hospitial_Img.jpg') 
    
    st.markdown("<h1 style='color: black;'>Hospital Re-Admission Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("<label style='color: black; font-style: italic;font-weight: bold;'>Age</label>", unsafe_allow_html=True)
    Age = st.number_input(' ', key='age_input', value=0)
    
    st.markdown("<label style='color: black; font-style: italic;font-weight: bold;'>Gender</label>", unsafe_allow_html=True)
    gen_display = ('Male', 'Female', 'Other')
    gen_options = list(range(len(gen_display)))
    gen = st.selectbox(" ", gen_options, key='gender_input', format_func=lambda x: gen_display[x])
    
    st.markdown("<label style='color: black; font-style: italic;font-weight: bold;'>Admission Type</label>", unsafe_allow_html=True)
    ad_display = ('Emergency', 'Urgent', 'Elective')
    ad_option = list(range(len(ad_display)))
    ad = st.selectbox(" ", ad_option, key='admission_type_input', format_func=lambda x: ad_display[x])
    
    st.markdown("<label style='color: black; font-style: italic;font-weight: bold;'>Diagnosis Type</label>", unsafe_allow_html=True)
    dia_display = ('Heart Disease', 'Diabetes', 'Injury', 'Infection')
    dia_option = list(range(len(dia_display)))
    dia = st.selectbox(" ", dia_option, key='diagnosis_type_input', format_func=lambda x: dia_display[x])
    
    st.markdown("<label style='color: black; font-style: italic;font-weight: bold;'>Number of Lab Procedures made</label>", unsafe_allow_html=True)
    Num_Lab_Procedures = st.number_input(' ', key='lab_procedures_input', value=0)
    
    st.markdown("<label style='color: black; font-style: italic;font-weight: bold;'>Number of Medications took</label>", unsafe_allow_html=True)
    Num_Medications = st.number_input(' ', key='medications_input', value=0)
    
    st.markdown("<label style='color: black; font-style: italic;font-weight: bold;'>A1C Result</label>", unsafe_allow_html=True)
    aic_display = ('Normal', 'Abnormal')
    aic_option = list(range(len(aic_display)))
    aic = st.selectbox(" ", aic_option, key='aic_input', format_func=lambda x: aic_display[x])

    if st.button('Submit'):
        import time
        with st.spinner("wait for it"):
            time.sleep(3)
            
        # Prepare the feature vector for prediction
        features = [[Age, gen, ad, dia, Num_Lab_Procedures, Num_Medications, aic]]
        
        # Convert categorical variables to numerical
        features[0][1] = gen_options.index(gen)
        features[0][2] = ad_option.index(ad)
        features[0][3] = dia_option.index(dia)
        features[0][6] = aic_option.index(aic)
        
        # Convert the list to a NumPy array for prediction
        features_array = np.array(features)

        # Make prediction
        prediction = model.predict(features_array)
        
        if prediction == 0:
            st.markdown("<p style='color: black; font-style: italic;font-weight: bold;'>Need to be Admitted</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: black; font-style: italic;font-weight: bold;'>Need not to be Admitted. Please take proper medications which have been prescribed</p>", unsafe_allow_html=True)

run()

