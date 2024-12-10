#Import Liberaries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from PIL import Image

# Create a title 
st.markdown(
    """
           <h1 style="color: #ff4b4b; font-size: 50px;font-weight: bold;">
            Diabetes Prediction Webapp
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Display the image
image = Image.open("diabetes_2.jpg")
st.image(image, use_container_width=True)

# Display the image caption
st.markdown(
    """
    <div style="text-align: center; color: #ff4b4b; font-size: 20px; font-weight: bold;">
        The only way to finish it is to start
    </div>
    """,
    unsafe_allow_html=True
)


# Import Data 
df = pd.read_csv('diabetes.csv')

#set a subbheader
st.markdown(
    """
    <h2 style="color: #ff4b4b; font-size: 24px;font-weight: bold;">
        Data Samples
    </h2>
    """,
    unsafe_allow_html=True
)

sampled_data =df.sample(n=10,random_state=123)
st.write(sampled_data)

# Exploratory Data Analysis

columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)
df.isnull().sum()
def median_target(var):   
    temp = df[df[var].notnull()]
    
    median_values = temp.groupby('Outcome')[[var]].median().reset_index()
    
    return median_values
columns = df.columns.drop("Outcome")

for i in columns:
    median_vals = median_target(i)

    df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = median_vals.loc[median_vals['Outcome'] == 0, i].values[0]
    
    df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = median_vals.loc[median_vals['Outcome'] == 1, i].values[0]
df.isnull().sum().sum()

# BloodPressure column cleaninng
q1=df.BloodPressure.quantile(0.25)
q3=df.BloodPressure.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
upper=q3+1.5*IQR
df.loc[df.BloodPressure>upper, "BloodPressure"]=upper
df.loc[df.BloodPressure < Lower, "BloodPressure"] = Lower

# SkinThickness column cleaninng
q1=df.SkinThickness.quantile(0.25)
q3=df.SkinThickness.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
upper=q3+1.5*IQR
df.loc[df.SkinThickness>upper, "SkinThickness"]=upper
df.loc[df.SkinThickness < Lower, "SkinThickness"] = Lower

# Insulin column cleaninng
q1=df.Insulin.quantile(0.25)
q3=df.Insulin.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
upper=q3+1.5*IQR
df.loc[df.Insulin>upper, "Insulin"]=upper

# BMI column cleaninng
q1=df.BMI.quantile(0.25)
q3=df.BMI.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
upper=q3+1.5*IQR
df.loc[df.BMI>upper, "BMI"]=upper

# DiabetesPedigreeFunction column cleaninng
q1=df.DiabetesPedigreeFunction.quantile(0.25)
q3=df.DiabetesPedigreeFunction.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
upper=q3+1.5*IQR
df.loc[df.DiabetesPedigreeFunction>upper, "DiabetesPedigreeFunction"]=upper



# Import Models
from sklearn.model_selection import train_test_split # to split data to train and test

from imblearn.over_sampling import RandomOverSampler # to handle imbalanced datasets

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score , recall_score , f1_score
from sklearn.metrics import classification_report

from collections import Counter # to count the occurrences of each element
# Data Preperation
x = df.drop('Outcome',axis=1) # Features
y = df.Outcome # target

over = RandomOverSampler(random_state=41)
x_over,y_over = over.fit_resample(x,y)

print("old shape {}".format(Counter(y)))
print("new shape {}".format(Counter(y_over)))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=41)

# get the feature imput from the users
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3) # first = min / second = max / third = the cursor in page web
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

   # store a dictionarry into a variable
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    #transform data in to data dataframe
    features = pd.DataFrame(user_data,index=[0])
    return features

#store the user input into a data frame
user_input = get_user_input()

#set a subbheader and display it 
st.markdown(
    """
    <h2 style="color: #ff4b4b; font-size: 24px;font-weight: bold;">
       User Input
    </h2>
    """,
    unsafe_allow_html=True
)
st.write(user_input)

model_1=LogisticRegression()
model_2=SVC()
model_3=RandomForestClassifier(n_estimators = 100,class_weight='balanced')
model_4=GradientBoostingClassifier(n_estimators=1000)

Columns = ['LogisticRegression','SVC','RandomForestClassifier','GradientBoostingClassifier']
acc_list = []
recall_list = []
f1_list = []

def cal(model):    
    # Learn Model
    model.fit(x_train,y_train)
    # Metrics
    prediction = model.predict(x_test)
    accurcy = accuracy_score(prediction,y_test)
    recall = recall_score(prediction,y_test)
    f1 = f1_score(prediction,y_test)
    cm = confusion_matrix(prediction,y_test)
    # Append metrics 
    acc_list.append(accurcy)
    recall_list.append(recall)
    f1_list.append(f1)
    # Display Results
    print("Used Model : ", model,'\n')
    print("Accurcy Score = " , accurcy,'\n')
    print("Recall Score = ",recall,'\n')
    print("f1 Score = ",f1,'\n')
    print('Confusion matrix\n--------------------------\n', cm)
    print('--------------------------\n')   

cal(model_1)
cal(model_2)
cal(model_3)
cal(model_4)


st.markdown(
    """
    <h2 style="color: #ff4b4b; font-size: 24px;font-weight: bold;">
       Performance Comparison of Different Models
    </h2>
    """,
    unsafe_allow_html=True
)
df2 = pd.DataFrame({'Algorithm':Columns,"Accurecy_Score":acc_list,"Recall_Score":recall_list,"F1_Score":f1_list})
st.write(df2)

# Create the plot
fig, ax = plt.subplots(figsize=(40, 25))
ax.plot(df2['Algorithm'], df2['Accurecy_Score'], label="Accuracy", marker='o')
ax.plot(df2['Algorithm'], df2['Recall_Score'], label="Recall", marker='o')
ax.plot(df2['Algorithm'], df2['F1_Score'], label="F1 Score", marker='o')

# Add labels and legend
ax.set_title("Performance Comparison of Different Models", fontsize=30)
ax.set_xlabel("Algorithm", fontsize=25)
ax.set_ylabel("Score", fontsize=25)
ax.legend(loc="upper left")

# Display the plot in Streamlit
st.pyplot(fig)

st.markdown(
    """
    <h2 style="color: #ff4b4b; font-size: 24px;font-weight: bold;">
       Accurcy Score
    </h2>
    """,
    unsafe_allow_html=True
)

st.write(str(round(accuracy_score(y_test, model_3.predict(x_test)) * 100, 2)) + '%')

# Predict with the RandomForest model using the user input
prediction = model_3.predict(user_input)

# Display classification result
st.markdown(
    """
    <h2 style="color: #ff4b4b; font-size: 24px; font-weight: bold;">
       Prediction
    </h2>
    """,
    unsafe_allow_html=True
)
st.write( "Diabetic" if prediction[0] == 1 else "Non-Diabetic")

















