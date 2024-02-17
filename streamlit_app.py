#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    
    st.title("""E-banking Usage level and Influence on Spending 
             Habits Among College of Business and Management Students""")
    
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
 
    st.write("""Both Gaussian and Bernoulli Naive Bayes are variants \
        of the Naive Bayes algorithm, a popular machine learning \
        technique for classification. However, they differ in \
        how they handle data and are suited for different types of datasets.""")

    st.write('Key Differences:')
    st.write('Data Type:')
    text = """ Gaussian Naive Bayes: Assumes features are continuous 
    and follow a normal distribution (bell-shaped curve). Examples: 
    height, weight, temperature."""
    st.write(text)

    text = """Bernoulli Naive Bayes: Assumes features are binary 
    (yes/no, true/false, present/absent). Examples: email spam 
    filter (spam/not spam), image pixel (black/white). Conditional 
    Probability Distribution:"""
    st.write(text)

    text = """This dataset potentially offers valuable insights 
    into real-world e-banking usage patterns among CBM students, 
    considering the influences of gender, course year level, 
    and family income."""

    st.write(text)

    # Create the logistic regression 
    clf = GaussianNB() 
    options = ['Gaussian Naive Bayes', 'Bernoulli Naive Bayes']
    selected_option = st.selectbox('Select the classifier', options)
    if selected_option=='Bernoulli Naive Bayes':
        clf = BernoulliNB()
    else:
        clf = GaussianNB()
    
    # Create the logistic regression 
    dbfile = 'encoded-ebanking.csv'
        
    if st.button('Start'):
        
        df = pd.read_csv(dbfile, header=0)
        st.subheader('The Dataset')
        # display the dataset
        st.dataframe(df, use_container_width=True)  

        #load the data and the labels
        X = df.values[:,0:-1]
        y = df.values[:,-1]          
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=0.2, random_state=42)
        
        clf.fit(X_train,y_train)
        y_test_pred = clf.predict(X_test)
        st.subheader('Confusion Matrix')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))
    
#run the app
if __name__ == "__main__":
    app()
