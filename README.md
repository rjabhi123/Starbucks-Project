# Udacity-Starbucks-Capstone-Project
 This repository has all the code and report for my Udacity Starbucks Capstone Project Capstone project.

# Installations
This project was written in Python, using Jupyter Notebook on Anaconda. The relevant Python packages for this project are as follows:

   * pandas
   * numpy
   * math
   * json
   * matplotlib
   * seaborn
   * from sklearn.model_selection import train_test_split, GridSearchCV
   * from sklearn.linear_model import LogisticRegression
   * from sklearn.neighbors import KNeighborsClassifier
   * from sklearn.tree import DecisionTreeClassifier
   * from sklearn.svm import SVC
   * from sklearn.ensemble import RandomForestRegressor
   * from sklearn.naive_bayes import GaussianNB 
   
  
# Project Motivation
This project is the Capstone project of my Machine Learning Engineer nanodegree with Udacity. As students in the nanodegree, we have the option to take part in the Starbucks Capstone Challenge. For the challenge, Udacity provided simulated data that mimics customer behavior on the Starbucks rewards mobile app.The objective is to try to find how Starbucks customers use the app, and how well is the current offers system. more importantly, to find patterns and show when and where to give specific offer to a specific customer.
  
# File Descriptions

1. The data used in the project is in the files : 
     * portfolio.json :-  containing offer ids and meta data about each offer (duration, type, etc.).
     * profile.json :- demographic data for each customer.
     * and transcript.json :- records for transactions, offers received, offers viewed, and offers completed.
     
2. Starbucks_Capstone_notebook.ipynb :- contains all the work.


# Implementation

This dataset contains simulated data that mimics customer behavior on the Starbucks rewarding system in their mobile application. Once every few days, Starbucks sends out an offer to users of the mobile app. The message can be an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. We are going to analyze three file:
  1. portfolio
  2. profile
  3. transcript
  
The process of our analysis will be by the following step: Define our Business question, understanding the Datasets, Data preparation and wrangling, analyze the data, model the data, compare model performance, and finally selecting one model and improving it.
   * Business Understanding :-  The objective here is to find patterns and show when and where to give specific offer to a specific                                     customer.
   * Data Understanding :- Let’s see table by table so we can have an overview.
   * Data preparation and wrangling :- Based on what we have seen in the previous step, there needs to be some work to prepare the data                                        for analysis and modeling.
   * Analyzing the Data : - For this part, it will be divided into Univariate Exploration and Multivariate Exploration.
   
        1.  Univariate Exploration :-  First, let’s start with the Univariate Exploration and try to answer the following questions :
   
               i)   What is the average income for Starbucks customers?
          
               ii)  What is the average age for Starbucks customers?
          
               iii) What is the most common promotion?
          
               iv)  What are the most common age group and gender?
          
               v)   Who are the most loyal customer (most transcripts)?
            
        2.  Multivariate Exploration :-  For our Multivariate Exploration, we will try to answer the following questions :   
        
               i)   What is the most common promotion for children, teens, young adult, adult and elderly customers?
             
               ii)  From profiles, which get more income, males or females?
             
               iii) Which type of promotions each gender likes?
          
   * Modeling the Data :- I tried to make a model that can identify which kind of offers we should give a customer. Because my model                               will guess the offer_type, I will only get those transcripts with offer id’s. So I will ignore all                                       transactions without offer id’s.
                          Since we have a simple classification problem, I will use accuracy to evaluate my models. We want to see how                             well our model by seeing the number of correct predictions vs total number of predictions. Why choose                                   accuracy? First let’s define accuracy, the ratio of the correctly labeled subjects to the whole pool of                                 subjects. Also, accuracy answers questions like: How many students did we correctly label out of all the                                 students? It’s similar to our situation right? because we want to see how many customers use Starbucks offers.                           Furthermore, Accuracy = (TP+TN)/(TP+FP+FN+TN). Not to forget, that this is a simple classification problem, so                           this is my opinion and reasoning on why to use the easiest (accuracy).
                          
      The models that I have used are :-  Logistic Regression, K-Nearest Neighbors, Decision Tree, Support Vector Machine, Random             Forest,and Naive Bayes.
                          
   * Compare model performance :- Now that we have trained the data, it’s time to evaluate their performance based on accuracy.
   * Model Improvements :- After using Grid Search with Logistic Regression we managed to get better results.
   * Conclusion :- In this project, I tried to analyze and make model to predict the best offer to give a Starbucks customer. First I                      explored the data and see what I have to change before start the analysis. Then I did some exploratory analysis on                      the data after cleaning. After that I trained the data, then choose one model and improved it to get better results.                    In conclusion, I think that Starbucks needs to focus more on adults and Males. Also, offer more BOGO and discounts to                    their customers.
   * Improvements :- I got to a point where I had good results and a good understanding of the data. But to make our results even                            better, I tried to improve my data collection and fixed issues which i had with NaN values. I took even more data                        like location and branch along with time when the transaction were completed. All these data can help us know when                      and where to give our offers.
   
   ### Metrics : -
   Since we have a simple classification problem, I will use accuracy to evaluate my models. We want to see how well our                    model by seeing the number of correct predictions vs total number of predicitons.
   Why choose accuracy? First let's define accuracy, the ratio of the correctly labeled subjects to the whole pool of subjects. Also,      accuracy answers questions like: How many students did we correctly label out of all the students? It's similar to our situation        right? because we want to see how many customers use Starbucks offers. Furthermore, Accuracy = (TP+TN)/(TP+FP+FN+TN). Not to forget,    that this is a simple classification problem, so this is my opinion and reasoning on why to use the easiest (accuracy).
   
   Reference :-  
           1.)  First [here]( https://towardsdatascience.com/accuracy-recall-precision-f-score-specificity-which-to-optimize-on-867d3f11124)
           2.)  second [here]( https://medium.com/thalus-ai/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b)
           
  * Cost function :- 
  
     1. Absolute  Error :- 
      
         Absolute Error is the amount of error in your measurements. It is the difference between the measured value and “true” value.            For example, if a scale states 90 pounds but you know your true weight is 89 pounds, then the scale has an absolute error of 90          lbs – 89 lbs = 1 lbs.

         This can be caused by your scale not measuring the exact amount you are trying to measure. For example, your scale may be                accurate to the nearest pound. If you weigh 89.6 lbs, the scale may “round up” and give you 90 lbs. In this case the absolute            error is 90 lbs – 89.6 lbs = .4 lbs.
       
         Reference :- [here](https://www.statisticshowto.com/absolute-error/)
     
     2.  Mean absolute percentage error :-
     
         The mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), is a measure of prediction          accuracy of a forecasting method in statistics, for example in trend estimation, also used as a loss function for regression            problems in machine learning.
        
         Reference :- [here](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
                           
# Results
The main results of the code can be found at the post available [here](https://medium.com/@rabhimanyu509/starbucks-project-9ca045b6a66c).
  
# Acknowledgements
I want to thank the Udacity Machine Learning Engineer Nano-Degree Program for giving me this opportunity to do the project.

