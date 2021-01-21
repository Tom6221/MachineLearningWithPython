# ==================================================================
# The best classifier
# Final project for the course "Machine Learning with python"
# (C) Thomas Bitterlich, T-Systems International
# 21.01.21 final version
# ==================================================================

# =================
# === imports
# =================
import pandas as pd
import numpy as np
import pydotplus
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import itertools

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss

import scipy.optimize as opt

from io import StringIO

# ===============================
# === some relevant parameters
# ===============================
# all the columns that will be used for the models
relevantColumns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

# the rate of data which is used for testing (must be larger than 0 and less than 1)
TestSize = 0.33

# Which of the models shall be executed ?
doKnn = True    # Execute k-nearest neighbor
doDt  = True    # Execute Decision Tree
doSvm = True    # Execute Support Vector Machines
doLr  = True    # Execute Logistic Regression


# ===============================
# === Functions
# ===============================

# ---------------------------------------
# --- write a data set to a temp csv file
# ---------------------------------------
def writeDataSetTo(dataSet, filename, maxZeile):
    tf = open(filename, "w")
    numZeile = 0
    for zeile in dataSet:
        if maxZeile == 0 or numZeile < maxZeile:
            numZeile = numZeile + 1
            # modify output, so that it is suitable for the german version of Excel
            z = str(zeile).replace("[", "").replace("]", "").replace(",", ";").replace(".", ",").replace("'", "")
            tf.write(str(numZeile) + "; " + z + "\n")
    tf.close()
    
    
# -----------------------------
# compute scores
# returns the scores as list
# -----------------------------
def calcAndPrintScores(y_test, yhat, yhat_prob):
    rawScore = []
    averageModes = ['micro', 'macro', 'samples', 'weighted']
    # 'binary': Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    #           'binary' excluded. Does not work in our case
    # 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance.
    # 'samples': Calculate metrics for each instance, and find their average (only meaningful for multilabel classification).
    
    # --- f1 score ---
    ausgabe = "F1 Score"
    for mode in averageModes:
        try:
            f1Score = f1_score(y_test, yhat, average=mode) 
            ausgabe = ausgabe + " mode " + mode + " = " + str(f1Score)
            rawScore.append(str(f1Score))
        except:
            ausgabe = ausgabe + " mode " + mode + "= ERR"
            rawScore.append('ERR')
    print (ausgabe)

    # --- jaccard score ---
    # Convert y_test and yhat to integers
    yt = pd.get_dummies(data=y_test, dtype=int)
    yh = pd.get_dummies(data=yhat, dtype=int)
    
    ausgabe = "Jaccard-Score"
    for mode in averageModes:
        try:
            jaccardScore = jaccard_score(yt, yh, average=mode)
            ausgabe = ausgabe + " mode " + mode + " = " + str(jaccardScore)
            #print ("Jaccard Score = " + str(jaccardScore))
            rawScore.append(str(jaccardScore))
        except:
            #print ("Jaccard Score " + mode + " reported an error")
            ausgabe = ausgabe + " mode " + mode + "= ERR"
            rawScore.append('ERR')
    print (ausgabe)
    
    # --- Log Loss score ---
    # Only useful for logistic regression, will be set to "ERR" in all other cases
    ausgabe = "Log Loss"
    try:
        logLoss = log_loss(yt, yh, yhat_prob)
        #print ("Log Loss Score = " + str(logLoss))
        rawScore.append(str(logLoss))
        usgabe = ausgabe + " = " + str(logLoss)
    except:
        #print ("Log Loss Score reported an error")
        rawScore.append('ERR')
        ausgabe = ausgabe + " = ERR"
        
    return rawScore



# -----------------------------
# k nearest neighbor prediction
# -----------------------------
def trainKnn(train, ytrain, test, ytest, TestSize):
    Kmax = 0
    Kgew = 0
    Ktrain = 0
    Ktest = 0
    AvgMax = 0
    AvgGewMax = 0
    TrainMax = 0
    TestMax = 0

    scoring = []
    
    print ("\n===================================")
    print ("===== k nearest neighbor model =====")
    print ("===================================")
    for k in range(1, 40):
        #print("======= k = " + str(k) + " =======")
        #Train Model and Predict  
        knc = KNeighborsClassifier(n_neighbors = k).fit(train,ytrain)

        # Predicting
        # we can use the model to predict the test set:
        xhat = knc.predict(train)
        yhat = knc.predict(test)
        
        #print ("xhat=" + str(xhat))
        #print ("yhat=" + str(yhat))
    
        # Accuracy evaluation
        # In multilabel classification, accuracy classification score is a function that computes subset accuracy. This function is equal to the jaccard_similarity_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
        TrainAcc = metrics.accuracy_score(ytrain, xhat)
        TestAcc = metrics.accuracy_score(ytest, yhat)
        AvgAcc = (TrainAcc + TestAcc) / 2.0
        AvgAccGew = (1.0 - TestSize) * TrainAcc + TestSize * TestAcc
        if AvgAcc > AvgMax:
            AvgMax = AvgAcc
            Kmax = k
        if AvgAccGew > AvgGewMax:
            AvgGewMax = AvgAccGew
            Kgew = k
        if TrainAcc > TrainMax:
            TrainMax = TrainAcc
            Ktrain = k
        if TestAcc > TestMax:
            TestMax = TestAcc
            Ktest = k
        print("K = " + str(k) + ", TrainAcc = " + str(TrainAcc) + ", TestAcc = " + str(TestAcc) + ", AvgAcc = " + str(AvgAcc) + ", AvgGew = " + str(AvgAccGew))

        # calculate and print required scores
        rawScore = calcAndPrintScores(y_test, yhat, None)
        
        scoring.append(["k-nearest neighbor", "k = " + str(k), rawScore])


    print("\nHöchste durchschnittliche Genauigkeit   " + str(AvgMax) + " bei K = " + str(Kmax))
    print("Höchste durchschn. gewichtete Genauigkeit " + str(AvgGewMax) + " bei K = " + str(Kgew))
    print("Höchste Trainingsdaten Genauigkeit        " + str(TrainMax) + " bei K = " + str(Ktrain))
    print("Höchste Testdaten Genauigkeit             " + str(TestMax) + " bei K = " + str(Ktest))    

    return scoring


# -----------------------------
# decision tree prediction
# -----------------------------
def trainDt(X_train, y_train, X_test, y_test, TestSize):
    print ("\n===============================")
    print ("===== decision tree model =====")
    print ("===============================")

    scoring = []

    for k in range(1,20):
        decisionTree = DecisionTreeClassifier(max_depth = k)

        # Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset 
        decisionTree.fit(X_train,y_train)
        
        # Prediction
        predTree = decisionTree.predict(X_test)
        #printDeviations(y_test, predTree)
        
        # calculate and print required scores
        rawScore = calcAndPrintScores(y_test, predTree, predTree)
        scoring.append(["decision tree", "k = " + str(k), rawScore])
        
    return scoring
    
        
# -----------------------------
# Support vector machine prediction
# -----------------------------
def trainSvm(X_train, y_train, X_test, y_test, TestSize):
    print ("\n=================================")
    print ("===== Support Vector Machine =====")
    print ("==================================")

    scoring = []

    # try different kernel types for the SVM
    KernelTypes = ['linear', 'poly', 'rbf', 'sigmoid']  # Specifies the kernel type to be used in the algorithm
    tolerance = 1e-3                                    # Tolerance for stopping criterion.
    degreeNumber = 5                                    # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    
    for kType in KernelTypes:
        print("======= Use Kernel type " + kType + " =======")
        svmTrain = svm.SVC(kernel=kType, degree=degreeNumber, tol=tolerance)
        svmTrain.fit(X_train, y_train) 

        # prediction
        yhat = svmTrain.predict(X_test)

        # Evaluation
        # Compute confusion matrix
        #svm_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
        #np.set_printoptions(precision = 2)
        #print (classification_report(y_test, yhat))
        
        # Plot non-normalized confusion matrix
        #plt.figure()
        #plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

        # calculate and print required scores
        rawScore = calcAndPrintScores(y_test, yhat, yhat)
        
        scoring.append(["support vector machines", "kernel type = " + str(kType), rawScore])
        
    return scoring
        
        
# -----------------------------
# Logistic regression prediction
# -----------------------------
def trainLr(X_train, y_train, X_test, y_test, TestSize):
    print ("\n==============================")
    print ("===== Logistic Regression =====")
    print ("===============================")

    scoring = []
    
    # train the model
    for solverToUse in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
        for i in range (1, 20):
            # value of C is counted up in steps of 0.1
            if i <= 10:
                Cvalue = round(0.1 * i, 1) # Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
            else:
                Cvalue = i
      
            print ("Linear regression for Solver " + solverToUse + " and C = " + str(Cvalue))
            LR = LogisticRegression(C=Cvalue, solver=solverToUse, max_iter=500).fit(X_train,y_train)
            
            # do prediction
            yhat = LR.predict(X_test)
            yhat_prob = LR.predict_proba(X_test)

            # calculate and print required scores
            rawScore = calcAndPrintScores(y_test, yhat, yhat_prob)
            
            scoring.append(["logistic regression", "solver = " + solverToUse + " and C="  + str(Cvalue), rawScore])
        
    return scoring

# ========================
# ======= M A I N ========
# ========================

# =========================
# ======= read data =======
# =========================
# I used the datset from https://github.com/shrikant-temburwar/Loan-Prediction-Dataset
# 
# The columns of the file are:
#
# Variable          Description
# -----------------+---------------------------------
#Loan_ID	        Unique Loan ID
#Gender	            Male/ Female
#Married	        Applicant married (Y/N)
#Dependents	        Number of dependents
#Education	        Applicant Education (Graduate/ Under Graduate)
#Self_Employed	    Self employed (Y/N)
#ApplicantIncome	Applicant income
#CoapplicantIncome	Coapplicant income
#LoanAmount	        Loan amount in thousands
#Loan_Amount_Term	Term of loan in months
#Credit_History	    credit history meets guidelines
#Property_Area	    Urban/ Semi Urban/ Rural
#Loan_Status	    Loan approved (Y/N)
dataSet = pd.read_csv('loanData.csv')
dataSet.head()

# =============================
# ======= Preprocessing =======
# =============================
print("--- Preprocessing training data")
# show column names and types
X = dataSet
X.info()

# drop columns which are not useful
X.drop('Loan_ID', axis=1)
# drop column which is for supervised training (will be used extra as yTrain)
X.drop('Loan_Status', axis=1)

# check for missing values in each column of X
X.apply(lambda x: sum(x.isnull()),axis=0) 

# modify empty entries to as useful value
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
    X[col] = X[col].fillna('unknown')
for col in ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    X[col] = X[col].fillna(0)

# see if data does not contain any empty cells now
for col in relevantColumns:
    print(X[col].value_counts())
    
# Numericalize strings
XTrain = pd.get_dummies(data=X, dtype=float)

# Show preprocessed data
print("Cleansed data")
print(XTrain)
writeDataSetTo(XTrain, "XTrain.csv", 0)


# Store the scores (F1, Jaccard and LogLoss as far as applicable) in an array to print out at the end
scores = []
# Header of the array
scores.append(['Algorithm', 'Parameters', 'F1Score-micro', 'F1Score-macro', 'F1Score-samples', 'F1Score-weighted', 'Jaccard Score-micro', 'Jaccard Score-macro', 'Jaccard Score-samples', 'Jaccard Score-weighted', 'Log Loss'])


# ==============================================================
# ======= Training / Test values for supervised learning =======
# ==============================================================
# === Do the Training and prediction
# These are our results for the supervised training
yTrain = dataSet['Loan_Status'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(XTrain, yTrain, test_size = TestSize, random_state = 1)


# =========================
# ======= Train knn =======
# =========================
# Train a k-Nearest neighbor model and do prediction for the test data
if doKnn:
    KNearestNeighbor = trainKnn(X_train, y_train, X_test, y_test, TestSize)
    scores.extend(KNearestNeighbor)

# ===================================
# ======= Train decision tree =======
# ===================================
# Train a decision tree model and do prediction for the test data
if doDt:
    decisionTree = trainDt(X_train, y_train, X_test, y_test, TestSize)
    scores.extend(decisionTree)

# ============================================
# ======= Train Support Vector Machine =======
# ============================================
# Train a Support Vector Machine model and do prediction for the test data
if doSvm:
    supportVectorMachine = trainSvm(X_train, y_train, X_test, y_test, TestSize)
    scores.extend(supportVectorMachine)

# =========================================
# ======= Train Logistic Regression =======
# =========================================
# Train a Logistic Regression model and do prediction for the test data
if doLr:
    logisticRegression = trainLr(X_train, y_train, X_test, y_test, TestSize)
    scores.extend(logisticRegression)

# =========================================
# Print out the result and store it in a
#  result.csv file (readble for Excel)
# =========================================
print (scores)
writeDataSetTo(scores, "results.csv", 0)
