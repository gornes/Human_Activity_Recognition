# Human Activity Recognition
Galvanize capstone project for classifying human activity of daily living using uci machine learning smartphones data set.  

## Motivation & Goal:
Innovative approaches to recognize activities of daily living (ADL) is essential input part for development of more interactive human-computer applications. Methods for understanding Human Activity Recognition (HAR) are developed by interpreting attributes derived from motion, location, physiological signals and environmental information. Project goal is to propose a machine learning method to perform ADL classification of published data set (http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphone). Data contains recordings from a group of individuals performing set of ADL (standing, sitting, laying, walking, walking upstairs and walking downstairs) while wearing a waist-mounted smart-phone with embedded internal sensors (accelerometers, gyroscopes and magnetometers). Effectiveness of machine learning method will be performed by comparing recognition accuracy of machine learning method with published Multi Class Hardware-Friendly Support Vector Machine (MC-HF-SVM) recognition accuracy.

## Method:
Data (10299 samples with ADL balanced 561 features)  will be partitioned into training and test set in proportion 70% and 30%. The partition will be randomized. The training data will be employed for training different classifiers such as Support Vector Machine (SVM) , Random Forrest (RF) and Boosting (AdaBoost). Cross validation will be employed to test classifiers performance and Yelp Metric Optimization Engine (MOE) for finding optimal parameters. Overall accuracy, recall and precision will be measured for determining best classifier. 

## Data:
1. “Human Activity Recognition Using Smartphones Data Set “ - UCI Machine Learning Depository; 
http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

## Tools:
Python (Pandas, Numpy, Scikit-Learn)

## References:
1. “Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support
Vector Machine” - Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L. Reyes-Ortiz 
http://www.icephd.org/sites/default/files/IWAAL2012.pdf

2. “Energy Efficient Smartphone-Based Activity Recognition using Fixed-Point Arithmetic” - Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz 
http://www.jucs.org/jucs_19_9/energy_efficient_smartphone_based/jucs_19_09_1295_1314_anguita.pdf

3. “Introducing MOE: Metric Optimization Engine; a new open source, machine learning service for optimal experiment design” - posted by DR. Scott Clark 
http://engineeringblog.yelp.com/2014/07/introducing-moe-metric-optimization-engine-a-new-open-source-machine-learning-service-for-optimal-ex.html

4. “dmlc XGBoost eXtrreme Gradient Boosting - An optimized general purpose gradient boosting library."
https://github.com/dmlc/xgboost





