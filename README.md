# Human Activity Recognition
Galvanize capstone project for classifying human activity of daily living using UCI Machine Learning Repository smartphones data set.  

## Motivation & Goal:
Innovative approaches to recognize activities of daily living (ADL) is essential input part for development of more interactive human-computer applications. Methods for understanding Human Activity Recognition (HAR) are developed by interpreting attributes derived from motion, location, physiological signals and environmental information. Project explores best machine learning methods for performing ADL classification of published data set (http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphone). Data contains recordings from a group of individuals performing set of ADL (standing, sitting, laying, walking, walking upstairs and walking downstairs) while wearing a waist-mounted smart-phone with embedded internal sensors (accelerometers, gyroscopes and magnetometers). Effectiveness of machine learning methods are compared with published Multi Class Hardware-Friendly Support Vector Machine (MC-HF-SVM) recognition accuracy.

## Method:
Data (10299 samples with ADL balanced 561 features) is partitioned into training and test set in proportion 70% and 30%. The partition is randomized. The training data is employed for training different classifiers such as Support Vector Machine (SVM) , Random Forrest (RF) and eXtreme Gradient Boosting (XGBoost). Cross validation with grid search and Yelp Metric Optimization Engine (MOE) are employed for finding optimal classifier parameters. Overall accuracy, recall and precision is measured for determining best classifier. 

## Data:
1. “Human Activity Recognition Using Smartphones Data Set “ - UCI Machine Learning Repository 
http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

## Tools:
Python (Pandas, Numpy, Scikit-Learn).

## References:
1. “Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support
Vector Machine” - Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L. Reyes-Ortiz 
http://www.icephd.org/sites/default/files/IWAAL2012.pdf

2. “Energy Efficient Smartphone-Based Activity Recognition using Fixed-Point Arithmetic” - Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz 
http://www.jucs.org/jucs_19_9/energy_efficient_smartphone_based/jucs_19_09_1295_1314_anguita.pdf

3. "Human Activity and Motion Disorder Recognition: Towards Smarter Interactive Cognitive Environments" - Jorge L. Reyes-Ortiz, Alessandro Ghio, Davide Anguita,Xavier Parra, Joan Cabestany, Andreu Catala
https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2013-11.pdf

4. “Introducing MOE: Metric Optimization Engine; a new open source, machine learning service for optimal experiment design” - posted by DR. Scott Clark 
http://engineeringblog.yelp.com/2014/07/introducing-moe-metric-optimization-engine-a-new-open-source-machine-learning-service-for-optimal-ex.html

5. “dmlc XGBoost eXtreme Gradient Boosting - An optimized general purpose gradient boosting library."
https://github.com/dmlc/xgboost





