# DATA SCIENCE PROJECT ON BREAST CANCER PREDICTION
## INTRODUCTION
Breast cancer has become the most recurrent type of health issue among women especially for women in middle age. Early detection of breast cancer can help women cure this disease and death rate can be reduced. In the present-day scenario, to observe breast cancer mammograms are used and they are known be the most effective scanning technique. In this project the prediction of cancer cells is done by machine learning technique.

## GOAL OF THE PROJECT
In this Project we will be focusing on comparing the performance of machine learning techniques on predicting whether the cancer is Benign or Malignant according to a provided dataset involving breast cancer diagnosis
Therefore we will apply machine learning techniques on the provided dataset to compare their coefficients and performance measures regarding predicting the target feature in the breast cancer dataset.
## METHODOLOGY 
This section presents the theoretical basis of the MLP, KNN, AB, Bagging, GB, and RF models and the development of the work to predict and diagnose breast cancer. A. Multi-layer Perceptron The MLP is an ANN type. It uses backpropagation to train the network. The MLP is composed of multiple layers, each of which is connected to all the others, forming a directed network. The MLP learns a feature from a set of inputs and combines the various features into a set of outputs. The layers usually have weights and polarization units that are adjusted during training. It should be noted that, with the exception of the input nodes, each node in the network is a neuron using a nonlinear activation function.
Artificial neural networks (ANN) or neural network systems are computing systems that mimic the functioning of a human brain. The main aim of the algorithm is to provide a faster result with more accuracy than an old or traditional system. if the algorithm has been given the data or an image  about a particular object then the algorithm will quickly be able to identify or categorise images that do not contain the said object. 
### DEVICE PROJECT IN-TO MULTIPLE STEPS:
1.	Data Collection 
2.	Loading data
3.	Domain Analysis 
4.	4.Basic Checks of data
5.	EDA (Univariate Data Analysis)
6.	Data Pre-processing 
7.	Feature Selection 
8.	Building ML Model
9.	Training & Model Evaluation
10.	Model Saving

### LOADING DATA:
load data in python using panda’s library

### DOMAIN ANALYSIS:
•	Understanding the meaning of each feature and grasping the significance of every attribute in defining sales effectiveness is essential. Status is the target Variable Represents the current status of the transaction or order.

### EDA (UNIVARIATE, BIVARIATE, MULTIVARIATE DATA ANALYSIS)
![image](https://github.com/KhairnarRutuja/Breast_Cancer_Prediction_Using_Artificial_Neural_Network_ANN/assets/135214279/c1f8fc07-34e5-4a81-be9a-1cd9da93c21d)
####OBSERVATION
#### Diagnosis (Malignant OR Benign):
* In this data the majority of 62.74% of patients have Malignant (M) type of cancer and the 37.26 of patients have Benign (B) type of cancer.

**Bivariate Analysis we used boxen plot, swarm plot and violin plot to find the Relationship between independant Feature with respect to Target Feature.**

**Multivariate Data Analysis**
![image](https://github.com/KhairnarRutuja/Breast_Cancer_Prediction_Using_Artificial_Neural_Network_ANN/assets/135214279/70c0e1c6-1c0f-415b-92a7-883d078989a9)
**OBSERVATION**
In this graph, we observe that when the mean area is at its minimum and the mean smoothness ranges from minimum to maximum, benign (B) type cancer is prevalent. Conversely, when the mean area increases and smoothness is also at its maximum, malignant (M) type cancer is predominant.
![image](https://github.com/KhairnarRutuja/Breast_Cancer_Prediction_Using_Artificial_Neural_Network_ANN/assets/135214279/9f5f75ac-c4cd-4c70-8ffd-af168df157b9)
**OBSERVATION**
In this graph, it is evident that when the radius standard error RADIUS SE is at its minimum and simultaneously the mean area is also at its minimum, benign (B) type cancer tends to be prevalent. Conversely, as the radius standard error increases and the mean area reaches its maximum, malignant (M) type cancer becomes predominant.

### DATA PREPROCESSING
**1. Missing Values:**
* The first step in data preprocessing is to check for missing values. Fortunately, in this dataset, no features have missing values.

**2. Encoding Categorical Variables:**
* Since there is only one categorical feature in the dataset, we can manually encode it into numerical values. This allows us to incorporate it into our machine learning models effectively.

**3. Handling Outliers:**
* Next, we address outliers present in some features. We use the Interquartile Range (IQR) method to identify and handle outliers. This helps ensure that the outliers do not unduly influence our models.

**4. Scaling:**
* After handling outliers, we apply feature scaling to ensure that all features have the same scale. Specifically, we use standard scaling to standardize the range of our features and bring them to a mean of 0 and a standard deviation of 1.

### FEATURE ENGINEERING
**1. Dropping Unique and Constant Features:**
* The first step in feature engineering is to identify and drop unique and constant features. In this dataset, we identified two unique features, namely 'id' and 'Unnamed: 32', which do not contribute to the predictive power of our model. Therefore, we dropped these features.

**2. Checking Correlation:**
* Before checking correlation, we ensured that all data types were appropriate for analysis. Once data types were confirmed, we checked the correlation between features using a heatmap. From the heatmap analysis, it was observed that there are no highly correlated features present. Therefore, we did not drop any feature based on correlation.

**3. Addressing Duplicates:**
•	I haven't addressed duplicates as I've compressed and merged labels, ensuring the dataset doesn't contain any duplicates due to label transformations

### MODEL CREATION & EVALUATION
**1. Define Independent and Dependent Variables:**
* The first step in model creation is to define the independent (features) and dependent (target) variables. The features are independent variables, while the target variable is dependent on these features.

**2. Splitting the Data:**
* Once the variables are defined, the data is split into training and testing sets. We typically use an 80-20 split, with 80% of the data used for training and 20% for testing.

**3. Checking for Data Balance:**
* Before training the model, it's essential to check if the data is balanced or not. Imbalanced data can lead to biased models. In this case, it was observed that the data is imbalanced. To address this issue, the Synthetic Minority Over-sampling Technique (SMOTE) was utilized to balance the data by oversampling the minority class.

**4. Model Selection:**
* For this project, the chosen algorithm is the Artificial Neural Network (ANN). ANNs are known for their ability to handle complex relationships in data and are suitable for tasks such as breast cancer prediction.

**5. Model Training and Evaluation:**
The ANN model is trained on the balanced training data. Once trained, the model is evaluated using various performance metrics such as accuracy, recall, precision, and F1 score. These metrics provide insights into the model's overall performance and its ability to correctly classify instances of both classes (malignant and benign).

### MODEL SAVING 
•	Save the model using pickle file.

### CONCLUSION
* The conducted work has determined that Artificial Neural Networks (ANNs) are highly suitable for predicting breast cancer in datasets. Specifically, it has been concluded that ANNs are the most efficient machine learning algorithm for this task. This conclusion is supported by several performance measures, including high accuracy levels achieved by the ANN algorithm. Moreover, the ANN demonstrated fast prediction speeds and short training times, further emphasizing its superiority in predictive capability and efficiency. Despite potential limitations, it is evident that Artificial Neural Networks excel in breast cancer prediction tasks, highlighting their effectiveness in healthcare applications.
* A training score of 100% and a testing score of 99% indicate very high performance of the model. This suggests that the model has learned the training data extremely well, achieving near-perfect accuracy. Additionally, the high testing score indicates that the model generalizes well to unseen data, performing almost as well on new data as it did on the training data. Overall, these scores suggest that the model is highly effective and reliable for the task it was trained on.

​


  

