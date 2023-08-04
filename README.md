# Bug-Prediction-Model
Bug classification and similar bug retrieval using machine learning
**Background:**
Software deveopement teams often encounter numerous bugs during the develpment and maintenance phases. Identifying the types of bugs and efficiently resolving them is crucial for the delivering high-quality software. Manual bug classification and searching for solutions from the existing tickets can be time consuming and error prone.

**Objective:**
The objectove was to develop an intelligent machine learning model that automates the bug classification process and make the retrieval of similar bugs faster and hassle free. The model inputs bug summaries and accurately classifies the bugs into predefined categories or clusters. Additionally, the system provides a feature to retrieve similar bugs based on the input summary, assisting developers in finding relevant solutions and accelerating the debugging process.

1) **Data Collection and Preprocessing:** Gathering a JIRA dataset with bugs' tikcets and correspoding bug summaries. Preprocessing techniques were applied to clean the data. The dimensions of the dataest is a prone errors as the training, testing and prediction datasets have to be dimensionally coherrent. NaN values must also be removed to avoid errors due to incompatibility of data types.

2) **Model developement:** Designing and training the supervised ML model to classify the bugs. Logistic regression and K-Means clustering were the different algorithms used for training the model.

3) **Similar bug retrieval:** Implementing a search mechanism by returning the most similar bug based on the similarity of the summary.

**Logistic Regression Model**(Refer to Bug_Prediction.ipynb in the main)

A detailed description of all the sections of code has been given in the notebook above each code snippet. 
