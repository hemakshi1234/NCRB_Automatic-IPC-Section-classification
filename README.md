# NCRB_Automatic-IPC-Section-classification
This is an API for classification of IPC section 302 , 307 and 376

<h3 align="left">About the Project</h3>

At present, while registering an FIR or Investigation, Investigation officers correlate the torts with the various penal codes and choose appropriate Act & Sections. This necessitates IO to have prior deep knowledge and a clear understanding of the criminal Law definitions and Interpretations. As the torts are predefined (definitions),
it is possible to predict the relevant sections from the compliant text or FIR description or Investigation reports using Artificial technologies (AI) like Machine Learning, NLP, and Deep Learning methodologies.

<h3 align="left">Functionality</h3>

A text analyzer for **Automatic Crime Head/Section Addition** using NLP has been built using **Python, Flask, Postman, NLTK, Machine Learning, and Natural Language**. The data has been collected from the Internet using news reports based on the crime committed under IPC sections 302, 307, and 376, along with their definitions, and was organized and structured in the form of labeled data in a CSV file to train the model.
The model with the highest accuracy is exposed as API using flask and Postman. Since a colossal amount of data to process is impossible for humans to do it alone. If machines are made solely responsible for sorting through data using text analysis models and machine learning, the benefits for the organization will be huge. The basic
functions being performed by model are taking input data and returning output after analyzing along with the IPC section. 

* The API provides the end user with two input functions. The first function takes a simple text as input and on executing this function it will categorize the text content sentence wise with corresponding predicted labels of IPC Section. 
* The second function takes a text file as input and on executing this function it will categorize the contents of text file sentence wise with corresponding predicted labels of IPC Section.

<h3 align="left">Tech Stack</h3>

* **Python:** The code for the text analyzer has been written in python
* **Natural Language Processing:** By using Natural Language Processing, we will make the computer truly understand more than just the objective definitions of the words. This analysis will help us segregate the data that has elements as defined in Sections 302, 307 and 376 IPC. It includes using Bag of Words model which is a way of extracting features from the text for use in modeling.
* **Machine Learning:** A classifier or classification algorithm has been used to identify whether a given piece of text is Sections 302, 307 and 376 IPC. In this case, we are using Support Vector Machine as our classifier because of higher accuracy.
* **NLTK:** NLTK (Natural Language Toolkit) is a popular open-source package in Python. Rather than building all tools from scratch, NLTK provides all common NLP Tasks. 
* **Flask:** Flask is a lightweight WSGI web application framework. It is designed to make getting started quick and easy, with the ability to scale up to complex applications.
* **Flasgger:** Flasgger is a Flask extension to help the creation of Flask APIs with documentation and live playground powered by SwaggerUI.
* **Postman:**  Postman is a great tool when trying to dissect RESTful APIs made by others or test ones we have made our self. It offers a sleek user interface with which to make HTML requests, without the hassle of writing a bunch of code just to test an API's functionality.
