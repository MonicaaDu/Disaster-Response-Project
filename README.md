# Disaster-Response-Project

## Description
This project is a part of Data Science Nanodegree Program by Udacity.
The project is to build a Natural Language Processing (NLP) model and show result in a web app.
This project is divided in the following key sections:
1. Data Processing: ETL Pipeline to extract data from source, clean data and save them in a database 
2. Machine Learning Pipeline to train a model able to classify text message in categories
3. Web App to show model results in real time.

## Installation
- Python 3.5+ (I used Python 3.7)
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly

## Executing Program
1. Run the following commands in the project's root directory to set up your database and model:
- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app. python run.py
3. Go to http://0.0.0.0:3001/

## Additional Matreial
- ETL Preparation Notebook: data processing step by step
- ML Pipeline Preparation Notebook: Machine Learning Model step by step
