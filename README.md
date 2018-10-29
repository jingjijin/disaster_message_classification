### disaster_message_classification
Real disaster messages used to build a classification web app 

# Project Summary:
In this project, a web application is built using data containing real messages
that were sent during disaster events. Data were multi-labeled with different
categories of disaster situations such as medical help, search and rescue, water
and food. Therefore, the trained model in this application can be used to
predict future, potentially disaster-related, messages into these categories so
that disaster relief agencies could make appropriate actions accordingly.


# Files
data folder:
 - 'disaster_messages.csv': file containing all disaster_messages
 - 'disaster_categories.csv': file containing labels to the disaster_messages
 - 'process_data.py': python script used to clean and store data into SQLite db
 - 'InsertDatabaseName.db': database to save clean data to

models folder:
 - 'train_classifier.py': python script used to build and save NLP ML model
 - 'classifier.pkl': saved model

app folder
 - template
  - 'master.html': main page of web app
  - 'go.html': classification result page of web app
 - 'run.py': Flask file that runs app



# Instructions to Run the Application
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/
        disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/
        classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
