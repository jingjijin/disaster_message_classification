### disaster_message_classification
Real disaster messages used to build a classification web app 

# Project Summary:
In this project, a web application is built using data containing real messages
that were sent during disaster events. Data were multi-labeled with different
categories of disaster situations such as medical help, search and rescue, water
and food. The trained machine learning model in this application can be used to
predict what categories the future, potentially disaster-related, messages fall into, 
thereby helping disaster relief agencies provide appropriate actions accordingly.


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

