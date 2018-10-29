'''
ML script

This machine learning script takes the database file path, creates and trains
a classifier, and stores the classifier into a pickle file to the specified
model file path.

'''

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle


def load_data(database_filepath):
    '''
    This function loads the data from Sqlite database file and returns messages
    data as well as the labels for the messages.

    inputs:
    database_filepath - path to the db file

    outputs:
    X - 'messages' (also the inputs for ML)
    y - 36 labels/categories (also the outputs for ML)
    cols - the names of 36 labels/categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name=database_filepath,con=engine)
    new_cols = []

    for col in df.columns:
        new_cols.append(col[5:-2])

    df.columns = new_cols
    X = np.array(df['message'])
    y = np.array(df.drop(['id','message','original','genre'],axis=1))
    cols = df.drop(['id','message','original','genre'],axis=1).columns
    return X, y, cols





def tokenize(text):
    '''
    This customized tokenize function tokenizes, strips, and
    lemmetizes the messages before saving them as clean tokens.

    inputs:
    text - messages that we want to tokenize

    outputs:
    clean_tokens - clean tokens transformed from messages
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens





def build_model():
    '''
    This function builds a machine learning model using pipeline and grid
    search.

    inputs:
    nothing to provide; models and parameters can be directly modified below

    outputs:
    grid - random forest classifier with a few searchable parameters are
    added to the grid function (model)

    '''
    rf = RandomForestClassifier()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenize,lowercase=True)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(rf))])

    parameters = {
     'classifier__estimator__max_depth': [10,20,None],
     'classifier__estimator__min_samples_leaf': [1,2,3],
     'classifier__estimator__n_estimators': [10,100,200]}

    grid = GridSearchCV(estimator=pipeline, param_grid=parameters)
    return grid




def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates the model using test data.

    inputs:
    model - estimator/classifier
    X_test - test data contain messages
    Y_test - test data contrain true labels
    category_names - names of the 36 labels

    outputs:
    prints out accuracy, precision and f1 score of the test data
    for each category
    '''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred,columns=category_names)
    Y_test = pd.DataFrame(Y_test,columns=category_names)

    results = pd.DataFrame(columns=['precision','recall','f1_score'],index=list(range(36)))
    i = 0
    for col in category_names:
        scores = precision_recall_fscore_support(Y_test[col], Y_pred[col], average='macro')
        precision = scores[0]
        recall = scores[1]
        f1_score = scores[2]
        results.iloc[i] = [precision,recall,f1_score]
        i +=1
    results.index = category_names
    print(results)




def save_model(model, model_filepath):
    '''
    This function saves the best ML model as a pickle file.

    inputs:
    model - the model from gridsearch
    model_filepath - path to save the models

    outputs:
    saves the model as a pickle file in desired path
    '''
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()




def main():
    '''
    The main function that loads db file and uses the data from db to train
    the classifier and save the trained model as a pickle file.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
