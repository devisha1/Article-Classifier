import pandas as pd
import time  # just for checking the time
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split  # training and test splits
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline  # pipeline for the classification.
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.multiclass import OneVsOneClassifier
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix  # Confusion matrix
from sklearn.model_selection import cross_val_score  # Cross validation score for the machine learning face
from sklearn.model_selection import GridSearchCV  # Grid search cv for checking multiple parameters
from sklearn.neighbors import KNeighborsClassifier  # K neigbour classifier.
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier.
from sklearn.ensemble import AdaBoostClassifier  # Ada boost classifier
from sklearn.tree import DecisionTreeClassifier  # Decision tree classifier
from sklearn.gaussian_process import GaussianProcessClassifier
import os
from sklearn.metrics import precision_recall_fscore_support  # for the precision, recall , f1 and support score

dir_path = os.path.dirname(os.path.realpath(__file__))  # Current working directory


class Model:
    """ A model class is used for the training purposes,
    it has different methods each has different responsibilities.
    """
    # TODO:
    pipeline = None
    df = None
    model = None
    model_score = None
    confusion_matrix = None
    classifier_name = None

    def __init__(self, df=None, pipeline=None):
        if pipeline is None:
            self.pipeline = self.get_pipeline()
            self.df = df.dropna()
        pass

    def set_classifier_pipeline(self, classifier):
        self.pipeline = Pipeline(
            [
                ('vectorizer', CountVectorizer(stop_words='english', max_df=0.8)),  # using the english stop words
                ('tfidf', TfidfTransformer()),  # tf idf transformer
                ('classifier', classifier)  # classifier any !
                # min 60 , max 66 , avg 64
            ]
        )  # A pipeline for the logistic Regression with the Multinomial , newtons solver.

        return

    # getting the pipeline
    def get_pipeline(self):
        # Logistic Classification multi nomial
        self.pipeline = Pipeline(
            [
                ('vectorizer', CountVectorizer(stop_words='english', max_df=0.8)),  # using the english stop words
                ('tfidf', TfidfTransformer()),  # tf idf transformer
                ('classifier', LogisticRegression(solver='newton-cg', multi_class='multinomial'))  # multiclass based
                # min 60 , max 66 , avg 64
            ]
        )  # A pipeline for the logistic Regression with the Multinomial , newtons solver.
        return self.pipeline  # Pipeline.


    # Train model
    def train_model(self, default_threshold=0.65, number_of_times_to_check=5):
        article = self.df['article']
        gender = self.df['gender']
        print(len(article))
        m_count = sum([1 for _ in gender if _ == 'm'])
        print('Training & Testing : count of males : ', m_count)
        print('Training & Testing : count of females', len(article) - m_count)

        x_train, x_test, y_train, y_test = train_test_split(article, gender, test_size=0.1)
        self.model = self.pipeline.fit(x_train, y_train)  # Fitting in the pipeline
        self.model_score = self.pipeline.score(x_test, y_test)
        predicted_probabilities_test = self.pipeline.predict_proba(x_test)
        classes_ = self.pipeline.classes_  # gets the classes (used for classification)
        predicted_x_test = []
        tp, fp, nc, pc = 0, 0, 0, 0  # True positive , false positive count, null count , predicted count
        # print(x_test[0])
        # exit(0)
        # TODO :  precision , recall , f1 and the support score here
        predicted_after_threshold, actual_after_threshold = list(), list()  # list creation.
        for actual_predicted, predicted_probability in list(zip(y_test, predicted_probabilities_test)):
            if np.max(predicted_probability) >= default_threshold:  # Compares with threshold
                pc += 1  # predicted count incremented
                # if the classes is an exact match then add the true positive count to be 1
                predicted_after_threshold.append(classes_[np.argmax(predicted_probability)].strip())
                actual_after_threshold.append(actual_predicted)  # Actual prediction appending .

                if classes_[np.argmax(predicted_probability)].strip() == actual_predicted:
                    tp += 1  # true postiive count added
                else:
                    fp += 1  # False positive count incremented  !
                predicted_x_test.append(classes_[np.argmax(predicted_probability)])  # Adds in the prediction list
            else:
                predicted_x_test.append('NONE')  # Since the model is not sure so it would be counted as NONE
                nc += 1  # none count is added in by 1
        p_, r_, f1_, s_ = precision_recall_fscore_support(actual_after_threshold,
                                                          predicted_after_threshold)  # getting the
        # precision , recall , f1 and the support score . !
        confusion_matrix = self.get_confusion_matrix(actual_after_threshold,predicted_after_threshold) # Confusion matrix

        print('classes : ', list(self.pipeline.classes_))
        print("Confusion Matrix : ",confusion_matrix)
        print('precision : ', list(p_))
        print('recall : ', list(r_))
        print('f1 score : ', list(f1_))
        print('support :', list(s_))
        #exit(0)
        accuracy = float(tp) / float(tp + fp)
        print(accuracy)

        # print("model score is :", self.model_score)
        # print("Checking cross validation:")
        # cross_val_scores = cross_val_score(self.pipeline, article, gender, cv=10)
        # print("cv_scores ", cross_val_scores)
        # print('maximum model score : ', np.max(cross_val_scores))
        # print('minimum model score :', np.min(cross_val_scores))
        # print('average model score :', np.mean(cross_val_scores))
        # print(self.model.classes_)  # Just checking the classes
        # print("model score is :", self.model_score)
        return accuracy * 100.0, tp, fp, nc, pc, len(
            y_test), self.model  # returns the accuracy , null count and the model

    def get_confusion_matrix(self, y_true, y_predicted):
        self.confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_predicted)
        return self.confusion_matrix


def apply_classification_using_models():
    print("Loading in the data set")
    M = Model(df=pd.read_csv('/Users/Devisha/Desktop/ArticleClassifier/Resources/author_article_gender.csv'))
    print("data has been loaded")
    # predict_proba() # < ---
    # Below is the classifiers list.
    classifier_list = {"logistic": LogisticRegression(solver='newton-cg')  # ,
                       # "Decision Tree Classifier :": DecisionTreeClassifier(),
                       # "Linear SVC": LinearSVC()  # ,
                       # "svc classification": SVC(kernel='rbf'),
                       # 'One vs one classifier': OneVsOneClassifier(estimator=1),
                       # "multi Naive ": MultinomialNB(),
                       # "bernouli naive bayes": BernoulliNB(),
                       # "Ada boost classifier": AdaBoostClassifier(),
                       # "sophisticated gradient decent": SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                       #                                               random_state=42),
                       # "Gaussian bayes classifier :": GaussianProcessClassifier(),
                       # "KNN classifier": KNeighborsClassifier(n_neighbors=10),
                       # "Random Forest": RandomForestClassifier() # HAHAHAHA :D yes!
                       }
    highest_score = 0.0
    highest_classifier_name = ""
    highest_model = None
    model = None  # Will be Holding the model for the storage
    for classifier_name, classifier in classifier_list.items():  # Use all the classifier provided in the list
        print('-' * 50)
        temp = time.time()
        print("Classifier being used : ", classifier_name)
        M.set_classifier_pipeline(classifier)  # Asks in for the classification
        model_score, tp, fp, nc, pc, ts, model = M.train_model()  # Trains the model
        if model_score > highest_score:
            highest_score = model_score
            highest_classifier_name = classifier_name
            highest_model = model
        print("Model evaluated in %s (seconds) :" % str(temp - time.time()))  # prints up the time
        print("accuracy :", model_score)
        print("Total test size: ", ts)
        print("Predicted : ", pc)
        print("True predicted :", tp)
        print("False predicted : ", fp)
        print("Not predicted: ", (ts - pc))  # Nor predicted count would be stored in here
        print('-' * 50)
    print(dir_path.replace('/MLClassifier', '/Resources' + os.sep + 'model.obj'))
    pickle.dump(highest_model, open(dir_path.replace('/MLClassifier', '/Resources//model.obj'), 'wb'))
    print("Model have been saved : ")
    print("-" * 100)
    print("Highest Score is : ", highest_score)
    print("Classifier is : ", highest_classifier_name)


apply_classification_using_models()
