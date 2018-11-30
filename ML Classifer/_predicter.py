import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC  # linear svc classififer
import pickle
import os
# TODO: <meta content="Paid Death Notices" name="online_sections"/
# TODO: REMOVE THE DEATH STUFF.
dir_path = os.path.dirname(os.path.realpath(__file__))  # Current working directory


class Predictor:
    model = None

    def __init__(self):
        self.model = self.load_model()

    # Since  a data set would be a list of the articles which doens't have the gender specified so.
    def predict_data(self, data_set=None, default_threshold=0.63):
        if data_set is None:
            data_set = pickle.load(open(dir_path.replace('/MLClassifier', '/Resources//'
                                                                          'without_author_article_xml_list.obj'), 'rb'))
        # print(data_set)
        # print(type(self.model))
        gender_prob_list = self.model.predict_proba(data_set)  # This will predict the probabilities list
        classes_ = self.model.classes_
        gender_predicted_list = []  # This will hold the results which are predicted
        male_count, female_count, not_predicted_count = 0, 0, 0  # Will hold all the counts
        article_considered_list = []
        for count, (prob_list) in enumerate(gender_prob_list):
            if np.max(prob_list) > default_threshold:
                gender_predicted_list.append(
                    classes_[np.argmax(prob_list)])  # appending the nme os the specific class which was created
                if classes_[np.argmax(prob_list)].strip() == 'm':
                    male_count += 1
                else:
                    female_count += 1
                article_considered_list.append(data_set[count])  # Adds in the article for the specific
            else:
                not_predicted_count += 1
                pass

        predicted_df = pd.DataFrame()  # This will hold the predicted results.
        print("Predicted number of articles : ", male_count + female_count)
        print("No of Articles classified as male : ", male_count)
        print("No of Articles classified as Female : ", female_count)
        print("Not Classified because of less confidence :", not_predicted_count)
        predicted_df['gender'] = pd.Series(gender_predicted_list)  # Adding in the gender list
        predicted_df['article'] = pd.Series(article_considered_list)  # adds in the article with that as well
        # gender_predicted_list = self.model.predict(data_set) This holds the defult value

        return predicted_df

    def load_model(self):
        try:
            self.model = pickle.load(open(dir_path.replace('/MLClassifier', '/Resources//model.obj'), 'rb'))
            return self.model
        except Exception as Ex:
            print("No model has been classified by the _model class", Ex)  # printing in the exception

    def predict_through_console(self, article):
        if article != '':
            return self.model.predict(article)


def choice():
    P = Predictor()  # Creating the predictor lass
    P.load_model()  # Loading in the model
    run_state = True
    print(
        "Kindly select one from the following Press 1,2 or q \n\n 1 - Predict the gender of articles which didn't had author name\n 2 - Give input of an article\n 3 - enter q or quit to leave")
    while run_state:
        print("-" * 100)
        _choice = str(input("Choice:"))
        if _choice.strip() == 'q' or _choice.strip() == 'quit':
            run_state = False  # this will break the program
        elif _choice.strip() == '1':
            df = P.predict_data(data_set=None)  # This calls in the Predict class and predicts the respectable results
            df.to_csv(dir_path.replace('/MLClassifier', '/Resources//predicted_non_author_article_files.csv')) # Saves to csv
            #with open(dir_path.replace('\MLClassifier', '\Resources\\predicted_non_author_article_files.csv'),
            #          'w') as file_writer:
                # print("total number of files Predicted :", len(li))
                # for (article, gender) in li:
                #     file_writer.writelines(str(gender) + ',' + article.replace(',', '') + '\n')
                # file_writer.close()
                # print("Prediction have been stored in the output file ")
        elif _choice.strip() == '2':
            article_input = str(raw_input("Article input :"))
            gender = P.predict_through_console([article_input])  # Calls in the predictor for the article input
            print("Gender : %s" % gender)
        else:
            print("invalid choice")


choice()
