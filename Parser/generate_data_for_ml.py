import pickle
import os
import time
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))  # Current working directory


# Returns the data set for ML (consisting of name, article, gender

def load_in_name_gender_data():
    try:
        print("Loading the common_names_gender_object")

        temp = time.time()

        common_names_gender = pickle.load(
            open(dir_path.replace('/Parser', '/Resources' + os.sep + 'common_names_gender_list.obj'), 'rb'))

        print("common names with gender loaded in", time.time() - temp, 'seconds')

    except Exception as Ex:
        print("Generating the common_names_gender object .")
        common_names_gender = []

        with open(dir_path.replace('/Parser', '/Resources//all_names_with_gender_list.txt'), 'r') as file_reader:
            for line in file_reader.readlines():
                name, gender = line.strip().split(',')
                common_names_gender.append((name, gender))  # Appends in the name and the gender list in form of sets

        pickle.dump(common_names_gender,
                    open(dir_path.replace('/Parser', '/Resources//common_names_gender_list.obj'), "wb"))
        # dumps in the pickle

    # Using the common gender names

    print("Loading in Author Article List")

    temp = time.time()

    author_article_list = pickle.load(
        open(dir_path.replace('/Parser', '/Resources' + os.sep + 'author_article_xml_list.obj'), 'rb'))

    print("Loaded AUTHOR-ARTICLE Data in ", time.time() - temp, ' seconds')

    # common_gender_names = pickle.load(open(dir_path.replace('\Parser', '\Resources\\common_names_gender_list.obj'),
    #                                        'rb'))
    # print(author_article_list[0][0])  # Author name.
    # print(author_article_list[0][1])  # Article.
    print(common_names_gender[0][0], common_names_gender[0][1])

    author_article_gender_list = []
    df = pd.DataFrame()
    print("Matching & Tagging in genders with respect to  author names .")
    for count, (author_name, author_article) in enumerate(author_article_list):

        for (name, gender) in common_names_gender:

            if (name in author_name and len(name) > 2) and (name == author_name.strip().split(' ')[0]):
                # Because we have multiple names so we got to check all the things.
                # print(author_name.strip().encode('ascii', 'ignore'),
                #      author_article.replace(',', ' ').strip().encode('ascii', 'ignore'), gender.strip())
                print("Found.: ", author_name, '---->', gender)
                author_article_gender_list.append(
                    (author_name.strip().encode('ascii', 'ignore'),
                     author_article.replace(',', ' ').strip().encode('ascii', 'ignore'), gender.strip()))

                # print("found :", count, author_name.strip().encode('ascii', 'ignore'), name, gender)
                break  # when found leave it as it is.
                # Now goes out

    df['author'] = [elements[0] for elements in author_article_gender_list]  # Author
    df['article'] = [elements[1] for elements in author_article_gender_list]  # Gender
    df['gender'] = [elements[2] for elements in author_article_gender_list]  # Article
    return df  # this returns the author article gender list


# This get the author article gender list.


df = load_in_name_gender_data()  # Calls first
df.to_csv(dir_path.replace('/Parser', '/Resources//author_article_gender.csv'))


# load_in_name_gender_data()
