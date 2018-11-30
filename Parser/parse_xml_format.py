from bs4 import *  # beautifulsoup pacakge for parsing
import os
import pickle  # for storing and loading of objects
import time

dir_path = os.path.dirname(os.path.realpath(__file__))  # Current working directory


# Gets all the file locations (xml file locations from the folder Data : XMLFILES
# TODO: Change it for having in all the folders.
def get_all_xml_files():
    data_dir_path = dir_path.replace('/Parser', '/Data//XMLFILES//')  # Replacing the data path

    xml_file_list = []

    for root, dirs, files in os.walk(data_dir_path):

        for _file in files:

            if _file.endswith(".xml"):
                xml_file_list.append(os.path.join(root, _file))

    return xml_file_list  # returns the xml file list


# Returns the data from all the xml files
def get_data_from_all_xml_files():
    author_exists_count = 0
    only_article_count = 0
    death_notice_count = 0

    total_xml_files = len(get_all_xml_files())
    print("Total Number of files : ", total_xml_files)  # total xml files !!!!
    # exit(0)
    # Loading in the already parsed files
    try:
        already_parsed_file_list = pickle.load(
            open(dir_path.replace('/Parser', '/Resources//already_parsed_file_list.obj'), 'rb'))  # the file list

        author_article_xml_data = pickle.load(
            open(dir_path.replace('/Parser', '/Resources//author_article_xml_list.obj'), 'rb'))  # the xml list
        without_article_xml_data = pickle.load(
            open(dir_path.replace('/Parser', '/Resources//without_author_article_xml_list.obj'), 'rb'))
    except Exception as Ex:
        print("Creating the parsing object file", Ex)
        # print("I am here !!")
        already_parsed_file_list = []  # Contains the location for the file

        author_article_xml_data = []  # Contains the article and the author xml data list
        without_article_xml_data = []  # Contains the articles which are without an article list
    # print("now here")
    for xml_file in get_all_xml_files():
        print(xml_file)
        if xml_file in already_parsed_file_list:

            continue
        else:
            already_parsed_file_list.append(
                xml_file)  # This adds in the xml file location so that it is not parsed again
            with open(xml_file, 'r') as xml_file_reader:
                data_from_file = xml_file_reader.read()

                soup = BeautifulSoup(data_from_file, 'lxml')  # Passing in the BS 4 for the LXML parsing(easiest to use)
                paid_notices = soup.findAll('meta', {'content': "Paid Death Notices"})
                if len(paid_notices) != 0:
                    death_notice_count += 1
                    print("Skipped : ", xml_file)
                    continue

                try:
                    # <meta content="Paid Death Notices" name="online_sections"/
                    # author_exists_count += 1
                    # print(author_exists_count)
                    author_name = soup.findAll('byline', {'class': 'print_byline'})[0].text.replace('By ', ' ').lower()
                    article_data = ''.join(
                        str(paragraph.text) for paragraph in soup.findAll('p'))  # Gets all the paragraphs from the xml
                    author_article_xml_data.append((author_name, article_data))  # (authorname, article data
                    # Appends in the xml in form of author name and article data
                    author_exists_count += 1
                    print("Parsed File :", xml_file, author_exists_count)
                # If the file doesn't have the author name and does have the article data then
                except Exception as Ex:
                    try:
                        article_data = ''.join(
                            str(paragraph.text) for paragraph in
                            soup.findAll('p'))  # Gets all the paragraphs from the xml
                        only_article_count += 1
                        without_article_xml_data.append(article_data)  # Adds in the article data without the tags
                    except Exception as Ex:
                        print("un parsable file :")
                        # print("author doesn't exists")
    print("Parsed Author XML FILES  : ", author_exists_count)
    print("Parsed Only article files : ", only_article_count)
    print("Death Notice Files Ignored : ", death_notice_count)
    print("Non Metadata files : ", (total_xml_files - (author_exists_count + only_article_count + death_notice_count)))
    print("Total Number of files : ", total_xml_files)  # total xml files !!!!
    if author_exists_count != 0:
        print("dumping author articles into object")
        pickle.dump(author_article_xml_data,
                    open(dir_path.replace('/Parser', '/Resources//author_article_xml_list.obj'),
                         'wb'))  # This dumps the List for authors,

        # dumps the xml file locations
        print("dumping xml files parsed. ")
        pickle.dump(already_parsed_file_list,
                    open(dir_path.replace('/Parser', '/Resources//already_parsed_file_list.obj'), 'wb'))
        pickle.dump(without_article_xml_data,
                    open(dir_path.replace('/Parser', '/Resources//without_author_article_xml_list.obj'), 'wb'))

    print("parsing completed")
    return  # YOU CAN USE THE RETURNING LATER
    # return author_article_xml_data  # returns the author name, article data list in form of sets


# loads and parses the xml files
def load_parse_xml_files():
    get_data_from_all_xml_files()


# checking by calling in for all the parsing files
load_parse_xml_files()
