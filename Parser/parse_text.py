from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

file_data = open('1815749.xml', 'r').read()  # Reading a sample file
soup = BeautifulSoup(file_data, 'lxml')  # Passing in the BS 4 for the LXML parsing(easiest to use)
# print(soup)
author_name = soup.findAll('byline', {'class': 'print_byline'})[0].text.replace('By ', '')   # Finds the byline in the xml format.
article_data = '. '.join(str(paragraph.text) for paragraph in soup.findAll('p'))  # Gets all the paragraphs from the xml

for _ in article_data.split('. '):
    print(_)
