The python snippets given here find the similarities between user/items for the resstaurents and hotels domains using topic embedding generated with the help of topic modeling and word embedding. Based-on the topic embedding similarities, we predict the ratings. The whole program is written in Django webframework, which can be downloaded from, https://www.djangoproject.com/. To store the data, we have used MySQL database. Django uses Object-relational mapping (ORM) concept which provides an easy query and database setup.
# RecTE
Recommender System using topic embeddings using rating data and topic embedding, which is an amalgamation of word embedding and topic modeling techniques. The novelty of RecTE lies in predicting item ratings using topic embeddings learned by incorporating local and global contextual information and
integrating them with user-based collaborative filtering. We are sharing 5 files which includes model creation to rating prediction. Below I am providing a small description of each python file --

## models.py
In Django, model.py definitive source of information about the data. It contains the essential fields and behaviors of the data we are storing. Generally, each model maps to a single database table. It define the structure of stored data, including the field types and possibly also their maximum size, default values, selection list options, help text for documentation, label text for forms.

## preprocess.py/preprocess1.py
These python file translate the word into id in documents by creating a dictionary with words and indexing for both target and source users/items.

## CoEmbedding.py
This pytohn file creates the topic embedding using word embedding and topic modeling.

## views.py
A view function, or view for short, is a Python function that takes a web request and returns a web response. This response can be the HTML contents of a web page, or a redirect, or a 404 error, or an XML document, or an image . . . or anything, really. The view itself contains whatever arbitrary logic is necessary to return that response. This code can live anywhere you want, as long as it’s on your Python path. There’s no other requirement. For the sake of putting the code somewhere, the convention is to put views in a file called views.py, placed in your project or application directory. The view file is therefore the main file of the program.
