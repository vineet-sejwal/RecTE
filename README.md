The python snippets given here find the similarities between user/items for the resstaurents and hotels domains using topic embedding generated with the help of topic modeling and word embedding. Based-on the topic embedding similarities, we predict the ratings. The whole program is written in Django webframework, which can be downloaded from, https://www.djangoproject.com/. To store the data, we have used MySQL database. Django uses Object-relational mapping (ORM) concept which provides an easy query and database setup.
# RecTE
Recommender System using topic embeddings using rating data and topic embedding, which is an amalgamation of word embedding and topic modeling techniques. The novelty of RecTE lies in predicting item ratings using topic embeddings learned by incorporating local and global contextual information and
integrating them with user-based collaborative filtering. We are sharing 5 files which includes model creation to rating prediction. Below I am providing a small description of each python file --

# models.py
