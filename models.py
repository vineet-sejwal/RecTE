from __future__ import unicode_literals

from django.db import models

# Create your models here.
class User_Context(models.Model):

    user_id = models.TextField(max_length=50,blank=True,null=True)

    user_votes = models.TextField(max_length=50,blank=True,null=True)

    review_id = models.TextField(max_length=50,blank=True,null=True)

    user_ratings = models.TextField(max_length=20,blank=True,null=True)

    user_reviews = models.TextField(max_length=10000,blank=True,null=True)

    reviews_date = models.TextField(max_length=20,blank=True,null=True)

    buisness_id = models.TextField(max_length=50,blank=True,null=True)

    review_classification = models.TextField(max_length=10,blank=True,null=True)

    specific_sentences = models.TextField(max_length=1000,blank=True,null=True)

    generic_sentences = models.TextField(max_length=1000,blank=True,null=True)

    def __unicode__(self):
		return self.user_id



class User_Context_20and50(models.Model):

    user_id = models.TextField(max_length=50,blank=True,null=True)

    review_id = models.TextField(max_length=50,blank=True,null=True)

    user_ratings = models.TextField(max_length=20,blank=True,null=True)

    user_reviews = models.TextField(max_length=10000,blank=True,null=True)

    buisness_id = models.TextField(max_length=50,blank=True,null=True)

    review_classification = models.TextField(max_length=10,blank=True,null=True)

    def __unicode__(self):
        return self.user_id


class Yelp_NYC(models.Model):

    reviewer_id = models.TextField(max_length=50,blank=True,null=True)

    product_id = models.TextField(max_length=50,blank=True,null=True)

    review_ratings = models.TextField(max_length=20,blank=True,null=True)

    review = models.TextField(max_length=10000,blank=True,null=True)

    review_classification = models.TextField(max_length=10,blank=True,null=True)

    specific_sentences = models.TextField(max_length=1000,blank=True,null=True)

    generic_sentences = models.TextField(max_length=1000,blank=True,null=True)


    def __unicode__(self):
        return self.reviewer_id

class Yelp_ZIP(models.Model):

    reviewer_id = models.TextField(max_length=50,blank=True,null=True)

    product_id = models.TextField(max_length=50,blank=True,null=True)

    review_ratings = models.TextField(max_length=20,blank=True,null=True)

    review = models.TextField(max_length=10000,blank=True,null=True)

    review_classification = models.TextField(max_length=10,blank=True,null=True)

    def __unicode__(self):
        return self.reviewer_id