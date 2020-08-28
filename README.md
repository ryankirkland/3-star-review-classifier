# Mixed Reviews: The Voice of the Customer

## Overview

<img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/Amazon-Logo.png'>

The following is an analysis of Amazon customer reviews within the rechargeable batteries category. The goal is run the text from the reviews through Natural Language Processing models after 3-star reviews have been run through a classification model to ensure every customer voice is heard. Latent topics gathered as a result of the NLP modeling will then be used to make recommendations in regards to customer service, product page content, and product development based on customer voice.

## Data and Technologies Used

<img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/reviews.png'>

#### Tech Used
- Requests (retrieving HTML)
- BeautifulSoup (Parsing HTML)
- Python 3
- Pandas
- NumPy
- SciKit Learn

## EDA

Some quick exploration to get an idea of the distribution of reviews by rating and whether or not they were verified purchase was done to understand any potential nuance to the scraped reviews:

- An initial concern was the overwhelming majority of reviews being 5-star having a heavy influence on the models' ability to classify. This only presented to be true when using Multinomial Naive Bayes to predict star rating on a scale from 1 - 5. Converting the 1-5 scale to a binary positive or negative classification resolved this issue.

<b> Reviews by Rating </b>

<img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/rating.png'>

- The presence of largely "Verified Purchase" reviews is important, as a significant presence of non-verified reviews is generally an indicator of gaming the system through fake review generation, which also creates a dataset of false information that does not accurately represent the sentiment of a product's true consumer.

<b> Reviews by Verified Status </b>

<img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/verified.png'>

## Models
### Classification

The chosen classifiers for comparison in this round of testing were Random Forest Classification and Multinomial Naive Bayes, as the original idea was to try to predict star rating. As mentioned prior, the MultinomialNB was heavily influenced by the presence of 5-star reviews, which led me to relabel the data where Positive was > 3 stars and Negative was < 3 stars - 3 star reviews are held out. After testing, Random Forest outperformed Naive Bayes with an accuracy score of 87.5% vs. the NB accuracy of 74.5%. NB did get a jump to accuracy (80.2%) after taking a random sample of 5-star reviews of equivalent count to the count of 1-star reviews to attempt to balance the scales. The same test was done on Random Forest, which hurt the RF's accuracy (dropping to 84%).

- <b> ROC Curve of Random Forest: </b>

<img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/roc.png' width=75%>

After moving forward with Random Forest, thresholds for probability of a review being positive were tested with a Profit Curve based on the confusion matrix multiplied by a cost-benefit matrix. With some assumptions made around the cost to contact a customer with a customer service rep, coupon offered, the likelihood of conversion after contact, and customer lifetime value, the threshold of 0.79 was deemed the most likely to save money against the total reviews.

- <b> Confusion Matrix and Profit Curve: </b>

<div>
  <img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/cs_conf_mat.png' width=49%>
  <img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/profit_curve.png' width=49%>
</div

A second threshold was determined, surprisingly at the default 0.5, for classification to be input into the NLP pipeline. Due to the sheer volume of 5-star reviews in this case, it is far more likely that false negatives would heavily sway the results of latent topic discovery, where false negatives would more likely be buried under the weight of true positives when analyzing the review content.

- <b> Confusion Matrix for NLP Feed: </b>

<img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/conf_mat.png' width=49%>

### Topic Modeling

After classifying the reviews, they are then processed through an NLP pipeline consisting of punctuation removal, word tokenization, lemmatization, the maintenance of unigrams with the addition of bigrams and trigrams for further context, review separation into Positive and Negative groups, then finally vectorization through Sklearn's TFIDF Vectorizer. The vectorized strings are passed through Non-Negative Matrix Factorization to uncover latent topics, which can be seen below:

- <b> Latent Topics for Negative Reviews of Rechargeable AA/AAA Batteries: </b>

<img src='https://github.com/ryankirkland/review-content-analysis/blob/master/img/neg_topics.png' width=80%>

Based on the above, it is clear that issues with holding charge should be top concern for brands selling rechargeable batteries. Positive topics largely discussed value for the money.

## Final Recommendation

- The positive feedback from the latent topics of the review indicates detail page content should be updated to speak to cost-savings and the true value for the money spent. Expectations should be set in regards to the length of holding a charge, but balanced out by the number of reuses, time-to-charge, and an estimation of cost savings when compared to the cost associated to purchasing disposables.

- Based on the negative feedback, product development research should largely be focused on how to get these rechargeables to hold charge for longer periods of time or bring the overall experience to at least parity to that of disposables (things like indications of charge level in products).

- From the profit curve above, a significant cost-savings can be realized from reaching out to those leaving negative reviews to troubleshoot with them and offer incentive to either keep their product after learning how to better use it or return the old product and use an incentive for repurchase.
