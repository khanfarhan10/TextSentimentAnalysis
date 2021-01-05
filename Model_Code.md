```python
from google.colab import drive
drive.mount('/content/drive/')
```

    Mounted at /content/drive/
    


```python
pip install scikit-plot
```

    Collecting scikit-plot
      Downloading https://files.pythonhosted.org/packages/7c/47/32520e259340c140a4ad27c1b97050dd3254fdc517b1d59974d47037510e/scikit_plot-0.3.7-py3-none-any.whl
    Requirement already satisfied: joblib>=0.10 in /usr/local/lib/python3.6/dist-packages (from scikit-plot) (1.0.0)
    Requirement already satisfied: scikit-learn>=0.18 in /usr/local/lib/python3.6/dist-packages (from scikit-plot) (0.22.2.post1)
    Requirement already satisfied: scipy>=0.9 in /usr/local/lib/python3.6/dist-packages (from scikit-plot) (1.4.1)
    Requirement already satisfied: matplotlib>=1.4.0 in /usr/local/lib/python3.6/dist-packages (from scikit-plot) (3.2.2)
    Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn>=0.18->scikit-plot) (1.19.4)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=1.4.0->scikit-plot) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=1.4.0->scikit-plot) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=1.4.0->scikit-plot) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=1.4.0->scikit-plot) (2.4.7)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from cycler>=0.10->matplotlib>=1.4.0->scikit-plot) (1.15.0)
    Installing collected packages: scikit-plot
    Successfully installed scikit-plot-0.3.7
    


```python
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,auc,roc_auc_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from scikitplot.metrics import plot_confusion_matrix
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
%matplotlib inline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
#read the data from google drive its name is Review.csv file
reviews=pd.read_csv("/content/drive/MyDrive/Reviews.csv")
reviews=reviews[:40000]
reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>B001E4KFG0</td>
      <td>A3SGXH7AUHU8GW</td>
      <td>delmartian</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1303862400</td>
      <td>Good Quality Dog Food</td>
      <td>I have bought several of the Vitality canned d...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>B00813GRG4</td>
      <td>A1D87F6ZCVE5NK</td>
      <td>dll pa</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1346976000</td>
      <td>Not as Advertised</td>
      <td>Product arrived labeled as Jumbo Salted Peanut...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>B000LQOCH0</td>
      <td>ABXLMWJIXXAIN</td>
      <td>Natalia Corres "Natalia Corres"</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1219017600</td>
      <td>"Delight" says it all</td>
      <td>This is a confection that has been around a fe...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>B000UA0QIQ</td>
      <td>A395BORC6FGVXV</td>
      <td>Karl</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1307923200</td>
      <td>Cough Medicine</td>
      <td>If you are looking for the secret ingredient i...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>B006K2ZZ7K</td>
      <td>A1UQRSCLF8GW1T</td>
      <td>Michael D. Bigham "M. Wassir"</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1350777600</td>
      <td>Great taffy</td>
      <td>Great taffy at a great price.  There was a wid...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Lets convert our sccores to three labels 0,1,2 for , negative(score>3),Positive(score>3),neutral(score==3)
def label(x):
    if x>3:
        return 1
    elif x<3:
        return 0
    elif x==3:
        return 2
reviews["Score"]=reviews["Score"].map(label)
reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>B001E4KFG0</td>
      <td>A3SGXH7AUHU8GW</td>
      <td>delmartian</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1303862400</td>
      <td>Good Quality Dog Food</td>
      <td>I have bought several of the Vitality canned d...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>B00813GRG4</td>
      <td>A1D87F6ZCVE5NK</td>
      <td>dll pa</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1346976000</td>
      <td>Not as Advertised</td>
      <td>Product arrived labeled as Jumbo Salted Peanut...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>B000LQOCH0</td>
      <td>ABXLMWJIXXAIN</td>
      <td>Natalia Corres "Natalia Corres"</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1219017600</td>
      <td>"Delight" says it all</td>
      <td>This is a confection that has been around a fe...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>B000UA0QIQ</td>
      <td>A395BORC6FGVXV</td>
      <td>Karl</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1307923200</td>
      <td>Cough Medicine</td>
      <td>If you are looking for the secret ingredient i...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>B006K2ZZ7K</td>
      <td>A1UQRSCLF8GW1T</td>
      <td>Michael D. Bigham "M. Wassir"</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1350777600</td>
      <td>Great taffy</td>
      <td>Great taffy at a great price.  There was a wid...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#lets check if our data is having duplicate values
reviews[reviews[["UserId","ProfileName","Time","Text"]].duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>B0001PB9FY</td>
      <td>A3HDKO7OW0QNK4</td>
      <td>Canadian Fan</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1107820800</td>
      <td>The Best Hot Sauce in the World</td>
      <td>I don't know if it's the cactus or the tequila...</td>
    </tr>
    <tr>
      <th>574</th>
      <td>575</td>
      <td>B000G6RYNE</td>
      <td>A3PJZ8TU8FDQ1K</td>
      <td>Jared Castle</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1231718400</td>
      <td>One bite and you'll become a "chippoisseur"</td>
      <td>I'm addicted to salty and tangy flavors, so wh...</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>1974</td>
      <td>B0017165OG</td>
      <td>A2EPNS38TTLZYN</td>
      <td>tedebear</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1312675200</td>
      <td>Pok Chops</td>
      <td>The pork chops from Omaha Steaks were very tas...</td>
    </tr>
    <tr>
      <th>2309</th>
      <td>2310</td>
      <td>B0001VWE0M</td>
      <td>AQM74O8Z4FMS0</td>
      <td>Sunshine</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1127606400</td>
      <td>Below standard</td>
      <td>Too much of the white pith on this orange peel...</td>
    </tr>
    <tr>
      <th>2323</th>
      <td>2324</td>
      <td>B0001VWE0C</td>
      <td>AQM74O8Z4FMS0</td>
      <td>Sunshine</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1127606400</td>
      <td>Below standard</td>
      <td>Too much of the white pith on this orange peel...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39976</th>
      <td>39977</td>
      <td>B001TZJ3OE</td>
      <td>A3908E1G8IL52G</td>
      <td>Jessica Ztardust</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1312243200</td>
      <td>Highly Addicting!</td>
      <td>I did not purchase this product off of Amazon....</td>
    </tr>
    <tr>
      <th>39977</th>
      <td>39978</td>
      <td>B001TZJ3OE</td>
      <td>AF1PV3DIC0XM7</td>
      <td>Robert Ashton</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1081555200</td>
      <td>Classic Condiment</td>
      <td>Mae Ploy Sweet Chili Sauce is becoming a stand...</td>
    </tr>
    <tr>
      <th>39978</th>
      <td>39979</td>
      <td>B001TZJ3OE</td>
      <td>A1VTHOTQFPRFVT</td>
      <td>jumperboy</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1301961600</td>
      <td>Just Okay, Too Sweet</td>
      <td>I was excited to try this sauce based on the r...</td>
    </tr>
    <tr>
      <th>39979</th>
      <td>39980</td>
      <td>B001TZJ3OE</td>
      <td>AAJ1IYOUIHWF</td>
      <td>D. Sun</td>
      <td>1</td>
      <td>11</td>
      <td>2</td>
      <td>1243036800</td>
      <td>Too much</td>
      <td>These are very large bottles. It is a good dip...</td>
    </tr>
    <tr>
      <th>39980</th>
      <td>39981</td>
      <td>B001TZJ3OE</td>
      <td>A3E3YJO2V3YZUM</td>
      <td>Lidgemeister</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>1295481600</td>
      <td>Guess I'm in the Minority</td>
      <td>I was looking for a good sweet and sour sauce ...</td>
    </tr>
  </tbody>
</table>
<p>2548 rows × 10 columns</p>
</div>




```python
# we can see in the above cell that there are duplicates in our data. Lets drop all of them.
print("Data set size before dropping duplicates",reviews.shape)
reviews_df = reviews.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep='first')
print("Data set size after dropping duplicates",reviews_df.shape)
```

    Data set size before dropping duplicates (40000, 10)
    Data set size after dropping duplicates (37452, 10)
    


```python
#let us find if our data have any missing values
# from now on we deal only with Text and Score columns, Text is our feature and score is our label.
reviews_df[["Text","Score"]].isnull().any()
```




    Text     False
    Score    False
    dtype: bool




```python
print("Amount of data retianed is : ", reviews_df.shape[0]/reviews.shape[0])
```

    Amount of data retianed is :  0.9363
    


```python
plt.bar(reviews_df["Score"].unique(),reviews_df["Score"].value_counts())
plt.xticks([0,1,2])
plt.show()
```


![png](Model_Code_files/Model_Code_9_0.png)



```python
plt.pie(reviews_df["Score"].value_counts(),autopct='%1.0f%%',radius=2,labels=reviews_df["Score"].unique(),colors=["g","r","y"])
plt.title("Labels")
```




    Text(0.5, 1.0, 'Labels')




![png](Model_Code_files/Model_Code_10_1.png)


# Observations
After removing duplicates and missing values we were able to retain 93.6% of actual data.
From the above bar plot we can clearly see that our data is imbalance

# 3. Data Preprocessing
Though we removed noise data, we need to make sure that our data is clean with text data comes a lot of unwanted characters, symbols, numbers and common words which adds no value to the model's performance so we will try to remove these unwanted characters to get a clean data


```python
#21,15,28
review34=reviews_df["Text"][34]
review34
```




    "Instant oatmeal can become soggy the minute the water hits the bowl. McCann's Instant Oatmeal holds its texture, has excellent flavor, and is good for you all at the same time. McCann's regular oat meal is excellent, too, but may take a bit longer to prepare than most have time for in the morning. This is the best instant brand I've ever eaten, and a very close second to the non-instant variety.<br /><br />McCann's Instant Irish Oatmeal, Variety Pack of Regular, Apples & Cinnamon, and Maple & Brown Sugar, 10-Count Boxes (Pack of 6)"




```python
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
reviews_df=reviews_df[["Text","Score"]]
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    


```python
print(stop_words)
```

    {'while', 'ours', 'out', 'by', 'be', "isn't", 'him', 'after', 'didn', 'mightn', 'theirs', 'y', 'wouldn', 'now', "doesn't", 'won', 'will', 'through', 'the', 'have', 'each', 'than', 'over', 'on', 'if', 'needn', 'own', 'at', 'why', 'to', "hadn't", 'does', 'and', 'weren', 'these', "weren't", 'no', 'once', 're', 'has', 'whom', 'not', 'ain', 'd', 'wasn', 'for', 'ourselves', 'but', 'ma', 'am', 'very', "you're", 'our', 'there', "mustn't", "shouldn't", 'other', 'further', 'or', 'in', 'before', 'any', 'couldn', 'are', "you'd", "she's", 'hadn', 'those', 'itself', 'who', 'same', 'what', 'then', "haven't", 'can', "don't", "should've", 'aren', 'both', 'mustn', 'herself', 'up', 'here', 'how', 'as', "shan't", 'was', 'more', 'a', "wasn't", "you've", 'few', 'having', 't', 'he', 'until', 'it', 'm', 'haven', 'been', 'again', 'o', 'yourselves', "needn't", "aren't", 'its', 'himself', 'too', "didn't", "wouldn't", 'my', 'should', 'during', 'shouldn', 'from', 'his', 'were', 'against', 'most', 'down', 'we', 'being', 'nor', 'where', 'isn', 'about', 'below', 'with', 'some', 'doesn', 'shan', "won't", 'off', 'them', 'i', "you'll", 'an', 'hers', 'did', 'myself', 'into', 'll', 'is', 'only', "it's", 'yours', 'of', "couldn't", 'because', 'so', 'hasn', 'do', 'above', 'all', 'you', 'doing', 'such', 'they', 'your', 've', 'had', 'under', 'just', 'her', "that'll", 'which', 'don', 'their', 'themselves', "mightn't", 'me', 'yourself', 'when', 'this', 'she', 'that', 's', "hasn't", 'between'}
    


```python
#let us remove word not from stop words, since it is the one of the most important word in classifing the review.
stop_words.remove("not")
```


```python
def text_Preprocessing(reviews):
#This will clean the text data, remove html tags, remove special characters and then tokenize the reviews to apply Stemmer on each word token"
        pre_processed_reviews=[]
        for review in tqdm(reviews):
            review= BeautifulSoup(review,'lxml').getText()    #remove html tags
            review=re.sub('\\S*\\d\\S*','',review).strip()
            review=re.sub('[^A-Za-z]+',' ',review)        #remove special chars\n",
            review=re.sub("n't","not",review)
            review=word_tokenize(str(review.lower())) #tokenize the reviews into word tokens
            # now we will split the review into words and then check if these words are in the stop words if so we will remove them, if not we will join
            review=' '.join(PorterStemmer().stem(word) for word in review if word not in stop_words)
            pre_processed_reviews.append(review.strip())
        return pre_processed_reviews
```


```python
preprocessed_reviews=text_Preprocessing(reviews_df["Text"])
preprocessed_reviews[34]
```

    100%|██████████| 37452/37452 [00:55<00:00, 679.30it/s]
    




    'mccann instant irish oatmeal varieti pack regular appl cinnamon mapl brown sugar box pack fan mccann steel cut oat thought give instant varieti tri found hardi meal not sweet great folk like post bariatr surgeri need food palat easili digest fiber make bloat'




```python
preprocessed_reviews=pd.DataFrame({"text":preprocessed_reviews,"sentiment":reviews_df.Score})
preprocessed_reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bought sever vital can dog food product found ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>product arriv label jumbo salt peanut peanut a...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>confect around centuri light pillowi citru gel...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>look secret ingredi robitussin believ found go...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>great taffi great price wide assort yummi taff...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
preprocessed_reviews.sentiment.value_counts()
```




    1    28875
    0     5597
    2     2980
    Name: sentiment, dtype: int64



# Observations:
we cleaned our text data, removed unnecessary tags
Though we cleaned our data, it is still in string format which computers won't understand, for this we use text featuration

# 3.1.Featurization


```python
#It is best practice to split the data Before we do text featurization 
reviews_train,reviews_test,sentiment_train,sentiment_test=train_test_split(preprocessed_reviews.text,preprocessed_reviews.sentiment)
print(reviews_train.shape,reviews_test.shape)
print(sentiment_train.shape,sentiment_test.shape)
```

    (28089,) (9363,)
    (28089,) (9363,)
    


```python
tfidf_model=TfidfVectorizer(ngram_range=(1,2),min_df=10, max_features=6000)
tfidf_model.fit(reviews_train,sentiment_train)
reviews_train_tfidf=tfidf_model.transform(reviews_train)
reviews_test_tfidf=tfidf_model.transform(reviews_test)
reviews_train_tfidf.shape,reviews_test_tfidf.shape
```




    ((28089, 6000), (9363, 6000))




```python
tfidf_df=pd.DataFrame(reviews_train_tfidf.toarray(),columns=tfidf_model.get_feature_names(),index=reviews_train.index)
tfidf_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abil</th>
      <th>abl</th>
      <th>abl buy</th>
      <th>abl find</th>
      <th>abl get</th>
      <th>abl make</th>
      <th>absolut</th>
      <th>absolut best</th>
      <th>absolut delici</th>
      <th>absolut favorit</th>
      <th>absolut love</th>
      <th>absorb</th>
      <th>acai</th>
      <th>accept</th>
      <th>access</th>
      <th>accid</th>
      <th>accident</th>
      <th>accompani</th>
      <th>accord</th>
      <th>account</th>
      <th>accur</th>
      <th>accustom</th>
      <th>acerola</th>
      <th>ach</th>
      <th>achiev</th>
      <th>acid</th>
      <th>acid coffe</th>
      <th>acid reflux</th>
      <th>acid tast</th>
      <th>acquir</th>
      <th>acquir tast</th>
      <th>across</th>
      <th>act</th>
      <th>action</th>
      <th>activ</th>
      <th>actual</th>
      <th>actual eat</th>
      <th>actual good</th>
      <th>actual like</th>
      <th>actual tast</th>
      <th>...</th>
      <th>yeah</th>
      <th>year</th>
      <th>year ago</th>
      <th>year love</th>
      <th>year not</th>
      <th>year old</th>
      <th>year round</th>
      <th>year sinc</th>
      <th>year tri</th>
      <th>year use</th>
      <th>yeast</th>
      <th>yellow</th>
      <th>yesterday</th>
      <th>yet</th>
      <th>yet not</th>
      <th>yield</th>
      <th>yogi</th>
      <th>yogi tea</th>
      <th>yogurt</th>
      <th>york</th>
      <th>yorki</th>
      <th>young</th>
      <th>younger</th>
      <th>yr</th>
      <th>yr old</th>
      <th>yuban</th>
      <th>yuck</th>
      <th>yum</th>
      <th>yum yum</th>
      <th>yummi</th>
      <th>zealand</th>
      <th>zero</th>
      <th>zero calori</th>
      <th>zing</th>
      <th>zip</th>
      <th>zip lock</th>
      <th>zipfizz</th>
      <th>ziploc</th>
      <th>ziwipeak</th>
      <th>zuke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36441</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32263</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18223</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21672</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27504</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.095156</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2655</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6810</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.096673</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.063068</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>33076</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13396</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.057754</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>28089 rows × 6000 columns</p>
</div>




```python
# Lets checkout the top features\n",
top_features=sorted(zip(tfidf_model.idf_,tfidf_model.get_feature_names()))
top10=top_features[:10]
```


```python
from wordcloud import WordCloud
plt.figure(figsize=(10,8))
wc = WordCloud(background_color="black",max_font_size=150, random_state=42)
wc.generate(str(top10))
plt.imshow(wc, interpolation='bilinear')
plt.suptitle('Top 10 words', size=30, y=0.88,color="r"); 
plt.axis("off")
plt.savefig("top10_words.png")
plt.show()
```


![png](Model_Code_files/Model_Code_27_0.png)


# 4. Modeling

# 4.1. Logistic regression


```python
# Logistic Regression with default parameters\n",
lr=LogisticRegression(max_iter=1000)
lr.fit(reviews_train_tfidf,sentiment_train)
lr_predict=lr.predict(reviews_test_tfidf)
plain_lr_f1=f1_score(sentiment_test,lr_predict,average="weighted")
plain_lr_f1
```




    0.8210884026537926




```python
plot_confusion_matrix(sentiment_test,lr_predict,normalize=True)
plt.title("Linear Regression with defult params")
plt.show()
```


![png](Model_Code_files/Model_Code_31_0.png)



```python
# we will tune the parameters of Logistic Regression with RandomizedsearchCV\n",
lr_params={"penalty":["l1","l2"],"C":[10**i for i in range(-4,4)]}
lr=LogisticRegression( max_iter=1000,solver="liblinear")
lr_rnm_clf=RandomizedSearchCV(lr,lr_params)
lr_rnm_clf.fit(oversampled_trainX,oversampled_trainY)
lr_rnm_clf.best_params_
```




    {'C': 100, 'penalty': 'l1'}




```python
lr_bal=LogisticRegression(**lr_rnm_clf.best_params_, max_iter=1000,solver="liblinear")
lr_bal.fit(oversampled_trainX,oversampled_trainY)
lr_bal_predict=lr_bal.predict(reviews_test_tfidf)
lr_bal_f1=f1_score(lr_bal_predict,sentiment_test,average="weighted")
lr_bal_f1
```




    0.7788067897580379




```python
plot_confusion_matrix(sentiment_test,lr_bal_predict,normalize=True)
plt.title("Logistic regression Confusion matrix",size=15)
```




    Text(0.5, 1.0, 'Logistic regression Confusion matrix')




![png](Model_Code_files/Model_Code_34_1.png)


# 4.2. Decision tree


```python
from sklearn.tree import DecisionTreeClassifier
dt_param={'max_depth':[i for i in range(5,2000,3)],'min_samples_split':[i for i in range(5,2000,3)]}
dt_clf=DecisionTreeClassifier()
rndm_clf=RandomizedSearchCV(dt_clf,dt_param)
rndm_clf.fit(oversampled_trainX,oversampled_trainY)
dt_best_params=rndm_clf.best_params_
```


```python
dt_clf=DecisionTreeClassifier(**dt_best_params)
dt_clf.fit(oversampled_trainX,oversampled_trainY)
dt_predict=dt_clf.predict(reviews_test_tfidf)
dt_f1=f1_score(sentiment_test,dt_predict,average="weighted")
dt_f1
```




    0.7014337953920822




```python
plot_confusion_matrix(sentiment_test,dt_predict,normalize=True)
plt.title("Decision Tree Confuison matrix",size=18)
```




    Text(0.5, 1.0, 'Decision Tree Confuison matrix')




![png](Model_Code_files/Model_Code_38_1.png)


# 4.3. Naive Bayes


```python
nb_params={"alpha":[10**i for i in range(-5,5)]}
nb_clf=MultinomialNB()
rndm_clf=RandomizedSearchCV(nb_clf,nb_params)
rndm_clf.fit(oversampled_trainX,oversampled_trainY)
rndm_clf.fit(oversampled_trainX,oversampled_trainY)
nb_best_params=rndm_clf.best_params_
nb_clf=MultinomialNB(**nb_best_params)
nb_clf.fit(oversampled_trainX,oversampled_trainY)
nb_predict=nb_clf.predict(reviews_test_tfidf)
nb_f1=f1_score(sentiment_test,nb_predict,average="weighted")
nb_f1
```




    0.7830211478634019




```python
plot_confusion_matrix(sentiment_test,nb_predict,normalize=True,cmap="Reds")
plt.title("Naive Bayes Confusion Matrix",size=15)
```




    Text(0.5, 1.0, 'Naive Bayes Confusion Matrix')




![png](Model_Code_files/Model_Code_41_1.png)


# 5. Model evaluation


```python
models=["LogesticRegression","DecisionTrees","NaiveBayes"]
f1_scores=[lr_bal_f1,dt_f1,nb_f1]
```


```python
plt.figure(figsize=(6,5))
plt.barh(models,f1_scores,color=['c','r','m'])
plt.title("F1 Scores of all models",size=20)
for index, value in enumerate(f1_scores):
    plt.text(0.9,index,str(round(value,2)))
plt.xlabel('F1_SCores',size=15)
plt.ylabel("Models",size=15)
plt.savefig("f1_scores.png")
plt.show()
```


![png](Model_Code_files/Model_Code_44_0.png)


# Obseravtions:
After cross checking the confusion matrices of above models, Naive Bayes is slightly better than rest of the models.
We will select Naive Bayes for our problem, lets Pickle the model for later use


```python
# lets save the model
import pickle
pickle.dump(nb_clf,open("nb_clf.pkl","wb"))
pickle.dump(tfidf_model,open("tfidf_model.pkl","wb"))
```
