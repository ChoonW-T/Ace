import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the Amazon reviews data
df = pd.read_excel('Amazon_Reviews.xlsx')

# Uncomment the following to visualize the distribution of review scores
# plt.style.use('ggplot')
# ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
# ax.set_xlabel('Review Stars')
# plt.show()

# Uncomment to view tokenized version of a sample review using nltk
# example = df['Text'][50]
# tokens = nltk.word_tokenize(example)
# tagged = nltk.pos_tag(tokens)
# entities = nltk.chunk.ne_chunk(tagged)
# entities.pprint()

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Compute VADER sentiment scores for each review
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

# Convert VADER results to DataFrame and merge with original data
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

# Plot VADER compound sentiment scores by Amazon review scores
sns.barplot(data=vaders, x='Score', y='compound').set_title('Compound Score by Amazon Star Review')
plt.show()

# Plot positive, neutral, and negative sentiment scores by Amazon review scores
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

# Initialize RoBERTa tokenizer and model for sentiment analysis
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# Function to get sentiment scores using RoBERTa
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }


# Compute both VADER and RoBERTa sentiment scores for each review
res = {}
for i, row in df.iterrows():
    try:
        text = row['Text']
        myid = row['Id']

        # VADER results
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}

        # RoBERTa results
        roberta_result = polarity_scores_roberta(text)

        # Combine both results
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

# Convert results to DataFrame and merge with original data
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# Plot pairwise relationships between sentiment scores, colored by Amazon review scores
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='tab10')
plt.show()