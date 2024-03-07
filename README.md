# nlpcode
#Text Classification Using Fasttext

import pandas as pd

df= pd.read_csv("ecommerce_dataset.csv", names=["category", "description"], header=None)
print(df.shape)
df.head(3)
df.dropna(inplace=True)
df.shape
df.category.unique()
df.category.replace("Clothing & Accessories", "Clothing_Accessories", inplace=True)
df['category'] = '__label__' + df['category'].astype(str)
df.head(5)
df['category_description'] = df['category'] + ' ' + df['description']
df.head(3)


---> Pre-procesing

#Remove punctuation
#Remove extra space
#Make the entire sentence lower case
import re

text = "  VIKI's | Bookcase/Bookshelf (3-Shelf/Shelve, White) | ? . hi"
text = re.sub(r'[^\w\s\']',' ', text)
text = re.sub(' +', ' ', text)
text.strip().lower()
def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower() 
    df['category_description'] = df['category_description'].map(preprocess)
df.head()


---> Train Test Split

    from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
train.shape, test.shape
train.to_csv("ecommerce.train", columns=["category_description"], index=False, header=False)
test.to_csv("ecommerce.test", columns=["category_description"], index=False, header=False)


----> Train the model and evaluate performance


import fasttext
model = fasttext.train_supervised(input="ecommerce.train")
model.test("ecommerce.test")


Now let's do prediction for few product descriptions

model.predict("wintech assemble desktop pc cpu 500 gb sata hdd 4 gb ram intel c2d processor 3")
model.predict("ockey men's cotton t shirt fabric details 80 cotton 20 polyester super combed cotton rich fabric")
model.predict("think and grow rich deluxe edition")
model.get_nearest_neighbors("painting")
model.get_nearest_neighbors("sony")
model.get_nearest_neighbors("banglore")
