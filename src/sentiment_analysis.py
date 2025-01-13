import logging
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class Batch:
    def __init__(self, data, batch_size=1000):
        self.current = -1
        self.data = data
        self.nchunks = int(len(data.index) / batch_size)
        self.parts = [(i * batch_size, (i + 1) * batch_size) for i in range(self.nchunks)]

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < self.nchunks:
            start, end = self.parts[self.current]
            return self.current, self.data.iloc[start:end]
        raise StopIteration

    def __len__(self):
        return self.nchunks


class ABSAProcessor:
    def __init__(self, df, path, name):
        tqdm.pandas()
        self.df: pd.DataFrame = df
        self.path = path
        self.name = name

    def aspect_extraction(self, ):
        docs = self.df["clean_tweet"].unique()
        stopwords = open(
            f"{self.path}/raw/stopwords.txt").read().splitlines()

        vectorizer = TfidfVectorizer(max_features=50, stop_words=stopwords, ngram_range=(1,2))
        result = vectorizer.fit_transform(docs)
        tfidf_matrix = pd.DataFrame(result.T.toarray(), index=vectorizer.get_feature_names_out())
        return tfidf_matrix.index.to_list()

    def _compute_sentiment(self, row, model, aspects):
        completed_task = model(row["tweet"], aspects=aspects)
        t = dict.fromkeys(aspects)

        for example in completed_task.examples:
            t[example.aspect] = example.scores

        return t

    def _compute_intensity_with_delta(self, v):
        v = eval(v)
        delta_1 = v[0] - v[1]
        delta_2 = v[0] - v[2]
        delta_3 = v[1] - v[2]
        if v[1] > delta_3 and v[1] > -delta_1:
            return -v[1]
        if v[2] > - delta_2 and v[2] > -delta_3:
            return v[2]
        else:
            return 0
        pass
    def _compute_intensity_max(self, v):
        v = eval(v)
        m = max(v)
        if m == v[0]:
            return 0
        if m == v[1]:
            return -v[1]
        if m == v[2]:
            return v[2]

    def _compute_intensity(self,v):
        ## neu/neg/pos
        v = eval(v)
        neu = v[0]
        neg = v[1]
        pos = v[2]
        if neg > abs(neg - neu) and neg > abs(neg - pos):
            return -neg
        if pos > abs(pos - neu) and pos > abs(pos - neg):
            return pos
        else:
            return 0
        pass
    def sentiment_intensity(self):
        df = self.df.drop(["tweet"], axis=1)
        for col in df.columns:
            print(col)
            df[col] = df[col].apply(self._compute_intensity)
        df = df.groupby("username").mean(numeric_only=True)

        print("Number of users:", df.shape)

        self.df = df
        # self.df.to_pickle(f"{self.path}/transformed/sentiment_intensity.pkl")
        self.df.to_csv(f"{self.path}/processed/sentiment_intensity_{self.name}.csv")

    def compute_sentiment(self, topics, cached=True):
        #import lib.aspect_based_sentiment_analysis as absa
        #model = absa.load()
        model = None

        batch_data = Batch(self.df)
        for i, part in batch_data:

            part_filename = os.path.join(self.path, f"transformed/sentiment_results_{self.name}_part{i}.csv")
            print(part_filename)
            if os.path.isfile(part_filename) and cached:
                continue

            part["result"] = part.progress_apply(self._compute_sentiment, model=model, aspects=topics, axis=1)
            part.to_csv(part_filename)

        self._merge_parts(len(batch_data))

    def _merge_parts(self, nchunks):
        df = pd.DataFrame()
        for i in range(0, nchunks):
            part = pd.read_csv(
                os.path.join(self.path, f"transformed/sentiment_results_{self.name}_part{i}.csv"),
                lineterminator='\n', usecols=["username", "tweet", "result"])
            df = pd.concat([df, part], ignore_index=True)
        df = pd.concat([df.drop(['result'], axis=1), pd.json_normalize(df['result'].apply(eval))], axis=1)

        self.df = df
        self.df.to_csv(f"{self.path}/transformed/sentiment_results_{self.name}.csv", index=False)

        print("Created dataset with " + str(len(df.index)) + " users")

    def sentiment_analysis(self, topics):
        self.compute_sentiment(topics)

        self.df = pd.read_csv(f"{self.path}/transformed/sentiment_results_{self.name}.csv",
                              index_col="username",
                              lineterminator='\n')
        self.sentiment_intensity()
