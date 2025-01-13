import requests
from tqdm import tqdm
import os
import json
import re
import pandas as pd
from bs4 import BeautifulSoup


def scrape(path):
    with open(os.path.join(path, "raw/original_entries.csv")) as f:
        ids = f.readlines()
    flag = True
    URL = "https://publish.twitter.com/oembed?dnt=true&omit_script=true&url=https://mobile.twitter.com/i/status/{}"
    df = []
    try:
        for idn in tqdm(ids[1:]):
            if flag:
                idn = idn.removeprefix("id_")

            res = requests.get(URL.format(idn))

            if res.status_code == 200:
                data = res.json()
                df.append(data)

    finally:
        import json
        with open(os.path.join(path, "transformed/original_entries.json"), "w") as f:
            json.dump(df, f, indent=4)


def extract_mentions(text):
    pattern = re.compile(r'@\w*')
    result = re.findall(pattern, text)
    return result


def extract_retweet(text):
    pattern = re.compile(r'&mdash;.+\(@(.+)\)')
    result = re.findall(pattern, text)
    return result


def transform(base_path):
    cols = ["id", "conversation_id", "created_at", "date", "time",
            "timezone", "user_id", "username", "name", "place", "tweet",
            "mentions", "urls", "photos", "replies_count", "retweets_count", "likes_count",
            "hashtags", "cashtags", "link", "retweet", "quote_url", "video", "near",
            "geo", "source", "user_rt_id", "user_rt", "retweet_id", "reply_to", "retweet_date"
            ]

    path = "transformed/direct_engagement_data.json"
    f = open(os.path.join(base_path, path))
    records = json.load(f)
    f.close()

    df = []

    for data in tqdm(records):
        row = {}

        # extract tweet id
        row["id"] = data['url'].split('/')[-1]

        # extract username
        row["username"] = data['author_url'].split('/')[-1]

        # extract tweet
        parsed = BeautifulSoup(data["html"])
        row["tweet"] = parsed.find('p').text

        # extract mentions
        row["mentions"] = extract_mentions(data["html"])

        # extract user rt
        row["user_rt"] = extract_retweet(data["html"])

        df.append(row)

    df = pd.DataFrame(df, columns=cols)
    df.to_csv(os.path.join(base_path, "transformed/direct_engagement.csv"))


if __name__ == '__main__':
    scrape("/home/miriam/PycharmProjects/echoChamberProject/data/rot")
