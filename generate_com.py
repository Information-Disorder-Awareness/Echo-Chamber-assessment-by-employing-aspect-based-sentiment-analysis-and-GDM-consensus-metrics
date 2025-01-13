import pandas as pd
from bertopic import BERTopic
from old.community_detection import community_detection


communities = community_detection("./data/rot", "rot", comType="weightComm")

with open("communities.txt", "w") as f:
    f.write(str(communities))
