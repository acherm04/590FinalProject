The following is my code because the colab notebook was unable to work:

import pandas as pd
from google.colab import files
import random

uploaded = files.upload()

df = pd.read_csv ("UofL_housing - Sheet1.csv")

df['minimum cost per semester'] = df['minimum cost per semester'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['maximum cost per semester'] = df['maximum cost per semester'].replace({'\$': '', ',': ''}, regex=True).astype(float)


df["text1"] = df['location'] + " is a " + df['style'] + " hall for " + df['student'] + " with a " + df['kitchen'] + " kitchen and a " + df['laundry'] + " laundry room."
df["text2"] = df['location'] + " is the cheapest option, priced at " + df['minimum cost per semester'].astype(str)
df["text3"] = df['location'] + " is the most expensive option, priced at " + df['maximum cost per semester'].astype(str)
df["text4"] = df['location'] + " is a housing option for " + df['student']

df["text"] = df.apply(lambda row: random.choice([row["text1"], row["text2"], row["text3"], row["text4"]]), axis=1)

texts = df["text"].tolist()

!pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

d = emb.shape[1]
index = faiss.IndexFlatIP(d)
index.add(emb.astype("float32"))

def search(query, k=5, required_style=None):
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    _, I = index.search(q_emb, k)

    results = df.iloc[I[0]]

    if required_style:
        results = results[results["style"].str.contains(required_style, case=False)]

    return results["text"].tolist()

def get_cheapest():
    return df.loc[df['minimum cost per semester'].idxmin()]
def get_most_expensive():
    return df.loc[df['maximum cost per semester'].idxmax()]

while True:
    q = input("Ask me about housing (or 'quit'): ").strip()
    if q.lower() in {"quit", "exit"}:
        break
    k = int(input("How many matches? "))

    # Check for location and room type query
    if ("type of rooms" in q.lower() or "types of rooms" in q.lower()):
        location_found = False
        for location in df['location']:
            if location.lower() in q.lower():
                location_row = df[df['location'].str.contains(location, case=False)].iloc[0]
                print(f"{location_row['location']} has the following types of rooms: {location_row['room type']}")
                location_found = True
                break
        if location_found:
            continue

    # Add logic for wifi, lease, and athletes
    if "wifi" in q.lower():
        print("All dorms have free wifi available for the students.")
        continue
    elif "lease" in q.lower():
        print("Both 9-month and 11.5-month leases are available for students. Most apartments are 11.5-months, while traditional and suite style dorms are 9-months.")
        continue
    elif "athletes" in q.lower():
        print("Bettie Johnson, University Pointe, and Denny Crum are apartments that house our UofL athletes. If you are an athlete, reach out to your team to see if there's special housing options for you.")
        continue
    elif "cheapest" in q.lower():
        cheapest_dorm = get_cheapest()
        print(f"{cheapest_dorm['location']} is the cheapest option, priced at {cheapest_dorm['minimum cost per semester']}")
        continue
    elif "expensive" in q.lower():
        most_expensive_dorm = get_most_expensive()
        print(f"{most_expensive_dorm['location']} is the most expensive option, priced at {most_expensive_dorm['maximum cost per semester']}")
        continue
    elif "freshmen" in q.lower():
        freshmen_dorms = df[df['student'].str.contains('freshmen', case=False)]
        if not freshmen_dorms.empty:
            print("Here are some housing options for freshmen:")
            for index, row in freshmen_dorms.iterrows():
                print(f"- {row['location']} is a {row['style']} hall for {row['student']}")
        else:
            print("I couldn't find any housing options specifically for freshmen.")
        continue
    elif "upperclassmen" in q.lower():
        upperclassmen_dorms = df[df['student'].str.contains('upperclassmen', case=False)]
        if not upperclassmen_dorms.empty:
            print("Here are some housing options for upperclassmen:")
            for index, row in upperclassmen_dorms.iterrows():
                print(f"- {row['location']} is a {row['style']} hall for {row['student']}")
        else:
            print("I couldn't find any housing options specifically for upperclassmen.")
        continue
    elif "laundry" in q.lower():
        if "in-unit" in q.lower():
            laundry_type = "in-unit"
        elif "shared facility" in q.lower():
            laundry_type = "shared facility"
        else:
            print("Please specify if you're looking for in-unit or shared facility laundry.")
            continue

        laundry_dorms = df[df['laundry'].str.contains(laundry_type, case=False)]
        if not laundry_dorms.empty:
            print(f"Here are some housing options with {laundry_type} laundry:")
            for index, row in laundry_dorms.iterrows():
                print(f"- {row['location']} has {row['laundry']} laundry.")
        else:
            print(f"I couldn't find any housing options with {laundry_type} laundry.")
        continue
    elif "kitchen" in q.lower():
        if "in-unit" in q.lower():
            kitchen_type = "in-unit"
        elif "shared facility" in q.lower():
            kitchen_type = "shared facility"
        else:
            print("Please specify if you're looking for in-unit or shared facility kitchen.")
            continue

        kitchen_dorms = df[df['kitchen'].str.contains(kitchen_type, case=False)]
        if not kitchen_dorms.empty:
            print(f"Here are some housing options with {kitchen_type} kitchen:")
            for index, row in kitchen_dorms.iterrows():
                print(f"- {row['location']} has a {row['kitchen']} kitchen.")
        else:
            print(f"I couldn't find any housing options with {kitchen_type} kitchen.")
        continue


    # Check if user wants only traditional halls
    required = None
    if "traditional" in q.lower():
        required = "traditional"
    elif "suite" in q.lower():
        required = "suite"
    elif "apartment" in q.lower():
        required = "apartment"

    matches = search(q, k=k + 5, required_style=required)  # pull extra for filtering
    print("\n—\n".join(matches[:k]))
