from src.core.config import settings


import requests
from bs4 import BeautifulSoup
import json
import time

def fetch_arxiv_ai_papers(categories, max_results=1000, save_path="data/arxiv_ai.jsonl"):
    base_url = "http://export.arxiv.org/api/query?"
    all_papers = []

    for cat in categories:
        print(f"Fetching category {cat}...")
        for start in range(0, max_results, 100):
            query = f"search_query=cat:{cat}&start={start}&max_results=100&sortBy=submittedDate&sortOrder=descending"
            response = requests.get(base_url + query)
            soup = BeautifulSoup(response.text, "xml")

            for entry in soup.find_all("entry"):
                paper = {
                    "id": entry.id.text,
                    "title": entry.title.text.strip().replace("\n", " "),
                    "abstract": entry.summary.text.strip().replace("\n", " "),
                    "authors": [a.text for a in entry.find_all("author")],
                    "category": cat,
                    "published": entry.published.text,
                    "url": entry.id.text,
                }
                paper["text"] = f"{paper['title']}. {paper['abstract']}"
                all_papers.append(paper)
            time.sleep(1)

    with open(save_path, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p) + "\n")

    print(f"Saved {len(all_papers)} papers to {save_path}")
    return all_papers

categories = ["cs.AI", "cs.CL", "cs.LG", "cs.CV", "cs.NE", "cs.IR"]
fetch_arxiv_ai_papers(categories, max_results=500, save_path=f"{settings.DATASET_PATH}/arxiv_ai.jsonl")

