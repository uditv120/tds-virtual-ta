import requests, json
from bs4 import BeautifulSoup
def fetch_discourse(base_url, pages=5, out="discourse_posts.json"):
    posts = []
    for p in range(1, pages+1):
        r = requests.get(f"{base_url}/latest?page={p}")
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a.title.raw-link"):
            url = base_url + a["href"]
            t = a.text.strip()
            pr = requests.get(url)
            ps = BeautifulSoup(pr.text, "html.parser")
            c = "\n".join(p.text for p in ps.select(".cooked p"))
            posts.append({"url": url, "title": t, "content": c})
    with open(out, "w") as f:
        json.dump(posts, f, indent=2)

if __name__ == "__main__":
    fetch_discourse("https://discourse.onlinedegree.iitm.ac.in/c/tools-in-data-science", pages=10)
