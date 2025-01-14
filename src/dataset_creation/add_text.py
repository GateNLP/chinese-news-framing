import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

TITLE_BREAK = "\n------\n\n"


def get_yes_no():
    i = input("yes (y) or no (n): ")
    while i != "y" and i != "n":
        i = input("answer not recognised, input 'y' or 'n'.")
    return i


def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = (
            soup.find("h1").get_text(strip=True)
            if soup.find("h1")
            else "No Title Found"
        )
        content = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        return title, content

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return "Error", "Error"


def text_from_url(url):
    title, content = scrape_url(url)
    text = title.strip() + TITLE_BREAK + content.strip()[:500]
    return text


if __name__ == "__main__":
    df = pd.read_csv("data/raw/chinese_news_framing_dataset.csv")

    # scrape
    tqdm.pandas(desc="Adding text from each web link in the dataset.")
    df["text"] = df["web_link"].progress_apply(text_from_url)
    print(df["text"])

    count_no_title = (
        df["text"].str.contains("No Title Found" + TITLE_BREAK, na=False).sum()
    )
    count_errors = df["text"].str.contains("Error" + TITLE_BREAK, na=False).sum()

    print("Number of samples with no title (to clean):")
    print(count_no_title)

    print("Number of samples with an error:")
    print(count_errors)

    print("Remove samples with an error?")
    ans = get_yes_no()

    if ans == "y":
        df = df[~df["text"].str.contains("Error" + TITLE_BREAK, na=False)]
        print(f"New length of df: {len(df)}")

    # save dataframe
    df.to_csv("data/chinese_news_framing_with_text.csv", index=False)
