import requests
from bs4 import BeautifulSoup
import os

os.makedirs("docs_text", exist_ok=True)

urls = {
    "tesla_q1_2024.txt": "https://www.sec.gov/Archives/edgar/data/1318605/000162828024017503/tsla-20240331.htm",
    "tesla_q2_2024.txt": "https://www.sec.gov/Archives/edgar/data/1318605/000162828024032662/tsla-20240630.htm",
    "tesla_q3_2024.txt": "https://www.sec.gov/Archives/edgar/data/1318605/000162828024043486/tsla-20240930.htm",
    "apple_q1_2024.txt": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/aapl-20231230.htm",
    "apple_q2_2024.txt": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240330.htm",
    "apple_q3_2024.txt": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240629.htm",
}

headers = {"User-Agent": "venugopal student@email.com"}

for filename, url in urls.items():
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        with open(f"docs_text/{filename}", "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"  Saved — {len(clean_text)} characters")

    except Exception as e:
        print(f"  Failed: {e}")

print("\nAll files downloaded!")
print("Now run: python ingest_text.py")