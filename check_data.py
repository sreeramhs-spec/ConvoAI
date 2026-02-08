import json
from pathlib import Path

# Count fixed URLs
with open('data/fixed_urls.json', 'r') as f:
    fixed_urls = json.load(f)
print(f'Fixed URLs: {len(fixed_urls)}')

# Check if processed chunks exist and count unique articles
if Path('data/processed/chunks.json').exists():
    with open('data/processed/chunks.json', 'r') as f:
        chunks = json.load(f)
    if chunks:
        unique_articles = set(chunk['article_url'] for chunk in chunks)
        print(f'Processed articles: {len(unique_articles)}')
        print(f'Total chunks: {len(chunks)}')
        if len(unique_articles) >= 10:
            print('Sample articles:')
            for i, url in enumerate(list(unique_articles)[:10]):
                print(f'  {i+1}. {url}')
    else:
        print('No chunks found')
else:
    print('No processed chunks file found')