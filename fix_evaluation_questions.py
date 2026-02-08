"""
Fix evaluation questions to include proper source URLs for MRR calculation
"""

import json
from pathlib import Path

def add_source_urls_to_questions():
    """Add source URLs to evaluation questions based on article titles"""
    
    # Load evaluation questions
    with open('data/processed/evaluation_questions.json', 'r') as f:
        eval_data = json.load(f)
    
    # Load chunks to create article title -> URL mapping
    with open('data/processed/chunks.json', 'r') as f:
        chunks = json.load(f)
    
    # Create mapping from article title to URL
    title_to_url = {}
    for chunk in chunks:
        title = chunk.get('article_title', '')
        url = chunk.get('article_url', '')
        if title and url:
            title_to_url[title] = url
    
    print(f"Created mapping for {len(title_to_url)} articles")
    
    # Update questions with source URLs
    updated_questions = []
    questions_with_urls = 0
    
    for question in eval_data['questions']:
        # Get article title from question
        article_title = question.get('article_title') or question.get('source_article')
        
        if article_title and article_title in title_to_url:
            question['source_url'] = title_to_url[article_title]
            questions_with_urls += 1
        else:
            # Try to find a close match
            for title, url in title_to_url.items():
                if article_title and article_title.lower() in title.lower():
                    question['source_url'] = url
                    questions_with_urls += 1
                    break
            
            if 'source_url' not in question:
                print(f"No URL found for article: {article_title}")
        
        updated_questions.append(question)
    
    # Update the evaluation data
    eval_data['questions'] = updated_questions
    eval_data['metadata']['questions_with_source_urls'] = questions_with_urls
    
    # Save updated questions
    with open('data/processed/evaluation_questions.json', 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"Updated {questions_with_urls}/{len(updated_questions)} questions with source URLs")
    return questions_with_urls

if __name__ == "__main__":
    add_source_urls_to_questions()