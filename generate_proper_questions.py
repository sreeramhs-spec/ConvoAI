"""
Generate proper evaluation questions from existing corpus
"""

import json
import random
import time
from pathlib import Path

def generate_evaluation_questions_from_corpus():
    """Generate 100 evaluation questions from existing corpus"""
    
    # Load existing chunks
    with open('data/processed/chunks.json', 'r') as f:
        chunks = json.load(f)
    
    # Group chunks by article
    articles = {}
    for chunk in chunks:
        url = chunk['article_url']
        title = chunk['article_title']
        if url not in articles:
            articles[url] = {
                'title': title,
                'url': url,
                'chunks': []
            }
        articles[url]['chunks'].append(chunk)
    
    print(f"Found {len(articles)} articles with chunks")
    
    # Generate questions
    questions = []
    question_templates = {
        'factual': [
            "What is {entity}?",
            "What are the main characteristics of {entity}?",
            "What is the definition of {entity}?",
            "What are the key features of {entity}?",
            "How is {entity} described?",
        ],
        'analytical': [
            "What are the advantages and disadvantages of {entity}?",
            "How does {entity} impact society?",
            "What are the implications of {entity}?",
            "How has {entity} evolved over time?",
            "What factors influence {entity}?",
        ],
        'contextual': [
            "What role does {entity} play in {context}?",
            "How does {entity} relate to {context}?",
            "What is the significance of {entity} in {context}?",
            "How does {entity} contribute to {context}?",
        ]
    }
    
    # Sample articles for questions
    sampled_articles = random.sample(list(articles.values()), min(100, len(articles)))
    
    current_id = 1
    for article in sampled_articles:
        if len(questions) >= 100:
            break
            
        # Choose question type
        question_types = ['factual'] * 40 + ['analytical'] * 40 + ['contextual'] * 20
        question_type = random.choice(question_types)
        
        # Choose difficulty
        difficulty_levels = ['easy'] * 40 + ['medium'] * 40 + ['hard'] * 20
        difficulty = random.choice(difficulty_levels)
        
        # Extract entity from article title
        title = article['title']
        entity = title
        
        # Choose template
        template = random.choice(question_templates[question_type])
        
        # Generate question
        if question_type == 'contextual':
            # For contextual questions, use a related domain
            contexts = ['modern technology', 'society', 'history', 'science', 'economics']
            context = random.choice(contexts)
            question = template.format(entity=entity, context=context)
        else:
            question = template.format(entity=entity)
        
        # Create ground truth answer from first chunk
        if article['chunks']:
            first_chunk = article['chunks'][0]
            ground_truth = first_chunk['content'][:500] + "..."
            source_chunk_id = first_chunk['chunk_id']
        else:
            ground_truth = f"Information about {entity}"
            source_chunk_id = "unknown"
        
        question_data = {
            'question': question,
            'type': question_type,
            'difficulty': difficulty,
            'entity': entity,
            'entity_type': 'CONCEPT',
            'source_chunk': source_chunk_id,
            'article_title': title,
            'source_url': article['url'],
            'ground_truth_answer': ground_truth,
            'question_id': f'q_{current_id:03d}',
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'source_article': title,
            'chunk_word_count': len(ground_truth.split()) if ground_truth else 0
        }
        
        questions.append(question_data)
        current_id += 1
    
    # Create metadata
    question_type_counts = {}
    difficulty_counts = {}
    
    for q in questions:
        qtype = q['type']
        difficulty = q['difficulty']
        question_type_counts[qtype] = question_type_counts.get(qtype, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    evaluation_data = {
        'metadata': {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'total_questions': len(questions),
            'question_types': question_type_counts,
            'difficulty_levels': difficulty_counts,
            'questions_with_source_urls': len([q for q in questions if q.get('source_url')])
        },
        'questions': questions
    }
    
    # Save questions
    with open('data/processed/evaluation_questions.json', 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"Generated {len(questions)} evaluation questions")
    print(f"Question types: {question_type_counts}")
    print(f"Difficulty levels: {difficulty_counts}")
    return questions

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    generate_evaluation_questions_from_corpus()