#!/usr/bin/env python3
"""
Intelligent Question Generation Script for RAG System Evaluation
Generates diverse, high-quality questions from Wikipedia articles using multiple strategies.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import spacy
from datetime import datetime

class AdvancedQuestionGenerator:
    """Advanced question generator using multiple NLP techniques."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.nlp = None
        self.load_nlp_model()
        
        # Question templates for different types
        self.question_templates = {
            'factual': [
                "What is {entity}?",
                "Who is {person}?",
                "Where is {location} located?",
                "When was {event} established?",
                "How does {concept} work?",
                "What are the main features of {topic}?",
                "What is the significance of {entity}?",
                "What role does {entity} play in {context}?"
            ],
            'analytical': [
                "How does {entity1} relate to {entity2}?",
                "What are the advantages and disadvantages of {concept}?",
                "What impact did {event} have on {domain}?",
                "How has {concept} evolved over time?",
                "What factors contribute to {phenomenon}?",
                "Why is {entity} important for {field}?",
                "What challenges does {entity} face?",
                "How does {entity} compare to similar {category}?"
            ],
            'contextual': [
                "In what context is {entity} most relevant?",
                "What are the practical applications of {concept}?",
                "How is {entity} used in modern {field}?",
                "What makes {entity} unique among {category}?",
                "What are the key components of {system}?",
                "How does {process} benefit {stakeholders}?",
                "What are the main characteristics of {entity}?",
                "In which scenarios would {entity} be most useful?"
            ]
        }
        
        # Question difficulty levels
        self.difficulty_patterns = {
            'easy': ['What is', 'Who is', 'Where is', 'When was'],
            'medium': ['How does', 'What are', 'Why is', 'Which'],
            'hard': ['Analyze', 'Compare', 'Evaluate', 'What factors', 'How has evolved']
        }
    
    def load_nlp_model(self):
        """Load spaCy model with fallback options."""
        models_to_try = ['en_core_web_sm', 'en_core_web_md', 'en']
        
        for model_name in models_to_try:
            try:
                self.nlp = spacy.load(model_name)
                print(f"✓ Loaded spaCy model: {model_name}")
                return
            except IOError:
                continue
        
        print("⚠ No spaCy model found. Installing en_core_web_sm...")
        try:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load('en_core_web_sm')
            print("✓ Successfully installed and loaded en_core_web_sm")
        except Exception as e:
            print(f"⚠ Could not load spaCy model: {e}")
            print("Using basic text processing without NER")
            self.nlp = None
    
    def extract_entities_and_concepts(self, text: str, title: str = "") -> Dict[str, List[str]]:
        """Extract entities and concepts from text."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities
            'EVENT': [],
            'CONCEPT': [],
            'DATE': [],
            'MISC': []
        }
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'EVENT']:
                    entities[ent.label_].append(ent.text)
                else:
                    entities['MISC'].append(ent.text)
        
        # Extract concepts using simple patterns
        concepts = self.extract_concepts_from_text(text, title)
        entities['CONCEPT'].extend(concepts)
        
        # Clean and deduplicate
        for key in entities:
            entities[key] = list(set([e.strip() for e in entities[key] if len(e.strip()) > 2]))
        
        return entities
    
    def extract_concepts_from_text(self, text: str, title: str = "") -> List[str]:
        """Extract key concepts using pattern matching."""
        concepts = []
        
        # Add title as primary concept
        if title:
            concepts.append(title)
        
        # Extract noun phrases and important terms
        if self.nlp:
            doc = self.nlp(text)
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3 and len(chunk.text) > 3:
                    concepts.append(chunk.text)
        
        # Pattern-based extraction
        patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper nouns
            r'\b[A-Z][a-z]{2,}\b',           # Capitalized words
            r'\bthe [a-z]+ of [A-Z][a-z]+\b', # "the X of Y" patterns
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def generate_questions_from_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple questions from a single chunk."""
        questions = []
        
        text = chunk.get('content', '') or chunk.get('chunk_text', '')
        title = chunk.get('article_title', 'Unknown')
        
        if len(text.strip()) < 50:  # Skip very short chunks
            return questions
        
        # Extract entities and concepts
        entities = self.extract_entities_and_concepts(text, title)
        
        # Generate different types of questions
        questions.extend(self.generate_factual_questions(entities, text, chunk))
        questions.extend(self.generate_analytical_questions(entities, text, chunk))
        questions.extend(self.generate_contextual_questions(entities, text, chunk))
        
        return questions
    
    def generate_factual_questions(self, entities: Dict, text: str, chunk: Dict) -> List[Dict]:
        """Generate factual questions."""
        questions = []
        templates = self.question_templates['factual']
        
        # Generate questions for each entity type
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
                
            for entity in entity_list[:2]:  # Limit to avoid too many questions
                if entity_type == 'PERSON':
                    template = random.choice([t for t in templates if '{person}' in t])
                    question = template.replace('{person}', entity)
                elif entity_type in ['ORG', 'GPE']:
                    template = random.choice([t for t in templates if '{entity}' in t])
                    question = template.replace('{entity}', entity)
                else:
                    template = random.choice([t for t in templates if '{entity}' in t or '{topic}' in t])
                    question = template.replace('{entity}', entity).replace('{topic}', entity)
                
                questions.append({
                    'question': question,
                    'type': 'factual',
                    'difficulty': 'easy',
                    'entity': entity,
                    'entity_type': entity_type,
                    'source_chunk': chunk.get('chunk_id', ''),
                    'article_title': chunk.get('article_title', ''),
                    'expected_content': text[:200] + "..."
                })
        
        return questions
    
    def generate_analytical_questions(self, entities: Dict, text: str, chunk: Dict) -> List[Dict]:
        """Generate analytical questions."""
        questions = []
        templates = self.question_templates['analytical']
        
        # Generate comparative and analytical questions
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        
        if len(all_entities) >= 2:
            entity1, entity2 = random.sample(all_entities, 2)
            template = random.choice([t for t in templates if '{entity1}' in t])
            question = template.replace('{entity1}', entity1).replace('{entity2}', entity2)
            
            questions.append({
                'question': question,
                'type': 'analytical',
                'difficulty': 'medium',
                'entities': [entity1, entity2],
                'source_chunk': chunk.get('chunk_id', ''),
                'article_title': chunk.get('article_title', ''),
                'expected_content': text[:300] + "..."
            })
        
        # Single entity analytical questions
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity = random.choice(entity_list)
                template = random.choice([t for t in templates if '{entity}' in t or '{concept}' in t])
                question = template.replace('{entity}', entity).replace('{concept}', entity)
                
                questions.append({
                    'question': question,
                    'type': 'analytical',
                    'difficulty': 'medium',
                    'entity': entity,
                    'entity_type': entity_type,
                    'source_chunk': chunk.get('chunk_id', ''),
                    'article_title': chunk.get('article_title', ''),
                    'expected_content': text[:300] + "..."
                })
                break
        
        return questions
    
    def generate_contextual_questions(self, entities: Dict, text: str, chunk: Dict) -> List[Dict]:
        """Generate contextual questions."""
        questions = []
        templates = self.question_templates['contextual']
        
        # Generate context-aware questions
        concepts = entities.get('CONCEPT', []) + entities.get('MISC', [])
        if concepts:
            concept = random.choice(concepts)
            template = random.choice(templates)
            question = template.replace('{entity}', concept).replace('{concept}', concept)
            
            # Try to infer field/domain from title
            title = chunk.get('article_title', '')
            field = self.infer_field_from_title(title)
            question = question.replace('{field}', field).replace('{category}', field)
            question = question.replace('{system}', concept).replace('{process}', concept)
            
            questions.append({
                'question': question,
                'type': 'contextual',
                'difficulty': 'hard',
                'concept': concept,
                'domain': field,
                'source_chunk': chunk.get('chunk_id', ''),
                'article_title': chunk.get('article_title', ''),
                'expected_content': text[:400] + "..."
            })
        
        return questions
    
    def infer_field_from_title(self, title: str) -> str:
        """Infer academic/professional field from article title."""
        field_keywords = {
            'science': ['physics', 'chemistry', 'biology', 'science', 'research', 'theory'],
            'technology': ['computer', 'software', 'algorithm', 'system', 'network', 'digital'],
            'history': ['war', 'battle', 'historical', 'ancient', 'century', 'empire'],
            'politics': ['government', 'political', 'president', 'minister', 'parliament', 'democracy'],
            'arts': ['music', 'painting', 'literature', 'artist', 'cultural', 'museum'],
            'geography': ['country', 'city', 'river', 'mountain', 'continent', 'region'],
            'medicine': ['medical', 'disease', 'treatment', 'health', 'clinical', 'therapy'],
            'business': ['company', 'corporation', 'market', 'economic', 'financial', 'industry']
        }
        
        title_lower = title.lower()
        for field, keywords in field_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return field
        
        return 'general studies'
    
    def generate_question_dataset(self, target_count: int = 100) -> List[Dict[str, Any]]:
        """Generate a comprehensive question dataset."""
        print(f"🎯 Generating {target_count} evaluation questions...")
        
        # Load processed chunks
        chunks_path = self.work_dir / "data" / "processed" / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"📚 Processing {len(chunks)} chunks...")
        
        all_questions = []
        articles_used = []
        
        # Shuffle chunks for diversity
        random.shuffle(chunks)
        
        for chunk in chunks:
            if len(all_questions) >= target_count:
                break
            
            article_title = chunk.get('article_title', 'Unknown')
            
            # Limit questions per article for diversity
            if articles_used.count(article_title) >= 3:
                continue
            
            questions = self.generate_questions_from_chunk(chunk)
            
            for question in questions:
                if len(all_questions) >= target_count:
                    break
                
                # Add metadata
                question.update({
                    'question_id': f"q_{len(all_questions) + 1:03d}",
                    'generated_at': datetime.now().isoformat(),
                    'source_article': article_title,
                    'chunk_word_count': chunk.get('chunk_word_count', 0)
                })
                
                all_questions.append(question)
                articles_used.append(article_title)
        
        print(f"✅ Generated {len(all_questions)} questions from {len(set(articles_used))} articles")
        
        # Balance question types and difficulties
        return self.balance_question_dataset(all_questions, target_count)
    
    def balance_question_dataset(self, questions: List[Dict], target_count: int) -> List[Dict]:
        """Balance the dataset by type and difficulty."""
        # Group by type and difficulty
        grouped = defaultdict(lambda: defaultdict(list))
        for q in questions:
            grouped[q['type']][q['difficulty']].append(q)
        
        # Target distribution
        type_targets = {'factual': 0.4, 'analytical': 0.35, 'contextual': 0.25}
        difficulty_targets = {'easy': 0.3, 'medium': 0.45, 'hard': 0.25}
        
        balanced_questions = []
        
        for q_type, type_ratio in type_targets.items():
            type_count = int(target_count * type_ratio)
            
            for difficulty, diff_ratio in difficulty_targets.items():
                diff_count = int(type_count * diff_ratio)
                available = grouped[q_type][difficulty]
                
                if available:
                    selected = random.sample(available, min(diff_count, len(available)))
                    balanced_questions.extend(selected)
        
        # Fill remaining slots if needed
        while len(balanced_questions) < target_count and questions:
            remaining = [q for q in questions if q not in balanced_questions]
            if remaining:
                balanced_questions.append(random.choice(remaining))
            else:
                break
        
        return balanced_questions[:target_count]
    
    def save_question_dataset(self, questions: List[Dict], output_format: str = 'both'):
        """Save questions in JSON and/or CSV format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        if output_format in ['json', 'both']:
            json_path = self.work_dir / "data" / "processed" / f"evaluation_questions_{timestamp}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'total_questions': len(questions),
                        'question_types': {
                            'factual': len([q for q in questions if q['type'] == 'factual']),
                            'analytical': len([q for q in questions if q['type'] == 'analytical']),
                            'contextual': len([q for q in questions if q['type'] == 'contextual'])
                        },
                        'difficulty_levels': {
                            'easy': len([q for q in questions if q['difficulty'] == 'easy']),
                            'medium': len([q for q in questions if q['difficulty'] == 'medium']),
                            'hard': len([q for q in questions if q['difficulty'] == 'hard'])
                        }
                    },
                    'questions': questions
                }, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved JSON dataset: {json_path}")
        
        # Save as CSV
        if output_format in ['csv', 'both']:
            import csv
            csv_path = self.work_dir / "data" / "processed" / f"evaluation_questions_{timestamp}.csv"
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if questions:
                    writer = csv.DictWriter(f, fieldnames=questions[0].keys())
                    writer.writeheader()
                    writer.writerows(questions)
            
            print(f"📊 Saved CSV dataset: {csv_path}")
        
        # Also save a standard name for easy access
        standard_json = self.work_dir / "data" / "processed" / "evaluation_questions.json"
        with open(standard_json, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_questions': len(questions),
                    'description': 'RAG System Evaluation Questions - Diverse set covering factual, analytical, and contextual queries'
                },
                'questions': questions
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Saved standard dataset: {standard_json}")
        
        return json_path if output_format != 'csv' else csv_path

def main():
    """Main function to generate question dataset."""
    work_dir = Path(".")
    generator = AdvancedQuestionGenerator(work_dir)
    
    print("🚀 Starting Question Generation Process")
    print("=" * 50)
    
    try:
        # Generate 100 questions
        questions = generator.generate_question_dataset(target_count=100)
        
        # Save in both formats
        output_path = generator.save_question_dataset(questions, 'both')
        
        print("\n📈 Dataset Statistics:")
        print(f"Total questions: {len(questions)}")
        
        # Print type distribution
        type_counts = defaultdict(int)
        difficulty_counts = defaultdict(int)
        for q in questions:
            type_counts[q['type']] += 1
            difficulty_counts[q['difficulty']] += 1
        
        print("Question types:")
        for q_type, count in type_counts.items():
            print(f"  • {q_type.capitalize()}: {count} ({count/len(questions)*100:.1f}%)")
        
        print("Difficulty levels:")
        for difficulty, count in difficulty_counts.items():
            print(f"  • {difficulty.capitalize()}: {count} ({count/len(questions)*100:.1f}%)")
        
        print("\n✅ Question generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()