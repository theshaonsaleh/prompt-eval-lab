# Sentiment Classification Dataset
# Public domain, hand-crafted for educational purposes
# 100 short sentences with binary sentiment labels (positive=1, negative=0)

SENTIMENT_DATASET = [
    # Positive examples (50)
    ("I love this sunny weather", 1),
    ("This book is absolutely wonderful", 1),
    ("Great job on the presentation", 1),
    ("The concert was amazing", 1),
    ("I feel fantastic today", 1),
    ("This is the best day ever", 1),
    ("Excellent work on that project", 1),
    ("The food tastes delicious", 1),
    ("I'm so happy about the news", 1),
    ("This movie is brilliant", 1),
    ("What a beautiful sunset", 1),
    ("I appreciate your help", 1),
    ("The service was outstanding", 1),
    ("This is incredibly useful", 1),
    ("I'm excited about tomorrow", 1),
    ("The team did a superb job", 1),
    ("This product exceeded my expectations", 1),
    ("I'm grateful for this opportunity", 1),
    ("The performance was spectacular", 1),
    ("This is exactly what I needed", 1),
    ("I'm thrilled with the results", 1),
    ("The design looks perfect", 1),
    ("I enjoy spending time here", 1),
    ("This solution works wonderfully", 1),
    ("I'm impressed by the quality", 1),
    ("The atmosphere is lovely", 1),
    ("This makes me very happy", 1),
    ("I'm pleased with the outcome", 1),
    ("The experience was delightful", 1),
    ("This is a fantastic idea", 1),
    ("I admire the creativity shown", 1),
    ("The results are remarkable", 1),
    ("This is better than expected", 1),
    ("I'm optimistic about the future", 1),
    ("The support team is helpful", 1),
    ("This is a positive development", 1),
    ("I'm satisfied with everything", 1),
    ("The quality is exceptional", 1),
    ("This brings me joy", 1),
    ("I'm proud of the achievement", 1),
    ("The event was memorable", 1),
    ("This is a great success", 1),
    ("I'm confident in the plan", 1),
    ("The interface is intuitive", 1),
    ("This is very encouraging", 1),
    ("I'm delighted by the progress", 1),
    ("The outcome is favorable", 1),
    ("This is truly remarkable", 1),
    ("I'm content with the decision", 1),
    ("The improvement is significant", 1),
    
    # Negative examples (50)
    ("I hate this rainy weather", 0),
    ("This book is terrible", 0),
    ("Poor performance on the test", 0),
    ("The concert was disappointing", 0),
    ("I feel awful today", 0),
    ("This is the worst day ever", 0),
    ("Terrible work on that project", 0),
    ("The food tastes disgusting", 0),
    ("I'm so sad about the news", 0),
    ("This movie is horrible", 0),
    ("What an ugly mess", 0),
    ("I'm frustrated by this", 0),
    ("The service was appalling", 0),
    ("This is completely useless", 0),
    ("I'm worried about tomorrow", 0),
    ("The team did a poor job", 0),
    ("This product failed to meet expectations", 0),
    ("I'm disappointed by this opportunity", 0),
    ("The performance was mediocre", 0),
    ("This is not what I wanted", 0),
    ("I'm unhappy with the results", 0),
    ("The design looks awful", 0),
    ("I dislike being here", 0),
    ("This solution doesn't work", 0),
    ("I'm unimpressed by the quality", 0),
    ("The atmosphere is unpleasant", 0),
    ("This makes me very sad", 0),
    ("I'm displeased with the outcome", 0),
    ("The experience was dreadful", 0),
    ("This is a terrible idea", 0),
    ("I'm concerned about the approach", 0),
    ("The results are poor", 0),
    ("This is worse than expected", 0),
    ("I'm pessimistic about the future", 0),
    ("The support team is unhelpful", 0),
    ("This is a negative development", 0),
    ("I'm dissatisfied with everything", 0),
    ("The quality is subpar", 0),
    ("This brings me sadness", 0),
    ("I'm ashamed of the failure", 0),
    ("The event was forgettable", 0),
    ("This is a complete failure", 0),
    ("I'm doubtful about the plan", 0),
    ("The interface is confusing", 0),
    ("This is very discouraging", 0),
    ("I'm troubled by the progress", 0),
    ("The outcome is unfavorable", 0),
    ("This is truly disappointing", 0),
    ("I'm upset with the decision", 0),
    ("The decline is significant", 0),
]

def get_dataset():
    """Return the complete dataset as (texts, labels) tuple."""
    texts = [text for text, _ in SENTIMENT_DATASET]
    labels = [label for _, label in SENTIMENT_DATASET]
    return texts, labels

def get_few_shot_examples(n=3):
    """Get n examples for few-shot prompting (balanced classes)."""
    examples = []
    pos_count = 0
    neg_count = 0
    
    for text, label in SENTIMENT_DATASET:
        if label == 1 and pos_count < n // 2 + 1:
            examples.append((text, label))
            pos_count += 1
        elif label == 0 and neg_count < n // 2:
            examples.append((text, label))
            neg_count += 1
        
        if len(examples) >= n:
            break
    
    return examples
