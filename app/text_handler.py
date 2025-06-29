from transformers import pipeline

# Load a zero-shot classifier (can predict from text with labels you give)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_text(input_text):
    labels = ["billing issue", "technical problem", "order status", "account issue", "general inquiry"]
    
    result = classifier(input_text, labels)
    top_label = result['labels'][0]
    score = result['scores'][0]

    return {
        "input": input_text,
        "predicted_label": top_label,
        "confidence": round(score, 3)
    }
