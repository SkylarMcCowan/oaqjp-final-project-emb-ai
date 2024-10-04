from transformers import pipeline

# Load the pre-trained emotion detection model
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

def emotion_detector(text_to_analyze):
    try:
        # Get the emotion predictions
        predictions = emotion_classifier(text_to_analyze)
        
        # Extract the emotion with the highest score
        top_emotion = max(predictions[0], key=lambda x: x['score'])
        
        return top_emotion['label']
    except Exception as e:
        return f"An error occurred: {e}"

# Testing the function in Python shell
if __name__ == "__main__":
    text = "I love this new technology."
    print(emotion_detector(text))