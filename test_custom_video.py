from inference import SentimentPredictor

predictor = SentimentPredictor('saved_models/best_model.pth')

# Test your video
video_path = '"C:\Users\nikhil pratee\Downloads\130513-747472688.mp4"'
result = predictor.predict(video_path)

print(f"\nVideo: {video_path}")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\nProbabilities:")
for label, prob in result['probabilities'].items():
    print(f"  {label}: {prob:.2%}")