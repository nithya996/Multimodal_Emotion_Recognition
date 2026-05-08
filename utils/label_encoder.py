def normalize_emotion(emotion):

    emotion = emotion.lower()

    if emotion in [
        "pleasant_surprise",
        "pleasant_surprised",
        "pleasant surprise",
        "pleasant-surprise",
        "surprised",
        "surprise",
        "ps"
    ]:
        return "pleasant_surprise"

    return emotion


emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "pleasant_surprise": 5,
    "sad": 6
}


reverse_emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "pleasant_surprise",
    6: "sad"
}