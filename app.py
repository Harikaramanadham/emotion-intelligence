import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "HarikaRamanadham04/emotion-intelligence-distilbert"
THRESHOLD = 0.30

tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH

)

model.eval()



def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits)[0]

    predictions = []

    for i, probability in enumerate(probabilities):
        if probability >= THRESHOLD:
            emotion = model.config.id2label[i]

            predictions.append(
                (emotion, float(probability))
            )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return predictions

st.set_page_config(
    page_title="Emotion Intelligence",
    page_icon="😊"
)
st.title("Emotion Intelligence")

st.write(
    "Enter a sentence to analyze the emotions expresses in the text"
)
user_text=st.text_area(
    "Enter Text:", placeholder="Example: I am really happy today!"
)

if st.button("Analyze Emotion"):
    if user_text:
        predictions = predict_emotion(user_text)
        if predictions:
            st.subheader("Predicted Emotions")
            for emotion, probability in predictions:
                st.write(
                    f"**{emotion.title()}** - {probability*100:1.2f}%"
                )

        else:
            st.write("No emotions detected above the threshold.")
    else:
        st.write("Please enter some text to analyze.")