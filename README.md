# Emotion Intelligence

Emotion Intelligence is a multi-label text emotion classification project that compares traditional machine learning approaches with a transformer model.

I built this project to explore a simple question:

**How well do traditional NLP models and transformer models recognize emotions in text, and where do they still struggle?**

The project compares TF-IDF based Logistic Regression and Linear SVM models with a fine-tuned DistilBERT model. I also built a Streamlit application where users can enter their own text and see the emotions predicted by the DistilBERT model.

## Live Demo

Try the deployed application: https://emotion-intelligence.streamlit.app

The trained DistilBERT model is hosted on Hugging Face:
https://huggingface.co/HarikaRamanadham04/emotion-intelligence-distilbert

## Dataset

I used the simplified version of the **GoEmotions** dataset from Google Research.

The dataset contains short text comments labeled with 27 emotion categories plus neutral. Since one sentence can contain more than one emotion, this is treated as a **multi-label classification problem**.

Dataset split:

| Split | Examples |
|---|---:|
| Training | 43,410 |
| Validation | 5,426 |
| Test | 5,427 |
| **Total** | **54,263** |

During exploratory analysis, I found that the emotion classes are not evenly distributed. The comments are also generally short, with an average length of about 12.8 words.

## Models

I started with two traditional machine learning baselines.

### Logistic Regression

The text was converted into numerical features using TF-IDF. Since the dataset is multi-label, I used a One-vs-Rest approach where a separate binary classifier is learned for each emotion.

### Linear SVM

I used the same TF-IDF representation with a Linear SVM. This provided a stronger traditional NLP baseline and allowed me to compare it with the transformer model.

### DistilBERT

For the contextual model, I fine-tuned **DistilBERT** for multi-label classification.

I chose DistilBERT because it provides contextual text representations while being smaller and more computationally practical than standard BERT. This was useful since the model was trained locally on CPU.

The model uses sigmoid outputs so each emotion is predicted independently.

## Threshold Selection

For multi-label classification, the model needs a threshold to decide whether each emotion should be included in the prediction.

Instead of automatically using 0.50, I tested several thresholds on the validation set.

A threshold of **0.30** gave the best Micro F1 score among the thresholds tested for that selection objective, so I used 0.30 for the final test evaluation.

The test set was kept separate from this threshold selection.

## Results

All three models were evaluated on the same held-out test set.

| Model | Precision | Recall | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7113 | 0.2963 | 0.4183 | 0.2366 |
| Linear SVM | 0.6691 | 0.3664 | 0.4735 | 0.3379 |
| DistilBERT | 0.6051 | 0.6268 | **0.6158** | **0.4505** |

Logistic Regression had the highest precision, but its recall was much lower, meaning it often did not predict emotions that were present.

Linear SVM improved recall and F1 compared with Logistic Regression.

DistilBERT had lower precision than the traditional models, but much higher recall and the highest Micro F1 and Macro F1 in this comparison.

## Looking at Model Predictions

I also tested the models with several sentences to understand their behavior beyond the overall metrics.

For example:

**Input:**  
`The ending of La La Land is depressing`

- Logistic Regression: No emotion predicted
- Linear SVM: No emotion predicted
- DistilBERT: Sadness, Disappointment

Another example:

**Input:**  
`What are your thoughts on the new Batman movie?`

- Logistic Regression: No emotion predicted
- Linear SVM: No emotion predicted
- DistilBERT: Curiosity, Neutral

These examples suggest that the contextual model can recognize some expressions that the TF-IDF models miss.

However, DistilBERT does not solve every case.

For example:

**Input:**  
`It sucks that I dropped my cookie on the floor because I really wanted to eat it`

None of the three models predicted an emotion above the chosen threshold.

This is an interesting limitation because the emotional meaning has to be inferred from the situation rather than from a direct emotion word.

These sentences are qualitative examples created for testing and are not manually annotated benchmark examples, so they are used to inspect model behavior rather than calculate performance.

## Streamlit Application

I built a simple Streamlit interface so the trained model can be tested with new sentences.

The user enters text, and the application returns all emotion labels whose model confidence score is above the selected threshold.

The trained DistilBERT model is hosted on Hugging Face and loaded by the Streamlit application during inference.

## Project Structure

```text
emotion-intelligence/
├── app.py
├── emotion_analysis.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

`emotion_analysis.ipynb` contains the data exploration, preprocessing, baseline models, DistilBERT training, evaluation, and model comparison.

`app.py` contains the Streamlit inference application.

## Running the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python -m streamlit run app.py
```

The application downloads the trained model from Hugging Face when it starts.

## Limitations

This version of the project has several limitations.

The DistilBERT model was fine-tuned for one epoch because training was performed locally on CPU. The dataset also has class imbalance, which makes some less common emotions harder to learn.

I currently use one global prediction threshold for all 28 labels. Different thresholds for individual emotion classes could potentially improve performance.

The model can also struggle when an emotion is indirect, depends on outside knowledge, or is expressed through more complicated language such as sarcasm.

## Future Work

I am interested in extending this project beyond text emotion classification and exploring how intelligent systems can better recognize human psychological and behavioral patterns.

One direction I would like to explore is combining different types of human signals, such as text, facial expressions, voice, and behavioral cues, to understand how they contribute to emotion prediction. I am particularly interested in cases where emotions are not directly expressed and need to be inferred from context or behavior.

On the technical side, the current model can also be improved by exploring class-specific thresholds, class imbalance handling, model explainability, and other transformer models.
