# Introvert vs Extrovert Personality Prediction

A machine learning model and Streamlit app that predicts whether a person is an **introvert** or an **extrovert** from seven everyday social habits, such as time spent alone, social event attendance and how drained they feel after socialising.

Built with Python, CatBoost, scikit-learn and Streamlit.

## Dataset

18,524 labelled people (`train.csv`) and 6,175 unlabelled people (`test.csv`) from the Kaggle Playground Series personality dataset.

| Personality | People | Share |
| --- | --- | --- |
| Extrovert | 13,699 | 74% |
| Introvert | 4,825 | 26% |

| Feature | Type | Meaning |
| --- | --- | --- |
| `Time_spent_Alone` | 0–11 | Hours spent alone per day |
| `Stage_fear` | Yes / No | Fear of public speaking or performing |
| `Social_event_attendance` | 0–10 | How often they attend social events |
| `Going_outside` | 0–7 | How often they go outside |
| `Drained_after_socializing` | Yes / No | Feels drained after socialising |
| `Friends_circle_size` | 0–15 | Number of close friends |
| `Post_frequency` | 0–10 | How often they post on social media |

## Approach

1. **Exploratory analysis:** class balance, missing values, a correlation heatmap and pair plots. `Time_spent_Alone` correlates most strongly with being an introvert (0.78), while social activity features correlate negatively.
2. **Cleaning:** every feature has 5–10% missing values. Numeric features are filled with the training median and Yes/No features with the training mode, so no rows are dropped.
3. **Class imbalance:** introverts are only 26% of the data, so the model uses inverse-frequency class weights (Extrovert 1.35, Introvert 3.84) to penalise missed introverts more.
4. **Model:** a `CatBoostClassifier` evaluated with stratified 5-fold cross-validation and early stopping, then retrained on all training data and saved for the app.

## Results

Out-of-fold predictions across all 18,524 people (5-fold stratified cross-validation):

| Metric | Score |
| --- | --- |
| Accuracy | 0.969 |
| Macro F1 | 0.96 |
| Introvert recall | 0.94 |

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Extrovert | 0.98 | 0.98 | 0.98 |
| Introvert | 0.94 | 0.94 | 0.94 |

**Confusion matrix** (rows = actual, columns = predicted):

| | Predicted Extrovert | Predicted Introvert |
| --- | --- | --- |
| **Actual Extrovert** | 13,426 | 273 |
| **Actual Introvert** | 300 | 4,525 |

### What the results show

- **The classes are very easy to separate.** Early stopping halts CatBoost after only a handful of trees, because accuracy stops improving almost immediately. A simple model captures nearly all the signal in these seven features.
- **Feeling drained after socialising is the strongest predictor** (25% of feature importance), followed by post frequency (19%) and time spent alone (15%).
- **The remaining errors are evenly split:** about 300 introverts are predicted as extroverts and about 270 the other way round. These are likely people whose habits sit between the two types.

## Run It Locally

```bash
git clone https://github.com/Teninlanimi/introvert-extrovert.git
cd introvert-extrovert
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, open `Introverts or Extroverts prediction.ipynb` and run all cells. It saves a new `personality_model.pkl`, `final_feature_list.txt` and `submission.csv`.

## Project Structure

| File | Purpose |
| --- | --- |
| `Introverts or Extroverts prediction.ipynb` | Analysis, cleaning, model training, evaluation and saving |
| `app.py` | Streamlit app: input form, prediction and probability chart |
| `personality_model.pkl` | The trained CatBoost model loaded by the app |
| `final_feature_list.txt` | The 7 input features, in the order the model expects |
| `train.csv`, `test.csv` | Training and test data |
| `submission.csv` | Predicted personality for every person in `test.csv` |
| `requirements.txt` | Python dependencies |
