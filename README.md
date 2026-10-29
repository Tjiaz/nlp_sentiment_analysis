# NLP Sentiment Analysis

This project builds a sentiment analysis model for movie reviews using spaCy's text classification pipeline. The model is trained on the IMDb-style ACL IMDb dataset and classifies review text as either positive or negative.

The repository contains a Jupyter notebook, `nlp-sentiment-analysis.ipynb`, which walks through:

- loading and preparing review data
- converting labels into spaCy text classification format
- training a binary text classifier
- evaluating the model with precision, recall, and F-score
- saving the trained model and plotting training metrics
- testing the final model on a sample review

## Project Overview

The goal of this project is to predict the sentiment of a written review using natural language processing techniques. It demonstrates how to:

- clean and normalize text data
- map labels to spaCy textcat categories
- train a text classification model from scratch
- evaluate model performance
- save the trained model for later inference

## Tech Stack

- Python 3
- Jupyter Notebook
- spaCy
- pandas
- NumPy
- matplotlib

## Dataset

This project uses the ACL IMDb dataset structure:

- `train/pos/`
- `train/neg/`

Each file contains a review as plain text. The notebook loads these files, normalizes line breaks, and converts each review into a spaCy-compatible training example.

## Model Details

The project uses a spaCy `textcat` model with two labels:

- `pos`
- `neg`

Training is performed using mini-batches and evaluation is tracked across iterations using:

- precision
- recall
- F-score

In the notebook, the model reaches a best F-score of approximately 0.8271 during validation.

## Repository Contents

- `nlp-sentiment-analysis.ipynb` — the full training and evaluation workflow
- `model_artifacts/` — saved trained spaCy model (generated after training)
- `training_metrics.png` — training performance plot (generated after training)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Tjiaz/nlp_sentiment_analysis.git
cd nlp_sentiment_analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -U spacy spacy-lookups-data pandas numpy matplotlib
python -m spacy download en_core_web_sm
```

### 4. Open the notebook

```bash
jupyter notebook
```

Then open `nlp-sentiment-analysis.ipynb` and run the cells in order.

## Usage

After training, the notebook includes a test function that loads the saved model and predicts sentiment for a sample review.

Example:

```python
import spacy

nlp = spacy.load("model_artifacts")
review = "This movie was absolutely fantastic and very engaging."
doc = nlp(review)
print(doc.cats)
```

The output will include the model's confidence scores for the `pos` and `neg` labels.

## Example Workflow

The notebook performs the following steps:

1. Reads text reviews from the dataset directory.
2. Converts each review into spaCy `Example` objects.
3. Splits the data into train/test sets.
4. Initializes a blank English spaCy pipeline with a text classifier.
5. Trains the model for multiple iterations.
6. Evaluates performance using validation data.
7. Saves the best model version to disk.
8. Tests the trained model on a sample review.

## Notes

- The notebook currently assumes the dataset is available in a Kaggle-style directory path. If running locally, update the `data_directory` path in the notebook to match your project structure.
- For optimal results, you may want to use a larger dataset or tune hyperparameters such as training iterations, batch size, or dropout.

## License

This project does not currently include a specific license file. If you plan to share or reuse it publicly, consider adding an appropriate open-source license.

## Acknowledgements

- spaCy for the NLP pipeline and text classification tooling
- ACL IMDb dataset for the training data
- Python ecosystem libraries used throughout the workflow
