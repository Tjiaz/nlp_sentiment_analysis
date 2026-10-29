# NLP Sentiment Analysis

A notebook-based natural language processing project for classifying movie reviews as positive or negative using spaCy's text classification pipeline.

This project demonstrates how to preprocess review text, build a supervised sentiment classification model, evaluate its performance, and save the trained model for inference.

## Overview

Sentiment analysis is one of the most common tasks in NLP. In this project, a binary classifier is trained on labeled review text to determine whether a review expresses a positive or negative sentiment.

The workflow includes:

- loading and cleaning text review data
- converting labels into spaCy-compatible category format
- training a text classification model from scratch
- tracking precision, recall, and F-score
- saving the trained model artifacts
- testing the final model on a sample review

## Project Goals

- Build a working sentiment classification model
- Learn how text classification works in spaCy
- Evaluate the model using standard classification metrics
- Package the training workflow in a reproducible notebook

## Tech Stack

- Python 3
- Jupyter Notebook
- spaCy
- pandas
- NumPy
- matplotlib

## Dataset

The project uses the IMDb-style ACL IMDb sentiment dataset, organized into positive and negative review folders.

Typical dataset structure:

```text
train/
  pos/
  neg/
```

Each text file contains a review, and the folder name defines the label.

## Model Architecture

The project uses spaCy's `textcat` component with two labels:

- `pos`
- `neg`

The model is trained using mini-batches and evaluated on a validation/test set with:

- precision
- recall
- F-score

## Repository Contents

```text
.
├── nlp-sentiment-analysis.ipynb
├── README.md
├── model_artifacts/
├── training_metrics.png
└── ...
```

- `nlp-sentiment-analysis.ipynb`: notebook containing the full training and evaluation workflow
- `README.md`: project documentation
- `model_artifacts/`: saved trained spaCy model
- `training_metrics.png`: visual training metrics plot

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

After training, the notebook loads the saved model and predicts sentiment for a sample review.

Example:

```python
import spacy

nlp = spacy.load("model_artifacts")
review = "This movie was absolutely fantastic and very engaging."
doc = nlp(review)
print(doc.cats)
```

The model returns probability scores for each class:

```python
{'pos': 0.98, 'neg': 0.02}
```

## Training Workflow

The notebook follows this pipeline:

1. Reads review files from the dataset directory
2. Cleans text by replacing HTML-style line breaks like `<br />`
3. Converts labels into spaCy text classification format
4. Splits data into training and test sets
5. Initializes a blank English spaCy model
6. Adds a text classification pipe
7. Trains the model over multiple iterations
8. Evaluates performance on the test set
9. Saves the best-performing model
10. Tests the model on custom input

## Example Result

The notebook reports validation performance metrics during training, with the best F-score reaching approximately 0.8271.

This indicates strong classification ability for the selected review dataset.

## Notes

- The notebook assumes the dataset is available in a Kaggle-style local directory structure. If running locally, update the `data_directory` path to match your environment.
- For stronger performance, you can expand the dataset, tune hyperparameters, or train for more iterations.
- This project is primarily an educational and experimental NLP workflow rather than a production-ready API service.

## License

No explicit license file is currently included in the repository. If you plan to publish or share the project publicly, consider adding an appropriate open-source license.

## Acknowledgements

- spaCy for the NLP pipeline and text classification tools
- IMDb sentiment dataset for the labeled review examples
- Python ecosystem libraries used throughout the workflow

## Contact

For questions or collaboration opportunities, feel free to reach out through the repository owner or project maintainer.
