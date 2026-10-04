# Multi-Class Toxicity Classification using RNN and LSTM

## 📌 Project Overview

This project is a **multi-class toxicity classification system** built using **RNN and LSTM** models with PyTorch.

The system accepts **text and/or images** as input and classifies the content into one of several safety-related categories.

For image input, an **image captioning model** is used to generate a textual description of the image. The generated caption is then combined with the user-provided text and passed through the same preprocessing and classification pipeline.

The project also includes a **backend, SQLite database, and Streamlit web application** that provide an interactive interface for making predictions and storing prediction results.

The main pipeline is:

```text
Text / Image
     ↓
Image Captioning (if image is provided)
     ↓
Text + Image Caption
     ↓
Data Cleaning & Preprocessing
     ↓
Tokenization
     ↓
Vocabulary / Sequence Conversion
     ↓
Padding
     ↓
Embedding
     ↓
RNN / LSTM
     ↓
Multi-Class Prediction
     ↓
SQLite Database
```

---

## 🎯 Project Objective

The main goal of this project is to understand and implement a complete multi-class text classification pipeline using recurrent neural networks.

The project compares two recurrent architectures:

* RNN
* LSTM

In addition to the machine learning models, the project extends the classifier into a complete application that can accept both text and image inputs.

For image inputs, an image captioning model converts the visual content into text before classification.

---

## 📊 Dataset

The original dataset contains text queries with their corresponding safety-related categories.

The original label mapping contains **9 categories**:

| Label | Category                  | Used for Training |
| ----: | ------------------------- | :---------------: |
|     0 | Safe                      |         ✅         |
|     1 | Violent Crimes            |         ✅         |
|     2 | Non-Violent Crimes        |         ✅         |
|     3 | unsafe                    |         ✅         |
|     4 | Unknown S-Type            |         ✅         |
|     5 | Sex-Related Crimes        |         ✅         |
|     6 | Suicide & Self-Harm       |         ✅         |
|     7 | Elections                 |         ❌         |
|     8 | Child Sexual Exploitation |         ❌         |

Although the original dataset contains 9 categories, the models were trained on the classes included in the training data.

The `Elections` and `Child Sexual Exploitation` categories were not included in the training stage.

---

## 🧹 Data Preparation

Before training the models, the original dataset went through several preparation steps.

### 1. Label Correction

During the initial data inspection, some text samples were found to have incorrect labels.

The labels were reviewed and corrected, and a new CSV file was created:

```text
train_true_labels.csv
```

The corrected labels were then used as the basis for the preprocessing and training stages.

### 2. Class Distribution Analysis

The class distribution was analyzed to identify underrepresented categories.

Some categories contained very few samples, which made learning their characteristics more difficult.

### 3. Manual Minority-Class Augmentation

Additional text records were manually added to extremely small minority classes.

This provided the models with more examples of underrepresented categories during training.

---

## 🔄 Text Preprocessing

The text was converted into numerical representations that could be processed by the neural networks.

### 1. Text Cleaning

The input text was cleaned before tokenization.

### 2. Tokenization

A custom tokenization process was used to split text into individual tokens.

Example:

```text
"this is a test"
```

becomes:

```text
["this", "is", "a", "test"]
```

### 3. Vocabulary Building

A vocabulary was created from the training data.

Each word was assigned a unique integer index.

Special tokens were also included:

```text
<PAD>
<UNK>
```

### 4. Text-to-Sequence

Each tokenized text was converted into a sequence of integer IDs.

### 5. Padding

Since text samples have different lengths, padding was applied so that sequences could be processed in batches.

---

## 🖼️ Image Captioning

The application also supports image input.

When the user uploads an image, the image is passed to an image captioning component based on:

```text
Salesforce/blip-image-captioning-base
```

The image captioning pipeline is:

```text
Input Image
     ↓
BLIP Processor
     ↓
BLIP Image Captioning Model
     ↓
Generated Caption
```

For example:

```text
Image
  ↓
"a man riding a bicycle on a street"
```

The generated caption is treated as text and can then be combined with the user's original text.

For example:

```text
User Text:
"Is this content safe?"

Image Caption:
"a man riding a bicycle on a street"

Combined Input:
"Is this content safe? a man riding a bicycle on a street"
```

The combined text then goes through the same preprocessing and classification pipeline used for text-only input.

---

## 🧠 Model Architecture

Two recurrent neural network architectures were implemented.

### RNN

The RNN architecture is:

```text
Input Sequence
      ↓
Embedding
      ↓
RNN
      ↓
Fully Connected Layer
      ↓
Multi-Class Output
```

The RNN processes the input sequence step by step while maintaining a hidden state.

### LSTM

The LSTM architecture is:

```text
Input Sequence
      ↓
Embedding
      ↓
LSTM
      ↓
Fully Connected Layer
      ↓
Multi-Class Output
```

LSTM uses gates to control which information should be retained or forgotten from previous time steps.

---

## ⚙️ Training

The models were implemented and trained using **PyTorch**.

The training process includes:

1. Loading the prepared dataset
2. Preprocessing the text
3. Converting text into numerical sequences
4. Creating PyTorch tensors
5. Creating training and testing datasets
6. Loading the data using DataLoaders
7. Forward propagation
8. Calculating the loss
9. Backpropagation
10. Updating model parameters
11. Evaluating the trained models

---

## ⚖️ Class Imbalance

Class imbalance was one of the main challenges in the project.

Several techniques were explored to improve minority-class learning, including:

* Manual minority-class augmentation
* Weighted sampling
* Class weighting
* Oversampling
* Different RNN/LSTM architectures
* Different preprocessing approaches

---

## 📈 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* Macro F1-score
* Weighted F1-score
* Confusion Matrix
* Classification Report

Macro F1-score was given particular attention because the dataset is imbalanced and it gives equal importance to each class.

---

## 📊 Results

### RNN

| Metric      | Result |
| ----------- | -----: |
| Accuracy    |   0.94 |
| Macro F1    |   0.84 |
| Weighted F1 |   0.94 |

### LSTM

| Metric      | Result |
| ----------- | -----: |
| Accuracy    |   0.99 |
| Macro F1    |   0.93 |
| Weighted F1 |   0.98 |

---

# 🏗️ Application Architecture

The trained models were integrated into a complete application.

The application consists of three main components:

```text
Streamlit Frontend
        ↓
      Backend
        ↓
 RNN / LSTM + BLIP
        ↓
 SQLite Database
```

### Streamlit

A **Streamlit web application** was developed to provide an interactive interface for the classifier.

The application allows the user to:

* Enter text
* Upload an image
* Use text and image together
* Select the classification model
* Choose between **RNN and LSTM**
* View the predicted category
* View the prediction confidence

### Backend

A backend layer was implemented to handle the application logic.

The backend connects the Streamlit interface with:

* Text preprocessing
* Image captioning
* RNN model
* LSTM model
* Prediction processing
* SQLite database

This keeps the application interface separate from the machine learning logic.

### SQLite Database

A **SQLite database** was added to store prediction results.

The database stores information related to each prediction, including:

* Input type
* Input text
* Selected model
* Image path
* Generated image caption
* Predicted label
* Confidence
* Prediction timestamp

This allows prediction results to be stored and retrieved locally without requiring an external database server.

---

# 🔀 Application Workflow

The application supports multiple input scenarios.

### Text Only

```text
User Text
    ↓
Preprocessing
    ↓
RNN / LSTM
    ↓
Prediction
    ↓
SQLite
```

### Image Only

```text
Image
  ↓
BLIP Image Captioning
  ↓
Generated Caption
  ↓
Preprocessing
  ↓
RNN / LSTM
  ↓
Prediction
  ↓
SQLite
```

### Text + Image

```text
User Text ──────────────┐
                        ↓
Image → BLIP → Caption → Combined Text
                              ↓
                        Preprocessing
                              ↓
                         RNN / LSTM
                              ↓
                         Prediction
                              ↓
                            SQLite
```

---

# 🖥️ Streamlit Application

The final application provides a simple interface for interacting with the trained models.

The user can select:

```text
Model:
○ RNN
○ LSTM
```

and provide:

```text
Text Input
+
Image Input (optional)
```

The application then processes the input and displays:

```text
Predicted Category
Confidence
Generated Image Caption (if an image was provided)
Selected Model
```

The prediction is also stored in the SQLite database.

---

## 🛠️ Technologies

* Python
* PyTorch
* Transformers
* BLIP
* Pandas
* NumPy
* Scikit-learn
* Pillow
* Streamlit
* SQLite

---

## 📁 Project Structure

```text
project/
│
├── train_true_labels.csv
├── data_preprocessing.py
├── backend.py
├── database.py
├── app.py
├── image_captioning.py
├── inference.py
│
├── RNN/
│   └── rnn.py
    └── ..
│
├── LSTM/
│   └── train_lstm.py
    └── .. 
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Malkah04/Cellula-week2.git
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 💡 Key Takeaway

This project demonstrates a complete **multi-class toxicity classification system using RNN and LSTM**, starting from raw text data and extending to a functional application.

The project includes:

* Data cleaning and label correction
* Minority-class augmentation
* Custom text preprocessing
* Tokenization and vocabulary building
* Text-to-sequence conversion
* Padding and embedding
* RNN and LSTM classification
* Image captioning using BLIP
* Text and image input support
* Backend integration
* SQLite database storage
* Streamlit web application
* Dynamic RNN/LSTM model selection

The final system connects the machine learning pipeline to a practical application where users can provide text, images, or both and receive a toxicity classification prediction.
