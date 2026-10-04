# Multi-Class Toxicity Classification using RNN and LSTM

## 📌 Project Overview

This project is a **multi-class toxicity classification system** built using **RNN and LSTM** models with PyTorch.

The system accepts **text and/or images** as input and classifies the content into one of **9 safety-related categories**.

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
9-Class Prediction
     ↓
SQLite Database
```

---

## 🎯 Project Objective

The main goal of this project is to implement a complete multi-class toxicity classification pipeline using recurrent neural networks.

The project compares two recurrent architectures:

* RNN
* LSTM

In addition to the machine learning models, the classifier was integrated into a complete application that can accept both text and image inputs.

For image inputs, an image captioning model converts the visual content into text before classification.

---

## 📊 Dataset

The dataset contains text queries with their corresponding safety-related categories.

The classification task contains **9 categories**:

| Label | Category                  |
| ----: | ------------------------- |
|     0 | Safe                      |
|     1 | Violent Crimes            |
|     2 | Non-Violent Crimes        |
|     3 | unsafe                    |
|     4 | Unknown S-Type            |
|     5 | Sex-Related Crimes        |
|     6 | Suicide & Self-Harm       |
|     7 | Elections                 |
|     8 | Child Sexual Exploitation |

All **9 categories** are included in the final classification task.

Therefore, the neural network output layer contains:

```python
num_classes = 9
```

---

## 🧹 Data Preparation

Before training the models, the dataset went through several preparation steps.

### 1. Label Correction

During the initial data inspection, some samples were found to have incorrect labels.

The labels were manually reviewed and corrected, and the corrected dataset was saved as:

```text
train_true_labels.csv
```

The corrected labels were then used for preprocessing and model training.

### 2. Duplicate and Data Inspection

The dataset was inspected for:

* Missing values
* Duplicate samples
* Incorrect labels
* Text length
* Class distribution

This helped identify data quality issues and understand the structure of the dataset before training.

### 3. Class Distribution Analysis

The class distribution was analyzed to identify underrepresented categories.

The dataset is highly imbalanced, with some classes containing significantly fewer samples than the majority classes.

### 4. Minority-Class Augmentation

Additional text records were manually added to extremely small minority classes.

This was done to provide the models with more examples of the underrepresented categories.

---

## 🔄 Text Preprocessing

The text was converted into numerical representations that could be processed by the neural networks.

### 1. Text Cleaning

The input text was cleaned before tokenization.

The same preprocessing pipeline is used during both training and inference to ensure consistency between the training data and the Streamlit application.

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

Two recurrent neural network architectures were implemented using PyTorch.

### RNN

```text
Input Sequence
      ↓
Embedding
      ↓
RNN
      ↓
Fully Connected Layer
      ↓
9-Class Output
```

### LSTM

```text
Input Sequence
      ↓
Embedding
      ↓
LSTM
      ↓
Fully Connected Layer
      ↓
9-Class Output
```

The LSTM uses gates to control which information should be retained or forgotten from previous time steps, allowing it to better handle dependencies across sequences.

---

## ⚙️ Training

The models were implemented and trained using **PyTorch**.

The training process includes:

1. Loading the prepared dataset
2. Preprocessing the text
3. Building the vocabulary
4. Tokenizing the text
5. Converting text into numerical sequences
6. Padding sequences
7. Creating PyTorch tensors
8. Splitting the data into training and test sets
9. Creating DataLoaders
10. Forward propagation
11. Calculating the loss
12. Backpropagation
13. Updating model parameters
14. Evaluating the trained models

---

## ⚖️ Class Imbalance

Class imbalance was one of the main challenges in the project.

Several techniques were explored to improve minority-class learning, including:

* Manual minority-class augmentation
* Weighted sampling
* Class weighting
* Oversampling
* Different RNN/LSTM architectures
* Different preprocessing configurations

Because the dataset is imbalanced, evaluation does not rely only on accuracy.

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

### Why Macro F1?

Macro F1-score was given particular attention because the dataset is highly imbalanced.

Macro F1 calculates the F1-score independently for each class and then gives every class equal importance.

This makes it more informative than accuracy when minority classes are important.

---

# 📊 Model Results

## LSTM Results

### Training Performance

| Metric      | Result |
| ----------- | -----: |
| Loss        | 0.4209 |
| Accuracy    | 92.28% |
| Macro F1    | 0.8264 |
| Weighted F1 | 0.9331 |

### Training Confusion Matrix

```text
[[686   0   2   0  36   0   0   0   0]
 [  4 547  11   9   1   0   0   1   0]
 [ 17   0 200  33   0   1   0   2   0]
 [  0   0   0  22   0   0   0   0   0]
 [  7   0   0   0  13   0   0   0   0]
 [  1   0   1   0   0  24   0   0   0]
 [  1   0   0   0   2   0  17   0   0]
 [  0   0   0   0   0   0   0  17   0]
 [  0   0   0   0   0   0   0   0  17]]
```

### Test Performance

| Metric      |     Result |
| ----------- | ---------: |
| Loss        |     0.7170 |
| Accuracy    |     86.16% |
| Macro F1    | **0.6875** |
| Weighted F1 |     0.8858 |

### Test Classification Report

| Category                  | Precision | Recall | F1-Score | Support |
| ------------------------- | --------: | -----: | -------: | ------: |
| Safe                      |      0.93 |   0.86 |     0.89 |     182 |
| Violent Crimes            |      0.99 |   0.97 |     0.98 |     144 |
| Non-Violent Crimes        |      0.92 |   0.70 |     0.79 |      63 |
| unsafe                    |      0.28 |   0.83 |     0.42 |       6 |
| Unknown S-Type            |      0.08 |   0.40 |     0.13 |       5 |
| Sex-Related Crimes        |      1.00 |   0.50 |     0.67 |       6 |
| Suicide & Self-Harm       |      0.57 |   0.80 |     0.67 |       5 |
| Elections                 |      0.80 |   1.00 |     0.89 |       4 |
| Child Sexual Exploitation |      0.75 |   0.75 |     0.75 |       4 |

The LSTM achieved strong performance on the majority classes, particularly **Violent Crimes** and **Safe**.

However, performance was lower on minority classes such as **Unknown S-Type**, **unsafe**, and **Sex-Related Crimes**.

The difference between training and test performance indicates some degree of overfitting, especially for the minority classes.

---

# RNN Results

### Training Performance

| Metric      | Result |
| ----------- | -----: |
| Loss        | 1.3483 |
| Accuracy    | 98.80% |
| Macro F1    | 0.9436 |
| Weighted F1 | 0.9874 |

### Training Confusion Matrix

```text
[[724   0   0   0   0   0   0   0   0]
 [  2 566   3   1   0   0   0   1   0]
 [  2   0 251   0   0   0   0   0   0]
 [  0   0   1  21   0   0   0   0   0]
 [  5   0   1   0  11   0   3   0   0]
 [  0   0   0   0   0  26   0   0   0]
 [  0   0   0   0   1   0  19   0   0]
 [  0   0   0   0   0   0   0  17   0]
 [  0   0   0   0   0   0   0   0  17]]
```

### Test Performance

| Metric      |     Result |
| ----------- | ---------: |
| Loss        |     1.7150 |
| Accuracy    | **89.74%** |
| Macro F1    |     0.6189 |
| Weighted F1 | **0.8934** |

### Test Confusion Matrix

```text
[[174   2   1   0   1   0   4   0   0]
 [  0 140   2   1   0   0   1   0   0]
 [  7   2  50   3   0   0   1   0   0]
 [  1   0   4   1   0   0   0   0   0]
 [  5   0   0   0   0   0   0   0   0]
 [  1   1   0   0   0   3   1   0   0]
 [  2   0   1   0   0   0   2   0   0]
 [  1   0   0   0   0   0   0   3   0]
 [  0   0   1   0   0   0   0   0   3]]
```

The RNN achieved a higher test accuracy than the LSTM.

However, the RNN achieved a lower Macro F1-score, which indicates that its performance is less balanced across the nine classes.

The model performs very well on the majority classes but struggles with several minority classes, especially **Unknown S-Type**, **unsafe**, and **Suicide & Self-Harm**.

The large difference between training Macro F1 (**0.9436**) and test Macro F1 (**0.6189**) also indicates noticeable overfitting.

---

# 🔍 Model Comparison

| Model    | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| -------- | ------------: | ------------: | ---------------: |
| **RNN**  |    **89.74%** |        0.6189 |       **89.34%** |
| **LSTM** |        86.16% |    **0.6875** |           88.58% |

### Analysis

The results show that **accuracy alone is not sufficient** for evaluating this classification task because the dataset is highly imbalanced.

The RNN achieved:

* Higher test accuracy
* Higher weighted F1-score

The LSTM achieved:

* Higher test Macro F1-score
* Better balance across the nine classes

Therefore, the results can be summarized as:

```text
RNN:
Better overall accuracy
Better majority-class performance

LSTM:
Better Macro F1
Better balance across classes
```

Since the dataset is highly imbalanced, **Macro F1 is considered an important metric when comparing the two models**.

---

## ⚠️ Minority-Class Challenge

The minority classes contain very few test samples.

For example:

| Category                  | Test Support |
| ------------------------- | -----------: |
| unsafe                    |            6 |
| Unknown S-Type            |            5 |
| Sex-Related Crimes        |            6 |
| Suicide & Self-Harm       |            5 |
| Elections                 |            4 |
| Child Sexual Exploitation |            4 |

Because these classes have very small test sets, a small number of incorrect predictions can significantly affect their precision, recall, and F1-score.

This is an important limitation of the current dataset and should be considered when interpreting the results.

---

# 🏗️ Application Architecture

The trained models were integrated into a complete application.

```text
                 Streamlit Frontend
                         ↓
                      Backend
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
        Image Captioning        Model Selection
              ↓                ┌────┴────┐
          BLIP Caption         ↓         ↓
              ↓               RNN       LSTM
              └───────────────┬─────────┘
                              ↓
                         Prediction
                              ↓
                       SQLite Database
```

---

## 🌐 Streamlit Application

A **Streamlit web application** was developed to provide an interactive interface for the classifier.

The application allows the user to:

* Enter text
* Upload an image
* Use text and image together
* Select the classification model
* Choose between **RNN and LSTM**
* Generate an image caption
* View the predicted category
* View prediction confidence
* Store the prediction in SQLite

The interface dynamically selects the model based on the user's choice.

Example:

```text
Model:
○ RNN
○ LSTM

Text:
[ User input ]

Image:
[ Optional image upload ]

[ Predict ]
```

The result displays information such as:

```text
Selected Model: LSTM
Predicted Category: Safe
Confidence: 0.91
Image Caption: "a person standing outside"
```

---

# 🔧 Backend

A backend layer was implemented to connect the Streamlit interface with the machine learning components.

The backend is responsible for:

* Receiving user input
* Handling text input
* Processing uploaded images
* Generating image captions
* Combining text and image captions
* Applying the same preprocessing used during training
* Selecting the RNN or LSTM model
* Running inference
* Returning the prediction and confidence
* Saving prediction results to SQLite

The general backend flow is:

```text
Streamlit
    ↓
Backend
    ↓
Input Processing
    ↓
Image Captioning (optional)
    ↓
Text + Caption
    ↓
Preprocessing
    ↓
Selected Model
    ↓
Prediction
    ↓
SQLite
```

---

# 🗄️ SQLite Database

A local **SQLite database** was added to store prediction history.

The database stores information including:

* Input type
* Input text
* Selected model
* Image path
* Generated image caption
* Predicted label
* Prediction confidence
* Prediction timestamp

Example database structure:

```text
predictions
├── id
├── input_type
├── input_text
├── model_type
├── image_path
├── image_caption
├── predicted_label
├── confidence
└── created_at
```

SQLite was selected because it is lightweight, local, and does not require a separate database server.

---

# 🔀 Application Workflow

## Text Only

```text
User Text
    ↓
Preprocessing
    ↓
Tokenization
    ↓
Sequence Conversion
    ↓
Padding
    ↓
RNN / LSTM
    ↓
9-Class Prediction
    ↓
Confidence
    ↓
SQLite
```

---

## Image Only

```text
Image
  ↓
BLIP Image Captioning
  ↓
Generated Caption
  ↓
Preprocessing
  ↓
Tokenization
  ↓
Sequence Conversion
  ↓
Padding
  ↓
RNN / LSTM
  ↓
9-Class Prediction
  ↓
Confidence
  ↓
SQLite
```

---

## Text + Image

```text
User Text ──────────────────┐
                            ↓
Image → BLIP → Caption → Combined Text
                              ↓
                         Preprocessing
                              ↓
                         Tokenization
                              ↓
                       Sequence Conversion
                              ↓
                            Padding
                              ↓
                         RNN / LSTM
                              ↓
                       9-Class Prediction
                              ↓
                          Confidence
                              ↓
                           SQLite
```

---

# 🖥️ Streamlit User Interface

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

After clicking the prediction button, the application processes the input and displays:

```text
Predicted Category
Confidence
Selected Model
Generated Image Caption (if an image was provided)
```

The prediction is also stored in the SQLite database.

---

# 🛠️ Technologies

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

# 📁 Project Structure

```text
project/
│
├── train_true_labels.csv
├── data_preprocessing.py
├── backend.py
├── database.py
├── app.py
│
├── ImageCaptioner/
│   └── ...
│
├── RNN/
│   └── rnn.py
│
├── LSTM/
│   └── train_lstm.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone <repository-url>
cd <project-directory>
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser.

---

# 💡 Key Takeaway

This project demonstrates a complete **9-class toxicity classification system using RNN and LSTM**, starting from data preparation and model training and extending to a functional web application.

The project includes:

* Data inspection and cleaning
* Manual label correction
* Duplicate and data quality analysis
* Class distribution analysis
* Minority-class augmentation
* Custom text preprocessing
* Tokenization
* Vocabulary building
* Text-to-sequence conversion
* Padding
* Embedding
* RNN classification
* LSTM classification
* 9-class toxicity prediction
* Image captioning using BLIP
* Text and image input support
* Backend integration
* SQLite database storage
* Streamlit web application
* Dynamic RNN/LSTM model selection
* Prediction confidence
* Prediction history storage

The final system allows users to provide **text, images, or both**, select either an **RNN or LSTM model**, and receive a classification prediction through an interactive Streamlit application.

The evaluation also demonstrates an important machine learning lesson: for highly imbalanced multi-class datasets, **accuracy alone does not fully describe model performance**. Macro F1 provides additional insight into how well the model performs across both majority and minority classes.
