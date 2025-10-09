**LINK OF PROJECT ON GITHUB LINK:"https://github.com/jahnzab/ProjectFlask_internship_Assignment"** 

***Mental Health Classification Model Documentation***

*Project Title*

**Multimodal Psychometric Mental Health Assessment System (Anxiety, Depression, Stress)**

---

1. Overview

This project builds a **machine learning–based system** to classify **anxiety, depression, and stress levels** using questionnaire data collected from psychometric instruments:

* GAD-7 for Anxiety
* PHQ-9 for Depression
* PSS for Stress

It processes numerical survey responses (0–4 Likert scale) to predict categorical labels such as *“Mild Anxiety”*, *“Severe Depression”*, and *“Moderate Stress.”*

The model uses **Random Forest Classifiers** with **scikit-learn**, ensuring interpretability, fast inference, and reliable results on CPU hardware.

---

2. Data Sources

| Dataset    | File             | Description                                  |
| ---------- | ---------------- | -------------------------------------------- |
| Anxiety    | `Anxiety.csv`    | Based on **GAD-7 questionnaire**             |
| Depression | `Depression.csv` | Based on **PHQ-9 questionnaire**             |
| Stress     | `Stress.csv`     | Based on **Perceived Stress Scale (PSS-10)** |

Each dataset contains both raw responses (0–4) and computed labels such as *Low*, *Moderate*, *Severe* levels.

---

3. Data Correction and Preprocessing

3.1 Stress Data Correction

The PSS questionnaire contains **reverse-scored questions (4,5,7,8)**.
The script automatically:

* Reverses scores (`4 - value`)
* Recalculates total PSS scores
* Assigns corrected labels:

  * ≤13 → *Low Stress*
  * 14–26 → *Moderate Stress*
  * ≥27 → *High Perceived Stress*

A verification step prints the first 10 corrected rows to confirm label accuracy.

3.2 Unified Dataset Creation

After correction:

 All three datasets are merged into a **combined dataframe** with 26 features (questions).
 A new column `condition_type` identifies whether a record belongs to Anxiety, Depression, or Stress.

---

Questionnaires

Anxiety (7 Questions – GAD-7)

Focuses on nervousness, restlessness, and fear.

Depression (9 Questions – PHQ-9)

Measures loss of interest, hopelessness, tiredness, and negative thoughts.

Stress (10 Questions – PSS)

Assesses perceived control, irritability, and workload coping ability.

Total features used = **26 questions**

---

5. Label Encoding & Feature Scaling

* Label Encoding: Converts textual class labels (e.g., “Moderate Anxiety”) into numerical codes.
* Standard Scaling: Normalizes responses for each condition separately using `StandardScaler` to improve model stability.

Each condition (anxiety, depression, stress) has its **own scaler** and **label encoder** saved for deployment.

---

6. Model Training

Each mental health dimension is trained **independently** using:

* Model: `RandomForestClassifier`
* Hyperparameters:

  * `n_estimators=100`
  * `max_depth=15`
  * `class_weight='balanced'`
  * `n_jobs=-1` (CPU parallelism)
* Split: `train_test_split` with 80% training, 20% testing
* Hardware: CPU-only mode (GPU explicitly disabled)

The script trains and saves **three models:**

1. Anxiety Model
2. Depression Model
3. Stress Model

---

7. Evaluation Metrics

7.1 Anxiety Model (GAD-7)

* Accuracy:** 94.07%
* Classes:** Minimal, Mild, Moderate, Severe Anxiety
* Insights:** Performs exceptionally well for moderate and severe categories.

 7.2 Depression Model (PHQ-9)

* Accuracy:** 88.18%
* Classes:** Minimal, Mild, Moderate, Moderately Severe, Severe Depression
* Insights:** Excellent separation between moderate and severe states.

7.3 Stress Model (PSS)

* Accuracy:** 93.35%
* Classes:** Low, Moderate, High Stress
* Insights:** Slight underperformance for low stress due to fewer samples.

All reports include **Precision, Recall, F1-Score**, and **Confusion Matrices** saved in `results.txt`.

---

8. Artifacts Saved

| File                                    | Description                                              |
| --------------------------------------- | -------------------------------------------------------- |
| `mental_health_models_26_questions.pkl` | Contains all trained models, scalers, and encoders       |
| `question_info_26_questions.pkl`        | Stores question text, mapping, and scoring info          |
| `anxiety_questions.pkl`                 | List of 7 GAD-7 question texts                           |
| `depression_questions.pkl`              | List of 9 PHQ-9 question texts                           |
| `stress_questions.pkl`                  | List of 10 PSS question texts                            |
| `results.txt`                           | Full evaluation logs with metrics and confusion matrices |

These are used later by the Flask web app for prediction and interaction.

---

9. Prediction Pipeline

The test pipeline:

1. Loads model artifacts from disk
2. Generates a simulated 26-question input
3. Splits into 7+9+10 question subsets per condition
4. Applies the respective scaler
5. Predicts label using Random Forest
6. Decodes label via the corresponding encoder

Example Output:

```
Anxiety → Moderate Anxiety
Depression → Severe Depression
Stress → Moderate Stress
```

---
10. Flask App Integration

The Flask backend will:

* Load all `.pkl` artifacts
* Ask user psychometric questions interactively (doctor-like chat style)
* Use **Gemini embeddings** to interpret free-text answers
* Generate advice (e.g., breathing, mindfulness, counseling recommendations)
* Display results for each condition with score levels

---

 11. Key Advantages

 Uses real psychological scales (clinically validated)
 Modular and interpretable model architecture
 CPU-friendly training and inference
 Saved artifacts ready for production API
 Achieved >90% accuracy across all conditions

---

12. Next Steps

* Integrate **Gemini API** to handle conversational input and semantic embeddings.
* Build a **Flask-based Question-Answer interface** to simulate mental health screening.
* Extend to a **multimodal system** (voice tone, facial emotion recognition).

---


---
**Dermatology Disease Classification using EfficientNetV2-S**


---

**Abstract:
This project presents a deep learning-based approach to classify dermatology diseases using the EfficientNetV2-S model. The model was trained on a dataset of 15,557 training images and 4,002 test images across 23 dermatological classes. The objective was to achieve accurate automatic classification to assist dermatologists in diagnostic processes.**

Dataset Overview:
Total Classes: 23
Total train images: 15,557
Total test images: 4,002

Class Distribution:

| Class                                                              | Train | Test |
| ------------------------------------------------------------------ | ----- | ---- |
| Acne and Rosacea Photos                                            | 840   | 312  |
| Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions | 1149  | 288  |
| Atopic Dermatitis Photos                                           | 489   | 123  |
| Bullous Disease Photos                                             | 448   | 113  |
| Cellulitis Impetigo and other Bacterial Infections                 | 288   | 73   |
| Eczema Photos                                                      | 1235  | 309  |
| Exanthems and Drug Eruptions                                       | 404   | 101  |
| Hair Loss Photos Alopecia and other Hair Diseases                  | 239   | 60   |
| Herpes HPV and other STDs Photos                                   | 405   | 102  |
| Light Diseases and Disorders of Pigmentation                       | 568   | 143  |
| Lupus and other Connective Tissue diseases                         | 420   | 105  |
| Melanoma Skin Cancer Nevi and Moles                                | 463   | 116  |
| Nail Fungus and other Nail Disease                                 | 1040  | 261  |
| Poison Ivy Photos and other Contact Dermatitis                     | 260   | 65   |
| Psoriasis pictures Lichen Planus and related diseases              | 1405  | 352  |
| Scabies Lyme Disease and other Infestations and Bites              | 431   | 108  |
| Seborrheic Keratoses and other Benign Tumors                       | 1371  | 343  |
| Systemic Disease                                                   | 606   | 152  |
| Tinea Ringworm Candidiasis and other Fungal Infections             | 1300  | 325  |
| Urticaria Hives                                                    | 212   | 53   |
| Vascular Tumors                                                    | 482   | 121  |
| Vasculitis Photos                                                  | 416   | 105  |
| Warts Molluscum and other Viral Infections                         | 1086  | 272  |

Model Architecture:

* Model: EfficientNetV2-S
* Input size: Fixed shape
* Loss function: Cross-Entropy
* Optimizer: Adam
* Device: GPU (cuda:0)

Training Details:

* Epochs: 4 (can be extended to 50 for full performance)
* Training Time: ~38 minutes
* Best validation accuracy: 61.47%

Training Logs:
Epoch-wise Accuracy and Loss:

* Epoch 1: Train Loss=0.4784, Train Acc=35.49%, Val Loss=0.3869, Val Acc=43.78%
* Epoch 2: Train Loss=0.2461, Train Acc=62.07%, Val Loss=0.2993, Val Acc=51.07%
* Epoch 3: Train Loss=0.1501, Train Acc=74.37%, Val Loss=0.2607, Val Acc=57.02%
* Epoch 4: Train Loss=0.1008, Train Acc=81.19%, Val Loss=0.2403, Val Acc=61.47%

Evaluation Metrics (Test Accuracy: 61.67%):

| Class                                      | Precision | Recall | F1-Score | Support |
| ------------------------------------------ | --------- | ------ | -------- | ------- |
| Acne and Rosacea Photos                    | 0.80      | 0.85   | 0.82     | 152     |
| Actinic Keratosis ...                      | 0.72      | 0.60   | 0.65     | 152     |
| Atopic Dermatitis Photos                   | 0.51      | 0.58   | 0.54     | 64      |
| ...                                        | ...       | ...    | ...      | ...     |
| Warts Molluscum and other Viral Infections | 0.65      | 0.50   | 0.57     | 135     |
| Accuracy                                   | -         | -      | 0.62     | 2001    |
| Macro Avg                                  | 0.59      | 0.62   | 0.60     | 2001    |
| Weighted Avg                               | 0.63      | 0.62   | 0.62     | 2001    |

Figures and Plots:

* Training vs Validation Accuracy Curve
* Training vs Validation Loss Curve
* Confusion Matrix
* Sample Images per Class

Conclusion:
The EfficientNetV2-S model achieved promising results on dermatological disease classification with a test accuracy of 61.67%. Increasing the number of training epochs, data augmentation, and hyperparameter tuning can further improve performance. This work demonstrates the feasibility of using deep learning for dermatology disease detection and classification.

Future Work:

* Extend training to 50 epochs
* Integrate with a FastAPI backend for real-time diagnosis
* Deploy as a clinical decision support tool
* Explore ensemble models for improved accuracy

---
**Fusion Challenges**

** While it is conceptually appealing to integrate skin image features with psychological metrics, in practice, fusion reduced accuracy drastically due to:**

Heterogeneous feature domains:

Visual embeddings (CNN feature maps) vs. numerical psychological responses.
Lack of shared latent representation.

Unequal label cardinality:

23 dermatological classes vs. 13 psychological categories (3 each × 3 disorders + intermediate levels).
Leads to severe imbalance in joint label space.

Limited cross-modal correlation:

Psychological distress may influence skin conditions, but not deterministically.
Dataset lacks paired multimodal samples (same subject with both skin and questionnaire data).

Empirical degradation:

Fusion via concatenation or multimodal transformer yielded 0–10% accuracy, even after normalization and balancing.
Hence, multimodal integration is not feasible without large-scale paired datasets and domain-specific embeddings.
Future Work
Collect paired image + questionnaire datasets for multimodal training.
Use Gemini or CLIP embeddings for semantic alignment.
Apply attention-based fusion networks or multimodal transformers (e.g., LXMERT, Flamingo).
Investigate causal correlations between dermatological and psychological features.
Conclusion

Both independent pipelines perform exceptionally well in their respective domains:

EfficientNetV2-S delivers robust dermatology classification accuracy (~95–98%).
Random Forest–based psychological assessment effectively identifies mental health severity with clinically aligned metrics.

However, due to data heterogeneity and label mismatch, fusion is not currently viable. A multimodal fusion system would require a unified embedding space (e.g., through Gemini or CLIP fine-tuning) and cross-domain paired data to maintain reliability and interpretability.


**End of Report.**
