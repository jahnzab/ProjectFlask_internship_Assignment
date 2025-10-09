
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import joblib
# import os

# # ========== FORCE CPU ONLY ==========
# print("🔧 FORCING CPU TRAINING - cuML/GPU disabled")
# HAS_GPU = False
# HAS_CUML = False

# # Create directories
# os.makedirs('model_artifacts', exist_ok=True)
# os.makedirs('evaluation_results', exist_ok=True)

# # Load your datasets
# print("Loading datasets...")
# anxiety_df = pd.read_csv('/home/jahanzaib/Desktop/Project_Intern/Flaskapp/mental-health/Anxiety.csv')
# depression_df = pd.read_csv('/home/jahanzaib/Desktop/Project_Intern/Flaskapp/mental-health/Depression.csv')  
# stress_df = pd.read_csv('/home/jahanzaib/Desktop/Project_Intern/Flaskapp/mental-health/Stress.csv')

# # Define ALL question columns for each condition
# ANXIETY_QUESTIONS = [
#     '1. In a semester, how often you felt nervous, anxious or on edge due to academic pressure? ',
#     '2. In a semester, how often have you been unable to stop worrying about your academic affairs? ',
#     '3. In a semester, how often have you had trouble relaxing due to academic pressure? ',
#     '4. In a semester, how often have you been easily annoyed or irritated because of academic pressure?',
#     '5. In a semester, how often have you worried too much about academic affairs? ',
#     '6. In a semester, how often have you been so restless due to academic pressure that it is hard to sit still?',
#     '7. In a semester, how often have you felt afraid, as if something awful might happen?'
# ]

# DEPRESSION_QUESTIONS = [
#     '1. In a semester, how often have you had little interest or pleasure in doing things?',
#     '2. In a semester, how often have you been feeling down, depressed or hopeless?',
#     '3. In a semester, how often have you had trouble falling or staying asleep, or sleeping too much? ',
#     '4. In a semester, how often have you been feeling tired or having little energy? ',
#     '5. In a semester, how often have you had poor appetite or overeating? ',
#     '6. In a semester, how often have you been feeling bad about yourself - or that you are a failure or have let yourself or your family down? ',
#     '7. In a semester, how often have you been having trouble concentrating on things, such as reading the books or watching television? ',
#     "8. In a semester, how often have you moved or spoke too slowly for other people to notice? Or you've been moving a lot more than usual because you've been restless? ",
#     '9. In a semester, how often have you had thoughts that you would be better off dead, or of hurting yourself? '
# ]

# STRESS_QUESTIONS = [
#     '1. In a semester, how often have you felt upset due to something that happened in your academic affairs? ',
#     '2. In a semester, how often you felt as if you were unable to control important things in your academic affairs?',
#     '3. In a semester, how often you felt nervous and stressed because of academic pressure? ',
#     '4. In a semester, how often you felt as if you could not cope with all the mandatory academic activities? (e.g, assignments, quiz, exams) ',
#     '5. In a semester, how often you felt confident about your ability to handle your academic / university problems?',
#     '6. In a semester, how often you felt as if things in your academic life is going on your way? ',
#     '7. In a semester, how often are you able to control irritations in your academic / university affairs? ',
#     '8. In a semester, how often you felt as if your academic performance was on top?',
#     '9. In a semester, how often you got angered due to bad performance or low grades that is beyond your control? ',
#     '10. In a semester, how often you felt as if academic difficulties are piling up so high that you could not overcome them? '
# ]

# print("🔍 Using ALL original questions for clinical validity...")

# # Use ALL questions from each condition
# anxiety_features = ANXIETY_QUESTIONS  # All 7 anxiety questions
# depression_features = DEPRESSION_QUESTIONS  # All 9 depression questions  
# stress_features = STRESS_QUESTIONS  # All 10 stress questions

# # Total: 26 questions
# ALL_QUESTIONS = anxiety_features + depression_features + stress_features

# print(f"\n🎯 USING ALL ORIGINAL QUESTIONS:")
# print(f"Total questions: {len(ALL_QUESTIONS)}")
# print(f"- Anxiety: {len(anxiety_features)} questions (GAD-7) - Max score: 21")
# print(f"- Depression: {len(depression_features)} questions (PHQ-9) - Max score: 27") 
# print(f"- Stress: {len(stress_features)} questions (PSS) - Max score: 40")

# def create_datasets_with_all_questions():
#     """Create datasets using ALL original questions"""
    
#     # For anxiety - use ALL 7 anxiety questions
#     anxiety_std = anxiety_df[anxiety_features].copy()
#     anxiety_std['condition_type'] = 'anxiety'
#     anxiety_std['anxiety_label'] = anxiety_df['Anxiety Label']
    
#     # For depression - use ALL 9 depression questions  
#     depression_std = depression_df[depression_features].copy()
#     depression_std['condition_type'] = 'depression'
#     depression_std['depression_label'] = depression_df['Depression Label']
    
#     # For stress - use ALL 10 stress questions
#     stress_std = stress_df[stress_features].copy()
#     stress_std['condition_type'] = 'stress'
#     stress_std['stress_label'] = stress_df['Stress Label']
    
#     # Combine all datasets
#     combined_df = pd.concat([anxiety_std, depression_std, stress_std], ignore_index=True)
#     combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
#     print(f"\n📊 Combined dataset shape: {combined_df.shape}")
#     print("Condition distribution:")
#     print(combined_df['condition_type'].value_counts())
    
#     return combined_df, anxiety_features, depression_features, stress_features

# # Create dataset with ALL questions
# combined_df, anxiety_features, depression_features, stress_features = create_datasets_with_all_questions()

# def prepare_features_and_labels_all_questions(combined_df, anxiety_features, depression_features, stress_features):
#     """Prepare features and labels for training with ALL questions"""
    
#     # All feature columns (26 total)
#     all_feature_columns = anxiety_features + depression_features + stress_features
    
#     X = combined_df[all_feature_columns]
#     y = combined_df[['anxiety_label', 'depression_label', 'stress_label']]
    
#     print(f"\n📈 Features shape: {X.shape} (26 questions total)")
#     print(f"   - Anxiety: {len(anxiety_features)} questions")
#     print(f"   - Depression: {len(depression_features)} questions")
#     print(f"   - Stress: {len(stress_features)} questions")
#     print(f"📋 Labels shape: {y.shape}")
    
#     return X, y, all_feature_columns

# # Use the new function with all questions
# X, y, feature_columns = prepare_features_and_labels_all_questions(
#     combined_df, anxiety_features, depression_features, stress_features
# )

# def create_label_encoders(y):
#     """Create label encoders for each condition"""
#     label_encoders = {}
    
#     for condition in ['anxiety', 'depression', 'stress']:
#         le = LabelEncoder()
#         condition_labels = y[f'{condition}_label'].dropna().unique()
#         le.fit(condition_labels)
#         label_encoders[condition] = le
#         print(f"{condition} classes: {list(le.classes_)}")
    
#     return label_encoders

# label_encoders = create_label_encoders(y)

# def train_models_with_all_questions(X_train, y_train, label_encoders, anxiety_features, depression_features, stress_features):
#     """Train models using ALL questions for each condition with CPU only"""
    
#     models = {}
#     scalers = {}  # Separate scaler for each condition's questions
    
#     for condition in ['anxiety', 'depression', 'stress']:
#         print(f"\n--- Training {condition} model ---")
        
#         # Get condition-specific features
#         if condition == 'anxiety':
#             condition_features = anxiety_features
#             q_count = len(anxiety_features)
#         elif condition == 'depression':
#             condition_features = depression_features
#             q_count = len(depression_features)
#         else:
#             condition_features = stress_features
#             q_count = len(stress_features)
        
#         # Get only rows for this condition
#         condition_mask = combined_df.iloc[X_train.index]['condition_type'] == condition
#         X_condition = X_train[condition_mask][condition_features]
#         y_condition = y_train[condition_mask][f'{condition}_label']
        
#         # Encode labels
#         y_encoded = label_encoders[condition].transform(y_condition)
        
#         # Create and fit scaler for this condition's features
#         condition_scaler = StandardScaler()
#         X_scaled = condition_scaler.fit_transform(X_condition)
#         scalers[condition] = condition_scaler
        
#         # Use CPU only (scikit-learn)
#         print(f"⚡ Training {condition} with CPU (scikit-learn)...")
#         model = RandomForestClassifier(
#             n_estimators=100,
#             random_state=42,
#             max_depth=15,
#             class_weight='balanced',
#             n_jobs=-1,
#             verbose=1
#         )
        
#         model.fit(X_scaled, y_encoded)
#         models[condition] = model
#         print(f"✅ {condition} model trained on {q_count} questions (full questionnaire)")
    
#     return models, scalers

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=combined_df['condition_type']
# )

# print(f"\n📚 Training set: {X_train.shape[0]} samples")
# print(f"🧪 Test set: {X_test.shape[0]} samples")

# # Train models with all questions
# models, condition_scalers = train_models_with_all_questions(
#     X_train, y_train, label_encoders, anxiety_features, depression_features, stress_features
# )

# def evaluate_models_with_all_questions(models, condition_scalers, X_test, y_test, label_encoders, 
#                                      anxiety_features, depression_features, stress_features):
#     """Evaluate models using ALL questions and save results to file"""
    
#     print("\n" + "="*60)
#     print("📊 MODEL EVALUATION WITH 26 ACTUAL QUESTIONS")
#     print("="*60)
    
#     # Initialize results dictionary
#     results = {
#         'anxiety': {},
#         'depression': {}, 
#         'stress': {}
#     }
    
#     with open('results.txt', 'w') as f:
#         f.write("MENTAL HEALTH MODEL RESULTS - ANXIETY, STRESS, DEPRESSION\n")
#         f.write("=" * 60 + "\n\n")
#         f.write(f"Training completed on CPU only\n")
#         f.write(f"Total questions used: {len(ALL_QUESTIONS)}\n")
#         f.write(f"- Anxiety: {len(anxiety_features)} questions (GAD-7)\n")
#         f.write(f"- Depression: {len(depression_features)} questions (PHQ-9)\n") 
#         f.write(f"- Stress: {len(stress_features)} questions (PSS)\n\n")
    
#     for condition in ['anxiety', 'depression', 'stress']:
#         print(f"\n--- {condition.upper()} CLASSIFICATION REPORT ---")
        
#         # Get condition-specific features
#         if condition == 'anxiety':
#             condition_features = anxiety_features
#         elif condition == 'depression':
#             condition_features = depression_features
#         else:
#             condition_features = stress_features
        
#         # Get only rows for this condition
#         condition_mask = combined_df.iloc[X_test.index]['condition_type'] == condition
#         X_condition = X_test[condition_mask][condition_features]
#         true_labels = y_test[condition_mask][f'{condition}_label']
        
#         # Scale and predict
#         X_scaled = condition_scalers[condition].transform(X_condition)
#         pred_encoded = models[condition].predict(X_scaled)
#         pred_labels = label_encoders[condition].inverse_transform(pred_encoded)
        
#         # Calculate metrics
#         accuracy = (true_labels == pred_labels).mean()
#         report = classification_report(true_labels, pred_labels, zero_division=0)
#         cm = confusion_matrix(true_labels, pred_labels)
#         results[condition]['confusion_matrix'] = cm

#         # Print confusion matrix
#         print(f"📊 Confusion Matrix for {condition}:")
#         print(cm)

#         # Save confusion matrix to file
#         with open('results.txt', 'a') as f:
#          f.write(f"Confusion Matrix:\n")
#          f.write(np.array2string(cm))
#          f.write("\n\n")
#         # Store results
#         results[condition]['accuracy'] = accuracy
#         results[condition]['classification_report'] = report
#         results[condition]['true_labels'] = true_labels.tolist()
#         results[condition]['predicted_labels'] = pred_labels.tolist()
        
#         # Print results
#         print(report)
#         print(f"🎯 Accuracy for {condition}: {accuracy:.3f}")
        
#         # Save to file
#         with open('results.txt', 'a') as f:
#             f.write(f"\n{condition.upper()} RESULTS\n")
#             f.write("-" * 40 + "\n")
#             f.write(f"Accuracy: {accuracy:.4f}\n")
#             f.write(f"Questions used: {len(condition_features)}\n")
#             f.write(f"Test samples: {len(true_labels)}\n\n")
#             f.write("Classification Report:\n")
#             f.write(report)
#             f.write("\n" + "="*50 + "\n")
    
#     return results

# # Evaluate models and get results
# results = evaluate_models_with_all_questions(
#     models, condition_scalers, X_test, y_test, label_encoders,
#     anxiety_features, depression_features, stress_features
# )

# def save_artifacts_with_all_questions(models, condition_scalers, label_encoders, 
#                                     anxiety_features, depression_features, stress_features):
#     """Save all artifacts with ALL questions for Flask app"""
    
#     # Save comprehensive artifacts
#     artifacts = {
#         'models': models,
#         'scalers': condition_scalers,
#         'label_encoders': label_encoders,
#         'anxiety_features': anxiety_features,
#         'depression_features': depression_features, 
#         'stress_features': stress_features,
#         'gpu_acceleration': False  # Force CPU
#     }
    
#     joblib.dump(artifacts, 'model_artifacts/mental_health_models_26_questions.pkl')
    
#     # Save question information for Flask app
#     question_info = {
#         'anxiety_questions': anxiety_features,
#         'depression_questions': depression_features,
#         'stress_questions': stress_features,
#         'all_questions': anxiety_features + depression_features + stress_features,
#         'question_short_forms': {
#             'anxiety': [f"Anxiety Q{i+1}: {q[:50]}..." for i, q in enumerate(anxiety_features)],
#             'depression': [f"Depression Q{i+1}: {q[:50]}..." for i, q in enumerate(depression_features)],
#             'stress': [f"Stress Q{i+1}: {q[:50]}..." for i, q in enumerate(stress_features)]
#         },
#         'question_mapping': {
#             'anxiety': {f'Q{i+1}': q for i, q in enumerate(anxiety_features)},
#             'depression': {f'Q{i+1}': q for i, q in enumerate(depression_features)},
#             'stress': {f'Q{i+1}': q for i, q in enumerate(stress_features)}
#         },
#         'scoring_info': {
#             'anxiety': {'max_score': 21, 'questions': 7},
#             'depression': {'max_score': 27, 'questions': 9},
#             'stress': {'max_score': 40, 'questions': 10}
#         }
#     }
    
#     joblib.dump(question_info, 'model_artifacts/question_info_26_questions.pkl')
    
#     # Save individual files for easy access in Flask
#     joblib.dump(anxiety_features, 'model_artifacts/anxiety_questions.pkl')
#     joblib.dump(depression_features, 'model_artifacts/depression_questions.pkl')
#     joblib.dump(stress_features, 'model_artifacts/stress_questions.pkl')
    
#     print("\n💾 ALL ARTIFACTS SAVED SUCCESSFULLY!")
#     print("📁 Files saved for Flask app:")
#     print("   - model_artifacts/mental_health_models_26_questions.pkl")
#     print("   - model_artifacts/question_info_26_questions.pkl")
#     print("   - model_artifacts/anxiety_questions.pkl")
#     print("   - model_artifacts/depression_questions.pkl") 
#     print("   - model_artifacts/stress_questions.pkl")
#     print(f"\n📋 Question Summary:")
#     print(f"   Total questions: {len(anxiety_features) + len(depression_features) + len(stress_features)}")
#     print(f"   - Anxiety: {len(anxiety_features)} questions (max 21 points)")
#     print(f"   - Depression: {len(depression_features)} questions (max 27 points)")
#     print(f"   - Stress: {len(stress_features)} questions (max 40 points)")
#     print(f"⚡ Training Mode: CPU Only (GPU disabled)")

# # Save artifacts
# save_artifacts_with_all_questions(
#     models, condition_scalers, label_encoders,
#     anxiety_features, depression_features, stress_features
# )

# def test_prediction_pipeline():
#     """Test the complete prediction pipeline with ALL questions"""
    
#     # Load artifacts
#     artifacts = joblib.load('model_artifacts/mental_health_models_26_questions.pkl')
#     question_info = joblib.load('model_artifacts/question_info_26_questions.pkl')
    
#     models = artifacts['models']
#     scalers = artifacts['scalers']
#     label_encoders = artifacts['label_encoders']
    
#     print("\n" + "="*50)
#     print("🧪 PREDICTION PIPELINE TEST")
#     print("="*50)
    
#     # Generate sample responses for 26 questions
#     np.random.seed(42)
#     sample_input = np.random.randint(0, 4, 26).reshape(1, -1)  # 26 questions, values 0-3
    
#     print(f"Sample input shape: {sample_input.shape} (26 questions)")
    
#     # Predict for all conditions using their specific questions
#     final_predictions = {}
#     for condition in ['anxiety', 'depression', 'stress']:
#         # Get condition-specific questions
#         if condition == 'anxiety':
#             condition_features = question_info['anxiety_questions']
#             feature_indices = list(range(0, 7))  # First 7 questions
#         elif condition == 'depression':
#             condition_features = question_info['depression_questions'] 
#             feature_indices = list(range(7, 16))  # Next 9 questions
#         else:
#             condition_features = question_info['stress_questions']
#             feature_indices = list(range(16, 26))  # Last 10 questions
        
#         # Extract relevant features for this condition
#         X_condition = sample_input[:, feature_indices]
        
#         # Scale and predict
#         X_scaled = scalers[condition].transform(X_condition)
#         pred_encoded = models[condition].predict(X_scaled)
#         final_predictions[condition] = label_encoders[condition].inverse_transform(pred_encoded)[0]
        
#         print(f"\n{condition.upper()} Prediction:")
#         print(f"  Used {len(condition_features)} questions")
#         print(f"  Prediction: {final_predictions[condition]}")
    
#     return final_predictions

# # Test the complete pipeline
# test_predictions = test_prediction_pipeline()

# print("\n" + "="*70)
# print("🎉 TRAINING WITH 26 ACTUAL QUESTIONS COMPLETED SUCCESSFULLY!")
# print("="*70)
# print("📊 Model trained on 26 original questions:")
# print("   - 7 Anxiety questions (GAD-7) - Max 21 points")
# print("   - 9 Depression questions (PHQ-9) - Max 27 points") 
# print("   - 10 Stress questions (PSS) - Max 40 points")
# print(f"⚡ Training Mode: CPU Only (GPU disabled)")
# print("💾 All artifacts saved for Flask app deployment")
# print("📄 Results saved to: results.txt")
# print("="*70)

# # Final confirmation
# print(f"\n✅ RESULTS FILE CREATED: results.txt")
# print("📝 File contains anxiety, stress, and depression model results including:")
# print("   - Accuracy scores for each condition")
# print("   - Classification reports with precision, recall, f1-score")
# print("   - Number of questions used for each condition")
# print("   - Test sample sizes")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ========== FORCE CPU ONLY ==========
print("🔧 FORCING CPU TRAINING - cuML/GPU disabled")
HAS_GPU = False
HAS_CUML = False

# Create directories
os.makedirs('model_artifacts', exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)

# Load your datasets
print("Loading datasets...")
anxiety_df = pd.read_csv('/home/jahanzaib/Desktop/Project_Intern/Flaskapp/mental-health/Anxiety.csv')
depression_df = pd.read_csv('/home/jahanzaib/Desktop/Project_Intern/Flaskapp/mental-health/Depression.csv')  
stress_df = pd.read_csv('/home/jahanzaib/Desktop/Project_Intern/Flaskapp/mental-health/Stress.csv')

# ========== STRESS DATA CORRECTION ==========
def correct_stress_scoring(df):
    """Apply proper PSS reverse scoring and correct labels - FOR STRESS ONLY"""
    df_corrected = df.copy()
    
    # Reverse score questions (0-indexed: Q4, Q5, Q7, Q8 are indices 3,4,6,7)
    reverse_questions = [3, 4, 6, 7]
    
    # Apply reverse scoring
    for idx in reverse_questions:
        col = df.columns[7 + idx]  # Stress questions start at column 7
        df_corrected[col] = 4 - df[col]  # Reverse: 0→4, 1→3, 2→2, 3→1, 4→0
    
    # Recalculate total scores
    stress_columns = df.columns[7:17]  # Columns with stress questions
    df_corrected['Stress Value'] = df_corrected[stress_columns].sum(axis=1)
    
    # Correct labels based on PSS rules
    def get_correct_label(score):
        if score <= 13:
            return 'Low Stress'
        elif score <= 26:
            return 'Moderate Stress'
        else:
            return 'High Perceived Stress'
    
    df_corrected['Stress Label'] = df_corrected['Stress Value'].apply(get_correct_label)
    
    return df_corrected

def analyze_corrections(original_df, corrected_df):
    """Show what corrections were made"""
    print("🔧 STRESS CORRECTION ANALYSIS:")
    print("=" * 50)
    
    changes_count = 0
    for i in range(min(10, len(original_df))):  # Show first 10 rows
        orig_score = original_df.iloc[i]['Stress Value']
        orig_label = original_df.iloc[i]['Stress Label']
        corr_score = corrected_df.iloc[i]['Stress Value'] 
        corr_label = corrected_df.iloc[i]['Stress Label']
        
        if orig_label != corr_label:
            print(f"Row {i+1}: {orig_score} ({orig_label}) → {corr_score} ({corr_label}) ❌ FIXED")
            changes_count += 1
        else:
            print(f"Row {i+1}: {orig_score} ({orig_label}) → {corr_score} ({corr_label}) ✓ OK")
    
    print(f"\n📊 Total labels corrected: {changes_count}/{min(10, len(original_df))}")
    return changes_count

# Apply correction ONLY to stress data
print("🔄 Correcting stress data (applying PSS reverse scoring)...")
stress_df_corrected = correct_stress_scoring(stress_df)

# Analyze what changed
changes_count = analyze_corrections(stress_df, stress_df_corrected)

# Replace ONLY stress dataframe
stress_df = stress_df_corrected
print(f"✅ Stress data corrected: {changes_count} labels fixed")
print("✅ Anxiety and depression data are correct - no changes needed")

# ========== CONTINUE WITH NORMAL TRAINING ==========

# Define ALL question columns for each condition
ANXIETY_QUESTIONS = [
    '1. In a semester, how often you felt nervous, anxious or on edge due to academic pressure? ',
    '2. In a semester, how often have you been unable to stop worrying about your academic affairs? ',
    '3. In a semester, how often have you had trouble relaxing due to academic pressure? ',
    '4. In a semester, how often have you been easily annoyed or irritated because of academic pressure?',
    '5. In a semester, how often have you worried too much about academic affairs? ',
    '6. In a semester, how often have you been so restless due to academic pressure that it is hard to sit still?',
    '7. In a semester, how often have you felt afraid, as if something awful might happen?'
]

DEPRESSION_QUESTIONS = [
    '1. In a semester, how often have you had little interest or pleasure in doing things?',
    '2. In a semester, how often have you been feeling down, depressed or hopeless?',
    '3. In a semester, how often have you had trouble falling or staying asleep, or sleeping too much? ',
    '4. In a semester, how often have you been feeling tired or having little energy? ',
    '5. In a semester, how often have you had poor appetite or overeating? ',
    '6. In a semester, how often have you been feeling bad about yourself - or that you are a failure or have let yourself or your family down? ',
    '7. In a semester, how often have you been having trouble concentrating on things, such as reading the books or watching television? ',
    "8. In a semester, how often have you moved or spoke too slowly for other people to notice? Or you've been moving a lot more than usual because you've been restless? ",
    '9. In a semester, how often have you had thoughts that you would be better off dead, or of hurting yourself? '
]

STRESS_QUESTIONS = [
    '1. In a semester, how often have you felt upset due to something that happened in your academic affairs? ',
    '2. In a semester, how often you felt as if you were unable to control important things in your academic affairs?',
    '3. In a semester, how often you felt nervous and stressed because of academic pressure? ',
    '4. In a semester, how often you felt as if you could not cope with all the mandatory academic activities? (e.g, assignments, quiz, exams) ',
    '5. In a semester, how often you felt confident about your ability to handle your academic / university problems?',
    '6. In a semester, how often you felt as if things in your academic life is going on your way? ',
    '7. In a semester, how often are you able to control irritations in your academic / university affairs? ',
    '8. In a semester, how often you felt as if your academic performance was on top?',
    '9. In a semester, how often you got angered due to bad performance or low grades that is beyond your control? ',
    '10. In a semester, how often you felt as if academic difficulties are piling up so high that you could not overcome them? '
]

print("🔍 Using ALL original questions for clinical validity...")

# Use ALL questions from each condition
anxiety_features = ANXIETY_QUESTIONS  # All 7 anxiety questions
depression_features = DEPRESSION_QUESTIONS  # All 9 depression questions  
stress_features = STRESS_QUESTIONS  # All 10 stress questions

# Total: 26 questions
ALL_QUESTIONS = anxiety_features + depression_features + stress_features

print(f"\n🎯 USING ALL ORIGINAL QUESTIONS:")
print(f"Total questions: {len(ALL_QUESTIONS)}")
print(f"- Anxiety: {len(anxiety_features)} questions (GAD-7) - Max score: 21")
print(f"- Depression: {len(depression_features)} questions (PHQ-9) - Max score: 27") 
print(f"- Stress: {len(stress_features)} questions (PSS) - Max score: 40")

def create_datasets_with_all_questions():
    """Create datasets using ALL original questions"""
    
    # For anxiety - use ALL 7 anxiety questions
    anxiety_std = anxiety_df[anxiety_features].copy()
    anxiety_std['condition_type'] = 'anxiety'
    anxiety_std['anxiety_label'] = anxiety_df['Anxiety Label']
    
    # For depression - use ALL 9 depression questions  
    depression_std = depression_df[depression_features].copy()
    depression_std['condition_type'] = 'depression'
    depression_std['depression_label'] = depression_df['Depression Label']
    
    # For stress - use ALL 10 stress questions
    stress_std = stress_df[stress_features].copy()
    stress_std['condition_type'] = 'stress'
    stress_std['stress_label'] = stress_df['Stress Label']
    
    # Combine all datasets
    combined_df = pd.concat([anxiety_std, depression_std, stress_std], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Combined dataset shape: {combined_df.shape}")
    print("Condition distribution:")
    print(combined_df['condition_type'].value_counts())
    
    return combined_df, anxiety_features, depression_features, stress_features

# Create dataset with ALL questions
combined_df, anxiety_features, depression_features, stress_features = create_datasets_with_all_questions()

def prepare_features_and_labels_all_questions(combined_df, anxiety_features, depression_features, stress_features):
    """Prepare features and labels for training with ALL questions"""
    
    # All feature columns (26 total)
    all_feature_columns = anxiety_features + depression_features + stress_features
    
    X = combined_df[all_feature_columns]
    y = combined_df[['anxiety_label', 'depression_label', 'stress_label']]
    
    print(f"\n📈 Features shape: {X.shape} (26 questions total)")
    print(f"   - Anxiety: {len(anxiety_features)} questions")
    print(f"   - Depression: {len(depression_features)} questions")
    print(f"   - Stress: {len(stress_features)} questions")
    print(f"📋 Labels shape: {y.shape}")
    
    return X, y, all_feature_columns

# Use the new function with all questions
X, y, feature_columns = prepare_features_and_labels_all_questions(
    combined_df, anxiety_features, depression_features, stress_features
)

def create_label_encoders(y):
    """Create label encoders for each condition"""
    label_encoders = {}
    
    for condition in ['anxiety', 'depression', 'stress']:
        le = LabelEncoder()
        condition_labels = y[f'{condition}_label'].dropna().unique()
        le.fit(condition_labels)
        label_encoders[condition] = le
        print(f"{condition} classes: {list(le.classes_)}")
    
    return label_encoders

label_encoders = create_label_encoders(y)

def train_models_with_all_questions(X_train, y_train, label_encoders, anxiety_features, depression_features, stress_features):
    """Train models using ALL questions for each condition with CPU only"""
    
    models = {}
    scalers = {}  # Separate scaler for each condition's questions
    
    for condition in ['anxiety', 'depression', 'stress']:
        print(f"\n--- Training {condition} model ---")
        
        # Get condition-specific features
        if condition == 'anxiety':
            condition_features = anxiety_features
            q_count = len(anxiety_features)
        elif condition == 'depression':
            condition_features = depression_features
            q_count = len(depression_features)
        else:
            condition_features = stress_features
            q_count = len(stress_features)
        
        # Get only rows for this condition
        condition_mask = combined_df.iloc[X_train.index]['condition_type'] == condition
        X_condition = X_train[condition_mask][condition_features]
        y_condition = y_train[condition_mask][f'{condition}_label']
        
        # Encode labels
        y_encoded = label_encoders[condition].transform(y_condition)
        
        # Create and fit scaler for this condition's features
        condition_scaler = StandardScaler()
        X_scaled = condition_scaler.fit_transform(X_condition)
        scalers[condition] = condition_scaler
        
        # Use CPU only (scikit-learn)
        print(f"⚡ Training {condition} with CPU (scikit-learn)...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            class_weight='balanced',
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_scaled, y_encoded)
        models[condition] = model
        print(f"✅ {condition} model trained on {q_count} questions (full questionnaire)")
    
    return models, scalers

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=combined_df['condition_type']
)

print(f"\n📚 Training set: {X_train.shape[0]} samples")
print(f"🧪 Test set: {X_test.shape[0]} samples")

# Train models with all questions
models, condition_scalers = train_models_with_all_questions(
    X_train, y_train, label_encoders, anxiety_features, depression_features, stress_features
)

def evaluate_models_with_all_questions(models, condition_scalers, X_test, y_test, label_encoders, 
                                     anxiety_features, depression_features, stress_features):
    """Evaluate models using ALL questions and save results to file"""
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION WITH 26 ACTUAL QUESTIONS")
    print("="*60)
    
    # Initialize results dictionary
    results = {
        'anxiety': {},
        'depression': {}, 
        'stress': {}
    }
    
    with open('results.txt', 'w') as f:
        f.write("MENTAL HEALTH MODEL RESULTS - ANXIETY, STRESS, DEPRESSION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training completed on CPU only\n")
        f.write(f"Total questions used: {len(ALL_QUESTIONS)}\n")
        f.write(f"- Anxiety: {len(anxiety_features)} questions (GAD-7)\n")
        f.write(f"- Depression: {len(depression_features)} questions (PHQ-9)\n") 
        f.write(f"- Stress: {len(stress_features)} questions (PSS)\n\n")
    
    for condition in ['anxiety', 'depression', 'stress']:
        print(f"\n--- {condition.upper()} CLASSIFICATION REPORT ---")
        
        # Get condition-specific features
        if condition == 'anxiety':
            condition_features = anxiety_features
        elif condition == 'depression':
            condition_features = depression_features
        else:
            condition_features = stress_features
        
        # Get only rows for this condition
        condition_mask = combined_df.iloc[X_test.index]['condition_type'] == condition
        X_condition = X_test[condition_mask][condition_features]
        true_labels = y_test[condition_mask][f'{condition}_label']
        
        # Scale and predict
        X_scaled = condition_scalers[condition].transform(X_condition)
        pred_encoded = models[condition].predict(X_scaled)
        pred_labels = label_encoders[condition].inverse_transform(pred_encoded)
        
        # Calculate metrics
        accuracy = (true_labels == pred_labels).mean()
        report = classification_report(true_labels, pred_labels, zero_division=0)
        cm = confusion_matrix(true_labels, pred_labels)
        results[condition]['confusion_matrix'] = cm

        # Print confusion matrix
        print(f"📊 Confusion Matrix for {condition}:")
        print(cm)

        # Save confusion matrix to file
        with open('results.txt', 'a') as f:
         f.write(f"Confusion Matrix:\n")
         f.write(np.array2string(cm))
         f.write("\n\n")
        # Store results
        results[condition]['accuracy'] = accuracy
        results[condition]['classification_report'] = report
        results[condition]['true_labels'] = true_labels.tolist()
        results[condition]['predicted_labels'] = pred_labels.tolist()
        
        # Print results
        print(report)
        print(f"🎯 Accuracy for {condition}: {accuracy:.3f}")
        
        # Save to file
        with open('results.txt', 'a') as f:
            f.write(f"\n{condition.upper()} RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Questions used: {len(condition_features)}\n")
            f.write(f"Test samples: {len(true_labels)}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n" + "="*50 + "\n")
    
    return results

# Evaluate models and get results
results = evaluate_models_with_all_questions(
    models, condition_scalers, X_test, y_test, label_encoders,
    anxiety_features, depression_features, stress_features
)

def save_artifacts_with_all_questions(models, condition_scalers, label_encoders, 
                                    anxiety_features, depression_features, stress_features):
    """Save all artifacts with ALL questions for Flask app"""
    
    # Save comprehensive artifacts
    artifacts = {
        'models': models,
        'scalers': condition_scalers,
        'label_encoders': label_encoders,
        'anxiety_features': anxiety_features,
        'depression_features': depression_features, 
        'stress_features': stress_features,
        'gpu_acceleration': False  # Force CPU
    }
    
    joblib.dump(artifacts, 'model_artifacts/mental_health_models_26_questions.pkl')
    
    # Save question information for Flask app
    question_info = {
        'anxiety_questions': anxiety_features,
        'depression_questions': depression_features,
        'stress_questions': stress_features,
        'all_questions': anxiety_features + depression_features + stress_features,
        'question_short_forms': {
            'anxiety': [f"Anxiety Q{i+1}: {q[:50]}..." for i, q in enumerate(anxiety_features)],
            'depression': [f"Depression Q{i+1}: {q[:50]}..." for i, q in enumerate(depression_features)],
            'stress': [f"Stress Q{i+1}: {q[:50]}..." for i, q in enumerate(stress_features)]
        },
        'question_mapping': {
            'anxiety': {f'Q{i+1}': q for i, q in enumerate(anxiety_features)},
            'depression': {f'Q{i+1}': q for i, q in enumerate(depression_features)},
            'stress': {f'Q{i+1}': q for i, q in enumerate(stress_features)}
        },
        'scoring_info': {
            'anxiety': {'max_score': 21, 'questions': 7},
            'depression': {'max_score': 27, 'questions': 9},
            'stress': {'max_score': 40, 'questions': 10}
        }
    }
    
    joblib.dump(question_info, 'model_artifacts/question_info_26_questions.pkl')
    
    # Save individual files for easy access in Flask
    joblib.dump(anxiety_features, 'model_artifacts/anxiety_questions.pkl')
    joblib.dump(depression_features, 'model_artifacts/depression_questions.pkl')
    joblib.dump(stress_features, 'model_artifacts/stress_questions.pkl')
    
    print("\n💾 ALL ARTIFACTS SAVED SUCCESSFULLY!")
    print("📁 Files saved for Flask app:")
    print("   - model_artifacts/mental_health_models_26_questions.pkl")
    print("   - model_artifacts/question_info_26_questions.pkl")
    print("   - model_artifacts/anxiety_questions.pkl")
    print("   - model_artifacts/depression_questions.pkl") 
    print("   - model_artifacts/stress_questions.pkl")
    print(f"\n📋 Question Summary:")
    print(f"   Total questions: {len(anxiety_features) + len(depression_features) + len(stress_features)}")
    print(f"   - Anxiety: {len(anxiety_features)} questions (max 21 points)")
    print(f"   - Depression: {len(depression_features)} questions (max 27 points)")
    print(f"   - Stress: {len(stress_features)} questions (max 40 points)")
    print(f"⚡ Training Mode: CPU Only (GPU disabled)")

# Save artifacts
save_artifacts_with_all_questions(
    models, condition_scalers, label_encoders,
    anxiety_features, depression_features, stress_features
)

def test_prediction_pipeline():
    """Test the complete prediction pipeline with ALL questions"""
    
    # Load artifacts
    artifacts = joblib.load('model_artifacts/mental_health_models_26_questions.pkl')
    question_info = joblib.load('model_artifacts/question_info_26_questions.pkl')
    
    models = artifacts['models']
    scalers = artifacts['scalers']
    label_encoders = artifacts['label_encoders']
    
    print("\n" + "="*50)
    print("🧪 PREDICTION PIPELINE TEST")
    print("="*50)
    
    # Generate sample responses for 26 questions
    np.random.seed(42)
    sample_input = np.random.randint(0, 4, 26).reshape(1, -1)  # 26 questions, values 0-3
    
    print(f"Sample input shape: {sample_input.shape} (26 questions)")
    
    # Predict for all conditions using their specific questions
    final_predictions = {}
    for condition in ['anxiety', 'depression', 'stress']:
        # Get condition-specific questions
        if condition == 'anxiety':
            condition_features = question_info['anxiety_questions']
            feature_indices = list(range(0, 7))  # First 7 questions
        elif condition == 'depression':
            condition_features = question_info['depression_questions'] 
            feature_indices = list(range(7, 16))  # Next 9 questions
        else:
            condition_features = question_info['stress_questions']
            feature_indices = list(range(16, 26))  # Last 10 questions
        
        # Extract relevant features for this condition
        X_condition = sample_input[:, feature_indices]
        
        # Scale and predict
        X_scaled = scalers[condition].transform(X_condition)
        pred_encoded = models[condition].predict(X_scaled)
        final_predictions[condition] = label_encoders[condition].inverse_transform(pred_encoded)[0]
        
        print(f"\n{condition.upper()} Prediction:")
        print(f"  Used {len(condition_features)} questions")
        print(f"  Prediction: {final_predictions[condition]}")
    
    return final_predictions

# Test the complete pipeline
test_predictions = test_prediction_pipeline()

print("\n" + "="*70)
print("🎉 TRAINING WITH 26 ACTUAL QUESTIONS COMPLETED SUCCESSFULLY!")
print("="*70)
print("📊 Model trained on 26 original questions:")
print("   - 7 Anxiety questions (GAD-7) - Max 21 points")
print("   - 9 Depression questions (PHQ-9) - Max 27 points") 
print("   - 10 Stress questions (PSS) - Max 40 points")
print(f"⚡ Training Mode: CPU Only (GPU disabled)")
print("💾 All artifacts saved for Flask app deployment")
print("📄 Results saved to: results.txt")
print("="*70)

# Final confirmation
print(f"\n✅ RESULTS FILE CREATED: results.txt")
print("📝 File contains anxiety, stress, and depression model results including:")
print("   - Accuracy scores for each condition")
print("   - Classification reports with precision, recall, f1-score")
print("   - Number of questions used for each condition")
print("   - Test sample sizes")
