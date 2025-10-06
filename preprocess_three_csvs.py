"""
Specialized Preprocessing for Anxiety, Depression, and Stress CSVs
Converts survey responses to descriptive text for model training
"""

import pandas as pd
import numpy as np
import os

# Response frequency mapping
RESPONSE_MAP = {
    0: 'never',
    1: 'rarely',
    2: 'sometimes',
    3: 'often',
    4: 'very often'
}

def process_anxiety_csv(csv_path):
    """
    Process Anxiety CSV with 7 questions
    """
    print(f"\n{'='*70}")
    print(f"Processing ANXIETY data: {csv_path}")
    print('='*70)
    
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    # Anxiety-specific questions (columns 8-14 in your CSV)
    anxiety_questions = [
        'In a semester, how often you felt nervous, anxious or on edge due to academic pressure?',
        'In a semester, how often have you been unable to stop worrying about your academic affairs?',
        'In a semester, how often have you had trouble relaxing due to academic pressure?',
        'In a semester, how often have you been easily annoyed or irritated because of academic pressure?',
        'In a semester, how often have you worried too much about academic affairs?',
        'In a semester, how often have you been so restless due to academic pressure that it is hard to sit still?',
        'In a semester, how often have you felt afraid, as if something awful might happen?'
    ]
    
    # Find actual column names (they might have slight variations)
    actual_q_cols = []
    for q in anxiety_questions:
        for col in df.columns:
            if q.lower()[:50] in col.lower():
                actual_q_cols.append(col)
                break
    
    if len(actual_q_cols) < 5:
        # Fallback: find numeric columns
        actual_q_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']][-7:]
    
    print(f"Found {len(actual_q_cols)} question columns")
    
    # Find label column
    label_col = [col for col in df.columns if 'Label' in col and 'Anxiety' in col]
    if label_col:
        label_col = label_col[0]
    else:
        label_col = df.columns[-1]  # Last column
    
    print(f"Label column: {label_col}")
    print(f"Unique labels: {df[label_col].nunique()}")
    
    processed_data = []
    
    for idx, row in df.iterrows():
        text_parts = []
        
        # Add demographic context
        gender = str(row.get('Gender', 'person')).lower()
        if gender in ['male', 'female']:
            text_parts.append(f"I am a {gender} student")
        
        # Process each anxiety question
        responses = []
        for i, q_col in enumerate(actual_q_cols[:7]):
            try:
                val = int(row[q_col])
                freq = RESPONSE_MAP.get(val, 'sometimes')
                
                # Create contextual statements
                if i == 0:
                    responses.append(f"I feel nervous and anxious {freq}")
                elif i == 1:
                    responses.append(f"I cannot stop worrying {freq}")
                elif i == 2:
                    responses.append(f"I have trouble relaxing {freq}")
                elif i == 3:
                    responses.append(f"I get easily annoyed {freq}")
                elif i == 4:
                    responses.append(f"I worry excessively {freq}")
                elif i == 5:
                    responses.append(f"I feel restless {freq}")
                elif i == 6:
                    responses.append(f"I feel afraid {freq}")
            except:
                continue
        
        text_parts.extend(responses)
        
        # Create final text
        text = ". ".join(text_parts) + "."
        
        # Get and simplify label
        original_label = str(row[label_col])
        simplified_label = simplify_anxiety_label(original_label)
        
        processed_data.append({
            'text': text,
            'label': simplified_label,
            'original_label': original_label,
            'category': 'anxiety'
        })
    
    result_df = pd.DataFrame(processed_data)
    print(f"✓ Processed {len(result_df)} anxiety samples")
    print(f"  Simplified labels: {result_df['label'].value_counts().to_dict()}")
    
    return result_df


def process_depression_csv(csv_path):
    """
    Process Depression CSV with 9 questions
    """
    print(f"\n{'='*70}")
    print(f"Processing DEPRESSION data: {csv_path}")
    print('='*70)
    
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    # Depression-specific questions
    depression_questions = [
        'In a semester, how often have you had little interest or pleasure in doing things?',
        'In a semester, how often have you been feeling down, depressed or hopeless?',
        'In a semester, how often have you had trouble falling or staying asleep, or sleeping too much?',
        'In a semester, how often have you been feeling tired or having little energy?',
        'In a semester, how often have you had poor appetite or overeating?',
        'In a semester, how often have you been feeling bad about yourself - or that you are a failure',
        'In a semester, how often have you been having trouble concentrating on things',
        'In a semester, how often have you moved or spoke too slowly',
        'In a semester, how often have you had thoughts that you would be better off dead'
    ]
    
    # Find actual column names
    actual_q_cols = []
    for q in depression_questions:
        for col in df.columns:
            if q.lower()[:40] in col.lower():
                actual_q_cols.append(col)
                break
    
    if len(actual_q_cols) < 5:
        # Fallback
        actual_q_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']][-9:]
    
    print(f"Found {len(actual_q_cols)} question columns")
    
    # Find label column
    label_col = [col for col in df.columns if 'Label' in col and 'Depression' in col]
    if label_col:
        label_col = label_col[0]
    else:
        label_col = df.columns[-1]
    
    print(f"Label column: {label_col}")
    print(f"Unique labels: {df[label_col].nunique()}")
    
    processed_data = []
    
    for idx, row in df.iterrows():
        text_parts = []
        
        # Add demographic context
        gender = str(row.get('Gender', 'person')).lower()
        if gender in ['male', 'female']:
            text_parts.append(f"I am a {gender} student")
        
        # Process each depression question
        responses = []
        for i, q_col in enumerate(actual_q_cols[:9]):
            try:
                val = int(row[q_col])
                freq = RESPONSE_MAP.get(val, 'sometimes')
                
                # Create contextual statements
                if i == 0:
                    responses.append(f"I have little interest in activities {freq}")
                elif i == 1:
                    responses.append(f"I feel down and depressed {freq}")
                elif i == 2:
                    responses.append(f"I have trouble sleeping {freq}")
                elif i == 3:
                    responses.append(f"I feel tired and lack energy {freq}")
                elif i == 4:
                    responses.append(f"I have appetite problems {freq}")
                elif i == 5:
                    responses.append(f"I feel like a failure {freq}")
                elif i == 6:
                    responses.append(f"I have trouble concentrating {freq}")
                elif i == 7:
                    responses.append(f"I move or speak slowly {freq}")
                elif i == 8:
                    responses.append(f"I have dark thoughts {freq}")
            except:
                continue
        
        text_parts.extend(responses)
        
        # Create final text
        text = ". ".join(text_parts) + "."
        
        # Get and simplify label
        original_label = str(row[label_col])
        simplified_label = simplify_depression_label(original_label)
        
        processed_data.append({
            'text': text,
            'label': simplified_label,
            'original_label': original_label,
            'category': 'depression'
        })
    
    result_df = pd.DataFrame(processed_data)
    print(f"✓ Processed {len(result_df)} depression samples")
    print(f"  Simplified labels: {result_df['label'].value_counts().to_dict()}")
    
    return result_df


def process_stress_csv(csv_path):
    """
    Process Stress CSV with 10 questions
    """
    print(f"\n{'='*70}")
    print(f"Processing STRESS data: {csv_path}")
    print('='*70)
    
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    # Stress-specific questions
    stress_questions = [
        'In a semester, how often have you felt upset due to something that happened in your academic affairs?',
        'In a semester, how often you felt as if you were unable to control important things in your academic affairs?',
        'In a semester, how often you felt nervous and stressed because of academic pressure?',
        'In a semester, how often you felt as if you could not cope with all the mandatory academic activities?',
        'In a semester, how often you felt confident about your ability to handle your academic',
        'In a semester, how often you felt as if things in your academic life is going on your way?',
        'In a semester, how often are you able to control irritations in your academic',
        'In a semester, how often you felt as if your academic performance was on top?',
        'In a semester, how often you got angered due to bad performance or low grades',
        'In a semester, how often you felt as if academic difficulties are piling up'
    ]
    
    # Find actual column names
    actual_q_cols = []
    for q in stress_questions:
        for col in df.columns:
            if q.lower()[:40] in col.lower():
                actual_q_cols.append(col)
                break
    
    if len(actual_q_cols) < 5:
        # Fallback
        actual_q_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']][-10:]
    
    print(f"Found {len(actual_q_cols)} question columns")
    
    # Find label column
    label_col = [col for col in df.columns if 'Label' in col and 'Stress' in col]
    if label_col:
        label_col = label_col[0]
    else:
        label_col = df.columns[-1]
    
    print(f"Label column: {label_col}")
    print(f"Unique labels: {df[label_col].nunique()}")
    
    processed_data = []
    
    for idx, row in df.iterrows():
        text_parts = []
        
        # Add demographic context
        gender = str(row.get('Gender', 'person')).lower()
        if gender in ['male', 'female']:
            text_parts.append(f"I am a {gender} student")
        
        # Process each stress question
        responses = []
        for i, q_col in enumerate(actual_q_cols[:10]):
            try:
                val = int(row[q_col])
                freq = RESPONSE_MAP.get(val, 'sometimes')
                
                # Create contextual statements
                if i == 0:
                    responses.append(f"I feel upset about academic matters {freq}")
                elif i == 1:
                    responses.append(f"I feel unable to control things {freq}")
                elif i == 2:
                    responses.append(f"I feel nervous and stressed {freq}")
                elif i == 3:
                    responses.append(f"I cannot cope with activities {freq}")
                elif i == 4:
                    # Reverse scored - higher is better
                    if val <= 2:
                        responses.append(f"I lack confidence in handling problems")
                elif i == 5:
                    # Reverse scored
                    if val <= 2:
                        responses.append(f"Things are not going my way")
                elif i == 6:
                    # Reverse scored
                    if val <= 2:
                        responses.append(f"I cannot control my irritations")
                elif i == 7:
                    # Reverse scored
                    if val <= 2:
                        responses.append(f"My performance is not satisfactory")
                elif i == 8:
                    responses.append(f"I get angered by bad performance {freq}")
                elif i == 9:
                    responses.append(f"Difficulties are piling up {freq}")
            except:
                continue
        
        text_parts.extend(responses)
        
        # Create final text
        text = ". ".join(text_parts) + "."
        
        # Get and simplify label
        original_label = str(row[label_col])
        simplified_label = simplify_stress_label(original_label)
        
        processed_data.append({
            'text': text,
            'label': simplified_label,
            'original_label': original_label,
            'category': 'stress'
        })
    
    result_df = pd.DataFrame(processed_data)
    print(f"✓ Processed {len(result_df)} stress samples")
    print(f"  Simplified labels: {result_df['label'].value_counts().to_dict()}")
    
    return result_df


def simplify_anxiety_label(label):
    """Simplify anxiety labels to 4 categories"""
    label_lower = str(label).lower()
    
    if 'minimal' in label_lower or 'no' in label_lower:
        return 'normal'
    elif 'mild' in label_lower:
        return 'anxiety'
    elif 'moderate' in label_lower:
        return 'anxiety'
    elif 'severe' in label_lower:
        return 'anxiety'
    else:
        return 'anxiety'


def simplify_depression_label(label):
    """Simplify depression labels to 4 categories"""
    label_lower = str(label).lower()
    
    if 'minimal' in label_lower or 'no depression' in label_lower or 'none' in label_lower:
        return 'normal'
    elif 'mild' in label_lower:
        return 'depression'
    elif 'moderate' in label_lower:
        return 'depression'
    elif 'severe' in label_lower:
        return 'depression'
    else:
        return 'depression'


def simplify_stress_label(label):
    """Simplify stress labels to 4 categories"""
    label_lower = str(label).lower()
    
    if 'low' in label_lower or 'minimal' in label_lower or 'no stress' in label_lower:
        return 'normal'
    elif 'moderate' in label_lower:
        return 'stress'
    elif 'high' in label_lower or 'severe' in label_lower:
        return 'stress'
    else:
        return 'stress'


def add_normal_samples(combined_df, target_percentage=0.25):
    """
    Add synthetic 'normal' samples to balance the dataset
    """
    print(f"\n{'='*70}")
    print("BALANCING DATASET")
    print('='*70)
    
    class_counts = combined_df['label'].value_counts()
    print(f"\nCurrent class distribution:")
    print(class_counts)
    
    total_samples = len(combined_df)
    normal_count = class_counts.get('normal', 0)
    target_normal = int(total_samples * target_percentage)
    
    samples_needed = max(0, target_normal - normal_count)
    
    if samples_needed > 0:
        print(f"\nAdding {samples_needed} 'normal' samples...")
        
        normal_templates = [
            "I am a student feeling mentally healthy. I manage my studies well. I feel calm and relaxed. I have good energy levels. I enjoy my activities. I sleep well at night. I maintain positive relationships.",
            "I am doing well academically and personally. I feel happy and content. I have no major worries. My stress levels are manageable. I feel positive about my future. I can concentrate well. I handle challenges effectively.",
            "I am coping well with academic pressure. I feel motivated and energetic. I have a positive outlook. My anxiety is minimal. I feel in control of my life. I am satisfied with my progress. I maintain work-life balance.",
            "I am a person with good mental health. I feel balanced and stable. I can handle stress effectively. I have supportive relationships. I am productive and focused. I feel optimistic about the future.",
            "I feel emotionally stable and content. I am managing my time well. I have healthy coping mechanisms. I enjoy learning and growing. I feel confident in my abilities. I maintain healthy sleep patterns.",
            "I am a student with good mental wellness. I feel energized and motivated. I can focus on my tasks. I handle pressure well. I maintain positive thoughts. I feel supported by others. I am achieving my goals."
        ]
        
        additional_normal = []
        for i in range(samples_needed):
            text = np.random.choice(normal_templates)
            additional_normal.append({
                'text': text,
                'label': 'normal',
                'original_label': 'Normal/Healthy',
                'category': 'normal'
            })
        
        normal_df = pd.DataFrame(additional_normal)
        combined_df = pd.concat([combined_df, normal_df], ignore_index=True)
        
        print(f"✓ Added {samples_needed} normal samples")
    
    print(f"\nFinal class distribution:")
    print(combined_df['label'].value_counts())
    
    return combined_df


def main():
    """
    Main preprocessing pipeline
    """
    print("="*70)
    print("MENTAL HEALTH DATA PREPROCESSING")
    print("Three CSV Files: Anxiety, Depression, Stress")
    print("="*70)
    
    # ===================================================================
    # UPDATE THESE PATHS WITH YOUR ACTUAL CSV FILE PATHS
    # ===================================================================
    
    ANXIETY_CSV = "/content/ProjectFlask_internship_Assignment/mental_health_dataset/Anxiety.csv"
    DEPRESSION_CSV = "/content/ProjectFlask_internship_Assignment/mental_health_dataset/Depression.csv"
    STRESS_CSV = "/content/ProjectFlask_internship_Assignment/mental_health_dataset/Stress.csv"
    OUTPUT_CSV = "/content/ProjectFlask_internship_Assignment/mental_health_processed.csv"
    
    # ===================================================================
    
    # Process each CSV
    try:
        anxiety_df = process_anxiety_csv(ANXIETY_CSV)
    except Exception as e:
        print(f"❌ Error processing anxiety: {e}")
        anxiety_df = pd.DataFrame()
    
    try:
        depression_df = process_depression_csv(DEPRESSION_CSV)
    except Exception as e:
        print(f"❌ Error processing depression: {e}")
        depression_df = pd.DataFrame()
    
    try:
        stress_df = process_stress_csv(STRESS_CSV)
    except Exception as e:
        print(f"❌ Error processing stress: {e}")
        stress_df = pd.DataFrame()
    
    # Combine all dataframes
    all_dfs = [df for df in [anxiety_df, depression_df, stress_df] if len(df) > 0]
    
    if not all_dfs:
        print("\n❌ No data was processed! Check file paths and column names.")
        return
    
    print(f"\n{'='*70}")
    print("COMBINING ALL DATA")
    print('='*70)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal samples before balancing: {len(combined_df)}")
    
    # Balance dataset
    combined_df = add_normal_samples(combined_df, target_percentage=0.25)
    
    # Keep only text and label columns
    final_df = combined_df[['text', 'label']]
    
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*70}")
    print("✅ PREPROCESSING COMPLETE!")
    print('='*70)
    print(f"\n✓ Processed data saved to:")
    print(f"  {OUTPUT_CSV}")
    print(f"\n✓ Total samples: {len(final_df)}")
    print(f"✓ Number of classes: {final_df['label'].nunique()}")
    print(f"✓ Format: text, label")
    
    print(f"\n📊 Final Statistics:")
    print(final_df['label'].value_counts())
    
    print(f"\n📝 Sample data:")
    for label in final_df['label'].unique():
        print(f"\n{label.upper()} example:")
        sample = final_df[final_df['label'] == label].iloc[0]
        print(f"  {sample['text'][:150]}...")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Update train_text_model.py:")
    print(f"      DATA_PATH = '{OUTPUT_CSV}'")
    print(f"   2. Run: python train_text_model.py")
    print(f"   3. Expected accuracy: 80-88%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
