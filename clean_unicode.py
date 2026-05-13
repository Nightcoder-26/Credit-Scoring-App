import os
import re

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace common emojis/symbols with ASCII text
    replacements = {
        r'[BANK]': '[BANK]',
        r'[DATA]': '[DATA]',
        r'[AI]': '[AI]',
        r'[PREDICT]': '[PREDICT]',
        r'[SHAP]': '[SHAP]',
        r'[ABOUT]': '[ABOUT]',
        r'[HOME]': '[HOME]',
        r'[EDA]': '[EDA]',
        r'[OK]': '[OK]',
        r'[BEST]': '[BEST]',
        r'[SAVED]': '[SAVED]',
        r'[SAMPLE]': '[SAMPLE]',
        r'[TUNE]': '[TUNE]',
        r'[UP]': '[UP]',
        r'[DOWN]': '[DOWN]',
        r'[TARGET]': '[TARGET]',
        r'[RUN]': '[RUN]',
        r'[WARN]': '[WARN]',
        r'[INFO]': '[INFO]',
        r'[DOWNLOADING]': '[DOWNLOADING]',
        r'[FOLDER]': '[FOLDER]',
        r'[TUNE]': '[FIX]',
        r'[TUNE]': '[TUNE]',
        r'-': '-',
        r'-': '-',
        r'*': '*',
        r'[WOW]': '[WOW]',
        r'[HOT]': '[HOT]',
        r'[TIP]': '[TIP]',
    }
    
    for char, replacement in replacements.items():
        content = content.replace(char, replacement)
    
    # Remove any other non-ASCII characters
    content = re.sub(r'[^\x00-\x7F]+', ' ', content)
    
    with open(filepath, 'w', encoding='ascii', errors='ignore') as f:
        f.write(content)
    print(f"Cleaned: {filepath}")

root_dir = r"C:\Users\Lenovo\Desktop\CodeAlpha_CreditScoring"
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.py'):
            clean_file(os.path.join(root, file))
