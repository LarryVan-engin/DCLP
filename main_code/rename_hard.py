import os
import pandas as pd
import re

dir_path = r'dataset_easyocr\hard_cases'
csv_path = os.path.join(dir_path, 'labels.csv')

def safe_filename(text):
    text = str(text)
    # Replace invalid Windows filename characters
    return re.sub(r'[<>:\"/\\|?*]', '-', text)

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    new_rows = []
    
    for idx, row in df.iterrows():
        old_filename = row['filename']
        gt = str(row['words'])
        pred = str(row.get('pred', 'None'))
        
        old_path = os.path.join(dir_path, old_filename)
        
        new_filename = f'GT_{safe_filename(gt)}___PRED_{safe_filename(pred)}___{idx}.jpg'
        new_path = os.path.join(dir_path, new_filename)
        
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            
            # Update row
            row['filename'] = new_filename
        
        new_rows.append(row)
        
    new_df = pd.DataFrame(new_rows)
    # Reorder columns to ensure 'filename' and 'words' are first
    cols = ['filename', 'words'] + [c for c in new_df.columns if c not in ['filename', 'words']]
    new_df = new_df[cols]
    new_df.to_csv(csv_path, index=False)
    print('Renamed all files and updated labels.csv')
else:
    print('labels.csv not found')
