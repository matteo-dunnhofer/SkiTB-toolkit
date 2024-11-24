import os
import json
import numpy as np
import zipfile


# Set the path to SkiTB's main folder
root_dir = './SkiTB'

# Set the path to folder containing tracker results
result_dir = './SkiTB-results'

# Set the tracker's name (its folder should be present in result_dir)
name = 'STARKski'

print(f'Exporting {name} for challenge. This might take a while...')

submission_file_json = os.path.join(f'challenge-submission.json')
submission_file_zip = os.path.join(f'challenge-submission.zip')

submission_dict = {}
submission_dict['challenge'] = 'visual_tracking'
submission_dict['version'] = '0.1'
submission_dict['tracker_name'] = name
submission_dict['results']  = {}
submission_dict['results']['AL']  = {}
submission_dict['results']['JP']  = {}
submission_dict['results']['FS']  = {}

for discipline in ['AL', 'JP', 'FS']:

    split_file = os.path.join(root_dir, discipline, f'{discipline}_train_val_test_date_60-40.json')

    f = open(split_file)
    split_dict = json.load(f)
    test_sequences = split_dict['test']
 
    for s, seq_name  in enumerate(test_sequences):

        record_dir = os.path.join(
                    result_dir, name, 'LT', discipline, 'ope', seq_name)

        boxes = np.genfromtxt(os.path.join(record_dir, 'boxes.txt'), delimiter=',')
        confs = np.genfromtxt(os.path.join(record_dir, 'confidences.txt'), delimiter='\n')

        submission_dict['results'][discipline][f'{seq_name}'] = {}
        submission_dict['results'][discipline][f'{seq_name}']['boxes'] = boxes.tolist()
        submission_dict['results'][discipline][f'{seq_name}']['confidences'] = confs.tolist()
        

# report the performance
json_str = json.dumps(submission_dict, indent=4, ensure_ascii=False)
with open(submission_file_json, 'w', encoding='utf-8') as f:
    f.write(json_str)
    f.close()

print(f'Submission json file saved to {submission_file_json}')

# zip the json file
zipfile.ZipFile(submission_file_zip, mode='w').write(submission_file_json)
print(f'Submission json file compressed to {submission_file_zip}')