import pandas as pd
import numpy as np
import json
import os

if __name__ == '__main__':
    # Specify the path to your JSON file
    json_file_path = './mining/output/version_history_mining.json'
    output_file = './mining/output/version_history.json'

    # Open the JSON file and read its contents
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # sort data by uniqueid (ascending)
    data = sorted(data, key=lambda x: x['uniqueid'])

    # convert list of dict to pandas dataframe
    df = pd.DataFrame(data)

    # create new column to count number of version history per each code snippet
    # df['number_of_versions'] = df['version_history'].apply(lambda x: len(x) if len(x) == 1 else len(x) - 1)
    df['number_of_versions'] = df['version_history'].apply(lambda x: len(x)-1)

    # create new column to calculate how many day a method exists
    df['days_to_exist'] = df['version_history'].apply(lambda x: sum([item['commit_days_to_prev_version'] for item in x]))

    # dataframe to json string (avoiding adding extra backslash)
    json_str = df.to_json(orient='records' ,date_format='iso')

    # reform string to json format
    parsed = json.loads(json_str)

    # export ouput file    
    with open(output_file, 'w') as json_file:
        json_file.write(json.dumps(parsed, indent=4))
        json_file.close()
    print("merging completed")
    