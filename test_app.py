# test_app.py
from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

@app.route('/')
def test():
    results = {}
    
    # List files in attached_assets
    try:
        results['files_in_directory'] = os.listdir('attached_assets')
    except Exception as e:
        results['directory_error'] = str(e)
    
    # Try to load each file
    data_files = [
        'air_data_average.json',
        'climate_data_average.json',
        'soil_data.json',
        'water_distance.json'
    ]
    
    results['file_content'] = {}
    
    for file in data_files:
        try:
            filepath = f'attached_assets/{file}'
            with open(filepath, 'r') as f:
                data = json.load(f)
                results['file_content'][file] = {
                    'loaded': True,
                    'keys_count': len(data.keys()),
                    'sample_keys': list(data.keys())[:5]
                }
        except Exception as e:
            results['file_content'][file] = {
                'loaded': False,
                'error': str(e)
            }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)