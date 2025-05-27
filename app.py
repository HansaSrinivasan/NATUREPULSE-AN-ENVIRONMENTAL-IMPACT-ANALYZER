from flask import Flask, render_template, jsonify, request, abort
import numpy as np
import pandas as pd
import logging
import os
import json
from data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create Flask app
app = Flask(__name__)

# Create global data loader instance
data_loader = DataLoader()

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/compare')
def compare():
    """Render the city comparison page."""
    return render_template('compare.html')

@app.route('/algorithms')
def algorithms():
    """Render the algorithms page."""
    return render_template('algorithms.html')

@app.route('/api/city/<city_name>')
def city_data(city_name):
    """API endpoint to get all data for a specific city."""
    try:
        # Use BST for city data lookup (O(log n) complexity)
        city_data = data_loader.get_city_data_bst(city_name)
        if city_data:
            return jsonify({
                'success': True,
                'city': city_name,
                'air_data': city_data.get('air_data', {}),
                'climate_data': city_data.get('climate_data', {}),
                'soil_data': city_data.get('soil_data', {}),
                'water_data': city_data.get('water_data', {})
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No data found for {city_name}'
            })
    except Exception as e:
        logging.error(f"Error in city_data API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/city_hash/<city_name>')
def city_data_hash(city_name):
    """API endpoint to get all data for a specific city using hash table (O(1) lookup)."""
    try:
        # Use hash table for city data lookup (O(1) complexity)
        city_data = data_loader.get_city_data_hash(city_name)
        if city_data:
            return jsonify({
                'success': True,
                'city': city_name,
                'air_data': city_data.get('air_data', {}),
                'climate_data': city_data.get('climate_data', {}),
                'soil_data': city_data.get('soil_data', {}),
                'water_data': city_data.get('water_data', {})
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No data found for {city_name}'
            })
    except Exception as e:
        logging.error(f"Error in city_data_hash API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/cities')
def get_cities():
    """API endpoint to get all available cities (sorted using merge sort)."""
    try:
        cities = data_loader.get_cities()
        return jsonify({
            'success': True,
            'cities': cities
        })
    except Exception as e:
        logging.error(f"Error in get_cities API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/optimal_water_network')
def optimal_water_network():
    """API endpoint to get optimal water network using Kruskal's MST algorithm."""
    try:
        network = data_loader.get_optimal_water_network()
        if network:
            return jsonify({
                'success': True,
                'nodes': network['nodes'],
                'links': network['links'],
                'total_weight': network['total_weight']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not calculate optimal water network'
            })
    except Exception as e:
        logging.error(f"Error in optimal_water_network API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/optimal_policy/<city_name>')
def optimal_policy(city_name):
    """API endpoint to get optimal environmental policy path using Multistage Graph DP."""
    try:
        # Get metric weights from query parameters
        air_weight = float(request.args.get('air_weight', 0.4))
        water_weight = float(request.args.get('water_weight', 0.3))
        soil_weight = float(request.args.get('soil_weight', 0.3))
        
        metrics = {
            'air': air_weight,
            'water': water_weight,
            'soil': soil_weight
        }
        
        result = data_loader.get_optimal_policy_path(city_name, metrics)
        
        if result:
            return jsonify({
                'success': True,
                'city': city_name,
                'optimal_path': result['path'],
                'total_cost': result['cost']
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Could not calculate optimal policy for {city_name}'
            })
    except Exception as e:
        logging.error(f"Error in optimal_policy API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/recommend_business')
def recommend_business():
    """API endpoint to recommend best city for business based on environmental factors."""
    try:
        # Get metric weights from query parameters
        air_weight = float(request.args.get('air_weight', 0.4))
        water_weight = float(request.args.get('water_weight', 0.3))
        soil_weight = float(request.args.get('soil_weight', 0.3))
        
        metrics = {
            'air': air_weight,
            'water': water_weight,
            'soil': soil_weight
        }
        
        result = data_loader.recommend_business_location(metrics)
        
        if result:
            return jsonify({
                'success': True,
                'recommended_city': result['city'],
                'score': result['score'],
                'factors': result['factors']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not generate business recommendations'
            })
    except Exception as e:
        logging.error(f"Error in recommend_business API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/compare', methods=['POST'])
def compare_cities():
    """API endpoint to compare environmental data between cities."""
    try:
        data = request.get_json()
        cities = data.get('cities', [])
        
        if len(cities) < 2:
            return jsonify({
                'success': False,
                'error': 'Please provide at least two cities for comparison'
            })
        
        comparison_data = {}
        
        for city in cities:
            city_data = data_loader.get_city_data_bst(city)
            if city_data:
                comparison_data[city] = city_data
        
        if not comparison_data:
            return jsonify({
                'success': False,
                'error': 'No data found for the selected cities'
            })
        
        return jsonify({
            'success': True,
            'comparison': comparison_data
        })
    except Exception as e:
        logging.error(f"Error in compare_cities API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)