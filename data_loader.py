import json
import logging
import os
import numpy as np
from algorithms import BST, HashTable, Graph, merge_sort, multistage_graph_dp

class DataLoader:
    """
    Class to load and manage environmental data from JSON files.
    Implements advanced data structures and algorithms for efficient data processing.
    """
    
    def __init__(self):
        """Initialize the DataLoader and load all data sources."""
        self.air_data = {}
        self.climate_data = {}
        self.soil_data = {}
        self.water_data = {}
        self.common_cities = []
        self.city_bst = BST()
        self.city_hash = HashTable()
        self.water_graph = Graph()
        
        # Load all data
        self.load_air_data()
        self.load_climate_data()
        self.load_soil_data()
        self.load_water_data()
        
        # Find common cities across all datasets
        self.find_common_cities()
        
        # Build data structures
        self._build_data_structures()
        
        logging.info(f"Data loaded for {len(self.common_cities)} cities")
    
    def load_air_data(self):
        """Load air quality data from JSON file."""
        try:
            if os.path.exists('sorted_air_data_average.json'):
                with open('sorted_air_data_average.json', 'r') as f:
                    self.air_data = json.load(f)
            elif os.path.exists('air_data_average.json'):
                with open('air_data_average.json', 'r') as f:
                    self.air_data = json.load(f)
        except Exception as e:
            logging.error(f"Error loading air data: {str(e)}")
    
    def load_climate_data(self):
        """Load climate data from JSON file."""
        try:
            if os.path.exists('sorted_climate_data_average.json'):
                with open('sorted_climate_data_average.json', 'r') as f:
                    self.climate_data = json.load(f)
            elif os.path.exists('climate_data_average.json'):
                with open('climate_data_average.json', 'r') as f:
                    self.climate_data = json.load(f)
        except Exception as e:
            logging.error(f"Error loading climate data: {str(e)}")
    
    def load_soil_data(self):
        """Load soil data from JSON file."""
        try:
            if os.path.exists('sorted_soil_data.json'):
                with open('sorted_soil_data.json', 'r') as f:
                    self.soil_data = json.load(f)
            elif os.path.exists('soil_data.json'):
                with open('soil_data.json', 'r') as f:
                    self.soil_data = json.load(f)
        except Exception as e:
            logging.error(f"Error loading soil data: {str(e)}")
    
    def load_water_data(self):
        """Load water proximity data from JSON file."""
        try:
            if os.path.exists('sorted_water_distance.json'):
                with open('sorted_water_distance.json', 'r') as f:
                    self.water_data = json.load(f)
            elif os.path.exists('water_distance.json'):
                with open('water_distance.json', 'r') as f:
                    self.water_data = json.load(f)
        except Exception as e:
            logging.error(f"Error loading water data: {str(e)}")
    
    def find_common_cities(self):
        """Find cities that exist in all datasets and sort using merge sort."""
        cities = set(self.air_data.keys())
        cities = cities.intersection(set(self.climate_data.keys()))
        cities = cities.intersection(set(self.soil_data.keys()))
        cities = cities.intersection(set(self.water_data.keys()))
        
        # Use our merge sort implementation to sort the cities
        self.common_cities = merge_sort(list(cities))
    
    def _build_data_structures(self):
        """Build BST, hash table, and water graph for efficient data operations."""
        # Build Binary Search Tree and Hash Table
        for city in self.common_cities:
            city_data = {
                'air_data': self.air_data.get(city, {}),
                'climate_data': self.climate_data.get(city, {}),
                'soil_data': self.soil_data.get(city, {}),
                'water_data': self.water_data.get(city, {})
            }
            self.city_bst.insert(city, city_data)
            self.city_hash.insert(city, city_data)
        
        # Build graph for water network optimization
        self._build_water_graph()
    
    def _build_water_graph(self):
        """Build graph of water connections between cities for MST using Kruskal's algorithm."""
        # Add edges between each pair of cities with water distance as weight
        for i, city1 in enumerate(self.common_cities):
            for j, city2 in enumerate(self.common_cities[i+1:], i+1):
                try:
                    water_dist1 = self.water_data.get(city1, {}).get('distance', 0)
                    water_dist2 = self.water_data.get(city2, {}).get('distance', 0)
                    
                    # Weight is the distance between cities based on their water source distances
                    # This is a simplification for demonstration purposes
                    weight = abs(water_dist1 - water_dist2) + 10  # Add a base distance
                    
                    self.water_graph.add_edge(city1, city2, weight)
                except Exception as e:
                    logging.warning(f"Error adding edge between {city1} and {city2}: {str(e)}")
    
    def get_cities(self):
        """Return the list of available cities sorted using merge sort."""
        return self.common_cities
    
    def get_city_data_bst(self, city):
        """Get data for a specific city using BST (O(log n) lookup)."""
        return self.city_bst.search(city)
    
    def get_city_data_hash(self, city):
        """Get data for a specific city using hash table (O(1) lookup)."""
        return self.city_hash.get(city)
    
    def get_air_data(self, city):
        """Get air quality data for a specific city."""
        city_data = self.get_city_data_bst(city)
        if city_data:
            return city_data.get('air_data', {})
        return {}
    
    def get_climate_data(self, city):
        """Get climate data for a specific city."""
        city_data = self.get_city_data_bst(city)
        if city_data:
            return city_data.get('climate_data', {})
        return {}
    
    def get_soil_data(self, city):
        """Get soil data for a specific city."""
        city_data = self.get_city_data_bst(city)
        if city_data:
            return city_data.get('soil_data', {})
        return {}
    
    def get_water_data(self, city):
        """Get water proximity data for a specific city."""
        city_data = self.get_city_data_bst(city)
        if city_data:
            return city_data.get('water_data', {})
        return {}
    
    def get_all_data(self, city):
        """Get all environmental data for a specific city."""
        return self.get_city_data_bst(city)
    
    def get_optimal_water_network(self):
        """
        Calculate the minimum cost pipeline network connecting cities and water sources
        using Kruskal's MST algorithm.
        """
        if not self.water_graph.edges:
            return None
        
        mst = self.water_graph.kruskal_mst()
        
        # Format the MST for visualization
        nodes = set()
        links = []
        total_weight = 0
        
        for u, v, weight in mst:
            nodes.add(u)
            nodes.add(v)
            links.append({'source': u, 'target': v, 'weight': weight})
            total_weight += weight
        
        return {
            'nodes': list(nodes),
            'links': links,
            'total_weight': total_weight
        }
    
    def get_optimal_policy_path(self, city, metrics=None):
        """
        Calculate optimal environmental policy implementation path for a city
        using Multistage Graph Dynamic Programming.
        
        Args:
            city: Name of the city
            metrics: Dictionary with weights for different metrics (air, water, soil)
        
        Returns:
            Dictionary with optimal path and cost
        """
        if metrics is None:
            metrics = {'air': 0.4, 'water': 0.3, 'soil': 0.3}
        
        city_data = self.get_city_data_bst(city)
        if not city_data:
            return None
        
        # We'll use 5 stages for a 5-year implementation plan
        stages = 5
        
        # Calculate optimal path using Multistage Graph DP
        optimal_path, total_cost = multistage_graph_dp(stages, metrics)
        
        return {
            'path': optimal_path,
            'cost': total_cost
        }
    
    def recommend_business_location(self, metrics=None):
        """
        Recommend best city for business location based on environmental factors
        using a weighted scoring system.
        
        Args:
            metrics: Dictionary with weights for different environmental factors
        
        Returns:
            Dictionary with recommended city and score
        """
        if metrics is None:
            metrics = {'air': 0.4, 'water': 0.3, 'soil': 0.3}
        
        best_city = None
        best_score = -float('inf')
        factors = {}
        
        for city in self.common_cities:
            city_data = self.get_city_data_hash(city)
            if not city_data:
                continue
            
            # Calculate score based on weighted metrics
            air_score = 100 - city_data['air_data'].get('aqi', 50)  # Lower AQI is better
            water_score = 100 - city_data['water_data'].get('distance', 50)  # Closer water is better
            soil_score = {'Red': 70, 'Black': 90, 'Alluvial': 85, 'Laterite': 60, 'Sandy': 50}.get(
                city_data['soil_data'].get('type', 'Sandy'), 50)
            
            # Weighted score
            score = (
                metrics.get('air', 0.4) * air_score +
                metrics.get('water', 0.3) * water_score +
                metrics.get('soil', 0.3) * soil_score
            )
            
            if score > best_score:
                best_score = score
                best_city = city
                factors = {
                    'air_quality': air_score,
                    'water_proximity': water_score,
                    'soil_quality': soil_score
                }
        
        if best_city:
            return {
                'city': best_city,
                'score': best_score,
                'factors': factors
            }
        
        return None