import numpy as np

class TreeNode:
    """Binary Search Tree Node for city data storage"""
    def __init__(self, city, data=None):
        self.city = city
        self.data = data or {}
        self.left = None
        self.right = None

class BST:
    """Binary Search Tree for efficient city data lookup"""
    def __init__(self):
        self.root = None
        
    def insert(self, city, data):
        """Insert a city with its associated data into the BST"""
        if not self.root:
            self.root = TreeNode(city, data)
        else:
            self._insert(self.root, city, data)
    
    def _insert(self, node, city, data):
        """Helper function for recursively inserting into the BST"""
        if city < node.city:
            if node.left is None:
                node.left = TreeNode(city, data)
            else:
                self._insert(node.left, city, data)
        elif city > node.city:
            if node.right is None:
                node.right = TreeNode(city, data)
            else:
                self._insert(node.right, city, data)
        else:
            # If city already exists, update data
            node.data = data
    
    def search(self, city):
        """Search for a city in the BST and return its data"""
        return self._search(self.root, city)
    
    def _search(self, node, city):
        """Helper function for recursively searching the BST"""
        if node is None:
            return None
        
        if city == node.city:
            return node.data
        elif city < node.city:
            return self._search(node.left, city)
        else:
            return self._search(node.right, city)
    
    def inorder_traversal(self):
        """Perform inorder traversal of the BST to get sorted cities"""
        cities = []
        self._inorder_traversal(self.root, cities)
        return cities
    
    def _inorder_traversal(self, node, cities):
        """Helper function for recursively traversing the BST"""
        if node:
            self._inorder_traversal(node.left, cities)
            cities.append(node.city)
            self._inorder_traversal(node.right, cities)


class HashTable:
    """Hash table for O(1) city data lookup"""
    def __init__(self, size=101):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        """Hash function for string keys (city names)"""
        hash_value = 0
        for char in key:
            hash_value += ord(char)
        return hash_value % self.size
    
    def insert(self, key, value):
        """Insert a key-value pair into the hash table"""
        hash_value = self._hash(key)
        
        # Check if key already exists and update if it does
        for i, (k, v) in enumerate(self.table[hash_value]):
            if k == key:
                self.table[hash_value][i] = (key, value)
                return
        
        # If key doesn't exist, append it
        self.table[hash_value].append((key, value))
    
    def get(self, key):
        """Get value associated with key"""
        hash_value = self._hash(key)
        
        for k, v in self.table[hash_value]:
            if k == key:
                return v
        
        return None
    
    def keys(self):
        """Return all keys in the hash table"""
        keys = []
        for bucket in self.table:
            for k, _ in bucket:
                keys.append(k)
        return keys


def merge_sort(arr):
    """Implementation of merge sort algorithm for sorting city names"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left, right):
    """Helper function for merging two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


class Graph:
    """Graph representation for city and water source connections"""
    def __init__(self):
        self.vertices = set()
        self.edges = []
    
    def add_edge(self, u, v, weight):
        """Add an edge to the graph"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges.append((u, v, weight))
    
    def kruskal_mst(self):
        """Kruskal's algorithm to find minimum spanning tree"""
        # Initialize a forest where each vertex is a separate tree
        parent = {vertex: vertex for vertex in self.vertices}
        rank = {vertex: 0 for vertex in self.vertices}
        
        def find(x):
            """Find function for union-find data structure"""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            """Union function for union-find data structure"""
            root_x = find(x)
            root_y = find(y)
            
            if root_x == root_y:
                return
            
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                if rank[root_x] == rank[root_y]:
                    rank[root_x] += 1
        
        # Sort edges by weight
        sorted_edges = sorted(self.edges, key=lambda x: x[2])
        
        mst = []
        
        # Process edges in order of increasing weight
        for u, v, weight in sorted_edges:
            if find(u) != find(v):  # Check if adding this edge creates a cycle
                union(u, v)
                mst.append((u, v, weight))
        
        return mst


def multistage_graph_dp(stages, metrics):
    """
    Multistage Graph Dynamic Programming for optimizing environmental policies
    
    stages: number of stages (time periods)
    metrics: dictionary of metrics to consider (air, water, soil, etc.)
    """
    n = stages
    
    # Create a graph with n stages
    graph = {}
    
    # Initialize with costs between stages based on metrics
    for i in range(n):
        graph[i] = {}
        for j in range(i+1, min(i+3, n)):  # Connect with next 2 stages
            # Cost is weighted sum of metrics
            cost = sum(metrics.get(m, 1) * np.random.rand() for m in metrics)
            graph[i][j] = cost
    
    # Initialize cost and path arrays
    cost = [float('inf')] * n
    path = [-1] * n
    cost[0] = 0
    
    # Calculate minimum cost for each stage
    for i in range(n-1):
        for j in graph[i]:
            if cost[i] + graph[i][j] < cost[j]:
                cost[j] = cost[i] + graph[i][j]
                path[j] = i
    
    # Reconstruct the optimal path
    optimal_path = []
    cur = n - 1
    while cur != 0:
        optimal_path.append(cur)
        cur = path[cur]
    optimal_path.append(0)
    optimal_path.reverse()
    
    return optimal_path, cost[n-1]


class EnvironmentalDataManager:
    """Main class to manage environmental data using the implemented algorithms"""
    def __init__(self):
        self.city_bst = BST()
        self.city_hash = HashTable()
        self.water_graph = Graph()
        self.air_data = {}
        self.climate_data = {}
        self.soil_data = {}
        self.water_data = {}
    
    def load_data(self):
        """Load data from JSON files and organize using our data structures"""
        try:
            # Load data from JSON files
            with open('air_data_average.json', 'r') as f:
                self.air_data = json.load(f)
            
            with open('climate_data_average.json', 'r') as f:
                self.climate_data = json.load(f)
            
            with open('soil_data.json', 'r') as f:
                self.soil_data = json.load(f)
            
            with open('water_distance.json', 'r') as f:
                self.water_data = json.load(f)
            
            # Find common cities across all datasets
            cities = set(self.air_data.keys())
            cities = cities.intersection(set(self.climate_data.keys()))
            cities = cities.intersection(set(self.soil_data.keys()))
            cities = cities.intersection(set(self.water_data.keys()))
            
            # Build BST and hash table
            for city in cities:
                city_data = {
                    'air_data': self.air_data.get(city, {}),
                    'climate_data': self.climate_data.get(city, {}),
                    'soil_data': self.soil_data.get(city, {}),
                    'water_data': self.water_data.get(city, {})
                }
                self.city_bst.insert(city, city_data)
                self.city_hash.insert(city, city_data)
            
            # Build water graph for MST
            self._build_water_graph()
            
            return True
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return False
    
    def _build_water_graph(self):
        """Build a graph connecting cities and water sources for MST calculation"""
        cities = self.city_hash.keys()
        
        # Add edges between each pair of cities with water distance as weight
        for city1 in cities:
            for city2 in cities:
                if city1 != city2:
                    water_dist1 = self.water_data.get(city1, {}).get('distance', 0)
                    water_dist2 = self.water_data.get(city2, {}).get('distance', 0)
                    
                    # Weight is the distance between cities based on their water source distances
                    # This is a simplification for demonstration purposes
                    weight = abs(water_dist1 - water_dist2) + 10  # Add a base distance
                    
                    self.water_graph.add_edge(city1, city2, weight)
    
    def get_optimal_water_network(self):
        """Get minimum spanning tree for water network using Kruskal's algorithm"""
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
    
    def get_city_data(self, city):
        """Get data for a specific city using BST (O(log n) lookup)"""
        return self.city_bst.search(city)
    
    def get_city_data_hash(self, city):
        """Get data for a specific city using hash table (O(1) lookup)"""
        return self.city_hash.get(city)
    
    def get_sorted_cities(self):
        """Return list of cities sorted alphabetically using merge sort"""
        cities = self.city_hash.keys()
        return merge_sort(cities)
    
    def get_optimal_policy_path(self, city, metrics=None):
        """
        Calculate optimal policy path for a city using Multistage Graph DP
        
        metrics: dictionary with weights for different metrics (air, water, soil)
        """
        if metrics is None:
            metrics = {'air': 0.4, 'water': 0.3, 'soil': 0.3}
        
        city_data = self.get_city_data(city)
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
        Recommend best city for business based on environmental factors
        
        metrics: dictionary with weights for different metrics
        """
        if metrics is None:
            metrics = {'air': 0.4, 'water': 0.3, 'soil': 0.3}
        
        cities = self.city_hash.keys()
        best_city = None
        best_score = -float('inf')
        factors = {}
        
        for city in cities:
            city_data = self.get_city_data_hash(city)
            
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