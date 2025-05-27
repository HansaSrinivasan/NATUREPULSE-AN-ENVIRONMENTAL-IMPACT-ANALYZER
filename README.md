# NATUREPULSE-AN-ENVIRONMENTAL-IMPACT-ANALYZER

The objective of this project is to develop a web-based application that enables users to visualize, analyze, and compare environmental metrics such as air quality, water distribution, soil nutrients, and climate indicators across different cities. The system is designed to aid in decision-making for environmental planning, infrastructure development, and sustainable policy formulation.

NaturePulse processes structured environmental data and employs algorithmic design techniques to deliver efficient, real-time results. The application features an interactive dashboard, city comparison tools, and algorithm-based simulations for real-world scenarios. It helps simulate pipeline networks and policy strategies based on actual environmental metrics.

To optimize the infrastructure planning module, Kruskal’s Minimum Spanning Tree (MST) algorithm is implemented using a greedy approach to compute the most cost-effective water pipeline connections across cities. The Multi-Stage Graph algorithm (Dynamic Programming) is used to determine the best policy route based on environmental data. For efficient data retrieval, hash tables are implemented to allow constant-time access to city-specific environmental records. Additionally, merge sort is used to organize data alphabetically during exports.

The backend is built using Python with Flask, while the frontend uses HTML, CSS, and JavaScript, supported by libraries like Chart.js for data visualization. The application also allows exporting environmental data in Excel format, making it useful for reports and further analysis.

ALGORITHMS & TECHNIQUES USED:
    Kruskal’s MST (Greedy Algorithm) 
    Multi-Stage Graph (Dynamic Programming)
    Hashing (Custom Hash Table for fast data retrieval)
    Merge Sort (for sorting city data)
    Binary Search Tree (for city lookup)

TOOLS & TECHNOLOGIES USED:
    Python, Flask
    HTML, CSS, JavaScript
    JSON for dataset storage
    Chart.js for visualization
    Visual Studio Code for development
