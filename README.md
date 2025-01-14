Heterogeneous Graph Neural Network for Medical Applications
This repository contains the implementation of a graph neural network (GNN) designed to analyze relationships in a medical dataset. The project focuses on building a heterogeneous graph representation of medical entities (e.g., symptoms, diseases, drugs) and predicting relationships between them. This approach has potential applications in patient triage, drug discovery, and healthcare optimization.

Key Features
1. Heterogeneous Graph Construction
Node Types: Symptoms, Diseases, Drugs, DSI (Drug-Symptom Interactions), TC (Therapeutic Categories).
Edge Types: Multiple datasets were preprocessed to define connections between these nodes.
The graph uses PyTorch Geometric's HeteroData class, allowing for flexible and scalable heterogeneous graph representation.
2. Preprocessing Pipelines
Implements custom CSV loaders to map entities into indices and encode features.
Uses embeddings from the sentence-transformers library to create high-dimensional representations of textual data.
3. Graph Neural Network Design
Utilizes SAGEConv (GraphSAGE) for message passing and representation learning.
Incorporates multiple layers for heterogeneous data handling, enabling complex relationship modeling.
4. Application of State-of-the-Art Libraries
PyTorch Geometric: For graph representation and learning.
Sentence Transformers: For feature embeddings, improving node representation quality.
t-SNE Visualization: To interpret and validate the learned embeddings.
Project Highlights
This project showcases:

Data Science Expertise: Preprocessing and integrating disparate datasets into a unified graph.
Graph Machine Learning: Hands-on experience with graph neural networks and heterogeneous data handling.
Domain Knowledge: Applying AI techniques to solve healthcare-related problems.
Efficient Engineering: Reusable and scalable methods for node and edge processing.
Visualization: Using t-SNE for dimensionality reduction and interpretability of graph embeddings.
Installation
Ensure you have Python 3.8 or higher installed. Then, install the required dependencies:

bash
Copy code
pip install torch-geometric sentence-transformers openai==0.28 pandas matplotlib
Usage
Prepare the Data: Place the CSV files (D_Di_res.csv, Di_Sy_res.csv, SDSI_Sy_res.csv, SDSI_TC_res.csv) in the data/ directory.

Run the Script:

bash
Copy code
python main.py
The script will preprocess the data, build the graph, train the model, and visualize results.

Outputs:

Node embeddings for each type.
Predicted edges between nodes (e.g., symptom-to-disease predictions).
Visualizations of the graph embeddings.
Future Work
This project can be extended to:

Integrate additional medical datasets for richer graph representations.
Explore uncertainty estimation in predictions using Bayesian methods.
Deploy the model in healthcare applications for real-time decision-making.
