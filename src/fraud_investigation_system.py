"""
FRAUD INVESTIGATION VISUALIZATION SYSTEM
========================================

A comprehensive fraud detection and visualization system that creates
interactive investigation dashboards for insurance claims analysis.

This system identifies fraud patterns through graph analysis and provides
investigators with an intuitive interface for exploring suspicious relationships.

Author: Senior Software Engineer
Technology Stack: Python, NetworkX, PyVis, Pandas
"""

import pandas as pd
import networkx as nx
try:
    # PyVis is used to generate the interactive HTML network visualization.
    from pyvis.network import Network
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: 'pyvis'. Install it in your environment with: pip install pyvis"
    ) from e
from networkx.algorithms import community as nx_community
import numpy as np
import warnings
import webbrowser
import os
import sys
import json
from typing import Dict, List, Tuple, Set, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Ensure console printing does not fail on Windows terminals with limited encodings.
# We prefer UTF-8 output, and fall back to replacing unsupported characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class FraudInvestigationSystem:
    """
    Main class for the fraud investigation visualization system.

    This system provides end-to-end functionality for:
    - Loading and processing insurance claims data
    - Building fraud detection graphs
    - Identifying suspicious patterns
    - Creating interactive investigation dashboards
    """

    def __init__(self):
        """Initialize the fraud investigation system."""
        self.graph = None
        self.dataset = None
        self.metrics = None
        self.suspicious_nodes = None
        self.clusters = None
        self.investigation_subgraph = None

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the graph-based insurance claims dataset for visualization.

        Returns:
            pd.DataFrame: Loaded dataset with graph entity identifiers
        """
        print("=" * 80)
        print("STEP 1: LOADING GRAPH DATASET FOR VISUALIZATION")
        print("=" * 80)

        # Define the path to the graph dataset
        dataset_path = "data/processed/insurance_claims_graph_dataset.csv"

        try:
            # Load the dataset from CSV file
            self.dataset = pd.read_csv(dataset_path)

            # Display dataset summary
            print(f"✅ Graph dataset loaded successfully from: {dataset_path}")
            print(f"📊 Dataset shape: {self.dataset.shape}")
            print(f"📋 Number of claims: {len(self.dataset)}")
            print(f"🔢 Number of features: {self.dataset.shape[1]}")
            print()

            # Display graph entity statistics
            print("🏗️  Graph Entity Statistics:")
            print(f"  • Unique policies: {self.dataset['policy_id'].nunique()}")
            print(f"  • Unique vehicles: {self.dataset['vehicle_id'].nunique()}")
            print(f"  • Unique locations: {self.dataset['location_id'].nunique()}")
            print(f"  • Unique ZIP codes: {self.dataset['zip_id'].nunique()}")
            print()

            # Display fraud statistics (handle NaN values)
            fraud_data = self.dataset['fraud_reported'].dropna()
            if len(fraud_data) > 0:
                fraud_count = fraud_data.sum()
                fraud_rate = (fraud_count / len(fraud_data)) * 100
                print(f"🚨 Fraud cases in dataset: {fraud_count}")
                print(f"📈 Fraud rate: {fraud_rate:.1f}%")
            else:
                print("⚠️  No fraud data available in dataset")
            print()

            print("🔍 Graph analysis is useful in fraud detection because:")
            print("  • Fraud rings often involve repeated connections between entities")
            print("  • Collusive behavior creates unusual network patterns")
            print("  • Traditional ML models may miss organized fraud networks")
            print("  • Graph metrics can identify central players in fraud schemes")
            print()

            return self.dataset

        except FileNotFoundError:
            print(f"❌ Error: Dataset file not found at {dataset_path}")
            print("Please ensure the feature-engineered dataset exists.")
            raise
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

    def initialize_graph(self) -> nx.Graph:
        """
        Initialize a NetworkX graph object for fraud detection.

        Returns:
            nx.Graph: Empty NetworkX graph object
        """
        print("=" * 80)
        print("STEP 2: INITIALIZING FRAUD DETECTION GRAPH")
        print("=" * 80)

        # Create an undirected graph for fraud analysis
        # Undirected because relationships are bidirectional in fraud networks
        self.graph = nx.Graph()

        print("✅ NetworkX graph initialized successfully.")
        print("📐 Graph type: Undirected")
        print()
        print("🏗️  Graph structure:")
        print("  • Nodes will represent entities:")
        print("    - Policy holders (policy_id)")
        print("    - Vehicles (vehicle_id)")
        print("    - Accident locations (location_id)")
        print("    - Geographic areas (zip_id)")
        print("  • Edges will represent relationships between entities")
        print("  • Real-world entity names preserved for investigation")
        print()

        return self.graph

    def create_nodes(self, graph: nx.Graph, dataframe: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create nodes for different entities in the insurance claims data.

        Args:
            graph (nx.Graph): NetworkX graph object
            dataframe (pd.DataFrame): Insurance claims dataset

        Returns:
            Dict[str, List[str]]: Dictionary mapping entity types to their node identifiers
        """
        print("=" * 80)
        print("STEP 3: CREATING ENTITY NODES")
        print("=" * 80)

        # Initialize dictionary to store node identifiers by entity type
        entity_nodes = {
            "policy_holders": [],
            "vehicles": [],
            "locations": [],
            "zip_codes": [],
        }

        # Identify confirmed fraud policies so we can color their nodes red.
        # A policy is considered fraud if it has at least one claim with fraud_reported == 1.
        # Cast IDs to string so they are safe to use in both NetworkX and PyVis.
        fraud_policy_ids = set(
            str(x)
            for x in dataframe.loc[dataframe["fraud_reported"] == 1, "policy_id"].dropna().unique().tolist()
        )

        # -------------------------------------------------------------------
        # Policy holder nodes (node_id = policy_id)
        # Visualization rules:
        #   - color = blue (default)
        #   - label = policy_id
        #   - if fraud_reported == 1 for that policy, color = red
        # -------------------------------------------------------------------
        print("👤 Creating policy holder nodes...")
        for policy_id_raw in dataframe["policy_id"].dropna().unique():
            policy_id = str(policy_id_raw)
            node_id = f"policy:{policy_id}"
            base_color = "#3498db"  # Blue
            is_fraud_policy = policy_id in fraud_policy_ids
            color = "#e74c3c" if is_fraud_policy else base_color  # Red for confirmed fraud policies

            graph.add_node(
                node_id,
                entity_type="policy_holder",
                policy_id=policy_id,
                label=str(policy_id),
                base_color=base_color,
                color=color,
                fraud_policy=is_fraud_policy,
            )
            entity_nodes["policy_holders"].append(node_id)

        print(f"  ✅ Created {len(entity_nodes['policy_holders'])} policy holder nodes")

        # -------------------------------------------------------------------
        # Vehicle nodes (node_id = vehicle_id)
        # Visualization rule:
        #   - color = orange
        #   - label = auto_make + auto_model + auto_year
        # -------------------------------------------------------------------
        print("🚗 Creating vehicle nodes...")
        vehicle_rows = dataframe.dropna(subset=["vehicle_id"]).drop_duplicates(subset=["vehicle_id"])
        for _, row in vehicle_rows.iterrows():
            vehicle_id = str(row["vehicle_id"])
            node_id = f"vehicle:{vehicle_id}"

            make = str(row.get("auto_make", "")).strip()
            model = str(row.get("auto_model", "")).strip()
            year = str(row.get("auto_year", "")).strip()
            vehicle_label = " ".join([x for x in [make, model, year] if x and x != "nan"]).strip()
            if not vehicle_label:
                vehicle_label = str(vehicle_id)

            graph.add_node(
                node_id,
                entity_type="vehicle",
                vehicle_id=vehicle_id,
                auto_make=row.get("auto_make"),
                auto_model=row.get("auto_model"),
                auto_year=row.get("auto_year"),
                label=vehicle_label,
                base_color="#f39c12",  # Orange
                color="#f39c12",
            )
            entity_nodes["vehicles"].append(node_id)

        print(f"  ✅ Created {len(entity_nodes['vehicles'])} vehicle nodes")

        # -------------------------------------------------------------------
        # Location nodes (node_id = location_id)
        # Visualization rule:
        #   - color = green
        #   - label = incident_city + incident_state
        # -------------------------------------------------------------------
        print("📍 Creating location nodes...")
        location_rows = dataframe.dropna(subset=["location_id"]).drop_duplicates(subset=["location_id"])
        for _, row in location_rows.iterrows():
            location_id = str(row["location_id"])
            node_id = f"location:{location_id}"

            city = str(row.get("incident_city", "")).strip()
            state = str(row.get("incident_state", "")).strip()
            location_label = " ".join([x for x in [city, state] if x and x != "nan"]).strip()
            if not location_label:
                location_label = str(location_id)

            graph.add_node(
                node_id,
                entity_type="location",
                location_id=location_id,
                incident_city=row.get("incident_city"),
                incident_state=row.get("incident_state"),
                label=location_label,
                base_color="#27ae60",  # Green
                color="#27ae60",
            )
            entity_nodes["locations"].append(node_id)

        print(f"  ✅ Created {len(entity_nodes['locations'])} location nodes")

        # -------------------------------------------------------------------
        # ZIP code nodes (node_id = zip_id)
        # Visualization rule:
        #   - color = purple
        #   - label = insured_zip
        # -------------------------------------------------------------------
        print("📮 Creating ZIP code nodes...")
        zip_rows = dataframe.dropna(subset=["zip_id"]).drop_duplicates(subset=["zip_id"])
        for _, row in zip_rows.iterrows():
            zip_id = str(row["zip_id"])
            node_id = f"zip:{zip_id}"

            insured_zip = row.get("insured_zip")
            zip_label = str(insured_zip) if pd.notna(insured_zip) else str(zip_id)

            graph.add_node(
                node_id,
                entity_type="zip_code",
                zip_id=zip_id,
                insured_zip=insured_zip,
                label=zip_label,
                base_color="#9b59b6",  # Purple
                color="#9b59b6",
            )
            entity_nodes["zip_codes"].append(node_id)

        print(f"  ✅ Created {len(entity_nodes['zip_codes'])} ZIP code nodes")
        print()

        print(f"📊 Total nodes created: {graph.number_of_nodes()}")
        print()

        return entity_nodes

    def create_edges(self, graph: nx.Graph, dataframe: pd.DataFrame) -> None:
        """
        Add edges to represent relationships between entities in claims.
        Also compute shared entity scores to identify potential fraud rings.

        Args:
            graph (nx.Graph): NetworkX graph object
            dataframe (pd.DataFrame): Insurance claims dataset
        """
        print("=" * 80)
        print("STEP 4: CREATING RELATIONSHIP EDGES")
        print("=" * 80)

        # Track entity usage for shared entity score computation
        vehicle_usage = {}
        location_usage = {}
        zip_usage = {}

        # Add edges between policy holders and vehicles
        print("🔗 Creating policy-to-vehicle edges...")
        vehicle_edge_count = 0
        for _, row in dataframe.iterrows():
            policy_id = row['policy_id']
            vehicle_id = row['vehicle_id']

            # Skip if either ID is null
            if pd.isna(policy_id) or pd.isna(vehicle_id):
                continue

            # Cast node IDs to string for compatibility with PyVis
            policy_id = str(policy_id)
            vehicle_id = str(vehicle_id)

            # Use namespaced node IDs to avoid collisions between entity types
            policy_node = f"policy:{policy_id}"
            vehicle_node = f"vehicle:{vehicle_id}"

            # Track vehicle usage for shared entity score
            if vehicle_id not in vehicle_usage:
                vehicle_usage[vehicle_id] = []
            vehicle_usage[vehicle_id].append(policy_id)

            # Add edge if both nodes exist
            if graph.has_node(policy_node) and graph.has_node(vehicle_node):
                graph.add_edge(
                    policy_node,
                    vehicle_node,
                    relationship_type='policy_vehicle',
                    claim_amount=row['total_claim_amount'],
                    fraud_flag=row['fraud_reported']
                )
                vehicle_edge_count += 1

        print(f"  ✅ Added {vehicle_edge_count} policy-to-vehicle edges")

        # Add edges between policy holders and incident locations
        print("📍 Creating policy-to-location edges...")
        location_edge_count = 0
        for _, row in dataframe.iterrows():
            policy_id = row['policy_id']
            location_id = row['location_id']

            # Skip if either ID is null
            if pd.isna(policy_id) or pd.isna(location_id):
                continue

            # Cast node IDs to string for compatibility with PyVis
            policy_id = str(policy_id)
            location_id = str(location_id)

            # Use namespaced node IDs to avoid collisions between entity types
            policy_node = f"policy:{policy_id}"
            location_node = f"location:{location_id}"

            # Track location usage for shared entity score
            if location_id not in location_usage:
                location_usage[location_id] = []
            location_usage[location_id].append(policy_id)

            # Add edge if both nodes exist
            if graph.has_node(policy_node) and graph.has_node(location_node):
                graph.add_edge(
                    policy_node,
                    location_node,
                    relationship_type='policy_location',
                    claim_amount=row['total_claim_amount'],
                    fraud_flag=row['fraud_reported']
                )
                location_edge_count += 1

        print(f"  ✅ Added {location_edge_count} policy-to-location edges")

        # Add edges between policy holders and ZIP codes
        print("📮 Creating policy-to-ZIP edges...")
        zip_edge_count = 0
        for _, row in dataframe.iterrows():
            policy_id = row['policy_id']
            zip_id = row['zip_id']

            # Skip if either ID is null
            if pd.isna(policy_id) or pd.isna(zip_id):
                continue

            # Cast node IDs to string for compatibility with PyVis
            policy_id = str(policy_id)
            zip_id = str(zip_id)

            # Use namespaced node IDs to avoid collisions between entity types
            policy_node = f"policy:{policy_id}"
            zip_node = f"zip:{zip_id}"

            # Track ZIP usage for shared entity score
            if zip_id not in zip_usage:
                zip_usage[zip_id] = []
            zip_usage[zip_id].append(policy_id)

            # Add edge if both nodes exist
            if graph.has_node(policy_node) and graph.has_node(zip_node):
                graph.add_edge(
                    policy_node,
                    zip_node,
                    relationship_type='policy_zip',
                    claim_amount=row['total_claim_amount'],
                    fraud_flag=row['fraud_reported']
                )
                zip_edge_count += 1

        print(f"  ✅ Added {zip_edge_count} policy-to-ZIP edges")
        print()

        # Compute shared entity scores for fraud ring detection
        print("🔍 Computing shared entity scores...")
        print("-" * 50)

        # Update shared entity scores for vehicles
        high_score_vehicles = 0
        for vehicle_id, policy_holders in vehicle_usage.items():
            if len(policy_holders) > 1:  # Vehicle used by multiple policy holders
                shared_score = len(policy_holders) - 1  # Score based on number of additional users
                vehicle_node = f"vehicle:{vehicle_id}"
                if graph.has_node(vehicle_node):
                    graph.nodes[vehicle_node]['shared_entity_score'] = shared_score
                    high_score_vehicles += 1

        print(f"🚗 Vehicles shared by multiple policy holders: {high_score_vehicles}")

        # Update shared entity scores for locations
        high_score_locations = 0
        for location_id, policy_holders in location_usage.items():
            if len(policy_holders) > 1:  # Location used by multiple policy holders
                shared_score = len(policy_holders) - 1
                location_node = f"location:{location_id}"
                if graph.has_node(location_node):
                    graph.nodes[location_node]['shared_entity_score'] = shared_score
                    high_score_locations += 1

        print(f"📍 Locations shared by multiple policy holders: {high_score_locations}")

        # Update shared entity scores for ZIP codes
        high_score_zips = 0
        for zip_id, policy_holders in zip_usage.items():
            if len(policy_holders) > 1:  # ZIP connects multiple policy holders
                shared_score = len(policy_holders) - 1
                zip_node = f"zip:{zip_id}"
                if graph.has_node(zip_node):
                    graph.nodes[zip_node]['shared_entity_score'] = shared_score
                    high_score_zips += 1

        print(f"📮 ZIP codes with multiple policy holders: {high_score_zips}")
        print()

        total_edges = graph.number_of_edges()
        print(f"📊 Total edges in graph: {total_edges}")
        print()
        print("💡 Shared entity scores help identify fraud rings where:")
        print("  • Multiple policy holders use the same vehicle")
        print("  • Multiple incidents occur at the same location")
        print("  • Many policy holders share the same geographic area")
        print()

    def compute_graph_metrics(self, graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        Compute centrality metrics to identify important nodes in the fraud network.

        Args:
            graph (nx.Graph): NetworkX graph object

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing centrality metrics for all nodes
        """
        print("=" * 80)
        print("STEP 5: COMPUTING GRAPH METRICS")
        print("=" * 80)

        # Degree centrality: measures how connected each node is in the network.
        print("📊 Computing degree centrality...")
        degree_centrality = nx.degree_centrality(graph)

        # Betweenness centrality: measures how often a node sits on shortest paths.
        print("🌉 Computing betweenness centrality...")
        betweenness_centrality = nx.betweenness_centrality(graph)

        # Store only the required metrics for investigation tooltips and suspicious detection.
        self.metrics = {
            "degree": degree_centrality,
            "betweenness": betweenness_centrality,
        }

        print("✅ Graph metrics computed successfully.")
        print()

        return self.metrics

    def detect_suspicious_entities(self, graph: nx.Graph, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Identify nodes with unusually high centrality values.

        Args:
            graph (nx.Graph): NetworkX graph object
            metrics (Dict[str, Dict[str, float]]): Centrality metrics for all nodes

        Returns:
            List[str]: List of suspicious node identifiers
        """
        print("=" * 80)
        print("STEP 6: DETECTING SUSPICIOUS ENTITIES")
        print("=" * 80)

        # Requirement: suspicious nodes are those with degree_centrality > 0.05.
        threshold = 0.05
        print(f"🔍 Flagging nodes with degree_centrality > {threshold:.2f} ...")

        suspicious_nodes = [node for node, value in metrics["degree"].items() if value > threshold]

        # Store for visualization highlighting (yellow).
        self.suspicious_nodes = suspicious_nodes

        print(f"🚨 Suspicious nodes detected: {len(self.suspicious_nodes)}")
        print()

        return self.suspicious_nodes

    def detect_clusters(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Use community detection to identify tightly connected clusters.

        Args:
            graph (nx.Graph): NetworkX graph object

        Returns:
            Dict[str, Any]: Dictionary containing cluster information
        """
        print("=" * 80)
        print("STEP 7: DETECTING FRAUD CLUSTERS")
        print("=" * 80)

        # Use the Louvain algorithm for community detection
        print("🔍 Running Louvain community detection...")
        try:
            detected_communities = nx_community.louvain_communities(graph, seed=42)
        except Exception:
            # Fallback to greedy modularity communities if Louvain fails
            print("  ⚠️  Louvain failed, using greedy modularity communities")
            detected_communities = nx_community.greedy_modularity_communities(graph)

        print(f"✅ Number of communities detected: {len(detected_communities)}")

        # Analyze community sizes
        community_sizes = [len(community) for community in detected_communities]
        print(f"📊 Community size range: {min(community_sizes)} to {max(community_sizes)}")
        print(f"📈 Average community size: {np.mean(community_sizes):.1f}")

        # Identify suspicious clusters (unusually large communities - top 10%)
        size_threshold = np.percentile(community_sizes, 90)
        suspicious_clusters = []

        print(f"🎯 Cluster size threshold (90th percentile): {size_threshold:.1f}")

        for i, community in enumerate(detected_communities):
            if len(community) >= size_threshold:
                # Analyze entity types in this community
                entity_types = {}
                for node in community:
                    node_attrs = graph.nodes[node]
                    entity_type = node_attrs.get('entity_type', 'unknown')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

                suspicious_clusters.append({
                    'cluster_id': i,
                    'size': len(community),
                    'nodes': list(community),
                    'entity_types': entity_types
                })

        print(f"🚨 Number of suspicious clusters (top 10% largest): {len(suspicious_clusters)}")
        print()
        print("💡 How fraud rings may appear as tightly connected clusters:")
        print("  • Groups of policy holders sharing vehicles")
        print("  • Clusters of claims at same locations")
        print("  • Networks of entities in geographic proximity")
        print()

        # Store cluster information
        self.clusters = {
            'all_communities': list(detected_communities),
            'suspicious_clusters': suspicious_clusters,
            'size_threshold': size_threshold
        }

        return self.clusters

    def build_investigation_subgraph(self, graph: nx.Graph, suspicious_nodes: List[str], max_nodes: int = 60) -> nx.Graph:
        """
        Create a focused investigation subgraph for visualization.

        Args:
            graph (nx.Graph): Original NetworkX graph object
            suspicious_nodes (List[str]): List of suspicious node identifiers
            max_nodes (int): Maximum number of nodes for the investigation graph

        Returns:
            nx.Graph: Focused investigation subgraph
        """
        print("=" * 80)
        print("STEP 8: BUILDING INVESTIGATION SUBGRAPH")
        print("=" * 80)

        # Build investigation subgraph with top suspicious nodes and their neighbors
        nodes_to_include = set()

        # Take top suspicious nodes (prioritize by degree centrality)
        suspicious_with_centrality = []
        for node in suspicious_nodes:
            if node in self.metrics['degree']:
                suspicious_with_centrality.append((node, self.metrics['degree'][node]))

        # Sort by centrality and take top nodes
        suspicious_with_centrality.sort(key=lambda x: x[1], reverse=True)
        top_suspicious = [node for node, _ in suspicious_with_centrality[:20]]
        nodes_to_include.update(top_suspicious)

        # Add immediate neighbors of top suspicious nodes
        for suspicious_node in top_suspicious:
            if suspicious_node in graph.nodes():
                neighbors = list(graph.neighbors(suspicious_node))
                # Add up to 2 neighbors per suspicious node to keep size manageable
                nodes_to_include.update(neighbors[:2])

        # Limit total nodes to prevent clutter
        if len(nodes_to_include) > max_nodes:
            nodes_to_include = set(list(nodes_to_include)[:max_nodes])

        # Create investigation subgraph
        self.investigation_subgraph = graph.subgraph(nodes_to_include)

        print(f"🎯 Building investigation graph with {len(nodes_to_include)} nodes")
        print("📊 Including top suspicious nodes and their immediate connections")
        print("🔍 Focused view enables detailed fraud pattern analysis")
        print()

        return self.investigation_subgraph

    def create_interactive_investigation_graph(self, subgraph: nx.Graph) -> str:
        """
        Create an interactive fraud investigation graph using PyVis.

        Args:
            subgraph (nx.Graph): Investigation graph (full graph or focused subgraph)

        Returns:
            str: Path to the generated HTML file
        """
        print("=" * 80)
        print("STEP 9: CREATING INTERACTIVE FRAUD INVESTIGATION NETWORK")
        print("=" * 80)

        # Initialize PyVis network (we will inject our own investigator UI controls,
        # so we disable the default PyVis menus).
        pyvis_network = Network(
            height='100%',
            width='100%',
            bgcolor='#ffffff',
            font_color='black',
            notebook=False,
            cdn_resources='in_line',
            select_menu=False,
            filter_menu=False
        )

        # Configure physics: optimized for faster initial load with better layout
        physics_options = """
        var options = {
          "physics": {
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {
              "gravitationalConstant": -1500,
              "centralGravity": 0.4,
              "springLength": 100,
              "springConstant": 0.03
            },
            "maxVelocity": 50,
            "stabilization": {
              "iterations": 200,
              "fit": true
            },
            "timestep": 0.5
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "zoomSpeed": 0.5,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          },
          "layout": {
            "improvedLayout": true
          }
        }
        """
        pyvis_network.set_options(physics_options)

        # Identify suspicious nodes (degree_centrality > 0.05) for investigator highlighting.
        suspicious_set = set(self.suspicious_nodes or [])

        # Pre-compute investigator filter sets.
        # These sets are exported to the browser so investigators can filter interactively.
        fraud_edge_keys: Set[Tuple[str, str]] = set()
        fraud_connected_nodes: Set[str] = set()
        shared_vehicle_nodes: Set[str] = set()
        frequent_location_nodes: Set[str] = set()

        # Fraud edges / fraud-connected nodes (fraud_reported == 1)
        for u, v, data in subgraph.edges(data=True):
            if int(data.get("fraud_flag", 0)) == 1:
                fraud_edge_keys.add(tuple(sorted([u, v])))
                fraud_connected_nodes.add(u)
                fraud_connected_nodes.add(v)

        # Shared vehicles: vehicle nodes used by multiple policies (shared_entity_score > 0)
        for node, attrs in subgraph.nodes(data=True):
            if attrs.get("entity_type") == "vehicle" and float(attrs.get("shared_entity_score", 0)) > 0:
                shared_vehicle_nodes.add(node)

        # Frequent accident locations: locations with more than N incidents.
        # An incident corresponds to a policy-location relationship, so degree() is an incident count.
        for node, attrs in subgraph.nodes(data=True):
            if attrs.get("entity_type") == "location":
                incident_count = subgraph.degree(node)
                # Store on node for the tooltip and client-side filtering.
                attrs["incident_count"] = incident_count
                if incident_count > 5:  # Default N for initial view; UI lets investigators change this.
                    frequent_location_nodes.add(node)

        # Add nodes with required visualization rules:
        #   - Shapes by entity type
        #   - Sizes by degree centrality
        #   - Colors by entity type with fraud policy override
        #   - Suspicious highlighting by yellow border (NOT fill)
        print("🎨 Adding nodes with required visualization rules...")
        for node in subgraph.nodes():
            node_attrs = subgraph.nodes[node]
            entity_type = node_attrs.get("entity_type", "unknown")

            # Centrality metrics for tooltips
            degree_centrality = self.metrics["degree"].get(node, 0.0)
            betweenness_centrality = self.metrics["betweenness"].get(node, 0.0)

            # Base color is stored on the node during node creation (blue/orange/green/purple)
            base_color = node_attrs.get("base_color", "#95a5a6")

            # Apply fraud highlighting: confirmed fraud policy nodes are red.
            if entity_type == "policy_holder" and node_attrs.get("fraud_policy", False):
                color = "#e74c3c"  # Red
            else:
                color = base_color

            # Node label is stored on the node during node creation.
            label = node_attrs.get("label", str(node))

            # Shapes by entity type (investigator-friendly).
            if entity_type == "policy_holder":
                shape = "dot"       # circle
            elif entity_type == "vehicle":
                shape = "triangle"
            elif entity_type == "location":
                shape = "square"
            elif entity_type == "zip_code":
                shape = "diamond"
            else:
                shape = "dot"

            # Extract the human-readable entity identifier for tooltips.
            if entity_type == "policy_holder":
                entity_id = node_attrs.get("policy_id", "")
            elif entity_type == "vehicle":
                entity_id = node_attrs.get("vehicle_id", "")
            elif entity_type == "location":
                entity_id = node_attrs.get("location_id", "")
            elif entity_type == "zip_code":
                entity_id = node_attrs.get("zip_id", "")
            else:
                entity_id = ""

            # Node size reflects importance: size = 12 + degree_centrality * 200 (capped at 45).
            node_size = 12 + (degree_centrality * 200)
            node_size = min(node_size, 45)

            # Shared entity score is computed during edge creation (e.g., reused vehicle/location/zip).
            shared_score = float(node_attrs.get("shared_entity_score", 0))

            # Number of connections = node degree in the graph.
            connections = subgraph.degree(node)

            # Tooltip shows the required investigation fields for analysts.
            # Format: clean text-based (HTML rendering is unreliable in tooltips)
            if entity_type == "policy_holder":
                tooltip = (
                    "POLICY HOLDER\n"
                    "─────────────\n"
                    f"Policy ID: {entity_id}\n"
                    f"Name: {label}\n"
                    f"Connections: {connections} relationships\n"
                    f"Importance: {degree_centrality:.1%}\n"
                    f"Fraud Risk: {node_attrs.get('fraud_policy', 'No')}\n"
                    f"Shared Entities: {shared_score:.0f}"
                )
            elif entity_type == "vehicle":
                tooltip = (
                    "VEHICLE\n"
                    "───────\n"
                    f"Vehicle ID: {entity_id}\n"
                    f"Type: {label}\n"
                    f"Shared by: {connections} policies\n"
                    f"Risk Factor: {degree_centrality:.1%}\n"
                    f"Frequency Score: {shared_score:.0f}"
                )
            elif entity_type == "location":
                tooltip = (
                    "LOCATION\n"
                    "────────\n"
                    f"Location ID: {entity_id}\n"
                    f"Name: {label}\n"
                    f"Claims: {connections} reported\n"
                    f"Incident Rate: {degree_centrality:.1%}\n"
                    f"Frequency: {int(node_attrs.get('incident_count', 0))} incidents"
                )
            elif entity_type == "zip_code":
                tooltip = (
                    "ZIP CODE\n"
                    "────────\n"
                    f"ZIP: {entity_id}\n"
                    f"Area: {label}\n"
                    f"Activity: {connections} claims\n"
                    f"Concentration: {degree_centrality:.1%}\n"
                    f"Risk Level: {shared_score:.0f}/10"
                )
            else:
                tooltip = (
                    "UNKNOWN ENTITY\n"
                    "──────────────\n"
                    f"Type: {entity_type}\n"
                    f"ID: {entity_id}\n"
                    f"Label: {label}\n"
                    f"Connections: {connections}"
                )

            # Suspicious nodes (degree_centrality > 0.05) get a yellow border and thicker outline.
            if node in suspicious_set:
                border_width = 4
                border_color = "#f1c40f"  # Yellow border
            else:
                border_width = 1
                border_color = "#2c3e50"  # Dark outline

            pyvis_network.add_node(
                node,
                label=label,
                color=color,
                size=node_size,
                title=tooltip,
                shape=shape,
                borderWidth=border_width,
                borderWidthSelected=border_width,
                borderColor=border_color,
                # Export attributes for client-side investigator filters.
                entity_type=entity_type,
                entity_id=str(entity_id),
                is_fraud_policy=bool(node_attrs.get("fraud_policy", False)),
                is_suspicious=bool(node in suspicious_set),
                is_shared_vehicle=bool(node in shared_vehicle_nodes),
                is_frequent_location=bool(node in frequent_location_nodes),
                incident_count=int(node_attrs.get("incident_count", 0)),
                shared_entity_score=float(node_attrs.get("shared_entity_score", 0)),
                degree_centrality=float(degree_centrality),
                betweenness_centrality=float(betweenness_centrality),
                font={"size": 12, "color": "black"},
            )

        # Add edges with fraud-specific styling
        print("🔗 Adding edges with fraud-specific styling...")
        for source, target, edge_data in subgraph.edges(data=True):
            fraud_flag = edge_data.get('fraud_flag', 0)
            claim_amount = edge_data.get('claim_amount', 0)

            # Edge styling rules:
            #   - Fraud edges: red, width 3
            #   - Normal edges: light grey, width 1
            if fraud_flag:
                edge_color = '#e74c3c'
                edge_width = 3
            else:
                edge_color = '#d0d3d4'
                edge_width = 1

            # Edge tooltip for investigators (supports the high-claim filter).
            edge_title = f"Claim Amount: ${claim_amount:,.2f}\nFraud Flag: {int(fraud_flag)}"

            pyvis_network.add_edge(
                source,
                target,
                color=edge_color,
                width=edge_width,
                title=edge_title,
                # Export edge attributes for client-side filters.
                fraud_flag=int(fraud_flag),
                claim_amount=float(claim_amount),
                hoverWidth=3
            )

        # Generate HTML content
        html_content = pyvis_network.generate_html()

        # ---------------------------------------------------------------
        # Structured UI layout (Title / Left panel / Center graph / Legend).
        # We inject CSS + HTML + JavaScript controls into the PyVis HTML.
        # ---------------------------------------------------------------

        # Title (top)
        title_html = """
<div id="icins-title" style="
  position: fixed;
  left: 0;
  right: 0;
  top: 0;
  height: 56px;
  z-index: 9999;
  display: flex;
  align-items: center;
  padding: 0 16px;
  background: #0b1320;
  color: #ffffff;
  font-family: 'Segoe UI', Arial, sans-serif;
  font-size: 18px;
  font-weight: 700;
  box-shadow: 0 2px 10px rgba(0,0,0,0.2);
">
  Insurance Fraud Investigation Network
</div>
"""

        # Investigation filter panel (left) - full height
        panel_html = f"""
<div id="icins-panel" style="
  position: fixed;
  left: 12px;
  top: 56px;
  bottom: 0;
  width: 320px;
  z-index: 9999;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 14px 14px 10px 14px;
  font-family: 'Segoe UI', Arial, sans-serif;
  font-size: 13px;
  overflow: auto;
  box-shadow: 0 6px 24px rgba(0,0,0,0.10);
">
  <button onclick="window.location.href = '/';" style="
    width: 100%;
    padding: 10px;
    margin-bottom: 12px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.3s;
  " onmouseover="this.style.background='#5568d3'" onmouseout="this.style.background='#667eea'">
    ← Back to Dashboard
  </button>

  <div style="font-weight:700; font-size:14px; margin-bottom:10px;">Investigation Filters</div>

  <label style="display:block; margin: 8px 0;">
    <input type="checkbox" id="f_showFraud" /> <b>Show Fraud Cases</b><br>
    <span style="color:#6b7280;">Display only nodes connected to fraud_reported = 1.</span>
  </label>

  <label style="display:block; margin: 8px 0;">
    <input type="checkbox" id="f_sharedVehicles" /> <b>Shared Vehicles</b><br>
    <span style="color:#6b7280;">Vehicles used by multiple policies (possible rings).</span>
  </label>

  <label style="display:block; margin: 8px 0;">
    <input type="checkbox" id="f_suspiciousNodes" /> <b>Suspicious Nodes</b><br>
    <span style="color:#6b7280;">degree_centrality &gt; </span>
    <input type="number" id="f_degreeThreshold" value="0.05" step="0.01" min="0" max="1"
           style="width:90px; padding:4px; margin-left:6px; border:1px solid #e5e7eb; border-radius:6px;" />
  </label>

  <div style="margin: 12px 0 6px 0;"><b>Frequent Accident Locations</b></div>
  <div style="color:#6b7280; margin-bottom:6px;">Locations with more than N incidents.</div>
  <div style="display:flex; gap:8px; align-items:center; margin-bottom:10px;">
    <input type="checkbox" id="f_frequentLocations" />
    <span>N &gt;</span>
    <input type="number" id="f_locationN" value="5" step="1" min="1" max="9999"
           style="width:90px; padding:4px; border:1px solid #e5e7eb; border-radius:6px;" />
  </div>

  <div style="margin: 12px 0 6px 0;"><b>High Claim Amount Connections</b></div>
  <div style="color:#6b7280; margin-bottom:6px;">Edges where claim_amount &gt; threshold.</div>
  <div style="display:flex; gap:8px; align-items:center;">
    <input type="checkbox" id="f_highClaimEdges" />
    <span>$</span>
    <input type="number" id="f_claimThreshold" value="50000" step="5000" min="0"
           style="width:120px; padding:4px; border:1px solid #e5e7eb; border-radius:6px;" />
  </div>

  <hr style="margin: 14px 0; border: none; border-top: 1px solid #eef2f7;" />

  <div style="margin: 12px 0 6px 0;"><b>Search by Policy ID</b></div>
  <input type="text" id="f_policySearch" placeholder="e.g., POL-001" 
         style="width:100%; padding:8px; border:1px solid #e5e7eb; border-radius:6px; margin-bottom:12px;" />

  <div style="margin: 12px 0 6px 0;"><b>Search by Vehicle Name</b></div>
  <input type="text" id="f_vehicleSearch" placeholder="e.g., SUV, Sedan" 
         style="width:100%; padding:8px; border:1px solid #e5e7eb; border-radius:6px; margin-bottom:12px;" />

  <div style="margin: 12px 0 6px 0;"><b>Search by Location</b></div>
  <input type="text" id="f_locationSearch" placeholder="e.g., Downtown, Airport" 
         style="width:100%; padding:8px; border:1px solid #e5e7eb; border-radius:6px; margin-bottom:12px;" />

  <div style="margin: 12px 0 6px 0;"><b>Search by ZIP Code</b></div>
  <input type="text" id="f_zipSearch" placeholder="e.g., 12345" 
         style="width:100%; padding:8px; border:1px solid #e5e7eb; border-radius:6px; margin-bottom:12px;" />

  <hr style="margin: 14px 0; border: none; border-top: 1px solid #eef2f7;" />

  <div style="margin-bottom:8px;"><b>Maximum nodes to display</b></div>
  <input type="range" id="f_nodeLimit" min="10" max="{max(10, subgraph.number_of_nodes())}" value="{min(500, subgraph.number_of_nodes())}" step="10"
         style="width: 100%;" />
  <div style="display:flex; justify-content:space-between; color:#6b7280; margin-top:6px;">
    <span>10</span>
    <span id="f_nodeLimitValue">{min(500, subgraph.number_of_nodes())}</span>
    <span>{subgraph.number_of_nodes()}</span>
  </div>

  <hr style="margin: 14px 0; border: none; border-top: 1px solid #eef2f7;" />

  <div style="margin-bottom:12px;"><b>Physics Control</b></div>
  <div style="display:flex; gap:8px; margin-bottom:12px;">
    <button id="btn_freezeLayout" onclick="freezeLayout_handler()" style="
      flex:1;
      padding:10px 12px;
      border:none;
      border-radius:8px;
      background:#eab308;
      color:#1a1a1a;
      font-weight:700;
      cursor:pointer;
      transition: background 0.2s;
    ">❄️ Freeze Layout</button>
    <button id="btn_enablePhysics" onclick="enablePhysics_handler()" style="
      flex:1;
      padding:10px 12px;
      border:none;
      border-radius:8px;
      background:#22c55e;
      color:white;
      font-weight:700;
      cursor:pointer;
      transition: background 0.2s;
    ">▶️ Enable Physics</button>
  </div>

  <div style="margin-top: 12px; display:flex; gap:10px;">
    <button id="btn_applyFilters" onclick="applyFilters_handler()" style="
      flex:1;
      padding:10px 12px;
      border:none;
      border-radius:10px;
      background:#2563eb;
      color:white;
      font-weight:700;
      cursor:pointer;
    ">Apply filters</button>
    <button id="btn_resetFilters" onclick="resetFilters_handler()" style="
      padding:10px 12px;
      border:1px solid #e5e7eb;
      border-radius:10px;
      background:white;
      font-weight:700;
      cursor:pointer;
    ">Reset</button>
  </div>
</div>
"""

        # Legend (top-right, not overlapping the panel controls)
        legend_html = """
<div id="icins-legend" style="
  position: fixed;
  right: 12px;
  top: 72px;
  width: 280px;
  z-index: 9999;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px 14px;
  font-family: 'Segoe UI', Arial, sans-serif;
  font-size: 13px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.10);
">
  <div style="font-weight: 700; margin-bottom: 8px;">Legend</div>
  <div style="display:flex; align-items:center; margin:4px 0;">
    <span style="display:inline-block;width:12px;height:12px;background:#3498db;border-radius:50%;margin-right:8px;"></span>
    Blue = Policy Holder
  </div>
  <div style="display:flex; align-items:center; margin:4px 0;">
    <span style="display:inline-block;width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:11px solid #f39c12;margin-right:8px;"></span>
    Orange = Vehicle
  </div>
  <div style="display:flex; align-items:center; margin:4px 0;">
    <span style="display:inline-block;width:12px;height:12px;background:#27ae60;margin-right:8px;"></span>
    Green = Location
  </div>
  <div style="display:flex; align-items:center; margin:4px 0;">
    <span style="display:inline-block;width:12px;height:12px;background:#9b59b6;transform:rotate(45deg);margin-right:8px;"></span>
    Purple = ZIP Code
  </div>
  <div style="display:flex; align-items:center; margin:4px 0;">
    <span style="display:inline-block;width:12px;height:12px;background:#e74c3c;border-radius:50%;margin-right:8px;"></span>
    Red = Confirmed Fraud
  </div>
  <div style="display:flex; align-items:center; margin:4px 0;">
    <span style="display:inline-block;width:12px;height:12px;border:3px solid #f1c40f;border-radius:50%;margin-right:8px;"></span>
    Yellow Border = Suspicious Node
  </div>
</div>
"""

        # Polyfill for ugt.clearMarks (prevents JavaScript errors)
        polyfill_html = """
<script>
  // Polyfill for performance measurement APIs that some libraries might use
  window.ugt = window.ugt || {};
  window.ugt.clearMarks = window.ugt.clearMarks || function() {};
  window.ugt.mark = window.ugt.mark || function() {};
  window.ugt.measure = window.ugt.measure || function() {};
</script>
"""

        # CSS to position graph canvas with proper layout (title + left panel)
        css_html = """
<style>
  html {
    height: 100%;
  }
  body { 
    height: 100%;
    margin: 0; 
    padding: 0; 
    overflow: hidden;
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  #mynetwork {
    position: fixed;
    top: 56px;
    left: 356px;
    right: 0;
    bottom: 0;
    width: calc(100% - 356px);
    height: calc(100% - 56px);
    border-left: 1px solid #eef2f7;
  }
  #mynetwork canvas {
    width: 100% !important;
    height: 100% !important;
  }
  /* Center and style the PyVis loading progress bar */
  .vis-loading-bar {
    position: fixed;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: 350px !important;
    height: 50px !important;
    background: #ffffff !important;
    border-radius: 25px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
    z-index: 10000 !important;
    display: flex !important;
    align-items: center !important;
    padding: 8px !important;
    box-sizing: border-box !important;
  }
  .vis-progress {
    height: 100% !important;
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    border-radius: 20px !important;
    transition: width 0.3s ease !important;
  }
  .vis-bar {
    display: none !important;
  }
</style>
"""

        # Custom investigation filtering logic (replaces PyVis default filter UI).
        # We filter edges first, then keep nodes that are endpoints of visible edges.
        js_html = f"""
<script>
  // Global handler functions for button clicks
  function freezeLayout_handler() {{
    if (typeof network !== 'undefined' && network) {{
      network.setOptions({{ physics: {{ enabled: false }} }});
      const freezeBtn = document.getElementById('btn_freezeLayout');
      const physicsBtn = document.getElementById('btn_enablePhysics');
      if (freezeBtn) freezeBtn.style.background = '#d4a500';
      if (physicsBtn) physicsBtn.style.background = '#b3e5fc';
      console.log('Physics frozen');
    }}
  }}

  function enablePhysics_handler() {{
    if (typeof network !== 'undefined' && network) {{
      network.setOptions({{ physics: {{ enabled: true }} }});
      const freezeBtn = document.getElementById('btn_freezeLayout');
      const physicsBtn = document.getElementById('btn_enablePhysics');
      if (freezeBtn) freezeBtn.style.background = '#eab308';
      if (physicsBtn) physicsBtn.style.background = '#22c55e';
      console.log('Physics enabled');
    }}
  }}

  // Global state for filters
  let allNodes_global = [];
  let allEdges_global = [];
  let FRAUD_CONNECTED_global = new Set();
  let SHARED_VEHICLES_global = new Set();

  function getChecked(id) {{
    const el = document.getElementById(id);
    return el ? el.checked : false;
  }}

  function getNumber(id, fallback) {{
    const el = document.getElementById(id);
    if (!el) return fallback;
    const v = parseFloat(el.value);
    return Number.isFinite(v) ? v : fallback;
  }}

  function updateSliderLabel() {{
    const slider = document.getElementById('f_nodeLimit');
    const label = document.getElementById('f_nodeLimitValue');
    if (slider && label) label.textContent = slider.value;
  }}

  function applyFilteredGraph(filteredNodes, filteredEdges) {{
    if (typeof nodes === 'undefined' || typeof edges === 'undefined' || typeof network === 'undefined') {{
      console.error('PyVis objects not available');
      return;
    }}
    
    if (!Array.isArray(filteredNodes) || filteredNodes.length === 0) {{
      console.warn('No nodes to display');
      return;
    }}
    
    if (!Array.isArray(filteredEdges)) {{
      console.warn('Invalid edges data');
      filteredEdges = [];
    }}

    try {{
      // Make deep copies to avoid reference issues
      const nodeCopies = filteredNodes.map(n => ({{...n}}));
      const edgeCopies = filteredEdges.map(e => ({{...e}}));
      
      console.log('Updating graph with', nodeCopies.length, 'nodes and', edgeCopies.length, 'edges');
      
      nodes.clear();
      edges.clear();
      nodes.add(nodeCopies);
      edges.add(edgeCopies);
      
      // Fit network to nodes
      setTimeout(() => {{
        try {{ network.fit({{ animation: false }}); }} catch (e) {{ console.warn('Fit error:', e); }}
      }}, 100);
    }} catch (e) {{
      console.error('Error updating graph:', e);
    }}
  }}

  function resetFilters() {{
    const ids = [
      'f_showFraud','f_sharedVehicles','f_suspiciousNodes',
      'f_frequentLocations','f_highClaimEdges'
    ];
    ids.forEach(x => {{ const el = document.getElementById(x); if (el) el.checked = false; }});
    const degree = document.getElementById('f_degreeThreshold'); if (degree) degree.value = '0.05';
    const locN = document.getElementById('f_locationN'); if (locN) locN.value = '5';
    const claimT = document.getElementById('f_claimThreshold'); if (claimT) claimT.value = '50000';
    const slider = document.getElementById('f_nodeLimit'); if (slider) slider.value = Math.min(500, allNodes_global.length);
    updateSliderLabel();
    applyFilteredGraph(allNodes_global, allEdges_global);
  }}

  function applyFilters() {{
    updateSliderLabel();

    const showFraud = getChecked('f_showFraud');
    const sharedVehicles = getChecked('f_sharedVehicles');
    const suspiciousOnly = getChecked('f_suspiciousNodes');
    const frequentLocs = getChecked('f_frequentLocations');
    const highClaimEdges = getChecked('f_highClaimEdges');
    
    // New text search filters
    const vehicleSearch = (document.getElementById('f_vehicleSearch')?.value || '').toLowerCase();
    const locationSearch = (document.getElementById('f_locationSearch')?.value || '').toLowerCase();
    const zipSearch = (document.getElementById('f_zipSearch')?.value || '').toLowerCase();
    const policySearch = (document.getElementById('f_policySearch')?.value || '').toLowerCase();

    // If only text searches and nothing is filled in, show full graph
    const anyCheckboxActive = showFraud || sharedVehicles || suspiciousOnly || frequentLocs || highClaimEdges;
    const anyTextSearch = vehicleSearch.length > 0 || locationSearch.length > 0 || zipSearch.length > 0 || policySearch.length > 0;
    
    if (!anyCheckboxActive && !anyTextSearch) {{
      console.log('No filters selected, showing full graph');
      applyFilteredGraph(allNodes_global, allEdges_global);
      return;
    }}

    const degreeThreshold = getNumber('f_degreeThreshold', 0.05);
    const locationN = getNumber('f_locationN', 5);
    const claimThreshold = getNumber('f_claimThreshold', 50000);
    const nodeLimit = parseInt(getNumber('f_nodeLimit', 500), 10);

    let filteredEdges = allEdges_global.filter(e => {{
      if (showFraud && !(e.fraud_flag === 1)) return false;
      if (highClaimEdges && !(e.claim_amount > claimThreshold)) return false;
      return true;
    }});

    const nodeSet = new Set();
    filteredEdges.forEach(e => {{ nodeSet.add(e.from); nodeSet.add(e.to); }});

    let filteredNodes = allNodes_global.filter(n => nodeSet.has(n.id));

    // Apply text searches
    if (vehicleSearch.length > 0) {{
      filteredNodes = filteredNodes.filter(n => {{
        const nodeLabel = (n.label || '').toLowerCase();
        const vehicleType = (n.vehicle_type || '').toLowerCase();
        return nodeLabel.includes(vehicleSearch) || vehicleType.includes(vehicleSearch);
      }});
    }}

    if (locationSearch.length > 0) {{
      filteredNodes = filteredNodes.filter(n => {{
        const location = (n.location || '').toLowerCase();
        return location.includes(locationSearch);
      }});
    }}

    if (zipSearch.length > 0) {{
      filteredNodes = filteredNodes.filter(n => {{
        const zip = (n.zip_code || '').toLowerCase();
        return zip.includes(zipSearch);
      }});
    }}

    if (policySearch.length > 0) {{
      filteredNodes = filteredNodes.filter(n => {{
        const policy = (n.policy_id || '').toLowerCase();
        const policyNum = (n.policy_number || '').toLowerCase();
        return policy.includes(policySearch) || policyNum.includes(policySearch);
      }});
    }}

    if (showFraud) {{
      filteredNodes = filteredNodes.filter(n => FRAUD_CONNECTED_global.has(n.id));
    }}

    if (sharedVehicles) {{
      const keep = new Set();
      filteredEdges.forEach(e => {{
        if (SHARED_VEHICLES_global.has(e.from) || SHARED_VEHICLES_global.has(e.to)) {{
          keep.add(e.from); keep.add(e.to);
        }}
      }});
      filteredNodes = filteredNodes.filter(n => keep.has(n.id));
      filteredEdges = filteredEdges.filter(e => keep.has(e.from) && keep.has(e.to));
    }}

    if (suspiciousOnly) {{
      filteredNodes = filteredNodes.filter(n => (n.degree_centrality || 0) > degreeThreshold);
      const keep = new Set(filteredNodes.map(n => n.id));
      filteredEdges = filteredEdges.filter(e => keep.has(e.from) && keep.has(e.to));
    }}

    if (frequentLocs) {{
      filteredNodes = filteredNodes.filter(n => {{
        if (n.entity_type !== 'location') return true;
        return (n.incident_count || 0) > locationN;
      }});
      const keep = new Set(filteredNodes.map(n => n.id));
      filteredEdges = filteredEdges.filter(e => keep.has(e.from) && keep.has(e.to));
    }}

    filteredNodes.sort((a, b) => (b.degree_centrality || 0) - (a.degree_centrality || 0));
    filteredNodes = filteredNodes.slice(0, Math.max(10, nodeLimit));

    const keepFinal = new Set(filteredNodes.map(n => n.id));
    filteredEdges = filteredEdges.filter(e => keepFinal.has(e.from) && keepFinal.has(e.to));

    // Safeguard: if result is empty, show full graph
    if (filteredNodes.length === 0 || filteredEdges.length === 0) {{
      console.warn('Filter resulted in empty graph, showing full graph instead');
      applyFilteredGraph(allNodes_global, allEdges_global);
    }} else {{
      console.log('Filter applied:', filteredNodes.length, 'nodes,', filteredEdges.length, 'edges');
      applyFilteredGraph(filteredNodes, filteredEdges);
    }}
  }}

  function applyFilters_handler() {{
    console.log('Apply filters called');
    applyFilters();
  }}

  function resetFilters_handler() {{
    console.log('Reset filters called');
    resetFilters();
  }}

  (function() {{
    if (typeof network === 'undefined' || typeof nodes === 'undefined' || typeof edges === 'undefined') {{
      console.warn('PyVis network variables not found.');
      return;
    }}

    // Populate global state from PyVis - create deep copies to avoid reference issues
    const rawNodes = nodes.get();
    const rawEdges = edges.get();
    
    allNodes_global = rawNodes.map(n => ({{...n}})); // Deep copy each node
    allEdges_global = rawEdges.map(e => ({{...e}})); // Deep copy each edge
    
    FRAUD_CONNECTED_global = new Set({json.dumps(sorted(list(fraud_connected_nodes)))});
    SHARED_VEHICLES_global = new Set({json.dumps(sorted(list(shared_vehicle_nodes)))});

    // Wire up slider events
    const slider = document.getElementById('f_nodeLimit');
    if (slider) slider.addEventListener('input', updateSliderLabel);

    // Initialize slider label
    updateSliderLabel();
    
    console.log('Filter system initialized:', allNodes_global.length, 'nodes,', allEdges_global.length, 'edges');
  }})();
</script>
"""

        # Inject the structured investigator UI right after the <body> tag.
        body_index = html_content.find("<body")
        if body_index != -1:
            body_end = html_content.find(">", body_index) + 1
            html_content = (
                html_content[:body_end]
                + polyfill_html
                + css_html
                + title_html
                + panel_html
                + legend_html
                + js_html
                + html_content[body_end:]
            )

        # Save HTML file with required name
        output_file = "fraud_investigation_network.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Interactive investigation network saved as: {output_file}")
        print()

        # Auto-open in browser
        try:
            file_path = os.path.abspath(output_file)
            webbrowser.open(f'file://{file_path}')
            print("🌐 Investigation dashboard automatically opened in your browser")
        except Exception:
            print(f"⚠️  Could not auto-open browser. Please open {output_file} manually.")

        print()
        print("🎯 Visualization Features Enabled:")
        print("  • Physics-based layout")
        print("  • Hover tooltips")
        print("  • Node filtering menu")
        print("  • Zoom and pan")
        print()

        return output_file

    def _create_investigation_tooltip(self, node: str, node_attrs: Dict, subgraph: nx.Graph) -> str:
        """
        Create a detailed investigation tooltip for a node.

        Args:
            node (str): Node identifier
            node_attrs (Dict): Node attributes
            subgraph (nx.Graph): Investigation subgraph

        Returns:
            str: Formatted tooltip HTML
        """
        entity_type = node_attrs.get('entity_type', 'unknown')
        degree_centrality = self.metrics['degree'].get(node, 0)
        shared_score = node_attrs.get('shared_entity_score', 0)
        connections = subgraph.degree(node)

        # Determine risk level
        if degree_centrality > 0.002:
            risk_level = "HIGH"
            risk_emoji = "🚨"
        elif degree_centrality > 0.001:
            risk_level = "MEDIUM"
            risk_emoji = "⚠️"
        else:
            risk_level = "LOW"
            risk_emoji = "✅"

        # Entity-specific information
        if entity_type == 'policy_holder':
            policy_id = node_attrs.get('policy_id', 'Unknown')
            return f"""👤 POLICY HOLDER INVESTIGATION
═══════════════════════════════
Policy ID: {policy_id}
Connected Entities: {connections}
Network Influence: {degree_centrality:.4f}
Shared Entity Score: {shared_score}
Fraud Risk Level: {risk_level} {risk_emoji}

Investigation Notes:
• Check for unusual claim patterns
• Verify vehicle ownership legitimacy
• Cross-reference with incident locations
• High connectivity may indicate fraud ring involvement"""

        elif entity_type == 'vehicle':
            make = node_attrs.get('vehicle_make', 'Unknown')
            model = node_attrs.get('vehicle_model', 'Unknown')
            year = node_attrs.get('vehicle_year', 'Unknown')
            return f"""🚗 VEHICLE INVESTIGATION
═══════════════════════════════
Vehicle: {make} {model} ({year})
Number of Claims: {connections}
Shared Entity Score: {shared_score}
Fraud Risk Level: {risk_level} {risk_emoji}

Investigation Notes:
• Used by {connections} different policy holders
• High shared usage may indicate fraud ring
• Verify vehicle identification numbers
• Check for staged accidents"""

        elif entity_type == 'location':
            city = node_attrs.get('incident_city', 'Unknown')
            state = node_attrs.get('incident_state', 'Unknown')
            return f"""📍 LOCATION INVESTIGATION
═══════════════════════════════
Location: {city}, {state}
Number of Incidents: {connections}
Shared Entity Score: {shared_score}
Fraud Risk Level: {risk_level} {risk_emoji}

Investigation Notes:
• Multiple accidents at same location
• May indicate staged accident location
• Verify location legitimacy
• Check for repair shop connections"""

        elif entity_type == 'zip_code':
            zip_code = node_attrs.get('insured_zip', 'Unknown')
            return f"""📮 GEOGRAPHIC AREA INVESTIGATION
═══════════════════════════════
ZIP Code: {zip_code}
Policy Holders: {connections}
Shared Entity Score: {shared_score}
Fraud Risk Level: {risk_level} {risk_emoji}

Investigation Notes:
• High concentration of policy holders
• May indicate geographic fraud ring
• Cross-reference with incident locations
• Check for organized fraud patterns"""

        else:
            return f"Unknown Entity: {node}"

    def _generate_investigation_insights(self, subgraph: nx.Graph) -> Dict[str, int]:
        """
        Generate investigation insights for the dashboard.

        Args:
            subgraph (nx.Graph): Investigation subgraph

        Returns:
            Dict[str, int]: Dictionary of investigation metrics
        """
        suspicious_set = set(self.suspicious_nodes)

        total_policies = len([
            n for n in subgraph.nodes()
            if self.graph.nodes[n].get('entity_type') == 'policy_holder'
        ])

        suspicious_policies = len([
            n for n in subgraph.nodes()
            if n in suspicious_set and self.graph.nodes[n].get('entity_type') == 'policy_holder'
        ])

        repeated_vehicles = len([
            n for n in subgraph.nodes()
            if self.graph.nodes[n].get('entity_type') == 'vehicle' and subgraph.degree(n) > 2
        ])

        repeated_locations = len([
            n for n in subgraph.nodes()
            if self.graph.nodes[n].get('entity_type') == 'location' and subgraph.degree(n) > 2
        ])

        return {
            'total_policies': total_policies,
            'suspicious_policies': suspicious_policies,
            'repeated_vehicles': repeated_vehicles,
            'repeated_locations': repeated_locations
        }

    def _create_dashboard_html(self, insights: Dict[str, int]) -> str:
        """
        Create the investigation dashboard HTML.

        Args:
            insights (Dict[str, int]): Investigation insights

        Returns:
            str: Dashboard HTML content
        """
        return f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 10px; font-family: 'Segoe UI', Arial, sans-serif; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
    <h1 style="text-align: center; margin: 0 0 20px 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
        🔍 FRAUD INVESTIGATION DASHBOARD
    </h1>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
        <!-- Investigation Insights Panel -->
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);">
            <h3 style="color: #ffd700; margin-top: 0; font-size: 1.3em;">📊 INVESTIGATION INSIGHTS</h3>
            <div style="font-size: 0.95em; line-height: 1.6;">
                <div style="margin-bottom: 8px;"><strong>Total Policies Analyzed:</strong> {insights['total_policies']}</div>
                <div style="margin-bottom: 8px;"><strong>Suspicious Policies:</strong> <span style="color: #ff6b6b; font-weight: bold;">{insights['suspicious_policies']}</span></div>
                <div style="margin-bottom: 8px;"><strong>Repeated Vehicles Detected:</strong> <span style="color: #ffd93d; font-weight: bold;">{insights['repeated_vehicles']}</span></div>
                <div style="margin-bottom: 8px;"><strong>Repeated Locations Detected:</strong> <span style="color: #6bcf7f; font-weight: bold;">{insights['repeated_locations']}</span></div>
            </div>
        </div>

        <!-- Graph Legend Panel -->
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);">
            <h3 style="color: #ffd700; margin-top: 0; font-size: 1.3em;">📖 GRAPH LEGEND</h3>
            <div style="font-size: 0.9em; line-height: 1.5;">
                <div style="margin-bottom: 6px;"><span style="color: #3498db; font-weight: bold;">👤 Blue dots</span> → Policy Holders</div>
                <div style="margin-bottom: 6px;"><span style="color: #27ae60; font-weight: bold;">🚗 Green triangles</span> → Vehicles</div>
                <div style="margin-bottom: 6px;"><span style="color: #e74c3c; font-weight: bold;">📍 Red squares</span> → Accident Locations</div>
                <div style="margin-bottom: 6px;"><span style="color: #f39c12; font-weight: bold;">📮 Orange diamonds</span> → ZIP Codes</div>
                <div style="margin-bottom: 6px;"><span style="color: #e74c3c; font-weight: bold;">🔴 Red edges</span> → Fraud flagged claims</div>
                <div style="margin-bottom: 6px;"><span style="color: #95a5a6; font-weight: bold;">⚪ Gray edges</span> → Normal claims</div>
                <div style="margin-top: 8px; padding: 6px; background: rgba(255,255,255,0.2); border-radius: 5px;">
                    <strong>Large nodes</strong> = Frequently appearing<br>
                    <strong>Red borders</strong> = High fraud risk
                </div>
            </div>
        </div>
    </div>

    <!-- Investigation Explanation Panel -->
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);">
        <h3 style="color: #ffd700; margin-top: 0; font-size: 1.3em;">🧠 FRAUD PATTERN EXPLANATION</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 0.9em; line-height: 1.5;">
            <div>
                <h4 style="color: #6bcf7f; margin: 0 0 8px 0;">Understanding Fraud Patterns:</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Shared vehicle:</strong> Multiple policies connected to the same vehicle</li>
                    <li><strong>Repeated location:</strong> Many accidents at the same location</li>
                    <li><strong>High connectivity:</strong> Entities linked to many claims</li>
                    <li><strong>Fraud ring:</strong> Tightly connected cluster of claims</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #6bcf7f; margin: 0 0 8px 0;">Investigation Actions:</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>🖱️ <strong>Hover</strong> for detailed investigation info</li>
                    <li>🔍 <strong>Zoom</strong> to examine specific areas</li>
                    <li>✋ <strong>Drag</strong> nodes to rearrange layout</li>
                    <li>🎯 <strong>Click</strong> to highlight connections</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Investigation Controls Panel -->
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px); margin-top: 15px;">
        <h3 style="color: #ffd700; margin-top: 0; font-size: 1.3em;">🛠 INVESTIGATION CONTROLS</h3>
        <div style="text-align: center;">
            <button onclick="network.setOptions({{physics:false}})" style="padding:10px 20px; margin:5px; border:none; border-radius:6px; background:#ffcc00; font-weight:bold; cursor:pointer;">
                ❄️ Freeze Layout
            </button>
            <button onclick="network.setOptions({{physics:true}})" style="padding:10px 20px; margin:5px; border:none; border-radius:6px; background:#28a745; color:white; font-weight:bold; cursor:pointer;">
                ⚡ Enable Physics
            </button>
            <button onclick="alert('🎯 INVESTIGATION GUIDE:\\n\\n1. Look for red-bordered nodes (high risk)\\n2. Follow red edges (fraud claims)\\n3. Find shared vehicles (green triangles)\\n4. Check repeated locations (red squares)\\n5. Identify clusters of connected entities\\n\\n💡 Use zoom and drag to explore patterns!')" style="padding:10px 20px; margin:5px; border:none; border-radius:6px; background:#007bff; color:white; font-weight:bold; cursor:pointer;">
                📋 Start Guided Tour
            </button>
        </div>
    </div>
</div>
"""

    def print_analysis_summary(self) -> Dict[str, Any]:
        """
        Print a comprehensive summary of the fraud analysis results.

        Returns:
            Dict[str, Any]: Summary statistics
        """
        print("=" * 80)
        print("STEP 10: ANALYSIS SUMMARY")
        print("=" * 80)

        # Basic network statistics
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        # Count confirmed fraud policies (policy nodes colored red)
        fraud_policies = [
            n for n, attrs in self.graph.nodes(data=True)
            if attrs.get("entity_type") == "policy_holder" and attrs.get("fraud_policy", False)
        ]

        # Rank suspicious nodes by degree centrality and show the top 5
        degree = self.metrics.get("degree", {}) if self.metrics else {}
        suspicious_nodes = self.suspicious_nodes or []
        top_suspicious = sorted(
            suspicious_nodes,
            key=lambda n: degree.get(n, 0.0),
            reverse=True
        )[:5]

        print("📊 INVESTIGATION SUMMARY:")
        print("-" * 40)
        print(f"Number of nodes: {total_nodes}")
        print(f"Number of edges: {total_edges}")
        print(f"Number of fraud policies: {len(fraud_policies)}")
        print()

        print("🟡 Top 5 suspicious nodes by degree centrality (degree_centrality > 0.05):")
        print("-" * 40)
        if not top_suspicious:
            print("No nodes exceeded the suspicious centrality threshold.")
        else:
            for i, node in enumerate(top_suspicious, 1):
                attrs = self.graph.nodes[node]
                entity_type = attrs.get("entity_type", "unknown")
                label = attrs.get("label", str(node))
                if entity_type == "policy_holder":
                    entity_id = attrs.get("policy_id", node)
                elif entity_type == "vehicle":
                    entity_id = attrs.get("vehicle_id", node)
                elif entity_type == "location":
                    entity_id = attrs.get("location_id", node)
                elif entity_type == "zip_code":
                    entity_id = attrs.get("zip_id", node)
                else:
                    entity_id = node
                print(f"{i}. {entity_id} | type={entity_type} | degree_centrality={degree.get(node, 0.0):.4f} | label={label}")
        print()

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "fraud_policies": len(fraud_policies),
            "suspicious_nodes": len(suspicious_nodes),
            "top_suspicious_nodes": top_suspicious,
        }

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute the complete fraud investigation analysis pipeline.

        Returns:
            Dict[str, Any]: Complete analysis results
        """
        print("🚀 FRAUD INVESTIGATION SYSTEM - COMPLETE ANALYSIS")
        print("=" * 80)
        print()

        try:
            # Step 1: Load dataset
            dataset = self.load_dataset()

            # Step 2: Initialize graph
            graph = self.initialize_graph()

            # Step 3: Create nodes
            self.create_nodes(graph, dataset)

            # Step 4: Create edges
            self.create_edges(graph, dataset)

            # Step 5: Compute metrics
            metrics = self.compute_graph_metrics(graph)

            # Step 6: Detect suspicious entities
            suspicious_nodes = self.detect_suspicious_entities(graph, metrics)

            # Step 7: Create interactive visualization (full graph)
            network_file = self.create_interactive_investigation_graph(graph)

            # Step 8: Print investigation summary
            summary = self.print_analysis_summary()

            print("=" * 80)
            print("✅ FRAUD INVESTIGATION ANALYSIS COMPLETED")
            print("=" * 80)
            print()
            print("🎯 Key Findings:")
            print(f"  • Number of nodes: {summary['total_nodes']}")
            print(f"  • Number of edges: {summary['total_edges']}")
            print(f"  • Number of fraud policies: {summary['fraud_policies']}")
            print(f"  • Suspicious nodes (degree_centrality > 0.05): {summary['suspicious_nodes']}")
            print()
            print("📊 Interactive network saved as: fraud_investigation_network.html")
            print("🌐 Open this file in a web browser to explore the fraud investigation network")
            print()

            return {
                'summary': summary,
                'network_file': network_file,
                'suspicious_nodes': suspicious_nodes
            }

        except FileNotFoundError as e:
            print(f"❌ Error: Dataset file not found - {e}")
            print("Please ensure the graph dataset exists at:")
            print("data/processed/insurance_claims_graph_dataset.csv")
            raise
        except Exception as e:
            # Print a detailed error for easier debugging in different environments.
            print(f"❌ Error during analysis: {repr(e)}")
            print("Please check the data and try again.")
            raise


def main():
    """
    Main function to execute the fraud investigation system.
    """
    # Initialize the fraud investigation system
    investigation_system = FraudInvestigationSystem()

    # Run the complete analysis
    try:
        results = investigation_system.run_complete_analysis()
        print("\n🎉 Fraud investigation completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Fraud investigation failed: {e}")
        return None


if __name__ == "__main__":
    main()
