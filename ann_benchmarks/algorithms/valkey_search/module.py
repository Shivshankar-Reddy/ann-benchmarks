import subprocess
import sys
import time
import os

from redis import Redis

from ..base.module import BaseANN


class ValkeySearchNGT(BaseANN):
    def __init__(self, metric, param):
        self.metric = metric
        
        # Extract parameters from the param dict (like NGT algorithms)
        self.edge_size_for_creation = int(param.get("edge_size_for_creation", 8))
        self.edge_size_for_search = int(param.get("edge_size_for_search", 12))
        self.epsilon_for_creation = float(param.get("epsilon_for_creation", 0.02))
        self.epsilon_for_search = float(param.get("epsilon_for_search", -0.3))
        
        self.index_name = "valkey_ngt_index"
        self.field_name = "v"
        
        # EF runtime (will be set in set_query_arguments) - not used in search but kept for compatibility
        self.ef_runtime = 50
        
        print(f"ValkeySearchNGT: edge_size_for_creation={self.edge_size_for_creation}")
        print(f"ValkeySearchNGT: edge_size_for_search={self.edge_size_for_search}")
        print(f"ValkeySearchNGT: epsilon_for_creation={self.epsilon_for_creation}")
        print(f"ValkeySearchNGT: epsilon_for_search={self.epsilon_for_search}")
        print(f"ValkeySearchNGT: metric={metric}")

    def fit(self, X):
        # Start Valkey-Search server in the background
        # Assuming valkey-server is in the PATH or we need to specify full path
        valkey_path = os.environ.get('VALKEY_PATH', 'valkey-server')
#        cmd = f"{valkey_path} --daemonize yes"
 #       print("Starting Valkey-Search:", cmd)
  #      subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

        # Sleep a bit to make sure the server is running
        print("Sleeping 3s to wait for Valkey-Search server to start up...")
        time.sleep(3)

        # Connect to Valkey-Search
        print("Connecting to Valkey-Search...")
        self.redis = Redis(host="localhost", port=6379, decode_responses=False)

        # Create index with NGT algorithm
        distance_metric = {"angular": "COSINE", "euclidean": "L2"}[self.metric]
        
        args = [
            "FT.CREATE",
            self.index_name,
            "ON", "HASH",
            "PREFIX", "1", "doc:",
            "SCHEMA",
            self.field_name,
            "VECTOR",
            "NGT",
            "14",  # number of remaining arguments
            "TYPE",
            "FLOAT32",
            "DIM",
            X.shape[1],
            "DISTANCE_METRIC",
            distance_metric,
            "EDGE_SIZE_FOR_CREATION",
            self.edge_size_for_creation,
            "EDGE_SIZE_FOR_SEARCH",
            self.edge_size_for_search,
            "EPSILON_FOR_CREATION",
            self.epsilon_for_creation,
            "EPSILON_FOR_SEARCH",
            self.epsilon_for_search,
        ]
        print("Running Valkey-Search command:", args)
        self.redis.execute_command(*args)

        # Insert vectors
        p = self.redis.pipeline(transaction=False)
        for i, v in enumerate(X):
            # Convert numpy array to bytes for storage
            vector_bytes = v.tobytes()
            p.execute_command("HSET", f"doc:{i}", self.field_name, vector_bytes)
            if i % 1000 == 999:
                p.execute()
                p.reset()
        p.execute()

    def set_query_arguments(self, epsilon):
        # Handle single epsilon value like PANNG algorithm
        print(f"ValkeySearchNGT: epsilon={epsilon}")
        # Convert epsilon like NGT algorithms: epsilon = epsilon - 1.0
        self.epsilon = epsilon - 1.0
        # Update the epsilon_for_search for the next index creation
        self.epsilon_for_search = self.epsilon
        self.name = f"ValkeySearchNGT(edge_creation={self.edge_size_for_creation}, edge_search={self.edge_size_for_search}, epsilon={epsilon})"

    def query(self, v, n):
        # Use runtime epsilon if available, otherwise use default
        epsilon_param = f" EPSILON {self.epsilon}" if hasattr(self, 'epsilon') else ""
        q = [
            "FT.SEARCH",
            self.index_name,
            f"*=>[KNN {n} @{self.field_name} $BLOB{epsilon_param}]",
            "NOCONTENT",
            "LIMIT",
            "0",
            str(n),
            "PARAMS",
            "2",
            "BLOB",
            v.tobytes(),
            "DIALECT",
            "2",
        ]
        result = self.redis.execute_command(*q)
        
        # Extract document IDs from the result
        # Result format: [total_count, doc_id1, doc_id2, ...]
        if len(result) > 1:
            # Extract document IDs (skip the first element which is the total count)
            doc_ids = []
            for i in range(1, len(result), 2):  # Skip every other element (values)
                if i < len(result):
                    doc_id = result[i].decode('utf-8')
                    # Extract the numeric part from "doc:123"
                    try:
                        numeric_id = int(doc_id.split(':')[1])
                        doc_ids.append(numeric_id)
                    except (IndexError, ValueError):
                        # If we can't parse the ID, skip it
                        continue
            return doc_ids
        return []

    def __str__(self):
        return f"ValkeySearchNGT(edge_creation={self.edge_size_for_creation}, edge_search={self.edge_size_for_search}, epsilon={self.epsilon + 1.0})" 
