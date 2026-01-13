This project implements a Python web API that checks the health of a system composed of multiple interdependent components modeled as a Directed Acyclic Graph (DAG)


Each component can depend on other components, and the system health is evaluated by:
- Traversing the DAG in a breadth-first (topological) order
- Performing asynchronous health checks for each component
- Aggregating results into a structured, tabular response
- Optionally generating a visual DAG representation highlighting failures


FEATURES:
- Accepts system relationships via JSON
- Validates DAG structure and detects cycles
- Traverses the graph using BFS (Kahn’s Algorithm)
- Performs asynchronous health checks using `asyncio` and `httpx`
- Returns component health in tabular JSON format
- Computes overall system health
- Outputs Graphviz DOT representation with failed nodes highlighted


ARCHITECTURE OVERVIEW:
- **FastAPI** for API framework
- **Pydantic** for request validation
- **Kahn’s Algorithm** for BFS/topological traversal
- **Async HTTP calls** for parallel health checks
- **Graphviz DOT** output for optional visualization


PROJECT STRUCTURE:
dag_health_api/
- main.py # Main FastAPI application
- requirements.txt # Python dependencies
- sample_request.json # Example input JSON
- README.md # Project documentation
- graph.dot # (Optional) Graphviz DOT output
- graph.png # png pic


INPUT FORMAT:
The API accepts a JSON payload describing system components and their dependencies.

Example (`sample_request.json`):

```json
{
  "components": [
    {
      "id": "Step 1",
      "depends_on": [],
      "health_url": "http://localhost:8000/demo/ok"
    },
    {
      "id": "Step 2",
      "depends_on": ["Step 1"],
      "health_url": "http://localhost:8000/demo/ok"
    }
  ]
}