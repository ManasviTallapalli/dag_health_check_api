from typing import List, Optional
from pydantic import BaseModel
from collections import deque, defaultdict

from fastapi import FastAPI, HTTPException

import asyncio
import httpx

import time


# 1. Input: The API accepts json input of a dag

# data model for a single component in our system
class Component(BaseModel):
    id: str
    depends_on: List[str] = []  # each item is a list with strings inside. If missing then default to empty list
    health_url: Optional[str] = None # optional but if present then its a string. if missing default to None

class SystemRequest(BaseModel):
    components: List[Component] # overall structure of the request body: list of components

app = FastAPI()

# ---------------------------------

# 2. Build graph from json so we can understand relationships
# list[Component] in the argument is called type hint. we are giving a hint about the type
def build_children_and_parents(components: list[Component]):
    # first put all nodes in a set so there are no duplicates
    nodes = set()
    for c in components:
        nodes.add(c.id)
        
    # now we have all nodes in a set, next lets define the parents and children for each node

    # given a child node, tell me all its parents
    children_to_parents = defaultdict(list) # in a defaultdict if a key is missing it automatically creates a default value which is an empty list here
    parents_to_children = defaultdict(list) # given a parent, tell me all its children

    # check if the dependency exists in list of nodes. return error if not existing
    for c in components:
        for p in c.depends_on:
            if p not in nodes:
                raise HTTPException(status_code=400, detail = f"Unknown dependency: {p}")


    # populating children and parent dictionaries
    for c in components:
        for p in c.depends_on:   #c.depends_on as c is a list object not dict for it to be c["depends_on"]
            parents_to_children[p].append(c.id)
            children_to_parents[c.id].append(p)

    # ensure children and parents exist for all nodes, if not it might be a root/leaf node and we need to handle this
    for n in nodes:
        parents_to_children.setdefault(n,[]) # if n is not a node in children then create it with an empty list
        children_to_parents.setdefault(n,[])

    
    # print("Parents:",parents)
    # print("Children:",children)

    return nodes, children_to_parents, parents_to_children


def bfs_for_dag(nodes, children_to_parents, parents_to_children):

    # store count of how many parents each node has
    count_of_parents_each_node = {}
    for node in nodes:
        p_count = len(children_to_parents[node])
        count_of_parents_each_node[node] = p_count

    # store nodes with no parents in a queue
    queue = deque()
    for node in nodes:
        if count_of_parents_each_node[node] == 0:
            queue.append(node)

    order = []
    processed = 0 # counts how many nodes we processed

    # bfs loop
    while queue:
        level_size = len(queue) # no of nodes on this level

        for _ in range(level_size):
            current = queue.popleft()
            order.append(current)
            processed += 1

            for child in parents_to_children[current]:
                count_of_parents_each_node[child] -= 1
                if count_of_parents_each_node[child] == 0:
                    queue.append(child)

    # detect if there are any cycles
    if processed != len(nodes):
        raise HTTPException(status_code=400, detail="Cycle detected. Not a DAG." )
    
    return order


# ---------------------------------

# 3. Async health checks

async def check_one_component_health(client: httpx.AsyncClient, comp: Component):
    """
    Checks the component health_url:
    - if missing then status=unknown
    - if http 200 = healthy
    - else = unhealthy
    """

    # if health_url is missing (as components are coming from external) we dont want it to fail. so mark them unknown
    if not comp.health_url:
        return {"status":"unknown", "http_status": None, "latency_ms": None, "error": "health_url_missing"}
    
    # measure the start time
    start_time = time.perf_counter()

    # try except block because network calls are unreliable and can crash your program if not handled.
    try:
        response = await client.get(comp.health_url, timeout=2.0) # wait makes it pause until response comes. imp!
        latency_ms = int((time.perf_counter() - start_time) * 1000) # returns in seconds so multiply by 1000 for milliseconds

        if response.status_code == 200:
            return {"status": "healthy", "http_status": response.status_code, "latency_ms": latency_ms, "error": None}
        else:
            return {"status": "unhealthy", "http_status": response.status_code, "latency_ms": latency_ms, "error": "Not 200"}
    #exception handling for catching connection refused, server down, DNS failure, timeout
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000) # define here again bc if try block fails due to an exception then latency is never calculated
        return {"status": "unhealthy", "http_status": None, "latency_ms": latency_ms, "error": str(e)}
    

# -----------------------------------------

# 4. Optional - build graph

# Your API can return a “graph description” (DOT text). Later you can convert it into an image (PNG/SVG) using Graphviz. 
# This function generates that DOT text and marks failed nodes red.

def build_dot_graph(nodes, parents_to_children, health_failed_nodes):
    lines = [
        "digraph G {", # means “start a directed graph named G”
        "rankdir = LR;"  # means “draw the graph left to right”
    ]
    for n in sorted(nodes):
        if n in health_failed_nodes:
            lines.append(f'"{n}" [style=filled, fillcolor=red, fontcolor=white];')
        else:
            lines.append(f'"{n}" [style=filled, fillcolor=lightgray];')

    # created edges
    for p in sorted(nodes):
        for c in parents_to_children[p]:
            lines.append(f'"{p}" -> "{c}";')

    lines.append("}") # close the dot graph
    return "\n".join(lines) # returns ["a", "b", "c"] -> "a\nb\nc" as graphviz expects one big DOT string

# -----------------------------------------

# 5. API endpoints
@app.post("/system/health") # Creates an HTTP POST endpoint
# async as we will use await next
async def system_health(req: SystemRequest):
    # 1. build graph
    nodes, children_to_parents, parents_to_children = build_children_and_parents(req.components)

    # result = build_children_and_parents(req.components)
    # nodes = result[0]
    # children_to_parents = result[1]
    # parents_to_children = result[2]

    # 2. bfs like traversal using kahns algorithm
    # order = kahns_alg(nodes, children_to_parents, parents_to_children)
    order = bfs_for_dag(nodes, children_to_parents, parents_to_children)

    # 3. look up table for components. fast O(1) lookup of components by IDs. avoids repeated list scanning
    component_map = {}
    for c in req.components:
        component_map[c.id] = c
    # comp_map = {c.id: c for c in req.components}


    # 4. async health checks
    async with httpx.AsyncClient() as client:
        tasks = [check_one_component_health(client, component_map[node_id]) for node_id in order] # this line just creates coroutines
        #tasks = []
        #for node_id in order:
        #    task = check_one_health(client, comp_map[node_id])
        #    tasks.append(task)

        results_list = await asyncio.gather(*tasks) # running all tasks together and collects results. if tasks = [task1, task2, task3] then *tasks = task1, task2, task3. This (*) is called unpacking
    # final results dictionary. we convert list to dict for later use
    results = {}
    for node_id, res in zip(order, results_list):
        results[node_id] = res

    # {node_id: res for node_id, res in zip(order, results_list)}


    # 5. build table
    rows = [] # a list where each item is a row in our table
    failed = set() # set used to store failed components, no duplicates

    for node_id in order: # for each node from kahns order
        res = results[node_id] # get results for the node
        if res["status"] == "unhealthy":
            failed.add(node_id)

        rows.append({
            "id": node_id,
            "status": res["status"],
            "http_status": res["http_status"],
            "latency_ms": res["latency_ms"],
            "error": res["error"],
            "depends_on": children_to_parents[node_id],
            "children": parents_to_children[node_id]
        })

    overall_status = "healthy" if len(failed) == 0 else "unhealthy"

    dot_graph = build_dot_graph(nodes,parents_to_children, failed)

    return {
        "overall_status": overall_status,
        "traversal_order": order,
        "rows": rows,
        "dot_graph": dot_graph
    }


# -----------------------------------------------

# 6. Demo endpoints

@app.get('/demo/ok')
def demo_ok():
    return {"status": "ok"}

@app.get('/demo/fail')
def demo_fail():
    raise HTTPException(status_code=500, detail = "demo failure")






            




    











        


    

    





