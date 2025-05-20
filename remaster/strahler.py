import networkx as nx


def label_streams(flowlines):
    flowlines = split_flowlines(flowlines)
    flowlines = flowlines.groupby("network_id").apply(
        compute_stream_order, include_groups=False
    )
    flowlines = flowlines.reset_index(level=0)

    # label mainstem and tributaries
    flowlines = assign_labels(flowlines)
    flowlines = flowlines.reset_index(drop=True)
    return flowlines


def lines_to_network(lines):
    """lines is goepandas dataframe with linestrings"""
    G = nx.DiGraph()
    for _, line in lines.iterrows():
        start = line.geometry.coords[0]
        end = line.geometry.coords[-1]
        G.add_edge(start, end, streamID=line["streamID"])
    return G


def assign_labels(flowlines):
    """
    Recursively label stream networks following mainstems and tributaries.

    Parameters:
    flowlines: GeoDataFrame with strahler order, mainstem, and network_id attributes

    Returns:
    GeoDataFrame with additional 'label' attribute
    """
    # Initialize stream labels column
    flowlines["label"] = None

    # Process each network separately
    for network_id, network_group in flowlines.groupby("network_id"):
        network_df = network_group.copy()
        G = lines_to_network(network_group)

        # Find outlet node (node with no outgoing edges)
        outlet_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
        if len(outlet_nodes) != 1:
            raise ValueError(f"Network {network_id} should have exactly one outlet")
        outlet_node = outlet_nodes[0]

        # Use a dictionary to track counters for each label prefix
        counter_dict = {}

        # Start recursive labeling from the outlet
        network_df = label_river_recursive(
            G, network_df, outlet_node, f"{network_id}", counter_dict
        )

        # Update the main flowlines dataframe
        flowlines.loc[network_df.index, "label"] = network_df["label"]

    return flowlines


def label_river_recursive(G, flowlines, current_node, current_label, counter_dict):
    """
    Recursively label streams, following mainstem first, then tributaries.

    Parameters:
    G: NetworkX DiGraph representation of the stream network
    flowlines: GeoDataFrame containing the stream segments
    current_node: Node to start labeling from (typically an outlet or junction)
    current_label: Label to assign to the mainstem from this node
    counter_dict: Dictionary to track tributary counters for each label prefix

    Returns:
    Updated flowlines GeoDataFrame with label column populated
    """
    # Get all incoming edges to current node
    upstream_edges = list(G.in_edges(current_node, data=True))

    if not upstream_edges:
        return flowlines  # No more upstream segments

    # Prepare data for sorting edges by priority
    edge_data = []
    for u, v, data in upstream_edges:
        stream_id = data["streamID"]
        strahler = flowlines.loc[stream_id, "strahler"]
        is_mainstem = flowlines.loc[stream_id, "mainstem"]
        length = flowlines.loc[stream_id, "geometry"].length
        edge_data.append((u, v, stream_id, strahler, is_mainstem, length))

    # Sort by mainstem flag first, then by Strahler order, then by length
    edge_data.sort(key=lambda x: (-int(x[4]), -x[3], -x[5]))

    # Process mainstem first, then tributaries
    if edge_data:
        # First edge after sorting is mainstem (highest priority)
        mainstem_u, mainstem_v, mainstem_stream_id = edge_data[0][:3]

        # Label the mainstem with current label
        flowlines.loc[mainstem_stream_id, "label"] = current_label

        # Recursively process mainstem first
        flowlines = label_river_recursive(
            G, flowlines, mainstem_u, current_label, counter_dict
        )

        # Then process each tributary with a new label
        for i in range(1, len(edge_data)):
            trib_u, trib_v, trib_stream_id = edge_data[i][:3]

            # Initialize counter for this prefix if not exists
            if current_label not in counter_dict:
                counter_dict[current_label] = 1

            # Create tributary label and increment counter
            trib_label = f"{current_label}.{counter_dict[current_label]}"
            counter_dict[current_label] += 1

            # Label this tributary edge
            flowlines.loc[trib_stream_id, "label"] = trib_label

            # Recursively label upstream of this tributary
            flowlines = label_river_recursive(
                G, flowlines, trib_u, trib_label, counter_dict
            )

    return flowlines


def split_flowlines(flowlines):
    graph = lines_to_network(flowlines)
    flowlines.index = flowlines["streamID"]

    outlets = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    if len(outlets) == 1:
        flowlines["network_id"] = 1
        return flowlines

    flowlines["network_id"] = None
    for i, outlet in enumerate(outlets, 1):
        upstream = nx.ancestors(graph, outlet)
        upstream.add(outlet)
        subgraph = graph.subgraph(upstream)
        streams = list(
            set(data["streamID"] for u, v, data in subgraph.edges(data=True))
        )
        flowlines.loc[streams, "network_id"] = i
    return flowlines


def compute_stream_order(lines):
    graph = lines_to_network(lines)
    # get root (no outgoing edges)
    root = [node for node in graph.nodes if graph.out_degree(node) == 0]
    if len(root) != 1:
        raise ValueError("There should be exactly one outlet node")
    strahler = calculate_strahler(graph, root[0])

    lines["strahler"] = None
    lines["strahler"] = lines["streamID"].apply(lambda x: lookup_strahler(graph, x))

    lines["mainstem"] = None
    mainstem_edges = find_mainstem(graph, root[0])
    lines["mainstem"] = lines["streamID"].apply(lambda x: x in mainstem_edges)

    lines["outlet"] = None
    lines["outlet"] = lines["geometry"].apply(lambda x: x.coords[-1] == root[0])
    return lines


def find_mainstem(G, outlet_node):
    """
    Find mainstem by following highest order stream upstream from outlet

    Parameters:
    G: NetworkX DiGraph with 'strahler' edge attribute
    outlet_node: starting node

    Returns:
    list of edge streamIDs representing mainstem path from headwater to outlet
    """
    mainstem_edges = []
    current_node = outlet_node

    while True:
        # Get all upstream edges
        upstream_edges = list(G.in_edges(current_node))

        # If no more upstream edges, we've reached a headwater
        if not upstream_edges:
            break

        # Find the edge with highest Strahler order
        max_order = -1
        next_node = None
        selected_edge = None

        for u, v in upstream_edges:
            order = G.edges[u, v]["strahler"]
            if order > max_order:
                max_order = order
                next_node = u
                selected_edge = (u, v)

        # Add edge to mainstem and continue upstream
        if selected_edge:
            # Store just the streamID of the edge
            stream_id = G.edges[selected_edge]["streamID"]
            mainstem_edges.append(stream_id)

        # Move to next node upstream
        current_node = next_node

    # Return in downstream order (from headwater to outlet)
    return mainstem_edges[::-1]


def lookup_strahler(graph, stream_id):
    """Get the Strahler number for a given stream ID"""
    edges = [
        (u, v) for u, v in graph.edges if graph.edges[u, v]["streamID"] == stream_id
    ]
    if len(edges) == 0:
        raise ValueError(f"Stream ID {stream_id} not found in graph edges")
    if len(edges) > 1:
        raise ValueError(f"Multiple edges found for Stream ID {stream_id}")
    edge = edges[0]
    return graph.edges[edge]["strahler"]


def calculate_strahler(G, root_node):
    # Assumes G is a directed graph (DiGraph) with flow direction

    def _strahler_recursive(node):
        # Get upstream edges
        in_edges = list(G.in_edges(node))

        if not in_edges:  # Leaf node/headwater
            return 1

        # Get Strahler numbers of upstream edges
        upstream_orders = []
        for u, v in in_edges:
            if "strahler" not in G.edges[u, v]:
                upstream_order = _strahler_recursive(u)
                G.edges[u, v]["strahler"] = upstream_order
            upstream_orders.append(G.edges[u, v]["strahler"])

        # Calculate Strahler number for current segment
        max_order = max(upstream_orders)
        if upstream_orders.count(max_order) > 1:
            strahler = max_order + 1
        else:
            strahler = max_order

        # Set order for edge going downstream from current node
        out_edges = list(G.out_edges(node))
        if out_edges:  # if not outlet
            u, v = out_edges[0]  # should only be one downstream edge
            G.edges[u, v]["strahler"] = strahler

        return strahler

    return _strahler_recursive(root_node)
