import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from rasterio.transform import rowcol
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import remove_small_holes


def define_valley_extent(reach, graph, dem, cost_threshold, min_hole_to_keep_fraction):
    sources = reach_graph_node_ids(reach, dem)

    costs = dijkstra(
        graph,
        indices=sources,
        return_predecessors=False,
        min_only=True,
        limit=cost_threshold,
    )
    costs = costs.reshape(dem.shape)

    extent = costs < cost_threshold
    min_hole_area = np.sum(extent) * min_hole_to_keep_fraction
    extent = remove_small_holes(extent, area_threshold=min_hole_area)
    return extent


def reach_graph_node_ids(reach, dem):
    # first get rowcol for each spatial coordinate
    xs, ys = reach.xy
    rows, cols = rowcol(dem.rio.transform(), xs, ys)

    # then get the graph id for each rowcol
    id_array = np.arange(dem.size).reshape(dem.shape)
    node_ids = []
    for row, col in zip(rows, cols):
        node_ids.append(id_array[row, col])
    return node_ids


def dem_to_graph(dem_array, walls):
    data, ids = _create_graph_data_numba(dem_array, walls)
    graph = csr_matrix(data, shape=(ids.size, ids.size))
    return graph


@njit
def _create_graph_data_numba(dem, walls):
    nrows, ncols = dem.shape
    ids = np.arange(dem.size).reshape(dem.shape)
    row_inds = []
    col_inds = []
    data = []
    for row in range(nrows):
        for col in range(ncols):
            start = ids[row, col]

            if walls is not None:
                if walls[row, col]:
                    continue

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = row + dx
                    ny = col + dy
                    end = ids[nx, ny]

                    if walls is not None:
                        if walls[nx, ny]:
                            continue

                    if 0 <= nx < nrows and 0 <= ny < ncols:
                        elev_cost = np.abs(dem[nx, ny] - dem[row, col])
                        euclidean_cost = 1 if dx == 0 or dy == 0 else 1.41
                        euclidean_cost = euclidean_cost * 0.1
                        cost = elev_cost + euclidean_cost

                        data.append(cost)
                        row_inds.append(start)
                        col_inds.append(end)

    return (data, (row_inds, col_inds)), ids
