import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from scipy.spatial import Delaunay
import networkx as nx
import copy
from collections import defaultdict, Counter
import warnings
warnings.simplefilter("error")
import sys

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def percolationSample(numBeads, numProbes, box_size, seedNumber, filepath):
    np.random.seed(seedNumber)
    filepath += str(numBeads) + '_' + str(seedNumber)
    print(filepath + '_tetra.txt')

    def replicate_points(points, box_size):
        """Replicate points across periodic boundaries."""
        replicas = []
        for dx in [-box_size, 0, box_size]:
            for dy in [-box_size, 0, box_size]:
                for dz in [-box_size, 0, box_size]:
                    if dx == dy == dz == 0:
                        continue
                    shift = np.array([dx, dy, dz])
                    replicas.append(points + shift)
        return np.vstack((points, *replicas))

    # generate random points in the box
    np.random.seed(seedNumber) # fix random seeds
    points = np.random.rand(numBeads, 3) * box_size  # Generate some random points

    # Replicate points to account for PBC
    points_PBC = replicate_points(points, box_size)

    # Compute the Voronoi tessellation and Delaunay tessellation
    tetra = Delaunay(points_PBC)
    vor = Voronoi(points_PBC)

    vertexSimplex = defaultdict(set)
    for tid, simplex in enumerate(tetra.simplices):
        for vid in simplex:
            vertexSimplex[vid].add(tid)

    def triangle_area(p, q, r):
        return 0.5 * np.linalg.norm(np.cross(q - p, r - p))

    def tetrahedron_incenter(A, B, C, D):
        # Areas of faces opposite each vertex
        alpha_A = triangle_area(B, C, D)  # face BCD
        alpha_B = triangle_area(A, C, D)  # face ACD
        alpha_C = triangle_area(A, B, D)  # face ABD
        alpha_D = triangle_area(A, B, C)  # face ABC

        # Compute weighted sum of vertices
        numerator = (alpha_A * A
                    + alpha_B * B
                    + alpha_C * C
                    + alpha_D * D)
        denominator = (alpha_A + alpha_B + alpha_C + alpha_D)

        # Return the incenter
        return numerator / denominator

    comAll = [] # array of COM
    for i in range(len(tetra.simplices)):
        point = points_PBC[tetra.simplices[i]]
        comAll.append(tetrahedron_incenter(point[0], point[1], point[2], point[3]))
    comAll = np.array(comAll)


    primaryStates = set()
    # if the COM of the tetra is inside the main box
    for i, simplex in enumerate(tetra.simplices):
        com = comAll[i]
        if np.max(com) < box_size and np.min(com) > 0:
            primaryStates.add(i)

    # tetra index in the primary cell
    def findPrimaryTetra(i):
        # if it is already inside
        if i in primaryStates:
            return i
        
        vertices = copy.deepcopy(tetra.simplices[i])
        for i in range(4):
            while vertices[i] >= numBeads:
                vertices[i] -= numBeads
        if len(set(vertices)) < 4:
            return -1
        
        for a in range(27):
            set1 = vertexSimplex[vertices[0]]
            for b in range(27):
                set2 = set1 & vertexSimplex[vertices[1]]
                if len(set2) > 0:
                    for c in range(27):
                        set3 = set2 & vertexSimplex[vertices[2]]
                        if len(set3) > 0:
                            for d in range(27):
                                setAll = set3 & vertexSimplex[vertices[3]]
                                if len(setAll) > 0:
                                    candidate = setAll.pop()
                                    if candidate in indexPrimary:
                                        return indexPrimary[candidate]
                                    if candidate in primaryStates:
                                        return candidate

                                vertices[3] += numBeads
                        vertices[2] += numBeads
                        vertices[3] %= numBeads
                vertices[1] += numBeads
                vertices[2] %= numBeads
                vertices[3] %= numBeads
            vertices[0] += numBeads
            vertices[1] %= numBeads
            vertices[2] %= numBeads
            vertices[3] %= numBeads

        return -1

    indexPrimary = dict()
    for i in range(len(tetra.simplices)):
        indexPrimary[i] = findPrimaryTetra(i)

    Counter(Counter(indexPrimary.values()).values())

    # construct graph and edge list
    def point_to_segment_distance(P, A, B):
        AB = B - A
        normAB2 = np.dot(AB, AB)
        if np.allclose(AB, 0):
            distances = np.linalg.norm(P - A, axis=1)
            return distances
        
        AP = P - A  # shape: (N, d)
        t = np.dot(AP, AB) / normAB2  # shape: (N,)
        t_clamped = np.clip(t, 0, 1)
        closest_points = A + np.outer(t_clamped, AB)
        return np.linalg.norm(P - closest_points, axis=1)

    def constructGraph(vor):
        """Filter edges based on distance to the nearest input point."""
        N = len(vor.vertices)
        G = nx.Graph()
        edgeList = []
        visited = set()
        for i, ridge in enumerate(vor.ridge_vertices): # loop over faces
            for vertex_idx in range(len(ridge)-1):     # loop over vertices on face
                v1, v2 = ridge[vertex_idx], ridge[vertex_idx + 1]
                if v1 < 0 or v2 < 0 or (v1, v2) in visited or (v2, v1) in visited:
                    continue  # Skip if edge goes to infinity
                visited.add((v1, v2))
                point1 = vor.vertices[v1]
                point2 = vor.vertices[v2]
                if point1[0] < -box_size/2 or point1[1] < -box_size/2 or point1[2] < -box_size/2 or point1[0] > 1.5*box_size or point1[1] > 1.5*box_size or point1[2] > 1.5*box_size:
                    continue
                if point2[0] < -box_size/2 or point2[1] < -box_size/2 or point2[2] < -box_size/2 or point2[0] > 1.5*box_size or point2[1] > 1.5*box_size or point2[2] > 1.5*box_size:
                    continue

                dist = min(point_to_segment_distance(vor.points, point1, point2))
                G.add_edge(v1, v2)
                edgeList.append([v1, v2, dist])

        print("Number of all nodes: ", N)
        print("Number of nodes: ", len(G.nodes()))
        print("Number of edges: ", len(edgeList))
        return G, edgeList

    fullGraph, edgeList = constructGraph(vor)
    edgeList.sort(key = lambda x: x[2])

    def ifPercolate(component, box_size, vor):
        largestVoidVertices = vor.vertices[np.array(list(component[0]))]
        visited = set()
        for i, vi in enumerate(largestVoidVertices):
            dists = np.linalg.norm(vor.points - vi, axis=1)
            closest_idx = np.argpartition(dists, 4)[:4]
            closest_idx = closest_idx[np.argsort(dists[closest_idx])] % numBeads
            closePoints = tuple(closest_idx)
            if closePoints in visited:
                return True
            else:
                visited.add(closePoints)
        return False

    l = len(edgeList)//2
    r = len(edgeList)
    m = 0
    while r-l>1:
        G = copy.deepcopy(fullGraph)
        m = (l+r) // 2
        for i in range(m):
            G.remove_edge(edgeList[i][0], edgeList[i][1])
        connected_components = list(nx.connected_components(G))
        connected_components = [[component,len(component)] for component in connected_components]
        connected_components.sort(key = lambda x:x[1], reverse=True)
        ifPerco = False
        for component in connected_components[:5]:
            if ifPercolate(component,box_size, vor):
                ifPerco = True
                break
        if len(connected_components) != 0 and ifPerco:
            l = m
        else:
            r = m
    mPerco = m

    rPerco = edgeList[m][2]
    r_cut = rPerco
    G = copy.deepcopy(fullGraph)
    for i in range(len(edgeList)):
        if edgeList[i][2] > r_cut:
            break
        G.remove_edge(edgeList[i][0], edgeList[i][1])

    connected_components = list(nx.connected_components(G))
    connected_components = [[component,len(component)] for component in connected_components]
    connected_components.sort(key = lambda x:x[1], reverse=True)
    ifPerco = False
    for component in connected_components[:5]:
        if ifPercolate(component, box_size, vor):
            ifPerco = True
            break
    if not ifPerco:
        component = connected_components[0]

    print(r_cut, ifPercolate(component, box_size, vor), len(list(component[0])))

    subG = G.subgraph(component[0])
    vertices = vor.vertices[list(component[0])]
    states = set()

    def segment_intersects_triangle(ABC, P, Q): # if a line segment pass through a triangle
        A, B, C = ABC[0], ABC[1], ABC[2]
        N = np.cross(B - A, C - A)
        norm_N = np.linalg.norm(N)
        d = Q - P
        denom = np.dot(d, N)
        t = -np.dot(P - A, N) / denom
        if t < 0 or t > 1:
            # Intersection point is not on the line segment.
            return False
        
        # barycentric coordinates: check if Intersection point is inside triangle
        X = P + t * d
        v0 = C - A
        v1 = B - A
        v2 = X - A
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        if u >= 0 and v >= 0 and (u + v) <= 1.0:
            return True
        else:
            return False
        
    def bisectionSearch(p1, p2):
        tetra1 = int(tetra.find_simplex(p1))
        tetra2 = int(tetra.find_simplex(p2))
        states.add(tetra1)
        states.add(tetra2)

        if tetra1 == tetra2:
            return
        
        # deal with werid case
        # 1. edges from VT may pass through some tetrahedron. So the two ends of the edge are not in neighboring tetrahedrons.
        # 2. edges from VT may pass through other tetrahedron and then end up in the neighboring tetrahedron.
        commonVertices = set(tetra.simplices[tetra1]) & set(tetra.simplices[tetra2])
        if len(commonVertices) < 3 or not segment_intersects_triangle(points_PBC[list(commonVertices)], p1, p2):
            midPoint = (p1 + p2) / 2
            bisectionSearch(p1, midPoint)
            bisectionSearch(midPoint, p2)

    for edge in list(subG.edges):
        bisectionSearch(vor.vertices[edge[0]], vor.vertices[edge[1]])
        t1 = int(tetra.find_simplex(vor.vertices[edge[0]]))
        t2 = int(tetra.find_simplex(vor.vertices[edge[1]]))
            
    states = list(states)
    states.sort()

    statePrimary = set() # id of the primary states
    for i in states:
        statePrimary.add(indexPrimary[i])
    if -1 in statePrimary:
        statePrimary.remove(-1)
        
    statePrimary = list(statePrimary) # statePrimary[new id] = original id
    statePrimary.sort()
    nStates = len(statePrimary)
    print("Number of primary states = ",nStates)

    def distance_to_bisector_intersection(A, B, C):
        """
        AB cross AC bisector at X, make sure AB is the longest side
        return X and AX distance
        """
        AB = B - A
        AC = C - A
        
        # Parameter t along AB for the intersection:
        t = np.linalg.norm(AC)**2 / (2 * np.dot(AC, AB))
        
        # Intersection point:
        X = A + t * AB
        # Distance from A to X:
        d = np.linalg.norm(X - A)
        return d, X

    def sphere_line_intersection_closer_to_B(A, B, C, r):
        d = B - A
        V = A - C
        d_norm_sq = np.dot(d, d)
        
        discriminant = (np.dot(V, d))**2 - d_norm_sq*(np.dot(V, V) - r**2)
        sqrt_disc = np.sqrt(discriminant)
        t = (-np.dot(V, d) + sqrt_disc) / d_norm_sq
        
        X = A + t * d
        distance = np.linalg.norm(X - A)
        
        return X, distance

    def computeExitArea(p1, p2, p3, r):
        c = np.linalg.norm(p1-p2)
        b = np.linalg.norm(p1-p3)
        a = np.linalg.norm(p3-p2)

        # make sure c is the longest
        if a > b and a > c:
            a, c = copy.deepcopy(c), copy.deepcopy(a)
            p1, p3 = copy.deepcopy(p3), copy.deepcopy(p1)
        if b > a and b > c:
            b, c = copy.deepcopy(c), copy.deepcopy(b)
            p2, p3 = copy.deepcopy(p3), copy.deepcopy(p2)
        s = (a + b + c) / 2
        areaTriangle = np.sqrt(s * (s - a) * (s - b) * (s - c))

        R = a*b*c/4/areaTriangle
        if R <= r:
            return 0
        
        # if obtuse triangle and two circles intersect outside the triangle
        d_AC, X_AC = distance_to_bisector_intersection(p1, p2, p3)
        d_BC, X_BC = distance_to_bisector_intersection(p2, p1, p3)
        #print(d_AC, d_BC, r)
        if r > d_AC and r > d_BC:
            return 0.0
        if r > d_AC or r > d_BC:
            if d_BC < d_AC: # make sure the side close to A is covered instead of B.
                a, b = b, a
                p1, p2 = p2, p1
                d_AC, d_BC = d_BC, d_AC
                X_AC, X_BC = X_BC, X_AC
                X, XA = sphere_line_intersection_closer_to_B(p1, p2, p3, r)
                b = r
                c = c - XA
                s = (a + b + c) / 2
                areaTriangle = np.sqrt(s * (s - a) * (s - b) * (s - c))
                angleX = np.arccos(np.clip(np.dot(p3-X, p2-X) / b / c, -1, 1))
                areaTriangle -= r*r/2 * (np.pi - angleX)
                if a < 2*r:
                    areaTriangle += r*r*np.arccos(a / (2 * r)) - (a / 4) * np.sqrt(4 * r**2 - a**2)
                return areaTriangle

        dA = areaTriangle * 2.0 / a
        dB = areaTriangle * 2.0 / b
        dC = areaTriangle * 2.0 / c
        areaTriangle -= np.pi * r * r / 2
        # deal with overlaps
        if a < 2*r:
            areaTriangle += r*r*np.arccos(a / (2 * r)) - (a / 4) * np.sqrt(4 * r**2 - a**2)
        if b < 2*r:
            areaTriangle += r*r*np.arccos(b / (2 * r)) - (b / 4) * np.sqrt(4 * r**2 - b**2)
        if c < 2*r:
            areaTriangle += r*r*np.arccos(c / (2 * r)) - (c / 4) * np.sqrt(4 * r**2 - c**2)
        # deal with truncation from the opsite side
        if dA < r:
            areaTriangle += r**2 * np.arccos(dA/r) - dA * np.sqrt(r**2 - dA**2)
        if dB < r:
            areaTriangle += r**2 * np.arccos(dB/r) - dB * np.sqrt(r**2 - dB**2)
        if dC < r:
            areaTriangle += r**2 * np.arccos(dC/r) - dC * np.sqrt(r**2 - dC**2)

        return areaTriangle
    
    # compute volume of tetra
    def tetra_volume(vertices):
        """Exact tetra volume from 4x3 array."""
        v = np.asarray(vertices, float).reshape(4,3)
        return abs(np.dot(v[1]-v[0], np.cross(v[2]-v[0], v[3]-v[0]))) / 6.0


    def sample_points_in_tetrahedron(vertices, n):
        rng = np.random.default_rng()
        w = rng.dirichlet(np.ones(4), size=n)        # (n,4)
        v = np.asarray(vertices, float).reshape(4,3) # (4,3)
        return w @ v

    def cavityVolume(vertices, r, nTrails=int(1e9)):
        v = np.asarray(vertices, float).reshape(4,3)
        V0 = tetra_volume(v)

        pts = sample_points_in_tetrahedron(v, nTrails)             # (n,3)
        d2 = ((pts[:,None,:] - v[None,:,:])**2).sum(axis=2)         # (n,4)
        covered = (d2 <= (r*r)).any(axis=1)                         # (n,)
        p_uncovered = 1.0 - covered.mean()

        est = V0 * p_uncovered
        return est

    transitionPrimary = set(statePrimary)
    newStates = list(transitionPrimary)[:]
    G = nx.Graph()
    while len(newStates) > 0:
        newState = newStates.pop()
        for neighbor in tetra.neighbors[newState]:
            vCommon = list(set(tetra.simplices[newState]) & set(tetra.simplices[neighbor]))
            exitArea = computeExitArea(points_PBC[vCommon[0]], points_PBC[vCommon[1]], points_PBC[vCommon[2]], r_cut)
            if exitArea > 0.0 and cavityVolume(points_PBC[tetra.simplices[neighbor]], r_cut, int(1e6)) > 0.0:
                if newState > indexPrimary[neighbor]:
                    G.add_edge(indexPrimary[neighbor], newState)
                else:
                    G.add_edge(newState, indexPrimary[neighbor])
                if indexPrimary[neighbor] not in transitionPrimary:
                    newStates.append(indexPrimary[neighbor])
                transitionPrimary.add(newState)
                transitionPrimary.add(indexPrimary[neighbor])
                
    transitionPrimary = list(transitionPrimary)
    transitionPrimary.sort()

    sccs = list(nx.connected_components(G))
    sccs = sorted(sccs, key=len, reverse = True)
    sccs = [list(comp) for comp in sccs]
    print("Number of clusters: ", len(sccs))

    idVol = dict()
    for cnt, i in enumerate(transitionPrimary):
        idVol[i] = cavityVolume(points_PBC[tetra.simplices[i]], r_cut, int(1e6))

     # compute channel area between neighboring tetrahedra
    channelArea = dict()
    for _, i in enumerate(transitionPrimary):
        for j in tetra.neighbors[i]:
            if i<indexPrimary[j] and indexPrimary[j] in transitionPrimary:
                vCommon = list(set(tetra.simplices[i]) & set(tetra.simplices[j]))
                exitArea = computeExitArea(points_PBC[vCommon[0]], points_PBC[vCommon[1]], points_PBC[vCommon[2]], r_cut)
                if exitArea > 0.0 and idVol[indexPrimary[j]] > 0.0:
                    channelArea[(i, indexPrimary[j])] = exitArea

    channelArea_df = pd.DataFrame(list(channelArea.keys()), columns=['i', 'j'])
    channelArea_df['area'] = channelArea.values()

    # Improved search: find all pairs (i, j) in channelArea_df that are not in G.edges()
    missing_edges = []
    edges_set = set(G.edges())
    for i, row in channelArea_df.iterrows():
        e1, e2 = row['i'], row['j']
        if (e1, e2) not in edges_set and (e2, e1) not in edges_set:
            missing_edges.append((e1, e2))
    print("missing edges: ", missing_edges)

    missing_edges = []
    for edge in G.edges():
        e1, e2 = edge
        if e1 > e2:
            e1, e2 = e2, e1
        if len(channelArea_df[(channelArea_df['i'] == e1) & (channelArea_df['j'] == e2)]) == 0:
            missing_edges.append((e1, e2))
    print("missing edges: ", missing_edges)


    points_id = [i for i in range(numBeads * 27)]
    points_image_id = [i%numBeads for i in range(numBeads * 27)]
    points_PBC_df = pd.DataFrame({'id': points_id, 'image_id': points_image_id})

    points_PBC_df['tetra_id'] = [[] for _ in range(len(points_PBC_df))]
    for tid, simplex in enumerate(tetra.simplices):
        for vid in simplex:
            points_PBC_df.loc[vid, 'tetra_id'].append(tid)

    tetra_id = [i for i in range(len(tetra.simplices))]
    tetra_obstacles_id = [sorted(i) for i in tetra.simplices]

    tetra_df = pd.DataFrame({'id': tetra_id, 'obstacles_id': tetra_obstacles_id})


    tetra_df['x'] = comAll[:,0]
    tetra_df['y'] = comAll[:,1]
    tetra_df['z'] = comAll[:,2]

    tetra_df['is_prime'] = (tetra_df['x'] > 0) & (tetra_df['x'] < box_size) & (tetra_df['y'] > 0) & (tetra_df['y'] < box_size) & (tetra_df['z'] > 0) & (tetra_df['z'] < box_size)
    tetra_df['image_id'] = tetra_df['id'].apply(lambda x: indexPrimary[x])

    tetra_df['is_percolation'] = tetra_df['id'].apply(lambda x: x in transitionPrimary)
    tetra_df = tetra_df[tetra_df['is_percolation']]
    tetra_df['percolation_id'] = tetra_df['is_percolation'].cumsum()
    tetra_df.loc[~tetra_df['is_percolation'], 'percolation_id'] = -1
    tetra_df.loc[tetra_df['is_percolation'], 'percolation_id'] = tetra_df.loc[tetra_df['is_percolation'], 'percolation_id']-1
    tetra_df.loc[tetra_df['is_percolation'], 'volume'] = tetra_df.loc[tetra_df['is_percolation'], 'id'].apply(lambda x: idVol[x])
    tetra_df['vol_fraction'] = tetra_df['volume'] / sum(idVol.values())
    tetra_df['vol_fraction'] = tetra_df['vol_fraction'].fillna(0)
    tetra_df['vol_fraction_cum'] = tetra_df['vol_fraction'].cumsum()
    

    tetra_df['num_probes'] = tetra_df['vol_fraction'].apply(lambda x: int(x*numProbes)+1)
    tetra_df.loc[~tetra_df['is_percolation'], 'num_probes'] = 0

    def tetra_volume(a, b, c, d) -> float:
        return abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0

    def point_in_tetra(v, p, tol: float = 1e-5) -> bool:
        v = np.asarray(v, float).reshape(4, 3)
        p = np.asarray(p, float).reshape(3)

        V  = tetra_volume(v[0], v[1], v[2], v[3])
        V1 = tetra_volume(p,    v[1], v[2], v[3])
        V2 = tetra_volume(v[0], p,    v[2], v[3])
        V3 = tetra_volume(v[0], v[1], p,    v[3])
        V4 = tetra_volume(v[0], v[1], v[2], p)

        return np.isclose(V1 + V2 + V3 + V4, V, atol=tol, rtol=0.0)

    probes_XYZ = []
    for i in range(numProbes):
        ifOverlap = True
        while ifOverlap:
            ifOverlap = False
            
            rd = np.random.random()
            ind = tetra_df.loc[tetra_df['vol_fraction_cum']>rd].iloc[0]['id']
            vertices = np.array([points_PBC[i] for i in tetra_df.loc[ind, 'obstacles_id']])
            probeTmp = sample_points_in_tetrahedron(vertices, 1)[0]
            
            if not point_in_tetra(vertices, probeTmp):
                raise SystemExit("Probe particle is not in the tetrahedron")
                
            for obstacle in points_PBC:
                if np.linalg.norm(probeTmp - obstacle) <= r_cut:
                    ifOverlap = True
                    break

        probes_XYZ.append(probeTmp)
    print("Number of probes generated: ", len(probes_XYZ))

    lengthUnit = r_cut*0.5

    tetra_df['x'] = tetra_df['x']/lengthUnit
    tetra_df['y'] = tetra_df['y']/lengthUnit
    tetra_df['z'] = tetra_df['z']/lengthUnit

    tetra_df['volume'] = tetra_df['volume']/lengthUnit**3

    channelArea_df['area'] = channelArea_df['area'] / lengthUnit**2

    # save configurations
    outfile = filepath + '.pos'
    with open(outfile, 'w') as output_fileID:
        output_fileID.write(f'{len(points) + len(probes_XYZ)}\n')
        output_fileID.write(f'Lattice="{box_size/lengthUnit} 0 0 0 {box_size/lengthUnit} 0 0 0 {box_size/lengthUnit}" Properties=species:S:1:pos:R:3:radius:R:1\n')

        for bid in range(len(points)):
            output_fileID.write(f'{0} {points[bid][0]/lengthUnit} {points[bid][1]/lengthUnit} {points[bid][2]/lengthUnit} {2.0}\n')
        for probe in probes_XYZ:
            output_fileID.write(f'{1} {probe[0]/lengthUnit} {probe[1]/lengthUnit} {probe[2]/lengthUnit} {0.0}\n')

    print('Done!')

    # save obstacles info
    outfile = filepath + '_tetra.txt'

    tetra_df["obstacles_id"] = tetra_df["obstacles_id"].apply(
        lambda x: [int(i) for i in x] if isinstance(x, (list, np.ndarray)) else int(x)
    )
    tetra_df[tetra_df['is_prime']].to_csv(outfile, sep=',', index = False, float_format='%.18f')

    outfile = filepath + '_area.txt'
    channelArea_df.to_csv(outfile, index=False, float_format='%.18f')



if __name__ == "__main__":
    # Check if at least one argument is provided
    if len(sys.argv) > 1:
        numBeads = int(sys.argv[1])
        numProbes =int( sys.argv[2])
        box_size = float(sys.argv[3])
        seedNumber = int(sys.argv[4])
        filepath = str(sys.argv[5])

        percolationSample(numBeads, numProbes, box_size, seedNumber, filepath)
    else:
        print("No argument provided")