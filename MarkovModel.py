import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import Delaunay
import networkx as nx
import copy
from collections import defaultdict, Counter
from types import MappingProxyType
import sys


def MarkovModel(numBeads, ratio, filePath, seedNumber):
    box_size = 100.0
    numBeads = int(numBeads)
    ratio = float(ratio)
    filePath += str(numBeads) + '_' + str(ratio) +'_'
    seedNumber = int(seedNumber)

    print("box size = ",box_size)
    print("num bead = ",numBeads)
    print("ratio   = ",ratio)
    print("filePath = ",filePath)
    print("seedNumb = ",seedNumber)

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

    print("Triangularization finished!")



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
    # if at least 3 vertices of the tetra is nside
    # or if 2 vertices inside the COM of the tetra is inside
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

    #Counter(Counter(indexPrimary.values()).values())

    # construct graph and edge list
    def circumRadius(p1, p2, p3):
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        cross_product = np.cross(p2 - p1, p3 - p1)
        area = np.linalg.norm(cross_product) / 2

        return (a * b * c) / (4 * area)

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

    print("Constructing Graph.")
    fullGraph, edgeList = constructGraph(vor)
    edgeList.sort(key = lambda x: x[2])
    print("Graph Constructed!")

    def unique_points_pbc(points, box_size, tol=1e-6):
        # Wrap points into the primary unit cell
        wrapped_points = points - box_size * np.floor(points / box_size)

        # remove duplicates
        wrapped_points = np.round(wrapped_points, decimals=int(-np.log10(tol)))  # Round to avoid floating point errors
        unique_points = np.unique(wrapped_points, axis=0)

        return unique_points

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

    print("Start finding percolation.")
    l = len(edgeList)//2
    r = len(edgeList)
    m = 0
    while r-l>1:
        G = copy.deepcopy(fullGraph)
        m = (l+r) // 2
        for i in range(m):
            G.remove_edge(edgeList[i][0], edgeList[i][1])
        print(m, edgeList[m][2])
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
    print("Percolation Found!")

    '''
    m = mPerco+dPerco
    G = copy.deepcopy(fullGraph)
    for i in range(m):
        G.remove_edge(edgeList[i][0], edgeList[i][1])
    '''
    rPerco = edgeList[m][2]
    r_cut = rPerco * ratio
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
    print("r = ", r_cut, ifPercolate(component, box_size, vor), "Number of vertices = ",len(list(component[0])))

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
        # 1. edges from VT may pass through some tetrahedron. So the two ends of are not in neighboring tetrahedrons.
        # 2. edges from VT may pass through other tetrahedron and then end up in the neighboring tetrahedron.
        commonVertices = set(tetra.simplices[tetra1]) & set(tetra.simplices[tetra2])
        if len(commonVertices) < 3 or not segment_intersects_triangle(points_PBC[list(commonVertices)], p1, p2):
            midPoint = (p1 + p2) / 2
            bisectionSearch(p1, midPoint)
            bisectionSearch(midPoint, p2)

    for edge in list(subG.edges):
        bisectionSearch(vor.vertices[edge[0]], vor.vertices[edge[1]])
            
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

    statePrimaryInverse = {value: key for key, value in enumerate(statePrimary)} # statePrimaryInverse[original id] = new id

    def sample_point_in_tetrahedron(vertices):
        rn = -np.log(np.random.rand(4))
        return np.dot(rn / np.sum(rn), vertices)

    nTrails = 10_000
    def cavityVolume(vertices, r):
        V0 = abs(np.dot(vertices[1]-vertices[0], np.cross(vertices[2]-vertices[0], vertices[3]-vertices[0]))) / 6.0

        cnt = 0
        for _  in range(nTrails):
            pt = sample_point_in_tetrahedron(vertices)
            for i in range(4):
                if np.linalg.norm(pt - vertices[i]) < r:
                    cnt += 1
                    break
        if cnt == nTrails:
            cnt -= 1
        return V0 * (nTrails - cnt) / nTrails


    idVol = np.zeros(nStates)
    for cnt, i in enumerate(statePrimary):
        idVol[cnt] = cavityVolume(points_PBC[tetra.simplices[i]], r_cut)

    def distance_to_bisector_intersection(A, B, C):
        """
        AB cross AC bisector at X, make sure AB is the longest side
        return X and AX distance
        """
        AB = B - A
        AC = C - A
        
        # Parameter t along AB for the intersection:
        t = np.linalg.norm(AC)**2 / (2 * np.dot(AC, AB))
        #print(np.linalg.norm(AB), np.linalg.norm(AC), np.linalg.norm(B-C))
        
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

    transitionPrimary = set()
    transitionRate = dict()
    transitionNeighbor = dict()

    cnt = 0
    G = nx.Graph()
    for i in statePrimary:
        V = idVol[statePrimaryInverse[i]]
        neighborTmp = []
        for neighbor in tetra.neighbors[i]:
            if indexPrimary[neighbor] in statePrimary:
                neighborVertices = points_PBC[tetra.simplices[neighbor]]
                vCommon = list(set(tetra.simplices[i]) & set(tetra.simplices[neighbor]))
                exitArea = computeExitArea(points_PBC[vCommon[0]], points_PBC[vCommon[1]], points_PBC[vCommon[2]], r_cut)
                if exitArea > 0:
                    transitionRate[(i, indexPrimary[neighbor])] = np.longdouble(np.power(exitArea, 0.75) / V)
                    transitionPrimary.add(i)
                    transitionPrimary.add(indexPrimary[neighbor])
                    neighborTmp.append(indexPrimary[neighbor])
                    G.add_edge(i, indexPrimary[neighbor])
        transitionNeighbor[i] = neighborTmp

    transitionRate = MappingProxyType(transitionRate)
    pairs = np.array(list(set(([(min(x), max(x)) for x in transitionRate.keys()]))))
    print("States not accessible: ", (set(statePrimary) - transitionPrimary))
    print("Largest transition rate: ", np.max(list(transitionRate.values())))

    dt = np.longdouble(0.1 / np.max(list(transitionRate.values())))
    transitionRateDt = np.zeros((nStates, nStates))
    for key, value in transitionRate.items():
        i, j = key
        i, j = statePrimaryInverse[i], statePrimaryInverse[j]
        transitionRateDt[i, j] = np.longdouble(transitionRate[key] * dt)

    windingNumberChange = dict() # winding number change from i to j
    for i in statePrimary:
        for j in statePrimary:
            if i >= j:
                continue
            dp = comAll[j] - comAll[i]
            dWN = [0, 0, 0]
            for d in range(3):
                if dp[d] > box_size / 2:
                    dWN[d] -= 1
                elif dp[d] < -box_size / 2:
                    dWN[d] += 1
            windingNumberChange[(i, j)] = tuple(dWN)
            windingNumberChange[(j, i)] = tuple([-x for x in dWN])

    output_filename = filePath + "COM.data"
    with open(output_filename, 'w') as output_fileID:
        for i in statePrimary:
            com = comAll[i]
            output_fileID.write(f'{com[0]} {com[1]} {com[2]} {idVol[statePrimaryInverse[i]]}\n')

    output_filename = filePath + "transitionMatrix.data"
    with open(output_filename, 'w') as output_fileID:
        for key, value in transitionRate.items():
            if key[0] < key[1]:
                value2 = windingNumberChange[key]
                output_fileID.write(f'{statePrimaryInverse[key[0]]} {statePrimaryInverse[key[1]]} {value} {transitionRate[(key[1], key[0])]} {value2[0]} {value2[1]} {value2[2]}\n')
    
    output_filename = filePath + "stateCOM.pos"
    with open(output_filename, 'w') as output_fileID:
        output_fileID.write(f'{len(states)}\n')
        output_fileID.write(f'Lattice="{box_size} 0 0 0 {box_size} 0 0 0 {box_size}" Properties=species:S:1:pos:R:3:radius:R:1\n')
        for i, com in enumerate(comAll[states]):
            output_fileID.write(f'{i} {com[0]} {com[1]} {com[2]} {1}\n')

    output_filename = filePath + "config.pos"
    vertices = vor.vertices[list(component[0])]
    with open(output_filename, 'w') as output_fileID:
        output_fileID.write(f'{len(points) + len(vertices)}\n')
        output_fileID.write(f'Lattice="{box_size} 0 0 0 {box_size} 0 0 0 {box_size}" Properties=species:S:1:pos:R:3:radius:R:1\n')
        for i, point in enumerate(points):
            output_fileID.write(f'{0} {point[0]} {point[1]} {point[2]} {r_cut}\n')
        for i, point in enumerate(vertices):
            output_fileID.write(f'{1} {point[0]} {point[1]} {point[2]} {0}\n')

if __name__ == "__main__":
    # Check if at least one argument is provided
    if len(sys.argv) > 1:
        numBeads = sys.argv[1]
        ratio = sys.argv[2]
        filepath = sys.argv[3]
        seedNumber = sys.argv[4]
        MarkovModel(numBeads, ratio, filepath, seedNumber)
    else:
        print("No argument provided")
