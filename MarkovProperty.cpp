/*
Input: position file. tetrahedron information file. Number of obstacles. Number of probes. Simulation time. dt.
Output: time step: number of transitions between tetrahedrons.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <tuple>
#include <map>
#include <unordered_map>

using namespace std;

// parameters
double rC = 0.0; // radius of cancer cells
int nC = 1000; // number of cancer cells
double rB = 1.0; //radius of beads
int nB = 1000; // number of beads
long double L = 0; //pow((nC*rC*rC*rC+nB*rB*rB*rB)*4.0/3.0*M_PI/phi, 1.0/3.0); // size of box
double step0 = 0.01; // cancer movement step
long double T = 1e6;  // number of time steps
int tSkip = -1; // save data every tSkip frames
int nFrame = 237;

// updating variables
vector<vector<double> > XYZ(nB, vector<double> (3, 0.0)); // position of beads: nB * 3
vector<vector<double> > pxyz(nC, vector<double> (3, 0.0)); // direction of probes: nC * 3
vector<vector<double> > xyz0(nC, vector<double> (3, 0.0)); // current position of probes: nC * 3
vector<long int> probeTetra(nC, -1); // current tetrahedron of probes: nC * 1
vector<vector<int> > neighborList(nC, vector<int> (nB+nC,-1)); // neighbor list: nC * (nB + nC)

// file positions
string pathname = "/Users/yitongzheng/Documents/Corey/BCM/diffusion/configuration/test/";
//const string pathname = "/gpfs/gibbs/pi/ohern/yz974/BCM/probeSize/0823/";
string markovFile = pathname + "markov.txt";
string tetraFile = pathname + "1000_42_tetra.txt";
string inputFile = pathname + "fireRadius_1000_1000_42.pos";
string posFile = pathname + "visual.pos";

struct tetra {
    vector<long int> obstacles_id;  // parsed into integers
    double x, y, z;
    bool is_percolation;
    double vol_fraction;
};

map<long int, tetra> tetraInfo;

static inline string trim(string s) {
    auto not_space = [](int ch){ return !std::isspace(ch); };
    s.erase(s.begin(), find_if(s.begin(), s.end(), not_space));
    s.erase(find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

vector<string> split_csv(const string &line) {
    vector<string> out;
    string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"'); // escaped quote
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

// Parse "[53, 76, 91, 2256]" -> {53, 76, 91, 2256}
vector<long int> parse_obstacles(const string &s_in, int numObstacles) {
    vector<long int> res;
    string s = s_in;
    string num;
    for (char c : s) {
        if (isdigit(c) || (c == '-' && num.empty())) {
            num.push_back(c);
        } else if (!num.empty()) {
            res.push_back(stoi(num) % numObstacles);
            num.clear();
        }
    }
    if (!num.empty()){
        res.push_back(stoi(num) % numObstacles);
    }
    return res;
}

bool to_bool(string s) {
    s = trim(s);
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s == "true" || s == "1" || s == "t" || s == "yes");
}

double parse_double(const string &s, double def = 0.0) {
    string t = trim(s);
    if (t.empty()) return def;
    return stod(t);
}

void newPhi() {
    static default_random_engine gen(42); // fixed seed for reproducibility
    //random_device rd;
    //default_random_engine gen(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);
    double p0;
    double x1, x2, s;
    for (int ci = 0; ci < nC; ci ++){
        do {
            x1 = dist(gen);
            x2 = dist(gen);
            s = x1 * x1 + x2 * x2;
        } while (s >= 1.0);

        pxyz[ci][0] = 2 * x1 * sqrt(1 - s);
        pxyz[ci][1] = 2 * x2 * sqrt(1 - s);
        pxyz[ci][2] = 1 - 2 * s;
    }
    return;
}

// neighborList[ci] = bi or cj + nb
void updateNeighborList(double neightborLengthBC, double neightborLengthCC){
    int cntNeigh;
    double dx, dy, dz;
    fill(neighborList.begin(), neighborList.end(), vector<int>(nB+nC, -1));
    for (int ci = 0; ci < nC; ci++){
        cntNeigh = 0;
        for (int bi = 0; bi < nB; bi++){
            dx = XYZ[bi][0] - xyz0[ci][0];
            dx -= L * round(dx / L);
            if (abs(dx) > neightborLengthBC) continue;

            dy = XYZ[bi][1] - xyz0[ci][1];
            dy -= L * round(dy / L);
            if (abs(dy) > neightborLengthBC) continue;

            dz = XYZ[bi][2] - xyz0[ci][2];
            dz -= L * round(dz / L);
            if (abs(dz) > neightborLengthBC) continue;
            
            if (dx*dx + dy*dy + dz*dz < neightborLengthBC*neightborLengthBC){
                neighborList[ci][cntNeigh] = bi;
                cntNeigh ++;
            }
        }
        // probe neighbors
        for (int cj = 0; cj < nC; cj++){
            if (cj == ci) {
                neighborList[ci][cntNeigh] = cj+nB;
                cntNeigh ++;
                continue;
            }
            dx = xyz0[cj][0] - xyz0[ci][0];
            dx -= L * round(dx / L);
            if (abs(dx) > neightborLengthCC) continue;

            dy = xyz0[cj][1] - xyz0[ci][1];
            dy -= L * round(dy / L);
            if (abs(dy) > neightborLengthCC) continue;

            dz = xyz0[cj][2] - xyz0[ci][2];
            dz -= L * round(dz / L);
            if (abs(dz) > neightborLengthCC) continue;
            
            if (dx*dx + dy*dy + dz*dz < neightborLengthCC*neightborLengthCC){
                neighborList[ci][cntNeigh] = cj+nB;
                cntNeigh ++;
            }
        }
    }
}

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

using Vec3 = vector<double>;   // expect size = 3

double dot(const Vec3& a, const Vec3& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return { a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0] };
}

Vec3 operator-(const Vec3& a, const Vec3& b) {
    return { a[0]-b[0], a[1]-b[1], a[2]-b[2] };
}

double tetraVolume(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    return fabs(dot(a-d, cross(b-d, c-d))) / 6.0;
}

bool pointInTetra(const vector<Vec3>& v, const Vec3& p, double tol=1e-5) {
    if (v.size() != 4) throw invalid_argument("Need 4 vertices for tetrahedron");

    double V  = tetraVolume(v[0], v[1], v[2], v[3]);
    double V1 = tetraVolume(p,    v[1], v[2], v[3]);
    double V2 = tetraVolume(v[0], p,    v[2], v[3]);
    double V3 = tetraVolume(v[0], v[1], p,    v[3]);
    double V4 = tetraVolume(v[0], v[1], v[2], p);

    double sum = V1 + V2 + V3 + V4;
    return fabs(sum - V) <= tol;
}

long int findTetra(int probeID){
    double dx, dy, dz;

    vector<double> point (3, 0.0);
    for(int j=0;j<3; j++){
        point[j] = xyz0[probeID][j];
    }

    // test if remain
    if(probeTetra[probeID] != -1){
        int tid = probeTetra[probeID];

        dx = point[0] - tetraInfo[tid].x;
        point[0] -= L * round(dx / L);
        dx -= L * round(dx / L);
        //if (abs(dx) > L/10) continue;

        dy = point[1] - tetraInfo[tid].y;
        point[1] -= L * round(dy / L);
        dy -= L * round(dy / L);
        //if (abs(dy) > L/10) continue;

        dz = point[2] - tetraInfo[tid].z;
        point[2] -= L * round(dz / L);
        dz -= L * round(dz / L);
        //if (abs(dz) > L/10) continue;
        vector<vector<double> > vertices (4, vector<double> (3, 0.0));
        for(int i=0; i<4; i++){
            vertices[i][0] = XYZ[tetraInfo[tid].obstacles_id[i]][0];
            dx = vertices[i][0] - tetraInfo[tid].x;
            vertices[i][0] -= L * round(dx / L);

            vertices[i][1] = XYZ[tetraInfo[tid].obstacles_id[i]][1];
            dy = vertices[i][1] - tetraInfo[tid].y;
            vertices[i][1] -= L * round(dy / L);

            vertices[i][2] = XYZ[tetraInfo[tid].obstacles_id[i]][2];
            dz = vertices[i][2] - tetraInfo[tid].z;
            vertices[i][2] -= L * round(dz / L);
        }

        // if the probe is inside the tetrahedron, return the tetrahedron id
        if (pointInTetra(vertices, point)){
            return tid;
        }
    }

    for (auto &kv : tetraInfo){
        if (kv.second.is_percolation or 1){
            dx = point[0] - kv.second.x;
            point[0] -= L * round(dx / L);
            //dx -= L * round(dx / L);
            //if (abs(dx) > L/10) continue;

            dy = point[1] - kv.second.y;
            point[1] -= L * round(dy / L);
            //dy -= L * round(dy / L);
            //if (abs(dy) > L/10) continue;

            dz = point[2] - kv.second.z;
            point[2] -= L * round(dz / L);
            //dz -= L * round(dz / L);
            //if (abs(dz) > L/10) continue;
            vector<vector<double> > vertices (4, vector<double> (3, 0.0));
            for(int i=0; i<4; i++){
                vertices[i][0] = XYZ[kv.second.obstacles_id[i]][0];
                dx = vertices[i][0] - kv.second.x;
                vertices[i][0] -= L * round(dx / L);

                vertices[i][1] = XYZ[kv.second.obstacles_id[i]][1];
                dy = vertices[i][1] - kv.second.y;
                vertices[i][1] -= L * round(dy / L);

                vertices[i][2] = XYZ[kv.second.obstacles_id[i]][2];
                dz = vertices[i][2] - kv.second.z;
                vertices[i][2] -= L * round(dz / L);
            }

            // if the probe is inside the tetrahedron, return the tetrahedron id
            if (pointInTetra(vertices, point)){
                if(kv.second.is_percolation == false){
                    cout << "Debug: " << probeID << " in " << kv.first << endl;
                    exit(1);
                }
                return kv.first;
            }
        }
    }
    //cout << "Probe " << probeID << " is not in any tetrahedron" << endl;
    //cout << point[0] << " " << point[1] << " " << point[2] << endl;
    //exit(1);
    return -1;
}

int main(int argc, char const *argv[]) {
    if (argc > 1){
        string nCStr             =   argv[1];
        string TStr              =   argv[2];
        string nFrameStr         =   argv[3];
        string tetraFileStr      =   argv[4];
        string inputStr          =   argv[5];
        string markovFileStr     =   argv[6];

        stringstream nCSS(nCStr);
        stringstream TSS(TStr);
        stringstream nFrameSS(nFrameStr);
        stringstream tetraFileSS(tetraFileStr);
        stringstream inputSS(inputStr);
        stringstream markovFileSS(markovFileStr);

        nCSS >> nC;
        TSS >> T;
        nFrameSS >> nFrame;
        tetraFileSS >> tetraFile;
        inputSS >> inputFile;
        markovFileSS >> markovFile;
    }

    // ******************** //
    //                      //
    //      READ  FILE      //
    //                      //
    // ******************** //
    string line;
    ifstream fileTmp(inputFile);
    if (!fileTmp) {
        std::cerr << "Cannot open the file!" << std::endl;
        return 1;
    }

    int fileEnd = 0;
    while (1){
        if (!getline(fileTmp, line)){
            break;
        }
        if (line[0] == 'L'){
            fileEnd += 1;
        }
    }

    if(fileEnd < nFrame){
        cout << "Not enough Frame!" << endl;
        exit(1);
    }

    ifstream file(inputFile);
    int cntFile = 0;
    while (cntFile < nFrame){
        if (!getline(file, line)){
            cout << "EOF,2" << endl;
            exit(1);
        }
        if (line[0] == 'L'){
            cntFile += 1;
        }
    }
    sscanf(line.c_str(),"Lattice=\"%Lf",&L);

    XYZ.resize(0);
    xyz0.resize(0);
    while (getline(file, line)) {
        int pid = 0;
        double x0, y0, z0, r0;
        vector<double> newVector;
        // Read the file line by line
        sscanf(line.c_str(),"%d %lf %lf %lf %lf", &pid, &x0, &y0, &z0, &r0);
        if (x0 < 0){x0 += L;}
        if (x0 > L){x0 -= L;}
        if (y0 < 0){y0 += L;}
        if (y0 > L){y0 -= L;}
        if (z0 < 0){z0 += L;}
        if (z0 > L){z0 -= L;}
        newVector = {x0, y0, z0};
        if (r0 > 1.0){
            XYZ.push_back(newVector);
            rB = r0;
        }
        else{
            xyz0.push_back(newVector);
            rC = r0;
            if (xyz0.size() == nC ){
                break;
            }
        }
    }
    nB = XYZ.size();

    int nC0 = xyz0.size(); // current number of cancer cells
    
    while (xyz0.size() != nC){
        random_device rd;
        default_random_engine gen(rd());
        uniform_real_distribution<> dist(-1.0, 1.0);
        for (int ci = 0; ci < nC0; ci++){
            vector<double> newVector;
            newVector = {xyz0[ci][0]+0.01*dist(gen)-0.005, xyz0[ci][1]+0.01*dist(gen)-0.005, xyz0[ci][2]+0.01*dist(gen)-0.005};
            xyz0.push_back(newVector);
            if (xyz0.size() == nC){
                break;
            }
        }
    }

    double dBC = rB + rC; // minimal distance between cancer cells and beads
    double dCC = rC + rC; // minimal distance between cancer cells

    // print parameter:
    cout << "L : " << L << endl;
    cout << "rC: " << rC << endl;
    cout << "nC: " << nC << endl;
    cout << "rB: " << rB << endl;
    cout << "nB: " << nB << endl;
    cout << "step: " << step0 << endl;
    cout << "T: " << T << endl;
    cout << "tSkip: " << tSkip << endl;


    // ******************** //
    //                      //
    //    Read Tetra info   //
    //                      //
    // ******************** //
    ifstream fin(tetraFile);
    if (!fin.is_open()) {
        cout << tetraFile << endl;
        cerr << "Error opening file\n";
        return 1;
    }

    getline(fin, line); // skip header

    while (getline(fin, line)) {
        if (line.empty()) continue;

        auto cols = split_csv(line);

        // Expect at least 13 columns from your format
        if (cols.size() < 12) continue;

        // Columns (0-based):
        // 0 id, 1 obstacles_id, 2 x, 3 y, 4 z, 5 is_prime, 6 image_id,
        // 7 is_percolation, 8 percolation_id, 9 volume (can be empty),
        // 10 vol_fraction, 11 vol_fraction_cum, 12 num_probes
        long int id = stoi(trim(cols[0]));
        vector<long int> obstacles = parse_obstacles(cols[1], nB);
        double x = parse_double(cols[2]);
        double y = parse_double(cols[3]);
        double z = parse_double(cols[4]);
        bool is_perc = to_bool(cols[7]);
        double vol_frac = parse_double(cols[10]);

        //if(is_perc) 
        tetraInfo[id] = tetra{obstacles, x, y, z, is_perc, vol_frac};
    }

    fin.close();
    int numTetra = tetraInfo.size();
    cout << "numTetra: " << numTetra << endl;
    // ******************** //
    //                      //
    //    Initialization    //
    //                      //
    // ******************** //

    // pxyz
    pxyz.resize(nC);
    for (int i = 0; i < pxyz.size(); ++i) {
        pxyz[i].resize(3, 0.0);
    }
    newPhi();
    // neighbor list
    neighborList.resize(nC);
    for (int i = 0; i < neighborList.size(); ++i) {
        neighborList[i].resize(nB+nC, -1);
    }

    // ******************** //
    //                      //
    //         FIRE         //
    //                      //
    // ******************** //
    int i=0;
    double P = 0;
    double fnorm = 0;
    double vnorm = 0;
    double alpha0 = 0.15;
    double alpha = alpha0;
    double finc = 1.1;
    double fdec = 0.5;
    double falpha = 0.99;

    double dt0 = 1e-5;
    double dtmax = 1.0;
    double dtmin = 0.0;

    double maxPos = 0.0;
    double rij, sij, dtmp, ftmp, vftmp;
    
    int npPos = 0;
    int npNeg = 0;
    int NMIN = 10;

    double Ftol = 1e-6;
    int fireit = 0;
    double fcheck = 10 * Ftol;

    vector<double> dr(3, 0.0);
    vector<double> dv((nC+nB)*3, 0.0);
    vector<double> dF((nC+nB)*3, 0.0);
    vector<double> drad(nC+nB, 0.0);
    vector<double> dpos((nC+nB)*3, 0.0);
    for (int bi = 0; bi < nB; bi ++){
        dpos[3*bi]     = XYZ[bi][0];
        dpos[3*bi + 1] = XYZ[bi][1];
        dpos[3*bi + 2] = XYZ[bi][2];
        drad[bi] = rB*1.0;
    }
    for (int ci = nB; ci < nB+nC; ci ++){
        dpos[3*ci] = xyz0[ci-nB][0];
        dpos[3*ci + 1] = xyz0[ci-nB][1];
        dpos[3*ci + 2] = xyz0[ci-nB][2];
        drad[ci] = rC*1.0;
    }

    int ifOverlap = 0;
    while(ifOverlap){
        ifOverlap = 0;
        // FIRE step 1. Compute P
        P = 0.0;
        for (int i = nB*3; i < (nC+nB)*3; i++)
            P += dv[i] * dF[i];
        // FIRE step 2. adjust simulation based on net motion of degrees of freedom
        if (P > 0) {
            npPos++;
            npNeg = 0;
            if (npPos > NMIN) {
                if (dt0 * finc < dtmax)
                    dt0 *= finc;
                alpha *= falpha;
            }
        }
        else{
            npPos = 0;
            npNeg++;
            if (npNeg > 1000){
                cerr << "    ** ERROR: During initial FIRE minimization, P < 0 for too long, so ending." << endl;
            }
            // take half step backwards, reset velocities
            for (i = nB*3; i < (nC+nB)*3; i++)
            {
                dpos[i] -= 0.5 * dt0 * dv[i];
                if (dpos[i] < 0) dpos[i] += L;
                if (dpos[i] > L) dpos[i] -= L;
                dv[i] = 0.0;
            }

            // decrease time step if past initial delay
            if (fireit > 20)
            {
                // decrease time step
                if (dt0 * fdec > dtmin)
                    dt0 *= fdec;

                // reset alpha
                alpha = alpha0;
            }
        }
        // FIRE step 3. First VV update
        for (i = nB*3; i < (nC+nB)*3; i++)
            dv[i] += 0.5 * dt0 * dF[i];
        // FIRE step 4. adjust velocity magnitude
        fnorm = 0.0;
        vnorm = 0.0;
        for (i = nB*3; i < (nC+nB)*3; i++) {
            fnorm += dF[i] * dF[i];
            vnorm += dv[i] * dv[i];
        }
        fnorm = sqrt(fnorm)/(nC+nB)/3;
        vnorm = sqrt(vnorm)/(nC+nB)/3;
        if (fnorm > 0) {
            for (i = nB*3; i < (nC+nB)*3; i++)
                dv[i] = (1 - alpha) * dv[i] + alpha * (vnorm / fnorm) * dF[i];
        }
        // FIRE step 4. Second VV update
        for (i = nB*3; i < (nC+nB)*3; i++) {
            dpos[i] += dt0 * dv[i];
            if (dpos[i] < 0) dpos[i] += L;
            if (dpos[i] > L) dpos[i] -= L;
            dF[i] = 0.0;
        }
        // FIRE step 5. Update forces
        for (int ci = nB; ci < nB+nC; ci++) {
            for (int cj = 0; cj < nB+nC; cj++) {
                if (ci == cj) {
                    continue;
                }
                sij = drad[ci] + drad[cj];
                rij = 0.0;
                for (int d = 0; d < 3; d++) {
                    dtmp = dpos[3 * cj + d] - dpos[3 * ci + d];
                    dtmp -= L * round(dtmp / L);
                    rij += dtmp * dtmp;
                    dr[d] = dtmp;
                }
                rij = sqrt(rij);
                // check distances
                if (rij < sij) {
                    //ifOverlap = 1;
                    cout << ci << " " << cj <<  endl;
                    // force magnitude
                    ftmp = 10.0 * (1.0 - (rij / sij)) / sij;
                    for (int d = 0; d < 3; d++)
                    {
                        vftmp = ftmp * (dr[d] / rij);
                        dF[3 * ci + d] -= vftmp;
                        dF[3 * cj + d] += vftmp;
                    }
                }
            }
        }
        // FIRE step 5. Final VV update
        for (i = nB*3; i < (nC+nB)*3; i++)
            dv[i] += 0.5 * dt0 * dF[i];
        // update forces to check
        fcheck = 0.0;
        for (i = nB*3; i < (nC+nB)*3; i++)
            fcheck += dF[i] * dF[i];
        fcheck = sqrt(fcheck / (nC));

        //print
        if (fireit % 10 == 0){
            cout << "    ********************************" << endl;
            cout << "    ** fireit = " << fireit << endl;
            cout << "    ** fcheck = " << fcheck << endl;
            cout << "    ** P = " << P << endl;
            cout << "    ** dt = " << dt0 << endl;
            cout << "    ** fnorm = " << fnorm << endl;
            cout << "    ** vnorm = " << vnorm << endl;
        }
        fireit ++;
    }
    cout << "    ********************************" << endl;
    cout << "    ** fireit = " << fireit << endl;
    cout << "    ** fcheck = " << fcheck << endl;
    cout << "    ** P = " << P << endl;
    cout << "    ** dt = " << dt0 << endl;
    cout << "    ** fnorm = " << fnorm << endl;
    cout << "    ** vnorm = " << vnorm << endl;
    
    // save positions
    for (int bi = 0; bi < nB; bi ++){
        XYZ[bi][0] = dpos[3*bi];
        XYZ[bi][1] = dpos[3*bi + 1];
        XYZ[bi][2] = dpos[3*bi + 2];
    }
    for (int ci = nB; ci < nB+nC; ci ++){
        xyz0[ci-nB][0] = dpos[3*ci];
        xyz0[ci-nB][1] = dpos[3*ci + 1];
        xyz0[ci-nB][2] = dpos[3*ci + 2];
    }

    // ******************** //
    //                      //
    //      Simulation      //
    //                      //
    // ******************** //

    double F0; // magnitude of a single force.
    double dx, dy, dz, dij;
    vector<double> xyzStart(3, 0.0);
    vector<double> xyzCollision(3, 0.0);
    vector<double> rHat(3, 0.0);
    double rHatNorm = 0.0, rHatP = 0.0;
    double A, B, C, delta; // solving equations
    double stepFraction, stepFractionTmp;
    int minBi = -1;
    int ifCollision = 1;
    int tNeighbor = 100;
    double neightborLengthBC = step0*double(tNeighbor) + dBC;
    double neightborLengthCC = 2*step0*double(tNeighbor) + dCC;
    double step;

    updateNeighborList(neightborLengthBC, neightborLengthCC);
    // update probeTetra
    for (int ci = 0; ci < nC; ci++){
        probeTetra[ci] = findTetra(ci);
        if(probeTetra[ci] == -1){
            //cout << ci << " " << xyz0[ci][0] << " " << xyz0[ci][1] << " " << xyz0[ci][2] << endl;
        }
    }

    ofstream markovfile(markovFile, ios::out);
    //ofstream posfile0(posFile, ios::out);
    for (long long t = 0; t < T; t++) {
        // loop cancer cells
        for (int ci = 0; ci < nC; ci++) {
            // diffusion
            //newPhi();
            ifCollision = 1;
            step = step0;
            xyzStart = xyz0[ci];

            // find earliest collision
            while (ifCollision){
                minBi = -1;
                ifCollision = 0;
                stepFraction = 1.0;
                A = step*step;
                for (int ni = 0; ni < nB+nC; ni++){
                    int bi = neighborList[ci][ni];
                    //cout << ci << " " << bi << endl;
                    if (bi < 0) break;
                    if (bi < nB) {
                        dx = xyzStart[0] - XYZ[bi][0];
                        dx -= L * round(dx / L);
                        dy = xyzStart[1] - XYZ[bi][1];
                        dy -= L * round(dy / L);
                        dz = xyzStart[2] - XYZ[bi][2];
                        dz -= L * round(dz / L);

                        // if overlap: do specular reflection
                        B = dx*pxyz[ci][0] + dy*pxyz[ci][1] + dz*pxyz[ci][2];
                        if (B < 0){
                            B *= 2.0*step;
                            C = dx*dx + dy*dy + dz*dz - dBC*dBC;
                            if (C < 0){
                                //cout << "Error: C < 0" << endl;
                                //cout << t << endl;
                                cout << ci << endl;
                                cout << bi << endl;
                                //cout << dx*dx + dy*dy + dz*dz << endl;
                                //cout << dBC*dBC << endl;
                                exit(1);
                            }
                            delta = B*B - 4.0*A*C;
                            if (delta > 0){
                                stepFractionTmp = (-B - sqrt(delta))/A*0.5;
                                if (stepFractionTmp < stepFraction){
                                    stepFraction = stepFractionTmp;
                                    minBi = bi;
                                    ifCollision = 1;
                                }
                            }
                        }
                    }
                    // probe particles collision
                    if (bi >= nB){
                        bi -= nB;
                        if (bi == ci){
                            continue;
                        }
                        dx = xyzStart[0] - xyz0[bi][0];
                        dx -= L * round(dx / L);
                        dy = xyzStart[1] - xyz0[bi][1];
                        dy -= L * round(dy / L);
                        dz = xyzStart[2] - xyz0[bi][2];
                        dz -= L * round(dz / L);

                        // if overlap: do specular reflection
                        B = dx*pxyz[ci][0] + dy*pxyz[ci][1] + dz*pxyz[ci][2];
                        if (B < 0){
                            B *= 2.0*step;
                            C = dx*dx + dy*dy + dz*dz - dCC*dCC;
                            if (C<0){
                                cout << ci << endl;
                                cout << bi << endl;
                                cout << "CC" << endl;
                                exit(1);
                            }
                            delta = B*B - 4.0*A*C;
                            if (delta > 0){
                                stepFractionTmp = (-B - sqrt(delta))/A*0.5;
                                if (stepFractionTmp < stepFraction){
                                    stepFraction = stepFractionTmp;
                                    minBi = bi+nB;
                                    ifCollision = 1;
                                }
                            }
                        }
                    }
                }
                // handle collisions
                if (ifCollision){
                    xyzStart[0] += stepFraction * pxyz[ci][0] * step;
                    xyzStart[1] += stepFraction * pxyz[ci][1] * step;
                    xyzStart[2] += stepFraction * pxyz[ci][2] * step;

                    if (minBi < nB){
                        rHat[0] = xyzStart[0] - XYZ[minBi][0];
                        rHat[1] = xyzStart[1] - XYZ[minBi][1];
                        rHat[2] = xyzStart[2] - XYZ[minBi][2];
                    }
                    else{
                        minBi -= nB;
                        rHat[0] = xyzStart[0] - xyz0[minBi][0];
                        rHat[1] = xyzStart[1] - xyz0[minBi][1];
                        rHat[2] = xyzStart[2] - xyz0[minBi][2];
                    }
                    rHat[0] -= L * round(rHat[0] / L);
                    rHat[1] -= L * round(rHat[1] / L);
                    rHat[2] -= L * round(rHat[2] / L);

                    rHatNorm = rHat[0]*rHat[0] + rHat[1]*rHat[1] + rHat[2]*rHat[2]; // dBC*dBC

                    rHatP = rHat[0]*pxyz[ci][0] + rHat[1]*pxyz[ci][1] + rHat[2]*pxyz[ci][2];
                    rHatP /= rHatNorm;

                    pxyz[ci][0] -= 2 * rHatP * rHat[0];
                    pxyz[ci][1] -= 2 * rHatP * rHat[1];
                    pxyz[ci][2] -= 2 * rHatP * rHat[2];

                    step *= 1-stepFraction;
                }
                // if no collision, update
                else{
                    xyz0[ci][0] = xyzStart[0] + stepFraction * pxyz[ci][0] * step;
                    xyz0[ci][1] = xyzStart[1] + stepFraction * pxyz[ci][1] * step;
                    xyz0[ci][2] = xyzStart[2] + stepFraction * pxyz[ci][2] * step;
                }
            }
        }

        // print transition Rate file
        for (int ci = 0; ci < nC; ci++){
            long int newTetra = findTetra(ci);
            if ((newTetra != probeTetra[ci]) & (t > tSkip)){
                markovfile << ci << " "<< probeTetra[ci] << " " << newTetra << " " << pxyz[ci][0] << " " << pxyz[ci][1] << " " << pxyz[ci][2] << endl;
            }
            probeTetra[ci] = newTetra;
        }

        // visualization
        /*
        posfile0 << 1+8 << endl;
        posfile0 << "Lattice=\"" << L << " 0 0 0 " << L << " 0 0 0 " << L << "\" Properties=species:S:1:pos:R:3:radius:R:1" << endl;
        for (int bi =0; bi < nB; bi++){
            //posfile0 << 0 << " " << fmod(fmod(XYZ[bi][0], L) + L, L) << " " << fmod(fmod(XYZ[bi][1], L) + L, L) << " " << fmod(fmod(XYZ[bi][2], L) + L, L) << " " << rB << endl;
        }
        for(int bi=0; bi < 4; bi ++){
            //cout << tetraInfo[10319].obstacles_id[bi] << endl;
            posfile0 << 0 << " " << fmod(fmod(XYZ[tetraInfo[t1].obstacles_id[bi]][0], L) + L, L) << " " << fmod(fmod(XYZ[tetraInfo[t1].obstacles_id[bi]][1], L) + L, L) << " " << fmod(fmod(XYZ[tetraInfo[t1].obstacles_id[bi]][2], L) + L, L) << " " << rB << endl;
            posfile0 << 1 << " " << fmod(fmod(XYZ[tetraInfo[t2].obstacles_id[bi]][0], L) + L, L) << " " << fmod(fmod(XYZ[tetraInfo[t2].obstacles_id[bi]][1], L) + L, L) << " " << fmod(fmod(XYZ[tetraInfo[t2].obstacles_id[bi]][2], L) + L, L) << " " << rB << endl;
        }
        for(const auto& kv: tetraInfo){
            for(int vi=0; vi<4; vi++){
                int bi=kv.second.obstacles_id[vi];
                //posfile0 << 0 << " " << fmod(fmod(XYZ[bi][0], L) + L, L) << " " << fmod(fmod(XYZ[bi][1], L) + L, L) << " " << fmod(fmod(XYZ[bi][2], L) + L, L) << " " << rB << endl;
            }
        }
        for (int ci =0; ci < nC; ci++){
            if(ci==35) posfile0 << 2 << " " << fmod(fmod(xyz0[ci][0], L) + L, L) << " " << fmod(fmod(xyz0[ci][1], L) + L, L) << " " << fmod(fmod(xyz0[ci][2], L) + L, L) << " " << rC << endl;
        }
        */

        // simulation progress
        if (t%int(T/10) == 0){
            cout << "t: " << t << endl;
        }
        
        // update neighbor list
        if (t%tNeighbor == 0){
            // create neighbor list
            updateNeighborList(neightborLengthBC, neightborLengthCC);
        }
    }

    markovfile.close();
    //posfile0.close();

    return 0;
}
