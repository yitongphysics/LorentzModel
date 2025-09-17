#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <tuple>
#include <map>
#include <cctype>
#include <unordered_map>

using namespace std;

// parameters
double rC = 1.0; // radius of cancer cells
int nC = 1000; // number of cancer cells
double rB = 1.0; //radius of beads
int nB = 0; // number of beads
long double L = 0; //pow((nC*rC*rC*rC+nB*rB*rB*rB)*4.0/3.0*M_PI/phi, 1.0/3.0); // size of box
int seedNumber = 42;
double radiusRatio = 1.0;

// updating variables
vector<vector<double> > XYZ(nB, vector<double> (3, 0.0)); // position of beads: nB * 3
vector<vector<double> > pxyz(nC, vector<double> (3, 0.0)); // direction of probes: nC * 3
vector<vector<double> > xyz0(nC, vector<double> (3, 0.0)); // current position of probes: nC * 3
vector<vector<double> > xyzPre(nC, vector<double> (3, 0.0)); // current position of probes: nC * 3
vector<vector<int> > neighborList(nC, vector<int> (nB+nC,-1)); // neighbor list: nC * (nB + nC)

// file positions
string pathname = "/Users/yitongzheng/Documents/Corey/BCM/diffusion/configuration/";
//const string pathname = "/gpfs/gibbs/pi/ohern/yz974/BCM/probeSize/0823/";
string outputFile = pathname + "fireRadius_1000.pos";
string inputFile = pathname + "1000_0.pos";
string tetraFile = pathname + "1000_0_tetra.txt";


struct tetra {
    vector<int> obstacles_id;  // parsed into integers
    double x, y, z;
    bool is_percolation;
    double vol_fraction;
};

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
vector<int> parse_obstacles(const string &s_in, int numObstacles) {
    vector<int> res;
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
    random_device rd;
    default_random_engine gen(rd());
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
void updateRadiusRatio(double newRadiusRatio){
    double dBC = rB + rC; // minimal distance between cancer cells and beads
    rC = (rB + rC) * newRadiusRatio; // allocate radius to probe according to radius ratio
    rB = dBC - rC;
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

bool detectOverlap(vector<vector<double> > XYZ, vector<vector<double> > xyz0, double rB, double rC){
    for(int ci=0; ci<xyz0.size(); ci++){
        for(int bi=0; bi<XYZ.size(); bi++){
            double dx = xyz0[ci][0] - XYZ[bi][0];
            double dy = xyz0[ci][1] - XYZ[bi][1];
            double dz = xyz0[ci][2] - XYZ[bi][2];

            dx -= L * round(dx / L);
            dy -= L * round(dy / L);
            dz -= L * round(dz / L);

            if(dx*dx + dy*dy + dz*dz < (rB+rC) * (rB+rC)){
                return true;
            }
        }
    }

    return false;
}

double tetraVolume(const vector<double> &a,
                   const vector<double> &b,
                   const vector<double> &c,
                   const vector<double> &d) {
    double x1 = b[0] - a[0], y1 = b[1] - a[1], z1 = b[2] - a[2];
    double x2 = c[0] - a[0], y2 = c[1] - a[1], z2 = c[2] - a[2];
    double x3 = d[0] - a[0], y3 = d[1] - a[1], z3 = d[2] - a[2];
    double det = x1 * (y2 * z3 - z2 * y3)
               - y1 * (x2 * z3 - z2 * x3)
               + z1 * (x2 * y3 - y2 * x3);
    return fabs(det) / 6.0;
}

bool insideTetra(vector<vector<double> > vertices, vector<double> probe){
    for(int i=0; i<4; i++){
        for(int j=0; j<3; j++){
            if(vertices[i][j] - probe[j] > L/2) vertices[i][j] -= L;
            if(vertices[i][j] - probe[j] < -L/2) vertices[i][j] += L;
        }
    }

    double V  = tetraVolume(vertices[0], vertices[1], vertices[2], vertices[3]);
    double V1 = tetraVolume(probe, vertices[1], vertices[2], vertices[3]);
    double V2 = tetraVolume(vertices[0], probe, vertices[2], vertices[3]);
    double V3 = tetraVolume(vertices[0], vertices[1], probe, vertices[3]);
    double V4 = tetraVolume(vertices[0], vertices[1], vertices[2], probe);

    double sumV = V1 + V2 + V3 + V4;
    double eps = 1e-9; // tolerance for floating-point error

    return fabs(sumV - V) < eps;
}

int lowestProbeDensity(unordered_map<int, tetra> tetraInfo, vector<vector<double> > xyz0){
    double density = numeric_limits<double>::max();
    double volume = -1;
    int lowestID = -1;

    for(auto &kv : tetraInfo){
        int cnt = 0;
        if(!kv.second.is_percolation) continue;

        vector<vector<double> > vertices(4, vector<double> (3, 0.0));
        for(int i=0; i<4; i++){
            for(int j=0; j<3; j++){
                vertices[i][j] = XYZ[kv.second.obstacles_id[i]][j];
            }
        }
        for(int pi=0; pi < xyz0.size(); pi++){
            if(insideTetra(vertices, xyz0[pi])) cnt += 1;
        }

        if(cnt / kv.second.vol_fraction < density){
            lowestID = kv.first;
            density = cnt / kv.second.vol_fraction;
        }
    }

    return lowestID;
}

int main(int argc, char const *argv[]) {
    if (argc > 1){
        string nBStr            =   argv[1];
        string nCStr            =   argv[2];
        string seedNumberStr    =   argv[3];
        string pathnameStr         =   argv[4];

        stringstream nBSS(nBStr);
        stringstream nCSS(nCStr);
        stringstream seedNumberSS(seedNumberStr);
        stringstream pathnameSS(pathnameStr);

        nBSS >> nB;
        nCSS >> nC;
        seedNumberSS >> seedNumber;
        pathnameSS >> pathname;

        outputFile = pathname + "/configuration/fireRadius_" + to_string(nB) + "_" + to_string(nC) + "_" + to_string(seedNumber) + ".pos";
        inputFile = pathname + "/configuration/" + to_string(nB) + "_" + to_string(seedNumber) + ".pos";
        tetraFile = pathname + "/configuration/" + to_string(nB) + "_" + to_string(seedNumber) + "_tetra.txt";
    }
    
    // ******************** //
    //                      //
    //      READ  FILE      //
    //                      //
    // ******************** //
    ifstream file(inputFile);
    if (!file) {
        cout << inputFile << endl;
        std::cerr << "Cannot open the file!" << std::endl;
        return 1;
    }

    string line;
    //getline(file, line);
    //getline(file, line);
    //sscanf(line.c_str(),"Lattice=\"%Lf",&L);
    int cntFile = 1;
    while (cntFile < 2){
        if (!getline(file, line)){
            cout << "EOF" << endl;
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
        if (pid == 0){
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
    double dxyz = 0.1;
    while (xyz0.size() < nC){
        random_device rd;
        default_random_engine gen(rd());
        uniform_real_distribution<> dist(-1.0, 1.0);
        for (int ci = 0; ci < nC0; ci++){
            vector<double> newVector;
            newVector = {xyz0[ci][0]+dxyz*dist(gen)-dxyz/2.0, xyz0[ci][1]+dxyz*dist(gen)-dxyz/2.0, xyz0[ci][2]+dxyz*dist(gen)-dxyz/2.0};
            xyz0.push_back(newVector);
            if (xyz0.size() == nC){
                break;
            }
        }
    }

    double dBC = rB + rC; // minimal distance between cancer cells and beads
    rC = dBC * 0.0; // allocate radius to probe according to radius ratio
    rB = dBC - rC;
    double dCC = rC + rC; // minimal distance between cancer cells

    // print parameter:
    cout << "L : " << L << endl;
    cout << "rC: " << rC << endl;
    cout << "nC: " << nC << endl;
    cout << "rB: " << rB << endl;
    cout << "nB: " << nB << endl;

    if(detectOverlap(XYZ, xyz0, rB, rC)){
        cout << "Overlap!" << endl;
        return 0;
    }
    // ******************** //
    //                      //
    //    Read Tetra info   //
    //                      //
    // ******************** //
    ifstream fin(tetraFile);
    if (!fin.is_open()) {
        cerr << "Error opening file\n";
        return 1;
    }

    unordered_map<int, tetra> tetraInfo;
    getline(fin, line); // skip header

    while (getline(fin, line)) {
        if (line.empty()) continue;

        auto cols = split_csv(line);
        // Expect at least 13 columns from your format
        if (cols.size() < 13) continue;

        // Columns (0-based):
        // 0 id, 1 obstacles_id, 2 x, 3 y, 4 z, 5 is_prime, 6 image_id,
        // 7 is_percolation, 8 percolation_id, 9 volume (can be empty),
        // 10 vol_fraction, 11 vol_fraction_cum, 12 num_probes
        int id = stoi(trim(cols[0]));
        vector<int> obstacles = parse_obstacles(cols[1], nB);
        double x = parse_double(cols[2]);
        double y = parse_double(cols[3]);
        double z = parse_double(cols[4]);
        bool is_perc = to_bool(cols[7]);
        double vol_frac = parse_double(cols[10]);

        tetraInfo[id] = tetra{obstacles, x, y, z, is_perc, vol_frac};
    }
    fin.close();

    // print tetra info
    if(false){
        for (auto &kv : tetraInfo) {
            cout << "id=" << kv.first << " obstacles_id=[";
            for (size_t i = 0; i < kv.second.obstacles_id.size(); i++) {
                if (i) cout << ",";
                cout << kv.second.obstacles_id[i];
            }
            cout << "] x=" << kv.second.x
                << " y=" << kv.second.y
                << " z=" << kv.second.z
                << " is_percolation=" << kv.second.is_percolation
                << " vol_fraction=" << kv.second.vol_fraction
                << "\n";
            
            break;
        }
        return 0;
    }

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
    double delta = rC * 0.4;
    double dx, dy, dz;

    // ******************** //
    //                      //
    //         FIRE         //
    //                      //
    // ******************** //
    int i=0;
    double P = 0;
    double fnorm = 0;
    double vnorm = 0;
    double alpha0 = 0.15; // 0.15
    double alpha = alpha0;
    double finc = 1.1;
    double fdec = 0.5;
    double falpha = 0.99; // 0.99

    double dt0 = 1e-5;
    double dtmax = 1e-1;
    double dtmin = 1e-10;

    double maxPos = 0.0;
    double rij, sij, dtmp, ftmp, vftmp;
    
    int npPos = 0;
    int npNeg = 0;
    int NMIN = 10;

    double Ftol = 1e-12;
    int fireit = 0;
    double fcheck = 10 * Ftol;

    ofstream posfile0(outputFile, ios::out);
    posfile0.precision(15);

    vector<double> dr(3, 0.0);
    vector<double> dv((nC+nB)*3, 0.0);
    vector<double> dF((nC+nB)*3, 0.0);
    vector<double> drad(nC+nB, 0.0);
    vector<double> dpos((nC+nB)*3, 0.0);
    for (int bi = 0; bi < nB; bi ++){
        dpos[3*bi]     = XYZ[bi][0];
        dpos[3*bi + 1] = XYZ[bi][1];
        dpos[3*bi + 2] = XYZ[bi][2];
        drad[bi] = rB * 1.001;
    }
    for (int ci = nB; ci < nB+nC; ci ++){
        dpos[3*ci] = xyz0[ci-nB][0];
        dpos[3*ci + 1] = xyz0[ci-nB][1];
        dpos[3*ci + 2] = xyz0[ci-nB][2];
        drad[ci] = rC * 1.001;
    }

    double radiusRatio0 = 0.0;
    double radiusRatioMax = -1.0;
    double rMax = 1.0;
    int cimax = -1;
    while (radiusRatio0 <= radiusRatio){
        updateRadiusRatio(radiusRatio0);
        for (int bi = 0; bi < nB; bi ++){
            drad[bi] = rB * 1.001;
        }
        for (int ci = nB; ci < nB+nC; ci ++){
            drad[ci] = rC * 1.001;
        }
        dBC = rB + rC;
        dCC = rC + rC;

        delta = rC * 0.4;
        if(rC == 0.0){
            delta = (rB + rC) * 0.2;
        }
        updateNeighborList(dBC + delta, dCC + delta);
        xyzPre = xyz0;

        fcheck = 10*Ftol;
        fireit = 0;
        int ifOverlap = 1;
        while(ifOverlap){
            ifOverlap = 0;
            rMax = 1.0;
            cimax = -1;
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
                for (int ni = 0; ni < nB+nC; ni++){
                    int cj = neighborList[ci - nB][ni];

                    if (cj < 0) break;
                    if (cj == ci){
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
                        ifOverlap = 1;
                        if (rij / sij < rMax){
                            rMax = rij / sij;
                            cimax = ci;
                        }
                        // force magnitude
                        ftmp = 10.0 * (1.0 - (rij / sij)) / sij;
                        ftmp = min(0.1, max(-0.1, ftmp));
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
                fcheck = max(fcheck, abs(dF[i]));
            //fcheck = sqrt(fcheck / (nC));

            if (0){
                cout << "    ********************************" << endl;
                cout << "    ** fireit = " << fireit << endl;
                cout << "    ** fcheck = " << fcheck << endl;
                cout << "    ** P = " << P << endl;
                cout << "    ** pos = " << npPos << endl;
                cout << "    ** neg = " << npNeg << endl;
                cout << "    ** dt = " << dt0 << endl;
                cout << "    ** fnorm = " << fnorm << endl;
                cout << "    ** vnorm = " << vnorm << endl;
                cout << "    ** rC    = " << rC << endl;
                cout << "    ** rB    = " << rB << endl;
                cout << "    ** rMax  = " << rMax << endl;
                cout << "    ** ci    = " << cimax << endl;
                cout << "    ** dF    = " << dF[cimax*3] << " " << dF[cimax*3 + 1] << " " << dF[cimax*3 + 2] << endl;
                cout << "    ** dv    = " << dv[cimax*3] << " " << dv[cimax*3 + 1] << " " << dv[cimax*3 + 2] << endl;
                cout << "    ** dx    = " << dpos[cimax*3] << " " << dpos[cimax*3 + 1] << " " << dpos[cimax*3 + 2] << endl;
            }
            
            //update neighbor list
            for(int ci = nB; ci < nB+nC; ci++){
                dx = dpos[3*ci] - xyzPre[ci-nB][0];
                dy = dpos[3*ci+1] - xyzPre[ci-nB][1];
                dz = dpos[3*ci+2] - xyzPre[ci-nB][2];
                if(pow(dx, 2) + pow(dy, 2) + pow(dz, 2) > delta * delta / 4.0){
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
                    delta = rC * 0.4;
                    if(rC == 0.0){
                        delta = (rB + rC) * 0.2;
                    }
                    updateNeighborList(dBC + delta, dCC + delta);
                    xyzPre = xyz0;
                    break;
                }
            }

            if (fireit % 100000 == 0 & fireit > 0){
                break;
            }
            fireit ++;
        }

        radiusRatio0 = radiusRatio0 + 0.001;

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

        if (~ifOverlap & (radiusRatio0 > radiusRatioMax)){
            radiusRatioMax = radiusRatio0;

            cout << "    ********************************" << endl;
            cout << "          C O N V E R G E D !       " << endl;
            cout << "    ********************************" << endl;
            cout << "    ** fireit = " << fireit << endl;
            cout << "    ** fcheck = " << fcheck << endl;
            cout << "    ** P = " << P << endl;
            cout << "    ** dt = " << dt0 << endl;
            cout << "    ** fnorm = " << fnorm << endl;
            cout << "    ** vnorm = " << vnorm << endl;
            cout << "    ** rC    = " << rC << endl;
            cout << "    ** rB    = " << rB << endl;
            cout << "    ** rMax  = " << rMax << endl;
            cout << "    ** ci    = " << cimax << endl;

            if (rC > 0.0){
                posfile0 << nC+nB << endl;
                posfile0 << "Lattice=\"" << L << " 0 0 0 " << L << " 0 0 0 " << L << "\" Properties=species:S:1:pos:R:3:radius:R:1" << endl;
                for (int bi =0; bi < nB; bi++){
                    posfile0 << 0 << " " << XYZ[bi][0] << " " << XYZ[bi][1] << " " << XYZ[bi][2] << " " << rB << endl;
                }
                for (int ci =0; ci < nC; ci++){
                    posfile0 << 1 << " " << xyz0[ci][0] << " " << xyz0[ci][1] << " " << xyz0[ci][2] << " " << rC << endl;
                }
            }
        }

        if (ifOverlap){
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dist(-1.0, 1.0);
            
            int lowestTetra = lowestProbeDensity(tetraInfo, xyz0);
            vector<vector<double> > detectProbe(1, vector<double> (3, 0.0));
            dxyz = 1e-3;

            while(true){
                dpos[cimax*3] = tetraInfo[lowestTetra].x   + dxyz*dist(gen)-dxyz/2.0;
                dpos[cimax*3+1] = tetraInfo[lowestTetra].y + dxyz*dist(gen)-dxyz/2.0;
                dpos[cimax*3+2] = tetraInfo[lowestTetra].z + dxyz*dist(gen)-dxyz/2.0;
                
                detectProbe[0][0] = dpos[cimax*3];
                detectProbe[0][1] = dpos[cimax*3+1];
                detectProbe[0][2] = dpos[cimax*3+2];
                if(~detectOverlap(XYZ, detectProbe, rB, rC)) break;
            }
            

            radiusRatio0 = 0.0;
            //exit(1);
        }
    }
    
    return 0;
}
