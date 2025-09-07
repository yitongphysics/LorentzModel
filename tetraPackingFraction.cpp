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

int numTrails = int(1e3);

int nC = 0; // number of probes
int nB = 0; // number of obstacles
double rB = 0.0;
double rC = 0.0;
long double L = 0.0;
int seedNumber=0;

vector<vector<double> > XYZ(nB, vector<double> (3, 0.0)); // position of beads: nB * 3
vector<vector<double> > xyz0(nC, vector<double> (3, 0.0)); // current position of probes: nC * 3

// file positions
string pathname = "/Users/yitongzheng/Documents/Corey/BCM/diffusion/configuration/";
//const string pathname = "/gpfs/gibbs/pi/ohern/yz974/BCM/probeSize/0823/";
string outputFile = pathname + "packingFraction_1000.txt";
string inputFile = pathname + "fireRadius_1000.pos";
string tetraFile = pathname + "1000_0_tetra.txt";


struct tetra {
    vector<int> obstacles_id;  // parsed into integers
    double x, y, z;
    bool is_percolation;
    double vol_fraction;
};
unordered_map<int, tetra> tetraInfo;

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

vector<double> sample_point_in_tetrahedron(const vector<vector<double> > vertices) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    vector<double> rn;
    rn.resize(4);
    for (int i = 0; i < 4; i++) {
        double u = dis(gen);
        rn[i] = -std::log(u);
    }

    double sum = std::accumulate(rn.begin(), rn.end(), 0.0);
    for (int i = 0; i < 4; i++) {
        rn[i] /= sum;
    }

    vector<double> point = {0.0, 0.0, 0.0};
    point.resize(4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            point[j] += rn[i] * vertices[i][j];
        }
    }

    return point;
}

int detectOverlap(vector<double> point){
    for(int bi=0; bi<XYZ.size(); bi++){
        double dx = point[0] - XYZ[bi][0];
        double dy = point[1] - XYZ[bi][1];
        double dz = point[2] - XYZ[bi][2];

        dx -= L * round(dx / L);
        dy -= L * round(dy / L);
        dz -= L * round(dz / L);

        if(dx*dx + dy*dy + dz*dz < (rB+rC) * (rB+rC)){
            return 2;
        }
    }

    for(int ci=0; ci<xyz0.size(); ci++){
        double dx = point[0] - xyz0[ci][0];
        double dy = point[1] - xyz0[ci][1];
        double dz = point[2] - xyz0[ci][2];

        dx -= L * round(dx / L);
        dy -= L * round(dy / L);
        dz -= L * round(dz / L);

        if(dx*dx + dy*dy + dz*dz < (rC) * (rC)){
            return 1;
        }
    }

    return 0;
}

double packingFraction(int tetra_id){
    vector<vector<double> > vertices (4, vector<double> (3, 0.0));
    int cnt1 = 0;
    int cnt2 = 0;

    for(int i=0; i<4; i++){
        for(int j=0;j<3; j++){
            vertices[i][j] = XYZ.at(tetraInfo[tetra_id].obstacles_id[i] % nB)[j];
        }
    }

    // shift together
    for(int i=1; i<4; i++){
        for(int j=0; j<3; j++){
            if(vertices[i][j] - vertices[0][j] > L/2) vertices[i][j] -= L;
            if(vertices[i][j] - vertices[0][j] < -L/2) vertices[i][j] += L;
        }
    }

    for(int i =0; i < numTrails; i++){
        int overlap = detectOverlap(sample_point_in_tetrahedron(vertices));

        if(overlap == 1) cnt1 += 1;
        if(overlap == 2) cnt2 += 1;
    }

    if(numTrails == cnt2){
        return 0.0;
    }

    return double(cnt1) / (numTrails - cnt2);
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

        outputFile = pathname + "/packingFraction/pf_" + to_string(nB) + "_" + to_string(nC) + "_" + to_string(seedNumber) + ".txt";
        inputFile = pathname + "/configuration/fireRadius_" + to_string(nB) + "_" + to_string(nC) + "_" + to_string(seedNumber) + ".pos";
        tetraFile = pathname + "/configuration/" + to_string(nB) + "_" + to_string(seedNumber) + "_tetra.txt";

    }
    cout << outputFile << endl;

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

    string line;
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
    int numTetra = tetraInfo.size();

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

    // save results
    ofstream pfFile(outputFile, ios::out);
    pfFile.precision(15);
    for (auto &kv : tetraInfo) {
        if(kv.second.is_percolation) pfFile << kv.first << " ";
    }
    pfFile << endl;

    while (true){
        if (!getline(file, line)){
            cout << "EOF" << endl;
            break;
        }

        if (line[0] == 'L'){
            sscanf(line.c_str(),"Lattice=\"%Lf",&L);
            XYZ.resize(0);
            xyz0.resize(0);
            while (getline(file, line)) {

                int pid = 0;
                double x0, y0, z0, r0;
                vector<double> newVector;
                // Read the file line by line
                sscanf(line.c_str(),"%d %lf %lf %lf %lf", &pid, &x0, &y0, &z0, &r0);
                if(pid > 2) break;

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

                }
            }
            nB = XYZ.size();
            nC = xyz0.size();
            
            cout << endl;
            cout << rC << endl;
            
            // save pf res
            for (auto &kv : tetraInfo) {
                if(kv.second.is_percolation) pfFile << double(packingFraction(kv.first)) << " ";
            }
            pfFile << endl;
        }
    }


    return 0;
}