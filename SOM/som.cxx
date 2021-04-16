#include <cmath>

#include <algorithm>
using std::random_shuffle;

#include <fstream>
using std::ifstream;

#include <limits>
using std::numeric_limits;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;

#include <random>
using std::default_random_engine;
using std::normal_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

class Sample {
    public:
        int type;
        vector<double> values;

        Sample(int _type, int length, double* _values) : type(_type), values(_values, _values + length) {
        }

        //overload so we can easily write to cout and other streams
        friend ostream & operator << (ostream &out, const Sample &s);
        
};

ostream & operator << (ostream &out, const Sample &s) {
    out << "[" << s.type << " :";
    for (int i = 0; i < s.values.size(); i++) {
        out << " " << s.values[i];
    }
    out << "]";

    return out;
}

void print_samples(const vector<Sample> &samples) {
    for (int i = 0; i < samples.size(); i++) {
        cout << samples[i] << endl;
    }
}


void normalize(vector<Sample> &samples) {
    int length = samples[0].values.size();

    double avgs[length];
    double stddevs[length];

    for (int i = 0; i < length; i++) {
        avgs[i] = 0.0;
        stddevs[i] = 0.0;
    }

    for (int i = 0; i < samples.size(); i++) {
        for (int j = 0; j < length; j++) {
            avgs[j] += samples[i].values[j];
        }
    }

    for (int i = 0; i < length; i++) {
        avgs[i] /= samples.size();
    }

    for (int i = 0; i < samples.size(); i++) {
        for (int j = 0; j < length; j++) {
            double diff = samples[i].values[j] - avgs[j];
            stddevs[j] += diff * diff;
        }
    }

    for (int i = 0; i < length; i++) {
        stddevs[i] /= samples.size();
        stddevs[i] = sqrt(stddevs[i]);
    }

    for (int i = 0; i < samples.size(); i++) {
        for (int j = 0; j < length; j++) {
            samples[i].values[j] = (samples[i].values[j] - avgs[j]) / stddevs[j];
        }
    }
}

double euclidian_distance(const vector<double> &v1, const vector<double> &v2) {
	double sum = 0.0;
	for (int i = 0; i < v1.size(); i++) {
		sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return sqrt(sum);
}


class Unit {
    public:
        vector<int> class_stats;
        vector<double> values;

        Unit(int length) : class_stats(length, 0.0), values(length, 0.0) {
        }

        void reset_class_stats() {
            for (int i = 0; i < class_stats.size(); i++) {
                class_stats[i] = 0;
            }
        }

        //so we can easily print the SOM to output
        friend ostream & operator << (ostream &out, const Unit &unit);
        
};

ostream & operator << (ostream &out, const Unit &unit) {
    out << "[";
    for (int i = 0; i < unit.values.size(); i++) {
        out << " " << std::setw(8) << std::setprecision(5) << std::fixed << unit.values[i];
    }

    out << " | BMU FOR ";
    for (int i = 0; i < unit.class_stats.size(); i++) {
        out << "(" << i << " - " << std::setw(3) << unit.class_stats[i] << ") ";
    }
    out << "]";

    return out;
}


class SelfOrganizingMap {
    public:
        double learning_rate;
        double radius;
        vector<Unit> units;

        SelfOrganizingMap(int number_units, double _learning_rate, double _radius, int unit_length) : learning_rate(_learning_rate), radius(_radius), units(number_units, Unit(unit_length)) {
            default_random_engine generator;
            normal_distribution<double> distribution(0.0,1.0);

            for (int i = 0; i < units.size(); i++) {
                for (int j = 0; j < units[i].values.size(); j++) {
                    units[i].values[j] = distribution(generator);
                }
            }
        }

        void reset_class_stats() {
            for (int i = 0; i < units.size(); i++) {
                units[i].reset_class_stats();
            }
        }

        void insert_sample(int current_epoch, int max_epochs, Sample &sample) {
            double min_distance = numeric_limits<double>::max();
            int bmu_index = 0;

            for (int i = 0; i < units.size(); i++) {
                double d = euclidian_distance(sample.values, units[i].values);
                cout << "distance to unit " << i << " is " << d << endl;

                if (d < min_distance) {
                    min_distance = d;
                    bmu_index = i;
                }
            }

            cout << "BMU was unit " << bmu_index << endl;
            units[bmu_index].class_stats[sample.type]++;

            double distance_modifier = 1.0 / (2.0 * radius * radius);
            double decayed_learning_rate = learning_rate * exp((double)current_epoch / (double)max_epochs);
            cout << "radius: " << radius << ", distance modifier: " << distance_modifier << ", decayed learning rate: " << decayed_learning_rate << endl;
            for (int i = 0; i < units.size(); i++) {

                double final_modifier = decayed_learning_rate * exp(-euclidian_distance(units[bmu_index].values, units[i].values) * distance_modifier);
                cout << "final modifier for unit " << i << " was: " << final_modifier << endl;

                for (int j = 0; j < sample.values.size(); j++) {
                    units[i].values[j] = units[i].values[j] + final_modifier * (sample.values[j] - units[i].values[j]);
                }
            }
        }

        //so we can easily print the SOM to output
        friend ostream & operator << (ostream &out, const SelfOrganizingMap &som);
        
};

ostream & operator << (ostream &out, const SelfOrganizingMap &som) {
    out << "[SOM:" << endl;
    for (int i = 0; i < som.units.size(); i++) {
        out << som.units[i] << endl;
    }
    out << "]";

    return out;
}


/*
double learning_rate = 0.1; //figure it out
double threshold = 0.1; //figure it out
double **units = new double[number_of_units][length];


void som_update(int length, double *input, int input_class) {
}
*/

void read_iris(string filename, vector<Sample> &samples) {
    ifstream infile(filename);
    string line;

    while (getline(infile, line)) {
        //cout << line << endl;
        int current_class = stof(line.substr(0,1));
        double v1 = stof(line.substr(2, 3));
        double v2 = stof(line.substr(6, 3));
        double v3 = stof(line.substr(10, 3));
        double v4 = stof(line.substr(14, 3));

        //cout << "class: " << current_class << ", v1: " << v1 << ", v2: " << v2 << ", v3: " << v3 << ", v4: " << v4 << endl;

        double *array = new double[4];
        array[0] = v1;
        array[1] = v2;
        array[2] = v3;
        array[3] = v4;
        samples.push_back(Sample(current_class, 4, array));
        delete [] array;
    }

    infile.close();
}


int main(int number_arguments, char** arguments) {
    if (number_arguments != 6) {
        cerr << "ERROR: incorrect usage. try:" << endl;
        cerr << arguments[0] << " <iris filename> <max epochs> <number units> <learning rate> <radius>" << endl;
        exit(1);
    }
    string input_filename(arguments[1]);
    int max_epochs = atoi(arguments[2]);
    int number_units = atoi(arguments[3]);
    double learning_rate = atof(arguments[4]);
    double radius = atof(arguments[5]);

    vector<Sample> samples;
    read_iris(input_filename, samples);

    //to test to make sure we read in the data correctly
    //print_samples(samples);

    normalize(samples);

    //to test to make sure we normalized the data correctly
    //print_samples(samples);

    int n_types = 0;
    for (int i = 0; i < samples.size(); i++) {
        if (samples[i].type > n_types) n_types = samples[i].type;
    }
    n_types++;
    cout << "there are " << n_types << " classes in the dataset." << endl;

    SelfOrganizingMap som(number_units, learning_rate, radius, n_types);
    //make sure we're initializing the SOM decently
    cout << som << endl;

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        random_shuffle(samples.begin(), samples.end());

        for (int i = 0; i < samples.size(); i++) {
            som.insert_sample(epoch, max_epochs, samples[i]);
        }

        cout << som << endl;
        som.reset_class_stats();
    }
}

