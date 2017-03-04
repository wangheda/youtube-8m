#include<iostream>
#include<fstream>
#include<string>
#include<unordered_map>
#include<sstream>
#include<chrono>
#include<cstring>
#include<vector>

// unix api
#include<dirent.h>
#include<unistd.h>

// 3rd party
#include"gzstream.h"
#include"annoylib.h"

#define DEBUG_VECTOR(vec) {for (unsigned i=0; i<5 && i<vec.size(); i++) { cerr << "DEBUG: "#vec << " has " << vec.at(i) << endl; } cerr << "DEBUG: "#vec << " has " << vec.size() << " items in total" << endl;}
#define DEBUG_VALUE(s) cerr << "DEBUG: "#s << " = " << s << endl;

using namespace std;

vector<string> get_files(const string& file_pattern) {
	vector<string> filenames;
	if (file_pattern != "") {
		string dir_name;
		string base_name;
		size_t star_pos;
		size_t slash_pos;
		if ((slash_pos = file_pattern.find_last_of('/')) != string::npos) {
			dir_name = file_pattern.substr(0, slash_pos);
			base_name = file_pattern.substr(slash_pos + 1);
		} else {
			dir_name = ".";
			base_name = file_pattern;
		}
		if ((star_pos = base_name.find('*')) != string::npos) {
			string prefix = base_name.substr(0, star_pos);
			string postfix = base_name.substr(star_pos + 1);
			unsigned prefix_len = prefix.length();
			unsigned postfix_len = postfix.length();

			DIR* dirp = opendir(dir_name.c_str());
			struct dirent* dp;
			while ((dp = readdir(dirp)) != 0) {
				string file_name(dp->d_name);
				if (file_name.length() >= prefix_len + postfix_len) {
					string fn_prefix = file_name.substr(0, prefix_len);
					string fn_postfix = file_name.substr(file_name.length() - postfix_len);
					if (fn_prefix == prefix && fn_postfix == postfix) {
						string filepath = dir_name + "/" + file_name;
						filenames.push_back(filepath);
					}
				}
			}
			closedir(dirp);
		} else {
			filenames.push_back(file_pattern);
		}
	}
	return filenames;
}

int parse_argument(
		const vector<string>& args,
		bool& use_gzip,
		string& train_pattern,
		string& validate_pattern,
		string& test_pattern,
		int& feature_size,
		int& tree_num,
		string& tree_file,
		string& task
		) {
	unsigned i = 0;
	while (i < args.size()) {
		if (args.at(i) == "-z") {
			use_gzip = true;
			i += 1;
		} else if (args.at(i) == "--help") {
			cerr << "Usage: knn_tool " << endl
				 << "       [-z] " << endl
				 << "       --task [train]/validate/test]" << endl
				 << "       --train train_filepattern" << endl
				 << "       --validate validate_filepattern" << endl
				 << "       --test test_filepattern" << endl
				 << "       --feature_size [1024]" << endl
				 << "       --tree_file [tmp.tree]" << endl
				 << "       --tree_num 20" << endl;
			return 0;
		} else if (args.at(i) == "--task") {
			if (i + 1 < args.size()) 
				task = args.at(i + 1);
			i += 2;
		} else if (args.at(i) == "--tree_num") {
			if (i + 1 < args.size()) 
				tree_num = stoi(args.at(i + 1));
			i += 2;
		} else if (args.at(i) == "--tree_file") {
			if (i + 1 < args.size()) 
				tree_file = args.at(i + 1);
			i += 2;
		} else if (args.at(i) == "--feature_size") {
			if (i + 1 < args.size()) 
				feature_size = stoi(args.at(i + 1));
			i += 2;
		} else if (args.at(i) == "--train") {
			if (i + 1 < args.size()) 
				train_pattern = args.at(i + 1);
			i += 2;
		} else if (args.at(i) == "--validate") {
			if (i + 1 < args.size()) 
				validate_pattern = args.at(i + 1);
			i += 2;
		} else if (args.at(i) == "--test") {
			if (i + 1 < args.size()) 
				test_pattern = args.at(i + 1);
			i += 2;
		} else {
			i++;
		}
	}
	DEBUG_VALUE(use_gzip);
	DEBUG_VALUE(feature_size);
	DEBUG_VALUE(train_pattern);
	DEBUG_VALUE(validate_pattern);
	DEBUG_VALUE(test_pattern);
	return 1;
}

int read_from_file(const string& filename, vector<string>& buffer, const bool use_gzip) {
	if (use_gzip) {
		igzstream in;
		in.open(filename.c_str());
		if (in.good()) {
			string tmp;
			while (getline(in, tmp)) {
				buffer.push_back(tmp);
			}
		} else return 1;
	} else {
		fstream in;
		in.open(filename.c_str());
		if (in.good()) {
			string tmp;
			while (getline(in, tmp)) {
				buffer.push_back(tmp);
			}
		} else return 1;
	}
	return 0;
}

class Sample {
protected:
	vector<int> labels;
	float *features;
	int feature_size;
public:
	Sample(const string& string_sample, const int feature_size) {
		this->features = NULL;
		this->feature_size = feature_size;
		stringstream ss(string_sample);
		int l_size;
		int f_size;
		ss >> l_size;
		for (int i = 0; i < l_size; i++) {
			int l; 
			ss >> l;
			labels.push_back(l);
		}
		ss >> f_size;
		if (f_size == feature_size && feature_size > 0) {
			features = new float[feature_size];
			for (int i = 0; i < f_size; i++) {
				float f;
				ss >> f;
				features[i] = f;
			}
		}
	}

	Sample(const Sample& sample) {
		this->labels = sample.labels;
		this->feature_size = sample.feature_size;
		if (sample.features != NULL && sample.feature_size > 0) {
			this->features = new float[feature_size];
			memcpy(this->features, sample.features, feature_size * sizeof(float));
		}
	}

	const float* get_features() const {
		return this->features;
	}

	const vector<int>& get_labels() const {
		return this->labels;
	}

	~Sample() {
		delete[] features;
	}
};

vector<Sample*> read_samples_from_files(const vector<string>& filenames, const int feature_size, const bool use_gzip) {
	vector<string> buffer;
	vector<Sample*> batch;
	for (auto fn: filenames) {
		buffer.clear();
		read_from_file(fn, buffer, use_gzip);
		for (auto s: buffer) {
			Sample* item = new Sample(s, feature_size);
			batch.push_back(item);
		}
	}
	return batch;
}

void save_sample_labels(const vector<Sample*>& samples, const string label_file) {
	fstream fs(label_file, ios::out);
	for (unsigned i = 0; i < samples.size(); i++) {
		auto l_list = samples.at(i)->get_labels();
		fs << i << " " << l_list.size();
		for (auto l: l_list) {
			fs << " " << l;
		}
		fs << endl;
	}
	fs.close();
}

void load_sample_labels(unordered_map<int, vector<int>>& labels, const string label_file) {
	labels.clear();
	fstream fs(label_file, ios::in);
	string line;
	while (getline(fs, line)) {
		stringstream ss(line);
		int key;
		int count;
		vector<int> label;
		ss >> key >> count;
		for (int i = 0; i < count; i++) {
			int l;
			ss >> l;
			label.push_back(l);
		}
		labels[key] = label;
	}
}

#define AINDEX AnnoyIndex<int, float, Euclidean, RandRandom>

AINDEX* build_trees(const vector<Sample*>& samples, const int feature_size, const int tree_num, const string& tree_file) {
	AINDEX* index = new AINDEX(feature_size);
	string build_msg = "Loading objects ...";
	DEBUG_VALUE(build_msg);
	for (unsigned i = 0; i < samples.size(); i++) {
		index->add_item(i, samples.at(i)->get_features());
		if (i % 10000 == 999) {
			float loaded_percent = (float) (i + 1) / samples.size();
			DEBUG_VALUE(loaded_percent);
		}
	}
	build_msg = "Building trees ... be patient ...";
	DEBUG_VALUE(build_msg);
	auto t_start = std::chrono::high_resolution_clock::now();
	index->build(tree_num);
	auto t_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t_end - t_start ).count();
	std::cerr << "Time elapsed "<< duration << " secs." << std::endl;
	build_msg = "Saving trees ... be patient ...";
	DEBUG_VALUE(build_msg);
	index->save(tree_file.c_str());
	save_sample_labels(samples, tree_file + ".label");
	build_msg = "Done";
	DEBUG_VALUE(build_msg);
	return index;
}

AINDEX* load_trees(const int feature_size, const string& tree_file) {
	AINDEX* index = new AINDEX(feature_size);
	index->load(tree_file.c_str());
	return index;
}

int main(int argc, char* argv[]) {
	bool use_gzip = false;
	string train_pattern = "";
	string validate_pattern = "";
	string test_pattern = "";
	int feature_size = 1024;
	int tree_num = 20;
	string tree_file = "tmp.tree";
	string task = "train";

	vector<string> args;
	for (int i = 0; i < argc; i++) {
		args.push_back(argv[i]);
	}
	if (parse_argument(args, use_gzip, train_pattern, validate_pattern, test_pattern, feature_size, tree_num, tree_file, task)) {
		if (task == "train") {
			if (train_pattern != "") {
				auto train_filenames = get_files(train_pattern);
				DEBUG_VECTOR(train_filenames);
				vector<Sample*> train_samples = read_samples_from_files(train_filenames, feature_size, use_gzip);
				AINDEX* index = build_trees(train_samples, feature_size, tree_num, tree_file);
			} else {
				cerr << "train_pattern needed" << endl;
			}
		} else if (task == "validate" && validate_pattern != "") {
			if (validate_pattern != "") {
				auto validate_filenames = get_files(validate_pattern);
				DEBUG_VECTOR(validate_filenames);
				AINDEX* index = load_trees(feature_size, tree_file);
				unordered_map<int, vector<int>> labels;
				load_sample_labels(labels, tree_file + ".label");
				// compute GAP
			} else {
				cerr << "validate_pattern needed" << endl;
			}
		} else if (task == "test" && test_pattern != "") {
			if (test_pattern != "") {
				auto test_filenames = get_files(test_pattern);
				DEBUG_VECTOR(test_filenames);
				AINDEX* index = load_trees(feature_size, tree_file);
				unordered_map<int, vector<int>> labels;
				load_sample_labels(labels, tree_file + ".label");
				// get predictions
			} else {
				cerr << "test_pattern needed" << endl;
			}
		}
	}
}
