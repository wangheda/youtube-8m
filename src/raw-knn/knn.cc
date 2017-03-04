#include<iostream>
#include<fstream>
#include<string>
#include<vector>

// unix api
#include<dirent.h>
#include<unistd.h>

// 3rd party
#include"gzstream.h"

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
		string& test_pattern
		) {
	unsigned i = 0;
	while (i < args.size()) {
		if (args.at(i) == "-z") {
			use_gzip = true;
			i += 1;
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
	DEBUG_VALUE(train_pattern);
	DEBUG_VALUE(validate_pattern);
	DEBUG_VALUE(test_pattern);
	return 0;
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

int main(int argc, char* argv[]) {
	bool use_gzip = false;
	string train_pattern = "";
	string validate_pattern = "";
	string test_pattern = "";

	vector<string> args;
	for (int i = 0; i < argc; i++) {
		args.push_back(argv[i]);
	}
	parse_argument(args, use_gzip, train_pattern, validate_pattern, test_pattern);

	auto train_filenames = get_files(train_pattern);
	DEBUG_VECTOR(train_filenames);
	auto validate_filenames = get_files(validate_pattern);
	DEBUG_VECTOR(validate_filenames);
	auto test_filenames = get_files(test_pattern);
	DEBUG_VECTOR(test_filenames);

	vector<string> buffer;
	if (train_filenames.size()) {
		read_from_file(train_filenames.at(0), buffer, use_gzip);
		DEBUG_VECTOR(buffer);
	}
}
