#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "common.h"
#include "result_verify.h"

using namespace std;
namespace fs = std::filesystem;

struct PointXYZ {
    float x, y, z;
};

// Load points from a .pcd file
bool loadPcd(const string &path, vector<PointXYZ> &out) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == -1) {
        cerr << "Failed to load PCD: " << path << endl;
        return false;
    }
    out.reserve(out.size() + cloud->points.size());
    for (auto &p : cloud->points) out.push_back({p.x, p.y, p.z});
    return true;
}

// Load points from a simple text file: each line with "x y z" (ignores lines without three numbers)
bool loadTxt(const string &path, vector<PointXYZ> &out) {
    ifstream in(path);
    if (!in.is_open()) return false;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        double x, y, z;
        if (ss >> x >> y >> z) {
            out.push_back({(float)x, (float)y, (float)z});
        }
    }
    return true;
}

bool loadPoints(const string &path, vector<PointXYZ> &out) {
    string ext = fs::path(path).extension().string();
    if (ext == ".pcd") return loadPcd(path, out);
    return loadTxt(path, out);
}

// helper: project a single lidar file onto a single image and save result
static bool projectAndSave(const string &lidar_file, const string &photo_file, const vector<float> &intrinsic, const vector<float> &extrinsic, const vector<float> &distortion, const string &out_file) {
    vector<PointXYZ> points;
    if (!loadPoints(lidar_file, points)) {
        cerr << "Failed to load lidar " << lidar_file << endl;
        return false;
    }
    cv::Mat img = cv::imread(photo_file);
    if (img.empty()) {
        cerr << "Cannot open image: " << photo_file << endl;
        return false;
    }

    // compute camera-axis depth range (only points in front of the camera)
    float min_cam_d = numeric_limits<float>::max();
    float max_cam_d = numeric_limits<float>::lowest();
    vector<double> cam_z_list; cam_z_list.reserve(points.size());
    for (auto &p : points) {
        double x_mm = p.x * 1000.0;
        double y_mm = p.y * 1000.0;
        double z_mm = p.z * 1000.0;
        // extrinsic is 3x4 row-major: extrinsic[8..11] are third row
        double cam_z = extrinsic[8] * x_mm + extrinsic[9] * y_mm + extrinsic[10] * z_mm + extrinsic[11];
        if (cam_z > 1e-6) { // only consider points in front
            double cam_d_m = cam_z / 1000.0; // meters
            cam_z_list.push_back(cam_d_m);
            if (cam_d_m < min_cam_d) min_cam_d = (float)cam_d_m;
            if (cam_d_m > max_cam_d) max_cam_d = (float)cam_d_m;
        }
    }
    if (cam_z_list.empty()) {
        cerr << "No points in front of camera; nothing to project." << endl;
        return false;
    }
    if (min_cam_d == max_cam_d) { min_cam_d = 0; max_cam_d = min_cam_d + 1e-3f; }

    int count = 0;
    for (auto &p : points) {
        float x = p.x, y = p.y, z = p.z;
        double x_mm = x * 1000.0;
        double y_mm = y * 1000.0;
        double z_mm = z * 1000.0;
        double cam_z = extrinsic[8] * x_mm + extrinsic[9] * y_mm + extrinsic[10] * z_mm + extrinsic[11];
        if (cam_z <= 1e-6) continue; // behind camera or at plane

        float uv[2] = {0,0};
        getTheoreticalUV(uv, intrinsic, extrinsic, x_mm, y_mm, z_mm);
        int u = static_cast<int>(floor(uv[0] + 0.5));
        int v = static_cast<int>(floor(uv[1] + 0.5));
        if (u < 0 || u >= img.cols || v < 0 || v >= img.rows) continue;

        // color by camera-axis depth
        float cam_d_m = static_cast<float>(cam_z / 1000.0);
        float t = (cam_d_m - min_cam_d) / (max_cam_d - min_cam_d);
        t = std::clamp(t, 0.0f, 1.0f);
        float hue = (1.0f - t) * 270.0f;
        cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(int(hue/2), 255, 255));
        cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Vec3b color = bgr.at<cv::Vec3b>(0,0);
        cv::circle(img, cv::Point(u,v), 2, cv::Scalar(color[0], color[1], color[2]), -1, cv::LINE_AA);
        ++count;
    }

    // save the projection on the original (no undistortion)
    try {
        if (!cv::imwrite(out_file, img)) {
            cerr << "cv::imwrite failed: returned false for " << out_file << endl;
            return false;
        }
    } catch (const cv::Exception &e) {
        cerr << "cv::imwrite threw exception: " << e.what() << endl;
        return false;
    }

    cout << "Projected " << count << " points: " << lidar_file << " -> " << photo_file << " -> " << out_file << endl;
    return true;
}

int main(int argc, char **argv) {
    // Usage: main1 <lidar_path_or_dir> <photo_path_or_dir> <intrinsic_path> <extrinsic_path> [output_path_or_dir]
    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <lidar_path_or_dir> <photo_path_or_dir> <intrinsic_path> <extrinsic_path> [output_path_or_dir]" << endl;
        return 0;
    }

    string lidar_path = argv[1];
    string photo_path = argv[2];
    string intrinsic_path = argv[3];
    string extrinsic_path = argv[4];
    string output_path = (argc >= 6) ? argv[5] : "./";

    // load intrinsics/extrinsics once
    vector<float> intrinsic; getIntrinsic(intrinsic_path, intrinsic);
    vector<float> extrinsic; getExtrinsic(extrinsic_path, extrinsic);
    vector<float> distortion; getDistortion(intrinsic_path, distortion);

    fs::path lidar_p(lidar_path);
    fs::path photo_p(photo_path);
    fs::path outp(output_path);

    // ensure output directory exists
    if (fs::exists(outp) && fs::is_regular_file(outp)) {
        // if a file was provided, use its parent as directory
        outp = outp.parent_path();
    }
    if (!fs::exists(outp)) {
        fs::create_directories(outp);
    }

    vector<fs::path> lidar_files;
    vector<fs::path> photo_files;

    auto is_lidar = [](const fs::path &p){ string e = p.extension().string(); return e==".pcd" || e==".txt" || e==".xyz" || e==".ply"; };
    auto is_image = [](const fs::path &p){ string e = p.extension().string(); if (e.size()>0 && e[0]=='.') e = e.substr(1); // remove dot
        // to lower
        for (auto &c : e) c = (char)tolower(c);
        for (auto &s: {"jpg","jpeg","png","bmp","tif","tiff"}) if (e==s) return true; return false; };

    if (fs::exists(lidar_p) && fs::is_directory(lidar_p)) {
        for (auto &it: fs::directory_iterator(lidar_p)) if (fs::is_regular_file(it.path()) && is_lidar(it.path())) lidar_files.push_back(it.path());
        sort(lidar_files.begin(), lidar_files.end());
    } else if (fs::exists(lidar_p) && fs::is_regular_file(lidar_p)) {
        lidar_files.push_back(lidar_p);
    } else {
        cerr << "Lidar path not found: " << lidar_path << endl; return 1;
    }

    if (fs::exists(photo_p) && fs::is_directory(photo_p)) {
        for (auto &it: fs::directory_iterator(photo_p)) if (fs::is_regular_file(it.path()) && is_image(it.path())) photo_files.push_back(it.path());
        sort(photo_files.begin(), photo_files.end());
    } else if (fs::exists(photo_p) && fs::is_regular_file(photo_p)) {
        photo_files.push_back(photo_p);
    } else {
        cerr << "Photo path not found: " << photo_path << endl; return 1;
    }

    // debug prints
    cerr << "Found " << lidar_files.size() << " lidar files and " << photo_files.size() << " photo files" << endl;
    for (size_t i=0;i<min<size_t>(photo_files.size(),5);++i) cerr << " photo["<<i<<"]="<<photo_files[i]<<"\n";

    // Pair files
    vector<pair<fs::path, fs::path>> pairs;
    if (lidar_files.size() == photo_files.size()) {
        for (size_t i=0;i<lidar_files.size();++i) pairs.emplace_back(lidar_files[i], photo_files[i]);
    } else {
        // try basename matching
        unordered_map<string, fs::path> photo_map;
        for (auto &p: photo_files) photo_map[p.stem().string()] = p;
        for (auto &lp: lidar_files) {
            string stem = lp.stem().string();
            if (photo_map.count(stem)) pairs.emplace_back(lp, photo_map[stem]);
            else {
                // try substring match
                bool found=false;
                for (auto &pf: photo_files) {
                    if (pf.stem().string().find(stem) != string::npos || stem.find(pf.stem().string()) != string::npos) {
                        pairs.emplace_back(lp, pf); found=true; break;
                    }
                }
                if (!found) cerr << "Warning: no matching photo for lidar " << lp << endl;
            }
        }
    }

    // fallback: if no pairs found, try index pairing up to min size
    if (pairs.empty() && !lidar_files.empty() && !photo_files.empty()) {
        size_t m = std::min(lidar_files.size(), photo_files.size());
        cerr << "Note: counts differ, fallback to index pairing first " << m << " files" << endl;
        for (size_t i = 0; i < m; ++i) pairs.emplace_back(lidar_files[i], photo_files[i]);
    }

    if (pairs.empty()) { cerr << "No pairs to process" << endl; return 1; }

    // process pairs
    for (auto &pr: pairs) {
        fs::path lidar_f = pr.first;
        fs::path photo_f = pr.second;
        string out_file = (outp / (photo_f.stem().string() + string("_proj.png"))).string();
        projectAndSave(lidar_f.string(), photo_f.string(), intrinsic, extrinsic, distortion, out_file);
    }

    cout << "All done." << endl;
    return 0;
}


// ./main1 "/Users/losehu/CameraLiDAR-Calib/example/hongwai/lidar" "/Users/losehu/CameraLiDAR-Calib/example/hongwai/photo" "/Users/losehu/CameraLiDAR-Calib/sign/int_hongwai.txt" "/Users/losehu/CameraLiDAR-Calib/result/extrinsic_hongwai.txt" ./proj_out