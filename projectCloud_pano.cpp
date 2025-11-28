#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem> // C++17
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "common.h"
#include "result_verify.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;


cv::Mat src_img;

int threshold_lidar = 30000;  
string input_pcd_dir, input_photo_path, output_path, intrinsic_path, extrinsic_path;

struct PointXYZRGB {
    float x, y, z;
};
// HSV -> RGB 转换
void HSVtoRGB(float h, float s, float v, int &r, int &g, int &b) {
    float c = v * s;
    float x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
    float m = v - c;
    float r1, g1, b1;
    if (h < 60)      { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120){ r1 = x; g1 = c; b1 = 0; }
    else if (h < 180){ r1 = 0; g1 = c; b1 = x; }
    else if (h < 240){ r1 = 0; g1 = x; b1 = c; }
    else if (h < 300){ r1 = x; g1 = 0; b1 = c; }
    else             { r1 = c; g1 = 0; b1 = x; }
    r = int((r1 + m) * 255);
    g = int((g1 + m) * 255);
    b = int((b1 + m) * 255);
}

#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cctype>


namespace fs = std::filesystem;

// 大小写无关地判断扩展名是否是 .pcd
static inline bool is_pcd_file(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return ext == ".pcd";
}

// 载入单个 .pcd 文件并压入 points，打印点数
static bool load_one_pcd(const fs::path& path, std::vector<PointXYZRGB>& points) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path.string(), cloud) == -1) {
        std::cerr << "Failed to load " << path << "\n";
        return false;
    }
    const std::size_t old_sz = points.size();
    points.reserve(points.size() + cloud.size());
    for (const auto& p : cloud.points) {
        points.push_back({p.x, p.y, p.z});  // 如果你的 PointXYZRGB 有 rgb 字段，可自行补默认值
    }
    std::cout << "Loaded " << path.filename().string()
              << " points=" << cloud.size()
              << " (total=" << points.size() << ")\n";
    return true;
}

// 既支持目录也支持单个 .pcd
void loadPointcloudFromPath(const std::string& path_str, std::vector<PointXYZRGB>& points) {
    fs::path p(path_str);
    if (!fs::exists(p)) {
        std::cerr << "Path not exists: " << p << "\n";
        return;
    }

    if (fs::is_regular_file(p)) {
        if (is_pcd_file(p)) {
            load_one_pcd(p, points);
        } else {
            std::cerr << "Not a .pcd file: " << p << "\n";
        }
        return;
    }

    if (fs::is_directory(p)) {
        // 如需排序，可先收集后排序
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(p)) {
            if (entry.is_regular_file() && is_pcd_file(entry.path())) {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end()); // 按文件名字典序；如需按数字名可改比较器

        for (const auto& f : files) {
            load_one_pcd(f, points);
        }
        std::cout << "Done. total points=" << points.size() << "\n";
        return;
    }

    std::cerr << "Unsupported path type: " << p << "\n";
}


int main(int argc, char **argv) {

    input_pcd_dir = "/Users/losehu/Documents/20250813/pcd/pcd14.pcd"; // 这里改成你的文件夹路径
    input_photo_path = "/Users/losehu/Documents/20250813/14.jpg";

    output_path = "./result.png";
    intrinsic_path = "../a.txt";
    extrinsic_path = "./extrinsic_pano.txt";
    threshold_lidar = 30000;

    src_img = cv::imread(input_photo_path);
    if (src_img.empty()) {
        cout << "No Picture found by filename: " << input_photo_path << endl;
        return 0;
    }

    vector<PointXYZRGB> pointcloud;
    loadPointcloudFromPath(input_pcd_dir, pointcloud);

    OcamModel ocam_model;


    get_ocam_model(ocam_model, intrinsic_path); // OcamModel 参数文件

    vector<float> extrinsic;
    getExtrinsic(extrinsic_path, extrinsic);




    cout << "Start to project the lidar cloud..." << endl;
    float theoryUV[2] = {0, 0};
    int myCount = 0;

    float min_depth_frame = std::numeric_limits<float>::max();
float max_depth_frame = std::numeric_limits<float>::lowest();

// 第一次遍历：统计深度范围
for (auto& pt : pointcloud) {
    float depth = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    if (depth < min_depth_frame) min_depth_frame = depth;
    if (depth > max_depth_frame) max_depth_frame = depth;
}
for (auto& pt : pointcloud) {
    // 1150 2330 939

    float x = pt.x, y = pt.y, z = pt.z;

// if(z>1.2)continue;
// std::cout<<x*1000<<" "<<y*1000<<" "<<z*1000<<std::endl;
// if (x*1000==1150&&y*1000==2330&&z*1000==939)
// {
//     exit(0);
// }
    // 计算深度（米）
    float depth = sqrt(x * x + y * y + z * z);
    float t = (depth - min_depth_frame) / (max_depth_frame - min_depth_frame);

    // 投影到图像坐标
    getTheoreticalUV_pano(theoryUV, ocam_model, extrinsic, x*1000, y*1000, z*1000);

    int u = floor(theoryUV[0] + 0.5);
    int v = floor(theoryUV[1] + 0.5);

    // 超出图像范围就跳过
    if (u < 0 || u >= src_img.cols || v < 0 || v >= src_img.rows) continue;

 t = std::clamp(t, 0.0f, 1.0f);

    // hue 从 0°(红) 到 270°(紫)
    float hue = (1.0f - t) * 270.0f;
    int r, g, b;
    HSVtoRGB(hue, 1.0f, 1.0f, r, g, b);

    cv::circle(src_img, Point(u, v), 5, Scalar(b, g, r), -1);

    ++myCount;
    if (myCount > threshold_lidar) break;
}


    // cv::Size imageSize = src_img.size();
    // cv::Mat map1, map2;
    // cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
    //     cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
    //     imageSize, CV_16SC2, map1, map2);
    // cv::remap(src_img, src_img, map1, map2, cv::INTER_LINEAR);

    cv::imshow("source", src_img);
    cv::waitKey(0);
    cv::imwrite(output_path, src_img);

    return 0;
}
