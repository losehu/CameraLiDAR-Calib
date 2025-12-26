#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "common.h"

using namespace std;
// ------------------- OcamModel 结构 -------------------
struct OcamModel
{
    std::vector<double> pol;
    std::vector<double> invpol;
    double xc, yc;
    double c, d, e;
    int width, height;
    int length_invpol;
    int length_pol;
};
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <climits>

namespace fs = std::filesystem;
void getTheoreticalUV(float *theoryUV, const vector<float> &intrinsic, const vector<float> &extrinsic, double x, double y, double z);
void getUVError_pano(const string &intrinsic_path, const string &extrinsic_path, const string &lidar_path, const string &photo_path, float *error, int threshold);
void getUVError(const string &intrinsic_path, const string &extrinsic_path, const string &lidar_path, const string &photo_path, float *error, int threshold, bool vaild );
void getUVErrorNewIntrinsic(const string &extrinsic_path, const string &lidar_path, const string &photo_path, float *error, int threshold, const vector<float> &intrinsic);
void get_ocam_model(OcamModel &model, const std::string &filename);
std::vector<std::string> list_images_sorted_by_number(const std::string &dir);
void drawCircleAt(cv::Mat &img,
                  int x, int y, int n,
                  const cv::Scalar &color,
                  int thickness, int lineType);
// 在已有图像上画圆
// img: 传入/传出图像
// x, y: 圆心像素坐标（x=列，y=行）
// n: 半径
// color: 颜色(B,G,R)，默认红色
// thickness: 线宽，-1 表示实心圆；默认 2
// lineType: 线型，默认抗锯齿
inline void drawCircleAt(cv::Mat &img,
                         int x, int y, int n,
                         const cv::Scalar &color = cv::Scalar(0, 0, 255),
                         int thickness = 2,
                         int lineType = cv::LINE_AA)
{
    CV_Assert(!img.empty());
    CV_Assert(n > 0);
    cv::circle(img, cv::Point(x, y), n, color, thickness, lineType);
}
void get_ocam_model(OcamModel &model, const std::string &filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // int length;
    // f >> length;
    // m.pol.resize(length);
    // for (int i = 0; i < length; i++) f >> m.pol[i];
    // f >> length;
    // m.invpol.resize(length);
    // for (int i = 0; i < length; i++) f >> m.invpol[i];
    // f >> m.xc >> m.yc >> m.c >> m.d >> m.e >> m.width >> m.height;

    std::string line;

    // Skip 2 lines
    std::getline(file, line);
    std::getline(file, line);

    // Read polynomial coefficients
    std::getline(file, line);
    std::istringstream iss_poly(line);
    iss_poly >> model.length_pol;

    model.pol.resize(model.length_pol);
    for (int i = 0; i < model.length_pol; ++i)
    {
        iss_poly >> model.pol[i];
    }

    // Skip 3 lines
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    // Read inverse polynomial coefficients
    std::getline(file, line);
    std::istringstream iss_invpoly(line);
    iss_invpoly >> model.length_invpol;
    model.invpol.resize(model.length_invpol);
    for (int i = 0; i < model.length_invpol; ++i)
    {
        iss_invpoly >> model.invpol[i];
    }

    // Skip 3 lines
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    // Read center coordinates
    std::getline(file, line);
    std::istringstream iss_center(line);
    iss_center >> model.xc >> model.yc;
    std::cout << "Center: (" << model.xc << ", " << model.yc << ")\n";
    // Skip 3 lines
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    // Read affine parameters
    std::getline(file, line);
    std::istringstream iss_affine(line);
    iss_affine >> model.c >> model.d >> model.e;

    // Skip 3 lines
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    // Read image size
    std::getline(file, line);
    std::istringstream iss_size(line);
    iss_size >> model.height >> model.width;

    file.close();
}

double polyval(const std::vector<double> &coeffs, double x)
{
    double y = 0.0;
    for (size_t i = 0; i < coeffs.size(); ++i)
    {
        y = y * x + coeffs[i];
    }
    return y;
}
cv::Point2d world2cam(const cv::Vec3d &point3D, const OcamModel &model)
{
    double normal = std::sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1] + point3D[2] * point3D[2]);
    cv::Vec3d point3D_copy;
    point3D_copy[0] = point3D[0] / normal; // Normalize the point to unit length
    point3D_copy[1] = point3D[1] / normal; // Normalize the point to unit length
    point3D_copy[2] = point3D[2] / normal; // Normalize the point to unit length

    double u = 0, v = 0;
    double xc = model.xc;
    double yc = model.yc;
    double c = model.c;
    double d = model.d;
    double e = model.e;
    cv::Point2d point2D[2];
    // #3D点在2D平面的投影长度
    double norm = std::sqrt(point3D_copy[0] * point3D_copy[0] + point3D_copy[1] * point3D_copy[1]);
    // #3D点与平面的夹角
    double theta = std::atan(point3D_copy[2] / norm);
    if (norm != 0)
    {
        double invnorm = 1 / norm;
        double t = theta;
        double rho = model.invpol[0];
        rho = polyval(model.invpol, t);

        // #rho是畸变半径
        double x = point3D_copy[0] * invnorm * rho;
        double y = point3D_copy[1] * invnorm * rho;
        u = x * c + y * d + xc;
        v = x * e + y + yc;
    }
    else
    {
        u = xc;
        v = yc;
    }
    u = int(u + 0.5);
    v = int(v + 0.5);
    return cv::Point2d(u, v);
}
void getTheoreticalUV_pano(float *uv, const OcamModel &model, const std::vector<float> &extrinsic, double x, double y, double z)
{

    double matrix2[3][4] = {{extrinsic[0], extrinsic[1], extrinsic[2], extrinsic[3]}, {extrinsic[4], extrinsic[5], extrinsic[6], extrinsic[7]}, {extrinsic[8], extrinsic[9], extrinsic[10], extrinsic[11]}};
    double matrix3[4][1] = {{x}, {y}, {z}, {1}};

    // transform into the opencv matrix
    cv::Mat matrixOut(3, 4, CV_64F, matrix2);
    cv::Mat coordinate(4, 1, CV_64F, matrix3);

    // calculate the result of u and v
    cv::Mat result = matrixOut * coordinate;
    float u = result.at<double>(0, 0);
    float v = result.at<double>(1, 0);
    float depth = result.at<double>(2, 0);

    // Transform LiDAR coordinates (x, y, z) to image coordinates (u, v) using world2cam
    cv::Vec3d point3D(u, v, depth);

    // Convert to camera coordinate`s (this should be customized for panoramic projection)
    cv::Point2d point2D = world2cam(point3D, model);

    // Set the theoretical UV values
    uv[0] = point2D.x;
    uv[1] = point2D.y;
}

void getUVError_pano(const std::string &intrinsic_path, const std::string &extrinsic_path,
                     const std::string &lidar_path, const std::string &photo_path, float *error, int threshold, const std::string &valid_path, bool vaild = false)
{
    std::ifstream inFile_lidar;
    std::ifstream inFile_photo;

    inFile_lidar.open(lidar_path);
    inFile_photo.open(photo_path);
    std::string lineStr_lidar;
    std::string lineStr_photo;

    int count = 0;
    float errorTotalU = 0;
    float errorTotalV = 0;
    float errorU = 0;
    float errorV = 0;

    // Load the camera parameters (intrinsics and extrinsics)
    OcamModel ocam_model;
    get_ocam_model(ocam_model, intrinsic_path);

    std::vector<float> extrinsic;
    getExtrinsic(extrinsic_path, extrinsic);
    auto files = list_images_sorted_by_number(valid_path);
    cv::Mat image;
    while (getline(inFile_lidar, lineStr_lidar) && getline(inFile_photo, lineStr_photo))
    {
        if (lineStr_lidar.size() > 10 && lineStr_photo.size() > 10)
        {
            if (count % 4 == 0 && vaild)
            {
                int now_image_idx = count / 4;
                std::string image_path = files[now_image_idx];
                image = cv::imread(image_path);
                if (image.empty())
                {
                    std::cerr << "Failed to load image: " << image_path << std::endl;
                    continue;
                }
            }
            double x, y, z, dataU, dataV;
            std::string str;
            std::stringstream line_lidar(lineStr_lidar);
            std::stringstream line_photo(lineStr_photo);

            // Read LiDAR data
            line_lidar >> str;
            x = str2double(str);
            line_lidar >> str;
            y = str2double(str);
            line_lidar >> str;
            z = str2double(str);

            // Read image pixel data
            line_photo >> str;
            dataU = str2double(str);
            line_photo >> str;
            dataV = str2double(str);

            // Calculate the theoretical UV from the LiDAR point and the camera parameters
            float theoryUV[2] = {0, 0};
            getTheoreticalUV_pano(theoryUV, ocam_model, extrinsic, x, y, z);
            if (vaild)
            {
                drawCircleAt(image, theoryUV[0], theoryUV[1], 20, cv::Scalar(0, 255, 0), -1); // 绿色实心圆

                if ((count + 1) % 4 == 0) 
                {
                    cv::imshow(int2str(count / 4), image);
                    cv::waitKey(0);
                    // exit(0);
                }
            }
            errorU = abs(dataU - theoryUV[0]);
            errorV = abs(dataV - theoryUV[1]);

            // if (errorU + errorV > threshold) {
            //     std::cout << "Data " << count << " has an error bigger than the threshold" << std::endl;
            //     std::cout << "XYZ: " << x << " " << y << " " << z << std::endl;
            //     std::cout << "ErrorU: " << errorU << " ErrorV: " << errorV << std::endl;
            //     std::cout << "**********************" << std::endl;
            // }

            errorTotalU += errorU;
            errorTotalV += errorV;
            ++count;
        }
        else if (lineStr_lidar.size() < 1 && lineStr_photo.size() < 1)
        {
            //             auto files = list_images_sorted_by_number("/Users/losehu/Documents/20250813");
            // for (auto& f : files) std::cout << f << "\n";
            // exit(0);
            break;
        }
        else if ((lineStr_lidar.size() < 10 && lineStr_photo.size() > 10) || (lineStr_lidar.size() > 10 && lineStr_photo.size() < 10))
        {
            std::cout << "LiDAR data and photo data not aligned!" << std::endl;
            exit(1);
        }
    }

    inFile_lidar.close();
    inFile_photo.close();

    error[0] = errorTotalU / count;
    error[1] = errorTotalV / count;
}

// 小工具：把扩展名转小写
static inline std::string lower_ext(const fs::path &p)
{
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    return e;
}

// 从文件名（不含扩展名）提取数字键；非数字返回 INT_MAX 以排在最后
static inline int numeric_key(const fs::path &p)
{
    const std::string stem = p.stem().string();
    try
    {
        size_t pos = 0;
        int val = std::stoi(stem, &pos);
        // 要求整个 stem 都是数字；否则按非数字处理
        if (pos == stem.size())
            return val;
    }
    catch (...)
    {
    }
    return INT_MAX;
}

// 列出并按数字文件名排序返回 .jpg/.jpeg 图片的绝对路径
std::vector<std::string> list_images_sorted_by_number(const std::string &dir)
{
    std::vector<fs::path> imgs;
    for (const auto &entry : fs::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
            continue;
        const fs::path &p = entry.path();
        std::string ext = lower_ext(p);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff"||  ext == ".JPG" || ext == ".JPEG" || ext == ".PNG" || ext == ".BMP" || ext == ".TIFF")
        {
            imgs.push_back(fs::absolute(p));
        }
    }

    std::sort(imgs.begin(), imgs.end(), [](const fs::path &a, const fs::path &b)
              {
        int ka = numeric_key(a);
        int kb = numeric_key(b);
        if (ka != kb) return ka < kb;
        // 数字键相同则回退到文件名字典序以获得稳定顺序
        return a.filename().string() < b.filename().string(); });

    std::vector<std::string> out;
    out.reserve(imgs.size());
    for (auto &p : imgs)
        out.push_back(p.string());
    return out;
}


// read mesured value and use theoretical U,V calculated to get the total error
void getUVError(const string &intrinsic_path, const string &extrinsic_path, const string &lidar_path, const string &photo_path, float* error, int threshold, const std::string &valid_path, bool vaild ) {
    ifstream inFile_lidar;
    ifstream inFile_photo;

    inFile_lidar.open(lidar_path);
    inFile_photo.open(photo_path);
    string lineStr_lidar;
    string lineStr_photo;

    int count = 0;
    float errorTotalU = 0;
    float errorTotalV = 0;
    float errorU = 0;
    float errorV = 0;

    vector<float> intrinsic;
    getIntrinsic(intrinsic_path, intrinsic);
    vector<float> extrinsic;
    getExtrinsic(extrinsic_path, extrinsic);
    cv::Mat image;
    auto files = list_images_sorted_by_number(valid_path);

    while(getline(inFile_lidar, lineStr_lidar) && getline(inFile_photo, lineStr_photo)) {
        if (lineStr_lidar.size() > 10 && lineStr_photo.size() > 10) {

             if (count % 4 == 0 && vaild)
            {
                int now_image_idx = count / 4;
                std::string image_path = files[now_image_idx];
                image = cv::imread(image_path);
                if (image.empty())
                {
                    std::cerr << "Failed to load image: " << image_path << std::endl;
                    continue;
                }
            }
            double x, y, z, dataU, dataV;
            string str;
            stringstream line_lidar(lineStr_lidar);
            stringstream line_photo(lineStr_photo);

            line_lidar >> str;
            x = str2double(str);

            line_lidar >> str;
            y = str2double(str);

            line_lidar >> str;
            z = str2double(str);

            line_photo >> str;
            dataU = str2double(str);

            line_photo >> str;
            dataV = str2double(str);

            float theoryUV[2] = {0, 0};
            getTheoreticalUV(theoryUV, intrinsic, extrinsic, x, y, z);

            errorU = abs(dataU - theoryUV[0]);
            errorV = abs(dataV - theoryUV[1]);
            if (errorU + errorV > threshold) {
                // cout << "Data " << count << " has a error bigger than the threshold" << endl;
                // cout << "xyz are " << x << " " << y << " " << z << endl;
                // cout << "ErrorU is " << errorU << " errorV is " << errorV << endl;
                // cout << "**********************" << endl;
            }
              if (vaild)
            {
                drawCircleAt(image, theoryUV[0], theoryUV[1], 20, cv::Scalar(0, 255, 0), -1); // 绿色实心圆
                if((count)%4==0)
                {              
                      drawCircleAt(image, dataU, dataV, 20, cv::Scalar(0, 0, 255), -1); // 绿色实心圆


                }else if((count)%4==1)
                {
                      drawCircleAt(image, dataU, dataV, 20, cv::Scalar(0, 255, 0), -1); // 绿色实心圆

                }else if((count)%4==2)
                {
                      drawCircleAt(image, dataU, dataV, 20, cv::Scalar(255, 0, 0), -1); // 绿色实心圆

                }else
                {
                      drawCircleAt(image, dataU, dataV, 20, cv::Scalar(255, 255, 0), -1); // 绿色实心圆

                }
                if ((count + 1) % 4 == 0)
                {
                    cv::imshow("show", image);
                    cv::waitKey(0);
                }
            }
            errorTotalU += errorU;
            errorTotalV += errorV;
            ++count;
        }
        else if(lineStr_lidar.size() < 1 && lineStr_photo.size() < 1) {  // stop reading the data when there is an empty line
            break;
        }
        else if ((lineStr_lidar.size() < 10 && lineStr_photo.size() > 10) || (lineStr_lidar.size() > 10 && lineStr_photo.size() < 10)) {
            cout << "Lidar data and photo data not aligned!" << endl;
            exit(1);
        }
    }
    inFile_lidar.close();
    inFile_photo.close();

    error[0] = errorTotalU/count;
    error[1] = errorTotalV/count;
}

void getUVErrorNewIntrinsic(const string &extrinsic_path, const string &lidar_path, const string &photo_path, float* error, int threshold, const vector<float> &intrinsic) {
    ifstream inFile_lidar;
    ifstream inFile_photo;

    inFile_lidar.open(lidar_path);
    inFile_photo.open(photo_path);
    string lineStr_lidar;
    string lineStr_photo;

    int count = 0;
    float errorTotalU = 0;
    float errorTotalV = 0;
    float errorU = 0;
    float errorV = 0;

    vector<float> extrinsic;
    getExtrinsic(extrinsic_path, extrinsic);

    while(getline(inFile_lidar, lineStr_lidar) && getline(inFile_photo, lineStr_photo)) {
        if (lineStr_lidar.size() > 10 && lineStr_photo.size() > 10) {  // ignore the index
            double x, y, z, dataU, dataV;
            string str;
            stringstream line_lidar(lineStr_lidar);
            stringstream line_photo(lineStr_photo);

            line_lidar >> str;
            x = str2double(str);

            line_lidar >> str;
            y = str2double(str);

            line_lidar >> str;
            z = str2double(str);

            line_photo >> str;
            dataU = str2double(str);

            line_photo >> str;
            dataV = str2double(str);

            float theoryUV[2] = {0, 0};
            getTheoreticalUV(theoryUV, intrinsic, extrinsic, x, y, z);

            errorU = abs(dataU - theoryUV[0]);
            errorV = abs(dataV - theoryUV[1]);
            if (errorU + errorV > threshold) {
                cout << "Data " << count << " has a error bigger than the threshold" << endl;
                cout << "xyz are " << x << " " << y << " " << z << endl;
                cout << "errorU is " << errorU << " errorV is " << errorV << endl;
                cout << "**********************" << endl;
            }
            errorTotalU += errorU;
            errorTotalV += errorV;
            ++count;
        }
        else if(lineStr_lidar.size() < 1 && lineStr_photo.size() < 1) {  // stop reading the data when there is an empty line
            break;
        }
        else if ((lineStr_lidar.size() < 10 && lineStr_photo.size() > 10) || (lineStr_lidar.size() > 10 && lineStr_photo.size() < 10)) {
            cout << "Lidar data and photo data not aligned!" << endl;
            exit(1);
        }
    }
    inFile_lidar.close();
    inFile_photo.close();

    error[0] = errorTotalU/count;
    error[1] = errorTotalV/count;
}

// calculate theoretical U and V from x,y,z
void getTheoreticalUV(float* theoryUV, const vector<float> &intrinsic, const vector<float> &extrinsic, double x, double y, double z) {
    // set the intrinsic and extrinsic matrix
    double matrix1[3][3] = {{intrinsic[0], intrinsic[1], intrinsic[2]}, {intrinsic[3], intrinsic[4], intrinsic[5]}, {intrinsic[6], intrinsic[7], intrinsic[8]}}; 
    double matrix2[3][4] = {{extrinsic[0], extrinsic[1], extrinsic[2], extrinsic[3]}, {extrinsic[4], extrinsic[5], extrinsic[6], extrinsic[7]}, {extrinsic[8], extrinsic[9], extrinsic[10], extrinsic[11]}};
    double matrix3[4][1] = {{x},{y},{z},{1}};
    
    // transform into the opencv matrix
    cv::Mat matrixIn(3, 3, CV_64F, matrix1);
    cv::Mat matrixOut(3, 4, CV_64F, matrix2);
    cv::Mat coordinate(4, 1, CV_64F, matrix3);
    
    // calculate the result of u and v
    cv::Mat result = matrixIn*matrixOut*coordinate;
    float u = result.at<double>(0, 0);
    float v = result.at<double>(1, 0);
    float depth = result.at<double>(2, 0);

    theoryUV[0] = u / depth;
    theoryUV[1] = v / depth;
}
