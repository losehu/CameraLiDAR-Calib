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
#include <algorithm>
#include <cctype>
#include <limits>   // for std::numeric_limits

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

// ====== 参数：调节步长 ======
static float kDegStep = 1.0f;       // 每次角度改变量（度）
static float kTransStepMM = 100.0f; // 每次平移改变量（毫米）

// HSV -> RGB（用于深度着色）
static inline void HSVtoRGB(float h, float s, float v, int &r, int &g, int &b)
{
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

// ============ 文件/点云加载 ============
static inline bool is_pcd_file(const fs::path &p)
{
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return ext == ".pcd";
}

static bool load_one_pcd(const fs::path &path, std::vector<PointXYZRGB> &points)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path.string(), cloud) == -1)
    {
        std::cerr << "Failed to load " << path << "\n";
        return false;
    }
    points.reserve(points.size() + cloud.size());
    for (const auto &p : cloud.points)
        points.push_back({p.x, p.y, p.z});
    std::cout << "Loaded " << path.filename().string()
              << " points=" << cloud.size()
              << " (total=" << points.size() << ")\n";
    return true;
}

static void loadPointcloudFromPath(const std::string &path_str, std::vector<PointXYZRGB> &points)
{
    fs::path p(path_str);
    if (!fs::exists(p))
    {
        std::cerr << "Path not exists: " << p << "\n";
        return;
    }
    if (fs::is_regular_file(p))
    {
        if (is_pcd_file(p))
            load_one_pcd(p, points);
        else
            std::cerr << "Not a .pcd file: " << p << "\n";
        return;
    }
    if (fs::is_directory(p))
    {
        std::vector<fs::path> files;
        for (const auto &entry : fs::directory_iterator(p))
        {
            if (entry.is_regular_file() && is_pcd_file(entry.path()))
                files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
        for (const auto &f : files)
            load_one_pcd(f, points);
        std::cout << "Done. total points=" << points.size() << "\n";
        return;
    }
    std::cerr << "Unsupported path type: " << p << "\n";
}

// ============ 外参工具：兼容 3x4(12) 与 4x4(16) ============
// 假设行主序：[ r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz  (若有则) 0 0 0 1 ]
static inline void vec_to_Rt(const std::vector<float> &ext, cv::Matx33f &R, cv::Vec3f &t)
{
    if (ext.size() != 12 && ext.size() != 16)
        throw std::runtime_error("extrinsic vector must have 12 (3x4) or 16 (4x4) floats.");

    R = Matx33f(
        ext[0], ext[1], ext[2],
        ext[4], ext[5], ext[6],
        ext[8], ext[9], ext[10]);
    t = Vec3f(ext[3], ext[7], ext[11]);
}

static inline std::vector<float> Rt_to_vec(const cv::Matx33f &R, const cv::Vec3f &t, bool want16)
{
    std::vector<float> ext;
    if (want16) ext.assign(16, 0.f);
    else        ext.assign(12, 0.f);

    // 行主序填充
    ext[0] = R(0, 0); ext[1] = R(0, 1); ext[2]  = R(0, 2); ext[3]  = t(0);
    ext[4] = R(1, 0); ext[5] = R(1, 1); ext[6]  = R(1, 2); ext[7]  = t(1);
    ext[8] = R(2, 0); ext[9] = R(2, 1); ext[10] = R(2, 2); ext[11] = t(2);

    if (want16){
        ext[12] = 0; ext[13] = 0; ext[14] = 0; ext[15] = 1;
    }
    return ext;
}

// ============ 欧拉角（度） -> 旋转矩阵（右手系, X=俯仰, Y=翻滚, Z=偏航） ============
static inline float deg2rad(float d) { return d * (float)CV_PI / 180.f; }

static inline cv::Matx33f Rx_deg(float deg)
{
    float a = deg2rad(deg);
    float c = cos(a), s = sin(a);
    return Matx33f(1, 0, 0, 0, c, -s, 0, s, c);
}
static inline cv::Matx33f Ry_deg(float deg)
{
    float a = deg2rad(deg);
    float c = cos(a), s = sin(a);
    return Matx33f(c, 0, s, 0, 1, 0, -s, 0, c);
}
static inline cv::Matx33f Rz_deg(float deg)
{
    float a = deg2rad(deg);
    float c = cos(a), s = sin(a);
    return Matx33f(c, -s, 0, s, c, 0, 0, 0, 1);
}

// 数值稳定：对 R 做一次正交化（SVD）
static inline cv::Matx33f orthonormalize(const cv::Matx33f &Rin)
{
    cv::Mat M(3, 3, CV_32F);
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) M.at<float>(r,c) = Rin(r,c);
    cv::SVD svd(M);
    cv::Mat R = svd.u * svd.vt;
    cv::Matx33f Rout;
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) Rout(r,c) = (float)R.at<float>(r,c);
    return Rout;
}

// ============ 投影与绘制 ============
struct DepthRange { float dmin, dmax; };

static DepthRange compute_depth_range(const std::vector<PointXYZRGB> &pts)
{
    DepthRange dr{std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
    for (auto &p : pts)
    {
        float d = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        dr.dmin = std::min(dr.dmin, d);
        dr.dmax = std::max(dr.dmax, d);
    }
    if (dr.dmax <= dr.dmin) dr.dmax = dr.dmin + 1e-6f;
    return dr;
}

// 打印当前“增量状态”（HUD 显示用）
static inline void print_state(float d_pitch_x, float d_roll_y, float d_yaw_z, const cv::Vec3f &tmm)
{
    printf("STATE  d_pitch_x=%.3f deg  d_roll_y=%.3f deg  d_yaw_z=%.3f deg  |  t(mm)=[%.2f, %.2f, %.2f]\n",
           d_pitch_x, d_roll_y, d_yaw_z, tmm[0], tmm[1], tmm[2]);
}

static void render_projection(const cv::Mat &base,
                              const std::vector<PointXYZRGB> &pointcloud,
                              const DepthRange &dr,
                              const OcamModel &ocam_model,
                              const std::vector<float> &extrinsic_any, // 12 或 16 都行
                              int threshold_lidar,
                              cv::Mat &out_img)
{
    out_img = base.clone();
    float theoryUV[2] = {0, 0};
    int cnt = 0;

    for (auto &pt : pointcloud)
    {
        float x = pt.x, y = pt.y, z = pt.z;
        float depth = std::sqrt(x * x + y * y + z * z);
        float t = (depth - dr.dmin) / (dr.dmax - dr.dmin);
        t = std::clamp(t, 0.0f, 1.0f);

        // 注意：向 getTheoreticalUV_pano 传入“同格式”的外参（12 就传 12，16 就传 16）
        std::vector<float> ext_pass = extrinsic_any; // 拷贝一份以防函数内部写入
        getTheoreticalUV_pano(theoryUV, ocam_model, ext_pass, x * 1000.f, y * 1000.f, z * 1000.f);

        int u = (int)std::floor(theoryUV[0] + 0.5f);
        int v = (int)std::floor(theoryUV[1] + 0.5f);
        if (u < 0 || u >= out_img.cols || v < 0 || v >= out_img.rows)
            continue;

        float hue = (1.0f - t) * 270.0f;
        int r, g, b;
        HSVtoRGB(hue, 1.0f, 1.0f, r, g, b);
        cv::circle(out_img, Point(u, v), 5, Scalar(b, g, r), -1);

        if (++cnt > threshold_lidar) break;
    }
}

// 叠加 HUD 文本
static void draw_hud(cv::Mat &img,
                     float pitch_deg, float roll_deg, float yaw_deg,
                     const cv::Vec3f &tmm)
{
    char buf[256];
    snprintf(buf, sizeof(buf),
             "Pitch(x): %.3f deg | Roll(y): %.3f deg | Yaw(z): %.3f deg",
             pitch_deg, roll_deg, yaw_deg);
    putText(img, buf, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);

    snprintf(buf, sizeof(buf),
             "t (mm): x=%.2f  y=%.2f  z=%.2f", tmm[0], tmm[1], tmm[2]);
    putText(img, buf, Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);

    putText(img, "[u/j]=+/- pitch(x)  [n/m]=-/+ roll(y)  [h/k]=-/+ yaw(z)",
            Point(20, 90),  FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200, 200, 200), 1);
    putText(img, "[w/s]=+/- x  [a/d]=-/+ y  [z/x]=-/+ z   [r]=reset  [p]=print  [q]=quit",
            Point(20, 115), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200, 200, 200), 1);
}

// ============ 主程序 ============
int main(int argc, char **argv)
{
    input_pcd_dir    = "/Users/losehu/Documents/20250813/pcd/pcd14.pcd";
    input_photo_path = "/Users/losehu/Documents/20250813/14.jpg";

    output_path   = "./result.png";
    intrinsic_path = "../a.txt";
    extrinsic_path = "./extrinsic_pano.txt";
    threshold_lidar = 30000;

    src_img = cv::imread(input_photo_path);
    if (src_img.empty())
    {
        std::cout << "No Picture found by filename: " << input_photo_path << std::endl;
        return 0;
    }

    std::vector<PointXYZRGB> pointcloud;
    loadPointcloudFromPath(input_pcd_dir, pointcloud);

    OcamModel ocam_model;
    get_ocam_model(ocam_model, intrinsic_path);

    // 读入初始外参（兼容 12 / 16）
    std::vector<float> extrinsic_any;
    getExtrinsic(extrinsic_path, extrinsic_any);
    if (extrinsic_any.size() != 12 && extrinsic_any.size() != 16)
    {
        std::cerr << "Extrinsic must have 12 (3x4) or 16 (4x4) floats, got "
                  << extrinsic_any.size() << std::endl;
        return -1;
    }
    bool want16 = (extrinsic_any.size() == 16);

    // 拆分为 R0, t0（基准）
    Matx33f R0;
    Vec3f   t0;
    try {
        vec_to_Rt(extrinsic_any, R0, t0);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // —— 当前“真状态”：每次按键直接更新它们（关键！）——
    Matx33f R_cur = R0;
    Vec3f   t_cur = t0;

    // HUD 显示的累计角度（非必须，用于提示）
    float cum_pitch_x = 0.f, cum_roll_y = 0.f, cum_yaw_z = 0.f;

    // 深度范围（一次性统计）
    DepthRange dr = compute_depth_range(pointcloud);

    // 窗口
    const string kWin = "Lidar->Image Projection (interactive extrinsic tuning)";
    namedWindow(kWin, WINDOW_AUTOSIZE);

    cout << "Controls:\n"
         << "  u/j : + / - Pitch (around X axis)\n"
         << "  n/m : - / + Roll  (around Y axis)\n"
         << "  h/k : - / + Yaw   (around Z axis)\n"
         << "  w/s : + / - translate X (mm)\n"
         << "  a/d : - / + translate Y (mm)\n"
         << "  z/x : - / + translate Z (mm)\n"
         << "  r   : reset to initial extrinsic\n"
         << "  p   : print current extrinsic (always shows 4x4)\n"
         << "  q/ESC: quit (save current view to result.png)\n";

    while (true)
    {
        // —— 渲染：使用当前真状态 R_cur / t_cur —— 
        std::vector<float> ext_now = Rt_to_vec(R_cur, t_cur, want16);

        Mat canvas;
        render_projection(src_img, pointcloud, dr, ocam_model, ext_now, threshold_lidar, canvas);
        draw_hud(canvas, cum_pitch_x, cum_roll_y, cum_yaw_z, t_cur);
        imshow(kWin, canvas);

        int key = waitKeyEx(0); // 阻塞等待按键
        bool changed = false;

        // 退出
        if (key == 27 /*ESC*/ || key == 'q' || key == 'Q')
        {
            imwrite(output_path, canvas);
            break;
        }

        // 重置
        if (key == 'r' || key == 'R')
        {
            R_cur = R0;
            t_cur = t0;
            cum_pitch_x = cum_roll_y = cum_yaw_z = 0.f;
            print_state(cum_pitch_x, cum_roll_y, cum_yaw_z, t_cur);
            continue;
        }

        // 打印 4x4（当前真状态）
        if (key == 'p' || key == 'P')
        {
            printf("extrinsic (4x4 print)\n");
            printf("% .6f % .6f % .6f % .6f\n", R_cur(0,0), R_cur(0,1), R_cur(0,2), t_cur[0]);
            printf("% .6f % .6f % .6f % .6f\n", R_cur(1,0), R_cur(1,1), R_cur(1,2), t_cur[1]);
            printf("% .6f % .6f % .6f % .6f\n", R_cur(2,0), R_cur(2,1), R_cur(2,2), t_cur[2]);
            printf("0 0 0 1\n");
            continue;
        }

        // ===== 旋转（左乘在当前相机坐标系）=====
        // [u/j] = +/− pitch (绕 X)
        if (key == 'u' || key == 'U') {
            R_cur = Rx_deg(+kDegStep) * R_cur;
            R_cur = orthonormalize(R_cur);
            cum_pitch_x += kDegStep;
            changed = true;
        }
        if (key == 'j' || key == 'J') {
            R_cur = Rx_deg(-kDegStep) * R_cur;
            R_cur = orthonormalize(R_cur);
            cum_pitch_x -= kDegStep;
            changed = true;
        }

        // [n/m] = −/+ roll (绕 Y)
        if (key == 'n' || key == 'N') {
            R_cur = Ry_deg(-kDegStep) * R_cur;
            R_cur = orthonormalize(R_cur);
            cum_roll_y -= kDegStep;
            changed = true;
        }
        if (key == 'm' || key == 'M') {
            R_cur = Ry_deg(+kDegStep) * R_cur;
            R_cur = orthonormalize(R_cur);
            cum_roll_y += kDegStep;
            changed = true;
        }

        // [h/k] = −/+ yaw (绕 Z)
        if (key == 'h' || key == 'H') {
            R_cur = Rz_deg(-kDegStep) * R_cur;
            R_cur = orthonormalize(R_cur);
            cum_yaw_z -= kDegStep;
            changed = true;
        }
        if (key == 'k' || key == 'K') {
            R_cur = Rz_deg(+kDegStep) * R_cur;
            R_cur = orthonormalize(R_cur);
            cum_yaw_z += kDegStep;
            changed = true;
        }

        // ===== 平移（在当前相机系，本地 xyz，单位 mm）=====
        // 若你的 t 是定义在 LiDAR/世界系，请改为：t_cur += R_cur * Vec3f(dx,dy,dz);
        if (key == 'w' || key == 'W') { t_cur[0] += kTransStepMM; changed = true; }
        if (key == 's' || key == 'S') { t_cur[0] -= kTransStepMM; changed = true; }

        if (key == 'a' || key == 'A') { t_cur[1] -= kTransStepMM; changed = true; }
        if (key == 'd' || key == 'D') { t_cur[1] += kTransStepMM; changed = true; }

        if (key == 'z' || key == 'Z') { t_cur[2] -= kTransStepMM; changed = true; }
        if (key == 'x' || key == 'X') { t_cur[2] += kTransStepMM; changed = true; }

        // 若有改动，则打印当前 HUD 状态
        if (changed) {
            print_state(cum_pitch_x, cum_roll_y, cum_yaw_z, t_cur);
        }
    }

    return 0;
}
