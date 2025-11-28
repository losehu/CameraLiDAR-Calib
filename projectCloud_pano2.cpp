// save as batch_extrinsic_renderer.cpp
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

// ======= 新增：生成并保存所有外参组合的函数 =======
static void generate_and_save_all_extrinsics(const Matx33f &R_base,
                                             const Vec3f &t_base,
                                             bool want16,
                                             const std::vector<PointXYZRGB> &pointcloud,
                                             const DepthRange &dr,
                                             const OcamModel &ocam_model,
                                             int threshold_lidar,
                                             const cv::Mat &base_img,
                                             const std::string &out_dir)
{
    // 创建输出目录
    fs::create_directories(out_dir);

    // 枚举：每个轴 ±90°（pitch X, roll Y, yaw Z）
    std::vector<int> signs = {+1, -1};

    // z: ±2000 mm
    std::vector<int> z_vals = {2000, -2000};

    // base magnitudes for x,y 有两种分配方式（不确定谁是30谁是9.44）
    double magA = 9.44;
    double magB = 30.0;

    int idx = 0;
    for (int sx : signs) // pitch sign
    for (int sy : signs) // roll sign
    for (int sz : signs) // yaw sign
    {
        float pitch_deg = 90.f * (float)sx;
        float roll_deg  = 90.f * (float)sy;
        float yaw_deg   = 90.f * (float)sz;

        // 构造旋转：遵循原程序中左乘到当前相机系（我们这里统一用 R = Rz * Ry * Rx）
        Matx33f R = Rz_deg(yaw_deg) * Ry_deg(roll_deg) * Rx_deg(pitch_deg);
        R = orthonormalize(R);

        for (int zsign : z_vals)
        {
            for (int swap_xy = 0; swap_xy < 2; ++swap_xy) // 0: x=9.44,y=30 ; 1: x=30,y=9.44
            {
                double base_x = swap_xy ? magB : magA;
                double base_y = swap_xy ? magA : magB;

                // x、y 都有正负两种
                for (int x_sign : signs)
                for (int y_sign : signs)
                {
                    double tx_mm = x_sign * base_x;
                    double ty_mm = y_sign * base_y;
                    double tz_mm = (double)zsign;

                    // 构造 t（单位 mm）
                    Vec3f tcur((float)tx_mm, (float)ty_mm, (float)tz_mm);

                    // 将 R, t 转成 ext vector（12 或 16）
                    std::vector<float> ext_now = Rt_to_vec(R, tcur, want16);

                    // 渲染
                    cv::Mat canvas;
                    render_projection(base_img, pointcloud, dr, ocam_model, ext_now, threshold_lidar, canvas);
                    // 在左上角标注参数
                    char hud[256];
                    snprintf(hud, sizeof(hud), "P%.0f R%.0f Y%.0f  x%.2f y%.2f z%.0f swap%d",
                             pitch_deg, roll_deg, yaw_deg, tx_mm, ty_mm, tz_mm, swap_xy);
                    putText(canvas, hud, Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 0), 1);

                    // 文件名（包含索引与参数，便于查找）
                    char fname[512];
                    snprintf(fname, sizeof(fname),
                             "%s/extr_%04d_P%+03.0f_R%+03.0f_Y%+03.0f_x%+.2f_y%+.2f_z%+.0f_swap%d.png",
                             out_dir.c_str(), idx, pitch_deg, roll_deg, yaw_deg, tx_mm, ty_mm, tz_mm, swap_xy);

                    // 保存
                    imwrite(fname, canvas);
                    if (idx % 10 == 0) {
                        std::cout << "Saved " << fname << " (" << idx << ")" << std::endl;
                    }
                    ++idx;
                }
            }
        }
    }
    std::cout << "Finished batch render: total saved = " << idx << " images to " << out_dir << std::endl;
}
// 保存外参矩阵 (4x4)
static void save_extrinsic_txt(const std::string &filename, const cv::Matx33f &R, const cv::Vec3f &t)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        std::cerr << "Cannot open " << filename << " for writing.\n";
        return;
    }

    ofs << "extrinsic\n";
    ofs << std::fixed << std::setprecision(6);

    // 前三行
    for (int i = 0; i < 3; i++)
    {
        ofs << std::setw(12) << R(i,0) << " "
            << std::setw(12) << R(i,1) << " "
            << std::setw(12) << R(i,2) << " "
            << std::setw(12) << t(i)
            << "\n";
    }
    // 最后一行固定
    ofs << "           0            0            0            1\n";

    ofs.close();
    std::cout << "Saved extrinsic to " << filename << std::endl;
}

int main(int argc, char **argv)
{
    input_pcd_dir    = "/Users/losehu/Documents/20250813/pcd/pcd14.pcd";
    input_photo_path = "/Users/losehu/Documents/20250813/14.jpg";

    output_path   = "./result";
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

    // 读取 extrinsic，取矩阵大小信息
    std::vector<float> extrinsic_any;
    getExtrinsic(extrinsic_path, extrinsic_any);
    if (extrinsic_any.size() != 12 && extrinsic_any.size() != 16)
    {
        std::cerr << "Extrinsic must have 12 or 16 floats\n";
        return -1;
    }
    bool want16 = (extrinsic_any.size() == 16);

    // 深度范围
    DepthRange dr = compute_depth_range(pointcloud);

    // 固定角度
    int ax = 0, ay = 0, az = 0;
    Matx33f R = Rz_deg(az) * Ry_deg(ay) * Rx_deg(ax);
    R = orthonormalize(R);
    R = R * Matx33f::diag(Vec3f(1.0f, 1.0f, -1.0f));

    // 平移组合
    std::vector<Vec3f> translations;
    for (float z : {100.f, 50.f})
    {
        for (float a : {9.44f, -9.44f})
        {
            for (float b : {30.f, -30.f})
            {
                // (x, y) = (a, b) 或 (b, a)
                translations.push_back(Vec3f(a, b, z));
                translations.push_back(Vec3f(b, a, z));
            }
        }
    }

    int idx = 0;
    for (auto &t : translations)
    {
        // 外参向量
        std::vector<float> ext = Rt_to_vec(R, t, want16);

        // 渲染
        Mat canvas;
        render_projection(src_img, pointcloud, dr, ocam_model, ext, threshold_lidar, canvas);

        // 保存
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "%s/vis_%03d_ax%d_ay%d_az%d_tx%.2f_ty%.2f_tz%.2f.png",
                 output_path.c_str(), idx, ax, ay, az, t[0], t[1], t[2]);
        std::cout << "Saving " << buf << std::endl;
        imwrite(buf, canvas);
char fname_ext[256];
snprintf(fname_ext, sizeof(fname_ext),
         "%s/extr_%03d.txt",
         output_path.c_str(), idx);
save_extrinsic_txt(fname_ext, R, t);

        idx++;
    }

    std::cout << "Done. Generated " << idx << " images." << std::endl;
    return 0;
}
