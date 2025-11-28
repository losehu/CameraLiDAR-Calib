#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <regex>
#include <algorithm>
#include <filesystem> // ★ 新增

#include "common.h"
#include <sstream>
namespace fs = std::filesystem;
#ifdef __APPLE__
#include <cstdio>
#include <cstdlib>


void silenceFrameworkLogs()
{
    // 关闭 Apple 的统一日志（对 os_log 有效）
    setenv("OS_ACTIVITY_MODE", "disable", 1);
    // 把 stderr 重定向到 /dev/null，屏蔽 NSLog/fprintf(stderr, ...)
    freopen("/dev/null", "w", stderr);
}
#endif

// ============ 画球工具 ============
bool drawSphereAtMeters(pcl::visualization::PCLVisualizer::Ptr viewer,
                        double x_m, double y_m, double z_m,
                        double radius_m = 0.30,
                        const std::string &id = "marker",
                        double r = 0.0, double g = 1.0, double b = 0.0,
                        double opacity = 0.6)
{
    if (!viewer)
        return false;
    viewer->removeShape(id);
    pcl::PointXYZ center(x_m, y_m, z_m);
    if (!viewer->addSphere(center, radius_m, r, g, b, id))
        return false;
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, id);
    return true;
}
bool drawSphereAtMillimeters(pcl::visualization::PCLVisualizer::Ptr viewer,
                             double x_mm, double y_mm, double z_mm,
                             double radius_mm = 300.0,
                             const std::string &id = "marker_mm",
                             double r = 0.0, double g = 1.0, double b = 0.0,
                             double opacity = 0.6)
{
    const double s = 0.001; // mm -> m
    return drawSphereAtMeters(viewer, x_mm * s, y_mm * s, z_mm * s,
                              radius_mm * s, id, r, g, b, opacity);
}

#define cam_path "saved_cam.cam"
pcl::visualization::PCLVisualizer::Ptr viewer(
    new pcl::visualization::PCLVisualizer("viewer"));

// —— 点选上限与记录 —— //
static const int MAX_POINTS = 4;
static std::vector<std::string> picked_ids;    // 点选球的ID
static std::vector<cv::Point3d> picked_pts_mm; // 点选坐标（mm）

// —— PCD 列表与游标 —— //
static std::vector<std::string> g_pcds;
static int g_cur_idx = -1;
static bool g_shift_down = false;
static std::vector<std::string> picked_ids_backup;
static std::vector<cv::Point3d> picked_pts_backup;

// —— 输出文件路径 —— //
static const std::string OUTPUT_FILE = "/Users/losehu/Documents/归档/Pano_Video/calib/sign/lidar_1120.txt";

// —— 当前点云数据（用于精确点选）—— //
static pcl::PointCloud<pcl::PointXYZ>::Ptr g_current_cloud(new pcl::PointCloud<pcl::PointXYZ>);

// —— 鼠标点击坐标（用于精确点选）—— //
static int g_mouse_x = -1;
static int g_mouse_y = -1;
static bool g_mouse_clicked = false;

// ============ 相机IO ============
static bool save_cam(const std::string &path, const pcl::visualization::Camera &cam)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return false;
    }
    out << "Clipping plane [near,far] " << cam.clip[0] << ", " << cam.clip[1] << "\n";
    out << "Focal point [x,y,z] " << cam.focal[0] << ", " << cam.focal[1] << ", " << cam.focal[2] << "\n";
    out << "Position [x,y,z] " << cam.pos[0] << ", " << cam.pos[1] << ", " << cam.pos[2] << "\n";
    out << "View up [x,y,z] " << cam.view[0] << ", " << cam.view[1] << ", " << cam.view[2] << "\n";
    out << "Camera view angle [degrees] " << cam.fovy << "\n";
    out << "Window size [x,y] " << cam.window_size[0] << ", " << cam.window_size[1] << "\n";
    out << "Window position [x,y] " << cam.window_pos[0] << ", " << cam.window_pos[1] << "\n";
    std::cout << "Camera parameters saved to: " << path << std::endl;
    return true;
}
static bool load_cam(const std::string &path, pcl::visualization::Camera &cam)
{
    std::ifstream in(path);
    if (!in)
        return false;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.find("Position") != std::string::npos)
            std::sscanf(line.c_str(), "Position [x,y,z] %lf, %lf, %lf", &cam.pos[0], &cam.pos[1], &cam.pos[2]);
        else if (line.find("Focal point") != std::string::npos)
            std::sscanf(line.c_str(), "Focal point [x,y,z] %lf, %lf, %lf", &cam.focal[0], &cam.focal[1], &cam.focal[2]);
        else if (line.find("View up") != std::string::npos)
            std::sscanf(line.c_str(), "View up [x,y,z] %lf, %lf, %lf", &cam.view[0], &cam.view[1], &cam.view[2]);
        else if (line.find("Clipping plane") != std::string::npos)
            std::sscanf(line.c_str(), "Clipping plane [near,far] %lf, %lf", &cam.clip[0], &cam.clip[1]);
        else if (line.find("Camera view angle") != std::string::npos)
            std::sscanf(line.c_str(), "Camera view angle [degrees] %lf", &cam.fovy);
        else if (line.find("Window size") != std::string::npos)
            std::sscanf(line.c_str(), "Window size [x,y] %lf, %lf", &cam.window_size[0], &cam.window_size[1]);
        else if (line.find("Window position") != std::string::npos)
            std::sscanf(line.c_str(), "Window position [x,y] %lf, %lf", &cam.window_pos[0], &cam.window_pos[1]);
    }
    return true;
}

// ============ 辅助：根据文件名提取 pcd(\d+) ============
static bool extract_need_id_from_path(const std::string &pcd_path, int &need_id)
{
    std::string stem = fs::path(pcd_path).stem().string(); // 如 "1" 或 "pcd1" 或 "pcd_001"
    std::regex re(R"((?:pcd)?[_\- ]*(\d+))", std::regex::icase);
    std::smatch m;
    if (std::regex_search(stem, m, re))
    {
        need_id = std::stoi(m[1]);
        return true;
    }
    return false;
}

// ============ 检查txt文件中是否已有该pcd的结果 ============
static bool pcdResultExists(const std::string &pcd_path)
{
    std::ifstream in(OUTPUT_FILE);
    if (!in.is_open())
        return false; // 文件不存在，说明没有记录
    
    std::string pcd_filename = fs::path(pcd_path).filename().string();
    std::string marker = "# " + pcd_filename;
    
    std::string line;
    while (std::getline(in, line))
    {
        if (line == marker)
            return true; // 找到了该pcd的标记
    }
    return false;
}
// 自动跳过所有已存在记录的PCD，返回是否找到未标注的PCD
bool skipLabeledPCDs()
{
    while (g_cur_idx < (int)g_pcds.size())
    {
        const std::string &pcd_path = g_pcds[g_cur_idx];
        
        // 如果该pcd已经有完整记录则跳过
        if (pcdResultExists(pcd_path))
        {
            std::cout << "[Skip] Already labeled: "
                      << fs::path(pcd_path).filename().string() << std::endl;
            g_cur_idx++;
        }
        else
            return true;   // 找到一个未标注的PCD
    }

    // 所有 pcd 都已经标注 → 退出程序
    std::cout << "All PCD files already labeled. Exit." << std::endl;
    exit(0);
}

// ============ 精确点选：根据鼠标坐标和相机参数找到最近的点 ============
static bool findNearestPointFromMouse(int mouse_x, int mouse_y, 
                                      pcl::PointXYZ &nearest_point, 
                                      double &min_distance)
{
    if (g_current_cloud->empty())
        return false;
    
    // 获取相机参数
    pcl::visualization::Camera cam;
    viewer->getCameraParameters(cam);
    
    // 获取窗口尺寸
    auto renWin = viewer->getRenderWindow();
    int *sz = renWin->GetSize();
    int window_width = sz[0];
    int window_height = sz[1];
    
    // 将鼠标坐标归一化到[-1, 1]
    double normalized_x = 2.0 * mouse_x / window_width - 1.0;
    double normalized_y = 1.0 - 2.0 * mouse_y / window_height; // Y轴翻转
    
    // 计算视场角的一半（弧度）
    double fov_rad = cam.fovy * M_PI / 180.0;
    double tan_fov = tan(fov_rad / 2.0);
    
    // 计算射线方向（在相机坐标系中）
    double aspect = cam.window_size[0] / cam.window_size[1];
    double ray_x = normalized_x * tan_fov * aspect;
    double ray_y = normalized_y * tan_fov;
    double ray_z = -1.0; // 相机看向-Z方向
    
    // 归一化射线方向
    double ray_len = sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;
    
    // 将射线方向转换到世界坐标系
    double view_dir[3] = {
        cam.focal[0] - cam.pos[0],
        cam.focal[1] - cam.pos[1],
        cam.focal[2] - cam.pos[2]
    };
    double view_len = sqrt(view_dir[0]*view_dir[0] + view_dir[1]*view_dir[1] + view_dir[2]*view_dir[2]);
    if (view_len < 1e-6)
        return false;
    view_dir[0] /= view_len;
    view_dir[1] /= view_len;
    view_dir[2] /= view_len;
    
    // 计算右向量（view_dir × view_up）
    double right[3] = {
        view_dir[1] * cam.view[2] - view_dir[2] * cam.view[1],
        view_dir[2] * cam.view[0] - view_dir[0] * cam.view[2],
        view_dir[0] * cam.view[1] - view_dir[1] * cam.view[0]
    };
    double right_len = sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (right_len < 1e-6)
        return false;
    right[0] /= right_len;
    right[1] /= right_len;
    right[2] /= right_len;
    
    // 重新计算上向量（right × view_dir）
    double up[3] = {
        right[1] * view_dir[2] - right[2] * view_dir[1],
        right[2] * view_dir[0] - right[0] * view_dir[2],
        right[0] * view_dir[1] - right[1] * view_dir[0]
    };
    
    // 世界坐标系中的射线方向
    double world_ray[3] = {
        ray_x * right[0] + ray_y * up[0] - ray_z * view_dir[0],
        ray_x * right[1] + ray_y * up[1] - ray_z * view_dir[1],
        ray_x * right[2] + ray_y * up[2] - ray_z * view_dir[2]
    };
    
    // 在点云中查找与射线最近的点
    min_distance = std::numeric_limits<double>::max();
    bool found = false;
    const double MAX_PICK_DISTANCE = 0.5; // 最大点选距离（米），超过这个距离认为无效
    
    for (const auto &pt : *g_current_cloud)
    {
        // 计算点到射线的距离
        double vec_to_point[3] = {
            pt.x - cam.pos[0],
            pt.y - cam.pos[1],
            pt.z - cam.pos[2]
        };
        
        // 投影到射线方向
        double proj_len = vec_to_point[0] * world_ray[0] + 
                          vec_to_point[1] * world_ray[1] + 
                          vec_to_point[2] * world_ray[2];
        
        // 如果点在射线后方，跳过
        if (proj_len < 0)
            continue;
        
        // 计算垂直距离
        double proj_vec[3] = {
            proj_len * world_ray[0],
            proj_len * world_ray[1],
            proj_len * world_ray[2]
        };
        
        double perp_vec[3] = {
            vec_to_point[0] - proj_vec[0],
            vec_to_point[1] - proj_vec[1],
            vec_to_point[2] - proj_vec[2]
        };
        
        double dist = sqrt(perp_vec[0]*perp_vec[0] + perp_vec[1]*perp_vec[1] + perp_vec[2]*perp_vec[2]);
        
        // 只考虑垂直距离（到射线的距离），不考虑深度
        if (dist < min_distance && dist < MAX_PICK_DISTANCE)
        {
            min_distance = dist;
            nearest_point = pt;
            found = true;
        }
    }
    
    return found;
}

// ============ 将当前pcd的点选结果写入txt文件 ============
static bool writePcdResultsToFile(const std::string &pcd_path)
{
    if (picked_pts_mm.empty())
        return false; // 没有点选结果，不写入
    
    std::ofstream out(OUTPUT_FILE, std::ios::app); // 追加模式
    if (!out.is_open())
    {
        std::cerr << "Failed to open output file: " << OUTPUT_FILE << std::endl;
        return false;
    }
    
    std::string pcd_filename = fs::path(pcd_path).filename().string();
    out << "# " << pcd_filename << "\n";
    
    for (size_t i = 0; i < picked_pts_mm.size(); ++i)
    {
        const auto &p = picked_pts_mm[i];
        int seq = static_cast<int>(i) + 1; // 1..N
        out << seq << "\n"
            << p.x << " " << p.y << " " << p.z << "\n";
    }
    
    out.flush();
    std::cout << "Results saved to " << OUTPUT_FILE << " for " << pcd_filename << std::endl;
    return true;
}

// ============ 绘制与当前 PCD 匹配的 LiDAR 球 ============
static void drawLidarMarkersForId(int need_id)
{
    std::ifstream inFile_lidar(OUTPUT_FILE);
    if (!inFile_lidar.is_open())
        return;

    std::string lineStr_lidar;
    int now_cnt = 0;
    while (std::getline(inFile_lidar, lineStr_lidar))
    {
        if (now_cnt / 9 == need_id - 1)
        {
            if (lineStr_lidar.size() > 10)
            {
                double x, y, z;
                std::string str;
                std::stringstream line_lidar(lineStr_lidar);
                line_lidar >> str;
                x = str2double(str);
                line_lidar >> str;
                y = str2double(str);
                line_lidar >> str;
                z = str2double(str);

                std::string name = "big_green_ball" + std::to_string((now_cnt % 9) / 2);
                if ((now_cnt % 9) /2 == 1)
                    drawSphereAtMillimeters(viewer, x, y, z, /*radius_mm=*/100.0, /*id=*/name, 1, 0, 0);
                else if ((now_cnt % 9) / 2 == 2)
                    drawSphereAtMillimeters(viewer, x, y, z, /*radius_mm=*/100.0, /*id=*/name, 0, 1, 0);
                else if ((now_cnt % 9) / 2 == 3)
                    drawSphereAtMillimeters(viewer, x, y, z, /*radius_mm=*/100.0, /*id=*/name, 0, 0, 1);
                else
                    // 改为黄色 (r=1,g=1,b=0) 与 get_point.py 的配色一致（红、绿、蓝、黄）
                    drawSphereAtMillimeters(viewer, x, y, z, /*radius_mm=*/100.0, /*id=*/name, 1, 1, 0);


            }
        }
        now_cnt += 1;
    }
}
// ============ 核心：加载并显示某个 PCD（清理旧状态，叠加激光球） ============
static bool loadOnePCDAndSetup(const std::string &pcd_path)
{
    // 清理旧形状（含之前的点选球 & 激光球）
    viewer->removeAllShapes();

    // 清空点选记录
    picked_ids.clear();  // Clear the picked IDs
    picked_pts_mm.clear();  // Clear the picked points

    // 替换点云
    viewer->removePointCloud("cloud");
    g_current_cloud->clear();
    if (pcl::io::loadPCDFile(pcd_path, *g_current_cloud) != 0)
    {
        std::cerr << "Failed to load: " << pcd_path << std::endl;
        return false;
    }

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_by_z(g_current_cloud, "z");
    if (color_by_z.isCapable())
        viewer->addPointCloud<pcl::PointXYZ>(g_current_cloud, color_by_z, "cloud");
    else
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(g_current_cloud, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(g_current_cloud, red, "cloud");
    }
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    // 解析 need_id 并叠加 LiDAR 球
    int need_id = 1;
    if (!extract_need_id_from_path(pcd_path, need_id))
        std::cout << "[id] not found\n";
    drawLidarMarkersForId(need_id);

    std::cout << fs::path(pcd_path).filename().string() << std::endl;
    return true;
}


// ============ 键盘回调 ============
void keyboardCallback(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
    auto viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);

if ((event.getKeySym() == "Shift_L" || event.getKeySym() == "Shift_R"))
{
    if (event.keyDown())
    {
        g_shift_down = true;

        // 备份
        picked_ids_backup = picked_ids;
        picked_pts_backup = picked_pts_mm;

        // 删除所有小球
        for (auto &id : picked_ids)
            viewer->removeShape(id);

        viewer->getRenderWindow()->Render();
    }
    else
    {
        g_shift_down = false;

        // 松开 Shift → 将临时数据同步到正式变量
        picked_ids = picked_ids_backup;
        picked_pts_mm = picked_pts_backup;

        // 恢复所有小球（按序号着色：红、绿、蓝、黄）
        for (size_t i = 0; i < picked_ids.size(); i++)
        {
            const auto &name = picked_ids[i];
            const auto &p = picked_pts_mm[i];
            // 颜色映射
            double r = 0, g = 1, b = 0;
            if (i == 0) { r = 1; g = 0; b = 0; }       // 第1个：红
            else if (i == 1) { r = 0; g = 1; b = 0; }  // 第2个：绿
            else if (i == 2) { r = 0; g = 0; b = 1; }  // 第3个：蓝
            else if (i == 3) { r = 1; g = 1; b = 0; }  // 第4个：黄
            drawSphereAtMillimeters(::viewer, p.x, p.y, p.z, 30.0, name, r, g, b);
        }

        viewer->getRenderWindow()->Render();
    }
}

    if ((event.getKeySym() == "p" || event.getKeySym() == "P") && event.keyDown())
    {
        pcl::visualization::Camera cam;
        viewer->getCameraParameters(cam);
        save_cam("saved_cam.cam", cam);
    }
    if ((event.getKeySym() == "l" || event.getKeySym() == "L") && event.keyDown())
    {
        pcl::visualization::Camera cam;
        if (load_cam("saved_cam.cam", cam))
        {
            auto renWin = viewer->getRenderWindow();
            int *sz = renWin->GetSize();
            cam.window_size[0] = sz[0];
            cam.window_size[1] = sz[1];

            viewer->setCameraParameters(cam);
            viewer->getRenderWindow()->Render();
            // 让裁剪范围和投影矩阵刷新一下
            viewer->getRendererCollection()->GetFirstRenderer()->ResetCameraClippingRange();
            // std::cout << "Loaded camera parameters from file\n";
        }
        else
        {
            std::cerr << "Failed to load camera parameters from file\n";
        }
    }
    if ((event.getKeySym() == "b" || event.getKeySym() == "B") && event.keyDown())
    {
        if (!picked_ids.empty())
        {
            const std::string last_id = picked_ids.back();
            viewer->removeShape(last_id);
            picked_ids.pop_back();
            if (!picked_pts_mm.empty())
                picked_pts_mm.pop_back();
            // 刷新一下
            viewer->getRendererCollection()->GetFirstRenderer()->ResetCameraClippingRange();
            viewer->getRenderWindow()->Render();
            viewer->spinOnce(1);
        }
    }
    // —— 按 Enter（Return/KP_Enter）保存"当前所有已经点过的点"到txt文件，然后切到下一张 —— //
    if ((event.getKeySym() == "Return" || event.getKeySym() == "KP_Enter") && event.keyDown())
    {
        // 1) 检查当前pcd是否已有结果，如果没有则写入文件
        if (!g_pcds.empty() && g_cur_idx >= 0 && g_cur_idx < (int)g_pcds.size())
        {
            const std::string &current_pcd = g_pcds[g_cur_idx];
            {
                const std::string &current_pcd = g_pcds[g_cur_idx];
                std::string pcd_filename = fs::path(current_pcd).filename().string();
                std::string marker = "# " + pcd_filename;
            
                // ---- Step 1: 读取整个文件 ----
                std::ifstream in(OUTPUT_FILE);
                std::vector<std::string> lines;
                if (in.is_open())
                {
                    std::string line;
                    bool skip_block = false;
                    while (std::getline(in, line))
                    {
                        // 如果遇到当前 PCD 的开头，开始跳过旧内容
                        if (line == marker)
                        {
                            skip_block = true;
                            continue;
                        }
            
                        // 当前是其他 PCD 开始（以 "# " 开头），恢复写入
                        if (skip_block && line.rfind("# ", 0) == 0)
                        {
                            skip_block = false;
                        }
            
                        // 只有不属于旧块的行才保存
                        if (!skip_block)
                            lines.push_back(line);
                    }
                    in.close();
                }
            
                // ---- Step 2: 覆盖整个文件（旧内容 + 新内容）----
                std::ofstream out(OUTPUT_FILE, std::ios::trunc);
                if (!out.is_open())
                {
                    std::cerr << "Failed to open output file: " << OUTPUT_FILE << std::endl;
                }
                else
                {
                    // 写回保留的所有内容
                    for (const auto &l : lines)
                        out << l << "\n";
            
                    // 追加新的 PCD 记录
                    out << "# " << pcd_filename << "\n";
                    for (size_t i = 0; i < picked_pts_mm.size(); ++i)
                    {
                        const auto &p = picked_pts_mm[i];
                        int seq = static_cast<int>(i) + 1; 
                        out << seq << "\n"
                            << p.x << " " << p.y << " " << p.z << "\n";
                    }
                    out.close();
                }
            
                std::cout << "Results for " << pcd_filename
                          << " saved (old record overwritten if existed)." << std::endl;
            }
            
        }

        // 2) 切到下一张 PCD（循环）
        if (!g_pcds.empty())
        {
            if (g_cur_idx + 1 < (int)g_pcds.size())
                g_cur_idx = (g_cur_idx + 1) % (int)g_pcds.size();
            else
                exit(0);
            loadOnePCDAndSetup(g_pcds[g_cur_idx]);
        }
    }

    // 下一帧
    if (event.getKeySym() == "n" && event.keyDown())
    {
        if (!g_pcds.empty())
        {
            g_cur_idx = (g_cur_idx + 1) % g_pcds.size();
            loadOnePCDAndSetup(g_pcds[g_cur_idx]);
        }
    }
    //上一帧（大写 N）
    if (event.getKeySym() == "N" && event.keyDown())
    {
        if (!g_pcds.empty())
        {
            g_cur_idx = (g_cur_idx - 1 + (int)g_pcds.size()) % (int)g_pcds.size();
            loadOnePCDAndSetup(g_pcds[g_cur_idx]);
        }
    }
}

// 前向声明
void visualization();

// ============ main ============
int main(int argc, char **argv)
{

#ifdef __APPLE__
    silenceFrameworkLogs();
#endif
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <PCD file or directory>" << std::endl;
        return 1;
    }

    const std::string input_path = argv[1];

    // 收集PCD列表（目录或单文件）
    if (fs::is_directory(input_path))
    {
        for (auto &entry : fs::directory_iterator(input_path))
        {
            if (!entry.is_regular_file())
                continue;
            if (entry.path().extension() == ".pcd" || entry.path().extension() == ".PCD")
                g_pcds.push_back(entry.path().string());
        }
        std::sort(g_pcds.begin(), g_pcds.end());
    }
    else
    {
        g_pcds.push_back(input_path);
    }

    if (g_pcds.empty())
    {
        std::cerr << "No PCD files found." << std::endl;
        return 1;
    }

    // 可选：加载相机参数
    pcl::visualization::Camera cam;
    if (load_cam("saved_cam.cam", cam))
    {
        auto renWin = viewer->getRenderWindow();
        int *sz = renWin->GetSize(); // sz[0] = width, sz[1] = height
        int w = sz[0];
        int h = sz[1];

        // 如果你要把相机里的 window_size 覆盖为当前窗口尺寸：
        cam.window_size[0] = w;
        cam.window_size[1] = h;

        viewer->setCameraParameters(cam);
        // 或者你想把窗口调整到相机里记录的尺寸：
        // viewer->setSize(static_cast<int>(cam.window_size[0]), static_cast<int>(cam.window_size[1]));

        // std::cout << "Loaded camera parameters from file" << std::endl;
    }
    else
    {
        std::cerr << "Failed to load camera parameters from file" << std::endl;
    }
    std::cout << "> Point picking enabled.  [n] next, [N] prev, [b] undo, [Enter] save to file and next\n";

    // 先加载第一帧
    g_cur_idx = 0;

    // ⬅ 添加：自动跳过已标注的PCD
    skipLabeledPCDs();
    
    // 加载第一个未标注的PCD
    if (!loadOnePCDAndSetup(g_pcds[g_cur_idx]))
        return 1;
    
    // 进入交互
    visualization();
    return 0;
}

// ============ 只做回调注册与spin ============
void visualization()
{

    // 键盘回调
    viewer->registerKeyboardCallback(keyboardCallback, (void *)viewer.get());

viewer->registerPointPickingCallback(
    [](const pcl::visualization::PointPickingEvent &e, void *)
{
    if (!g_shift_down)
        return;   // 只有 Shift + 左键 才处理

    int point_idx = e.getPointIndex();
    if (point_idx == -1)
        return;

    if (picked_ids_backup.size() >= MAX_POINTS)
        return;

    // 从点云取点（最准确）
    if (point_idx < 0 || point_idx >= (int)g_current_cloud->size())
        return;

    const auto &pt = (*g_current_cloud)[point_idx];
    double x_mm = pt.x * 1000.0;
    double y_mm = pt.y * 1000.0;
    double z_mm = pt.z * 1000.0;

    int seq = (int)picked_ids_backup.size() + 1;
    std::string name = "picked_ball" + std::to_string(seq);

    // 按序号设置颜色：1->红, 2->绿, 3->蓝, 4->黄
    double r = 0, g = 1, b = 0;
    if (seq == 1) { r = 1; g = 0; b = 0; }
    else if (seq == 2) { r = 0; g = 1; b = 0; }
    else if (seq == 3) { r = 0; g = 0; b = 1; }
    else if (seq == 4) { r = 1; g = 1; b = 0; }

    drawSphereAtMillimeters(viewer, x_mm, y_mm, z_mm, 30.0, name, r, g, b);

    picked_ids_backup.push_back(name);
    picked_pts_backup.emplace_back(x_mm, y_mm, z_mm);
});

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    }
}

