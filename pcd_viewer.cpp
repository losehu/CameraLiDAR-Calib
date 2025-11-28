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
#include <filesystem>
#include <map>

#include "common.h"

namespace fs = std::filesystem;

#ifdef __APPLE__
#include <cstdio>
#include <cstdlib>

void silenceFrameworkLogs()
{
    setenv("OS_ACTIVITY_MODE", "disable", 1);
    freopen("/dev/null", "w", stderr);
}
#endif

// ============ 画球工具 ============
bool drawSphereAtMillimeters(pcl::visualization::PCLVisualizer::Ptr viewer,
                             double x_mm, double y_mm, double z_mm,
                             double radius_mm = 100.0,
                             const std::string &id = "marker_mm",
                             double r = 0.0, double g = 1.0, double b = 0.0,
                             double opacity = 0.6)
{
    if (!viewer)
        return false;
    viewer->removeShape(id);
    const double s = 0.001; // mm -> m
    pcl::PointXYZ center(x_mm * s, y_mm * s, z_mm * s);
    if (!viewer->addSphere(center, radius_mm * s, r, g, b, id))
        return false;
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, id);
    return true;
}

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

// ============ 数据结构：每个PCD对应的4个点 ============
struct PCDPoints
{
    std::string pcd_filename;
    std::vector<cv::Point3d> points; // 最多4个点(mm)
};

// ============ 解析txt文件 ============
bool parseTxtFile(const std::string &txt_path, std::map<std::string, PCDPoints> &data)
{
    std::ifstream in(txt_path);
    if (!in.is_open())
    {
        std::cerr << "Failed to open txt file: " << txt_path << std::endl;
        return false;
    }

    std::string line;
    PCDPoints current_pcd;
    int point_seq = 0;
    
    while (std::getline(in, line))
    {
        // 去除首尾空格
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty())
            continue;
        
        // 检查是否是PCD文件名（以.pcd结尾或以#开头）
        if (line.find(".pcd") != std::string::npos || line[0] == '#')
        {
            // 保存之前的PCD数据
            if (!current_pcd.pcd_filename.empty())
            {
                data[current_pcd.pcd_filename] = current_pcd;
            }
            
            // 开始新的PCD
            if (line[0] == '#')
                current_pcd.pcd_filename = line.substr(2); // 去掉"# "
            else
                current_pcd.pcd_filename = line;
            current_pcd.points.clear();
            point_seq = 0;
        }
        // 检查是否是序号行（1, 2, 3, 4）
        else if (line == "1" || line == "2" || line == "3" || line == "4")
        {
            point_seq = std::stoi(line);
        }
        // 坐标行（包含三个数字）
        else
        {
            std::istringstream iss(line);
            double x, y, z;
            if (iss >> x >> y >> z)
            {
                current_pcd.points.emplace_back(x, y, z);
            }
        }
    }
    
    // 保存最后一个PCD的数据
    if (!current_pcd.pcd_filename.empty())
    {
        data[current_pcd.pcd_filename] = current_pcd;
    }
    
    return true;
}

// ============ 全局变量 ============
pcl::visualization::PCLVisualizer::Ptr viewer(
    new pcl::visualization::PCLVisualizer("PCD Viewer"));

static std::vector<std::string> g_pcds;
static int g_cur_idx = -1;
static std::map<std::string, PCDPoints> g_txt_data;
static pcl::PointCloud<pcl::PointXYZ>::Ptr g_current_cloud(new pcl::PointCloud<pcl::PointXYZ>);

// ============ 显示当前PCD及其对应的点 ============
static bool loadOnePCDAndSetup(const std::string &pcd_path)
{
    // 清理旧形状
    viewer->removeAllShapes();

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

    // 获取PCD文件名
    std::string pcd_filename = fs::path(pcd_path).filename().string();
    std::cout << "\n[" << (g_cur_idx + 1) << "/" << g_pcds.size() << "] " << pcd_filename << std::endl;

    // 查找并显示对应的点
    auto it = g_txt_data.find(pcd_filename);
    if (it != g_txt_data.end())
    {
        const PCDPoints &pcd_points = it->second;
        std::cout << "Found " << pcd_points.points.size() << " points in txt file" << std::endl;
        
        for (size_t i = 0; i < pcd_points.points.size() && i < 4; ++i)
        {
            const auto &p = pcd_points.points[i];
            std::string name = "point_" + std::to_string(i + 1);
            
            // 按序号设置颜色：1->红, 2->绿, 3->蓝, 4->黄
            double r = 0, g = 1, b = 0;
            if (i == 0) { r = 1; g = 0; b = 0; }       // 第1个：红
            else if (i == 1) { r = 0; g = 1; b = 0; }  // 第2个：绿
            else if (i == 2) { r = 0; g = 0; b = 1; }  // 第3个：蓝
            else if (i == 3) { r = 1; g = 1; b = 0; }  // 第4个：黄
            
            drawSphereAtMillimeters(viewer, p.x, p.y, p.z, 100.0, name, r, g, b);
            std::cout << "  Point " << (i + 1) << ": (" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
        }
    }
    else
    {
        std::cout << "No points found in txt file for this PCD" << std::endl;
    }

    return true;
}

// ============ 键盘回调 ============
void keyboardCallback(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
    auto viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);

    // 按P保存相机参数
    if ((event.getKeySym() == "p" || event.getKeySym() == "P") && event.keyDown())
    {
        pcl::visualization::Camera cam;
        viewer->getCameraParameters(cam);
        save_cam("saved_cam.cam", cam);
    }
    
    // 按L加载相机参数
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
            viewer->getRendererCollection()->GetFirstRenderer()->ResetCameraClippingRange();
        }
        else
        {
            std::cerr << "Failed to load camera parameters from file\n";
        }
    }

    // 按Enter或n切换到下一个PCD
    if (((event.getKeySym() == "Return" || event.getKeySym() == "KP_Enter" || event.getKeySym() == "n") && event.keyDown()))
    {
        if (!g_pcds.empty())
        {
            g_cur_idx = (g_cur_idx + 1) % g_pcds.size();
            if (g_cur_idx == 0)
            {
                std::cout << "\n=== Reached end of PCD list, looping back to start ===" << std::endl;
            }
            loadOnePCDAndSetup(g_pcds[g_cur_idx]);
        }
    }
    
    // 按N切换到上一个PCD
    if (event.getKeySym() == "N" && event.keyDown())
    {
        if (!g_pcds.empty())
        {
            g_cur_idx = (g_cur_idx - 1 + (int)g_pcds.size()) % (int)g_pcds.size();
            loadOnePCDAndSetup(g_pcds[g_cur_idx]);
        }
    }
}

// ============ main ============
int main(int argc, char **argv)
{
#ifdef __APPLE__
    silenceFrameworkLogs();
#endif

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <PCD directory> <txt file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./pcd_back ./sign/lidar_911.txt" << std::endl;
        return 1;
    }

    const std::string pcd_dir = argv[1];
    const std::string txt_file = argv[2];

    // 1. 读取txt文件
    std::cout << "Loading txt file: " << txt_file << std::endl;
    if (!parseTxtFile(txt_file, g_txt_data))
    {
        std::cerr << "Failed to parse txt file" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << g_txt_data.size() << " PCD entries from txt file" << std::endl;

    // 2. 收集PCD文件列表
    if (!fs::is_directory(pcd_dir))
    {
        std::cerr << "Not a valid directory: " << pcd_dir << std::endl;
        return 1;
    }

    for (auto &entry : fs::directory_iterator(pcd_dir))
    {
        if (!entry.is_regular_file())
            continue;
        if (entry.path().extension() == ".pcd" || entry.path().extension() == ".PCD")
            g_pcds.push_back(entry.path().string());
    }
    std::sort(g_pcds.begin(), g_pcds.end());

    if (g_pcds.empty())
    {
        std::cerr << "No PCD files found in directory: " << pcd_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << g_pcds.size() << " PCD files" << std::endl;

    // 3. 加载相机参数（可选）
    pcl::visualization::Camera cam;
    if (load_cam("saved_cam.cam", cam))
    {
        auto renWin = viewer->getRenderWindow();
        int *sz = renWin->GetSize();
        cam.window_size[0] = sz[0];
        cam.window_size[1] = sz[1];
        viewer->setCameraParameters(cam);
    }

    // 4. 注册键盘回调
    viewer->registerKeyboardCallback(keyboardCallback, (void *)viewer.get());

    // 5. 加载第一个PCD
    g_cur_idx = 0;
    if (!loadOnePCDAndSetup(g_pcds[g_cur_idx]))
        return 1;

    std::cout << "\n=== Controls ===" << std::endl;
    std::cout << "  [Enter/n] - Next PCD" << std::endl;
    std::cout << "  [N] - Previous PCD" << std::endl;
    std::cout << "  [p] - Save camera parameters" << std::endl;
    std::cout << "  [l] - Load camera parameters" << std::endl;
    std::cout << "================\n" << std::endl;

    // 6. 进入可视化循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    }

    return 0;
}
