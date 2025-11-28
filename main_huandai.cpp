#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "common.h"
#include "result_verify.h"

#include <opencv2/opencv.hpp>
#include <stdexcept>

// 生成绕X轴旋转90度倍数的矩阵
Eigen::Matrix3f rotation_x(int degrees)
{
    float rad = degrees * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    Eigen::Matrix3f R;
    R << 1.0f, 0.0f, 0.0f,
        0.0f, cos_a, -sin_a,
        0.0f, sin_a, cos_a;
    return R;
}

// 生成绕Y轴旋转90度倍数的矩阵
Eigen::Matrix3f rotation_y(int degrees)
{
    float rad = degrees * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    Eigen::Matrix3f R;
    R << cos_a, 0.0f, sin_a,
        0.0f, 1.0f, 0.0f,
        -sin_a, 0.0f, cos_a;
    return R;
}

// 生成绕Z轴旋转90度倍数的矩阵
Eigen::Matrix3f rotation_z(int degrees)
{
    float rad = degrees * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    Eigen::Matrix3f R;
    R << cos_a, -sin_a, 0.0f,
        sin_a, cos_a, 0.0f,
        0.0f, 0.0f, 1.0f;
    return R;
}

// 比较两个矩阵是否相等（考虑浮点误差）
bool matrices_equal(const Eigen::Matrix3f &a, const Eigen::Matrix3f &b, float tolerance = 1e-6f)
{
    return (a - b).norm() < tolerance;
}

// 生成所有90度旋转组合
std::vector<Eigen::Matrix3f> generate_all_90_degree_rotations()
{
    std::vector<Eigen::Matrix3f> rotations;
    std::vector<Eigen::Matrix3f> unique_rotations;

    // 所有可能的旋转角度（90度的倍数）
    std::vector<int> angles = {0, 90, 180, 270};

    // 生成所有可能的旋转组合
    for (int x : angles)
    {
        Eigen::Matrix3f Rx = rotation_x(x);

        for (int y : angles)
        {
            Eigen::Matrix3f Ry = rotation_y(y);

            for (int z : angles)
            {
                Eigen::Matrix3f Rz = rotation_z(z);

                // 组合旋转：R = Rz * Ry * Rx
                Eigen::Matrix3f R = Rz * Ry * Rx;
                rotations.push_back(R);
            }
        }
    }

    // 去除重复的矩阵
    for (const auto &R : rotations)
    {
        bool is_duplicate = false;
        for (const auto &existing : unique_rotations)
        {
            if (matrices_equal(R, existing))
            {
                is_duplicate = true;
                break;
            }
        }
        if (!is_duplicate)
        {
            unique_rotations.push_back(R);
        }
    }

    return unique_rotations;
}

// 打印旋转矩阵
void print_rotation(const Eigen::Matrix3f &R, int index)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // 将接近0的值显示为0，接近±1的值显示为±1
            float value = R(i, j);
            if (std::abs(value) < 1e-6f)
                value = 0.0f;
            else if (std::abs(value - 1.0f) < 1e-6f)
                value = 1.0f;
            else if (std::abs(value + 1.0f) < 1e-6f)
                value = -1.0f;
        }
    }
}
// 可选：float 坐标的重载（会就近取整到像素中心）
inline bool drawDot(cv::Mat &img, const cv::Point2f &ptf,
                    const cv::Scalar &bgr = {0, 0, 255},
                    int radius = 3, int thickness = cv::FILLED, int lineType = cv::LINE_AA)
{
    return drawDot(img, cv::Point(cvRound(ptf.x), cvRound(ptf.y)), bgr, radius, thickness, lineType);
}

void print_polynomial(const std::vector<double> &poly)
{
    for (size_t i = 0; i < poly.size(); ++i)
    {
        std::cout << std::fixed << std::setprecision(17) << poly[i];
        if (i != poly.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

// ------------------- 多项式求值（支持泛型 T） -------------------
template <typename T>
T polyval(const std::vector<double> &coeffs, const T &x)
{
    T y = T(0);
    for (size_t i = 0; i < coeffs.size(); ++i)
    {
        y = y * x + T(coeffs[i]);
    }
    return y;
}

// ------------------- world2cam（支持自动求导） -------------------
template <typename T>
Eigen::Matrix<T, 2, 1> world2cam1(const Eigen::Matrix<T, 3, 1> &point3D, const OcamModel &model)
{
    // std::cout<<"SBSBSIN"<<point3D<<std::endl;
    // exit(0);
    T u = T(0), v = T(0);
    T xc = T(model.xc);
    T yc = T(model.yc);
    T c = T(model.c);
    T d = T(model.d);
    T e = T(model.e);
    T norm = sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1]);
    if (norm != T(0))
    {
        T invnorm = T(1) / norm;
        T theta = atan(point3D[2] / norm);
        T rho = polyval(model.invpol, theta);

        T x = point3D[0] * invnorm * rho;
        T y = point3D[1] * invnorm * rho;

        u = x * c + y * d + xc;
        v = x * e + y + yc;
    }
    else
    {
        u = xc;
        v = yc;
    }
    Eigen::Matrix<T, 2, 1> uv;
    uv << u, v;
    return uv;
}

// ------------------- 残差类 -------------------
class ExternalCaliOcam
{
public:
    ExternalCaliOcam(PnPData p, const OcamModel &model) : pd(p), ocam_model(model) {}

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residuals) const
    {

        Eigen::Quaternion<T> q_incre(_q[3], _q[0], _q[1], _q[2]);
        Eigen::Matrix<T, 3, 1> t_incre(_t[0], _t[1], _t[2]);

        Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));

        // std::cout<<p_l<<std::endl;
        Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
        T norm = p_c.norm();
        if (norm != T(0))
        {
            p_c /= norm; // Normalize the point to unit length
        }
        Eigen::Matrix<T, 2, 1> uv = world2cam1(p_c, ocam_model);
        // std::cout << "world2cam1OUT:" << uv<< std::endl;

        residuals[0] = uv[0] - T(pd.u);
        residuals[1] = uv[1] - T(pd.v);
        return true;
    }

    static ceres::CostFunction *Create(PnPData p, const OcamModel &model)
    {
        return (new ceres::AutoDiffCostFunction<ExternalCaliOcam, 2, 4, 3>(
            new ExternalCaliOcam(p, model)));
    }

private:
    PnPData pd;
    const OcamModel ocam_model;
};

// ------------------- 主函数 -------------------
int main()
{

    OcamModel ocam_model;
    // string intrinsic_path = "./intrinsic_pano.txt";
    string intrinsic_path = "intrinsic_pano.txt"; // 全景内参

    int error_threshold = 12;

    get_ocam_model(ocam_model, intrinsic_path); // OcamModel 参数文件

    std::vector<PnPData> pData;
    // getData("lidar_point.txt", "cam_point.txt", pData);
    string lidar_path = "./lidar_point.txt"; // 雷达标注点
    // string lidar_path = "../fake_lidar.txt"; // 雷达标注点

    string photo_path = "./cam_point.txt";                  // 全景标注点
    string extrinsic_path = "./extrinsic_pano.txt";         // 输出的外参路径
    string vaild_path = "/Users/losehu/Documents/20250813"; // 验证的全景图
    getData(lidar_path, photo_path, pData);
    // 生成所有90度旋转矩阵
    auto all_rotations = generate_all_90_degree_rotations();
    std::cout << "共生成 " << all_rotations.size() << " 个不同的90度旋转矩阵\n"
              << std::endl;
    double error_shit = 1e10;
    for (size_t i = 0; i < all_rotations.size(); ++i)
    {
        std::cout << "尝试旋转矩阵 " << i + 1 << "/" << all_rotations.size() << ":" << std::endl;
        Eigen::Matrix3f R_init_f = all_rotations[i];
        Eigen::Matrix3d R_init = R_init_f.cast<double>();

        Eigen::Quaterniond q(R_init);

        double ext[7] = {q.x(), q.y(), q.z(), q.w(), 0.0, 0.0, 0.0};

        Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
        Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);

        ceres::Manifold *q_parameterization = new ceres::EigenQuaternionManifold();
        ceres::Problem problem;
        problem.AddParameterBlock(ext, 4, q_parameterization);
        problem.AddParameterBlock(ext + 4, 3);

        for (auto &val : pData)
        {
            // std::cout << "Point: (" << val.x << ", " << val.y << ", " << val.z << "), Pixel: (" << val.u << ", " << val.v << ")\n";

            ceres::CostFunction *cost_function = ExternalCaliOcam::Create(val, ocam_model);
            problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
        }
    // problem.SetParameterBlockConstant(ext + 4);    // 锁死平移

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;

        float error[2] = {0, 0};
        Eigen::Matrix3d rot = m_q.toRotationMatrix();

        writeExt(extrinsic_path, rot, m_t);

        getUVError_pano(intrinsic_path, extrinsic_path, lidar_path, photo_path, error, error_threshold, vaild_path, false);

        if (error[0] + error[1] < error_shit)
        {
            cout << "find better:" << error[0] + error[1] << endl;
            writeExt(extrinsic_path, rot, m_t);
            error_shit = error[0] + error[1];
        }
    }

    cout << "最好的重投影误差是: " << error_shit << endl;
}


// int main() {
//     // -------- 基本路径 / 参数 --------
//     OcamModel ocam_model;
//     std::string intrinsic_path = "../a.txt";                  // 全景内参
//     std::string lidar_path     = "./lidar_point.txt";         // 雷达标注点
//     std::string photo_path     = "./cam_point.txt";           // 全景标注点
//     std::string extrinsic_path = "./extrinsic_pano.txt";      // 外参输出路径
//     std::string vaild_path     = "/Users/losehu/Documents/20250813";
//     int error_threshold = 12;

//     get_ocam_model(ocam_model, intrinsic_path);

//     std::vector<PnPData> pData;
//     getData(lidar_path, photo_path, pData);

//     // -------- 用你的外参作为 初始值 --------
//     Eigen::Matrix3d R_fixed;
//     R_fixed <<   -0.923021  ,0.372418  0.096633,
//                0.340026,  0.907107 -0.248070,
//               -0.180042, -0.196116 ,-0.963910;



//     Eigen::Vector3d t_fixed;
//     t_fixed << 307.028992,   -169.879883,  -330.180054;

//     // 如果你的雷达点 p_l 是以“米”为单位，而上述 t_fixed 是“毫米”，
//     // 请把下面这行取消注释，以统一单位到米：
//     // t_fixed /= 1000.0;

//     // Ceres 参数：四元数按 {x,y,z,w}，平移 {tx,ty,tz}
//     Eigen::Quaterniond q_init(R_fixed);
//     double ext[7] = { q_init.x(), q_init.y(), q_init.z(), q_init.w(),
//                       t_fixed.x(), t_fixed.y(), t_fixed.z() };

//     Eigen::Map<Eigen::Quaterniond> m_q(ext);       // 映射到四元数
//     Eigen::Map<Eigen::Vector3d>    m_t(ext + 4);   // 映射到平移

//     // -------- 搭建 Ceres 问题 --------
//     ceres::Problem problem;
//     ceres::Manifold* q_manifold = new ceres::EigenQuaternionManifold();
//     problem.AddParameterBlock(ext, 4, q_manifold);
//     problem.AddParameterBlock(ext + 4, 3);

//     for (const auto& val : pData) {
//         ceres::CostFunction* cost = ExternalCaliOcam::Create(val, ocam_model);
//         problem.AddResidualBlock(cost, nullptr, ext, ext + 4);
//     }

//     // （可选）如果你希望只优化 R 或只优化 t，可锁定另一个参数块：
//     // problem.SetParameterBlockConstant(ext);        // 锁死旋转
//     // problem.SetParameterBlockConstant(ext + 4);    // 锁死平移

//     // -------- 求解 --------
//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_SCHUR;
//     options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//     options.minimizer_progress_to_stdout = true;

//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
//     std::cout << summary.BriefReport() << std::endl;

//     // -------- 写出外参并评估误差 --------
//     Eigen::Matrix3d R_opt = m_q.toRotationMatrix();
//     writeExt(extrinsic_path, R_opt, m_t);

//     float error[2] = {0.f, 0.f};
//     getUVError_pano(intrinsic_path, extrinsic_path, lidar_path, photo_path,
//                     error, error_threshold, vaild_path, /*save_debug=*/false);

//     std::cout << "优化后重投影误差: u=" << error[0]
//               << "  v=" << error[1]
//               << "  sum=" << (error[0] + error[1]) << std::endl;

//     return 0;
// }
