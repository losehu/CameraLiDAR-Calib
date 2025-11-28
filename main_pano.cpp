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

        Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
        T norm = p_c.norm();
        if (norm != T(0))
        {
            p_c /= norm; // Normalize the point to unit length
        }
        // 建议在函数顶部统一成 T 类型
        const T PI = T(M_PI); // 或 C++20: std::numbers::pi_v<T>
        const T TWO_PI = T(2.0) * PI;
        const T W = T(7680); // 画幅宽
        const T H = T(3840); // 画幅高

        // … p_c 已单位化 …

        // 经纬（与你当前定义一致）
        const T lon_rad = ceres::atan2(p_c.y(), p_c.x());
        const T xy_norm = ceres::sqrt(p_c.x() * p_c.x() + p_c.y() * p_c.y());
        const T lat_rad = ceres::atan2(p_c.z(), xy_norm);

        // 投影到像素（保持你原来的公式，但全用 T 常量）
        const T u = (PI - lon_rad) * W / (T(2.0) * PI);
        const T v = (T(0.5) * PI - lat_rad) * H / PI;
        residuals[0] = u - T(pd.u);
        residuals[1] = v - T(pd.v);

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

    int error_threshold = 12;

    std::vector<PnPData> pData;
    // getData("lidar_point.txt", "cam_point.txt", pData);
    string lidar_path = "/Users/losehu/Documents/归档/Pano_Video/calib/sign/913_LIDAr.txt"; // 雷达标注点
    // string lidar_path = "../fake_lidar.txt"; // 雷达标注点

    string photo_path = "/Users/losehu/Documents/归档/Pano_Video/calib/sign/913_img.txt";        // 全景标注点
    string extrinsic_path = "./extrinsic_yuyan_913.txt";    // 输出的外参路径
    string vaild_path = "./"; // 验证的全景图
    getData(lidar_path, photo_path, pData);
    // 生成所有90度旋转矩阵
    auto all_rotations = generate_all_90_degree_rotations();
    std::cout << "共生成 " << all_rotations.size() << " 个不同的90度旋转矩阵\n"
              << std::endl;
    double error_shit = 1e10;
    for (size_t i = 0; i < all_rotations.size(); ++i)
    {
        //     Eigen::Matrix3d R_fix;
        //     R_fix  << -0.709058 , -0.245465 , 0.661047 ,
        // 0.235907 , -0.966013,  -0.105667  ,
        // 0.664518,  0.081022  ,0.742867  ;
        //     Eigen::Vector3d t_fix( 812.51,  811.595,  -1085.49);
        //     Eigen::Quaterniond q_fix(R_fix);
        //     double ext[7] = { q_fix.x(), q_fix.y(), q_fix.z(), q_fix.w(),
        //                       t_fix.x(), t_fix.y(), t_fix.z() };

        // std::cout << "尝试旋转矩阵 " << i + 1 << "/" << all_rotations.size() << ":" << std::endl;
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

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;

        float error[2] = {0, 0};
        Eigen::Matrix3d rot = m_q.toRotationMatrix();
//         rot<<-0.927944,-0.134850, 0.347470,
//  0.139088,-0.990197,-0.012841,
//  0.345795, 0.036413, 0.937603;
// m_t<<849.864990, 702.129028, -78.075012;
        writeExt("./tmp.txt", rot, m_t);

        getUVError_yuyan("./tmp.txt", lidar_path, photo_path, error, error_threshold, vaild_path, 0);

        if (error[0] + error[1] < error_shit)
        {

            cout << "find better:" << error[0] + error[1] << endl;
            writeExt(extrinsic_path, rot, m_t);
            
            error_shit = error[0] + error[1];
            if (error_shit<64)
            {
                        // getUVError_yuyan("./tmp.txt", lidar_path, photo_path, error, error_threshold, vaild_path, 1);

            }
        }
    }

    cout << "最好的重投影误差是: " << error_shit << endl;
}
