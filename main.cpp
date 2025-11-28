#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <pcl/conversions.h>
#include <vector>
#include <string>
#include <ceres/manifold.h>  // 需要包含 Manifold 头文件

#include "common.h"
#include "result_verify.h"

using namespace std;

typedef pcl::PointXYZRGB PointType;
Eigen::Matrix3d inner;
string lidar_path, photo_path, intrinsic_path, extrinsic_path;
int error_threshold;
vector<float> init;

// void getParameters();
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


class external_cali {
public:
    external_cali(PnPData p, const std::vector<float> &dist) {
        pd = p;
        distortion = dist;
    }

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residuals) const {
        Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};

        // 雷达点
        Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));

        // 相机坐标系
        Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;

        // 归一化坐标
        T x = p_c[0] / p_c[2];
        T y = p_c[1] / p_c[2];

        // ---------------------
        // 畸变修正
        // ---------------------
        T r2 = x * x + y * y;
        T k1 = T(distortion[0]);
        T k2 = T(distortion[1]);
        T p1 = T(distortion[2]);
        T p2 = T(distortion[3]);
        T k3 = T(distortion[4]);

        T radial = T(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        T x_distorted = x * radial + T(2.0) * p1 * x * y + p2 * (r2 + T(2.0) * x * x);
        T y_distorted = y * radial + p1 * (r2 + T(2.0) * y * y) + T(2.0) * p2 * x * y;

        // 投影到像素
        Eigen::Matrix<T, 3, 1> p_2;
        p_2[0] = innerT(0,0) * x_distorted + innerT(0,1) * y_distorted + innerT(0,2);
        p_2[1] = innerT(1,0) * x_distorted + innerT(1,1) * y_distorted + innerT(1,2);
        p_2[2] = innerT(2,0) * x_distorted + innerT(2,1) * y_distorted + innerT(2,2);

        // 残差
        residuals[0] = p_2[0] / p_2[2] - T(pd.u);
        residuals[1] = p_2[1] / p_2[2] - T(pd.v);

        return true;
    }

    static ceres::CostFunction *Create(PnPData p, const std::vector<float> &dist) {
        return (new ceres::AutoDiffCostFunction<external_cali, 2, 4, 3>(
            new external_cali(p, dist)));
    }

private:
    PnPData pd;
    std::vector<float> distortion;
};


int main() {
    // getParameters();

    vector<PnPData> pData;
    string lidar_path="/Users/losehu/Documents/归档/Pano_Video/calib/sign/lidar_point_pianzhen.txt";
    string photo_path="/Users/losehu/Documents/归档/Pano_Video/calib/sign/photo_point_pianzhen.txt";
    string intrinsic_path="/Users/losehu/Documents/归档/Pano_Video/calib/sign/int_pianzhen.txt";
    string extrinsic_path="/Users/losehu/Documents/归档/Pano_Video/calib/result/extrinsic_pianzhen.txt";
    string show_path="./";///home/nanyuan/偏振外参数据";
    // string lidar_path="/home/nanyuan/calib/leida_hongwai.txt";
    // string photo_path="/home/nanyuan/calib/xiangji_hongwai.txt";
    // string intrinsic_path="/home/nanyuan/calib/int_hongwai.TXT";
    // string extrinsic_path="/home/nanyuan/calib/extrinsic_hongwai.txt";
    // string show_path="/home/nanyuan/红外外参数据";

    int error_threshold=12;
    init = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f,  0.0f, -1.0f, 0.0f,
        0.0f,  1.0f, 0.0f,  0.0f
    };
    // init = {
    //     0.0f, -1.0f, 0.0f, 0.0f,
    //     0.0f,  0.0f, -1.0f, 0.0f,
    //     1.0f,  0.0f, 0.0f,  0.0f
    // };

        vector<float> distortion;

    getDistortion(intrinsic_path, distortion);



    getData(lidar_path, photo_path, pData);
    vector<float> intrinsic;
    getIntrinsic(intrinsic_path, intrinsic);
    Eigen::Matrix4d extrin;
    inner << intrinsic[0], intrinsic[1], intrinsic[2],
    intrinsic[3], intrinsic[4], intrinsic[5],
    intrinsic[6], intrinsic[7], intrinsic[8];


    auto all_rotations = generate_all_90_degree_rotations();
    std::cout << "共生成 " << all_rotations.size() << " 个不同的90度旋转矩阵\n"
              << std::endl;
    double error_shit = 1e10;











    for (size_t i = 0; i < all_rotations.size(); ++i)
    {


  Eigen::Matrix3f R_init_f = all_rotations[i];
        Eigen::Matrix3d R_init = R_init_f.cast<double>();

        Eigen::Quaterniond q(R_init);

        double ext[7] = {q.x(), q.y(), q.z(), q.w(), 0.0, 0.0, 0.0};


    Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
    Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);
    ceres::Manifold* q_parameterization = new ceres::EigenQuaternionManifold();  // 新方式
    ceres::Problem problem;

    problem.AddParameterBlock(ext, 4, q_parameterization);
    problem.AddParameterBlock(ext + 4, 3);
  
    for(auto val : pData) {
        ceres::CostFunction *cost_function;
        cost_function = external_cali::Create(val,distortion);
        problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    Eigen::Matrix3d rot = m_q.toRotationMatrix();
    writeExt("tmp.txt", rot, m_t);
    cout << rot << endl;
    cout << m_t << endl;
        
        float error[2] = {0, 0};
//         rot<<-0.927944,-0.134850, 0.347470,
//  0.139088,-0.990197,-0.012841,
//  0.345795, 0.036413, 0.937603;
// m_t<<849.864990, 702.129028, -78.075012;

getUVError(intrinsic_path , "tmp.txt", lidar_path, photo_path, error, error_threshold,show_path,0);
        if (error[0] + error[1] < error_shit)
        {

            cout << "find better:" << error[0] + error[1] << endl;
            writeExt(extrinsic_path, rot, m_t);
            error_shit = error[0] + error[1];
            if (error_shit<138)
            
            {
            getUVError(intrinsic_path, extrinsic_path, lidar_path, photo_path, error, error_threshold,show_path,1);


            }
        }
    }

    cout << "最好的重投影误差是: " << error_shit << endl;
}
