// external_cali_ocam_opt_intrinsics.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <cctype>

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace fs = std::filesystem;

// ========================= file utils (MUST be before list_* functions) =========================
static inline std::string lower_ext(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c){ return (unsigned char)std::tolower(c); });
    return e;
}

// 提取文件名（不含扩展名）的“纯数字”作为排序key；不是纯数字就给一个很大的key排到最后
static inline int numeric_key(const fs::path& p) {
    const std::string stem = p.stem().string(); // "0001" 或 "frame_0001"
    // 1) 先尝试整串就是数字
    try {
        size_t pos = 0;
        int v = std::stoi(stem, &pos);
        if (pos == stem.size()) return v;
    } catch(...) {}

    // 2) 再尝试提取最后一段连续数字（适配 frame_0001 / img-12 这种命名）
    int best = std::numeric_limits<int>::max();
    for (int i = (int)stem.size() - 1; i >= 0; --i) {
        if (std::isdigit((unsigned char)stem[i])) {
            int j = i;
            while (j >= 0 && std::isdigit((unsigned char)stem[j])) j--;
            // [j+1, i] 是一段数字
            try {
                int v = std::stoi(stem.substr(j+1, i-j));
                best = v;
            } catch(...) {}
            break;
        }
    }
    return best;
}

static inline bool numeric_less_path(const fs::path& a, const fs::path& b) {
    int ka = numeric_key(a), kb = numeric_key(b);
    if (ka != kb) return ka < kb;
    return a.filename().string() < b.filename().string();
}

static std::vector<std::string> list_images_sorted_by_number(const std::string& dir) {
    std::vector<fs::path> imgs;
    for (auto& e: fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        fs::path p = e.path();
        std::string ext = lower_ext(p);
        if (ext==".jpg" || ext==".jpeg" || ext==".png" || ext==".bmp" || ext==".tiff") {
            imgs.push_back(fs::absolute(p));
        }
    }
    std::sort(imgs.begin(), imgs.end(), numeric_less_path);
    std::vector<std::string> out;
    out.reserve(imgs.size());
    for (auto& p: imgs) out.push_back(p.string());
    return out;
}

static std::vector<std::string> list_pcd_sorted_by_number(const std::string& dir) {
    std::vector<fs::path> files;
    for (auto& e: fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        fs::path p = e.path();
        std::string ext = lower_ext(p);
        if (ext==".pcd") files.push_back(fs::absolute(p));
    }
    std::sort(files.begin(), files.end(), numeric_less_path);
    std::vector<std::string> out;
    out.reserve(files.size());
    for (auto& p: files) out.push_back(p.string());
    return out;
}

// ========================= config utils (simple YAML: key: value) =========================
static inline std::string trim_copy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

static std::string strip_inline_comment(const std::string& line) {
    bool in_single = false;
    bool in_double = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '\'' && !in_double) {
            in_single = !in_single;
        } else if (c == '"' && !in_single) {
            in_double = !in_double;
        } else if (c == '#' && !in_single && !in_double) {
            return line.substr(0, i);
        }
    }
    return line;
}

static inline std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2) {
        char first = s.front();
        char last = s.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return s.substr(1, s.size() - 2);
        }
    }
    return s;
}

static std::unordered_map<std::string, std::string> parse_yaml_kv(const fs::path& path) {
    std::unordered_map<std::string, std::string> out;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Failed to open config: " << path << "\n";
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        line = strip_inline_comment(line);
        line = trim_copy(line);
        if (line.empty()) continue;
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::string key = trim_copy(line.substr(0, pos));
        std::string val = trim_copy(line.substr(pos + 1));
        if (key.empty()) continue;
        out[key] = strip_quotes(val);
    }
    return out;
}

static std::unordered_map<std::string, std::string> load_config_kv(const fs::path& path) {
    auto data = parse_yaml_kv(path);
    auto it = data.find("config_path");
    if (it == data.end() || it->second.empty()) return data;
    fs::path nested = it->second;
    if (nested.is_relative()) {
        fs::path from_parent = path.parent_path() / nested;
        if (fs::exists(from_parent)) {
            nested = from_parent;
        } else {
            fs::path from_cwd = fs::path(nested);
            if (fs::exists(from_cwd)) nested = from_cwd;
        }
    }
    return parse_yaml_kv(nested);
}

static std::string get_required_value(const std::unordered_map<std::string, std::string>& cfg,
                                      const std::string& key) {
    auto it = cfg.find(key);
    if (it == cfg.end() || it->second.empty()) {
        std::cerr << "Config missing '" << key << "'\n";
        std::exit(1);
    }
    return it->second;
}

static std::string get_optional_value(const std::unordered_map<std::string, std::string>& cfg,
                                      const std::string& key,
                                      const std::string& fallback) {
    auto it = cfg.find(key);
    if (it == cfg.end() || it->second.empty()) return fallback;
    return it->second;
}

// ========================= Data =========================
struct PnPData {
    double x, y, z;   // LiDAR point
    double u, v;      // image pixel (u=col, v=row)
};

// ========================= Ocam Model =========================
struct OcamModel {
    std::vector<double> pol;
    std::vector<double> invpol;
    double xc = 0.0, yc = 0.0; // xc = column center, yc = row center
    double c = 1.0, d = 0.0, e = 0.0;
    int width = 0, height = 0;
    int length_invpol = 0;
    int length_pol = 0;
};

static inline double str2double(const std::string& s) {
    std::stringstream ss(s);
    double v;
    ss >> v;
    return v;
}
static inline std::string int2str(int x) { return std::to_string(x); }
static inline bool has_space(const std::string& s) {
    return s.find(' ') != std::string::npos || s.find('\t') != std::string::npos;
}

// ========================= Read ocam model =========================
// IMPORTANT: center line is "row column" => yc=row, xc=col.
void get_ocam_model(OcamModel& model, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening ocam file: " << filename << "\n";
        std::exit(1);
    }
    std::string line;

    std::getline(file, line);
    std::getline(file, line);

    std::getline(file, line);
    {
        std::istringstream iss(line);
        iss >> model.length_pol;
        model.pol.resize(model.length_pol);
        for (int i = 0; i < model.length_pol; ++i) iss >> model.pol[i];
    }

    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    std::getline(file, line);
    {
        std::istringstream iss(line);
        iss >> model.length_invpol;
        model.invpol.resize(model.length_invpol);
        for (int i = 0; i < model.length_invpol; ++i) iss >> model.invpol[i];
    }

    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    std::getline(file, line);
    {
        std::istringstream iss(line);
        double row, col;
        iss >> row >> col;
        model.yc = row;
        model.xc = col;
    }

    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    std::getline(file, line);
    {
        std::istringstream iss(line);
        iss >> model.c >> model.d >> model.e;
    }

    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    std::getline(file, line);
    {
        std::istringstream iss(line);
        iss >> model.height >> model.width;
    }

    std::cout << std::fixed << std::setprecision(6)
              << "[ocam] xc(col)=" << model.xc
              << " yc(row)=" << model.yc
              << " c=" << model.c << " d=" << model.d << " e=" << model.e
              << " H=" << model.height << " W=" << model.width
              << " invpol=" << model.invpol.size()
              << " pol=" << model.pol.size() << "\n";
}

// ========================= Polyval (MATLAB order) =========================
template <typename T>
T polyval_matlab(const std::vector<double>& coeffs, const T& x) {
    T y = T(0);
    for (size_t i = 0; i < coeffs.size(); ++i) y = y * x + T(coeffs[i]);
    return y;
}

// ========================= world2cam =========================
static inline cv::Point2d world2cam_cv(const cv::Vec3d& p, const OcamModel& m) {
    double n = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    cv::Vec3d pn = (n > 0) ? cv::Vec3d(p[0]/n, p[1]/n, p[2]/n) : cv::Vec3d(0,0,1);

    double X = pn[0], Y = pn[1], Z = pn[2];
    double norm_xy = std::sqrt(X*X + Y*Y);

    double u, v;
    if (norm_xy > 1e-12) {
        double theta = std::atan2(Z, norm_xy);
        double rho   = polyval_matlab<double>(m.invpol, theta);
        double invn  = 1.0 / norm_xy;

        double x = X * invn * rho;
        double y = Y * invn * rho;

        u = x * m.c + y * m.d + m.xc;
        v = x * m.e + y + m.yc;
    } else {
        u = m.xc;
        v = m.yc;
    }
    return {u, v};
}

static void create_equirect_remap(const OcamModel& m,
                                  int out_w,
                                  int out_h,
                                  double lon_min_deg,
                                  double lon_max_deg,
                                  double lat_min_deg,
                                  double lat_max_deg,
                                  cv::Mat& mapx,
                                  cv::Mat& mapy) {
    mapx = cv::Mat(out_h, out_w, CV_32FC1);
    mapy = cv::Mat(out_h, out_w, CV_32FC1);
    const double lon_range = lon_max_deg - lon_min_deg;
    const double lat_range = lat_max_deg - lat_min_deg;
    for (int y = 0; y < out_h; ++y) {
        const double v =  (double)y / (double)(out_h) ;
        const double lat_deg = -lat_max_deg + v * lat_range;
        const double lat = lat_deg * M_PI / 180.0;
        const double cos_lat = std::cos(lat);
        const double sin_lat = std::sin(lat);

        for (int x = 0; x < out_w; ++x) {
            const double u =  (double)x / (double)(out_w);
            const double lon_deg = -lon_min_deg - u * lon_range;
            const double lon = lon_deg * M_PI / 180.0;

            cv::Vec3d dir(cos_lat * std::cos(lon),
                          cos_lat * std::sin(lon),
                          sin_lat);
            cv::Point2d src = world2cam_cv(dir, m);
            mapx.at<float>(y, x) = static_cast<float>(src.x);
            mapy.at<float>(y, x) = static_cast<float>(src.y);
        }
    }
}

static inline bool project_to_equirect_uv(const Eigen::Vector3d& pc,
                                          int out_w,
                                          int out_h,
                                          double lon_min_deg,
                                          double lon_max_deg,
                                          double lat_min_deg,
                                          double lat_max_deg,
                                          int& u,
                                          int& v) {
    const double norm = pc.norm();
    if (norm <= 0.0) return false;

    const double x = pc.x() / norm;
    const double y = pc.y() / norm;
    const double z = pc.z() / norm;

    const double lon = std::atan2(y, x) * 180.0 / M_PI;
    const double lat = std::asin(z) * 180.0 / M_PI;

    const double lon_low = -lon_max_deg;
    const double lon_high = -lon_min_deg;
    const double lat_low = -lat_max_deg;
    const double lat_high = -lat_min_deg;
    if (lon < lon_low || lon > lon_high ||
        lat < lat_low || lat > lat_high) {
        return false;
    }

    const double lon_range = lon_max_deg - lon_min_deg;
    const double lat_range = lat_max_deg - lat_min_deg;
    const double uf = (-lon_min_deg - lon) / lon_range * out_w;
    const double vf = (lat + lat_max_deg) / lat_range * out_h;

    int ui = (int)std::lround(uf);
    int vi = (int)std::lround(vf);
    if (ui == out_w) ui = out_w - 1;
    if (vi == out_h) vi = out_h - 1;
    if ((unsigned)ui >= (unsigned)out_w || (unsigned)vi >= (unsigned)out_h) {
        return false;
    }
    u = ui;
    v = vi;
    return true;
}

template <typename T>
static inline Eigen::Matrix<T,2,1> world2cam_eigen(const Eigen::Matrix<T,3,1>& p, const OcamModel& m) {
    T X = p[0], Y = p[1], Z = p[2];
    T norm_xy = sqrt(X*X + Y*Y);

    T u = T(m.xc), v = T(m.yc);
    if (norm_xy != T(0)) {
        T theta = atan2(Z, norm_xy);
        T rho   = polyval_matlab<T>(m.invpol, theta);
        T invn  = T(1) / norm_xy;

        T x = X * invn * rho;
        T y = Y * invn * rho;

        u = x * T(m.c) + y * T(m.d) + T(m.xc);
        v = x * T(m.e) + y + T(m.yc);
    }
    Eigen::Matrix<T,2,1> uv;
    uv << u, v;
    return uv;
}

// ========================= Read raw pairs =========================
static void read_pairs_raw(const std::string& lidar_path,
                           const std::string& photo_path,
                           std::vector<PnPData>& out_raw) {
    std::ifstream inL(lidar_path), inP(photo_path);
    if (!inL.is_open() || !inP.is_open()) {
        std::cerr << "Cannot open lidar/photo file.\n";
        std::exit(1);
    }

    std::string lineL, lineP;
    int count = 0;

    while (std::getline(inL, lineL) && std::getline(inP, lineP)) {
        bool sL = has_space(lineL);
        bool sP = has_space(lineP);

        if (sL && sP) {
            std::stringstream sl(lineL), sp(lineP);
            std::string str;
            PnPData p{};

            sl >> str; p.x = str2double(str);
            sl >> str; p.y = str2double(str);
            sl >> str; p.z = str2double(str);

            sp >> str; p.u = str2double(str);
            sp >> str; p.v = str2double(str);

            out_raw.push_back(p);
            ++count;
        } else if (lineL.empty() && lineP.empty()) {
            break;
        } else if (sL != sP) {
            std::cerr << "Lidar/photo not aligned!\n";
            std::cerr << "L: " << lineL << "\nP: " << lineP << "\n";
            std::exit(1);
        }
    }
    std::cout << "Loaded raw pairs: " << count << "\n";
}

// ========================= Init rotations =========================
static Eigen::Matrix3f rotation_x(int deg) {
    float r = deg * float(M_PI) / 180.f;
    float c = std::cos(r), s = std::sin(r);
    Eigen::Matrix3f R;
    R << 1,0,0,  0,c,-s,  0,s,c;
    return R;
}
static Eigen::Matrix3f rotation_y(int deg) {
    float r = deg * float(M_PI) / 180.f;
    float c = std::cos(r), s = std::sin(r);
    Eigen::Matrix3f R;
    R << c,0,s,  0,1,0,  -s,0,c;
    return R;
}
static Eigen::Matrix3f rotation_z(int deg) {
    float r = deg * float(M_PI) / 180.f;
    float c = std::cos(r), s = std::sin(r);
    Eigen::Matrix3f R;
    R << c,-s,0,  s,c,0,  0,0,1;
    return R;
}
static bool mat_equal(const Eigen::Matrix3f& a, const Eigen::Matrix3f& b, float tol=1e-6f) {
    return (a-b).norm() < tol;
}
static std::vector<Eigen::Matrix3f> generate_all_90_degree_rotations() {
    std::vector<int> ang = {0,90,180,270};
    std::vector<Eigen::Matrix3f> all, uniq;
    for (int x: ang) for (int y: ang) for (int z: ang) {
        Eigen::Matrix3f R = rotation_z(z) * rotation_y(y) * rotation_x(x);
        all.push_back(R);
    }
    for (auto& R: all) {
        bool dup=false;
        for (auto& U: uniq) { if (mat_equal(R,U)) { dup=true; break; } }
        if (!dup) uniq.push_back(R);
    }
    return uniq;
}

// ========================= flip variants =========================
struct Flip {
    int sx, sy, sz; // each in {-1,+1}
    Eigen::Matrix3d M() const {
        Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
        A(0,0) = (double)sx;
        A(1,1) = (double)sy;
        A(2,2) = (double)sz;
        return A;
    }
};

static std::vector<Flip> all_flips() {
    std::vector<Flip> v;
    // 你原始代码这里强制只用 -1（如果你想全枚举，改成 {-1,+1}）
    int s[2] = {-1,1};
    for (int sx: s) for (int sy: s) for (int sz: s) v.push_back({sx,sy,sz});
    return v;
}

enum class Mode { Forward, Inverse }; // Forward: pc=R*pl+t, Inverse: pc=R^T*(pl-t)

// ========================= Residual (fixed intrinsics) with flip + mode =========================
class ExternalCaliOcam {
public:
    ExternalCaliOcam(const PnPData& p, const OcamModel& model, const Eigen::Matrix3d& A, Mode mode)
        : pd(p), m(model), A_(A), mode_(mode) {}

    template <typename T>
    bool operator()(const T* const _q, const T* const _t, T* residuals) const {
        Eigen::Quaternion<T> q(_q[3], _q[0], _q[1], _q[2]);
        Eigen::Matrix<T,3,1> t(_t[0], _t[1], _t[2]);

        Eigen::Matrix<T,3,1> pl(T(pd.x), T(pd.y), T(pd.z));
        Eigen::Matrix<T,3,1> pc;

        if (mode_ == Mode::Forward) {
            pc = q.toRotationMatrix()*pl + t;
        } else {
            pc = q.toRotationMatrix().transpose() * (pl - t);
        }

        Eigen::Matrix<T,3,3> A = A_.cast<T>();
        pc = A * pc;

        // normalize (direction)
        T n = pc.norm();
        if (n != T(0)) pc /= n;

        Eigen::Matrix<T,2,1> uv = world2cam_eigen(pc, m);
        residuals[0] = uv[0] - T(pd.u);
        residuals[1] = uv[1] - T(pd.v);
        return true;
    }

    static ceres::CostFunction* Create(const PnPData& p, const OcamModel& model, const Eigen::Matrix3d& A, Mode mode) {
        return new ceres::AutoDiffCostFunction<ExternalCaliOcam, 2, 4, 3>(
            new ExternalCaliOcam(p, model, A, mode)
        );
    }

private:
    PnPData pd;
    OcamModel m;
    Eigen::Matrix3d A_;
    Mode mode_;
};

// ========================= NEW: Residual (opt intrinsics) using DynamicAutoDiff =========================
template <typename T>
static inline T polyval_runtime(const T* coeffs, int n, const T& x) {
    // MATLAB order: y = (((a0)*x + a1)*x + a2)...
    T y = T(0);
    for (int i = 0; i < n; ++i) y = y * x + coeffs[i];
    return y;
}

struct ExternalCaliOcamDyn {
    ExternalCaliOcamDyn(const PnPData& p, int invpol_len, const Eigen::Matrix3d& A, Mode mode)
        : pd(p), K(invpol_len), A_(A), mode_(mode) {}

    template <typename T>
    bool operator()(T const* const* params, T* residuals) const {
        // params[0] = q(4), params[1] = t(3), params[2] = invpol(K), params[3] = aff(5)
        const T* qptr = params[0];
        const T* tptr = params[1];
        const T* inv  = params[2];
        const T* aff  = params[3];

        Eigen::Quaternion<T> q(qptr[3], qptr[0], qptr[1], qptr[2]);
        Eigen::Matrix<T,3,1> t(tptr[0], tptr[1], tptr[2]);

        Eigen::Matrix<T,3,1> pl(T(pd.x), T(pd.y), T(pd.z));
        Eigen::Matrix<T,3,1> pc;

        if (mode_ == Mode::Forward) pc = q.toRotationMatrix()*pl + t;
        else                        pc = q.toRotationMatrix().transpose() * (pl - t);

        pc = A_.cast<T>() * pc;

        // ocam projection with variable intrinsics
        const T X = pc[0], Y = pc[1], Z = pc[2];
        const T norm_xy = sqrt(X*X + Y*Y);

        // aff = [xc, yc, c, d, e]
        T u = aff[0]; // xc
        T v = aff[1]; // yc

        if (norm_xy != T(0)) {
            const T theta = atan2(Z, norm_xy);
            const T rho   = polyval_runtime(inv, K, theta);
            const T invn  = T(1) / norm_xy;

            const T x = X * invn * rho;
            const T y = Y * invn * rho;

            const T c = aff[2], d = aff[3], e = aff[4];
            u = x * c + y * d + aff[0];
            v = x * e + y + aff[1];
        }

        residuals[0] = u - T(pd.u);
        residuals[1] = v - T(pd.v);
        return true;
    }

    static ceres::CostFunction* Create(const PnPData& p, int invpol_len,
                                       const Eigen::Matrix3d& A, Mode mode) {
        auto* f = new ExternalCaliOcamDyn(p, invpol_len, A, mode);
        auto* cost = new ceres::DynamicAutoDiffCostFunction<ExternalCaliOcamDyn>(f);
        cost->AddParameterBlock(4);            // q
        cost->AddParameterBlock(3);            // t
        cost->AddParameterBlock(invpol_len);   // invpol
        cost->AddParameterBlock(5);            // xc,yc,c,d,e
        cost->SetNumResiduals(2);
        return cost;
    }

    PnPData pd;
    int K;
    Eigen::Matrix3d A_;
    Mode mode_;
};

// ========================= [INSURANCE] Translation prior =========================
struct TranslationPrior {
    TranslationPrior(double w, double tx0, double ty0, double tz0)
        : w_(w), tx0_(tx0), ty0_(ty0), tz0_(tz0) {}

    template <typename T>
    bool operator()(const T* const _t, T* residuals) const {
        const T w = T(w_);
        residuals[0] = w * (_t[0] - T(tx0_));
        residuals[1] = w * (_t[1] - T(ty0_));
        residuals[2] = w * (_t[2] - T(tz0_));
        return true;
    }

    static ceres::CostFunction* Create(double w, double tx0=0.0, double ty0=0.0, double tz0=0.0) {
        return new ceres::AutoDiffCostFunction<TranslationPrior, 3, 3>(
            new TranslationPrior(w, tx0, ty0, tz0)
        );
    }

    double w_;
    double tx0_, ty0_, tz0_;
};

// ========================= NEW: Intrinsics priors (strongly recommended) =========================
struct InvpolPrior {
    InvpolPrior(const std::vector<double>& inv0, double w) : inv0_(inv0), w_(w) {}

    template <typename T>
    bool operator()(T const* const* params, T* residuals) const {
        // params[0] is the invpol parameter block (length = inv0_.size())
        const T* inv = params[0];
        const T w = T(w_);
        for (int i = 0; i < (int)inv0_.size(); ++i) {
            residuals[i] = w * (inv[i] - T(inv0_[i]));
        }
        return true;
    }

    static ceres::CostFunction* Create(const std::vector<double>& inv0, double w) {
        auto* functor = new InvpolPrior(inv0, w);
        auto* cost = new ceres::DynamicAutoDiffCostFunction<InvpolPrior>(functor);
        cost->AddParameterBlock((int)inv0.size());   // one block: invpol
        cost->SetNumResiduals((int)inv0.size());     // residuals: same length
        return cost;
    }

    std::vector<double> inv0_;
    double w_;
};


struct AffinePrior {
    AffinePrior(double w, const double* a0) : w_(w) { for(int i=0;i<5;i++) a0_[i]=a0[i]; }

    template <typename T>
    bool operator()(const T* const a, T* residuals) const {
        for(int i=0;i<5;i++) residuals[i] = T(w_) * (a[i] - T(a0_[i]));
        return true;
    }

    static ceres::CostFunction* Create(double w, const double* a0) {
        return new ceres::AutoDiffCostFunction<AffinePrior, 5, 5>(new AffinePrior(w, a0));
    }

    double w_;
    double a0_[5];
};

// ========================= projection + error + viz =========================
static inline void project_RT(float uv[2], const OcamModel& m,
                              const Eigen::Matrix3d& R,
                              const Eigen::Vector3d& t,
                              const Eigen::Matrix3d& A,
                              Mode mode,
                              double x, double y, double z)
{
    Eigen::Vector3d pl(x,y,z);
    Eigen::Vector3d pc;

    if (mode == Mode::Forward) pc = R*pl + t;
    else                      pc = R.transpose() * (pl - t);

    pc = A * pc;

    cv::Point2d pix = world2cam_cv(cv::Vec3d(pc.x(), pc.y(), pc.z()), m);
    uv[0] = (float)pix.x;
    uv[1] = (float)pix.y;
}

static void compute_error_RT(const OcamModel& m,
                             const Eigen::Matrix3d& R,
                             const Eigen::Vector3d& t,
                             const Eigen::Matrix3d& A,
                             Mode mode,
                             const std::vector<PnPData>& pts,
                             float error[2])
{
    double sumU=0, sumV=0;
    int n=0;
    for (auto& p: pts) {
        float uv[2];
        project_RT(uv, m, R, t, A, mode, p.x, p.y, p.z);
        sumU += std::abs(p.u - uv[0]);
        sumV += std::abs(p.v - uv[1]);
        n++;
    }
    if (n==0) { error[0]=error[1]=1e9; return; }
    error[0] = (float)(sumU/n);
    error[1] = (float)(sumV/n);
}

static inline void drawCircleAt(cv::Mat& img, int x, int y, int r, const cv::Scalar& bgr, int thickness) {
    if (img.empty()) return;
    cv::circle(img, cv::Point(x,y), r, bgr, thickness, cv::LINE_AA);
}

static void visualize_RT(const OcamModel& m,
                         const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& t,
                         const Eigen::Matrix3d& A,
                         Mode mode,
                         const std::vector<PnPData>& pts,
                         const std::string& valid_path)
{
    auto files = list_images_sorted_by_number(valid_path);
    if (files.empty()) {
        std::cerr << "No images found in " << valid_path << "\n";
        return;
    }

    cv::Mat img;
    for (int i=0; i<(int)pts.size(); ++i) {
        if (i % 4 == 0) {
            int idx = i/4;
            if (idx >= (int)files.size()) break;
            img = cv::imread(files[idx]);
            if (img.empty()) continue;
        }

        float uvth[2];
        project_RT(uvth, m, R, t, A, mode, pts[i].x, pts[i].y, pts[i].z);

        // 理论点：紫色
        drawCircleAt(img, (int)std::lround(uvth[0]), (int)std::lround(uvth[1]), 14, cv::Scalar(255,0,255), -1);

        // 数据点：按组内索引上色
        cv::Scalar cmap[4] = {
            cv::Scalar(0,0,255),
            cv::Scalar(0,255,0),
            cv::Scalar(255,0,0),
            cv::Scalar(255,255,0)
        };
        drawCircleAt(img, (int)std::lround(pts[i].u), (int)std::lround(pts[i].v), 14, cmap[i%4], -1);

        if ((i+1) % 4 == 0) {
            cv::imshow(int2str(i/4), img);
            cv::waitKey(0);
        }
    }
}

static void visualize_pcd_sequence(const OcamModel& m,
                                   const Eigen::Matrix3d& R_L2C,
                                   const Eigen::Vector3d& t_L2C,
                                   const Eigen::Matrix3d& A,
                                   const std::string& pcd_dir,
                                   const std::string& img_dir,
                                   bool pcd_m_to_mm = true,
                                   int step = 3,
                                   int radius = 1,
                                   double min_depth = 1e-6)
{

    auto imgs = list_images_sorted_by_number(img_dir);
    auto pcds = list_pcd_sorted_by_number(pcd_dir);

    if (imgs.empty() || pcds.empty()) {
        std::cerr << "No images or pcds found.\n";
        std::cerr << "img_dir=" << img_dir << "  pcd_dir=" << pcd_dir << "\n";
        return;
    }

    size_t N = std::min(imgs.size(), pcds.size());
    if (imgs.size() != pcds.size()) {
        std::cerr << "[Warn] image count=" << imgs.size()
                  << " pcd count=" << pcds.size()
                  << " => process first " << N << " pairs\n";
    }

    const double lon_min_deg = -180.0;
    const double lon_max_deg = 180.0;
    const double lat_min_deg = -6.0;
    const double lat_max_deg = 39.0;
    const int eq_height = 512;
    const int eq_width = (int)std::lround(eq_height *
        (lon_max_deg - lon_min_deg) / (lat_max_deg - lat_min_deg));

    cv::Mat mapx, mapy;
    create_equirect_remap(m, eq_width, eq_height,
                          lon_min_deg, lon_max_deg,
                          lat_min_deg, lat_max_deg,
                          mapx, mapy);

    std::cout << "Projecting ALL pcd points (equirect): frames=" << N
              << " step=" << step
              << " pcd_m_to_mm=" << (pcd_m_to_mm?"true":"false") << "\n";

    for (size_t i = 0; i < N; ++i) {
        cv::Mat src = cv::imread(imgs[i]);
        if (src.empty()) {
            std::cerr << "Failed to load image: " << imgs[i] << "\n";
            continue;
        }
        cv::Mat pano;
        cv::remap(src, pano, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                  cv::Scalar(0,0,0));

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcds[i], *cloud) != 0) {
            std::cerr << "Failed to load pcd: " << pcds[i] << "\n";
            continue;
        }

        int drawn = 0;
        const int s = std::max(1, step);

        std::vector<double> depths;
        depths.reserve(cloud->points.size());
        for (const auto &pt : cloud->points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            depths.push_back(std::sqrt((double)pt.x*pt.x + (double)pt.y*pt.y + (double)pt.z*pt.z));
        }
        if (depths.empty()) {
            std::cerr << "No valid points in cloud\n";
            continue;
        }
        std::vector<double> depths_sorted = depths;
        std::sort(depths_sorted.begin(), depths_sorted.end());
        size_t idx90 = (size_t)std::floor(0.9 * depths_sorted.size());
        if (idx90 >= depths_sorted.size()) idx90 = depths_sorted.size() - 1;
        double p90 = depths_sorted[idx90];
        if (p90 <= 0) p90 = depths_sorted.back();

        for (size_t k = 0; k < cloud->points.size(); k += s) {
            const auto& pt = cloud->points[k];
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

            double depth = std::sqrt((double)pt.x*pt.x + (double)pt.y*pt.y + (double)pt.z*pt.z);

            Eigen::Vector3d pl(pt.x, pt.y, pt.z);
            if (pcd_m_to_mm) pl *= 1000.0;

            Eigen::Vector3d pc = R_L2C * pl + t_L2C;
            pc = A * pc;

            if (pc.norm() < min_depth) continue;

            int u = 0, v = 0;
            if (!project_to_equirect_uv(pc, pano.cols, pano.rows,
                                        lon_min_deg, lon_max_deg,
                                        lat_min_deg, lat_max_deg,
                                        u, v)) {
                continue;
            }

            float tnorm = static_cast<float>(depth / p90);
            if (tnorm > 1.0f) tnorm = 1.0f;
            float hue = (1.0f - tnorm) * 270.0f;

            cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(static_cast<int>(hue/2.0f + 0.5f), 255, 255));
            cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            cv::Vec3b col = bgr.at<cv::Vec3b>(0,0);

            cv::circle(pano, cv::Point(u, v), radius,
                       cv::Scalar(col[0], col[1], col[2]), -1, cv::LINE_AA);
            drawn++;
        }

        fs::path ip(imgs[i]);

        cv::imshow("show", pano);
        cv::waitKey(0);


    }
}

// ========================= write extrinsic (Lidar->Camera) =========================
static void writeExt_L2C(const std::string& path, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Cannot write extrinsic: " << path << "\n";
        return;
    }
    out << "extrinsic\n";
    out << R(0,0) << " " << R(0,1) << " " << R(0,2) << " " << t.x() << "\n";
    out << R(1,0) << " " << R(1,1) << " " << R(1,2) << " " << t.y() << "\n";
    out << R(2,0) << " " << R(2,1) << " " << R(2,2) << " " << t.z() << "\n";
    out << "0 0 0 1\n";
}

// ========================= write refined intrinsics (simple text) =========================
static void write_ocam_intrinsics_full(const std::string& path, const OcamModel& m) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Cannot write intrinsics: " << path << "\n";
        return;
    }
    out << std::setprecision(16);
    out << "#polynomial coefficients for the DIRECT mapping function (ocam_model.ss in MATLAB). These are used by cam2world\n\n";
    out << m.pol.size();
    for (auto v : m.pol) out << " " << v;
    out << "\n\n";

    out << "#polynomial coefficients for the inverse mapping function (ocam_model.invpol in MATLAB). These are used by world2cam\n\n";
    out << m.invpol.size();
    for (auto v : m.invpol) out << " " << v;
    out << "\n\n";

    out << "#center: \"row\" and \"column\", starting from 0 (C convention)\n\n";
    out << m.yc << " " << m.xc << "\n\n";

    out << "#affine parameters \"c\", \"d\", \"e\"\n\n";
    out << m.c << " " << m.d << " " << m.e << "\n\n";

    out << "#image size: \"height\" and \"width\"\n\n";
    out << m.height << " " << m.width << "\n";
}

// ========================= main =========================
int main() {
    // ========================= knobs =========================
    const double huber_delta_px = 8.0;   // 像素尺度：6~15
    const double t_prior_w      = 0.01;  // 平移正则：t 单位为 mm 时 0.001~0.05 之间试
    const int max_iter = 100;

    // refine intrinsics priors (very important)
    const double w_invpol = 1e-3;   // 1e-4~1e-2
    const double w_aff    = 1e-2;   // 1e-3~1e-1

    const fs::path config_path = fs::path("config") / "config.yaml";
    auto config = load_config_kv(config_path);
    if (config.empty()) {
        std::cerr << "Config is empty: " << config_path << "\n";
        return 1;
    }

    std::string intrinsic_path = get_required_value(config, "intrinsics_path");
    std::string lidar_path     = get_required_value(config, "lidar_out");
    std::string photo_path     = get_required_value(config, "photo_out");
    std::string extrinsic_out  = get_required_value(config, "extrinsic_out");
    std::string intrin_out     = get_optional_value(config, "intrinsics_better", "./result/ocam_refined.txt");
    std::string valid_path     = get_required_value(config, "image_dir");

    OcamModel ocam;
    get_ocam_model(ocam, intrinsic_path);

    std::vector<PnPData> pts;
    read_pairs_raw(lidar_path, photo_path, pts);
    if (pts.empty()) return 1;

    auto inits = generate_all_90_degree_rotations();
    auto flips = all_flips();

    std::cout << "Initial rotations: " << inits.size() << "\n";
    std::cout << "Flip variants: " << flips.size() << "\n";
    std::cout << "Modes: 2 (Forward/Inverse)\n";

    // best (we will output as Lidar->Camera)
    double best_cost = 1e30;
    Eigen::Matrix3d bestR_L2C = Eigen::Matrix3d::Identity();
    Eigen::Vector3d bestt_L2C = Eigen::Vector3d::Zero();
    Eigen::Matrix3d bestA = Eigen::Matrix3d::Identity();
    Mode bestMode = Mode::Forward;
    Flip bestFlip{+1,+1,+1};

    // ========================= Stage A: search best extrinsic (intrinsics fixed) =========================
    for (auto& fl : flips) {
        Eigen::Matrix3d A = fl.M();

        for (int mode_id=0; mode_id<2; ++mode_id) {
            Mode mode = (mode_id==0) ? Mode::Forward : Mode::Inverse;

            std::cout << "\n--- Trying flip(sx,sy,sz)=("<<fl.sx<<","<<fl.sy<<","<<fl.sz<<")"
                      << " mode=" << ((mode==Mode::Forward)?"Forward":"Inverse") << " ---\n";

            for (size_t i=0; i<inits.size(); ++i) {
                Eigen::Matrix3d R0 = inits[i].cast<double>();
                Eigen::Quaterniond q0(R0);
                q0.normalize();

                // ext = [qx,qy,qz,qw, tx,ty,tz]
                double ext[7] = {q0.x(), q0.y(), q0.z(), q0.w(), 0.0, 0.0, 0.0};

                ceres::Problem problem;
                ceres::Manifold* q_manifold = new ceres::EigenQuaternionManifold();
                problem.AddParameterBlock(ext, 4, q_manifold);
                problem.AddParameterBlock(ext+4, 3);

                for (auto& p: pts) {
                    ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta_px);
                    problem.AddResidualBlock(ExternalCaliOcam::Create(p, ocam, A, mode), loss, ext, ext+4);
                }
                problem.AddResidualBlock(TranslationPrior::Create(t_prior_w, 0.0, 0.0, 0.0), nullptr, ext+4);

                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.max_num_iterations = max_iter;
                options.minimizer_progress_to_stdout = false;
                options.use_nonmonotonic_steps = true;

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                Eigen::Quaterniond q(ext[3], ext[0], ext[1], ext[2]);
                q.normalize();
                Eigen::Matrix3d R = q.toRotationMatrix();
                Eigen::Vector3d t(ext[4], ext[5], ext[6]);

                float err[2]{0,0};
                compute_error_RT(ocam, R, t, A, mode, pts, err);
                double cost = err[0] + err[1];

                if (cost < best_cost) {
                    best_cost = cost;
                    bestA = A;
                    bestMode = mode;
                    bestFlip = fl;

                    // unify output to Lidar->Camera
                    if (mode == Mode::Forward) {
                        bestR_L2C = R;
                        bestt_L2C = t;
                    } else {
                        bestR_L2C = R.transpose();
                        bestt_L2C = -R.transpose() * t;
                    }

                    std::cout << ">>> New BEST sum=" << best_cost
                              << "  flip=("<<fl.sx<<","<<fl.sy<<","<<fl.sz<<")"
                              << "  mode=" << ((mode==Mode::Forward)?"Forward":"Inverse")
                              << "  init=" << i << "\n";
                }
            }
        }
    }

    std::cout << "\n========== BEST RESULT (Stage A, intrinsics fixed) ==========\n";
    std::cout << "Best sum error = " << best_cost << "\n";
    std::cout << "Best flip(sx,sy,sz)=("<<bestFlip.sx<<","<<bestFlip.sy<<","<<bestFlip.sz<<")\n";
    std::cout << "Best mode = " << ((bestMode==Mode::Forward)?"Forward":"Inverse") << "\n";

    // ========================= Stage B: refine extrinsic + intrinsics (one run) =========================
    {
        std::cout << "\n========== REFINE (Stage B, optimize extrinsics + intrinsics) ==========\n";
        // init q,t from best L->C
        double q_ref[4];
        {
            Eigen::Quaterniond q(bestR_L2C);
            q.normalize();
            q_ref[0]=q.x(); q_ref[1]=q.y(); q_ref[2]=q.z(); q_ref[3]=q.w();
        }
        double t_ref[3] = { bestt_L2C.x(), bestt_L2C.y(), bestt_L2C.z() };

        // intrinsics params
        std::vector<double> invpol_ref = ocam.invpol;
        double aff_ref[5] = { ocam.xc, ocam.yc, ocam.c, ocam.d, ocam.e };
        double aff0[5]    = { ocam.xc, ocam.yc, ocam.c, ocam.d, ocam.e };

        ceres::Problem prob;

        ceres::Manifold* q_manifold = new ceres::EigenQuaternionManifold();
        prob.AddParameterBlock(q_ref, 4, q_manifold);
        prob.AddParameterBlock(t_ref, 3);
        prob.AddParameterBlock(invpol_ref.data(), (int)invpol_ref.size());
        prob.AddParameterBlock(aff_ref, 5);

        // residuals (use unified L->C => Mode::Forward)
        for (auto& p: pts) {
            ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta_px);
            prob.AddResidualBlock(
                ExternalCaliOcamDyn::Create(p, (int)invpol_ref.size(), bestA, Mode::Forward),
                loss,
                q_ref, t_ref, invpol_ref.data(), aff_ref
            );
        }

        // priors (MUST, or intrinsics will drift)
        prob.AddResidualBlock(InvpolPrior::Create(ocam.invpol, w_invpol), nullptr, invpol_ref.data());
        prob.AddResidualBlock(AffinePrior::Create(w_aff, aff0), nullptr, aff_ref);

        // mild t prior around current t_ref
        prob.AddResidualBlock(TranslationPrior::Create(t_prior_w, t_ref[0], t_ref[1], t_ref[2]), nullptr, t_ref);

        // bounds (recommended)
        prob.SetParameterLowerBound(aff_ref, 0, 0.0);
        prob.SetParameterUpperBound(aff_ref, 0, (double)ocam.width);
        prob.SetParameterLowerBound(aff_ref, 1, 0.0);
        prob.SetParameterUpperBound(aff_ref, 1, (double)ocam.height);
        prob.SetParameterLowerBound(aff_ref, 2, 0.1); // c > 0

        ceres::Solver::Options opt;
        opt.linear_solver_type = ceres::DENSE_SCHUR;
        opt.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        opt.max_num_iterations = 80;
        opt.minimizer_progress_to_stdout = true;
        opt.use_nonmonotonic_steps = true;

        ceres::Solver::Summary sum;
        ceres::Solve(opt, &prob, &sum);

        // write back refined params
        Eigen::Quaterniond qf(q_ref[3], q_ref[0], q_ref[1], q_ref[2]);
        qf.normalize();
        bestR_L2C = qf.toRotationMatrix();
        bestt_L2C = Eigen::Vector3d(t_ref[0], t_ref[1], t_ref[2]);

        ocam.invpol = invpol_ref;
        ocam.xc = aff_ref[0]; ocam.yc = aff_ref[1];
        ocam.c  = aff_ref[2]; ocam.d  = aff_ref[3]; ocam.e  = aff_ref[4];

        std::cout << "\n[Refine done] intrinsics:\n"
                  << "xc="<<ocam.xc<<" yc="<<ocam.yc
                  << " c="<<ocam.c<<" d="<<ocam.d<<" e="<<ocam.e << "\n";
    }

    // ========================= output =========================
    std::cout << "\n========== FINAL OUTPUT ==========\n";
    std::cout << "Writing Lidar->Camera extrinsic to: " << extrinsic_out << "\n";
    writeExt_L2C(extrinsic_out, bestR_L2C, bestt_L2C);

    std::cout << "Writing refined ocam intrinsics to: " << intrin_out << "\n";
    write_ocam_intrinsics_full(intrin_out, ocam);

    // final error
    float final_err[2]{0,0};
    compute_error_RT(ocam, bestR_L2C, bestt_L2C, bestA, Mode::Forward, pts, final_err);
    std::cout << "Final avg error (using L->C forward + flip) u="<<final_err[0]
              << " v="<<final_err[1] << " sum="<<(final_err[0]+final_err[1]) << "\n";

    // visualize points
    visualize_RT(ocam, bestR_L2C, bestt_L2C, bestA, Mode::Forward, pts, valid_path);

    // ====== 全点云投影到对应图片 ======
    std::string pcd_dir  = get_optional_value(config, "lidar_dir", "");
    if (pcd_dir.empty()) {
        pcd_dir = get_optional_value(config, "lidar_dir", "");
    }
    std::string img_dir  = valid_path;

    bool pcd_m_to_mm = true; // 外参平移是 mm、pcd 通常是 m => true
    int step = 5;

    if (!pcd_dir.empty()) {
        visualize_pcd_sequence(ocam,
                               bestR_L2C,
                               bestt_L2C,
                               bestA,
                               pcd_dir,
                               img_dir,
                               pcd_m_to_mm,
                               step,
                               1);
    } else {
        std::cerr << "Config missing 'lidar_dir' (or 'lidar_dir'); skip pcd projection.\n";
    }

    return 0;
}
