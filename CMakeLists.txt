cmake_minimum_required(VERSION 3.10)

# ---------------- Project & C++ Standard ----------------
project(MyProject LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- macOS/Homebrew hints ------------------
# 把 Homebrew 的常见前缀都加上（含 Qt/Freetype/VTK）
list(PREPEND CMAKE_PREFIX_PATH
  "/opt/homebrew"
  "/opt/homebrew/opt/qt"
  "/opt/homebrew/opt/freetype"
  "/opt/homebrew/opt/vtk"
)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# =============== Abseil (absl) ==========================
find_package(absl QUIET CONFIG)
if(NOT absl_FOUND)
  find_package(absl QUIET MODULE)
endif()
if(NOT absl_FOUND)
  message(STATUS "Abseil not found via find_package, trying manual setup...")
  find_path(ABSL_INCLUDE_DIR absl/base/config.h PATHS /opt/homebrew/include)
  find_library(ABSL_BASE_LIB     absl_base     PATHS /opt/homebrew/lib)
  find_library(ABSL_STRINGS_LIB  absl_strings  PATHS /opt/homebrew/lib)
  find_library(ABSL_LOG_LIB      absl_log_internal_check_op PATHS /opt/homebrew/lib)
  if(ABSL_INCLUDE_DIR AND ABSL_BASE_LIB AND ABSL_STRINGS_LIB AND ABSL_LOG_LIB)
    message(STATUS "Found Abseil manually: ${ABSL_INCLUDE_DIR}")
    add_library(absl::base INTERFACE IMPORTED)
    set_target_properties(absl::base PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ABSL_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${ABSL_BASE_LIB}"
    )
    add_library(absl::strings INTERFACE IMPORTED)
    set_target_properties(absl::strings PROPERTIES
      INTERFACE_LINK_LIBRARIES "${ABSL_STRINGS_LIB}"
    )
    add_library(absl::log_internal_check_op INTERFACE IMPORTED)
    set_target_properties(absl::log_internal_check_op PROPERTIES
      INTERFACE_LINK_LIBRARIES "${ABSL_LOG_LIB}"
    )
    set(absl_FOUND TRUE)
  endif()
endif()

# =============== JsonCpp ================================
find_package(JsonCpp QUIET CONFIG)
if (NOT JsonCpp_FOUND)
  find_package(jsoncpp REQUIRED CONFIG)
endif()
if (TARGET jsoncpp_lib AND NOT TARGET JsonCpp::JsonCpp)
  add_library(JsonCpp::JsonCpp INTERFACE IMPORTED)
  set_target_properties(JsonCpp::JsonCpp PROPERTIES
    INTERFACE_LINK_LIBRARIES jsoncpp_lib)
endif()

# =============== 通用依赖 ===============================
find_package(Boost REQUIRED COMPONENTS thread)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
# 需要可视化 → 列出 PCL 可视化模块；如果你不需要可视化，可把 visualization 从组件里删掉
find_package(PCL REQUIRED COMPONENTS common io filters kdtree features visualization)
find_package(Threads REQUIRED)

# =============== GUI + 字体（VTK 的 GUI 依赖）==========
# 先找到 Qt6 和 Freetype，再找带 Qt 组件的 VTK
find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)
find_package(Freetype REQUIRED)

# =============== VTK（含 Qt 组件）=======================
# 只保留这一处 VTK 查找，列出常见且 PCL 可视化会用到的组件
find_package(VTK REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  FiltersCore
  FiltersSources
  IOImage
  IOLegacy
  RenderingCore
  RenderingOpenGL2
  RenderingFreeType       # 依赖 Freetype
  InteractionStyle
  GUISupportQt            # 依赖 Qt6
  RenderingQt             # 依赖 Qt6
)

# =============== 汇总通用库 =============================
set(COMMON_LIBS
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  Threads::Threads
  JsonCpp::JsonCpp
  Qt6::Widgets
  Qt6::OpenGL
  Qt6::OpenGLWidgets
  Freetype::Freetype
  VTK::CommonCore
  VTK::CommonDataModel
  VTK::FiltersCore
  VTK::FiltersSources
  VTK::IOImage
  VTK::IOLegacy
  VTK::RenderingCore
  VTK::RenderingOpenGL2
  VTK::RenderingFreeType
  VTK::InteractionStyle
  VTK::GUISupportQt
  VTK::RenderingQt
)

# Eigen3 目标（有的打包导出此目标）
if (TARGET Eigen3::Eigen)
  list(APPEND COMMON_LIBS Eigen3::Eigen)
endif()



# =============== 可执行目标封装 =========================
function(add_my_executable target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE ${COMMON_LIBS})

  # 如果手动设置了 Abseil 的头文件位置，把它加到 include
  if(ABSL_INCLUDE_DIR)
    target_include_directories(${target_name} PRIVATE ${ABSL_INCLUDE_DIR})
  endif()

  # 某些环境下 PCL 需要定义
  target_compile_definitions(${target_name} PRIVATE PCL_NO_PRECOMPILE=1)
endfunction()

# =============== 你的可执行文件 =========================
add_my_executable(main_pano       main_pano.cpp)
add_my_executable(main             main.cpp)
add_my_executable(pcd_show         pcd_show.cpp)
add_my_executable(unfold           ../unfold.cpp)
add_my_executable(projectpano      projectCloud_pano.cpp)
add_my_executable(projectpano2     projectCloud_pano2.cpp)
add_my_executable(yuyan            main_yuyan.cpp)
add_my_executable(yuyan_show       projectCloud_yuyan.cpp)
# add_my_executable(pano_yaw         main_pano_yaw.cpp)
add_my_executable(projectCloud_pano2_show projectCloud_pano2_show.cpp)

add_my_executable(pcd_viewer       pcd_viewer.cpp)
# =============== VTK 9+ 模块自动初始化 ==================
if (VTK_FOUND AND COMMAND vtk_module_autoinit)
  vtk_module_autoinit(
    TARGETS
      main_pano main pcd_show unfold projectpano projectpano2 yuyan yuyan_show  pcd_viewer
    MODULES
      ${VTK_LIBRARIES}
  )
endif()
