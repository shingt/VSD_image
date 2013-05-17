cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(vsd)

set(OPENCV_MANUAL_PATH /usr/local/include/opencv /usr/local/include/opencv2)
set(RESEARCH_ROOT_PATH ${PROJECT_SOURCE_DIR}/../)
set(UTILS_PATH ${RESEARCH_ROOT_PATH}/utils/)
set(HOG_PATH ${RESEARCH_ROOT_PATH}/hog/)
set(MANUAL_CV_LIBS opencv_core opencv_imgproc opencv_video opencv_highgui opencv_objdetect 
  opencv_nonfree opencv_flann opencv_ml opencv_features2d)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OPENCV_MANUAL_PATH})
include_directories(${RESEARCH_ROOT_PATH})
include_directories(${UTILS_PATH})
include_directories(${HOG_PATH})

#include_directories(${OpenCV_INCLUDE_DIR})   # if homebrew

# CMakeで変数名を表示する方法
#message(STATUS "${OpenCV_INCLUDE_DIR}")

link_directories(/usr/local/lib)
add_executable(vsd vsd.cpp ${UTILS_PATH}/cvutils.cpp ${UTILS_PATH}/CategoryFileManager.cpp ${HOG_PATH}/hog.cpp)

#target_link_libraries(vsd ${OpenCV_LIBS})    # if homebrew
target_link_libraries(vsd ${MANUAL_CV_LIBS})
