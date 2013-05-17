//
//  vsd.h
//
//  Created by Shinichi Goto on 4/30/13.
//  Copyright (c) 2012 Shinichi Goto All rights reserved.
//

#ifndef VSD_H_
#define VSD_H_

#ifndef CV_H_
#include <cv.h>
#endif //CV_H_

#ifndef HIGHGUI_H_
#include <highgui.h>
#endif //HIGHGUI_H_

enum VSDImageFeature {
  VSD_SIFT = 0,
  VSD_SURF,
};

enum VSDLeraning {
  VSD_SVM = 0,
  VSD_ANN,
};


// 特徴量を管理するクラスを作る必要あり．codebookも保持する？


//
// SURFDescs
//   ...1つの画像中でのSURFを表す
typedef cv::Mat SURFDescs;

//
// Bag-of-Words
//
typedef std::vector<float> BoW;


//
// vsdManager
//   ...contains Bag-of-Features for each category.
//
class VSDManager
{
public:
  VSDManager ();
  VSDManager (std::string &category_name, 
              unsigned int category);
  VSDManager &operator=(const VSDManager &vsdManager);

  std::string getCategoryName() const {return category_name;};

  int getCategory () const {return bofs_category;}
  std::vector<BoW> getBoFs () const {return bofs;}

  void createBoFsForEachCategory (const cv::Mat& vwords,
                                  const std::vector<std::string> &images_path_array);

private:
  std::string category_name;
  int bofs_category;
  std::vector<BoW> bofs;
};


#endif //VSD_H_
