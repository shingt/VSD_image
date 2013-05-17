//
//  vsd.cpp
//
//  Created by Shinichi Goto on 4/30/13.
//  Copyright (c) 2012 Shinichi Goto All rights reserved.
//


#define USE_SVM 1
#define USE_ANN 1

#include <cxcore.h>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "vsd.h"

#ifndef HOG_H_
#include "hog.h"
#endif //HOG_H_

#ifndef CVUTILS_H_
#include "cvutils.h"
#endif //CVUTILS_H_

#ifndef CATEGORY_FILE_MANAGER_H_
#include "categoryFileManager.h"
#endif //CATEGORY_FILE_MANAGER_H_


// NOTE: These file should be in the images folder 
const std::string TRAIN_DATA_FILE_NAME("test_train_data_file_names.txt");
const std::string TEST_DATA_FILE_NAME("test_test_data_file_names.txt");

// parameter
const int DIM_SURF = 128;         // The number of dimension of SURF
const int ITERATION = 20;         // The number of iterations of k-means
const int COLOR_HIST_BIN = 64;   // For color histogram
const int CODEBOOK_SIZE = 100;     // The size of codebook
const int GRID = 10;              // The pixel interval of grid sampling
const int kNearestNeighbor = 1;                // k-NN

  
  template <class T>
std::ostream &operator<<(std::ostream &o, const std::vector<T> &v)
{
  o << "{ ";
  for (int i = 0; i < (int) v.size (); ++i) o << v[i] << " ";
  o << "}";
  return o;
}



#pragma mark - Public methods -

VSDManager::VSDManager ()
{};

VSDManager::VSDManager (std::string &category_name, 
                        unsigned int category)
  : category_name (category_name),
    bofs_category (category)
{
//  std::cout << "### " << category + 1 << " th category was created" << std::endl;
};

VSDManager &VSDManager::operator=(const VSDManager &vsdManager)
{
  this->category_name = vsdManager.getCategoryName ();
  this->bofs_category = vsdManager.getCategory ();
  return (*this);
}

//
// calcFeatures
//
//   @param[in] src...grey-scale source image.
//   @param[out] output descriptor
//
void calcFeatures (const cv::Mat &src, cv::Mat &descriptor)
{
  std::vector<cv::KeyPoint> kp_vec;     // キーポイント格納コンテナ(dummy)
  CV_Assert (!src.empty () && src.depth () == CV_8U);
  cv::SURF calc_surf (500, 4, 2, true);
  calc_surf (src, cv::Mat (), kp_vec, descriptor);
  assert (descriptor.cols == DIM_SURF);
} 


#pragma mark - Learning Algorithms -

//
// learnSVM
// @param[in] features
// @param[in] target 
// @param[out] SVM
//
void learnSVM (const cv::Mat &features, const cv::Mat &target, CvSVM &SVM)
{
  if (features.rows != target.rows) {
    std::cout << "### Error : Size of trainData and targetData didn't match." << std::endl;
    std::cout << "### trainData : cols : " << features.cols << " rows : " << features.rows << std::endl;
    std::cout << "### targetData : cols : " << target.cols << " rows : " << target.rows << std::endl;
    return;
  }
  std::cout << "### in learnSVM" << std::endl;
 
  CvSVMParams svm_params;
  // C_SVC...C-support vector classification
  svm_params.svm_type = CvSVM::C_SVC;
  svm_params.kernel_type = CvSVM::RBF;
  svm_params.term_crit = cvTermCriteria (CV_TERMCRIT_ITER, 10000, 1e-5);

  // TODO: Need to specify C and gamma.
  SVM.train (features, target, cv::Mat (), cv::Mat (), svm_params);
}  


//
// learnANN
// @param[in] features
// @param[in] target
// @param[out] ANN
//
void learnANN (const cv::Mat &features, const cv::Mat &target, CvANN_MLP &ANN)
{
  if (features.rows != target.rows) {
    std::cout << "### Error : Size of trainData and targetData didn't match." << std::endl;
    std::cout << "### trainData : cols : " << features.cols << " rows : " << features.rows << std::endl;
    std::cout << "### targetData : cols : " << target.cols << " rows : " << target.rows << std::endl;
    return;
  }
  std::cout << "### in learnANN" << std::endl;

  // Parameters
  int num_input_neurons = features.cols;
  int num_hidden_neurons = 250;
  int num_output_neurons = target.cols;
  int num_iterations = 1000;
  double bp_dw_scale = 0.1;
  // REVIEW: Threshold might be set.
  CvTermCriteria term_crit = cvTermCriteria (CV_TERMCRIT_ITER, num_iterations, 0);
  CvANN_MLP_TrainParams::CvANN_MLP_TrainParams ann_params (term_crit, CvANN_MLP_TrainParams::BACKPROP, bp_dw_scale);

  // Generate
  cv::Mat layerSizes = (cv::Mat_<int> (3, 1) 
      << num_input_neurons, num_hidden_neurons, num_output_neurons);
  ANN.create (layerSizes, CvANN_MLP::SIGMOID_SYM);

  // targetをANNのtarget用に変換する必要あり
  cv::Mat target_ann = cv::Mat::zeros (features.rows, num_output_neurons, CV_64FC1);
  for (int j = 0; j < features.rows; ++j) {
    if (target.at<int>(j, 0) == 1)
      target_ann.at<double>(j, 0) = 1.;
  }
  std::cout << "target: " << std::endl << target_ann << std::endl;
  assert (features.rows == target_ann.rows && target_ann.cols == num_output_neurons);
  ANN.train (features, target_ann, cv::Mat (), cv::Mat (), ann_params);
}  


#pragma mark - Image Features -

//
// generaeteCodebook
// NOTE: The same codebook should be used in test phase as well.
//
// @param[in] features...all features used to generate codebook.
// @param[out] codebook...output codebook.
//
void generateCodebook (const cv::Mat &features, cv::Mat &codebook)
{
  cv::Mat labels_dummy;   // 各特徴ベクトルがどのクラスタに割り当てられたかを表す（不使用）
  cv::kmeans (features, CODEBOOK_SIZE, labels_dummy, cv::TermCriteria
      (CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, ITERATION, 0.01), 1, 0,
      codebook);
  std::cout << "### Codebook was created successfully." << std::endl;
}


//
// createBoW
//
// @param[in] descriptors...extracted features.
// @param[in] codebook...calculated codebook.
// @param[out] bow...output bag-of-words.
//
void createBoW (const cv::Mat &descriptors, const cv::Mat &codebook, 
                std::vector<float> &bow)
{
  // 各局所特徴量に関して最近傍のvisual-wordを見つけて投票
  int num_of_features = descriptors.rows;
  cv::Mat indices = cv::Mat::zeros (num_of_features, kNearestNeighbor, CV_32SC1); // 最も近いvisual-wordのインデックス
  cv::Mat dists = cv::Mat::zeros (num_of_features, kNearestNeighbor, CV_64FC1);  // その距離
  cv::flann::Index flannIndex (codebook, cv::flann::KDTreeIndexParams ());
  cv::flann::SearchParams searchParams = cv::flann::SearchParams ();

  // Nearest Neighbor
  flannIndex.knnSearch (descriptors, indices, dists, kNearestNeighbor, searchParams);

  // Create histogram
  std::vector<int> hist (codebook.rows);
  std::fill (hist.begin (), hist.end (), 0);
  for (int j = 0; j < indices.rows; j++) {
    // 現在は最近傍法なのでindicesの最初の要素だけを見ればよい
    //int idx = CV_MAT_ELEM (indices, int, j, 0);
    int idx = indices.at<int> (j, 0);     // FIXME: 画素アクセス
    hist[idx]++;
  }

  // Normalization
  assert (bow.empty ());
  for (int i = 0; i < hist.size (); ++i) {
    float val_normalized = (float) hist[i] / (float) num_of_features;
    bow.push_back (val_normalized);
  }
}


void extractAndPushFeatures (const cv::Mat &image, std::vector<SURFDescs> &features_SURF, 
                             cv::Mat &features_color_histogram)
{
  // Resize image
  cv::Mat image_resized;
  SG::fixImageSize (image, image_resized, 400);

  // Create gray-scale image
  cv::Mat gray;
  cv::cvtColor (image, gray, CV_BGR2GRAY);

  // calculate SURF
  SURFDescs descriptor;            // 画像中のすべてのキーポイントに対するSURF 
  calcFeatures (gray, descriptor); 
  assert (descriptor.cols == DIM_SURF);
  features_SURF.push_back (descriptor);

  // calculate HOG
  //      int cell_size = 20, block_size = 3, num_grad_bins = 9;
  //      HOG hog (image, cell_size, block_size, num_grad_bins);
  //      features_HOG.push_back (hog.getFeatures ());

  // calculate color histogram
  std::vector<float> color_hist;
  SG::calcRegularizedColorHistogram (image, COLOR_HIST_BIN, color_hist);
  assert (!color_hist.empty ());
  cv::Mat temp (color_hist);
  cv::Mat temp_t = temp.t ();
  features_color_histogram.push_back (temp_t);
}


#pragma mark - Main -

int main (int argc, char **argv)
{
  // ----------------------------------
  //        Training Phase
  // ----------------------------------
  printf ("\n----------------------------------\n        Training Phase\n----------------------------------\n\n");

  //
  // Read training data using file manager
  //
  CategoryFileManager trainingFileManager (TRAIN_DATA_FILE_NAME);
  if (!trainingFileManager.generateFileDatas ()) {
    std::cout << "### Failed to generate training data." << std::endl;
    return -1;
  }
  std::vector<std::string> training_categories = trainingFileManager.getCategories ();
  int num_training_categories = training_categories.size ();

  //
  // Display training categories and numbers
  //
  std::cout << "### Training data categories : " << std::endl;
  for (int index = 0; index < num_training_categories; ++index) {
    std::cout << "Category " << index << " : " << training_categories[index] << std::endl;
  }
  std::cout << std::endl;

  // 全特徴量を抽出
  std::vector<SURFDescs> features_SURF;
//  cv::Mat features_HOG;
  cv::Mat features_color_histogram (0, COLOR_HIST_BIN, CV_32FC1);
  cv::Mat target;
  
  int num_of_actual_images = 0;

  // For each category 
  for (int index_category = 0; index_category < num_training_categories; ++index_category) {
    std::string category_file_name;
    std::vector<std::string> training_file_paths = trainingFileManager.getFilePaths (index_category);
    int num_images = training_file_paths.size ();

    // For each image in current category
    for (int index_image = 0; index_image < num_images; index_image++) {
      cv::Mat image = cv::imread (training_file_paths[index_image], CV_LOAD_IMAGE_COLOR);
      if (!image.data) {
        std::cout << "### Couldn't load the image : " << training_file_paths[index_image] << std::endl;
        continue;
      }
      std::cout << training_file_paths[index_image] << std::endl;
      num_of_actual_images++;

      // Extract features and prepare target
      extractAndPushFeatures (image, features_SURF, features_color_histogram);
      target.push_back (index_category);
    }
  }

  assert (features_color_histogram.rows == num_of_actual_images);

  // SURF用codebookの生成
  assert (features_SURF.size () == num_of_actual_images);
  cv::Mat features_SURF_all;
  // 全ての画像のSURFに対して
  for (std::vector<cv::Mat>::iterator it = features_SURF.begin ();
       it != features_SURF.end (); it++) {
    SURFDescs descriptor = *it;
    // For each descriptor in image
    for (int row_in_desc = 0, row_max = descriptor.rows; row_in_desc < row_max; ++row_in_desc) {
      features_SURF_all.push_back (descriptor.row (row_in_desc));
    }
  }
  cv::Mat codebook_for_SURF (CODEBOOK_SIZE, num_of_actual_images, CV_32FC1);
  generateCodebook (features_SURF_all, codebook_for_SURF);
  assert (codebook_for_SURF.cols == DIM_SURF && codebook_for_SURF.rows == CODEBOOK_SIZE);

  // HOG用codebookの生成
//  cv::Mat codebook_for_HOG (CODEBOOK_SIZE, num_of_images, CV_32FC1);
//  generateCodebook (features_HOG, codebook_for_HOG);


  // After creating codebook

  // Bag-of-SURF生成
  cv::Mat vec_bag_of_SURF (0, CODEBOOK_SIZE, CV_32FC1);    // すべての画像のBag-of-SURFを含む
  for (int i = 0; i < num_of_actual_images; i++) {
    SURFDescs descriptor = features_SURF[i];
    BoW bag_of_SURF;
    createBoW (descriptor, codebook_for_SURF, bag_of_SURF);
    SG::concatVectorToMatVertical (bag_of_SURF, vec_bag_of_SURF);
  }
  assert (vec_bag_of_SURF.rows == num_of_actual_images);
//  std::cout << "### color " << std::endl << color_hist << std::endl; 
  
  // Bag-of-HOG生成
//  cv::Mat bag_of_HOG;
//  for (int i = 0; i < nuM_of_images; ++i) {
//    std::vector hog_desc;
//    createBoW ( , codebook_for_HOG, hog_desc);
//    bag_of_HOG.push_back (hog_desc);
//  }
//  assert (bag_of_HOG.rows == num_of_images);

  // 全特徴量の合成
//  assert ((bag_of_SURF.rows == bag_of_HOG.rows) && (bag_of_HOG.rows == features_color_histogram.rows));
  assert ((vec_bag_of_SURF.rows == features_color_histogram.rows));

  cv::Mat all_training_features; 
  all_training_features = SG::concatMatsHorizontal (vec_bag_of_SURF, features_color_histogram);

#if USE_SVM
  // Support Vector Machine
  CvSVM SVM;
  learnSVM (all_training_features, target, SVM);
  std::cout << "### Finished learning SVM." << std::endl;
#endif // USE_SVM

#if USE_ANN
  // Artificial Neural Network
  CvANN_MLP ANN;
  learnANN (all_training_features, target, ANN);
  std::cout << "### Finished learning ANN." << std::endl;
#endif // USE_ANN


  // ------------------------------ 
  //        Test Phase
  // ------------------------------
  printf ("\n----------------------------------\n        Test Phase\n----------------------------------\n\n");
  
  // Read test data using file manager
  CategoryFileManager testFileManager (TEST_DATA_FILE_NAME);
  if (!testFileManager.generateFileDatas ()) {
    std::cout << "### Failed to generate test data." << std::endl;
    return -1;
  }
 
  std::vector<std::string> test_categories = testFileManager.getCategories ();
  int num_test_categories = test_categories.size ();

  // Display test categories
  std::cout << "### Test data categories : " << std::endl;
  for (int index = 0; index < num_test_categories; ++index) {
    std::cout << "Category " << index << " : " << test_categories[index] << std::endl;
  }
  std::cout << std::endl;

  // 各カテゴリに対して
  for (int current_category = 0; current_category < num_training_categories; ++current_category) {
    std::cout << "--- Current Category : " << current_category << " ----------------------" << std::endl;

    std::vector<std::string> test_file_paths = testFileManager.getFilePaths (current_category);
    int num_images = test_file_paths.size ();
    int num_actual_images = 0;

#if USE_SVM
    int num_correct_svm = 0;
#endif//USE_SVM
#if USE_ANN
    int num_correct_ann = 0;
#endif//USE_ANN

    // カテゴリ内の全画像の特徴量を抽出，BoF生成，SVM判定
    for (int index_image = 0; index_image < num_images; ++index_image) {
      cv::Mat image = cv::imread (test_file_paths[index_image]);
      if (image.empty ()) {
        std::cout << "### No image found : " << test_file_paths[index_image] << std::endl;
      } else {
        std::cout << "### File name : " << test_file_paths[index_image] << std::endl;
        num_actual_images++;

        cv::Mat resized_image;
        SG::fixImageSize (image, resized_image, 400);
   
        //
        // Feature Extraction 
        //
        cv::Mat gray;
        cv::cvtColor (resized_image, gray, CV_BGR2GRAY);

        // SURF
        SURFDescs descriptor;                 // 画像中のすべてのキーポイントに対するSURF 
        calcFeatures (gray, descriptor); 
        assert (descriptor.cols == DIM_SURF);

        // Bo-SURF
        BoW bag_of_SURF;
        createBoW (descriptor, codebook_for_SURF, bag_of_SURF);
        assert (!bag_of_SURF.empty ());
       
        // Color histogram
        std::vector<float> color_hist;
        SG::calcRegularizedColorHistogram (resized_image, COLOR_HIST_BIN, color_hist);
 
        // 特徴量を合成
        bag_of_SURF.insert (bag_of_SURF.end (), color_hist.begin (), color_hist.end ());
        cv::Mat test_features (bag_of_SURF);

        //
        // Classification
        //
      
        int result_category = -1;
#if USE_SVM
        result_category = SVM.predict (test_features.t ());

        std::cout << "Number of support vectors : " << SVM.get_support_vector_count () << std::endl;
        std::cout << "SVM Result: ";
        if (result_category == current_category) {
          std::cout << "Correct" << std::endl;
          ++num_correct_svm;
        } else {
          std::cout << "Wrong" << std::endl;
        } 
        std::cout << "SVM Result category: " << result_category << std::endl;
#endif // USE_SVM

#if USE_ANN
        cv::Mat outputs;
        ANN.predict (test_features.t (), outputs);
        assert (outputs.rows == 1 && outputs.cols == 1);

        // XXX: Sigmoid function should be used?
        float output_num = outputs.at<float> (0, 0);
        std::cout << output_num << std::endl;
        if (output_num >= 0.) {
          result_category = 1;
        } else {
          result_category = 0;
        }
        std::cout << "ANN Result: ";
        if (result_category == current_category) {
          std::cout << "Correct" << std::endl;
          ++num_correct_ann;
        } else {                                 
          std::cout << "Wrong" << std::endl;
        } 
        std::cout << "ANN Result category : " << result_category << std::endl;
#endif // USE_ANN
        std::cout << std::endl;
      }
    }

    // カテゴリ内での正解率を表示
#if USE_SVM
    std::cout << "### SVM Category Result : " << num_correct_svm << " / " << num_actual_images << std::endl;
    std::cout << "### SVM Accuracy : " << 100. * float (num_correct_svm) / float (num_actual_images) 
      << " %" << std::endl << std::endl;
#endif//USE_SVM

#if USE_SVM
    std::cout << "### ANN Category Result : " << num_correct_ann << " / " << num_actual_images << std::endl;
    std::cout << "### ANN Accuracy : " << 100. * float (num_correct_ann) / float (num_actual_images) 
      << " %" << std::endl << std::endl;
#endif//USE_SVM

  }
  return 0;
}
