/****************************************************************************
 *
 * This file is part of the ViSP software.
 * Copyright (C) 2005 - 2015 by Inria. All rights reserved.
 *
 * This software is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * ("GPL") version 2 as published by the Free Software Foundation.
 * See the file LICENSE.txt at the root directory of this source
 * distribution for additional information about the GNU GPL.
 *
 * For using ViSP with software that can not be combined with the GNU
 * GPL, please contact Inria about acquiring a ViSP Professional
 * Edition License.
 *
 * See http://visp.inria.fr for more information.
 *
 * This software was developed at:
 * Inria Rennes - Bretagne Atlantique
 * Campus Universitaire de Beaulieu
 * 35042 Rennes Cedex
 * France
 *
 * If you have questions regarding the use of this file, please contact
 * Inria at visp@inria.fr
 *
 * This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
 * WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Description:
 * Track blobs that belong to the same object.
 * Compute the pose knowing the 3D model.
 *
 * Authors:
 * Souriya Trinh
 *
 *****************************************************************************/

#ifndef __vpBlobTracker_h__
#define __vpBlobTracker_h__

#include <iostream>
#include <numeric>
#include <map>

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpPoint.h>
#include <visp3/core/vpKalmanFilter.h>
#include <visp3/core/vpImageConvert.h>

// OpenCV is required
#if (VISP_HAVE_OPENCV_VERSION >= 0x020403)
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


class vpBlobInfo {
public:
  typedef enum BLOB_TYPE {
      NOT_TRACKED, PREDICTION, NEAREST, PROJECTION
  } BLOB_TYPE;

  //Blob area
  double m_blobArea;
  //Blob contour
  std::vector<vpImagePoint> m_blobContourPts;
  //Blob index
  size_t m_blobIndex;
  //History of 2D image positions
  std::vector<vpImagePoint> m_imPts;
  //2D Kalman filter
  vpKalmanFilter m_kalmanFilter;
  //Limit of the history
  size_t m_maxHistory;
  //3D model point
  vpPoint m_pt;
  //History of status
  std::vector<BLOB_TYPE> m_statusHistory;

  vpBlobInfo() : m_blobArea(-1.0), m_blobContourPts(), m_blobIndex(), m_imPts(), m_kalmanFilter(),
      m_maxHistory(30), m_pt(), m_statusHistory() {
  }

  void cleanVectors() {
    if(m_imPts.size() > m_maxHistory) {
      m_imPts.erase(m_imPts.begin());
    }

    if(m_statusHistory.size() > m_maxHistory) {
      m_statusHistory.erase(m_statusHistory.begin());
    }
  }
};


class VISP_EXPORT vpObjectBlobTracker {
public:
  //! Method to use to predict the position of a given 3D point in the image plane for the current frame
  typedef enum PREDICTION_TYPE {
      NO_PREDICTION, KALMAN_2D, KALMAN_3D, PROJECTION, CONSTANT_VELOCITIE, MEDIAN_FLOW
  } PREDICTION_TYPE;

  //! Blob type
  typedef enum BLOB_TYPE {
      TOO_SMALL, TOO_BIG, GOOD_AREA, AGGREGATE //Chunk contours
  } BLOB_TYPE;

  //! Morphology operation type
  typedef enum MORPHOLOGY_TYPE {
    MORPHOLOGY_ERODE    = cv::MORPH_ERODE,
    MORPHOLOGY_DILATE   = cv::MORPH_DILATE,
    MORPHOLOGY_OPEN     = cv::MORPH_OPEN,
    MORPHOLOGY_CLOSE    = cv::MORPH_CLOSE,
    MORPHOLOGY_GRADIENT = cv::MORPH_GRADIENT,
    MORPHOLOGY_TOPHAT   = cv::MORPH_TOPHAT,
    MORPHOLOGY_BLACKHAT = cv::MORPH_BLACKHAT,
    NO_MORPHOLOGY
  } MORPHOLOGY_TYPE;

  //! Structuring element type
  typedef enum MORPHOLOGY_SHAPE_TYPE {
    MORPHOLOHY_SHAPE_RECTANGLE  = cv::MORPH_RECT,
    MORPHOLOGY_SHAPE_CROSS      = cv::MORPH_CROSS,
    MORPHOLOGY_SHAPE_ELLIPSE    = cv::MORPH_ELLIPSE
  } MORPHOLOGY_SHAPE_TYPE;

  //! Pose estimation type
  typedef enum POSE_ESTIMATION_METHOD {
    VVS_POSE_ESTIMATION,
    VVS_POSE_ESTIMATION_WITH_PREVIOUS_POSE_INITIAL_GUESS,
    OPENCV_POSE_ESTIMATION,
    OPENCV_POSE_ESTIMATION_WITH_PREVIOUS_POSE_INITIAL_GUESS
  } POSE_ESTIMATION_METHOD;


private:
  //! Camera intrinsic parameters
  vpCameraParameters m_cam;
  //! Homogeneous matrix between the object and the camera frame
  vpHomogeneousMatrix m_cMo;
  //! Homogeneous matrix between the object and the camera frame computed by OpenCV, could be used for comparison later
  vpHomogeneousMatrix m_cMo_OpenCV;
  //! Homogeneous matrix between the object and the camera frame predicted by the Kalman filter
  vpHomogeneousMatrix m_cMo_predicted;
  //! Homogeneous matrix between the object and the camera frame computed without the initial guess,
  //! could be used for comparison later
  vpHomogeneousMatrix m_cMo_raw;
  //! Image with the detected contour
  cv::Mat m_contourImage;
  //! Covariance matrix
  vpMatrix m_covarianceMatrix;
  //! Flag sets to true when the covariance matrix has been correctly computed
  bool m_covarianceMatrixOk;
  //! Flag sets to true when it is the first call
  bool m_first;
  //! Number of frame per second
  double m_fps;
  //! Image with history of the blob locations
  vpImage<unsigned char> m_heatMap;
  //! Map of blob types
  std::map<BLOB_TYPE, std::vector<std::vector<cv::Point> > > m_mapOfBlobTypes;
  //! Map of object blobs
  std::map<std::string, vpBlobInfo> m_mapOfObjectBlobs;
  //! Map of predicted blob locations
  std::map<std::string, vpImagePoint> m_mapOfPredictedBlobPoints;
  //! Binary image that should contains all the object blobs
  cv::Mat m_matImgBinarized;
  //! Median displacement flow
  vpImagePoint m_medianFlow;
  //! Threshold for the minimum blob area to be considered as an object blob
  double m_minBlobArea;
  //! Threshold for the maximum blob area to be considered as an object blob
  double m_maxBlobArea;
  //! Threshold for the maximum distance to match a detected blob with his prediction
  double m_maxDist;
  //! Pose estimation type (VVS initialized by Dementhon or Lagrange, OpenCV, VVS (or OpenCV) initialized by the previous pose)
  POSE_ESTIMATION_METHOD m_poseEstimationMethod;
  //! Kalman filter for the pose
  vpKalmanFilter m_poseKalmanFilter;
  //! If true the heat map is computed
  bool m_useComputeHeatMap;
  //! If true the previous pose is used to initialize the VVS pose estimation if the previous pose is ok
  bool m_usePreviousPoseGuess;
  //! List of the current detected blobs
  std::vector<vpBlobInfo> m_vectorOfProbableObjectBlobs;


public:
  vpObjectBlobTracker();
  vpObjectBlobTracker(const double fps);
  vpObjectBlobTracker(const std::map<std::string, vpPoint> &mapOfBlobs, const double fps);
  vpObjectBlobTracker(const vpObjectBlobTracker &blobTracker);

  vpObjectBlobTracker& operator=(const vpObjectBlobTracker &blobTracker);

  vpImage<unsigned char> binarize(const vpImage<vpRGBa> &I_color, const unsigned char threshold,
      const MORPHOLOGY_TYPE morphology=NO_MORPHOLOGY, const unsigned int kernelSize=1,
      const MORPHOLOGY_SHAPE_TYPE shape=MORPHOLOGY_SHAPE_ELLIPSE, const int nbIterations=1, const bool useHSV=false);
  vpImage<unsigned char> binarize(const vpImage<unsigned char> &I, const unsigned char threshold,
      const MORPHOLOGY_TYPE morphology=NO_MORPHOLOGY, const unsigned int kernelSize=1,
      const MORPHOLOGY_SHAPE_TYPE shape=MORPHOLOGY_SHAPE_ELLIPSE, const int nbIterations=1);

  void display(const vpImage<unsigned char> &I, const bool displayLegend=false);
  void display(const vpImage<vpRGBa> &I, const bool displayLegend=false);

  inline void getBinaryImage(vpImage<unsigned char> &I_binary) const {
    vpImageConvert::convert(m_matImgBinarized, I_binary);
  }

  inline vpCameraParameters getCameraParameters() const {
    return m_cam;
  }

  inline bool getCovarianceMatrix(vpMatrix &covarianceMatrix) const {
    //if m_covarianceMatrixOk == false, covarianceMatrix will be equal to DBL_MAX
    covarianceMatrix = m_covarianceMatrix;
    return m_covarianceMatrixOk;
  }

  vpHomogeneousMatrix getFilteredPose() const;

  inline double getFps() const {
    return m_fps;
  }

  inline vpImage<unsigned char> geHeatMap() const {
    return m_heatMap;
  }

  inline double getMaxBlobArea() const {
    return m_maxBlobArea;
  }

  inline double getMaxBlobDistanceMatching() const {
    return m_maxDist;
  }

  inline vpHomogeneousMatrix getPose() const {
    return m_cMo;
  }

  inline void getPose(vpHomogeneousMatrix &cMo) const {
    cMo = m_cMo;
  }

  inline POSE_ESTIMATION_METHOD getPoseEstimationMethod() const {
    return m_poseEstimationMethod;
  }
  
  inline vpKalmanFilter getPoseKalmanFilter() const {
    return m_poseKalmanFilter;
  }

  inline bool getUseComputeHeatMap() const {
    return m_useComputeHeatMap;
  }

  std::vector<vpImagePoint> initClick(const vpImage<unsigned char> &I, const std::string &filename,
      const bool refinePosition=true);
  std::vector<vpImagePoint> initClick(const vpImage<vpRGBa> &I, const std::string &filename,
      const bool refinePosition=true);

  void initFromPose(const vpImage<unsigned char> &I_binary, const vpHomogeneousMatrix &cMo_init);

  void loadModel(const std::string &filename);

  /*!
    Reset all the internal variables.
  */
  inline void reset() {
    m_cam = vpCameraParameters();
    m_cMo.eye();
    m_cMo_OpenCV.eye();
    m_cMo_predicted.eye();
    m_cMo_raw.eye();
    m_contourImage.release();
    m_covarianceMatrix = vpMatrix(6,6);
    m_covarianceMatrixOk = false;
    m_first = true;
    m_fps = 25.0;
    m_heatMap.destroy();
    m_mapOfBlobTypes.clear();
    m_mapOfObjectBlobs.clear();
    m_mapOfPredictedBlobPoints.clear();
    m_matImgBinarized.release();
    m_medianFlow = vpImagePoint();
    m_minBlobArea = 5.0;
    m_maxBlobArea = 250.0;
    m_maxDist = 12.0;
    m_poseEstimationMethod = VVS_POSE_ESTIMATION_WITH_PREVIOUS_POSE_INITIAL_GUESS;
    m_poseKalmanFilter = vpKalmanFilter();
    m_useComputeHeatMap = true;
    m_usePreviousPoseGuess = false;
    m_vectorOfProbableObjectBlobs.clear();
  }

  inline void setCameraParameters(const vpCameraParameters &cam) {
    m_cam = cam;
  }

  void setFps(const double fps);

  inline void setMaxBlobArea(const double maxArea) {
    m_maxBlobArea = maxArea;
  }

  inline void setMaxBlobMatchingDistance(const double maxDist) {
    m_maxDist = maxDist;
  }

  void setMaxNbPrediction(const size_t nb);

  inline void setPoseEstimationMethod(const POSE_ESTIMATION_METHOD &method) {
    m_poseEstimationMethod = method;
  }

  inline void setUseComputeHeatMap(const bool use) {
    m_useComputeHeatMap = use;
  }

  void track(const vpImage<vpRGBa> &I_color, const unsigned char threshold, const MORPHOLOGY_TYPE morphology=NO_MORPHOLOGY,
      const unsigned int kernelSize=1, const MORPHOLOGY_SHAPE_TYPE shape=MORPHOLOGY_SHAPE_ELLIPSE, const int nbIterations=1,
      const bool useHSV=false, bool (*func)(vpHomogeneousMatrix *)=NULL, const PREDICTION_TYPE &predictionType=PROJECTION);
  void track(const vpImage<unsigned char> &I, const bool doBinarize=true, const unsigned char threshold=0,
      const MORPHOLOGY_TYPE morphology=NO_MORPHOLOGY, const unsigned int kernelSize=1,
      const MORPHOLOGY_SHAPE_TYPE shape=MORPHOLOGY_SHAPE_ELLIPSE, const int nbIterations=1,
      bool (*func)(vpHomogeneousMatrix *)=NULL, const PREDICTION_TYPE &predictionType=PROJECTION);

  friend std::ostream& operator<<(std::ostream& os, const vpPoint& pt);
  friend std::ostream& operator<<(std::ostream& os, const vpObjectBlobTracker& blobTracker);


private:
  std::vector<vpBlobInfo> aggregateContours(const std::vector<std::vector<cv::Point> > contours);

  void computeHeatMap();

  vpImagePoint computeMedianBlobFlow();

  void computePose(bool (*func)(vpHomogeneousMatrix *)=NULL);

  std::vector<vpBlobInfo> findBlobCentroid();

  double getCoherentAngle(const double prevAngle, const double currentAngle);

  bool getNearestBlob(const vpBlobInfo &predictedPoint, const std::vector<vpBlobInfo> &vectorOfProbableLightBlobs,
      vpImagePoint &nearest, double &min_dist, size_t &nearest_index, double &min_area,
      const std::vector<size_t> &alreadyProcessIndex, const double max_dist=12.0);

  void initPoseKalmanFilter();

  bool isInclusiveContour(const std::vector<cv::Point> &contours1, const std::vector<cv::Point> &contours2,
      const double inclusivePercentage=0.1, const double minAngleDeviation=160.0, const double distFactor=2.0);

  double pointDistance(const cv::Point &pt1, const cv::Point &pt2);

  vpImagePoint refinePositionWithCentroid(const vpImagePoint &pt);

  void setPoseKalmanFilterTimeUpdate();

  void updateBlobPositions(std::map<std::string, vpBlobInfo> &mapOfBlobs,
      const std::vector<vpBlobInfo> &vectorOfProbableLightBlobs, const PREDICTION_TYPE predictionType=PROJECTION);

  void updatePoseKalmanFilter(const vpColVector &z_measurement);
};

#endif
#endif
