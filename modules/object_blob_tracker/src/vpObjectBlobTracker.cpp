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

#include <iostream>
#include <algorithm>
#include <map>

#include <visp3/object_blob_tracker/vpObjectBlobTracker.h>
#include <visp3/core/vpDisplay.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/core/vpMeterPixelConversion.h>
#include <visp3/core/vpPolygon.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpImageTools.h>
#include <visp3/core/vpMath.h>
#include <visp3/vision/vpPose.h>

// OpenCV is required
#if (VISP_HAVE_OPENCV_VERSION >= 0x020403)

#include <opencv2/calib3d/calib3d.hpp>

typedef struct matching_info_t {
  std::string name;
  double dist;
  size_t blob_index;
  vpImagePoint nearest;
  double area;

  matching_info_t(const std::string &n, const double d, const size_t idx,
      const vpImagePoint &imPt, const double _area) :
    name(n), dist(d), blob_index(idx), nearest(imPt), area(_area) {

  }
} matching_info_t;

bool sortMatching(const matching_info_t &m1, const matching_info_t &m2) {
  if(m1.dist < 0.0) {
    return false;
  }

  if(m2.dist < 0.0) {
    return true;
  }

  return m1.dist < m2.dist;
}

vpObjectBlobTracker::vpObjectBlobTracker() : m_cam(), m_cMo(), m_cMo_OpenCV(), m_cMo_predicted(), m_cMo_raw(), m_contourImage(),
    m_covarianceMatrix(6,6), m_covarianceMatrixOk(false), m_first(true), m_fps(25), m_heatMap(), m_mapOfBlobTypes(),
    m_mapOfObjectBlobs(), m_mapOfPredictedBlobPoints(), m_matImgBinarized(), m_medianFlow(), m_minBlobArea(5.0),
    m_maxBlobArea(250.0), m_maxDist(12.0), m_poseEstimationMethod(VVS_POSE_ESTIMATION_WITH_PREVIOUS_POSE_INITIAL_GUESS),
    m_poseKalmanFilter(), m_useComputeHeatMap(true), m_usePreviousPoseGuess(false), m_vectorOfProbableObjectBlobs() {

  initPoseKalmanFilter();
}

vpObjectBlobTracker::vpObjectBlobTracker(const double fps) : m_cam(), m_cMo(), m_cMo_OpenCV(), m_cMo_predicted(), m_cMo_raw(),
    m_contourImage(), m_covarianceMatrix(6,6), m_covarianceMatrixOk(false), m_first(true), m_fps(fps), m_heatMap(),
    m_mapOfBlobTypes(), m_mapOfObjectBlobs(), m_mapOfPredictedBlobPoints(), m_matImgBinarized(), m_medianFlow(),
    m_minBlobArea(5.0), m_maxBlobArea(250.0), m_maxDist(12.0),
    m_poseEstimationMethod(VVS_POSE_ESTIMATION_WITH_PREVIOUS_POSE_INITIAL_GUESS), m_poseKalmanFilter(),
    m_useComputeHeatMap(true), m_usePreviousPoseGuess(false), m_vectorOfProbableObjectBlobs() {

  initPoseKalmanFilter();
}

vpObjectBlobTracker::vpObjectBlobTracker(const std::map<std::string, vpPoint> &mapOfBlobs, const double fps) :
    m_cam(), m_cMo(), m_cMo_OpenCV(), m_cMo_predicted(), m_cMo_raw(), m_contourImage(), m_covarianceMatrix(6,6),
    m_covarianceMatrixOk(false), m_first(true), m_fps(fps), m_heatMap(), m_mapOfBlobTypes(), m_mapOfObjectBlobs(),
    m_mapOfPredictedBlobPoints(), m_matImgBinarized(), m_medianFlow(), m_minBlobArea(5.0), m_maxBlobArea(250.0),
    m_maxDist(12.0), m_poseEstimationMethod(VVS_POSE_ESTIMATION_WITH_PREVIOUS_POSE_INITIAL_GUESS), m_poseKalmanFilter(),
    m_useComputeHeatMap(true), m_usePreviousPoseGuess(false), m_vectorOfProbableObjectBlobs() {
  for(std::map<std::string, vpPoint>::const_iterator it = mapOfBlobs.begin(); it != mapOfBlobs.end(); ++it) {
    m_mapOfObjectBlobs[it->first].m_pt = it->second;
  }

  initPoseKalmanFilter();
}

vpObjectBlobTracker::vpObjectBlobTracker(const vpObjectBlobTracker &blobTrackerSource) {
  *this = blobTrackerSource;
}

vpObjectBlobTracker& vpObjectBlobTracker::operator=(const vpObjectBlobTracker &blobTrackerSource) {
  m_cam = blobTrackerSource.m_cam;
  m_cMo = blobTrackerSource.m_cMo;
  m_cMo_OpenCV = blobTrackerSource.m_cMo_OpenCV;
  m_cMo_predicted = blobTrackerSource.m_cMo_predicted;
  m_cMo_raw = blobTrackerSource.m_cMo_raw;
  m_contourImage = blobTrackerSource.m_contourImage.clone();
  m_covarianceMatrix = blobTrackerSource.m_covarianceMatrix;
  m_covarianceMatrixOk = blobTrackerSource.m_covarianceMatrixOk;
  m_first = blobTrackerSource.m_first;
  m_fps = blobTrackerSource.m_fps;
  m_heatMap = blobTrackerSource.m_heatMap;
  m_mapOfBlobTypes = blobTrackerSource.m_mapOfBlobTypes;
  m_mapOfObjectBlobs = blobTrackerSource.m_mapOfObjectBlobs;
  m_mapOfPredictedBlobPoints = blobTrackerSource.m_mapOfPredictedBlobPoints;
  m_matImgBinarized = blobTrackerSource.m_matImgBinarized.clone();
  m_medianFlow = blobTrackerSource.m_medianFlow;
  m_minBlobArea = blobTrackerSource.m_minBlobArea;
  m_maxBlobArea = blobTrackerSource.m_maxBlobArea;
  m_maxDist = blobTrackerSource.m_maxDist;
  m_poseEstimationMethod = blobTrackerSource.m_poseEstimationMethod;
  m_poseKalmanFilter = blobTrackerSource.m_poseKalmanFilter;
  m_useComputeHeatMap = blobTrackerSource.m_useComputeHeatMap;
  m_usePreviousPoseGuess = blobTrackerSource.m_usePreviousPoseGuess;
  m_vectorOfProbableObjectBlobs = blobTrackerSource.m_vectorOfProbableObjectBlobs;

  return *this;
}

/*!
  Aggregate contours (blobs) that are likely to belong to the same object.

  \param contours : List of contours, each contour is represented by a list of 2D points.

  \return The list of blobs.
*/
std::vector<vpBlobInfo> vpObjectBlobTracker::aggregateContours(const std::vector<std::vector<cv::Point> > contours) {
  std::vector<double> vectorOfContourAreas;
  std::vector<cv::Moments> mu(contours.size());
  for(size_t i = 0; i < contours.size(); i++) {
    double contour_area = cv::contourArea(contours[i]);
    vectorOfContourAreas.push_back(contour_area);
    mu[i] = moments(contours[i], false);
  }

  ///  Get the mass centers:
  std::vector<cv::Point2d> vectorOfContourCentroids(contours.size());
  for (size_t i = 0; i < contours.size(); i++) {
    vectorOfContourCentroids[i] = cv::Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
  }

  m_mapOfBlobTypes.clear();
  std::vector<vpBlobInfo> vectorOfProbableLightBlobs;

  bool aggregate = false;
  std::vector<size_t> vectorOfAlreadyCheck;
  for(size_t i = 0; i < contours.size(); i++) {
    if(std::find(vectorOfAlreadyCheck.begin(), vectorOfAlreadyCheck.end(), i) == vectorOfAlreadyCheck.end()) {
      double contour_area = cv::contourArea(contours[i]);
      cv::Rect boundingBoxCurrent = cv::boundingRect(contours[i]);

      aggregate = false;
      for(size_t j = 0; j < contours.size(); j++) {
        if(i != j) {
          cv::Rect boundingBoxNext = cv::boundingRect(contours[j]);
          cv::Rect rectIntersection = boundingBoxCurrent & boundingBoxNext;

          if(rectIntersection.width > (std::min)(boundingBoxNext.width, boundingBoxCurrent.width) / 2.0
              && rectIntersection.height > (std::min)(boundingBoxNext.height, boundingBoxCurrent.height) / 2.0) {
            if(boundingBoxNext.area() > boundingBoxCurrent.area()) {
              if(isInclusiveContour(contours[j], contours[i])) {
                aggregate = true;
                vectorOfAlreadyCheck.push_back(i);
                vectorOfAlreadyCheck.push_back(j);
                m_mapOfBlobTypes[AGGREGATE].push_back(contours[j]);
              }
            } else {
              if(isInclusiveContour(contours[i], contours[j])) {
                aggregate = true;
                vectorOfAlreadyCheck.push_back(i);
                vectorOfAlreadyCheck.push_back(j);
                m_mapOfBlobTypes[AGGREGATE].push_back(contours[j]);
              }
            }
          }
        }
      }

      if(aggregate) {
        m_mapOfBlobTypes[AGGREGATE].push_back(contours[i]);
      } else if(contour_area > m_minBlobArea && contour_area < m_maxBlobArea) {
        vpBlobInfo blobInfo;
        for(std::vector<cv::Point>::const_iterator it = contours[i].begin(); it != contours[i].end(); ++it) {
          blobInfo.m_blobContourPts.push_back(vpImagePoint(it->y, it->x));
        }
        blobInfo.m_blobArea = vectorOfContourAreas[i];
        blobInfo.m_imPts.push_back(vpImagePoint(vectorOfContourCentroids[i].y, vectorOfContourCentroids[i].x));
        vectorOfProbableLightBlobs.push_back(blobInfo);
        m_mapOfBlobTypes[GOOD_AREA].push_back(contours[i]);
      } else if(contour_area <= m_minBlobArea) {
        m_mapOfBlobTypes[TOO_SMALL].push_back(contours[i]);
      } else {
        m_mapOfBlobTypes[TOO_BIG].push_back(contours[i]);
      }
    }
  }

  return vectorOfProbableLightBlobs;
}

/*!
  Binarize the given image in order to detect the blobs.

  \param I_color : Color image to binarize.
  \param threshold : I_binarize[i][j] = 0 if grayscale(I_color[i][j]) < threshold otherwise 255.
  \param morphology : Morphology type to use (i.e. erosion, dilation, opening, etc.).
  \param kernelSize : Size of the kernel for the morphology operation.
  \param shape : Morphology shape to use (i.e. rectangle, cross or ellipse).
  \param nbIterations : Number of morphology operation iterations.
  \param useHSV : If true, will convert the RGB image to HSV and use the value channel as the binary image.
*/
vpImage<unsigned char> vpObjectBlobTracker::binarize(const vpImage<vpRGBa> &I_color, const unsigned char threshold,
    const MORPHOLOGY_TYPE morphology, const unsigned int kernelSize, const MORPHOLOGY_SHAPE_TYPE shape,
    const int nbIterations, const bool useHSV) {
  unsigned int w = I_color.getWidth(), h = I_color.getHeight();
  vpImage<unsigned char> I_binary(h, w);

  //Binarize the input color image for blob detection step
  if(useHSV) {
    //Binarize the value channel in the HSV color space
    unsigned int size = I_color.getSize();
    vpImage<unsigned char> I_hue(h, w), I_saturation(h, w);
    vpImageConvert::RGBaToHSV((unsigned char *) I_color.bitmap, I_hue.bitmap, I_saturation.bitmap, I_binary.bitmap, size);

    vpImageTools::binarise(I_binary, threshold, threshold, (unsigned char) 0, (unsigned char) 0, (unsigned char) 255, true);

    if(morphology != NO_MORPHOLOGY) {
      cv::Mat element = cv::getStructuringElement(shape,
          cv::Size(2*((int) kernelSize) + 1, 2*((int) kernelSize)+1),
          cv::Point((int) kernelSize, (int) kernelSize));

      // Apply the morphology operation
      cv::Mat matImg;
      vpImageConvert::convert(I_binary, matImg);
      cv::morphologyEx(matImg, matImg, morphology, element, cv::Point(-1, -1), nbIterations);
      vpImageConvert::convert(matImg, I_binary);
    }
  } else {
    vpImage<unsigned char> I;
    vpImageConvert::convert(I_color, I);
    I_binary = binarize(I, threshold, morphology, kernelSize, shape, nbIterations);
  }

  return I_binary;
}

/*!
  Binarize the given image in order to detect the blobs.

  \param I : Grayscale image to binarize.
  \param threshold : I_binarize[i][j] = 0 if I[i][j] < threshold otherwise 255.
  \param morphology : Morphology type to use (i.e. erosion, dilation, opening, etc.).
  \param kernelSize : Size of the kernel for the morphology operation.
  \param shape : Morphology shape to use (i.e. rectangle, cross or ellipse).
  \param nbIterations : Number of morphology operation iterations.
*/
vpImage<unsigned char> vpObjectBlobTracker::binarize(const vpImage<unsigned char> &I, const unsigned char threshold,
    const MORPHOLOGY_TYPE morphology, const unsigned int kernelSize, const MORPHOLOGY_SHAPE_TYPE shape,
    const int nbIterations) {
  vpImage<unsigned char> I_binary = I;

  //Binarize the input grayscale image for the blob detection step
  vpImageTools::binarise(I_binary, threshold, threshold, (unsigned char) 0, (unsigned char) 0, (unsigned char) 255, true);

  cv::Mat matImg;
  vpImageConvert::convert(I_binary, matImg);
  if(morphology != NO_MORPHOLOGY) {
    cv::Mat element = cv::getStructuringElement(shape,
        cv::Size(2*((int) kernelSize) + 1, 2*((int) kernelSize) + 1),
        cv::Point((int) kernelSize, (int) kernelSize));

    // Apply the morphology operation
    cv::morphologyEx(matImg, matImg, morphology, element, cv::Point(-1, -1), nbIterations);
    vpImageConvert::convert(matImg, I_binary);
  }

  return I_binary;
}

void vpObjectBlobTracker::computeHeatMap() {
  if(m_heatMap.getSize() == 0 || (int) m_heatMap.getWidth() != m_matImgBinarized.cols ||
      (int) m_heatMap.getHeight() != m_matImgBinarized.rows) {
    m_heatMap.resize((unsigned int) m_matImgBinarized.rows, (unsigned int) m_matImgBinarized.cols, 0);
  }

  for(int i = 0; i < m_matImgBinarized.rows; i++) {
    for(int j = 0; j < m_matImgBinarized.cols; j++) {
      if(m_contourImage.at<unsigned char>(i,j)) {
        m_heatMap[i][j] = vpMath::saturate<unsigned char>(m_heatMap[i][j] + 25);
      } else {
        m_heatMap[i][j] = vpMath::saturate<unsigned char>(m_heatMap[i][j] - 10);
      }
    }
  }
}

vpImagePoint vpObjectBlobTracker::computeMedianBlobFlow() {
  vpImagePoint medianFlow(0,0);

  std::vector<double> flowX, flowY;
  for(std::map<std::string, vpBlobInfo >::const_iterator it = m_mapOfObjectBlobs.begin(); it != m_mapOfObjectBlobs.end(); ++it) {
    if(it->second.m_statusHistory.back() == vpBlobInfo::NEAREST) {
      if(it->second.m_imPts.size() > 1) {
        vpImagePoint previousImPt = it->second.m_imPts[it->second.m_imPts.size()-2];
        flowX.push_back(it->second.m_imPts.back().get_u()-previousImPt.get_u());
        flowY.push_back(it->second.m_imPts.back().get_v()-previousImPt.get_v());
      }
    }
  }

  if(!flowX.empty() && !flowY.empty()) {
    medianFlow.set_uv(vpMath::getMedian(flowX), vpMath::getMedian(flowY));
  }
  return medianFlow;
}

/*!
  Compute the pose.

  \param cMo : The pose represented by an homogeneous matrix.
  \param func : Pointer to a function to check (if not null) if a given pose is ok or not.
*/
void vpObjectBlobTracker::computePose(bool (*func)(vpHomogeneousMatrix *)) {
  m_covarianceMatrixOk = false;

  vpPose pose;
  std::vector<cv::Point2d> imagePoints;
  std::vector<cv::Point3d> objectPoints;
  //Add the current list of detected object blobs
  for(std::map<std::string, vpBlobInfo >::iterator it = m_mapOfObjectBlobs.begin();
      it != m_mapOfObjectBlobs.end(); ++it) {
    if(it->second.m_statusHistory.back() == vpBlobInfo::NEAREST) {
      double x = 0.0, y = 0.0;
      vpPixelMeterConversion::convertPoint(m_cam, it->second.m_imPts.back(), x, y);
      it->second.m_pt.set_x(x);
      it->second.m_pt.set_y(y);

      pose.addPoint(it->second.m_pt);
      imagePoints.push_back(cv::Point2d(it->second.m_imPts.back().get_u(), it->second.m_imPts.back().get_v()));
      objectPoints.push_back(cv::Point3d(it->second.m_pt.get_oX(), it->second.m_pt.get_oY(), it->second.m_pt.get_oZ()));
    }
  }

  if(imagePoints.size() == objectPoints.size() && objectPoints.size() >= 4) {
    switch(m_poseEstimationMethod) {
    case VVS_POSE_ESTIMATION:
    {
      //Always use Dementhon or Lagrange to initialize the VVS pose estimation
      vpHomogeneousMatrix cMo_dementhon, cMo_lagrange;
      try {
        pose.computePose(vpPose::DEMENTHON, cMo_dementhon);
        pose.computePose(vpPose::LAGRANGE, cMo_lagrange);
      } catch(vpException &e) {
        m_usePreviousPoseGuess = false;
        throw e;
      }

//      double r_dementhon = std::numeric_limits<double>::max(), r_lagrange = std::numeric_limits<double>::max();
      double r_dementhon = DBL_MAX, r_lagrange = DBL_MAX;
      r_dementhon = pose.computeResidual(cMo_dementhon);
      r_lagrange = pose.computeResidual(cMo_lagrange);

      vpHomogeneousMatrix cMo = (r_dementhon < r_lagrange) ? cMo_dementhon : cMo_lagrange;

      try {
        pose.setCovarianceComputation(true);
        pose.computePose(vpPose::VIRTUAL_VS, cMo);
        m_covarianceMatrix = pose.getCovarianceMatrix();
        m_covarianceMatrixOk = true;
      } catch(vpException &e) {
        m_usePreviousPoseGuess = false;
//        m_covarianceMatrix = std::numeric_limits<double>::max();
        m_covarianceMatrix = DBL_MAX;
        throw e;
      }

      //Set the raw VVS pose estimation
      m_cMo_raw = cMo;
      //Set the current pose
      m_cMo = cMo;
    }
    break;

    case VVS_POSE_ESTIMATION_WITH_PREVIOUS_POSE_INITIAL_GUESS:
    {
      vpHomogeneousMatrix cMo;

      if(m_usePreviousPoseGuess) {
        if(func != NULL && func(&m_cMo)) {
          //Initialize the VVS with the previous pose
          cMo = m_cMo;
        } else if(func == NULL) {
          //Initialize the VVS with the previous pose
          cMo = m_cMo;
        } else {
          m_usePreviousPoseGuess = false;
        }
      }

      if(m_usePreviousPoseGuess) {
        try {
          pose.setCovarianceComputation(true);
          pose.computePose(vpPose::VIRTUAL_VS, cMo);
          m_covarianceMatrix = pose.getCovarianceMatrix();
          m_covarianceMatrixOk = true;
        } catch(vpException &e) {
          m_usePreviousPoseGuess = false;
//          m_covarianceMatrix = std::numeric_limits<double>::max();
          m_covarianceMatrix = DBL_MAX;
          throw e;
        }

        //Set the current pose
        m_cMo = cMo;
      } else {
        //Use Dementhon or Lagrange to initialize the VVS pose estimation
        vpHomogeneousMatrix cMo_dementhon, cMo_lagrange;
        try {
          pose.computePose(vpPose::DEMENTHON, cMo_dementhon);
          pose.computePose(vpPose::LAGRANGE, cMo_lagrange);
        } catch(vpException &e) {
          m_usePreviousPoseGuess = false;
          throw e;
        }

//        double r_dementhon = std::numeric_limits<double>::max(), r_lagrange = std::numeric_limits<double>::max();
        double r_dementhon = DBL_MAX, r_lagrange = DBL_MAX;
        r_dementhon = pose.computeResidual(cMo_dementhon);
        r_lagrange = pose.computeResidual(cMo_lagrange);

        vpHomogeneousMatrix cMo = (r_dementhon < r_lagrange) ? cMo_dementhon : cMo_lagrange;

        try {
          pose.setCovarianceComputation(true);
          pose.computePose(vpPose::VIRTUAL_VS, cMo);
          m_covarianceMatrix = pose.getCovarianceMatrix();
          m_covarianceMatrixOk = true;
        } catch(vpException &e) {
          m_usePreviousPoseGuess = false;
//          m_covarianceMatrix = std::numeric_limits<double>::max();
          m_covarianceMatrix = DBL_MAX;
          throw e;
        }

        //Set the raw VVS pose estimation
        m_cMo_raw = cMo;
        //Set the current pose
        m_cMo = cMo;
      }
    }
    break;

    case OPENCV_POSE_ESTIMATION:
    {
      //Compute the pose with OpenCV
      cv::Mat cameraMatrix =
          (cv::Mat_<double>(3, 3) << m_cam.get_px(), 0, m_cam.get_u0(), 0, m_cam.get_py(), m_cam.get_v0(), 0, 0, 1);

      cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
      cv::Mat rvec, tvec;
      cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false);

      vpTranslationVector translationVec = vpTranslationVector(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
      vpThetaUVector thetaUVector = vpThetaUVector(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));

      //Set the raw OpenCV pose estimation
      m_cMo_OpenCV = vpHomogeneousMatrix(translationVec, thetaUVector);
      //Set the current pose
      m_cMo = m_cMo_OpenCV;

      //Cannot compute the covariance matrix
//      m_covarianceMatrix = std::numeric_limits<double>::max();
      m_covarianceMatrix = DBL_MAX;
    }
    break;

    case OPENCV_POSE_ESTIMATION_WITH_PREVIOUS_POSE_INITIAL_GUESS:
    {
      vpHomogeneousMatrix cMo;

      if(m_usePreviousPoseGuess) {
        if(func != NULL && func(&m_cMo)) {
          //Initialize the VVS with the previous pose
          cMo = m_cMo;
        } else if(func == NULL) {
          //Initialize the VVS with the previous pose
          cMo = m_cMo;
        } else {
          m_usePreviousPoseGuess = false;
        }
      }

      vpTranslationVector translationVec;
      vpThetaUVector thetaUVector;
      cMo.extract(translationVec);
      cMo.extract(thetaUVector);
      cv::Mat tvec = (cv::Mat_<double>(3, 1) << translationVec[0], translationVec[1], translationVec[2]);
      cv::Mat rvec = (cv::Mat_<double>(3, 1) << thetaUVector[0], thetaUVector[1], thetaUVector[2]);

      //Compute the pose with OpenCV
      cv::Mat cameraMatrix =
          (cv::Mat_<double>(3, 3) << m_cam.get_px(), 0, m_cam.get_u0(), 0, m_cam.get_py(), m_cam.get_v0(), 0, 0, 1);

      cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
      cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, m_usePreviousPoseGuess);

      translationVec = vpTranslationVector(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
      thetaUVector = vpThetaUVector(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));

      //Set the raw OpenCV pose estimation
      m_cMo_OpenCV = vpHomogeneousMatrix(translationVec, thetaUVector);
      //Set the current pose
      m_cMo = m_cMo_OpenCV;

      //Cannot compute the covariance matrix
//      m_covarianceMatrix = std::numeric_limits<double>::max();
      m_covarianceMatrix = DBL_MAX;
    }
    break;

    default:
      throw vpException(vpException::fatalError, "Problem with switch(m_poseEstimationMethod) !");
      break;
    }


    if(func != NULL && func(&m_cMo)) {
      m_usePreviousPoseGuess = true;
    } else {
      m_usePreviousPoseGuess = false;
    }
  } else if(imagePoints.size() == objectPoints.size() && objectPoints.size() == 3) {
    //TODO: Add pose estimation using 3 points ?
    throw vpException(vpException::fatalError, "Not enough points to compute the pose !");
  } else {
    //Not enough points to compute the pose
    throw vpException(vpException::fatalError, "Not enough points to compute the pose !");
  }
}

void vpObjectBlobTracker::display(const vpImage<unsigned char> &I, const bool displayLegend) {
  vpImagePoint position_legend; // = vpImagePoint(I.getHeight()-(20*m_mapOfBlobs.size()), 20);
  double interline_size = 20, offset_x = 20;
  int inflate_x = 10, inflate_y = 10;

//  double minX = std::numeric_limits<double>::max(), minY = std::numeric_limits<double>::max();
  double minX = DBL_MAX, minY = DBL_MAX;
  double maxX = 0, maxY = 0;
  if(displayLegend) {
    for(std::map<std::string, vpBlobInfo>::const_iterator it = m_mapOfObjectBlobs.begin();
        it != m_mapOfObjectBlobs.end(); ++it) {
      vpImagePoint imPt = it->second.m_imPts.back();
      minX = imPt.get_u() < minX ? imPt.get_u() : minX;
      minY = imPt.get_v() < minY ? imPt.get_v() : minY;
      maxX = imPt.get_u() > maxX ? imPt.get_u() : maxX;
      maxY = imPt.get_v() > maxY ? imPt.get_v() : maxY;
    }

    //Get the model bounding box
    cv::Rect bounding_box((int) minX, (int) minY, (int) (maxX-minX), (int) (maxY-minY));


    //Find the best quadrant to display the legend
    cv::Rect top_left(0, 0, (int) (I.getWidth()/2.0 + inflate_x), (int) (I.getHeight()/2.0 + inflate_y));
    cv::Rect top_right((int) (I.getWidth()/2.0 - inflate_x), 0, (int) (I.getWidth()/2.0 + inflate_x), (int) (I.getHeight()/2.0 + inflate_y));
    cv::Rect bottom_right((int) (I.getWidth()/2.0 - inflate_x), (int) (I.getHeight()/2.0 - inflate_y),
        (int) (I.getWidth()/2.0 + inflate_x), (int) (I.getHeight()/2.0 + inflate_y));
    cv::Rect bottom_left(0, (int) (I.getHeight()/2.0 - inflate_y), (int) (I.getWidth()/2.0 + inflate_x), (int) (I.getHeight()/2.0 + inflate_y));

    cv::Rect rect_intersection1 = bounding_box & top_left;
    cv::Rect rect_intersection2 = bounding_box & top_right;
    cv::Rect rect_intersection3 = bounding_box & bottom_right;
    cv::Rect rect_intersection4 = bounding_box & bottom_left;

    //Best default position is bottom left
    if(rect_intersection4.area() == 0) {
      position_legend = vpImagePoint(I.getHeight() - (interline_size*m_mapOfObjectBlobs.size()), offset_x);
    } else if(rect_intersection1.area() == 0) {
      //Left display is better than right
      position_legend = vpImagePoint(I.getHeight()/2.0 - (interline_size*m_mapOfObjectBlobs.size()), offset_x);
    } else {
      int minArea = rect_intersection1.area();
      position_legend = vpImagePoint(I.getHeight()/2.0 - (interline_size*m_mapOfObjectBlobs.size()), offset_x);

      if(minArea > rect_intersection2.area()) {
        minArea = rect_intersection2.area();
        position_legend = vpImagePoint(I.getHeight()/2.0 - (interline_size*m_mapOfObjectBlobs.size()),
            I.getWidth()/2.0 + offset_x);
      }

      if(minArea > rect_intersection3.area()) {
        minArea = rect_intersection3.area();
        position_legend = vpImagePoint(I.getHeight() - (interline_size*m_mapOfObjectBlobs.size()),
            I.getWidth()/2.0 + offset_x);
      }

      if(minArea > rect_intersection4.area()) {
        minArea = rect_intersection4.area();
        position_legend = vpImagePoint(I.getHeight() - (interline_size*m_mapOfObjectBlobs.size()), offset_x);
      }
    }
  }

  for(std::map<std::string, vpBlobInfo>::const_iterator it = m_mapOfObjectBlobs.begin();
      it != m_mapOfObjectBlobs.end(); ++it) {
    vpColor legend_color = vpColor::gray, cross_color = vpColor::red;

    if(it->second.m_statusHistory.back() == vpBlobInfo::NEAREST) {
      legend_color = vpColor::red;
      cross_color = vpColor::red;
    } else if(it->second.m_statusHistory.back() == vpBlobInfo::PREDICTION ||
        it->second.m_statusHistory.back() == vpBlobInfo::PROJECTION) {
      legend_color = vpColor::orange;
      cross_color = vpColor::orange;
    } else if(it->second.m_statusHistory.back() == vpBlobInfo::PREDICTION) {
      legend_color = vpColor::green;
      cross_color = vpColor::green;
    } else if(it->second.m_statusHistory.back() == vpBlobInfo::PROJECTION) {
      legend_color = vpColor::orange;
      cross_color = vpColor::orange;
    }

    vpDisplay::displayCross(I, it->second.m_imPts.back(), 12, cross_color, 2);

    if(displayLegend) {
      vpImagePoint offset(-6, -10);
      size_t index_underscore = it->first.find_first_of("_");
      //Display numbers
      vpDisplay::displayText(I, it->second.m_imPts.back()+offset, it->first.substr(0, index_underscore), legend_color);

      std::string s = it->first;
      std::replace(s.begin(), s.end(), '_', ' ');
      s.insert(index_underscore, ")");
      //Display legend
      vpDisplay::displayText(I, position_legend, s, legend_color);

      position_legend.set_i(position_legend.get_i() + interline_size);
    }
  }
}

void vpObjectBlobTracker::display(const vpImage<vpRGBa> &I, const bool displayLegend) {
  vpImagePoint position_legend; // = vpImagePoint(I.getHeight()-(20*m_mapOfBlobs.size()), 20);
  double interline_size = 20.0, offset_x = 20.0;
  int inflate_x = 10, inflate_y = 10;

//  double minX = std::numeric_limits<double>::max(), minY = std::numeric_limits<double>::max();
  double minX = DBL_MAX, minY = DBL_MAX;
  double maxX = 0, maxY = 0;
  if(displayLegend) {
    for(std::map<std::string, vpBlobInfo>::const_iterator it = m_mapOfObjectBlobs.begin();
        it != m_mapOfObjectBlobs.end(); ++it) {
      vpImagePoint imPt = it->second.m_imPts.back();
      minX = imPt.get_u() < minX ? imPt.get_u() : minX;
      minY = imPt.get_v() < minY ? imPt.get_v() : minY;
      maxX = imPt.get_u() > maxX ? imPt.get_u() : maxX;
      maxY = imPt.get_v() > maxY ? imPt.get_v() : maxY;
    }

    //Get the model bounding box
    cv::Rect bounding_box((int) minX, (int) minY, (int) (maxX-minX), (int) (maxY-minY));


    //Find the best quadrant to display the legend
    cv::Rect top_left(0, 0, (int) (I.getWidth()/2.0 + inflate_x), (int) (I.getHeight()/2.0 + inflate_y));
    cv::Rect top_right((int) (I.getWidth()/2.0 - inflate_x), 0, (int) (I.getWidth()/2.0 + inflate_x), (int) (I.getHeight()/2.0 + inflate_y));
    cv::Rect bottom_right((int) (I.getWidth()/2.0 - inflate_x), (int) (I.getHeight()/2.0 - inflate_y),
        (int) (I.getWidth()/2.0 + inflate_x), (int) (I.getHeight()/2.0 + inflate_y));
    cv::Rect bottom_left(0, (int) (I.getHeight()/2.0 - inflate_y), (int) (I.getWidth()/2.0 + inflate_x), (int) (I.getHeight()/2.0 + inflate_y));

    cv::Rect rect_intersection1 = bounding_box & top_left;
    cv::Rect rect_intersection2 = bounding_box & top_right;
    cv::Rect rect_intersection3 = bounding_box & bottom_right;
    cv::Rect rect_intersection4 = bounding_box & bottom_left;

    //Best default position is bottom left
    if(rect_intersection4.area() == 0) {
      position_legend = vpImagePoint((double) I.getHeight() - (interline_size*m_mapOfObjectBlobs.size()), offset_x);
    } else if(rect_intersection1.area() == 0) {
      //Left display is better than right
      position_legend = vpImagePoint(I.getHeight()/2.0 - (interline_size*m_mapOfObjectBlobs.size()), offset_x);
    } else {
      int minArea = rect_intersection1.area();
      position_legend = vpImagePoint(I.getHeight()/2.0 - (interline_size*m_mapOfObjectBlobs.size()), offset_x);

      if(minArea > rect_intersection2.area()) {
        minArea = rect_intersection2.area();
        position_legend = vpImagePoint(I.getHeight()/2.0 - (interline_size*m_mapOfObjectBlobs.size()),
            I.getWidth()/2.0 + offset_x);
      }

      if(minArea > rect_intersection3.area()) {
        minArea = rect_intersection3.area();
        position_legend = vpImagePoint((double) I.getHeight() - (interline_size*m_mapOfObjectBlobs.size()),
            I.getWidth()/2.0 + offset_x);
      }

      if(minArea > rect_intersection4.area()) {
        minArea = rect_intersection4.area();
        position_legend = vpImagePoint((double) I.getHeight() - (interline_size*m_mapOfObjectBlobs.size()), offset_x);
      }
    }
  }

  for(std::map<std::string, vpBlobInfo>::const_iterator it = m_mapOfObjectBlobs.begin();
      it != m_mapOfObjectBlobs.end(); ++it) {
    vpColor legend_color = vpColor::gray, cross_color = vpColor::red;

    if(it->second.m_statusHistory.back() == vpBlobInfo::NEAREST) {
      vpDisplay::displayCross(I, it->second.m_imPts.back(), 12, vpColor::red, 2);
      legend_color = vpColor::red;
      cross_color = vpColor::red;
    } else if(it->second.m_statusHistory.back() == vpBlobInfo::PREDICTION ||
        it->second.m_statusHistory.back() == vpBlobInfo::PROJECTION) {
      legend_color = vpColor::orange;
      cross_color = vpColor::orange;
    } else if(it->second.m_statusHistory.back() == vpBlobInfo::PREDICTION) {
      legend_color = vpColor::green;
      cross_color = vpColor::green;
    } else if(it->second.m_statusHistory.back() == vpBlobInfo::PROJECTION) {
      legend_color = vpColor::orange;
      cross_color = vpColor::orange;
    }

    vpDisplay::displayCross(I, it->second.m_imPts.back(), 12, cross_color, 2);

    if(displayLegend) {
      vpImagePoint offset(-6, -10);
      size_t index_underscore = it->first.find_first_of("_");
      //Display numbers
      vpDisplay::displayText(I, it->second.m_imPts.back()+offset, it->first.substr(0, index_underscore), legend_color);

      std::string s = it->first;
      std::replace(s.begin(), s.end(), '_', ' ');
      s.insert(index_underscore, ")");
      //Display legend
      vpDisplay::displayText(I, position_legend, s, legend_color);

      position_legend.set_i(position_legend.get_i() + interline_size);
    }
  }
}

std::vector<vpBlobInfo> vpObjectBlobTracker::findBlobCentroid() {
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::Mat matImgBinarizedTmp;
  m_matImgBinarized.copyTo(matImgBinarizedTmp);
  cv::findContours(matImgBinarizedTmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  //Get the list of detected aggregated blobs
  std::vector<vpBlobInfo> vectorOfProbableObjectBlobs = aggregateContours(contours);
  if(m_contourImage.empty() || m_contourImage.size() != m_matImgBinarized.size()) {
    m_contourImage.create(m_matImgBinarized.size(), m_matImgBinarized.type());
  }
  m_contourImage = cv::Scalar(0);

  int i = 0;
  for(std::vector<std::vector<cv::Point> >::iterator it = m_mapOfBlobTypes[GOOD_AREA].begin();
      it != m_mapOfBlobTypes[GOOD_AREA].end(); ++it, i++) {
    cv::drawContours(m_contourImage, m_mapOfBlobTypes[GOOD_AREA], i, cv::Scalar(255), -1);
  }

  return vectorOfProbableObjectBlobs;
}

double vpObjectBlobTracker::getCoherentAngle(const double prevAngle, const double currentAngle) {
  if((prevAngle < 0) == (currentAngle < 0)) {
    return currentAngle;
  } else {
    if(prevAngle > M_PI_2) {
      return currentAngle + M_PI*2.0;
    } else if(prevAngle < -M_PI_2) {
      return currentAngle - M_PI*2.0;
    } else {
      return currentAngle;
    }
  }
}

vpHomogeneousMatrix vpObjectBlobTracker::getFilteredPose() const {
  // Filtered translation
  vpTranslationVector filteredTranslation;
  filteredTranslation[0] = m_poseKalmanFilter.Xest[0];
  filteredTranslation[1] = m_poseKalmanFilter.Xest[1];
  filteredTranslation[2] = m_poseKalmanFilter.Xest[2];

  // Filtered euler angles
  vpRzyxVector zyx_vector;
  zyx_vector.buildFrom(m_poseKalmanFilter.Xest[9], m_poseKalmanFilter.Xest[10], m_poseKalmanFilter.Xest[11]);

  vpRotationMatrix rotationMatrix(zyx_vector);
  vpHomogeneousMatrix cMo_filtered(filteredTranslation, rotationMatrix);

  return cMo_filtered;
}

bool vpObjectBlobTracker::getNearestBlob(const vpBlobInfo &predictedPoint,
    const std::vector<vpBlobInfo> &vectorOfProbableLightBlobs,
    vpImagePoint &nearest, double &min_dist, size_t &nearest_index, double &min_area,
    const std::vector<size_t> &alreadyProcessIndex, const double max_dist) {
//  min_dist = std::numeric_limits<double>::max();
  min_dist = DBL_MAX;
  bool find_nearest = false;

  size_t i = 0;
  for(std::vector<vpBlobInfo>::const_iterator it1 = vectorOfProbableLightBlobs.begin();
      it1 != vectorOfProbableLightBlobs.end(); ++it1, i++) {
    if(std::find(alreadyProcessIndex.begin(), alreadyProcessIndex.end(), i) == alreadyProcessIndex.end()) {

//      double dist = std::numeric_limits<double>::max();
      double dist = DBL_MAX;
      for(std::vector<vpImagePoint>::const_iterator it2 = it1->m_blobContourPts.begin();
          it2 != it1->m_blobContourPts.end(); ++it2) {
        double contour_dist = vpImagePoint::distance(*it2, predictedPoint.m_imPts.back());
        if(contour_dist < dist) {
          dist = contour_dist;
        }
      }

      if(predictedPoint.m_blobArea < 0) {
        if(dist <= max_dist && min_dist > dist) {
          find_nearest = true;
          min_dist = dist;
          min_area = it1->m_blobArea;
          nearest = it1->m_imPts.back();
          nearest_index = i;
        }
      } else {
        bool isSimilarBlob = (predictedPoint.m_blobArea * 0.5 <= it1->m_blobArea) &&
            (it1->m_blobArea <= predictedPoint.m_blobArea * 2.0);
        if(dist <= max_dist && min_dist > dist && isSimilarBlob) {
          find_nearest = true;
          min_dist = dist;
          min_area = it1->m_blobArea;
          nearest = it1->m_imPts.back();
          nearest_index = i;
        }
      }
    }
  }

  return find_nearest;
}

/*!
  Initialize the blob position by mouse clicking.

  \param I : Image where we will click on blob positions.
  \param filename : The path to the file that contains the list of the blobs visible in the image.
  \param refinePosition : If true, the 2D coordinate from mouse clicking will be refined using the blob centroid position.

  \return The list of 2D raw coordinates for each mouse click.
*/
std::vector<vpImagePoint> vpObjectBlobTracker::initClick(const vpImage<unsigned char> &I, const std::string &filename,
    const bool refinePosition) {
  std::vector<vpImagePoint> vectorOfClicks;
  std::map<std::string, vpImagePoint> mapOfInitPositions;
  bool askForClicks = true;

  //Check if an .init.pos file already exists
  std::string posFilename = filename + ".pos";
  if(vpIoTools::checkFilename(posFilename)) {
    std::ifstream f_pos(posFilename.c_str());

    if(f_pos.is_open()) {
      int nbPoints = 0;
      f_pos >> nbPoints;

      //Display the saved position
      vpDisplay::display(I);

      for(int cpt = 0; cpt < nbPoints;) {
        std::string blobName = "";
        double i = 0.0, j = 0.0;
        f_pos >> blobName >> i >> j;
        if(blobName.substr(0,1) != "#") {
          if(m_mapOfObjectBlobs.find(blobName) != m_mapOfObjectBlobs.end()) {
            mapOfInitPositions[blobName] = vpImagePoint(i, j);
            vectorOfClicks.push_back(vpImagePoint(i, j));
            vpDisplay::displayCross(I, (int) i, (int) j, 12, vpColor::red, 2);
          } else {
            std::cerr << "The point: " << blobName << " does not exist !" << std::endl;
          }
          cpt++;
        }
      }

      std::cout << "No modification : left click " << std::endl;
      std::cout << "Modify initial pose : right click " << std::endl ;

      vpDisplay::displayText(I, 15, 10,
                "left click to validate, right click to modify initial pose",
                vpColor::red);
      vpDisplay::flush(I);

      vpMouseButton::vpMouseButtonType button;
      vpDisplay::getClick(I, button, true);
      if(button == vpMouseButton::button1) {
        askForClicks = false;
      }
    } else {
      throw vpException(vpException::ioError, "Problem when opening file: " + posFilename);
    }
  }

  if(askForClicks) {
    //Read .init file that contains the name of the points to initialize
    std::ifstream f_init(filename.c_str());

    if(f_init.is_open()) {
      int nbPoints = 0;
      f_init >> nbPoints;

      std::vector<std::string> vectorOfPointNames((size_t) nbPoints);
      for(int cpt = 0; cpt < nbPoints;) {
        std::string blobName = "";
        f_init >> blobName;
        if(blobName.substr(0,1) != "#") {
          if(m_mapOfObjectBlobs.find(blobName) != m_mapOfObjectBlobs.end()) {
            vectorOfPointNames[(size_t) cpt] = blobName;
          } else {
            std::cerr << "The point: " << blobName << " does not exist !" << std::endl;
          }
          cpt++;
        }
      }

      while(askForClicks) {
        vectorOfClicks.clear();
        mapOfInitPositions.clear();

        vpDisplay::display(I);
        vpDisplay::flush(I);

        //Iterate over the names of the blobs to initialize
        for(std::vector<std::string>::const_iterator it1 = vectorOfPointNames.begin(); it1 != vectorOfPointNames.end(); ++it1) {
          vpDisplay::display(I);
          vpImagePoint imPt;
          vpDisplay::getClick(I, imPt, true);
          std::cout << "Click at " << imPt << std::endl;

          if(refinePosition) {
            vpImagePoint refineImPt = refinePositionWithCentroid(imPt);
            if(!vpMath::equal(refineImPt.get_i(), -1) && !vpMath::equal(refineImPt.get_j(), -1)) {
              imPt = refineImPt;
              std::cout << "Refine at " << imPt << std::endl;
            }
          }

          vectorOfClicks.push_back(imPt);
          mapOfInitPositions[*it1] = imPt;

          for(std::vector<vpImagePoint>::const_iterator it2 = vectorOfClicks.begin(); it2 != vectorOfClicks.end(); ++it2) {
            vpDisplay::displayCross(I, *it2, 8, vpColor::red, 2);
          }
          vpDisplay::flush(I);
        }

        //Ask if all the clicks are Ok
        vpDisplay::display(I);

        for(std::vector<vpImagePoint>::const_iterator it2 = vectorOfClicks.begin(); it2 != vectorOfClicks.end(); ++it2) {
          vpDisplay::displayCross(I, *it2, 8, vpColor::red, 2);
        }

        std::cout << "No modification : left click " << std::endl;
        std::cout << "Modify initial pose : right click " << std::endl ;

        vpDisplay::displayText(I, 15, 10,
                  "left click to validate, right click to modify initial pose",
                  vpColor::red);

        vpDisplay::flush(I);

        vpMouseButton::vpMouseButtonType button;
        vpDisplay::getClick(I, button, true);
        if(button == vpMouseButton::button1) {
          askForClicks = false;
        }
      }
    } else {
      throw vpException(vpException::ioError, "Problem when opening file: " + filename);
    }
  }

  //Initialize the blobs
  for(std::map<std::string, vpImagePoint>::const_iterator it = mapOfInitPositions.begin();
      it != mapOfInitPositions.end(); ++it) {
    std::string blobName = it->first;
    double i = it->second.get_i(), j = it->second.get_j();

    bool nameExist = m_mapOfObjectBlobs.find(blobName) != m_mapOfObjectBlobs.end();
    if(nameExist) {
      //Add the 2D coordinate
      m_mapOfObjectBlobs[blobName].m_imPts.push_back(vpImagePoint(i, j));

      //Change the status to nearest as we have initialized the 2D location
      //(it was set to NOT_TRACKED in loadModel)
      m_mapOfObjectBlobs[blobName].m_statusHistory.back() = vpBlobInfo::NEAREST;

      //Set the initial state vector
      m_mapOfObjectBlobs[blobName].m_kalmanFilter.Xest[0] = j;
      m_mapOfObjectBlobs[blobName].m_kalmanFilter.Xest[1] = i;
    }
  }

  //Save the click position in a file
  std::ofstream f_pos_save(posFilename.c_str());
  if(f_pos_save.is_open()) {
    //Write the number of points
    f_pos_save << mapOfInitPositions.size() << std::endl;

    for(std::map<std::string, vpImagePoint>::const_iterator it = mapOfInitPositions.begin();
        it != mapOfInitPositions.end(); ++it) {
      f_pos_save << it->first << " " << it->second.get_i() << " " << it->second.get_j() << std::endl;
    }
  } else {
    throw vpException(vpException::ioError, "Problem when opening file: " + posFilename);
  }

  //Compute the pose
  computePose();
  std::cout << "Init pose=\n" << m_cMo << std::endl;

  //Set the initial state vector
  vpTranslationVector translation_est;
  vpRotationMatrix rotationMatrix_est;
  m_cMo.extract(translation_est);
  m_cMo.extract(rotationMatrix_est);

  vpRzyxVector zyx_est(rotationMatrix_est);
  for(unsigned int i = 0; i < 3; i++) {
    m_poseKalmanFilter.Xest[i] = translation_est[i];
    m_poseKalmanFilter.Xest[i+9] = zyx_est[i];
  }

  return vectorOfClicks;
}

/*!
  Initialize the blob position by mouse clicking.

  \param I_color : Image where we will click on blob positions.
  \param filename : The path to the file that contains the list of the blobs visible in the image.
  \param refinePosition : If true, the 2D coordinate from mouse clicking will be refined using the blob centroid position.
  The binary image must be previously set in order to automatically find the blobs.

  \return The list of 2D raw coordinates for each mouse click.
*/
std::vector<vpImagePoint> vpObjectBlobTracker::initClick(const vpImage<vpRGBa> &I_color, const std::string &filename,
    const bool refinePosition) {
  std::vector<vpImagePoint> vectorOfClicks;
  std::map<std::string, vpImagePoint> mapOfInitPositions;
  bool askForClicks = true;

  //Check if an .init.pos file already exists
  std::string posFilename = filename + ".pos";
  if(vpIoTools::checkFilename(posFilename)) {
    std::ifstream f_pos(posFilename.c_str());

    if(f_pos.is_open()) {
      int nbPoints = 0;
      f_pos >> nbPoints;

      //Display the saved position
      vpDisplay::display(I_color);

      for(int cpt = 0; cpt < nbPoints;) {
        std::string blobName = "";
        double i = 0.0, j = 0.0;
        f_pos >> blobName >> i >> j;
        if(blobName.substr(0,1) != "#") {
          if(m_mapOfObjectBlobs.find(blobName) != m_mapOfObjectBlobs.end()) {
            mapOfInitPositions[blobName] = vpImagePoint(i, j);
            vectorOfClicks.push_back(vpImagePoint(i, j));
            vpDisplay::displayCross(I_color, (int) i, (int) j, 12, vpColor::red, 2);
          } else {
            std::cerr << "The point: " << blobName << " does not exist !" << std::endl;
          }
          cpt++;
        }
      }

      std::cout << "No modification : left click " << std::endl;
      std::cout << "Modify initial pose : right click " << std::endl ;

      vpDisplay::displayText(I_color, 15, 10,
                "left click to validate, right click to modify initial pose",
                vpColor::red);
      vpDisplay::flush(I_color);

      vpMouseButton::vpMouseButtonType button;
      vpDisplay::getClick(I_color, button, true);
      if(button == vpMouseButton::button1) {
        askForClicks = false;
      }
    } else {
      throw vpException(vpException::ioError, "Problem when opening file: " + posFilename);
    }
  }

  if(askForClicks) {
    //Read .init file that contains the name of the points to initialize
    std::ifstream f_init(filename.c_str());

    if(f_init.is_open()) {
      int nbPoints = 0;
      f_init >> nbPoints;

      std::vector<std::string> vectorOfPointNames((size_t) nbPoints);
      for(int cpt = 0; cpt < nbPoints;) {
        std::string blobName = "";
        f_init >> blobName;
        if(blobName.substr(0,1) != "#") {
          if(m_mapOfObjectBlobs.find(blobName) != m_mapOfObjectBlobs.end()) {
            vectorOfPointNames[(size_t) cpt] = blobName;
          } else {
            std::cerr << "The point: " << blobName << " does not exist !" << std::endl;
          }
          cpt++;
        }
      }

      while(askForClicks) {
        vectorOfClicks.clear();
        mapOfInitPositions.clear();

        vpDisplay::display(I_color);
        vpDisplay::flush(I_color);

        //Iterate over the names of the blobs to initialize
        for(std::vector<std::string>::const_iterator it1 = vectorOfPointNames.begin(); it1 != vectorOfPointNames.end(); ++it1) {
          vpDisplay::display(I_color);
          vpImagePoint imPt;
          vpDisplay::getClick(I_color, imPt, true);
          std::cout << "Click at " << imPt << std::endl;

          if(refinePosition) {
            vpImagePoint refineImPt = refinePositionWithCentroid(imPt);
            if(!vpMath::equal(refineImPt.get_i(), -1) && !vpMath::equal(refineImPt.get_j(), -1)) {
              imPt = refineImPt;
              std::cout << "Refine at " << imPt << std::endl;
            }
          }

          vectorOfClicks.push_back(imPt);
          mapOfInitPositions[*it1] = imPt;

          for(std::vector<vpImagePoint>::const_iterator it2 = vectorOfClicks.begin(); it2 != vectorOfClicks.end(); ++it2) {
            vpDisplay::displayCross(I_color, *it2, 8, vpColor::red, 2);
          }
          vpDisplay::flush(I_color);
        }

        //Ask if all the clicks are Ok
        vpDisplay::display(I_color);

        for(std::vector<vpImagePoint>::const_iterator it2 = vectorOfClicks.begin(); it2 != vectorOfClicks.end(); ++it2) {
          vpDisplay::displayCross(I_color, *it2, 8, vpColor::red, 2);
        }

        std::cout << "No modification : left click " << std::endl;
        std::cout << "Modify initial pose : right click " << std::endl ;

        vpDisplay::displayText(I_color, 15, 10,
                  "left click to validate, right click to modify initial pose",
                  vpColor::red);

        vpDisplay::flush(I_color);

        vpMouseButton::vpMouseButtonType button;
        vpDisplay::getClick(I_color, button, true);
        if(button == vpMouseButton::button1) {
          askForClicks = false;
        }
      }
    } else {
      throw vpException(vpException::ioError, "Problem when opening file: " + filename);
    }
  }

  //Initialize the blobs
  for(std::map<std::string, vpImagePoint>::const_iterator it = mapOfInitPositions.begin();
      it != mapOfInitPositions.end(); ++it) {
    std::string blobName = it->first;
    double i = it->second.get_i(), j = it->second.get_j();

    bool nameExist = m_mapOfObjectBlobs.find(blobName) != m_mapOfObjectBlobs.end();
    if(nameExist) {
      //Add the 2D coordinate
      m_mapOfObjectBlobs[blobName].m_imPts.push_back(vpImagePoint(i, j));

      //Change the status to nearest as we have initialized the 2D location
      //(it was set to NOT_TRACKED in loadModel)
      m_mapOfObjectBlobs[blobName].m_statusHistory.back() = vpBlobInfo::NEAREST;

      //Set the initial state vector
      m_mapOfObjectBlobs[blobName].m_kalmanFilter.Xest[0] = j;
      m_mapOfObjectBlobs[blobName].m_kalmanFilter.Xest[1] = i;
    }
  }

  //Save the click position in a file
  std::ofstream f_pos_save(posFilename.c_str());
  if(f_pos_save.is_open()) {
    //Write the number of points
    f_pos_save << mapOfInitPositions.size() << std::endl;

    for(std::map<std::string, vpImagePoint>::const_iterator it = mapOfInitPositions.begin();
        it != mapOfInitPositions.end(); ++it) {
      f_pos_save << it->first << " " << it->second.get_i() << " " << it->second.get_j() << std::endl;
    }
  } else {
    throw vpException(vpException::ioError, "Problem when opening file: " + posFilename);
  }

  //Compute the pose
  computePose();
  std::cout << "Init pose=\n" << m_cMo << std::endl;

  //Set the initial state vector
  vpTranslationVector translation_est;
  vpRotationMatrix rotationMatrix_est;
  m_cMo.extract(translation_est);
  m_cMo.extract(rotationMatrix_est);

  vpRzyxVector zyx_est(rotationMatrix_est);
  for(unsigned int i = 0; i < 3; i++) {
    m_poseKalmanFilter.Xest[i] = translation_est[i];
    m_poseKalmanFilter.Xest[i+9] = zyx_est[i];
  }

  return vectorOfClicks;
}

void vpObjectBlobTracker::initFromPose(const vpImage<unsigned char> &I_binary, const vpHomogeneousMatrix &cMo_init) {
  //Reset some variables, keep some other variables like the model info for example
  m_covarianceMatrix = DBL_MAX;
  m_covarianceMatrixOk = false;
  m_first = true;
  m_heatMap.resize(0,0);
  m_mapOfBlobTypes.clear();
  m_mapOfPredictedBlobPoints.clear();
  m_medianFlow = vpImagePoint(0,0);
  m_usePreviousPoseGuess = false;
  m_vectorOfProbableObjectBlobs.clear();

  //Set the binary image
  vpImageConvert::convert(I_binary, m_matImgBinarized);

  //Get the current blobs
  m_vectorOfProbableObjectBlobs = findBlobCentroid();

  std::vector<size_t> alreadyProcessIndex;
  //Project the object blobs using the inital pose
  for(std::map<std::string, vpBlobInfo>::iterator it = m_mapOfObjectBlobs.begin(); it != m_mapOfObjectBlobs.end(); ++it) {
    //Clear history
    it->second.m_imPts.clear();
    it->second.m_statusHistory.clear();

    //Clear contours
    it->second.m_statusHistory.clear();

    //Project
    it->second.m_pt.project(cMo_init);
    vpImagePoint imPt;
    vpMeterPixelConversion::convertPoint(m_cam, it->second.m_pt.get_x(), it->second.m_pt.get_y(), imPt);
    it->second.m_imPts.push_back(imPt);

    //Find nearest blob
    double minDist = DBL_MAX;
    size_t nearestIndex = 0;
    double minArea = 0.0;
    vpImagePoint nearestBlob;
    if(getNearestBlob(it->second, m_vectorOfProbableObjectBlobs, nearestBlob, minDist, nearestIndex,
        minArea, alreadyProcessIndex, m_maxDist)) {
      alreadyProcessIndex.push_back(nearestIndex);
     it->second.m_statusHistory.push_back(vpBlobInfo::NEAREST);
    } else {
      it->second.m_statusHistory.push_back(vpBlobInfo::PROJECTION);
    }

    //Set the initial state vector
    it->second.m_kalmanFilter.Xest[0] = imPt.get_u();
    it->second.m_kalmanFilter.Xest[1] = imPt.get_v();
    //Reset error covariance matrix ?
  }


  //Set the initial pose
  m_cMo = cMo_init;
  m_cMo_OpenCV = cMo_init;
  m_cMo_predicted = cMo_init;
  m_cMo_raw = cMo_init;


  //Set the initial state vector
  vpTranslationVector translation_est;
  vpRotationMatrix rotationMatrix_est;
  m_cMo.extract(translation_est);
  m_cMo.extract(rotationMatrix_est);

  vpRzyxVector zyx_est(rotationMatrix_est);
  for(unsigned int i = 0; i < 3; i++) {
    m_poseKalmanFilter.Xest[i] = translation_est[i];
    m_poseKalmanFilter.Xest[i+9] = zyx_est[i];
    //Reset error covariance matrix ?
  }
}

/*!
  Initialize the Kalman filter for the pose.
*/
void vpObjectBlobTracker::initPoseKalmanFilter() {
  unsigned int nStates = 18;         // the number of states
  unsigned int nMeasurements = 6;    // the number of measured states
  double dt = 1.0 / m_fps;  // time between measurements (1/FPS)

//  m_poseKalmanFilter.verbose(true);
  m_poseKalmanFilter.init(nStates, nMeasurements, (unsigned int) 1);
  vpMatrix Q(nStates, nStates);
  Q.diag(1e-5);
  m_poseKalmanFilter.Q = Q;

  vpMatrix R(nMeasurements, nMeasurements);
  R.diag(1e-4);
  m_poseKalmanFilter.R = R;

  vpMatrix Ppost(nStates, nStates);
  Ppost.diag(1.0);
  m_poseKalmanFilter.Pest = Ppost;

  /* DYNAMIC MODEL */
// [1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0 0 0]
// [0 1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0 0]
// [0 0 1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0]
// [0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2 0 0]
// [0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2 0]
// [0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2]
// [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
  vpMatrix F(nStates, nStates);
  F.diag();

// position
  F[0][3] = dt; //x
  F[1][4] = dt; //y
  F[2][5] = dt; //z
  F[3][6] = dt; //v_x
  F[4][7] = dt; //v_y
  F[5][8] = dt; //v_z
  F[0][6] = 0.5 * pow(dt, 2); // a_x
  F[1][7] = 0.5 * pow(dt, 2); // a_y
  F[2][8] = 0.5 * pow(dt, 2); // a_z

// orientation
  F[9][12] = dt; // phi
  F[10][13] = dt; // theta
  F[11][14] = dt; // psi
  F[12][15] = dt; // phi_d
  F[13][16] = dt; // theta_d
  F[14][17] = dt; // psi_d
  F[9][15] = 0.5 * pow(dt, 2); // phi_dd
  F[10][16] = 0.5 * pow(dt, 2); // theta_dd
  F[11][17] = 0.5 * pow(dt, 2); // psi_dd

  m_poseKalmanFilter.F = F;

  /* MEASUREMENT MODEL */
// [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
// [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
// [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
  vpMatrix H(nMeasurements, nStates);
  H[0][0] = 1; // x
  H[1][1] = 1; // y
  H[2][2] = 1; // z
  H[3][9] = 1; // roll
  H[4][10] = 1; // pitch
  H[5][11] = 1; // yaw

  m_poseKalmanFilter.H = H;
}

/*!
  Check if a given contour is likely to belong to another contours.
  Two contours are likely to belong to each other if at least a minimum percentage of points in the first contour
  have their angles with the points in the second contour comprise in a wide range.

  \param contours1 : The first contour.
  \param contours2 : The second contour.
  \param inclusivePercentage : Percentage of inclusion.
  \param minAngleDeviation : Percentage of inclusion.
  \param distFactor : Percentage of inclusion.
*/
bool vpObjectBlobTracker::isInclusiveContour(const std::vector<cv::Point> &contours1, const std::vector<cv::Point> &contours2,
    const double inclusivePercentage, const double minAngleDeviation, const double distFactor) {
  std::vector<cv::Point> approx_contours1, approx_contours2;

  cv::approxPolyDP(contours1, approx_contours1, 3.0, true);
  cv::approxPolyDP(contours2, approx_contours2, 3.0, true);

  cv::Rect boundingBox1 = cv::boundingRect(contours1);
  double max_dist = (std::max)(boundingBox1.width, boundingBox1.height) / distFactor;

  int nb = 0;
  for(std::vector<cv::Point>::const_iterator it1 = approx_contours2.begin(); it1 != approx_contours2.end(); ++it1) {
    std::vector<double> vectorOfAngles;
    for(std::vector<cv::Point>::const_iterator it2 = approx_contours1.begin(); it2 != approx_contours1.end(); ++it2) {
      if(pointDistance(*it1, *it2) < max_dist) {
        double angle = std::atan2((double) (it2->y-it1->y), (double) (it2->x-it1->x));
        vectorOfAngles.push_back(angle);
      }

      if(vectorOfAngles.size() > 1) {
        std::sort(vectorOfAngles.begin(), vectorOfAngles.end());
        if(vpMath::deg(vectorOfAngles.back()-vectorOfAngles.front()) > minAngleDeviation) {
          nb++;
        }
      }
    }
  }

  if(nb / (double)approx_contours2.size() > inclusivePercentage) {
    return true;
  }

  return false;
}

/*!
  Load the list of 3D points to track.

  \param filename : The path to the file that contains the list of 3D points to track.
*/
void vpObjectBlobTracker::loadModel(const std::string &filename) {
  m_mapOfObjectBlobs.clear();
  std::ifstream f(filename.c_str());

  if(f.is_open()) {
    int nbPoints = 0;
    f >> nbPoints;

    for(int i = 0; i < nbPoints;) {
      std::string blobName = "";
      double oX = 0.0, oY = 0.0, oZ = 0.0;
      f >> blobName >> oX >> oY >> oZ;
      if(blobName.substr(0,1) != "#") {
        i++;
        m_mapOfObjectBlobs[blobName].m_pt.setWorldCoordinates(oX, oY, oZ);
        //Set the status to non-track as there is no 2D information for now
        m_mapOfObjectBlobs[blobName].m_statusHistory.push_back(vpBlobInfo::NOT_TRACKED);
      }
    }
  }

  for(std::map<std::string, vpBlobInfo>::iterator it = m_mapOfObjectBlobs.begin(); it != m_mapOfObjectBlobs.end(); ++it) {
    //Initialize the 2D Kalman filter for each point
    double delta_t = 1.0;
    vpKalmanFilter kalmanFilter(6, 2, 1);
    vpMatrix F(6,6);
    F.diag();
    F[0][2] = delta_t;  F[0][4] = 0.5*delta_t*delta_t;
    F[1][3] = delta_t;  F[1][5] = 0.5*delta_t*delta_t;
    F[2][4] = delta_t;
    F[3][5] = delta_t;
    kalmanFilter.F = F;

    vpMatrix H(2,6);
    H[0][0] = 1.0;
    H[1][1] = 1.0;
    kalmanFilter.H = H;

    vpMatrix Q(6,6);
    Q.diag(1e-4);
    kalmanFilter.Q = Q;

    vpMatrix R(2,2);
    R.diag(1e-1);
    kalmanFilter.R = R;

    vpMatrix Ppost(6,6);
    Ppost.diag(0.1);
    kalmanFilter.Pest = Ppost;

    it->second.m_kalmanFilter = kalmanFilter;
  }
}

/*!
  Compute the distance between two cv::Point.

  \param pt1 : The first point.
  \param pt2 : The second point.

  \return The distance between the two points.
*/
double vpObjectBlobTracker::pointDistance(const cv::Point &pt1, const cv::Point &pt2) {
  return sqrt( (double) ( (pt1.x-pt2.x)*(pt1.x-pt2.x) + (pt1.y-pt2.y)*(pt1.y-pt2.y) ) );
}

/*!
  Given a 2D coordinate, this method will return the coordinate of the centroid for the corresponding blob.

  \param pt : 2D coordinate.

  \return The coordinate of the centroid for the corresponding blob.
*/
vpImagePoint vpObjectBlobTracker::refinePositionWithCentroid(const vpImagePoint &pt) {
  std::vector<vpBlobInfo> vectorOfBlobs = findBlobCentroid();
  vpImagePoint centroid(-1, -1);

  //Find the blob that contains the point
  for(std::vector<vpBlobInfo>::const_iterator it = vectorOfBlobs.begin(); it != vectorOfBlobs.end(); ++it) {
    vpPolygon polygon;
    polygon.buildFrom(it->m_blobContourPts);
    if(polygon.isInside(pt)) {
      //Calculate the centroid coordinate
      std::vector<cv::Point2f> contours(it->m_blobContourPts.size());
      for(size_t i = 0; i < it->m_blobContourPts.size(); i++) {
        contours[i] = cv::Point2f((float) it->m_blobContourPts[i].get_u(), (float) it->m_blobContourPts[i].get_v());
      }
      cv::Moments moment = cv::moments(contours, false);
      cv::Point2d cv_centroid = cv::Point2d(moment.m10 / moment.m00, moment.m01 / moment.m00);
      centroid = vpImagePoint(cv_centroid.y, cv_centroid.x);
      return centroid;
    }
  }

  return centroid;
}

/*!
  Set the number of frame per second.

  \param fps : Frame per second.
*/
void vpObjectBlobTracker::setFps(const double fps) {
  m_fps = fps;
  setPoseKalmanFilterTimeUpdate();
}

/*!
  Set the maximum number of predictions when an object is lost.

  \param nb : Maximum number of predictions.
*/
void vpObjectBlobTracker::setMaxNbPrediction(const size_t nb) {
  for(std::map<std::string, vpBlobInfo>::iterator it = m_mapOfObjectBlobs.begin(); it != m_mapOfObjectBlobs.end(); ++it) {
    size_t old_max = it->second.m_maxHistory;
    it->second.m_maxHistory = nb;

    if(nb < old_max) {
      if(it->second.m_imPts.size() > nb) {
        it->second.m_imPts.erase(it->second.m_imPts.begin(), it->second.m_imPts.begin() +
            ((int) it->second.m_imPts.size() - (int) nb));
      }

      if(it->second.m_statusHistory.size() > nb) {
        it->second.m_statusHistory.erase(it->second.m_statusHistory.begin(), it->second.m_statusHistory.begin() +
            ((int) it->second.m_statusHistory.size() - (int) nb));
      }
    }
  }
}

void vpObjectBlobTracker::setPoseKalmanFilterTimeUpdate() {
  unsigned int nStates = 18;         // the number of states
  double dt = 1.0 / m_fps;  // time between measurements (1/FPS)

  /* DYNAMIC MODEL */
// [1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0 0 0]
// [0 1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0 0]
// [0 0 1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0]
// [0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
// [0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2 0 0]
// [0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2 0]
// [0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2]
// [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
  vpMatrix F(nStates, nStates);
  F.diag();

// position
  F[0][3] = dt; //x
  F[1][4] = dt; //y
  F[2][5] = dt; //z
  F[3][6] = dt; //v_x
  F[4][7] = dt; //v_y
  F[5][8] = dt; //v_z
  F[0][6] = 0.5 * pow(dt, 2); // a_x
  F[1][7] = 0.5 * pow(dt, 2); // a_y
  F[2][8] = 0.5 * pow(dt, 2); // a_z

// orientation
  F[9][12] = dt; // phi
  F[10][13] = dt; // theta
  F[11][14] = dt; // psi
  F[12][15] = dt; // phi_d
  F[13][16] = dt; // theta_d
  F[14][17] = dt; // psi_d
  F[9][15] = 0.5 * pow(dt, 2); // phi_dd
  F[10][16] = 0.5 * pow(dt, 2); // theta_dd
  F[11][17] = 0.5 * pow(dt, 2); // psi_dd

  m_poseKalmanFilter.F = F;
}

/*!
  Track a list of points which are reprojected by blobs in the image plane.

  \param I_color : Color image.
  \param threshold : I_binarize[i][j] = 0 if grayscale(I_color[i][j]) < threshold otherwise 255.
  \param morphology : Morphology type to use (i.e. erosion, dilation, opening, etc.).
  \param kernelSize : Size of the kernel for the morphology operation.
  \param shape : Morphology shape to use (i.e. rectangle, cross or ellipse).
  \param nbIterations : Number of morphology operation iterations.
  \param useHSV : If true, will convert the RGB image to HSV and use the value channel as the binary image.
  \param func : Pointer to a function which permits to check (if not null) if a given pose is ok or not.
  \param predictionType : Type of prediction to used.
*/
void vpObjectBlobTracker::track(const vpImage<vpRGBa> &I_color, const unsigned char threshold,
    const MORPHOLOGY_TYPE morphology, const unsigned int kernelSize, const MORPHOLOGY_SHAPE_TYPE shape,
    const int nbIterations, const bool useHSV, bool (*func)(vpHomogeneousMatrix *), const PREDICTION_TYPE &predictionType) {
  // Binarize the RGB image
  vpImage<unsigned char> I_binary = binarize(I_color, threshold, morphology, kernelSize, shape, nbIterations, useHSV);
  vpImageConvert::convert(I_binary, m_matImgBinarized);
  track(I_binary, false, threshold, morphology, kernelSize, shape, nbIterations, func, predictionType);
}

/*!
  Track a list of points which are reprojected by blobs in the image plane.

  \param I : Grayscale image.
  \param doBinarize : If true, the input image will be binarize, otherwise it is already a binary image.
  \param threshold : I_binarize[i][j] = 0 if I[i][j] < threshold otherwise 255.
  \param morphology : Morphology type to use (i.e. erosion, dilation, opening, etc.).
  \param kernelSize : Size of the kernel for the morphology operation.
  \param shape : Morphology shape to use (i.e. rectangle, cross or ellipse).
  \param nbIterations : Number of morphology operation iterations.
  \param func : Pointer to a function which permits to check (if not null) if a given pose is ok or not.
  \param predictionType : Type of prediction to used.
*/
void vpObjectBlobTracker::track(const vpImage<unsigned char> &I, const bool doBinarize, const unsigned char threshold,
    const MORPHOLOGY_TYPE morphology, const unsigned int kernelSize, const MORPHOLOGY_SHAPE_TYPE shape,
    const int nbIterations, bool (*func)(vpHomogeneousMatrix *), const PREDICTION_TYPE &predictionType) {
  if(doBinarize) {
    vpImage<unsigned char> I_binary = binarize(I, threshold, morphology, kernelSize, shape, nbIterations);
    vpImageConvert::convert(I_binary, m_matImgBinarized);
  } else {
    vpImageConvert::convert(I, m_matImgBinarized);
  }

  //Find blob centroids in the current image
  m_vectorOfProbableObjectBlobs = findBlobCentroid();


  //Prediction step for each blob Kalman filters
  for(std::map<std::string, vpBlobInfo>::iterator it = m_mapOfObjectBlobs.begin();
      it != m_mapOfObjectBlobs.end(); ++it) {
    if(it->second.m_statusHistory.back() != vpBlobInfo::NOT_TRACKED) {
      it->second.m_kalmanFilter.prediction();
    }
  }


  //Predict the pose using the Kalman filter
  m_poseKalmanFilter.prediction();
  vpColVector Xpre = m_poseKalmanFilter.Xpre;
  vpColVector Xpost = m_poseKalmanFilter.Xest;

  vpColVector state_predicted = m_poseKalmanFilter.Xpre;
  vpTranslationVector translation_predicted;
  vpRzyxVector zyx_predicted;
  for(unsigned int i = 0; i < 3; i++) {
    translation_predicted[i] = state_predicted[i];
    zyx_predicted[i] = state_predicted[i+9];
  }

  vpRotationMatrix rotationMatrix_predicted(zyx_predicted);
  m_cMo_predicted.buildFrom(translation_predicted, rotationMatrix_predicted);


  if(m_first) {
    m_first = false;
    updateBlobPositions(m_mapOfObjectBlobs, m_vectorOfProbableObjectBlobs, vpObjectBlobTracker::PROJECTION);
  } else {
    updateBlobPositions(m_mapOfObjectBlobs, m_vectorOfProbableObjectBlobs, predictionType);
  }


  //Compute the pose with detected and matched blobs
  computePose(func);


  //Correct the pose Kalman filter
  //Get the measured translation and rotation
  vpColVector z_measurement(6);
  vpRotationMatrix rotationMatrix_measured(m_cMo);
  vpRzyxVector zyx_measured(rotationMatrix_measured);
  for(unsigned int i = 0; i < 3; i++) {
    z_measurement[i] = m_cMo[i][3];
    z_measurement[i+3] = getCoherentAngle(zyx_predicted[i], zyx_measured[i]);
  }

  updatePoseKalmanFilter(z_measurement);

  vpColVector state_corrected = m_poseKalmanFilter.Xest;
  vpRzyxVector zyx_corrected;
  for(unsigned int i = 0; i < 3; i++) {
    zyx_corrected[i] = state_corrected[i+9];
  }


  //Update the median flow vector
  m_medianFlow = computeMedianBlobFlow();

  //Add blob index that are already matched
  std::vector<size_t> alreadyProcessIndex;
  for(std::map<std::string, vpBlobInfo>::iterator it = m_mapOfObjectBlobs.begin();
      it != m_mapOfObjectBlobs.end(); ++it) {
    if(it->second.m_statusHistory.back() == vpBlobInfo::NEAREST) {
      alreadyProcessIndex.push_back(it->second.m_blobIndex);
    }
  }

  //Part where we try to reinitialize the lost blobs
  //Recover position
  for(std::map<std::string, vpBlobInfo>::iterator it = m_mapOfObjectBlobs.begin();
      it != m_mapOfObjectBlobs.end(); ++it) {
    bool updateKalmanFilter = false;

    if(it->second.m_statusHistory.back() == vpBlobInfo::PREDICTION) {
      //We did not find a blob that matches the location of the predicted point
      //so we try to check if there is a blob that could match the projection of this point
      vpPoint pt = it->second.m_pt;
      pt.project(m_cMo);

      double u = 0.0, v = 0.0;
      vpMeterPixelConversion::convertPoint(m_cam, pt.get_x(), pt.get_y(), u, v);
      vpImagePoint projectedImPt(v, u), nearest;

      double min_dist;
      size_t index = 0;
      double area = 0.0;
      vpBlobInfo projectedBlobInfo;
      projectedBlobInfo.m_imPts.push_back(projectedImPt);
      projectedBlobInfo.m_blobArea = it->second.m_blobArea;
      if(getNearestBlob(projectedBlobInfo, m_vectorOfProbableObjectBlobs, nearest, min_dist, index, area, alreadyProcessIndex)) {
        it->second.m_imPts.back() = nearest;
        it->second.m_blobArea = area;
        it->second.m_statusHistory.back() = vpBlobInfo::NEAREST;
        updateKalmanFilter = true;
      } else if(predictionType == MEDIAN_FLOW) {
        //Update the prediction location using the current computed median flow vector
        vpImagePoint previousImPt = it->second.m_imPts[it->second.m_imPts.size()-2];
        it->second.m_imPts.back() = previousImPt + m_medianFlow;
        it->second.m_blobArea = -1.0;
        updateKalmanFilter = true;
      }
    } else if(it->second.m_statusHistory.back() == vpBlobInfo::PROJECTION) {
      //Project the point using the current computed cMo
      vpPoint pt = it->second.m_pt;
      pt.project(m_cMo);

      double u = 0.0, v = 0.0;
      vpMeterPixelConversion::convertPoint(m_cam, pt.get_x(), pt.get_y(), u, v);
      it->second.m_imPts.back().set_uv(u, v);
      it->second.m_statusHistory.back() = vpBlobInfo::PROJECTION;
      it->second.m_blobArea = -1.0;

      //Try to see if there is a blob that would match the projected point
      vpImagePoint nearest;
      double min_dist;
      size_t index = 0;
      double area = 0.0;
      vpBlobInfo projectedBlobInfo;
      projectedBlobInfo.m_imPts.push_back(it->second.m_imPts.back());
      projectedBlobInfo.m_blobArea = it->second.m_blobArea;
      if(getNearestBlob(projectedBlobInfo, m_vectorOfProbableObjectBlobs, nearest, min_dist, index, area, alreadyProcessIndex)) {
        updateKalmanFilter = true;
        it->second.m_imPts.back() = nearest;
        it->second.m_blobArea = area;
        it->second.m_statusHistory.back() = vpBlobInfo::NEAREST;
      }
    } else if(it->second.m_statusHistory.back() == vpBlobInfo::NOT_TRACKED) {
      //Project the point using the computed cMo
      vpPoint pt = it->second.m_pt;
      pt.project(m_cMo);

      double u = 0.0, v = 0.0;
      vpMeterPixelConversion::convertPoint(m_cam, pt.get_x(), pt.get_y(), u, v);
      it->second.m_imPts.push_back(vpImagePoint(v,u));
      it->second.m_statusHistory.back() = vpBlobInfo::PROJECTION;
      //Should not be necessary
//      it->second.m_blobArea = -1.0;

      it->second.m_kalmanFilter.Xest[0] = u;
      it->second.m_kalmanFilter.Xest[1] = v;
    }

    if(updateKalmanFilter) {
      //Update the 2D Kalman filter
      vpColVector z_measurement(2);
      z_measurement[0] = it->second.m_imPts.back().get_u();
      z_measurement[1] = it->second.m_imPts.back().get_v();
      it->second.m_kalmanFilter.filtering(z_measurement);
    }
  }

  //If a lost point is predicted for a too long time, set its status to projection
  for(std::map<std::string, vpBlobInfo>::iterator it1 = m_mapOfObjectBlobs.begin();
      it1 != m_mapOfObjectBlobs.end(); ++it1) {
    bool onlyPrediction = true;
    for(std::vector<vpBlobInfo::BLOB_TYPE>::const_iterator it2 = it1->second.m_statusHistory.begin();
        it2 != it1->second.m_statusHistory.end(); ++it2) {
      if(*it2 != vpBlobInfo::PREDICTION) {
        onlyPrediction = false;
        break;
      }
    }

    if(onlyPrediction) {
      it1->second.m_statusHistory.back() = vpBlobInfo::PROJECTION;
      it1->second.m_blobArea = -1.0;
    }
  }

  if(m_useComputeHeatMap) {
    computeHeatMap();
  }

  //Erase first element if the size is too  big
  for(std::map<std::string, vpBlobInfo>::iterator it = m_mapOfObjectBlobs.begin(); it != m_mapOfObjectBlobs.end(); ++it) {
    it->second.cleanVectors();
  }
}

void vpObjectBlobTracker::updateBlobPositions(std::map<std::string, vpBlobInfo > &mapOfBlobs,
    const std::vector<vpBlobInfo> &vectorOfProbableLightBlobs, const PREDICTION_TYPE predictionType) {
  std::map<std::string, vpImagePoint> mapOfPredictedImPts;
  for(std::map<std::string, vpBlobInfo>::iterator it = mapOfBlobs.begin();
      it != mapOfBlobs.end(); ++it) {

    if(it->second.m_statusHistory.back() != vpBlobInfo::NOT_TRACKED) {
      vpImagePoint predictedPoint;
      switch (predictionType) {
        case vpObjectBlobTracker::NO_PREDICTION:
          //Predicted point == last 2D position
          predictedPoint = it->second.m_imPts.back();
          break;

        case vpObjectBlobTracker::KALMAN_2D:
          //Predicted point == prediction step of the 2D kalman filter
          predictedPoint.set_uv(it->second.m_kalmanFilter.Xest[0], it->second.m_kalmanFilter.Xest[1]);
          break;

        case vpObjectBlobTracker::KALMAN_3D:
          {
            //Predicted point == projection of the 3D point using the cMo from the prediction step of the
            //3D Kalman filter
            vpColVector state_predicted = m_poseKalmanFilter.Xpre;
            vpTranslationVector translation_predicted;
            vpRzyxVector zyx_predicted;
            for(unsigned int i = 0; i < 3; i++) {
              translation_predicted[i] = state_predicted[i];
              zyx_predicted[i] = state_predicted[i+9];
            }
            vpRotationMatrix rotationMatrix_predicted(zyx_predicted);
            vpHomogeneousMatrix cMo_predict(translation_predicted, rotationMatrix_predicted);

            vpPoint modelPt = it->second.m_pt;
            modelPt.project(cMo_predict);
            double u = 0.0, v = 0.0;
            vpMeterPixelConversion::convertPoint(m_cam, modelPt.get_x(), modelPt.get_y(), u, v);
            vpImagePoint projectedPt(v, u);
            predictedPoint = projectedPt;
          }
          break;

        case vpObjectBlobTracker::PROJECTION:
          {
            //Predicted point == projection of the 3D point using the last cMo
            vpPoint modelPt = it->second.m_pt;
            modelPt.project(m_cMo);
            double u = 0.0, v = 0.0;
            vpMeterPixelConversion::convertPoint(m_cam, modelPt.get_x(), modelPt.get_y(), u, v);
            vpImagePoint projectedPt(v, u);
            predictedPoint = projectedPt;
          }
          break;

        case vpObjectBlobTracker::CONSTANT_VELOCITIE:
          //Predicted point == last position + last velocity
          predictedPoint = it->second.m_imPts.back();

          if(it->second.m_imPts.size() >= 2) {
            vpImagePoint shift = it->second.m_imPts[it->second.m_imPts.size()-1] -
                it->second.m_imPts[it->second.m_imPts.size()-2];
            predictedPoint = it->second.m_imPts[it->second.m_imPts.size()-1] + shift;
          }
          break;

        case vpObjectBlobTracker::MEDIAN_FLOW:
          //Use the median flow calculated in the previous iteration to predict the current location
          predictedPoint = it->second.m_imPts[it->second.m_imPts.size()-1] + m_medianFlow;
          break;

        default:
          std::cerr << "Problem with the chosen type of prediction !" << std::endl;
          break;
      }

      mapOfPredictedImPts[it->first] = predictedPoint;
      m_mapOfPredictedBlobPoints[it->first] = predictedPoint;
    }
  }

  std::vector<size_t> alreadyProcessIndex;
  std::vector<std::string> alreadyProcessName;
  for(size_t i = 0; i < mapOfBlobs.size(); i++) {
    std::vector<matching_info_t> vectorOfPotentialMatches;

    //Calculate all the possible matches
    for(std::map<std::string, vpBlobInfo>::iterator it = mapOfBlobs.begin();
        it != mapOfBlobs.end(); ++it) {
      if(it->second.m_statusHistory.back() != vpBlobInfo::NOT_TRACKED) {
        if(std::find(alreadyProcessName.begin(), alreadyProcessName.end(), it->first) == alreadyProcessName.end()) {
          vpImagePoint predictedPoint = mapOfPredictedImPts[it->first];
          vpImagePoint nearest;
          double dist;
          size_t index;
          double area = 0.0;

          vpBlobInfo predictedBlobInfo;
          predictedBlobInfo.m_imPts.push_back(predictedPoint);
          predictedBlobInfo.m_blobArea = it->second.m_blobArea;
          bool isNearestExist = getNearestBlob(predictedBlobInfo, vectorOfProbableLightBlobs, nearest, dist, index, area,
              alreadyProcessIndex, m_maxDist);
          if(isNearestExist) {
            matching_info_t m(it->first, dist, index, nearest, area);
            vectorOfPotentialMatches.push_back(m);
          } else {
            matching_info_t m(it->first, -1.0, index, predictedPoint, it->second.m_blobArea);
            vectorOfPotentialMatches.push_back(m);
          }
        }
      }
    }

    //Sort the possible matches
    std::sort(vectorOfPotentialMatches.begin(), vectorOfPotentialMatches.end(), sortMatching);

    if(vectorOfPotentialMatches.size() > 0) {
      //Get best match (with the smaller distance between the predicted and the corresponding blob)
      matching_info_t m_best = vectorOfPotentialMatches.front();

      if(m_best.dist > -1.0) {
        //Matching with a blob
        alreadyProcessIndex.push_back(m_best.blob_index);
        alreadyProcessName.push_back(m_best.name);
        mapOfBlobs[m_best.name].m_imPts.push_back(m_best.nearest);
        mapOfBlobs[m_best.name].m_statusHistory.push_back(vpBlobInfo::NEAREST);
        mapOfBlobs[m_best.name].m_blobIndex = m_best.blob_index;
        mapOfBlobs[m_best.name].m_blobArea = m_best.area;

        vpColVector z_measurement(2);
        z_measurement[0] = m_best.nearest.get_u();
        z_measurement[1] = m_best.nearest.get_v();
        mapOfBlobs[m_best.name].m_kalmanFilter.filtering(z_measurement);
      }
    }
  }

  //We got a list of points successfully matched with a blob and so another list of points not successfully matched
  for(std::map<std::string, vpBlobInfo>::iterator it = mapOfBlobs.begin();
      it != mapOfBlobs.end(); ++it) {
    //For each not successfully matched blob
    if(std::find(alreadyProcessName.begin(), alreadyProcessName.end(), it->first) == alreadyProcessName.end()) {
      if(it->second.m_statusHistory.back() == vpBlobInfo::PROJECTION) {
        //Project the point using the cMo of the previous iteration
        vpPoint pt = it->second.m_pt;
        pt.project(m_cMo);

        double u = 0.0, v = 0.0;
        vpMeterPixelConversion::convertPoint(m_cam, pt.get_x(), pt.get_y(), u, v);
        it->second.m_imPts.push_back(vpImagePoint(v,u));
        it->second.m_statusHistory.push_back(vpBlobInfo::PROJECTION);
        it->second.m_blobArea = -1.0;
      } else if(it->second.m_statusHistory.back() != vpBlobInfo::NOT_TRACKED) {
        //Point which has not been matched with a blob
        vpImagePoint predictedPoint = mapOfPredictedImPts[it->first];
        it->second.m_imPts.push_back(predictedPoint);
        it->second.m_statusHistory.push_back(vpBlobInfo::PREDICTION);
        it->second.m_blobArea = -1.0;
      }
    }
  }
}

/*!
  Update the Kalman filter with the current measures.

  \param KF : The Kalman filter.
  \param measurement : The current measures.
  \param filteredTranslation : The filtered translation.
  \param filteredRotation : The filtered rotation matrix.
*/
void vpObjectBlobTracker::updatePoseKalmanFilter(const vpColVector &z_measurement) {
// The "correct" phase that is going to use the predicted value and our measurement
  m_poseKalmanFilter.filtering(z_measurement);
}

std::ostream& operator<< (std::ostream& stream, const vpPoint& pt) {
  stream << "oX=" << pt.get_oX() << " ; oY=" << pt.get_oY() << " ; oZ=" << pt.get_oZ() << std::endl;
  return stream;
}

std::ostream& operator<< (std::ostream& stream, const vpObjectBlobTracker& blobTracker) {
  for(std::map<std::string, vpBlobInfo>::const_iterator it = blobTracker.m_mapOfObjectBlobs.begin();
      it != blobTracker.m_mapOfObjectBlobs.end(); ++it) {
    stream << it->first << ":\n" << it->second.m_pt << std::endl;
  }
  return stream;
}

#elif !defined(VISP_BUILD_SHARED_LIBS)

// Work arround to avoid warning: libvisp_object_blob_tracker.a(vpObjectBlobTracker.cpp.o) has no symbols
void dummy_vpObjectBlobTracker() {};

#endif
