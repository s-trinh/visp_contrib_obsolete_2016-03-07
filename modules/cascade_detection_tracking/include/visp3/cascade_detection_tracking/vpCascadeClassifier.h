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
 * Detect an object using a cascade of classifiers with the possibility to
 * to track the lost objects with the ViSP template tracker.
 *
 * Authors:
 * Souriya Trinh
 *
 *****************************************************************************/

#ifndef __vpCascadeClassifier_h__
#define __vpCascadeClassifier_h__

#include <iostream>

#include <visp3/core/vpConfig.h>
#include <visp3/tt/vpTemplateTrackerWarpSRT.h>
#include <visp3/tt/vpTemplateTrackerSSDInverseCompositional.h>


// OpenCV is required
#if (VISP_HAVE_OPENCV_VERSION >= 0x020403)
#include <opencv2/opencv.hpp>


class VISP_EXPORT vpObjectDetection {
public:
  //! Different possible status of an object:
  //! DETECTION_STATUS: Flag sets when the classifier has detected the object
  //! PREDICTION_STATUS: Flag sets when the object has been lost and the template tracker is not enabled.
  //! A constant velocity model is assumed.
  //! TRACKING_STATUS: Flag sets when the object has been lost and the object is track during a certain number of frames.
  //! TOO_DARK_STATUS: Flag sets if the object is too dark (a brightness filtering is enabled), for visualization.
  //! TOO_BRIGHT_STATUS: Flag sets if the object is too bright (a brightness filtering is enabled), for visualization.
  //! FILTERED_STATUS: Flag sets if the object is eliminated using a custom filtering function, for visualization.
  enum vpObjectStatus {DETECTION_STATUS, PREDICTION_STATUS, TRACKING_STATUS, TOO_DARK_STATUS, TOO_BRIGHT_STATUS, FILTERED_STATUS};
  //! Current object bounding box
  vpRect m_boundingBox;
  //! Vector of object bounding boxes history
  std::vector<vpRect> m_boundingBoxHistory;
  //! Vector of status history
  std::vector<vpObjectStatus> m_statusHistory;
  //! Flag set to true to notify that the object should be deleted
  bool m_trackingHasDiverged;
  //! Flag set to true when the object has been detected
  bool m_update;

  vpObjectDetection() : m_boundingBox(), m_boundingBoxHistory(), m_statusHistory(), m_trackingHasDiverged(false),
      m_update(false),m_maxVectorHistory(30), m_warp(), m_tracker(NULL), m_zone_ref(),
      m_zone_cur(), m_p(), m_target(), m_area_zone_cur(0.0), m_area_zone_prev(0.0) {

    m_tracker = new vpTemplateTrackerSSDInverseCompositional(&m_warp);
    m_tracker->setSampling(2, 2);
    m_tracker->setLambda(0.001);
    m_tracker->setIterationMax(5);
    m_tracker->setPyramidal(2, 1);
  }

  vpObjectDetection(const vpRect &r, const vpObjectStatus &status) : m_boundingBox(),
      m_boundingBoxHistory(), m_statusHistory(), m_trackingHasDiverged(false), m_update(false), m_maxVectorHistory(30), m_warp(),
      m_tracker(NULL), m_zone_ref(), m_zone_cur(), m_p(), m_target(), m_area_zone_cur(0.0), m_area_zone_prev(0.0) {
    addDetection(r, status);

    m_tracker = new vpTemplateTrackerSSDInverseCompositional(&m_warp);
    m_tracker->setSampling(2, 2);
    m_tracker->setLambda(0.001);
    m_tracker->setIterationMax(5);
    m_tracker->setPyramidal(2, 1);
  }

  vpObjectDetection(const vpObjectDetection &object) {
    m_boundingBox = object.m_boundingBox;
    m_boundingBoxHistory = object.m_boundingBoxHistory;
    m_statusHistory = object.m_statusHistory;
    m_trackingHasDiverged = object.m_trackingHasDiverged;
    m_update = object.m_update;
    m_maxVectorHistory = object.m_maxVectorHistory;
    m_warp = object.m_warp;

    m_tracker = new vpTemplateTrackerSSDInverseCompositional(&m_warp);
    //Work around as there is no assignment operators in vpTemplateTrackerSSDInverseCompositional ?
    m_tracker->setSampling(2, 2);
    m_tracker->setLambda(0.001);
    m_tracker->setIterationMax(5);
    m_tracker->setPyramidal(2, 1);

    m_zone_ref = object.m_zone_ref;
    m_zone_cur = object.m_zone_cur;
    m_p = object.m_p;
    m_target = object.m_target;
    m_area_zone_cur = object.m_area_zone_cur;
    m_area_zone_prev = object.m_area_zone_prev;
  }

  vpObjectDetection &operator=(const vpObjectDetection &source) {
    if (&source != this) {
      if(m_tracker != NULL) {
        delete m_tracker;
      }

      m_boundingBox = source.m_boundingBox;
      m_statusHistory = source.m_statusHistory;
      m_boundingBoxHistory = source.m_boundingBoxHistory;
      m_update = source.m_update;
      m_maxVectorHistory = source.m_maxVectorHistory;
      m_warp = source.m_warp;

      m_tracker = new vpTemplateTrackerSSDInverseCompositional(&m_warp);
      //Work around as there is no assignment operators in vpTemplateTrackerSSDInverseCompositional ?
      m_tracker->setSampling(2, 2);
      m_tracker->setLambda(0.001);
      m_tracker->setIterationMax(5);
      m_tracker->setPyramidal(2, 1);

      m_zone_ref = source.m_zone_ref;
      m_zone_cur = source.m_zone_cur;
      m_p = source.m_p;
      m_target = source.m_target;
      m_area_zone_cur = source.m_area_zone_cur;
      m_area_zone_prev = source.m_area_zone_prev;
      m_trackingHasDiverged = source.m_trackingHasDiverged;
    }
    return *this;
  }

  ~vpObjectDetection() {
    if(m_tracker != NULL) {
      delete m_tracker;
    }
  }

  /*!
    Add the bounding box of the object in the current image.

    \param r: Current bounding box.
    \param status: Status of the detection (if the bounding box was found by the detection, by the template tracker, ...).
  */
  void addDetection(const vpRect &r, const vpObjectStatus &status) {
    m_boundingBox = r;
    m_statusHistory.push_back(status);
    m_boundingBoxHistory.push_back(r);
    m_update = true;
  }

  /*!
    Clean internal vectors (with a fix maximal vector size) to avoid too much memory occupation.
  */
  void clean() {
    if(m_statusHistory.size() > m_maxVectorHistory) {
      m_statusHistory.erase(m_statusHistory.begin());
    }

    if(m_boundingBoxHistory.size() > m_maxVectorHistory) {
      m_boundingBoxHistory.erase(m_boundingBoxHistory.begin());
    }
  }

  /*!
    Initialize the template tracker using the supplied image and with the current bounding box.

    \param I: Image used to initialize the template tracker.
  */
  void initTracking(const vpImage<unsigned char> &I) {
    double scale = 0.05; // reduction factor
    double x = m_boundingBox.getLeft(), y = m_boundingBox.getTop();
    double width = m_boundingBox.getWidth();
    double height = m_boundingBox.getHeight();

    std::vector<vpImagePoint> corners;
    corners.push_back(vpImagePoint(y + scale * height, x + scale * width));
    corners.push_back(vpImagePoint(y + scale * height, x + (1 - scale) * width));
    corners.push_back(vpImagePoint(y + (1 - scale) * height, x + (1 - scale) * width));
    corners.push_back(vpImagePoint(y + (1 - scale) * height, x + scale * width));

    try {
      m_tracker->resetTracker();
      m_tracker->initFromPoints(I, corners, true);
      m_tracker->track(I);
      m_zone_ref = m_tracker->getZoneRef();
      m_p = m_tracker->getp();
      m_warp.warpZone(m_zone_ref, m_p, m_zone_cur);
      m_area_zone_prev = m_zone_cur.getArea();
      m_area_zone_cur = m_zone_cur.getArea();
    } catch (vpException &e) {
      std::cerr << e.what() << std::endl;
//      throw e;
      m_trackingHasDiverged = true;
    }
  }

  /*!
    Return the intersection rectangle.

    \param r1: First rectangle.
    \param r2: Second rectangle.
    \return The intersection rectangle, which can be zero if there is no intersection
  */
  vpRect getIntersectionRectangle(const vpRect &r1, const vpRect &r2) {
    double left = std::max(r1.getLeft(), r2.getLeft());
    double right = std::min(r1.getRight(), r2.getRight());
    double bottom = std::min(r1.getBottom(), r2.getBottom());
    double top = std::max(r1.getTop(), r2.getTop());

    vpRect rIntersec;
    if(left < right && bottom > top) {
      rIntersec = vpRect(left, top, (right-left), (bottom-top));
    }

    return rIntersec;
  }

  /*!
    Track the object using the template tracker.

    \param I: Image where we want to track the object.
  */
  void track(const vpImage<unsigned char> &I) {
    if(!m_trackingHasDiverged) {
      try {
        m_tracker->track(I);
        m_p = m_tracker->getp();
        m_warp.warpZone(m_zone_ref, m_p, m_zone_cur);

        m_area_zone_cur = m_zone_cur.getArea();
        double min_size_change = 0.8, max_size_change = 1.2;
        double size_ratio = m_area_zone_cur / m_area_zone_prev;

        //Try to avoid too big changes in size
        if(size_ratio < min_size_change || size_ratio > max_size_change) {
          m_trackingHasDiverged = true;
        } else {
          vpRect r = m_zone_cur.getBoundingBox();
          vpRect rImage(0, 0, I.getWidth(), I.getHeight());
          vpRect rIntersect = getIntersectionRectangle(rImage, r);

          if(rIntersect.getWidth() > 0 && rIntersect.getHeight() > 0) {
            double ratioW = rIntersect.getWidth() / r.getWidth();
            double ratioH = rIntersect.getHeight() / r.getHeight();

            //Try to avoid cases where the object leaves the image bounds
            if(ratioW < 0.5 || ratioH < 0.5) {
              m_trackingHasDiverged = true;
            } else if(r.getWidth() < 30 || r.getHeight() < 30) {
              m_trackingHasDiverged = true;
            } else {
              addDetection(r, vpObjectDetection::TRACKING_STATUS);
              m_area_zone_prev = m_area_zone_cur;
            }
          } else {
            m_trackingHasDiverged = true;
          }
        }
      } catch(vpException &e) {
        std::cerr << e.what() << std::endl;
  //      throw e;
        m_trackingHasDiverged = true;
      }
    }
  }

private:
  //! Maximal size for the different vector sizes
  size_t m_maxVectorHistory;
  //! Warp type for the template tracker
  vpTemplateTrackerWarpSRT m_warp;
  //! Template tracker
  vpTemplateTrackerSSDInverseCompositional *m_tracker;
  //! Reference zone to track the object
  vpTemplateTrackerZone m_zone_ref;
  //! Current zone where was found the object
  vpTemplateTrackerZone m_zone_cur;
  //! m_p
  vpColVector m_p;
  //! Bounding box returned by the template tracker
  vpRect m_target;
  //! Area of the current zone
  double m_area_zone_cur;
  //! Area of the previous zone
  double m_area_zone_prev;
};

class VISP_EXPORT vpCascadeClassifier {
public:
  //! No filtering, keep detection if the mean intensity falls between a defined range, same thing but
  //! with median intensity, use peak histogram to determine if the detection is to bright or to dark
  enum vpBrightnessFilterType {NO_BRIGHTNESS_FILTER, MEAN_INTENSITIE, MEDIAN_INTENSITIE, HISTOGRAM};
  //! NO_DETECTION_FILTER will return all the detected objects returned by the cascade classifier
  //! BIGGEST_OBJECT will keep only the biggest detection
  //! MATCHING will match the current detections with the list of the current objects
  enum vpDetectionFilterType {NO_DETECTION_FILTER, BIGGEST_OBJECT, MATCHING};
  //! Strategy to use when an object has not been detected in the current image
  //! NO_RECOVER: no strategy, will keep the last known bounding box
  //! PREDICTION: will predict the current position using a constant velocity assumption
  //! TEMPLATE_TRACKING: will use the template tracker to find the object
  enum vpRecoveryStrategyType {NO_RECOVER, PREDICTION, TEMPLATE_TRACKING};

  vpCascadeClassifier();
  vpCascadeClassifier(const std::string &classifierDataFile);


  void detect(const vpImage<vpRGBa> &I, bool (*func)(const vpImage<vpRGBa> &I, const vpRect &boundingBox)=NULL,
      const double scaleFactor=1.1, const int minNeighbors=3, const cv::Size &minSize=cv::Size(),
      const cv::Size &maxSize=cv::Size());

  void detect(const vpImage<unsigned char> &I, bool (*func)(const vpImage<unsigned char> &I, const vpRect &boundingBox)=NULL,
      const double scaleFactor=1.1, const int minNeighbors=3, const cv::Size &minSize=cv::Size(),
      const cv::Size &maxSize=cv::Size());

  void displayAllDetections(const vpImage<unsigned char> &I, const unsigned int thickness=2);
  void displayAllDetections(const vpImage<vpRGBa> &I, const unsigned int thickness=2);

  void displayDetectedObjects(const vpImage<unsigned char> &I, const unsigned int thickness=2);
  void displayDetectedObjects(const vpImage<vpRGBa> &I, const unsigned int thickness=2);

  void filterTooBrightObjects(const vpImage<unsigned char> &I, std::vector<vpRect> &objectBoundingBoxes);
  void filterTooDarkObjects(const vpImage<unsigned char> &I, std::vector<vpRect> &objectBoundingBoxes);

  /*!
    Get the brightness filter type.

    \return The brightness filter type.
  */
  inline vpBrightnessFilterType getBrightnessFilterType() const {
    return m_brightnessFilterType;
  }

  /*!
    Get the vector of bounding boxes corresponding to the detections in the current image.

    \return The vector of bounding boxes.
  */
  inline std::vector<vpRect> getDetectedBoundingBoxes() const {
    return m_objectBoundingBoxes;
  }

  /*!
    Get the map of detected and track objects. The key corresponds to the id of the object. The same id is attributed to the
    same detected object in time.

    \return The map of detected and track objects.
  */
  inline std::map<int, vpObjectDetection> getDetectedObjects() const {
    return m_mapOfDetectedObjects;
  }

  /*!
    Get the maximum cooldown time for a lost object id before it can be reused.

    \return The maximum cooldown time of a lost object id.
  */
  inline int getMaxCooldownTime() const {
    return m_maxCooldownTime;
  }

  /*!
    Get the maximum number of iterations during the time where an object has not been detected by the cascade classifier.

    \return The maximum number of iterations of the life time of an object.
  */
  inline size_t getMaxLifeTime() const {
    return m_maxLifeTime;
  }

  /*!
    Get the minimum overlapping percentage to decide if a detection belongs to the object or not.

    \return The minimum overlapping percentage.
  */
  inline double getMinOverlappingPercentage() const {
    return m_minOverlappingPercentage;
  }

  /*!
    Get the threshold to decide if a detection is too dark or not.

    \return The threshold to decide if a detection is too dark or not.
  */
  inline double getTooDarkIntensitieThreshold() const {
    return m_tooDarkIntensitieThreshold;
  }

  /*!
    Get the threshold to decide if a detection is too bright or not.

    \return The threshold to decide if a detection is too bright or not.
  */
  inline double getTooBrightIntensitieThreshold() const {
    return m_tooBrightIntensitieThreshold;
  }

  /*!
    Get the recovery strategy type (what to do when an object has not been detected).

    \return The recovery strategy type.
  */
  inline vpRecoveryStrategyType getRecoveryStrategy() const {
    return m_recoveryStrategyType;
  }

  /*!
    Get the flag that indicates if a prediction (for the position of the object in the current image)
    must be used before matching the detections with the objects.

    \return True if a prediction must be used, false otherwise.
  */
  inline bool getUsePredictionForMatching() const {
    return m_usePredictionForMatching;
  }

  bool load(const std::string &classifierDataFile);

  /*!
    Set the brightness filter type.

    \param type: The brightness filter type.
  */
  inline void setBrightnessFilterType(const vpBrightnessFilterType &type) {
    m_brightnessFilterType = type;
  }

  /*!
    Set the maximum cooldown time for a lost object id before it can be reused.

    \param maxTime: The maximum cooldown time of a lost object id.
  */
  inline void setMaxCooldownTime(const int maxTime) {
    m_maxCooldownTime = maxTime;
  }

  /*!
    Set the maximum life time of an object.

    \param size: The maximum life time of an object.
  */
  inline void setMaxLifeTime(const size_t size) {
    m_maxLifeTime = size;
  }

  /*!
    Set the minimum overlapping percentage to decide if a detection belongs to the object or not.

    \return The minimum overlapping percentage.
  */
  inline void setMinOverlappingPercentage(const double percentage) {
    if(percentage > 0 && percentage <= 100.0) {
      m_minOverlappingPercentage = percentage;
    } else {
      std::cerr << "percentage=" << percentage << " but should be between ]0 ; 1] !" << std::endl;
    }
  }

  /*!
    Set the recovery strategy.

    \param strategy: The recovery strategy.
  */
  inline void setRecoveryStrategy(const vpRecoveryStrategyType strategy) {
    m_recoveryStrategyType = strategy;
  }

  /*!
    Set the threshold to decide if a detection is too dark or not.

    \param threshold: The threshold.
  */
  inline void setTooDarkIntensitieThreshold(const double threshold) {
    m_tooDarkIntensitieThreshold = threshold;
  }

  /*!
    Set the threshold to decide if a detection is too bright or not.

    \param threshold: The threshold.
  */
  inline void setTooBrightIntensitieThreshold(const double threshold) {
    m_tooBrightIntensitieThreshold = threshold;
  }

  /*!
    Set the flag to decide if a prediction (on the object positions) must be used when
    matching the current detections with the objects.

    \param use: The flag.
  */
  inline void setUsePredictionForMatching(const bool use) {
    m_usePredictionForMatching = use;
  }


private:
  //! Brightness filter type
  vpBrightnessFilterType m_brightnessFilterType;
  //! Cascade classifier
  cv::CascadeClassifier m_classifierDetector;
  //! Detection filter type
  vpDetectionFilterType m_detectionFilterType;
  //! Threshold to reject a detection when too bright with histogram method
  double m_histogramRatioTooBrightThreshold;
  //! Threshold to reject a detection when too dark with histogram method
  double m_histogramRatioTooDarkThreshold;
  //! Map of detected objects, the key represents the id of the object
  std::map<int, vpObjectDetection> m_mapOfDetectedObjects;
  //! Map of lost object id to avoid to attribute a lost id immediately
  std::map<int, int> m_mapOfLostObjects;
  //! Max cooldown time
  int m_maxCooldownTime;
  //! Maximal life time of an object when it is not detected by the cascade classifier
  size_t m_maxLifeTime;
  //! Minimal overlapping percentage to match a detection with an object
  double m_minOverlappingPercentage;
  //! Vector of current bounding boxes
  std::vector<vpRect> m_objectBoundingBoxes;
  //! Previous image used to initialize the template tracker with the last known bounding box
  vpImage<unsigned char> m_prevI;
  //! Recover strategy type
  vpRecoveryStrategyType m_recoveryStrategyType;
  //! Threshold to reject a detection when too bright
  double m_tooBrightIntensitieThreshold;
  //! Threshold to reject a detection when too dark
  double m_tooDarkIntensitieThreshold;
  //! Flag set to true if a prediction must be used before matching a detection with an object
  bool m_usePredictionForMatching;
  //! Vector of currently detected objects
  std::vector<vpObjectDetection> m_vectorOfDetectedObjects;

  inline void clear() {
    m_mapOfDetectedObjects.clear();
    m_vectorOfDetectedObjects.clear();
    m_objectBoundingBoxes.clear();
  }

  void computeDetection(const vpImage<unsigned char> &I, const double scaleFactor=1.1, const int minNeighbors=3,
      const cv::Size &minSize=cv::Size(), const cv::Size &maxSize=cv::Size());

  void computeMatching(const vpImage<unsigned char> &I);

  void eraseLostObjects();

  vpRect getBiggestDetection(const std::vector<vpRect> &objectBoundingBoxes);

  int getId();

  double getMeanIntensity(const vpImage<unsigned char> &I, const vpRect &rect);
  double getMedian(std::vector<double> &v);
  double getMedianIntensity(const vpImage<unsigned char> &I, const vpRect &rect);

  double getPercentageOfOverlap(const vpRect &r1, const vpRect &r2);

  void match(const vpImage<unsigned char> &I, const std::vector<vpRect> &objectBoundingBoxes);
};

#endif
#endif
