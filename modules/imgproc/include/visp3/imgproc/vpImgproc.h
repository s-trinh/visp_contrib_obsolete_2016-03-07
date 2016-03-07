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
 * Functions for basic image processing functions.
 *
 * Authors:
 * Souriya Trinh
 *
 *****************************************************************************/



/*!
  \file vpImgproc.h
  \brief Basic image processing functions.

*/

#ifndef __vpImgproc_h__
#define __vpImgproc_h__

#include <visp3/core/vpImage.h>


namespace vp
{
  enum RETINEX_LEVEL {
      RETINEX_UNIFORM = 0, RETINEX_LOW = 1, RETINEX_HIGH = 2
  };

  VISP_EXPORT void adjust(vpImage<unsigned char> &I, const double alpha, const double beta);
  VISP_EXPORT void adjust(const vpImage<unsigned char> &I1, vpImage<unsigned char> &I2, const double alpha, const double beta);
  VISP_EXPORT void adjust(vpImage<vpRGBa> &I, const double alpha, const double beta);
  VISP_EXPORT void adjust(const vpImage<vpRGBa> &I1, vpImage<vpRGBa> &I2, const double alpha, const double beta);

  VISP_EXPORT void equalizeHistogram(vpImage<unsigned char> &I);
  VISP_EXPORT void equalizeHistogram(const vpImage<unsigned char> &I1, vpImage<unsigned char> &I2);
  VISP_EXPORT void equalizeHistogram(vpImage<vpRGBa> &I, const bool useHSV=false);
  VISP_EXPORT void equalizeHistogram(const vpImage<vpRGBa> &I1, vpImage<vpRGBa> &I2, const bool useHSV=false);

  VISP_EXPORT void gammaCorrection(vpImage<unsigned char> &I, const double gamma);
  VISP_EXPORT void gammaCorrection(const vpImage<unsigned char> &I1, vpImage<unsigned char> &I2, const double gamma);
  VISP_EXPORT void gammaCorrection(vpImage<vpRGBa> &I, const double gamma);
  VISP_EXPORT void gammaCorrection(const vpImage<vpRGBa> &I1, vpImage<vpRGBa> &I2, const double gamma);

  VISP_EXPORT void retinex(vpImage<vpRGBa> &I, const int scale=240, const int scaleDiv=3,
      const int level=RETINEX_UNIFORM, const double dynamic=1.2, const int kernelSize=-1);
  VISP_EXPORT void retinex(const vpImage<vpRGBa> &I1, vpImage<vpRGBa> &I2, const int scale=240, const int scaleDiv=3,
      const int level=RETINEX_UNIFORM, const double dynamic=1.2, const int kernelSize=-1);

  VISP_EXPORT void stretchContrast(vpImage<unsigned char> &I);
  VISP_EXPORT void stretchContrast(const vpImage<unsigned char> &I1, vpImage<unsigned char> &I2);
  VISP_EXPORT void stretchContrast(vpImage<vpRGBa> &I);
  VISP_EXPORT void stretchContrast(const vpImage<vpRGBa> &I1, vpImage<vpRGBa> &I2);

  VISP_EXPORT void stretchContrastHSV(vpImage<vpRGBa> &I);
  VISP_EXPORT void stretchContrastHSV(const vpImage<vpRGBa> &I1, vpImage<vpRGBa> &I2);

  VISP_EXPORT void unsharpMask(vpImage<unsigned char> &I, const unsigned int size=7, const double weight=0.6);
  VISP_EXPORT void unsharpMask(const vpImage<unsigned char> &I, vpImage<unsigned char> &Ires,
      const unsigned int size=7, const double weight=0.6);
  VISP_EXPORT void unsharpMask(vpImage<vpRGBa> &I, const unsigned int size=7, const double weight=0.6);
  VISP_EXPORT void unsharpMask(const vpImage<vpRGBa> &I, vpImage<vpRGBa> &Ires,
      const unsigned int size=7, const double weight=0.6);
}

#endif
