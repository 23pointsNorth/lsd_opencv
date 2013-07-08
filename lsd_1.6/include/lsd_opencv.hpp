/*///////////////////////////////////////////////////////////////////////////////////////
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//*/

#ifndef _OPENCV_LSD_HPP_
#define _OPENCV_LSD_HPP_
#ifdef __cplusplus

#include <opencv2/core/core.hpp>

namespace cv {

class LSD
{
public:
/**
 * Create an LSD object. Specifying scale, number of subdivisions for the image, should the lines be refined and other constants as follows:
 *
 * @param _refine       Should the lines found be refined? E.g. breaking arches into smaller line approximations. 
 *                      If disabled, execution is faster.
 * @param _subdivision  The factor by which each dimension of the image will be divided into. 2 -> generates 2x2 rois and finds lines in them.
 *                      Note: Using smalled images (higher subdivision factor) will find fines lines.
 * @param _scale        The scale of the image that will be used to find the lines. Range (0..1].
 * @param _sigma_scale  Sigma for Gaussian filter is computed as sigma = _sigma_scale/_scale.
 * @param _quant        Bound to the quantization error on the gradient norm. 
 * @param _ang_th       Gradient angle tolerance in degrees.
 * @param _log_eps      Detection threshold: -log10(NFA) > _log_eps
 * @param _density_th   Minimal density of aligned region points in rectangle.
 * @param _n_bins       Number of bins in pseudo-ordering of gradient modulus.
 */
    LSD(bool _refine = true, int _subdivision = 1, double _scale = 0.8, 
        double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5, 
        double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);

/**
 * Detect lines in the input image with the specified ROI.
 *
 * @param _image    A grayscale(CV_8UC1) input image. 
 * @param _lines    Return: A vector of Vec4i elements specifying the beginning and ending point of a line.
 *                          Where Vec4i is (x1, y1, x2, y2), point 1 is the start, point 2 - end. 
 *                          Returned lines are strictly oriented depending on the gradient.
 * @param _roi      Return: ROI of the image, where lines are to be found. If specified, the returning 
 *                          lines coordinates are image wise.
 * @param width     Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param prec      Return: Vector of precisions with which the lines are found.
 * @param nfa       Return: Vector containing number of false alarms in the line region, with precision of 10%. 
 *                          The bigger the value, logarithmically better the detection.
 *                              * -1 corresponds to 10 mean false alarms
 *                              * 0 corresponds to 1 mean false alarm
 *                              * 1 corresponds to 0.1 mean false alarms
 */
    void detect(const cv::InputArray& _image, cv::OutputArray& _lines, cv::Rect _roi = cv::Rect(),
                cv::OutputArray& width = cv::noArray(), cv::OutputArray& prec = cv::noArray(),
                cv::OutputArray& nfa = cv::noArray());

/**
 * Draw lines on the given canvas.
 *
 * @param image     The image, where lines will be drawn. 
 *                  Should have the size of the image, where the lines were found
 * @param lines     The lines that need to be drawn
 */    
    static void drawSegments(cv::Mat& image, const std::vector<cv::Vec4i>& lines);

/**
 * Draw both vectors on the image canvas. Uses blue for lines 1 and red for lines 2.
 *
 * @param image     The image, where lines will be drawn. 
 *                  Should have the size of the image, where the lines were found
 * @param lines1    The first lines that need to be drawn. Color - Blue.
 * @param lines2    The second lines that need to be drawn. Color - Red.
 * @return          The number of mismatching pixels between lines1 and lines2.
 */
    static int compareSegments(cv::Size& size, const std::vector<cv::Vec4i>& lines1, const std::vector<cv::Vec4i> lines2, cv::Mat* image = 0);
    
/*
 * Shows the lines in a window.
 *
 * @param name      The name of the window where the lines will be shown.
 * @param image     The image that will be used as a background. 
 * @param lines     The lines that need to be drawn.
 */    
    static void showSegments(const std::string& name, const cv::Mat& image, const std::vector<cv::Vec4i>& lines);

/*
 * Shows the 2 vector of lines drawn on a window.
 *
 * @param name      The name of the window where the lines will be shown.
 * @param size      The size that will be used to create an image to draw the lines, if no image is specified.
 * @param lines1    The lines that need to be drawn with blue.
 * @param lines2    The lines that need to be drawn with red.
 * @param image     A optional pointer to an image that may be used as a background.
 * @return          The number of non overlapping pixels. 
 */    
    static int showSegments(const std::string& name, cv::Size size, const std::vector<cv::Vec4i>& lines1, const std::vector<cv::Vec4i> lines2, cv::Mat* image = 0);

private:
    cv::Mat image;
    cv::Mat_<double> scaled_image;
    double *scaled_image_data;
    cv::Mat_<double> angles;     // in radians 
    double *angles_data;
    cv::Mat_<double> modgrad;
    double *modgrad_data;
    cv::Mat_<uchar> used;

    int img_width;
    int img_height;
    double LOG_NT;

    cv::Rect roi;
    int roix, roiy;

    const double SCALE;
    const bool doRefine;
    const int SUBDIVISION;
    const double SIGMA_SCALE;
    const double QUANT;
    const double ANG_TH;
    const double LOG_EPS;
    const double DENSITY_TH;
    const int N_BINS;

    struct RegionPoint {
        int x;
        int y;
        uchar* used;
        double angle;
        double modgrad;
    };


    typedef struct coorlist_s
    {
        cv::Point2i p;
        struct coorlist_s* next;
    } coorlist;

    typedef struct rect_s
    {
        double x1, y1, x2, y2;    // first and second point of the line segment
        double width;             // rectangle width
        double x, y;              // center of the rectangle
        double theta;             // angle
        double dx,dy;             // (dx,dy) is vector oriented as the line segment
        double prec;              // tolerance angle
        double p;                 // probability of a point with angle within 'prec'
    } rect;

/**
 * Detect lines in the whole input image.
 *
 * @param _image        A grayscale(CV_8UC1) input image. 
 * @param lines         Return: A vector of Vec4i elements specifying the beginning and ending point of a line.
 *                              Where Vec4i is (x1, y1, x2, y2), point 1 is the start, point 2 - end. 
 *                              Returned lines are strictly oriented depending on the gradient.
 * @param widths        Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param precisions    Return: Vector of precisions with which the lines are found.
 * @param nfas          Return: Vector containing number of false alarms in the line region, with precision of 10%. 
 *                              The bigger the value, logarithmically better the detection.
 *                                  * -1 corresponds to 10 mean false alarms
 *                                  * 0 corresponds to 1 mean false alarm
 *                                  * 1 corresponds to 0.1 mean false alarms
 */
    void flsd(const cv::Mat_<double>& _image,
              std::vector<cv::Vec4i>& lines, 
              std::vector<double>* widths, std::vector<double>* precisions, 
              std::vector<double>* nfas);

/**
 * Finds the angles and the gradients of the image. Generates a list of pseudo ordered points.
 *
 * @param threshold The minimum value of the angle that is considered defined, otherwise NOTDEF
 * @param n_bins    The number of bins with which gradients are ordered by, using bucket sort. 
 * @param list      Return: Vector of coordinate points that are pseudo ordered by magnitude. 
 *                  Pixels would be ordered by norm value, up to a precision given by max_grad/n_bins.
 */
    void ll_angle(const double& threshold, const unsigned int& n_bins, std::vector<coorlist>& list);

/**
 * Grow a region starting from point s with a defined precision, 
 * returning the containing points size and the angle of the gradients.
 *
 * @param s         Starting point for the region.
 * @param reg       Return: Vector of points, that are part of the region
 * @param reg_size  Return: The size of the region.
 * @param reg_angle Return: The mean angle of the region.
 * @param prec      The precision by which each region angle should be aligned to the mean.
 */
    void region_grow(const cv::Point2i& s, std::vector<RegionPoint>& reg,
                     int& reg_size, double& reg_angle, const double& prec);

/**
 * Finds the bounding rotated rectangle of a region.
 *
 * @param reg       The region of points, from which the rectangle to be constructed from.
 * @param reg_size  The number of points in the region.
 * @param reg_angle The mean angle of the region.
 * @param prec      The precision by which points were found.
 * @param p         Probability of a point with angle within 'prec'.
 * @param rec       Return: The generated rectangle.
 */
    void region2rect(const std::vector<RegionPoint>& reg, const int reg_size, const double reg_angle,
                    const double prec, const double p, rect& rec) const;

/**
 * Compute region's angle as the principal inertia axis of the region.
 * @return          Regions angle.
 */
    double get_theta(const std::vector<RegionPoint>& reg, const int& reg_size, const double& x,
                     const double& y, const double& reg_angle, const double& prec) const;

/**
 * An estimation of the angle tolerance is performed by the standard deviation of the angle at points 
 * near the region's starting point. Then, a new region is grown starting from the same point, but using the 
 * estimated angle tolerance. If this fails to produce a rectangle with the right density of region points, 
 * 'reduce_region_radius' is called to try to satisfy this condition.
 */ 
    bool refine(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
                const double prec, double p, rect& rec, const double& density_th);

/**
 * Reduce the region size, by elimination the points far from the starting point, until that leads to 
 * rectangle with the right density of region points or to discard the region if too small.
 */
    bool reduce_region_radius(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
                const double prec, double p, rect& rec, double density, const double& density_th);

/** 
 * Try some rectangles variations to improve NFA value. Only if the rectangle is not meaningful (i.e., log_nfa <= log_eps).
 * @return      The new NFA value.
 */
    double rect_improve(rect& rec) const;

/** 
 * Calculates the number of correctly aligned points within the rectangle.
 * @return      The new NFA value.
 */
    double rect_nfa(const rect& rec) const;

/** 
 * Computes the NFA values based on the total number of points, points that agree.
 * n, k, p are the binomial parameters. 
 * @return      The new NFA value.
 */
    double nfa(const int& n, const int& k, const double& p) const;

/** 
 * Is the point at place 'address' aligned to angle theta, up to precision 'prec'?
 * @return      Whether the point is aligned.
 */
    bool isAligned(const int& address, const double& theta, const double& prec) const;
};

}
#endif
#endif
