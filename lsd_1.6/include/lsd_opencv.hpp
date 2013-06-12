/*----------------------------------------------------------------------------*/
/** @file lsd_opencv.h
    LSD OpenCV module header
    @author dani
 */
/*----------------------------------------------------------------------------*/

#ifndef LSD_OPENCV_H_
#define LSD_OPENCV_H_

#include <opencv2/core/core.hpp>

typedef struct lineSegment_s
{
    cv::Point2f begin;
    cv::Point2f end;
    double width;
    double p;
    double NFA;
} lineSegment;


typedef struct coorlist_s
{
  cv::Point p;
  struct coorlist_s* next;
} coorlist;


class LSD
{
public:
    void flsd(const cv::Mat& _image, const double& scale, std::vector<cv::Point2f>& begin, std::vector<cv::Point2f>& end, 
        std::vector<double>& width, std::vector<double>& prec, std::vector<double>& nfa, cv::Rect roi = cv::Rect());
    void flsd(const cv::Mat& _image, const double& scale, std::vector<lineSegment>& lines, cv::Rect roi = cv::Rect());

private:
    cv::Mat image;
    cv::Mat scaled_image;
    cv::Mat angles;
    cv::Mat modgrad;
    cv::Mat used;

    void ll_angle(const double& threshold, const unsigned int& n_bins, std::vector<coorlist*>& list);
    inline void region_grow(const cv::Point2d& s, std::vector<cv::Point2d>& reg, int& reg_size, double& reg_angle, double prec);
    inline void region2rect();
    inline bool refine();
    inline double rect_improve();

};

#endif /* !LSD_OPENCV_H_ */
/*----------------------------------------------------------------------------*/
