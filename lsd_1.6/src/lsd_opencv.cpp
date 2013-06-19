#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#include "lsd_opencv.hpp"

#include "opencv2/opencv.hpp"
//core, imgproc

using namespace cv;


LSD::LSD(double _scale, int _subdivision, bool _refine, double _sigma_scale, double _quant, double _ang_th, double _log_eps, double _density_th, int _n_bins)
{
    CV_Assert(_scale > 0 && _sigma_scale > 0 && _quant >= 0 &&
              _ang_th > 0 && _ang_th < 180 && _density_th >= 0 && _density_th < 1 &&
              _n_bins > 0 && _subdivision > 0);
    
    scale = _scale;
    doRefine = _refine;
    subdivision = _subdivision;
    SIGMA_SCALE = _sigma_scale;
    QUANT = _quant;
    ANG_TH = _ang_th;
    LOG_EPS = _log_eps;
    DENSITY_TH = _density_th;
    N_BINS = _n_bins;
}

void LSD::flsd(const Mat& _image, std::vector<Point2f>& begin, std::vector<Point2f>& end, 
    std::vector<double>& width, std::vector<double>& prec, std::vector<double>& nfa, Rect roi)
{
    //call the other method,
}

void LSD::flsd(const Mat& _image, std::vector<lineSegment>& lines, Rect roi)
{
    CV_Assert(_image.data != NULL && _image.channels() == 1);
    CV_Assert(scale > 0);

    _image.convertTo(image, CV_64FC1);

    // Angle tolerance
    double prec = M_PI * ANG_TH / 180.0;
    double p = ANG_TH / 180.0;
    double rho = QUANT / sin(prec);    // gradient magnitude threshold
 
    vector<coorlist> list;
    if (scale != 1)
    {
        //TODO: Asses Gaussian blur, as scaling down applies.
        Mat gaussian_img;
        double sigma = (scale < 1.0)?(SIGMA_SCALE / scale):(SIGMA_SCALE);
        double prec = 3.0;
        unsigned int h =  (unsigned int) ceil(sigma * sqrt(2.0 * prec * log(10.0)));
        int ksize = 1+2*h; // kernel size 
        // Create a Gaussian kernel
        Mat kernel = getGaussianKernel(ksize, sigma, CV_64FC1);
        // Apply to the image
        filter2D(image, gaussian_img, image.depth(), kernel, Point(-1, -1));
        // Scale image to needed size
        resize(gaussian_img, scaled_image, Size(), scale, scale);
        //resize(image, scaled_image, Size(), scale, scale);
        imshow("Gaussian image", scaled_image);
        ll_angle(rho, BIN_SIZE, list);
    }
    else
    {
        scaled_image = image;
        ll_angle(rho, BIN_SIZE, list);
    }
    
    // double* q = (double*) angles.data;
    // //memcpy(&q, angles.data, sizeof(double));
    // std::cout << q[0] << std::endl;
    // std::cout << "@CHECK: Angles >" << (double)angles.at<unsigned char>(0,0) << "<>" << 
    //     (double)angles.data[0] << "<"<< std::endl; // should be <double>(x,y) not <uchar>
    
    
    /* Number of Tests - NT
        The theoretical number of tests is Np.(XY)^(5/2)
        where X and Y are number of columns and rows of the image.
        Np corresponds to the number of angle precisions considered.
        As the procedure 'rect_improve' tests 5 times to halve the
        angle precision, and 5 more times after improving other factors,
        11 different precision values are potentially tested. Thus,
        the number of tests is
        11 * (X*Y)^(5/2)
        whose logarithm value is
        log10(11) + 5/2 * (log10(X) + log10(Y)).
    */
    int width = scaled_image.cols;
    int height = scaled_image.rows; 
    
    double logNT = 5.0 * (log10((double)width) + log10((double)height)) / 2.0 + log10(11.0);
    int min_reg_size = (int) (-logNT/log10(p)); /* minimal number of points in region that can give a meaningful event */
    //std::cout << "Min region size: " << min_reg_size << std::endl;

    // Initialize region only when needed
    Mat region = Mat::zeros(scaled_image.size(), CV_8UC1);
    Mat used = Mat::zeros(scaled_image.size(), CV_8UC1); // zeros = NOTUSED
    vector<cv::Point2i> reg(width * height);
    
    // std::cout << "Search." << std::endl;
    // Search for line segments 
    unsigned int ls_count = 0;
    unsigned int list_size = list.size();
    for(unsigned int i = 0; i < list_size; ++i)
    {
        // std::cout << "Inside for 1: size " << list.size() << " image size: " << image.size() << std::endl;
        unsigned int adx = list[i].p.x + list[i].p.y * width;
        // std::cout << "adx " << adx << std::endl;
        // std::cout << "Used: " << used.data[adx] << std::endl;
        if((used.data[adx] == NOTUSED) && (angles_data[adx] != NOTDEF))
        {
            // std::cout << "Inside for 2 " << std::endl;
            int reg_size;
            double reg_angle;
            region_grow(list[i].p, reg, reg_size, reg_angle, prec, used);
            
            // Ignore small regions
            if(reg_size < min_reg_size) { continue; }
            
            //std::cout << "Region size: " << reg_size << std::endl;
            // Construct rectangular approximation for the region
            rect rec;
            region2rect(reg, reg_size, reg_angle, prec, p, rec);
            if(!refine(reg, reg_size, reg_angle, prec, p, rec, DENSITY_TH, used)) { continue; }

            // Compute NFA
            double log_nfa = rect_improve();
            //if(log_nfa <= LOG_EPS) { continue; }

            // Found new line
            ++ls_count;

            // Add the offset
            rec.x1 += 0.5; rec.y1 += 0.5;
            rec.x2 += 0.5; rec.y2 += 0.5;

            // scale the result values if a sub-sampling was performed
            if(scale != 1.0)
            {
                rec.x1 /= scale; rec.y1 /= scale;
                rec.x2 /= scale; rec.y2 /= scale;
                rec.width /= scale;
            }
            
            lines.push_back(lineSegment(
                Point(rec.x1, rec.y1), 
                Point(rec.x2, rec.y2),
                rec.width,
                rec.p,
                log_nfa));

            //std::cout << "Lines size: " << lines.size() << std::endl;
            for(unsigned int el = 0; el < reg_size; el++)
            {
                region.data[reg[i].x + reg[i].y * width] = ls_count;
            }

        }
    
    }
 
}

void LSD::ll_angle(const double& threshold, const unsigned int& n_bins, std::vector<coorlist>& list)
{
    angles = cv::Mat(scaled_image.size(), CV_64FC1);
    modgrad = cv::Mat(scaled_image.size(), CV_64FC1);
    
    angles_data = angles.ptr<double>(0);
    modgrad_data = modgrad.ptr<double>(0);
    scaled_image_data = scaled_image.ptr<double>(0);

    int width = scaled_image.cols;
    int height = scaled_image.rows;

    // Undefined the down and right boundaries 
    cv::Mat w_ndf(1, width, CV_64FC1, NOTDEF);
    cv::Mat h_ndf(height, 1, CV_64FC1, NOTDEF);
    w_ndf.row(0).copyTo(angles.row(height - 1));
    h_ndf.col(0).copyTo(angles.col(width -1));
    
    /* Computing gradient for remaining pixels */
    CV_Assert(scaled_image.isContinuous());   // Accessing image data linearly
    double max_grad = 0.0;
    for(int x = 0; x < width - 1; ++x)
    {
        for(int y = 0; y < height - 1; ++y)
        {
            /*
                Norm 2 computation using 2x2 pixel window:
                    A B
                    C D
                and
                    DA = D-A,  BC = B-C.
                Then
                    gx = B+D - (A+C)   horizontal difference
                    gy = C+D - (A+B)   vertical difference
                DA and BC are just to avoid 2 additions.
            */
            int adr = y * width + x; 
            double DA = scaled_image_data[adr + width + 1] - scaled_image_data[adr];
            double BC = scaled_image_data[adr + 1] - scaled_image_data[adr + width];
            double gx = DA + BC;    /* gradient x component */
            double gy = DA - BC;    /* gradient y component */
            double norm = std::sqrt((gx * gx + gy * gy)/4.0);   /* gradient norm */
            
            modgrad_data[adr] = norm;    /* store gradient*/

            if (norm <= threshold)  /* norm too small, gradient no defined */
            {
                angles_data[adr] = NOTDEF;
            }
            else
            {
                angles_data[adr] = std::atan2(gx, -gy);   /* gradient angle computation */
                if (norm > max_grad) { max_grad = norm; }
            }

        }
    }
    // std::cout << "Max grad: " << max_grad << std::endl;

    /* Compute histogram of gradient values */
    // // SLOW! 
    // std::vector<std::vector<cv::Point> > range(n_bins);
    // //for(int i = 0; i < n_bins; ++i) {range[i].reserve(width*height/n_bins); }
    // double bin_coef = (double) n_bins / max_grad;
    // for(int x = 0; x < width - 1; ++x)
    // {
    //     for(int y = 0; y < height - 1; ++y)
    //     {
    //         double norm = modgrad.data[y * width + x];
    //         /* store the point in the right bin according to its norm */
    //         int i = (unsigned int) (norm * bin_coef);
    //         range[i].push_back(cv::Point(x, y));
    //     }
    // }

    list = vector<coorlist>(width * height);
    vector<coorlist*> range_s(n_bins, NULL);
    vector<coorlist*> range_e(n_bins, NULL);
    unsigned int count = 0;
    double bin_coef = (double) (n_bins - 1) / max_grad;

    for(int x = 0; x < width - 1; ++x)
    {
        for(int y = 0; y < height - 1; ++y)
        {
            double norm = modgrad_data[y * width + x];
            /* store the point in the right bin according to its norm */
            int i = (unsigned int) (norm * bin_coef);
            //std::cout << "before assignment" << std::endl;
            if(range_e[i] == NULL)
            {
                // std::cout << "asdsad" << std::endl;
                range_e[i] = range_s[i] = &list[count];
                ++count;
            }
            else
            {
                range_e[i]->next = &list[count];
                range_e[i] = &list[count];
                ++count;
            }
            //range_e[i] = new coorlist();
            //std::cout << "after assignment" << std::endl;
            range_e[i]->p = cv::Point(x, y);
            // std::cout << "between" << std::endl;
            range_e[i]->next = NULL;
            // std::cout << "loop end" << std::endl;
            
        }
    }

    // std::cout << "make list" << std::endl;
    // Sort
    int idx = n_bins - 1;
    for(;idx > 0 && range_s[idx] == NULL; idx--);
    coorlist* start = range_s[idx];
    coorlist* end = range_e[idx];
    if(start != NULL)
    {
        while(idx > 0)
        {
            --idx;
            if(range_s[idx] != NULL)
            {
                end->next = range_s[idx];
                end = range_e[idx];
            }
        }
    }

    // std::cout << "End" << std::endl;
    //imshow("Angles", angles);
}

void LSD::region_grow(const cv::Point2i& s, std::vector<cv::Point2i>& reg, 
                      int& reg_size, double& reg_angle, double& prec, cv::Mat& used)
{
    int width = angles.cols;
    int height = angles.rows;

    // Point to this region
    reg_size = 1;
    reg[0] = s;
    int addr = s.x + s.y * width;
    reg_angle = angles_data[addr];
    double sumdx = cos(reg_angle);
    double sumdy = sin(reg_angle);
    used.data[addr] = USED;

    //Try neighboring regions
    for(int i=0; i<reg_size; ++i)
        for(int xx = reg[i].x - 1; xx <= reg[i].x + 1; ++xx)
            for(int yy = reg[i].y - 1; yy <= reg[i].y + 1; ++yy)
            {
                int c_addr = xx + yy * width;
                if((xx >= 0 && yy>=0) && (xx < width && yy < height) &&
                   (used.data[c_addr] != USED) &&
                   (isAligned(c_addr, reg_angle, prec)))
                {
                    // Add point
                    used.data[c_addr] = USED;
                    reg[reg_size].x = xx;
                    reg[reg_size].y = yy;
                    ++reg_size;

                    // Update region's angle
                    sumdx += cos(angles_data[c_addr]);
                    sumdy += sin(angles_data[c_addr]);
                    reg_angle = atan2(sumdy, sumdx);
                }
            }
}

void LSD::region2rect(const std::vector<cv::Point2i>& reg, const int reg_size, const double reg_angle, 
                      const double prec, const double p, rect& rec) const
{
    double x = 0, y = 0, sum = 0;
    for(int i = 0; i < reg_size; ++i)
    {
        cv::Point2i p = reg[i];
        double weight = modgrad_data[p.x + p.y * modgrad.cols];
        x += (double)p.x * weight;
        y += (double)p.y * weight;
        sum += weight;
    }
    // Weighted sum must differ from 0
    CV_Assert(sum > 0);
    
    x /= sum;
    y /= sum;

    double theta = get_theta(reg, reg_size, x, y, reg_angle, prec);

    // Find length and width
    double dx = cos(theta);
    double dy = sin(theta);
    double l_min = 0, l_max = 0, w_min = 0, w_max = 0;

    for(int i = 0; i < reg_size; ++i)
    {
        double regdx = (double)reg[i].x - x;
        double regdy = (double)reg[i].y - y;
        
        double l = regdx * dx + regdy * dy;
        double w = -regdx * dy + regdy * dx;

        if(l > l_max) l_max = l;
        if(l < l_min) l_min = l;
        if(w > w_max) w_max = w;
        if(w < w_min) w_min = w;
    }

    // Store values
    rec.x1 = x + l_min * dx;
    rec.y1 = y + l_min * dy;
    rec.x2 = x + l_max * dx;
    rec.y2 = y + l_max * dy;
    rec.width = w_max - w_min;
    rec.x = x;
    rec.y = y;
    rec.theta = theta;
    rec.dx = dx;
    rec.dy = dy;
    rec.prec = prec;
    rec.p = p;

    // Min width of 1 pixel
    if(rec.width < 1.0) rec.width = 1.0;
}

double LSD::get_theta(const std::vector<cv::Point2i>& reg, const int& reg_size, const double& x, 
                      const double& y, const double& reg_angle, const double& prec) const
{
    double Ixx = 0.0;
    double Iyy = 0.0;
    double Ixy = 0.0;

    // compute inertia matrix 
    for(int i = 0; i < reg_size; ++i)
    {
        double weight = modgrad_data[reg[i].x + reg[i].y * modgrad.cols];
        double dx = (double)reg[i].x - x;
        double dy = (double)reg[i].y - y;
        Ixx += dy * dy * weight;
        Iyy += dx * dx * weight;
        Ixy -= dx * dy * weight;
    }

    // Check if inertia matrix is null
    CV_Assert(!(double_equal(Ixx, 0) && double_equal(Iyy, 0) && double_equal(Ixy, 0)));

    // Compute smallest eigenvalue
    double lambda = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy));

    // compute angle
    double theta = (fabs(Ixx)>fabs(Iyy))?atan2(lambda - Ixx, Ixy):atan2(Ixy, lambda - Iyy);

    // Correct angle by 180 deg if necessary 
    if(angle_diff(theta, reg_angle) > prec) { theta += M_PI; }

    return theta;
}

bool LSD::refine(std::vector<cv::Point2i>& reg, int& reg_size, double reg_angle, 
                double prec, double p, rect& rec, const double& density_th, cv::Mat& used)
{
    double density = (double) reg_size / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density >= density_th) { return true; }

    // Try to reduce angle tolerance
    double xc = (double) reg[0].x;
    double yc = (double) reg[0].y;
    double ang_c = angles_data[reg[0].x + reg[0].y * angles.cols];
    double sum = 0, s_sum = 0;
    int n = 0;

    for (int i = 0; i < reg_size; ++i)
    {
        used.data[reg[i].x + reg[i].y * used.cols] = NOTUSED;
        if (dist(xc, yc, reg[i].x, reg[i].y) < rec.width)
        {
            double angle = angles_data[reg[i].x + reg[i].y * angles.cols];
            double ang_d = angle_diff_signed(angle, ang_c);
            sum += ang_d;
            s_sum += ang_d * ang_d;
            ++n;
        }
    }
    double mean_angle = sum / (double) n;
    // 2 * standard deviation
    double tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / (double) n + mean_angle * mean_angle ); 

    // Try new region
    region_grow(Point(reg[0].x, reg[0].y), reg, reg_size, reg_angle, tau, used);

    if (reg_size < 2) { return false; }

    region2rect(reg, reg_size, reg_angle, prec, p, rec);
    density = (double) reg_size / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density < density_th) 
    { 
        return reduce_region_radius(reg, reg_size, reg_angle, prec, p, rec, density, density_th, used);
    }
    else
    {
        return true;
    }
}

bool LSD::reduce_region_radius(std::vector<cv::Point2i>& reg, int& reg_size, double reg_angle, 
                double prec, double p, rect& rec, double density, const double& density_th, cv::Mat& used)
{
    //compute region's radius */
    double xc = (double) reg[0].x;
    double yc = (double) reg[0].y;
    double rad1 = dist(xc, yc, rec.x1, rec.y1);
    double rad2 = dist(xc, yc, rec.x2, rec.y2);
    double rad = rad1 > rad2 ? rad1 : rad2;

    while(density < density_th)
    {
        rad *= 0.75; // reduce region's radius to 75% of its value
        // remove points from the region and update 'used' map 
        for(int i = 0; i < reg_size; ++i)
        {
            if( dist( xc, yc, (double) reg[i].x, (double) reg[i].y ) > rad )
            {
                // Remove point from the region 
                used.data[reg[i].x + reg[i].y * used.cols] = NOTUSED;
                reg[i].x = reg[reg_size - 1].x;
                reg[i].y = reg[reg_size - 1].y;
                --reg_size;
                --i; // to avoid skipping one point 
            }
        }

        if(reg_size < 2) { return false; }

        // Re-compute rectangle 
        region2rect(reg, reg_size ,reg_angle, prec, p, rec);

        // Re-compute region points density
        density = (double) reg_size / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);
    }

    return true;
}

inline double LSD::dist(double x1, double y1, double x2, double y2)
{
  return sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

double LSD::rect_improve()
{
    // test return
    return LOG_EPS;
}

inline bool LSD::isAligned(const int& address, const double& theta, const double& prec)
{
    double a = angles_data[address];
    if (a == NOTDEF) { return false; }

    // It is assumed that 'theta' and 'a' are in the range [-pi,pi] 
    double n_theta = theta - a;
    if(n_theta < 0.0) { n_theta = -n_theta; }
    if(n_theta > M_3_2_PI)
    {
        n_theta -= M_2__PI;
        if( n_theta < 0.0 ) n_theta = -n_theta;
    }

    return n_theta <= prec;
}

// Absolute value angle difference 
inline double LSD::angle_diff(const double& a, const double& b) const
{
    double diff = a - b;
    while(diff <= -M_PI) diff += M_2__PI;
    while(diff >   M_PI) diff -= M_2__PI;
    if(diff < 0.0) diff = -diff;
    return diff;
}

// Signed angle difference 
inline double LSD::angle_diff_signed(const double& a, const double& b) const
{
    double diff = a - b;
    while(diff <= -M_PI) diff += M_2__PI;
    while(diff >   M_PI) diff -= M_2__PI;
    return diff;
}

// Compare doubles by relative error.
inline bool LSD::double_equal(const double& a, const double& b) const
{
    // trivial case
    if(a == b) return true;

    double abs_diff = fabs(a - b);
    double aa = fabs(a);
    double bb = fabs(b);
    double abs_max = (aa > bb)? aa : bb;

    if(abs_max < DBL_MIN) abs_max = DBL_MIN;

    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}
