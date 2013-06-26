#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cfloat>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> // Only for imshow

#include "lsd_opencv.hpp"
#include "lsd_const.hpp"

using namespace cv;

inline double distSq(const double x1, const double y1, const double x2, const double y2)
{
    return (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1);
}

inline double dist(const double x1, const double y1, const double x2, const double y2)
{
    return sqrt(distSq(x1, y1, x2, y2));
}

// Signed angle difference
inline double angle_diff_signed(const double& a, const double& b)
{
    double diff = a - b;
    while(diff <= -M_PI) diff += M_2__PI;
    while(diff >   M_PI) diff -= M_2__PI;
    return diff;
}

// Absolute value angle difference
inline double angle_diff(const double& a, const double& b)
{
    return std::fabs(angle_diff_signed(a, b));
}

// Compare doubles by relative error.
inline bool double_equal(const double& a, const double& b)
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

LSD::LSD(double _scale, int _subdivision, bool _refine, double _sigma_scale, double _quant, double _ang_th, double _log_eps, double _density_th, int _n_bins)
        :SCALE(_scale), doRefine(_refine), SUBDIVISION(_subdivision), SIGMA_SCALE(_sigma_scale), QUANT(_quant), ANG_TH(_ang_th), LOG_EPS(_log_eps), DENSITY_TH(_density_th), N_BINS(_n_bins)
{
    CV_Assert(_scale > 0 && _sigma_scale > 0 && _quant >= 0 &&
              _ang_th > 0 && _ang_th < 180 && _density_th >= 0 && _density_th < 1 &&
              _n_bins > 0 && _subdivision > 0);
}

void LSD::detect(const cv::InputArray& _image, cv::OutputArray& _lines, cv::Rect _roi,
                cv::OutputArray& width, cv::OutputArray& prec,
                cv::OutputArray& nfa)
{
    Mat_<double> img = _image.getMat();
    CV_Assert(!img.empty() && img.channels() == 1);

    // If default, then convert the whole image, else just the specified by roi
    roi = _roi;
    if (roi.area() == 0)
    {
        img.convertTo(image, CV_64FC1);
    }
    else
    {
        roix = roi.x;
        roiy = roi.y;
        img(roi).convertTo(image, CV_64FC1);
    }

    std::vector<Vec4i> lines;
    std::vector<double>* w = (width.needed())?(new std::vector<double>()):0;
    std::vector<double>* p = (prec.needed())?(new std::vector<double>()):0;
    std::vector<double>* n = (nfa.needed())?(new std::vector<double>()):0;

    flsd(image, lines, w, p, n);

    Mat(lines).copyTo(_lines);
    if (w) Mat(*w).copyTo(width); 
    if (p) Mat(*p).copyTo(prec);
    if (n) Mat(*n).copyTo(nfa);

    delete w;
    delete p;
    delete n;
}

void LSD::flsd(const Mat_<double>& _image, std::vector<Vec4i>& lines, 
    std::vector<double>* widths, std::vector<double>* precisions, 
    std::vector<double>* nfas)
{
    // Angle tolerance
    const double prec = M_PI * ANG_TH / 180.0;
    const double p = ANG_TH / 180.0;
    const double rho = QUANT / sin(prec);    // gradient magnitude threshold
 
    vector<coorlist> list;
    if (SCALE != 1)
    {
        //TODO: Asses Gaussian blur, as scaling down applies.
        Mat gaussian_img;
        const double sigma = (SCALE < 1.0)?(SIGMA_SCALE / SCALE):(SIGMA_SCALE);
        const double sprec = 3.0;
        const unsigned int h =  (unsigned int)(ceil(sigma * sqrt(2.0 * sprec * log(10.0))));
        Size ksize(1 + 2 * h, 1 + 2 * h); // kernel size 
        GaussianBlur(image, gaussian_img, ksize, sigma);
        // Scale image to needed size
        resize(gaussian_img, scaled_image, Size(), SCALE, SCALE);
        ll_angle(rho, N_BINS, list);
    }
    else
    {
        scaled_image = image;
        ll_angle(rho, N_BINS, list);
    }

    const double logNT = 5.0 * (log10(double(img_width)) + log10(double(img_height))) / 2.0 + log10(11.0);
    const int min_reg_size = int(-logNT/log10(p)); // minimal number of points in region that can give a meaningful event 
    
    // // Initialize region only when needed
    // Mat region = Mat::zeros(scaled_image.size(), CV_8UC1);
    used = Mat_<uchar>::zeros(scaled_image.size()); // zeros = NOTUSED
    vector<RegionPoint> reg(img_width * img_height);
    
    // Search for line segments 
    unsigned int ls_count = 0;
    unsigned int list_size = list.size();
    for(unsigned int i = 0; i < list_size; ++i)
    {
        unsigned int adx = list[i].p.x + list[i].p.y * img_width;
        if((used.data[adx] == NOTUSED) && (angles_data[adx] != NOTDEF))
        {
            int reg_size;
            double reg_angle;
            region_grow(list[i].p, reg, reg_size, reg_angle, prec);
            
            // Ignore small regions
            if(reg_size < min_reg_size) { continue; }
            
            // Construct rectangular approximation for the region
            rect rec;
            region2rect(reg, reg_size, reg_angle, prec, p, rec);

            if (doRefine)
            {
                if(!refine(reg, reg_size, reg_angle, prec, p, rec, DENSITY_TH)) { continue; }
            }

            // Compute NFA
            double log_nfa = rect_improve();
            //if(log_nfa <= LOG_EPS) { continue; }
            
            // Found new line
            ++ls_count;

            // Add the offset
            rec.x1 += 0.5; rec.y1 += 0.5;
            rec.x2 += 0.5; rec.y2 += 0.5;

            // scale the result values if a sub-sampling was performed
            if(SCALE != 1.0)
            {
                rec.x1 /= SCALE; rec.y1 /= SCALE;
                rec.x2 /= SCALE; rec.y2 /= SCALE;
                rec.width /= SCALE;
            }
            
            if (roi.area()) // if a roi has been given by the user, adjust coordinates
            {
                rec.x1 += roix;
                rec.y1 += roiy;
                rec.x2 += roix;
                rec.y2 += roiy;
            }

            //Store the relevant data
            lines.push_back(cv::Vec4i(rec.x1, rec.y1, rec.x2, rec.y2));
            if (widths) widths->push_back(rec.width);
            if (precisions) precisions->push_back(rec.p);
            if (nfas) nfas->push_back(log_nfa);

            // //Add the linesID to the region on the image
            // for(unsigned int el = 0; el < reg_size; el++)
            // {
            //     region.data[reg[i].x + reg[i].y * width] = ls_count;
            // }

        }
    
    }
 
}

void LSD::ll_angle(const double& threshold, const unsigned int& n_bins, std::vector<coorlist>& list)
{
    //Initialize data
    angles = cv::Mat_<double>(scaled_image.size());
    modgrad = cv::Mat_<double>(scaled_image.size());
    
    angles_data = angles.ptr<double>(0);
    modgrad_data = modgrad.ptr<double>(0);
    scaled_image_data = scaled_image.ptr<double>(0);

    img_width = scaled_image.cols; 
    img_height = scaled_image.rows;

    // Undefined the down and right boundaries 
    angles.row(img_height - 1).setTo(NOTDEF);
    angles.col(img_width - 1).setTo(NOTDEF);
    
    // Computing gradient for remaining pixels
    CV_Assert(scaled_image.isContinuous() && 
              modgrad.isContinuous() && 
              angles.isContinuous());   // Accessing image data linearly
    double max_grad = 0.0;
    for(int y = 0; y < img_height - 1; ++y)
    {
        for(int addr = y * img_width, addr_end = addr + img_width - 1; addr < addr_end; ++addr)
        {
            double DA = scaled_image_data[addr + img_width + 1] - scaled_image_data[addr];
            double BC = scaled_image_data[addr + 1] - scaled_image_data[addr + img_width];
            double gx = DA + BC;    // gradient x component 
            double gy = DA - BC;    // gradient y component 
            double norm = std::sqrt((gx * gx + gy * gy)/4.0);   // gradient norm 
            
            modgrad_data[addr] = norm;    // store gradient

            if (norm <= threshold)  // norm too small, gradient no defined 
            {
                angles_data[addr] = NOTDEF;
            }
            else
            {
                angles_data[addr] = cv::fastAtan2(gx, -gy) * DEG_TO_RADS;   // gradient angle computation
                if (norm > max_grad) { max_grad = norm; }
            }

        }
    }
    
    // Compute histogram of gradient values
    list = vector<coorlist>(img_width * img_height);
    vector<coorlist*> range_s(n_bins);
    vector<coorlist*> range_e(n_bins);
    unsigned int count = 0;
    double bin_coef = double(n_bins - 1) / max_grad;

    for(int y = 0; y < img_height - 1; ++y)
    {
        const double* norm = modgrad_data + y * img_width;
        for(int x = 0; x < img_width - 1; ++x, ++norm)
        {
            // store the point in the right bin according to its norm 
            int i = int((*norm) * bin_coef);
            if(!range_e[i])
            {
                range_e[i] = range_s[i] = &list[count];
                ++count;
            }
            else
            {
                range_e[i]->next = &list[count];
                range_e[i] = &list[count];
                ++count;
            }
            range_e[i]->p = cv::Point(x, y);
            range_e[i]->next = 0;
        }
    }

    // Sort
    int idx = n_bins - 1;
    for(;idx > 0 && !range_s[idx]; --idx);
    coorlist* start = range_s[idx];
    coorlist* end = range_e[idx];
    if(start)
    {
        while(idx > 0)
        {
            --idx;
            if(range_s[idx])
            {
                end->next = range_s[idx];
                end = range_e[idx];
            }
        }
    }
}

void LSD::region_grow(const cv::Point2i& s, std::vector<RegionPoint>& reg,
                      int& reg_size, double& reg_angle, const double& prec)
{
    // Point to this region
    reg_size = 1;
    reg[0].x = s.x;
    reg[0].y = s.y;
    int addr = s.x + s.y * img_width;
    reg[0].used = used.data + addr;
    reg_angle = angles_data[addr];
    reg[0].angle = reg_angle;
    reg[0].modgrad = modgrad_data[addr];

    float sumdx = cos(reg_angle);
    float sumdy = sin(reg_angle);
    *reg[0].used = USED;

    //Try neighboring regions
    for(int i = 0; i < reg_size; ++i)
    {
        const RegionPoint& rpoint = reg[i];
        int xx_min = std::max(rpoint.x - 1, 0), xx_max = std::min(rpoint.x + 1, img_width - 1);
        int yy_min = std::max(rpoint.y - 1, 0), yy_max = std::min(rpoint.y + 1, img_height - 1);
        for(int yy = yy_min; yy <= yy_max; ++yy)
        {
                int c_addr = xx_min + yy * img_width;
                for(int xx = xx_min; xx <= xx_max; ++xx, ++c_addr)
                {
                    if((used.data[c_addr] != USED) &&
                       (isAligned(c_addr, reg_angle, prec)))
                    {
                        // Add point
                        used.data[c_addr] = USED;
                        RegionPoint& region_point = reg[reg_size];
                        region_point.x = xx;
                        region_point.y = yy;
                        region_point.used = &(used.data[c_addr]);
                        region_point.modgrad = modgrad_data[c_addr];
                        const double& angle = angles_data[c_addr];
                        region_point.angle = angle;
                        ++reg_size;

                        // Update region's angle
                        sumdx += cos(float(angle));
                        sumdy += sin(float(angle));
                        // reg_angle is used in the isAligned, so it needs to be updates?
                        reg_angle = cv::fastAtan2(sumdy, sumdx) * DEG_TO_RADS;
                    }
            }
        }
    }
}

void LSD::region2rect(const std::vector<RegionPoint>& reg, const int reg_size, const double reg_angle,
                      const double prec, const double p, rect& rec) const
{
    double x = 0, y = 0, sum = 0;
    for(int i = 0; i < reg_size; ++i)
    {
        const RegionPoint& pnt = reg[i];
        const double& weight = pnt.modgrad;
        x += double(pnt.x) * weight;
        y += double(pnt.y) * weight;
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
        double regdx = double(reg[i].x) - x;
        double regdy = double(reg[i].y) - y;
        
        double l = regdx * dx + regdy * dy;
        double w = -regdx * dy + regdy * dx;

        if(l > l_max) l_max = l;
        else if(l < l_min) l_min = l;
        if(w > w_max) w_max = w;
        else if(w < w_min) w_min = w;
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

double LSD::get_theta(const std::vector<RegionPoint>& reg, const int& reg_size, const double& x,
                      const double& y, const double& reg_angle, const double& prec) const
{
    double Ixx = 0.0;
    double Iyy = 0.0;
    double Ixy = 0.0;

    // compute inertia matrix 
    for(int i = 0; i < reg_size; ++i)
    {
        const double& regx = reg[i].x; 
        const double& regy = reg[i].y;
        const double& weight = reg[i].modgrad;
        double dx = regx - x;
        double dy = regy - y;
        Ixx += dy * dy * weight;
        Iyy += dx * dx * weight;
        Ixy -= dx * dy * weight;
    }

    // Check if inertia matrix is null
    CV_Assert(!(double_equal(Ixx, 0) && double_equal(Iyy, 0) && double_equal(Ixy, 0)));

    // Compute smallest eigenvalue
    double lambda = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy));

    // Compute angle
    double theta = (fabs(Ixx)>fabs(Iyy))?cv::fastAtan2(lambda - Ixx, Ixy):cv::fastAtan2(Ixy, lambda - Iyy); // in degs
    theta *= DEG_TO_RADS;

    // Correct angle by 180 deg if necessary 
    if(angle_diff(theta, reg_angle) > prec) { theta += M_PI; }

    return theta;
}

bool LSD::refine(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
                const double prec, double p, rect& rec, const double& density_th)
{
    double density = double(reg_size) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density >= density_th) { return true; }

    // Try to reduce angle tolerance
    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    const double& ang_c = reg[0].angle;
    double sum = 0, s_sum = 0;
    int n = 0;

    for (int i = 0; i < reg_size; ++i)
    {
        *(reg[i].used) = NOTUSED;
        if (dist(xc, yc, reg[i].x, reg[i].y) < rec.width)
        {
            const double& angle = reg[i].angle;
            double ang_d = angle_diff_signed(angle, ang_c);
            sum += ang_d;
            s_sum += ang_d * ang_d;
            ++n;
        }
    }
    double mean_angle = sum / double(n);
    // 2 * standard deviation
    double tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / double(n) + mean_angle * mean_angle ); 

    // Try new region
    region_grow(Point(reg[0].x, reg[0].y), reg, reg_size, reg_angle, tau);

    if (reg_size < 2) { return false; }

    region2rect(reg, reg_size, reg_angle, prec, p, rec);
    density = double(reg_size) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density < density_th) 
    { 
        return reduce_region_radius(reg, reg_size, reg_angle, prec, p, rec, density, density_th);
    }
    else
    {
        return true;
    }
}

bool LSD::reduce_region_radius(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
                const double prec, double p, rect& rec, double density, const double& density_th)
{
    // Compute region's radius
    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    double radSq1 = distSq(xc, yc, rec.x1, rec.y1);
    double radSq2 = distSq(xc, yc, rec.x2, rec.y2);
    double radSq = radSq1 > radSq2 ? radSq1 : radSq2;

    while(density < density_th)
    {
        radSq *= 0.75*0.75; // reduce region's radius to 75% of its value
        // remove points from the region and update 'used' map 
        for(int i = 0; i < reg_size; ++i)
        {
            if(distSq( xc, yc, double(reg[i].x), double(reg[i].y)) > radSq)
            {
                // Remove point from the region 
                *(reg[i].used) = NOTUSED;
                std::swap(reg[i], reg[reg_size - 1]);
                --reg_size;
                --i; // to avoid skipping one point 
            }
        }

        if(reg_size < 2) { return false; }

        // Re-compute rectangle 
        region2rect(reg, reg_size ,reg_angle, prec, p, rec);

        // Re-compute region points density
        density = double(reg_size) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);
    }

    return true;
}

double LSD::rect_improve()
{
    // test return
    return LOG_EPS;
}

inline bool LSD::isAligned(const int& address, const double& theta, const double& prec) const
{
    const double& a = angles_data[address];
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


void LSD::drawSegments(cv::Mat& image, const std::vector<cv::Vec4i>& lines)
{
    CV_Assert(!image.empty() && (image.channels() == 1 || image.channels() == 3));

    Mat gray;
    if (image.channels() == 1)
    {
        gray = image;
    }
    else if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    }
    
    // Create a 3 channel image in order to draw colored lines
    vector<Mat> planes;
    planes.push_back(gray);
    planes.push_back(gray);
    planes.push_back(gray);

    merge(planes, image);

    // Draw segments
    for(unsigned int i = 0; i < lines.size(); ++i)
    {
        Point b(lines[i][0], lines[i][1]);
        Point e(lines[i][2], lines[i][3]);
        line(image, b, e, Scalar(0, 0, 255), 1);
    }
}

void LSD::showSegments(const std::string& name, const cv::Mat& image, const std::vector<cv::Vec4i>& lines)
{
    Mat img = image.clone();
    drawSegments(img, lines);
    imshow(name.c_str(), img);
}

int LSD::compareSegments(cv::Size& size, const std::vector<cv::Vec4i>& lines1, const std::vector<cv::Vec4i> lines2, cv::Mat* image)
{
    //CV_Assert(!image.empty() && (image.channels() == 1 || image.channels() == 3));
    
    if (image && image->size() != size) size = image->size();
    CV_Assert(size.area());

    Mat_<uchar> I1 = Mat_<uchar>::zeros(size);
    Mat_<uchar> I2 = Mat_<uchar>::zeros(size);

    // Draw segments
    for(unsigned int i = 0; i < lines1.size(); ++i)
    {
        Point b(lines1[i][0], lines1[i][1]);
        Point e(lines1[i][2], lines1[i][3]);
        line(I1, b, e, Scalar::all(255), 1);
    }
    for(unsigned int i = 0; i < lines2.size(); ++i)
    {
        Point b(lines2[i][0], lines2[i][1]);
        Point e(lines2[i][2], lines2[i][3]);
        line(I2, b, e, Scalar::all(255), 1);
    }

    // Count the pixels that don't agree
    Mat Ixor;
    bitwise_xor(I1, I2, Ixor);
    int N = countNonZero(Ixor);

    if (image)
    {
        Mat Ig;
        if (image->channels() == 1)
        {
            cv::cvtColor(*image, *image, CV_GRAY2BGR);   
        }
        CV_Assert(image->isContinuous() && I1.isContinuous() && I2.isContinuous());
        for (unsigned int i = 0; i < I1.total(); ++i)
        {
            uchar i1 = I1.data[i];
            uchar i2 = I2.data[i];
            if (i1 || i2)
            {
                image->data[3*i + 1] = 0;
                if (i1) image->data[3*i] = 255;
                else image->data[3*i] = 0;
                if (i2) image->data[3*i + 2] = 255;
                else image->data[3*i + 2] = 0;
            }
        }
    }
    
    return N;
}

int LSD::showSegments(const std::string& name, cv::Size size, const std::vector<cv::Vec4i>& lines1, const std::vector<cv::Vec4i> lines2, cv::Mat* image)
{
    Mat img;
    if (!image) img = Mat_<uchar>::zeros(size); 
    else img = image->clone();
    int n = compareSegments(size, lines1, lines2, &img);
    imshow(name.c_str(), img);
    return n;
}
