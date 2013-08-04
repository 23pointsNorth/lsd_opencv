.. seealso::

    :ocv:class:`LineSegmentDetector`


LineSegmentDetector
-------------------
Line segment detector class, following the algorithm described at [Rafael12]_.

.. ocv:class:: LineSegmentDetector : public Algorithm


createLineSegmentDetectorPtr
----------------------------
Creates a smart pointer to a LineSegmentDetector object and initializes it.

.. ocv:function:: Ptr<LineSegmentDetector> createLineSegmentDetectorPtr(int _refine = LSD_REFINE_STD, double _scale = 0.8, double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5, double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024)

    :param _refine: The way found lines will be refined:

        * **LSD_REFINE_NONE** - No refinement applied.

        * **LSD_REFINE_STD**  - Standard refinement is applied. E.g. breaking arches into smaller straighter line approximations.

        * **LSD_REFINE_ADV**  - Advanced refinement. Number of false alarms is calculated, lines are refined through increase of precision, decrement in size, etc.

    :param scale: The scale of the image that will be used to find the lines. Range (0..1].

    :param sigma_scale: Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.

    :param quant: Bound to the quantization error on the gradient norm.

    :param ang_th: Gradient angle tolerance in degrees.

    :param log_eps: Detection threshold: -log10(NFA) > _log_eps. Used only when advancent refinement is chosen.

    :param density_th: Minimal density of aligned region points in the enclosing rectangle.

    :param n_bins: Number of bins in pseudo-ordering of gradient modulus.

The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want to edit those, as to tailor it for their own application.


LineSegmentDetector::detect
---------------------------
Finds lines in the input image. See the lsd_lines.cpp sample for possible usage.

.. ocv:function:: void detect(const InputArray _image, OutputArray _lines, OutputArray width = noArray(), OutputArray prec = noArray(), OutputArray nfa = noArray())

    :param _image A grayscale (CV_8UC1) input image.
        If only a roi needs to be selected, use ::
        lsd_ptr->detect(image(roi), lines, ...);
        lines += Scalar(roi.x, roi.y, roi.x, roi.y);

    :param lines: A vector of Vec4i elements specifying the beginning and ending point of a line. Where Vec4i is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly oriented depending on the gradient.

    :param width: Vector of widths of the regions, where the lines are found. E.g. Width of line.

    :param prec: Vector of precisions with which the lines are found.

    :param nfa: Vector containing number of false alarms in the line region, with precision of 10%. The bigger the value, logarithmically better the detection.

        * -1 corresponds to 10 mean false alarms
        * 0 corresponds to 1 mean false alarm
        * 1 corresponds to 0.1 mean false alarms

    This vector will be calculated only when the objects type is LSD_REFINE_ADV.


LineSegmentDetector::drawSegments
---------------------------------
Draws the line segments on a given image.

.. ocv:function:: void drawSegments(InputOutputArray image, const InputArray lines)

    :param image: The image, where the liens will be drawn. Should be bigger or equal to the image, where the lines were found.

    :param lines: A vector of the lines that needed to be drawn.


LineSegmentDetector::compareSegments
------------------------------------
Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.

.. ocv:function:: int compareSegments(const Size& size, const InputArray lines1, const InputArray lines2, Mat* image = 0)

    :param size: The size of the image, where the lines were found.
    :param lines1: The first group of lines that needs to be drawn. It is visualized in blue color.
    :param lines2: The second group of lines. They visualized in red color.
    :param image: Optional image, where the lines will be drawn. The image is converted to grayscale before displaying, leaving lines1 and lines2 in the above mentioned colors.


.. [Rafael12] Rafael Grompone von Gioi, Jérémie Jakubowicz, Jean-Michel Morel, and Gregory Randall, LSD: a Line Segment Detector, Image Processing On Line, vol. 2012. http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
