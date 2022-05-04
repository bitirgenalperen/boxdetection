#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <string.h>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

// Global variable to set according to given arguments
String retrieve_dir_name, save_dir_name;
String retrieve_color_files, retrieve_mono_files;
String save_color_files, save_mono_files;
int color_var = 0;
int color_states[2] = {0,0};
int batch_size = 4; // default batch size is 4
int search_radius = 50;
int thresh = 80;
int blockSize = 5;
int apertureSize = 3;
int margin = 600;
int center_threshold = 300;
double area_ratio_threshold = 1.25;
double k = 0.06;
RNG rng(12345);


int argumentHandler(int arg_count, char** given_args) {
    // argument handling
    // argv[1] :: Image color variation(color, mono, both) (color_var)
    // argv[2] :: Name of the directory to retrieve images (retrieve_dir_name)
    // argv[3] :: Name of the directory to save the images with bounding boxes (save_dir_name)
    // argv[4] :: Batch size (batch_size)
    if (arg_count > 4) {
        cout << "You have entered " << arg_count - 1 << " arguments:" << endl;
        color_var = atoi(given_args[1]);
        retrieve_dir_name = given_args[2];
        save_dir_name = given_args[3];

        // Directory adjustments for retrieve and save images  
        if (color_var >= 1) { // color files
            color_states[1] = 1;
            retrieve_color_files = retrieve_dir_name + "/images/*.tiff";
            save_color_files = save_dir_name + "/";
            cout << "Retrieve Colored Images: " << retrieve_color_files << endl;
            cout << "Save Colored Images: " << save_color_files << endl;

        }
        if (color_var <= -1) { // mono files
            color_states[0] = 1;
            retrieve_mono_files = retrieve_dir_name + "/images_mono/*.tiff";
            save_mono_files = save_dir_name + "/";
            cout << "Retrieve Mono Images: " << retrieve_mono_files << endl;
            cout << "Save Mono Images: " << save_mono_files << endl;
        }

        // check if batch-size given
        // if given, also check the value
        // if not given, deafult value 4 will be preserved
        if (arg_count == 5) {
            batch_size = atoi(given_args[4]);
            if (batch_size < 1) {
                cout << "IMPROPER BATCH-SIZE" << endl;
                return -1;
            } else {
                cout << "Batch-size: " << batch_size << endl;
                return 0;
            }
        } else {
            return 0;
        }

    } else if (arg_count > 1) {
        cout << "Missing arg(s)!" << endl;
        return 1;
    } else {
        cout << "No arg given!" << endl;
        return 2;
    }
}


void gammaCorrection(const Mat& src, Mat& dst, const float gamma) {
    float invGamma = 1 / gamma;
    Mat table(1, 256, CV_8U);
    uchar* p = table.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = (uchar)(pow(i / 255.0, invGamma) * 255);
    }
    LUT(src, table, dst);
}


void processImages(String get_dir, String save_dir) {

    // counter variables
    int m, n;
    size_t count = 0;
    // vectors holding the file names
    vector<String> file_names;
    // vectors holding images
    vector<Mat> images;

    Mat reader;

    glob(get_dir, file_names, false);
    count = file_names.size();
    for (m = 0; m < count; m++) {
        // read and store images of batch-size
        cout << m << "  :: vector-size:   " << images.size() + 1 << endl;
        reader = imread(file_names[m]);
        if (reader.empty()) {
            cout << "CANNOT READ" << endl;
            break;
        }
        images.push_back(reader);

        if ((m + 1) % batch_size == 0 or m == (count - 1)) {
            // process images
            // get bounding boxes and rotation angle
            for (n = 0; n < batch_size; n++) {
                Mat src, src_blur, src_gray;
                Mat frame1, frame2, canny_output;
                Scalar color;
                src = images[n];

                // color adjustments
                cvtColor(src, src_gray, COLOR_BGR2GRAY);
                gammaCorrection(src_gray, src_gray, 1.8);

                // edge detection
                Canny(src_gray, frame1, 30, 80, 3, true);
                GaussianBlur(frame1, frame1, Size(5, 5), 3);
                frame2 = frame1.clone();
                canny_output = frame1.clone();

                //
                vector<vector<Point> > contours;
                vector<Vec4i> hierarchy;

                findContours(frame2, contours, hierarchy, RETR_TREE,
                             CHAIN_APPROX_SIMPLE, Point(0, 0));

                vector<RotatedRect> minRects(contours.size());
                for (size_t i = 0; i < contours.size(); i++) {
                    minRects[i] = minAreaRect(contours[i]);
                }

                // post-process the found rectangles
                vector<RotatedRect> savedRects = minRects;
                vector<RotatedRect> bestFits;
                int margin_x = src.rows * 0.2, margin_y = src.cols * 0.3;
                for (size_t i = 0; i < savedRects.size(); i++) {

                    color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256),
                                   rng.uniform(0, 256));
                    // contour
                    // drawContours(drawer, contours, (int)i, color);
                    // rotated rectangle
                    Size rect_size = savedRects[i].size;

                    if ((rect_size.area() > 1500000) and
                        (rect_size.area() <
                         8000000)) { // almost totally in or does not
                                     // cover the whole area
                        cout << rect_size.area() << endl;
                        bestFits.push_back(savedRects[i]);
                    }

                    else if (rect_size.area() > 400000) {
                        cout << rect_size.area() << endl;
                        Point2f rect_center = savedRects[i].center;
                        int rect_x = rect_center.x, rect_y = rect_center.y;

                        if (((rect_x < margin_x) or
                             (rect_x > (src.rows - margin_x))) or
                            ((rect_y < margin_y) or
                             (rect_y > (src.cols - margin_y)))) {
                            bestFits.push_back(savedRects[i]);
                        }
                    }
                }

                // filter the overlapping rectangles
                vector<int> filter(bestFits.size());
                for (size_t i = 0; i < bestFits.size(); i++) {
                    filter[i] = 1;
                }
                for (int i = bestFits.size() - 1; i >= 0; i--) {
                    RotatedRect cur_rect = bestFits[i];
                    bool bounding = false;
                    for (int j = 0; j < bestFits.size(); j++) {
                        RotatedRect int_rect = bestFits[j];
                        float center_diff =
                            norm(int_rect.center - cur_rect.center);
                        float area_diff =
                            abs(int_rect.size.area() - cur_rect.size.area());
                        float area_ratio =
                            max(int_rect.size.area(), cur_rect.size.area()) /
                            min(int_rect.size.area(), cur_rect.size.area());
                        if ((center_diff < 300) and (area_ratio < 3)) {
                            if (int_rect.size.area() < cur_rect.size.area()) {
                                filter[i] = 1;
                                filter[j] = 0;
                            }
                            if (int_rect.size.area() > cur_rect.size.area()) {
                                filter[i] = 0;
                                filter[j] = 1;
                            }
                        }
                    }
                }
                //write the details on image
                for (size_t i = 0; i < bestFits.size(); i++) {
                    if (filter[i] == 1) {
                        RotatedRect cur_rect = bestFits[i];
                        Point2f rect_points[4];
                        cur_rect.points(rect_points);

                        for (int j = 0; j < 4; j++) {
                            line(src, rect_points[j], rect_points[(j + 1) % 4],
                                 color, 25);
                            circle(src, rect_points[j], 5, Scalar(0, 0, 255),
                                   10);
                            String corner_pos =
                                "(" + to_string(int(rect_points[j].x)) + ", " +
                                to_string(int(rect_points[j].y)) + ")";
                            putText(src, corner_pos, rect_points[j], 1, 8,
                                    Scalar(0, 0, 255), 10);
                        }
                        String title_angle_area =
                            "Angle: " + to_string(int(cur_rect.angle)) +
                            " :: " + "Area: " +
                            to_string(int(cur_rect.size.area()));

                        String title_size =
                            "Width-Height: (" +
                            to_string(int(cur_rect.size.width)) + ", " +
                            to_string(int(cur_rect.size.height)) + ")";
                        putText(
                            src, title_angle_area,
                            Point(cur_rect.center.x - cur_rect.size.width / 2,
                                  cur_rect.center.y - 100),
                            1, 8, Scalar(0, 0, 255), 10);
                        putText(
                            src, title_size,
                            Point(cur_rect.center.x - cur_rect.size.width / 2,
                                  cur_rect.center.y),
                            1, 8, Scalar(0, 0, 255), 10);
                    }
                    // save processed image
                    // save corresponding data
                    String file_name = file_names[m - batch_size + n + 1];
                    imwrite(file_name, src);
                }

            }
            cout << "Clear vector" << endl;
            images.clear();
        }
    }
}


int main(int argc, char** argv){
    if (argumentHandler(argc, argv))
        return 1;
    if (color_var >= 0) { // colored images
        cout << "Starting to Colored Images.." << endl;
        processImages(retrieve_color_files,save_color_files);
    }
    if (color_var <= 0) { // mono images
        cout << "Starting to Mono Images..." << endl;
        processImages(retrieve_mono_files,save_mono_files);
    }
    return 0;
}
