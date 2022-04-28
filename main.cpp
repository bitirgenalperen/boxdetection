#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <string.h>
#include <vector>

using namespace cv;
using namespace std;

// Global variable to set according to given arguments
String retrieve_dir_name, save_dir_name;
String retrieve_color_files, retrieve_mono_files;
String save_color_files, save_mono_files;
int color_var = 0;
int color_states[2] = {0,0};
int batch_size = 4; // default batch size is 4
int search_radius = 60;
int thresh = 80;
int blockSize = 5;
int apertureSize = 3;
int margin = 300;
int center_threshold = 300;
double area_ratio_threshold = 1.25;
double k = 0.06;
RNG rng(12345);


int argumentHandler(int arg_count, char** given_args) {
    // argument handling
    // argv[1] :: Image color variation(color, mono, both) (color_var)
    // argv[2] :: Name of the directory to retrieve images (retrieve_dir_name)
    // argv[3] :: Name of the directory to save the images with bounding boxes
    // (save_dir_name) argv[4] :: Batch size (batch_size)
    if (arg_count > 4) {
        cout << "You have entered " << arg_count - 1 << " arguments:" << endl;
        color_var = atoi(given_args[1]);
        retrieve_dir_name = given_args[2];
        save_dir_name = given_args[3];

        // Directory adjustments for retrieve and save images  
        if (color_var >= 1) { // color files
            color_states[1] = 1;
            retrieve_color_files = retrieve_dir_name + "/images/*.tiff";
            save_color_files = retrieve_dir_name + "/images/";
            cout << "Retrieve Colored Images: " << retrieve_color_files << endl;
            cout << "Save Colored Images: " << save_color_files << endl;
        }
        if (color_var <= -1) { // mono files
            color_states[0] = 1;
            retrieve_mono_files = retrieve_dir_name + "/images_mono/*.tiff";
            save_mono_files = retrieve_dir_name + "/images_mono/";
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



void processColorImages() {

    // counter variables
    size_t i, j;
    size_t color_count = 0;
    // vectors holding the file names
    vector<String> colored_file_names;
    // vectors holding images
    vector<Mat> color_images;

    Mat reader;

    glob(retrieve_color_files, colored_file_names, false);
    color_count = colored_file_names.size();
    for (i = 0; i < color_count; i++) {
        // read and store images of batch-size
        cout << i << "  :: vector-size:   " << color_images.size() + 1 << endl;
        reader = imread(colored_file_names[i]);
        if (reader.empty()) {
            cout << "CANNOT READ" << endl;
            break;
        }
        color_images.push_back(reader);

        if ((i+1) % batch_size == 0 or i == (color_count - 1)) {
            // process images
            // get bounding boxes and rotation angle
            for (j = 0; j < batch_size; j++) {
                Mat src = color_images[j];
                Mat src_blur, src_gray, frame1, frame2, canny_output;
                int padding_y = src.rows * 0.01;
                copyMakeBorder(src, src, padding_y, padding_y, 0, 0, BORDER_CONSTANT, Scalar(0, 0, 0));
                GaussianBlur(src, src_blur, Size(7, 7), 3);
                cvtColor(src_blur, src_gray, COLOR_BGR2GRAY);
                Canny(src_gray, frame1, 25, 90, 3, true);
                frame2 = frame1.clone();
                canny_output = frame1.clone();

                vector<vector<Point> > contours;
                vector<Vec4i> hierarchy;
                findContours(canny_output, contours, hierarchy, RETR_TREE,
                             CHAIN_APPROX_SIMPLE, Point(0, 0));

                Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256),
                                      rng.uniform(0, 256));

                Mat dst = Mat::zeros(src_blur.size(), CV_32FC1);
                cornerHarris(frame1, dst, blockSize, apertureSize, k);
                Mat dst_norm, dst_norm_scaled;
                normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
                convertScaleAbs(dst_norm, dst_norm_scaled);
                for (int i = 0; i < dst_norm.rows; i++) {
                    for (int j = 0; j < dst_norm.cols; j++) {
                        if ((int)dst_norm.at<float>(i, j) > thresh) {
                            int hold_max = (int)dst_norm.at<float>(i, j);
                            int search_max = (int)dst_norm.at<float>(i, j);
                            for (int k = -search_radius; k < search_radius;
                                 k++) {
                                for (int l = -search_radius; l < search_radius;
                                     l++) {
                                    int xx = min(max(i + k, 0), dst_norm.rows);
                                    int yy = min(max(j + l, 0), dst_norm.cols);
                                    if ((int)dst_norm.at<float>(xx, yy) >
                                        search_max) {
                                        search_max =
                                            (int)dst_norm.at<float>(xx, yy);
                                    }
                                }
                            }
                            if (search_max == hold_max) {
                                circle(frame2, Point(j, i), 35, color, 35, 8,
                                       0);
                            }
                        }
                    }
                }

                contours.clear();
                hierarchy.clear();
                findContours(frame2, contours, hierarchy, RETR_TREE,
                             CHAIN_APPROX_SIMPLE, Point(0, 0));

                vector<RotatedRect> minRects(contours.size());
                for (size_t i = 0; i < contours.size(); i++) {
                    minRects[i] = minAreaRect(contours[i]);
                }

                vector<RotatedRect> savedRects;

                for (size_t k = 0; k < minRects.size(); k++) {
                    bool flag_save = true;
                    for (size_t l = 0; l < minRects.size(); l++) {
                        if ((norm(minRects[k].center - minRects[l].center) <
                             margin) and
                            ((minRects[l].size.area() >
                              minRects[k].size.area()) and
                             ((minRects[l].size.area() /
                               minRects[k].size.area()) <
                              area_ratio_threshold))) {
                            flag_save = false;
                            break;
                        }
                    }
                    if (flag_save) {
                        savedRects.push_back(minRects[k]);
                    }
                }

                for (size_t i = 0; i < savedRects.size(); i++) {

                    color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256),
                                   rng.uniform(0, 256));
                    // contour
                    // drawContours(drawer, contours, (int)i, color);
                    // rotated rectangle
                    Size rect_size = savedRects[i].size;

                    if ((rect_size.area() > 300000) and
                        (rect_size.area() <
                         9000000)) { // almost totally in or does not cover the
                                     // whole area
                        Point2f rect_points[4];
                        savedRects[i].points(rect_points);
                        for (int j = 0; j < 4; j++) {
                            line(src, rect_points[j], rect_points[(j + 1) % 4],
                                 color, 25);
                        }
                    }

                    else if (rect_size.area() > 200000) {
                        Point2f rect_center = savedRects[i].center;
                        int rect_x = rect_center.x, rect_y = rect_center.y;
                        if (((rect_x < margin) or
                             (rect_x > (src.rows - margin))) and
                            ((rect_y < margin) or
                             (rect_y > (src.cols - margin)))) {

                            Point2f rect_points[4];
                            savedRects[i].points(rect_points);
                            for (int j = 0; j < 4; j++) {
                                line(src, rect_points[j],
                                     rect_points[(j + 1) % 4], color, 25);
                            }
                        }
                    }
                }

                // save processed image
                // save corresponding data
                String file_name = "processed_" + save_color_files +
                                   colored_file_names[i - batch_size + j];
                String file_name2 = "interestpoint_" + file_name;
                imwrite(file_name, src);
                imwrite(file_name2, frame2);
            }

            cout << "Clear vector" << endl;
            color_images.clear();
        }
    }

}

void processMonoImages() {
    // counter variables
    size_t i, j;
    size_t mono_count = 0;
    // vector holding the file names
    vector<String> mono_file_names;
    // vectors holding images
    vector<Mat> mono_images;

    Mat reader, reader_gamma,reader_ab;

    glob(retrieve_mono_files, mono_file_names, false);
    mono_count = mono_file_names.size();
    for (i = 0; i < mono_count; i++) {
        cout << i << "  :: vector-size:   " << mono_images.size() + 1 << endl;
        reader = imread(mono_file_names[i]);
        if (reader.empty()) {
            cout << "CANNOT READ" << endl;
            break;
        }
        // pre-process mono-chromatic images: gamma, alpha-beta adjustments
        gammaCorrection(reader,reader_gamma,2.2);
        reader_gamma.convertTo(reader_ab,-1,1.7,10);
        mono_images.push_back(reader_ab);
        if ((i + 1) % batch_size == 0 or i == (mono_count - 1)) {
            // process images
            // get bounding boxes and rotation angle
            for (j = 0; j < batch_size; j++) {
                Mat src = mono_images[j];
                Mat src_blur, src_gray, frame1, frame2, canny_output;
                int padding_y = src.rows * 0.01;
                copyMakeBorder(src, src, padding_y, padding_y, 0, 0, BORDER_CONSTANT, Scalar(0, 0, 0));
                GaussianBlur(src, src_blur, Size(7, 7), 3);
                cvtColor(src_blur, src_gray, COLOR_BGR2GRAY);
                Canny(src_gray, frame1, 25, 90, 3, true);
                frame2 = frame1.clone();
                canny_output = frame1.clone();


                vector<vector<Point> > contours;
                vector<Vec4i> hierarchy;
                findContours(canny_output, contours, hierarchy, RETR_TREE,
                             CHAIN_APPROX_SIMPLE, Point(0, 0));

                Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256),
                                      rng.uniform(0, 256));


                Mat dst = Mat::zeros(src_blur.size(), CV_32FC1);
                cornerHarris(frame1, dst, blockSize, apertureSize, k);
                Mat dst_norm, dst_norm_scaled;
                normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
                convertScaleAbs(dst_norm, dst_norm_scaled);
                for (int i = 0; i < dst_norm.rows; i++) {
                    for (int j = 0; j < dst_norm.cols; j++) {
                        if ((int)dst_norm.at<float>(i, j) > thresh) {
                            int hold_max = (int)dst_norm.at<float>(i, j);
                            int search_max = (int)dst_norm.at<float>(i, j);
                            for (int k = -search_radius; k < search_radius;
                                 k++) {
                                for (int l = -search_radius; l < search_radius;
                                     l++) {
                                    int xx = min(max(i + k, 0), dst_norm.rows);
                                    int yy = min(max(j + l, 0), dst_norm.cols);
                                    if ((int)dst_norm.at<float>(xx, yy) >
                                        search_max) {
                                        search_max =
                                            (int)dst_norm.at<float>(xx, yy);
                                    }
                                }
                            }
                            if (search_max == hold_max) {
                                circle(frame2, Point(j, i), 35, color, 35, 8,
                                       0);
                            }
                        }
                    }
                }

                contours.clear();
                hierarchy.clear();
                findContours(frame2, contours, hierarchy, RETR_TREE,
                             CHAIN_APPROX_SIMPLE, Point(0, 0));

                vector<RotatedRect> minRects(contours.size());
                for (size_t i = 0; i < contours.size(); i++) {
                    minRects[i] = minAreaRect(contours[i]);
                }


                // post-process the found rectangles
                vector<RotatedRect> savedRects;

                for (size_t k = 0; k < minRects.size(); k++) {
                    bool flag_save = true;
                    for (size_t l = 0; l < minRects.size(); l++) {
                        if ((norm(minRects[k].center - minRects[l].center) <
                             margin) and
                            ((minRects[l].size.area() >
                              minRects[k].size.area()) and
                             ((minRects[l].size.area() /
                               minRects[k].size.area()) <
                              area_ratio_threshold))) {
                            flag_save = false;
                            break;
                        }
                    }
                    if (flag_save) {
                        savedRects.push_back(minRects[k]);
                    }
                }

                for (size_t i = 0; i < savedRects.size(); i++) {

                    color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256),
                                   rng.uniform(0, 256));
                    // contour
                    // drawContours(drawer, contours, (int)i, color);
                    // rotated rectangle
                    Size rect_size = savedRects[i].size;

                    if ((rect_size.area() > 300000) and
                        (rect_size.area() <
                         9000000)) { // almost totally in or does not cover the
                                     // whole area
                        Point2f rect_points[4];
                        savedRects[i].points(rect_points);
                        for (int j = 0; j < 4; j++) {
                            line(src, rect_points[j], rect_points[(j + 1) % 4],
                                 color, 25);
                        }
                    }

                    else if (rect_size.area() > 200000) {
                        Point2f rect_center = savedRects[i].center;
                        int rect_x = rect_center.x, rect_y = rect_center.y;
                        if (((rect_x < margin) or
                             (rect_x > (src.rows - margin))) and
                            ((rect_y < margin) or
                             (rect_y > (src.cols - margin)))) {

                            Point2f rect_points[4];
                            savedRects[i].points(rect_points);
                            for (int j = 0; j < 4; j++) {
                                line(src, rect_points[j],
                                     rect_points[(j + 1) % 4], color, 25);
                            }
                        }
                    }
                }

                // save processed image
                // save corresponding data
                String file_name =
                    "processed_" + save_mono_files + mono_file_names[i - batch_size + j];
                String file_name2 = "interestpoint_" +  file_name;
                imwrite(file_name, src);
                imwrite(file_name2, frame2);


            }
            // after each batch clear the vector
            cout << "Clear vector" << endl;
            mono_images.clear();
        }
    }

}


int main(int argc, char** argv) {
 
    if (argumentHandler(argc, argv))
        return 1;

    if (color_var >= 0) { // colored images
        cout << "Starting to Colored Images.." << endl;
        processColorImages();
    }
    if (color_var <= 0) { // mono images
        cout << "Starting to Mono Images..." << endl;
        processMonoImages();
    }

    return 0;
}
