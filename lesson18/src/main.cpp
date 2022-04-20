#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <set>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <memory>

#include <libutils/rasserts.h>
#include <libutils/fast_random.h>


// Эта функция говорит нам правда ли пиксель отмаскирован, т.е. отмечен как "удаленный", т.е. белый
// j - rows i - cols
bool isPixelMasked(cv::Mat &mask, int j, int i) {
    rassert(j >= 0 && j < mask.rows, 372489347280017);
    rassert(i >= 0 && i < mask.cols, 372489347280018);
    rassert(mask.type() == CV_8UC3, 2348732984792380019);

    if(mask.at<cv::Vec3b>(j,i)[0] >= 255 && mask.at<cv::Vec3b>(j,i)[1] >= 255 && mask.at<cv::Vec3b>(j,i)[2] >= 255)
        return true;
    return false;
}
/*
 * (-2,2) (-1,2) (0,2) (1,2) (2,2)
 * (-2,1) (-1,1) (0,1) (1,1) (2,1)
 * (-2,0) (-1,0) (0,0) (1,0) (2,0)
 * (-2,-1) (-1,-1) (0,-1) (1,-1) (1,-2)
 * (-2,-2) (-1,-2) (0,-2) (2,-1) (2,-2)
*/
std::vector<int> dx = {-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2};
std::vector<int> dy = {2,2,2,2,2,1,1,1,1,1,0,0,0,0,0,-1,-1,-1,-1,-1,-2,-2,-2,-2,-2};
double estimateQuality(cv::Mat &image, cv::Mat &mask, int r1, int c1, int r2, int c2){
    const int colsNum = image.cols;
    const int rowsNum = image.rows;
    double quol = 0;
    for(int i = 0; i < dx.size();++i){
        if(r1+dx[i] >= 0 && r1+dx[i] < rowsNum && r2+dx[i] >= 0 && r2+dx[i] < rowsNum && c1+dy[i] >= 0 && c1+dy[i] < colsNum && c2+dy[i] >= 0 && c2+dy[i] < colsNum){
            if(isPixelMasked(mask, r2+dx[i], c2+dy[i])){
                quol = 1e8;
            }
            double d1 = abs(image.at<cv::Vec3b>(r1+dx[i],c1+dy[i])[1] - image.at<cv::Vec3b>(r2+dx[i],c2+dy[i])[1]);
            double d2 = abs(image.at<cv::Vec3b>(r1+dx[i],c1+dy[i])[2] - image.at<cv::Vec3b>(r2+dx[i],c2+dy[i])[2]);
            double d0 = abs(image.at<cv::Vec3b>(r1+dx[i],c1+dy[i])[0] - image.at<cv::Vec3b>(r2+dx[i],c2+dy[i])[0]);
            quol += abs(d0+d1+d2);
        }
        else
            quol = 1e8;
    }
    return quol;
}

void refinement(cv::Mat& mask, cv::Mat& shift, cv::Mat& original, std::pair<int,int>& i, FastRandom& random){
    int dc = random.next(2, original.cols-3) - i.second;
    int dr = random.next(2,original.rows-3) - i.first;
    while(isPixelMasked(mask, dr + i.first, dc + i.second)){
        dc = random.next(2, original.cols-3) - i.second;
        dr = random.next(2,original.rows-3) - i.first;
    }
    int r = shift.at<cv::Vec2i>(i.first, i.second)[0];
    int c = shift.at<cv::Vec2i>(i.first, i.second)[1];
    int escNew = estimateQuality(original, mask, i.first, i.second, i.first + dr, i.second + dc);
    int esc = estimateQuality(original, mask, i.first, i.second, i.first + r, i.second + c);
    if(escNew < esc){
        original.at<cv::Vec3b>(i.first, i.second) = original.at<cv::Vec3b>(dr+i.first,dc+i.second);
        shift.at<cv::Vec2i>(i.first, i.second)[0] = dr;
        shift.at<cv::Vec2i>(i.first, i.second)[0] = dc;
    }
}

std::vector<int> ddx = {1,0,-1,0};
std::vector<int> ddy = {0,1,0,-1};

void propagation(cv::Mat& mask, cv::Mat& shift, cv::Mat& original, std::pair<int,int>& i, FastRandom& random){
    int r = shift.at<cv::Vec2i>(i.first, i.second)[0];
    int c = shift.at<cv::Vec2i>(i.first, i.second)[1];
    int esc = estimateQuality(original, mask, i.first, i.second, i.first + r, i.second + c);
    for(int j = 0; j < 4; ++j){
        if(i.first+ddx[j] >= 0 && i.first+ddx[j] < original.rows && i.second+ddy[j] >= 0 && i.second+ddy[j] < original.cols && isPixelMasked(mask, i.first+ddx[j], i.second+ddy[j])){
            int nr = shift.at<cv::Vec2i>(i.first+ddx[j], i.second+ddy[j])[0];
            int nc = shift.at<cv::Vec2i>(i.first+ddx[j], i.second+ddy[j])[1];
            if(nr+i.first >= 0 && nr+i.first < original.rows && nc+i.second >= 0 && nc+i.second < original.cols){
                int escNew = estimateQuality(original, mask, i.first, i.second, i.first + nr, i.second + nc);
                if (esc > escNew){
                    original.at<cv::Vec3b>(i.first, i.second) = original.at<cv::Vec3b>(nr+i.first,nc+i.second);
                    shift.at<cv::Vec2i>(i.first, i.second)[0] = nr;
                    shift.at<cv::Vec2i>(i.first, i.second)[0] = nc;
                }
            }
        }
    }
}

void search(cv::Mat& mask, cv::Mat& shift, cv::Mat& original, FastRandom& random){
    std::vector<std::pair<int,int>> masked;
    for(int i = 0; i < original.cols; ++i){
        for(int j = 0; j < original.rows; ++j){
            if(isPixelMasked(mask, j, i)){
                masked.emplace_back(std::make_pair(j,i));
            }
        }
    }
    for(int xd = 0; xd < 100; ++xd){
        for(auto& i : masked){
            refinement(mask, shift, original, i, random);
            propagation(mask, shift, original, i, random);
        }
    }
}

void run(int caseNumber, std::string caseName) {
    std::cout << "_________Case #" << caseNumber << ": " <<  caseName << "_________" << std::endl;

    cv::Mat original = cv::imread("lesson18/data/" + std::to_string(caseNumber) + "_" + caseName + "/" + std::to_string(caseNumber) + "_original.jpg");
    cv::Mat mask = cv::imread("lesson18/data/" + std::to_string(caseNumber) + "_" + caseName + "/" + std::to_string(caseNumber) + "_mask.png");
    rassert(!original.empty(), 324789374290018);
    rassert(!mask.empty(), 378957298420019);
    rassert(mask.rows == original.rows && mask.cols == original.cols, 896324873461343274);
    std::cout << "Image resolution: " << mask.rows << ' ' << mask.cols << std::endl;

    std::string resultsDir = "lesson18/resultsData/";
    if (!std::filesystem::exists(resultsDir)) { // если папка еще не создана
        std::filesystem::create_directory(resultsDir); // то создаем ее
    }
    resultsDir += std::to_string(caseNumber) + "_" + caseName + "/";
    if (!std::filesystem::exists(resultsDir)) { // если папка еще не создана
        std::filesystem::create_directory(resultsDir); // то создаем ее
    }

    cv::imwrite(resultsDir + "0original.png", original);
    cv::imwrite(resultsDir + "1mask.png", mask);

    int maskedPixelNumber = 0;
    for(int i = 0; i < original.cols; ++i){
        for(int j = 0; j < original.rows; ++j){
            if(isPixelMasked(mask, j, i)){
                original.at<cv::Vec3b>(j,i) = mask.at<cv::Vec3b>(j,i);
                maskedPixelNumber++;
            }
        }
    }
    std::cout << "Number of masked pixels: " << maskedPixelNumber << '/' << original.cols*original.rows << " = " << static_cast<double>(maskedPixelNumber)/static_cast<double>(original.cols*original.rows) * 100. << '%' << std::endl;
    FastRandom random(32542341); // этот объект поможет вам генерировать случайные гипотезы
    const int colsNum = original.cols;
    const int rowsNum = original.rows;
    cv::imwrite(resultsDir + "3randomShifting.png", original);
    std::vector<std::pair<cv::Mat, cv::Mat>> pyramid;
    std::vector<cv::Mat> shiftpyr;
    cv::Mat tempOrg = original.clone();
    cv::Mat tempMsk = mask.clone();
    for(int i = 0; i < 8; ++i){
        cv::Mat shift(tempOrg.rows, tempOrg.cols, CV_32SC2);
        pyramid.emplace_back(std::make_pair(tempOrg.clone(), tempMsk.clone()));
        std::cout << "layer: " << i << " " << tempOrg.cols << " " << tempOrg.rows << std::endl;
        shiftpyr.emplace_back(shift.clone());
        cv::pyrDown(tempOrg, tempOrg);
        cv::pyrDown(tempMsk, tempMsk);
    }
    for(int c = 0; c < pyramid.back().first.cols; ++c){
        for(int r = 0; r < pyramid.back().first.rows; ++r){
            if(isPixelMasked(pyramid.back().second, r, c)){
                int dc = random.next(2, colsNum-3) - c;
                int dr = random.next(2,rowsNum-3) - r;
                while(isPixelMasked(pyramid.back().second, dr + r, dc + c)){
                    dc = random.next(2, colsNum-3) - c;
                    dr = random.next(2,rowsNum-3) - r;
                }
                shiftpyr.back().at<cv::Vec2i>(r, c) = cv::Vec2i(dr, dc);
                pyramid.back().first.at<cv::Vec3b>(r, c) = pyramid.back().first.at<cv::Vec3b>(dr + r,dc + c);
            }
        }
    }
    //r, c -> (2r,2c); (2r+1, 2c); (2r + 2c+1); (2r + 1, 2c + 1);
    //r, c -> r/2 c/2
    for(int i = pyramid.size()-1; i >= 0; --i){
        search(pyramid[i].second, shiftpyr[i], pyramid[i].first, random);

        std::cout << "save image " << i << std::endl;
        cv::imwrite(resultsDir + "ture_orig" + std::to_string(i) + ".png", original);

        if(i > 0){
            for(int c = 0; c < pyramid[i-1].first.cols; ++c){
                for(int r = 0; r < pyramid[i-1].first.rows; ++r){
                    if(isPixelMasked(pyramid[i-1].second, r, c)){
                        pyramid[i-1].first.at<cv::Vec3b>(r,c) = pyramid[i].first.at<cv::Vec3b>(r/2, c/2);
                        shiftpyr[i-1].at<cv::Vec2i>(r,c) = shiftpyr[i-1].at<cv::Vec2i>(r/2,c/2);
                    }
                }
            }
        }
        //shiftnew.at<cv::Vec3b>(1,1) = original.at<cv::Vec3b>(1,1);
    }
    std::cout << "save final image"  << std::endl;
    cv::imwrite(resultsDir + "final.png", original);
}


int main() {
    try {
        run(1, "mic");
        // TODO протестируйте остальные случаи:
        //run(2, "flowers");
        //run(3, "baloons");
        //run(4, "brickwall");
        //run(5, "old_photo");
        //run(6, "your_data"); // TODO придумайте свой случай для тестирования (рекомендуется не очень большое разрешение, например 300х300)

        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception! " << e.what() << std::endl;
        return 1;
    }
}
