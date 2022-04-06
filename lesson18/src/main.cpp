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
double estimateQuality(cv::Mat &image, int r1, int c1, int r2, int c2){
    std::vector<int> dx = {-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2};
    std::vector<int> dy = {2,2,2,2,2,1,1,1,1,1,0,0,0,0,0,-1,-1,-1,-1,-1,-2,-2,-2,-2,-2};
    const int colsNum = image.cols;
    const int rowsNum = image.rows;
    double quol = 0;
    for(int i = 0; i < dx.size();++i){
        if(r1+dx[i] >= 0 && r1+dx[i] < rowsNum && r2+dx[i] >= 0 && r2+dx[i] < rowsNum && c1+dy[i] >= 0 && c1+dy[i] < colsNum && c2+dy[i] >= 0 && c2+dy[i] < colsNum){
            double d1 = abs(image.at<cv::Vec3b>(r1+dx[i],c1+dy[i])[1] - image.at<cv::Vec3b>(r2+dx[i],c2+dy[i])[1]);
            double d2 = abs(image.at<cv::Vec3b>(r1+dx[i],c1+dy[i])[2] - image.at<cv::Vec3b>(r2+dx[i],c2+dy[i])[2]);
            double d0 = abs(image.at<cv::Vec3b>(r1+dx[i],c1+dy[i])[0] - image.at<cv::Vec3b>(r2+dx[i],c2+dy[i])[0]);
            quol += abs(d0+d1+d2);
        }
    }
    return quol;
}

void run(int caseNumber, std::string caseName) {
    std::cout << "_________Case #" << caseNumber << ": " <<  caseName << "_________" << std::endl;

    cv::Mat original = cv::imread("lesson18/data/" + std::to_string(caseNumber) + "_" + caseName + "/" + std::to_string(caseNumber) + "_original.jpg");
    cv::Mat mask = cv::imread("lesson18/data/" + std::to_string(caseNumber) + "_" + caseName + "/" + std::to_string(caseNumber) + "_mask.png");
    rassert(!original.empty(), 324789374290018);
    rassert(!mask.empty(), 378957298420019);

    // TODO напишите rassert сверяющий разрешение картинки и маски
    // TODO выведите в консоль это разрешение картинки
    rassert(mask.rows == original.rows && mask.cols == original.cols, 896324873461343274);
    std::cout << "Image resolution: " << mask.rows << ' ' << mask.cols << std::endl;

    // создаем папку в которую будем сохранять результаты - lesson18/resultsData/ИМЯ_НАБОРА/
    std::string resultsDir = "lesson18/resultsData/";
    if (!std::filesystem::exists(resultsDir)) { // если папка еще не создана
        std::filesystem::create_directory(resultsDir); // то создаем ее
    }
    resultsDir += std::to_string(caseNumber) + "_" + caseName + "/";
    if (!std::filesystem::exists(resultsDir)) { // если папка еще не создана
        std::filesystem::create_directory(resultsDir); // то создаем ее
    }

    // сохраняем в папку с результатами оригинальную картинку и маску
    cv::imwrite(resultsDir + "0original.png", original);
    cv::imwrite(resultsDir + "1mask.png", mask);

    // TODO замените белым цветом все пиксели в оригинальной картинке которые покрыты маской
    // TODO сохраните в папку с результатами то что получилось под названием "2_original_cleaned.png"
    // TODO посчитайте и выведите число отмаскированных пикселей (числом и в процентах) - в таком формате:
    // Number of masked pixels: 7899/544850 = 1%
    int maskedPixelNumber = 0;
    std::vector<std::pair<int,int>> masked;
    for(int i = 0; i < original.cols; ++i){
        for(int j = 0; j < original.rows; ++j){
            if(isPixelMasked(mask, j, i)){
                original.at<cv::Vec3b>(j,i) = mask.at<cv::Vec3b>(j,i);
                maskedPixelNumber++;
                masked.emplace_back(std::make_pair(j,i));
            }
        }
    }
    std::cout << "Number of masked pixels: " << maskedPixelNumber << '/' << original.cols*original.rows << " = " << static_cast<double>(maskedPixelNumber)/static_cast<double>(original.cols*original.rows) * 100. << '%';
    FastRandom random(32542341); // этот объект поможет вам генерировать случайные гипотезы
    const int colsNum = original.cols;
    const int rowsNum = original.rows;
    // TODO 10 создайте картинку хранящую относительные смещения - откуда брать донора для заплатки, см. подсказки про то как с нею работать на сайте
    cv::Mat shift(mask.rows, mask.cols, CV_32SC2);
    // TODO 11 во всех отмаскированных пикселях: заполните эту картинку с относительными смещениями - случайными смещениями (но чтобы они и их окрестность 5х5 не выходила за пределы картинки)
    // TODO 12 во всех отмаскированных пикселях: замените цвет пиксела А на цвет пикселя Б на который указывает относительное смещение пикселя А
    for(auto& i : masked){
        int dc = random.next(2, colsNum-3) - i.second;
        int dr = random.next(2,rowsNum-3) - i.first;
        while(isPixelMasked(mask, dr + i.first, dc + i.second)){
            dc = random.next(2, colsNum-3) - i.second;
            dr = random.next(2,rowsNum-3) - i.first;
        }
        shift.at<cv::Vec2i>(i.first, i.second) = cv::Vec2i(dr, dc);
        original.at<cv::Vec3b>(i.first, i.second) = original.at<cv::Vec3b>(dr + i.first,dc + i.second);
    }
    // TODO 13 сохраните получившуюся картинку на диск
    cv::imwrite(resultsDir + "3randomShifting.png", original);
    // TODO 14 выполняйте эти шаги 11-13 много раз, например 1000 раз (оберните просто в цикл, сохраняйте картинку на диск только на каждой десятой или сотой итерации)
    // TODO 15 теперь давайте заменять значение относительного смещения на новой только если новая случайная гипотеза - лучше старой, добавьте оценку "насколько смещенный патч 5х5 похож на патч вокруг пикселя если их наложить"

    for(int xd = 0; xd < 500; ++xd){
        for(auto& i : masked){
            int dc = random.next(2, colsNum-3) - i.second;
            int dr = random.next(2,rowsNum-3) - i.first;
            while(isPixelMasked(mask, dr + i.first, dc + i.second)){
                dc = random.next(2, colsNum-3) - i.second;
                dr = random.next(2,rowsNum-3) - i.first;
            }
            int r = shift.at<cv::Vec2i>(i.first, i.second)[0];
            int c = shift.at<cv::Vec2i>(i.first, i.second)[1];
            int escNew = estimateQuality(original, i.first, i.second, i.first + dr, i.second + dc);
            int esc = estimateQuality(original, i.first, i.second, i.first + r, i.second + c);
            if(escNew < esc){
                original.at<cv::Vec3b>(i.first, i.second) = original.at<cv::Vec3b>(dr+i.first,dc+i.second);
                shift.at<cv::Vec2i>(i.first, i.second)[0] = dr;
                shift.at<cv::Vec2i>(i.first, i.second)[0] = dc;
            }
        }
        if(xd % 100 == 0){
            std::cout << "save image " << xd / 100 + 4 << std::endl;
            cv::imwrite(resultsDir + "ture_orig" + std::to_string(xd / 100 + 4) + ".png", original);
        }
    }


    // Ориентировочный псевдокод-подсказка получившегося алгоритма:
    // cv::Mat shifts(...); // матрица хранящая смещения, изначально заполнена парами нулей
    // cv::Mat image = original; // текущая картинка
    // for (100 раз) {
    //     for (пробегаем по всем пикселям j,i) {
    //         if (если этот пиксель не отмаскирован)
    //             continue; // пропускаем т.к. его менять не надо
    //         cv::Vec2i dxy = смотрим какое сейчас смещение для этого пикселя в матрице смещения
    //         int (nx, ny) = (i + dxy.x, j + dxy.y); // ЭТО НЕ КОРРЕКТНЫЙ КОД, но он иллюстрирует как рассчитать координаты пикселя-донора из которого мы хотим брать цвет
    //         currentQuality = estimateQuality(image, j, i, ny, nx, 5, 5); // эта функция (создайте ее) считает насколько похож квадрат 5х5 приложенный центром к (i, j)
    //                                                                                                                        на квадрат 5х5 приложенный центром к (nx, ny)
    //
    //         int (rx, ry) = random.... // создаем случайное смещение относительно нашего пикселя, воспользуйтесь функцией random.next(...);
    //                                      (окрестность вокруг пикселя на который укажет смещение - не должна выходить за пределы картинки и не должна быть отмаскирована)
    //         randomQuality = estimateQuality(image, j, i, j+ry, i+rx, 5, 5); // оцениваем насколько похоже будет если мы приложим эту случайную гипотезу которую только что выбрали
    //
    //         if (если новое качество случайной угадайки оказалось лучше старого) {
    //             то сохраняем (rx,ry) в картинку смещений
    //             и в текущем пикселе кладем цвет из пикселя на которого только что смотрели (цент окрестности по смещению)
    //             (т.е. мы не весь патч сюда кладем, а только его центральный пиксель)
    //         } else {
    //             а что делать если новая случайная гипотеза хуже чем то что у нас уже есть?
    //         }
    //     }
    //     не забываем сохранить на диск текущую картинку
    //     а как численно оценить насколько уже хорошую картинку мы смогли построить? выведите в консоль это число
    // }
}


int main() {
    try {
        //run(1, "mic");
        // TODO протестируйте остальные случаи:
        run(2, "flowers");
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
