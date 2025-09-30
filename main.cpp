#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    // 1. 读入两张图
    Mat img1 = imread("images/img1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("images/img2.jpg", IMREAD_GRAYSCALE);
    if(img1.empty() || img2.empty()) {
        cout << "无法读取图片" << endl;
        return -1;
    }

    // 2. 提取 ORB 特征
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

    // 3. 特征匹配 (用 Hamming 距离)
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 4. 画出匹配结果
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("ORB Matches", img_matches);
    waitKey(0);

    return 0;
}