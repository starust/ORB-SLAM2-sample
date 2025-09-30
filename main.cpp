#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

int main() {
    // 图片路径
    vector<string> filenames = {
        "images/img1.jpg",
        "images/img2.jpg",
        "images/img3.jpg",
        "images/img4.jpg",
        "images/img5.jpg",
        "images/img6.jpg",
        "images/img7.jpg"
    };

    // ORB 特征提取器
    Ptr<ORB> orb = ORB::create(3000); // 提取 3000 个特征点

    // BFMatcher 用汉明距离
    BFMatcher matcher(NORM_HAMMING);

    // 顺序处理相邻图片对
    for (size_t i = 0; i + 1 < filenames.size(); i++) {
        Mat img1 = imread(filenames[i], IMREAD_GRAYSCALE);
        Mat img2 = imread(filenames[i + 1], IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            cout << "❌ 无法读取 " << filenames[i] << " 或 " << filenames[i+1] << endl;
            continue;
        }

        // 提取 ORB 特征点与描述子
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        orb->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
        orb->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

        cout << "✅ " << filenames[i] << " vs " << filenames[i+1]
             << " | keypoints1=" << keypoints1.size()
             << ", keypoints2=" << keypoints2.size() << endl;

        // knnMatch 匹配
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

        // ratio test 过滤
        const float ratio_thresh = 0.75f;
        vector<DMatch> good_matches;
        for (size_t k = 0; k < knn_matches.size(); k++) {
            if (knn_matches[k][0].distance < ratio_thresh * knn_matches[k][1].distance) {
                good_matches.push_back(knn_matches[k][0]);
            }
        }

        cout << "   匹配数量: " << good_matches.size() << endl;

        // 绘制匹配结果
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                    Scalar(0, 255, 0), Scalar(255, 0, 0),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        imshow("ORB Matches", img_matches);

        // 按任意键切换到下一对，按 ESC 退出
        int key = waitKey(0);
        if (key == 27) break; // ESC 退出
    }

    return 0;
}
