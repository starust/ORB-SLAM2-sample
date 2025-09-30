#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

void saveKeypoints(const vector<KeyPoint>& kps, int idx) {
    ofstream f("results/keypoints_" + to_string(idx) + ".csv");
    f << "id,u,v\n";
    for (size_t i = 0; i < kps.size(); i++) {
        f << i << "," << kps[i].pt.x << "," << kps[i].pt.y << "\n";
    }
}

void saveMatches(const vector<KeyPoint>& k1, const vector<KeyPoint>& k2,
                 const vector<DMatch>& matches, int i, int j) {
    ofstream f("results/matches_" + to_string(i) + "_" + to_string(j) + ".csv");
    f << "id_i,u_i,v_i,id_j,u_j,v_j,distance\n";
    for (const auto& m : matches) {
        const auto& p1 = k1[m.queryIdx].pt;
        const auto& p2 = k2[m.trainIdx].pt;
        f << m.queryIdx << "," << p1.x << "," << p1.y << ","
          << m.trainIdx << "," << p2.x << "," << p2.y << ","
          << m.distance << "\n";
    }
}

int main() {
    vector<string> filenames = {
        "images/img1.jpg", "images/img2.jpg", "images/img3.jpg",
        "images/img4.jpg", "images/img5.jpg", "images/img6.jpg",
        "images/img7.jpg"
    };

    Ptr<ORB> orb = ORB::create(3000);
    BFMatcher matcher(NORM_HAMMING);

    // 确保 results 文件夹存在
    system("mkdir results");

    vector<vector<KeyPoint>> all_keypoints;
    vector<Mat> all_desc;

    // 提取每张图的特征点和描述子
    for (size_t i = 0; i < filenames.size(); i++) {
        Mat img = imread(filenames[i], IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "无法读取图像 " << filenames[i] << endl;
            return -1;
        }
        vector<KeyPoint> kps;
        Mat desc;
        orb->detectAndCompute(img, Mat(), kps, desc);
        all_keypoints.push_back(kps);
        all_desc.push_back(desc);

        // 保存关键点
        saveKeypoints(kps, i+1);
        cout << "保存 keypoints_" << i+1 << ".csv" << endl;
    }

    // 相邻帧匹配
    for (size_t i = 0; i < filenames.size()-1; i++) {
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(all_desc[i], all_desc[i+1], knn_matches, 2);

        vector<DMatch> good_matches;
        for (auto& m : knn_matches) {
            if (m[0].distance < 0.75 * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }

        saveMatches(all_keypoints[i], all_keypoints[i+1], good_matches, i+1, i+2);
        cout << "保存 matches_" << i+1 << "_" << i+2 << ".csv" << endl;
    }

    return 0;
}
