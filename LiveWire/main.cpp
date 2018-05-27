#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <cstdio>
#include <queue>
using namespace cv;

const double inf = 0x3f3f3f3f3f;

Mat img, img_gray, img_value, fz, fg, I, pre, cost, vis, G, img_draw, in_que, skip, img_canny;
bool g_flag;

Point start;

void Mat_init(){
    I.create(img.rows, img.cols, CV_64FC2);
    G.create(img.rows, img.cols, CV_64FC1);
    fg.create(img.rows, img.cols, CV_64FC1);
    pre.create(img.rows, img.cols, CV_32SC2);
    cost.create(img.rows, img.cols, CV_64FC1);
    vis.create(img.rows, img.cols, CV_8UC1);
    fz.create(img.rows, img.cols, CV_64FC1);
    in_que.create(img.rows, img.cols, CV_8UC1);
    skip.create(img.rows, img.cols, CV_8UC1);
}

double getcost(Point p, Point q, bool flag){
    double Ixp = I.at<Vec2d>(p.y, p.x)[0];
    double Iyp = I.at<Vec2d>(p.y, p.x)[1];
    double Ixq = I.at<Vec2d>(q.y, q.x)[0];
    double Iyq = I.at<Vec2d>(q.y, q.x)[1];
    double fZ = 0.0, fG = 0.0, fD = 0.0;
    fZ = fz.at<uchar>(q.y, q.x);
    fG = fg.at<double>(q.y, q.x);
    double judge = 0;
    judge = Iyp * (q.x - p.x) + (-Ixp) * (q.y - p.y);
    double dp = 0.0, dq = 0.0;
    // q - p
    if (judge >= 0){
        dp = Iyp * (q.x - p.x) + (-Ixp) * (q.y - p.y);
        dq = (q.x - p.x) * Iyq + (q.y - p.y) * (-Ixq);
    }
    // p - q
    else{
        dp = Iyp * (p.x - q.x) + (-Ixp) * (p.y - q.y);
        dq = (p.x - q.x) * Iyq + (p.y - q.y) * (-Ixq);
    }
    if (!flag){
        dp /= sqrt(2);
        dq /= sqrt(2);
    }
    double pi = acos(-1.0);
    fD = (acos(dp) + acos(dq)) / pi;
    if (flag)
        fG /= sqrt(2); // vertical or horizontal neighbor
    double ans = 0.43 * fZ + 0.43 * fD + 0.14 * fG;
    return ans;
}

struct P{
    double first;
    Point second;
    P(){}
    P(double a,Point b){
        first = a;
        second = b;
    }
    bool operator > (const P &b) const{
        return first > b.first;
    }
    bool operator < (const P &b) const{
        return first < b.first;
    }
};

void find_min_path(Point start){
    int dir_x[] = {1, 0, -1, 0, 1, 1, -1, -1};
    int dir_y[] = {0, 1, 0, -1, 1, -1, 1, -1};
    vis.setTo(0);
    cost.setTo(inf);
    in_que.setTo(0);
    skip.setTo(0);
    cost.at<double>(start.y, start.x) = 0;
    in_que.at<uchar>(start.y, start.x) = 1;
    std::priority_queue < P, std::vector<P>, std::greater<P> > que;
    que.push(P(0, start));
    while (!que.empty()){
        P cur = que.top();
        que.pop();
        Point p = cur.second;
        in_que.at<uchar>(p.y, p.x) = 0;
        if (skip.at<uchar>(p.y, p.x) == 1)
            continue;
        vis.at<uchar>(p.y, p.x) = 1;
        for (int i = 0; i < 8; i++){
            int tx = p.x + dir_x[i];
            int ty = p.y + dir_y[i];
            if (tx < 0 || tx >= img.cols || ty < 0 || ty >= img.rows)
                continue;
            if (vis.at<uchar>(ty, tx) == 1)
                continue;
            Point q = Point(tx, ty);
            double tmp;
            if (i <= 3){
                tmp = cost.at<double>(p.y, p.x) + getcost(p, q, true);
            }
            else{
                tmp = cost.at<double>(p.y, p.x) + getcost(p, q, false);
            }
            if (in_que.at<uchar>(q.y, q.x) == 1 && tmp < cost.at<double>(q.y, q.x)){
                skip.at<uchar>(q.y, q.x) = 1;
                continue;
            }
            
            if (in_que.at<uchar>(q.y, q.x) == 0){
                cost.at<double>(q.y, q.x) = tmp;
                pre.at<Vec2i>(q.y, q.x)[0] = p.x;
                pre.at<Vec2i>(q.y, q.x)[1] = p.y;
                in_que.at<uchar>(q.y, q.x) = 1;
                que.push(P(cost.at<double>(q.y, q.x), q));
            }
        }
    }
}

void onMouse(int event, int x, int y, int flags, void *param){
    if (event == EVENT_LBUTTONDOWN){
        start = Point(x, y);
        g_flag = true;
        find_min_path(start);
        img.copyTo(img_draw);
        imshow("example", img_draw);
    }
    else if (event == EVENT_MOUSEMOVE && g_flag){
        img.copyTo(img_draw);
        Point cur = Point(x, y);
        Point tmp;
        while (cur != start){
            tmp = Point(pre.at<Vec2i>(cur.y, cur.x)[0], pre.at<Vec2i>(cur.y, cur.x)[1]);
            line(img_draw, cur, tmp, Scalar(0, 255, 0), 2);
            if (tmp == start) break;
            cur = tmp;
        }
        imshow("example", img_draw);
    }
    else if (event == EVENT_LBUTTONUP){
        g_flag = false;
        img.copyTo(img_draw);
        imshow("example", img_draw);
    }
}

int main(){
    std::string filepath = "/Users/hxh/Desktop/Girl.bmp";
    namedWindow("example");
    img = imread(filepath);
    Mat_init();
    cvtColor(img, img_gray, CV_BGR2GRAY);
    img_gray.copyTo(img_value);
    GaussianBlur(img_value, img_value, Size(3, 3), 0, 0, BORDER_DEFAULT);
    Canny(img_gray, img_canny, 50, 100);
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            if (img_canny.at<uchar>(i, j) == 255) fz.at<uchar>(i, j) = 0;
            else    fz.at<uchar>(i, j) = 1;
        }
    }
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols - 1; j++){
            I.at<Vec2d>(i, j)[0] = (img_value.at<uchar>(i, j + 1) - img_value.at<uchar>(i, j)) / 255.0;
        }
        I.at<Vec2d>(i, img.cols - 1)[0] = I.at<Vec2d>(i, img.cols - 2)[0];
    }
    for (int j = 0; j < img.cols; j++){
        for (int i = 0; i < img.rows - 1; i++){
            I.at<Vec2d>(i, j)[1] = (img_value.at<uchar>(i + 1, j) - img_value.at<uchar>(i, j)) / 255.0;
        }
        I.at<Vec2d>(img.rows - 1, j)[1] = I.at<Vec2d>(img.rows - 2, j)[1];
    }
    double max_G = 0.0;
    for (int i = 0; i < I.rows; i++){
        for (int j = 0; j < I.cols; j++){
            G.at<double>(i, j) = sqrt(I.at<Vec2d>(i, j)[0] * I.at<Vec2d>(i, j)[0] + I.at<Vec2d>(i, j)[1] * I.at<Vec2d>(i, j)[1]);
            max_G = max(max_G, G.at<double>(i, j));
        }
    }
    for (int i = 0; i < fg.rows; i++){
        for (int j = 0; j < fg.cols; j++){
            fg.at<double>(i, j) = 1.0 - G.at<double>(i, j) / max_G;
        }
    }
    std::cout << "calculate of fz and fg completed!" << std::endl;
    setMouseCallback("example", onMouse, 0);
    imshow("example", img);
    waitKey(0);
}





