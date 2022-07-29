#include "../common/common.hpp"
#include "../common/RenderImage.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

void* handle = NULL;
RIFrameInfo depth = { 0 };
RIFrameInfo rgb = { 0 };

Mat frame, blur_frame, dil_frame, ero_frame, after_frame;
//预设四色的hsv范围值
vector<vector<int>> ColorValues{ {0, 18, 0, 179, 255, 62},      //black
                                                    {80, 43, 46, 124, 255, 255},    //blue
                                                    {0, 43, 46, 5, 255, 255},          //red
                                                    {35, 43, 46, 77, 255, 255} };   //green

//初始化+开启设备
void HIK_initialization()
{
    MV3D_RGBD_VERSION_INFO stVersion;
    ASSERT_OK(MV3D_RGBD_GetSDKVersion(&stVersion));
    LOGD("dll version: %d.%d.%d", stVersion.nMajor, stVersion.nMinor, stVersion.nRevision);

    ASSERT_OK(MV3D_RGBD_Initialize());

    unsigned int nDevNum = 0;
    ASSERT_OK(MV3D_RGBD_GetDeviceNumber(DeviceType_Ethernet | DeviceType_USB, &nDevNum));
    LOGD("MV3D_RGBD_GetDeviceNumber success! nDevNum:%d.", nDevNum);
    ASSERT(nDevNum);

    // 查找设备
    std::vector<MV3D_RGBD_DEVICE_INFO> devs(nDevNum);
    ASSERT_OK(MV3D_RGBD_GetDeviceList(DeviceType_Ethernet | DeviceType_USB, &devs[0], nDevNum, &nDevNum));
    for (unsigned int i = 0; i < nDevNum; i++)
    {
        LOG("Index[%d]. SerialNum[%s] IP[%s] name[%s].\r\n", i, devs[i].chSerialNumber, devs[i].SpecialInfo.stNetInfo.chCurrentIp, devs[i].chModelName);
    }

    //打开设备
    unsigned int nIndex = 0;
    ASSERT_OK(MV3D_RGBD_OpenDevice(&handle, &devs[nIndex]));
    LOGD("OpenDevice success.");

    // 开始工作流程
    ASSERT_OK(MV3D_RGBD_Start(handle));
    LOGD("Start work success.");
}

//关闭释放设备
void HIK_stop()
{
    ASSERT_OK(MV3D_RGBD_Stop(handle));
    ASSERT_OK(MV3D_RGBD_CloseDevice(&handle));
    ASSERT_OK(MV3D_RGBD_Release());

    LOGD("Main done!");
}

String coltypes(int type)
{
    String color_type;
    switch (type)
    {
    case 0:
        color_type = "black";
        break;
    case 1:
        color_type = "blue";
        break;
    case 2:
        color_type = "red";
        break;
    case 3:
        color_type = "green";
        break;
    }
    return color_type;
}

Point getcontours(Mat image)
{
    //轮廓点向量contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //矩形点向量和矩形向量
    vector<vector<Point>> conpoly(contours.size());
    vector<Rect> boundirect(contours.size());

    Point myPoint(0, 0);

    for (int i = 0; i < contours.size(); i++)
    {
        int area = contourArea(contours[i]);

        //控制检测像素面积大小
        if (area> 3000 && area < 10000)
        {
            //多边形曲线拟合
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conpoly[i], 0.03 * peri, true);

            LOGD("contours: area(%d);  conpoly[%d]: size(%d)", area, i, conpoly[i].size());

            boundirect[i] = boundingRect(conpoly[i]);

            //图框的左上角
            myPoint.x = boundirect[i].x;
            myPoint.y = boundirect[i].y;

            //图像绘图
            //drawContours(frame, conpoly, i, Scalar(255, 0, 255), 2);
            rectangle(after_frame, boundirect[i].tl(), boundirect[i].br(), Scalar(0, 0, 255), 2);  
            LOGD("draw finish!");
        }
    }

    return myPoint;
}

void getcolor(Mat ero_frame)
{
    Mat hsv_frame;
    cvtColor(ero_frame, hsv_frame, COLOR_RGB2HSV);

    for (int i = 0; i < ColorValues.size(); i++)
    {
        Mat mask;
        Scalar lower(ColorValues[i][0], ColorValues[i][1], ColorValues[i][2]);
        Scalar upper(ColorValues[i][3], ColorValues[i][4], ColorValues[i][5]);
        
        //二值化图像
        inRange(hsv_frame, lower, upper, mask);

        Point po = getcontours(mask);

        if (po.x != 0)
        {
            LOGD("findPoint: x(%d), y(%d)", po.x, po.y);

            String t_color = coltypes(i);
            putText(after_frame, t_color, {po.x -5, po.y }, FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
        }
    }
}

int main(int argc, char** argv)
{
    //初始化相机并打开
    HIK_initialization();

    MV3D_RGBD_FRAME_DATA stFrameData = { 0 };

    while (1)
    {
        // 获取图像数据
        int nRet = MV3D_RGBD_FetchFrame(handle, &stFrameData, 5000);
        if (MV3D_RGBD_OK == nRet)
        {
            //分析获取每帧数据
            parseFrame(&stFrameData, &depth, &rgb);

            LOGD("rgb: FrameNum(%d), height(%d), width(%d)。", rgb.nFrameNum, rgb.nHeight, rgb.nWidth);
            
            //转换两个相同图像，一个处理，一个绘图
            Mat rgb_frame(rgb.nHeight, rgb.nWidth, CV_8UC3, rgb.pData);
            cvtColor(rgb_frame, frame, COLOR_BGR2RGB);
            cvtColor(rgb_frame, after_frame, COLOR_BGR2RGB);

            //平滑处理，扩大再侵蚀
            GaussianBlur(frame, blur_frame, Size(7, 7), 5, 0);
            Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
            dilate(blur_frame, dil_frame, kernel);
            erode(dil_frame, ero_frame, kernel);

            //寻找小块颜色
            getcolor(ero_frame);

            namedWindow("color_detection", 0);
            resizeWindow("color_detection", 640, 360);
            imshow("color_detection", after_frame);

            waitKey(1);
        }
    }

    //相机关闭，线程释放
    HIK_stop();

   return  0;
}

