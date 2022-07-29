#include "../common/common.hpp"
#include "../common/RenderImage.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

void* handle = NULL;
RIFrameInfo depth = { 0 };
RIFrameInfo rgb = { 0 };
CascadeClassifier faceCascade;

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

    //改变分辨率参数，0x00010001为 1280×720， 0x00020002为 640×360
    //MV3D_RGBD_PARAM stparam;
    //stparam.enParamType = ParamType_Enum;
    //stparam.ParamInfo.stEnumParam.nCurValue = 0x00010001;
    //ASSERT_OK(MV3D_RGBD_SetParam(handle, MV3D_RGBD_ENUM_RESOLUTION, &stparam));

      //改变曝光参数
      //MV3D_RGBD_PARAM stparam;
      //stparam.enParamType = ParamType_Float;;
      //stparam.ParamInfo.stFloatParam.fCurValue = 100.0000;
      //ASSERT_OK(MV3D_RGBD_SetParam(handle, MV3D_RGBD_FLOAT_EXPOSURETIME, &stparam));
      //LOGD("EXPOSURETIME: (%f)", stparam.ParamInfo.stFloatParam.fCurValue);

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

int main(int argc, char** argv)
{

    HIK_initialization();

    MV3D_RGBD_FRAME_DATA stFrameData = { 0 };

    while (1)
    {
        // 获取图像数据
        int nRet = MV3D_RGBD_FetchFrame(handle, &stFrameData, 5000);
        if (MV3D_RGBD_OK == nRet)
        {
            LOGD("MV3D_RGBD_FetchFrame success.");

            //分析获取每帧数据
            parseFrame(&stFrameData, &depth, &rgb);
            Mat rgb_frame(rgb.nHeight, rgb.nWidth, CV_8UC3, rgb.pData);

            LOGD("rgb: FrameNum(%d), height(%d), width(%d)。", rgb.nFrameNum, rgb.nHeight, rgb.nWidth);
            
            //B、R通道交换，显示正常彩色图像
            Mat frame;
            cvtColor(rgb_frame, frame, COLOR_RGB2BGR);

            //以下进行人脸识别
            faceCascade.load("haarcascade_frontalface_default.xml");

            if (faceCascade.empty())
            {
                LOGD("xml not found!");
            }

            vector<Rect> faces;
            faceCascade.detectMultiScale(frame, faces, 1.1, 10);

            for (int i = 0; i < faces.size(); i++)
            {
                LOGD("catch_face: x(%d), y(%d)", faces[i].x, faces[i].y);
                rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(255, 255, 255), 4);
            }

            namedWindow("HIK_face", 0);
            resizeWindow("HIK_face", 640, 360);
            imshow("HIK_face", frame);
            waitKey(1);
        }
    }

    HIK_stop();

    return  0;
}

