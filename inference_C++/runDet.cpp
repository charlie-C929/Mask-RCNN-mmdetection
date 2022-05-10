#include <opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgproc/imgproc_c.h>
#include <opencv4/opencv2/dnn.hpp>
#include <iostream>  
#include <onnxruntime_cxx_api.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include "roi_align.h"
#include "roi_align.cpp"
#include "grid_sample.h"
#include "gridSample.cpp"
#include "argparse.hpp"


using namespace cv;     //当定义这一行后，cv::imread可以直接写成imread
using namespace std;
using namespace Ort;
using namespace cv::dnn;

void vector2Mat(vector< vector<float> > src,Mat & dst,int type);
// Create a new, _empty_ cv::Mat with the row size of OrigSamples
void vector2Mat(vector< vector<float> > src,Mat & dst,int type)
{
    Mat temp(src.size(),src.at(0).size(),type);
    for(int i=0; i<temp.rows; ++i)
        for(int j=0; j<temp.cols; ++j)
            temp.at<float>(i, j) = src.at(i).at(j);
    temp.copyTo(dst);
}

// 图像处理  标准化处理
void PreProcess(const Mat& image, Mat& image_blob)
{
	Mat input;
	image.copyTo(input);

    
	//数据处理 标准化
	std::vector<Mat> channels, channel_p;
	split(input, channels);
	Mat R, G, B;
	B = channels.at(0);
	G = channels.at(1);
	R = channels.at(2);

	B = (B / 255. - 0.5) / 0.5;
	G = (G / 255. - 0.5) / 0.5;
	R = (R / 255. - 0.5) / 0.5;

	channel_p.push_back(R);
	channel_p.push_back(G);
	channel_p.push_back(B);
    

	Mat outt;
	merge(channel_p, outt);
	image_blob = outt;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("program_name");

    program.add_argument("pic")
        .help("The picture you want to inference");
    program.add_argument("--model")
        .required()
        .help("The model path you want to use to inference");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    const char* model_path = program.get<std::string>("--model").data();
    std::cout << model_path << endl;

    auto pic_path = program.get<std::string>("pic");
    
   //environment （设置为VERBOSE（ORT_LOGGING_LEVEL_VERBOSE）时，方便控制台输出时看到是使用了cpu还是gpu执行）
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
	Ort::SessionOptions session_options;
	// 使用1个线程执行op,若想提升速度，增加线程数
	session_options.SetIntraOpNumThreads(1);
	//CUDA加速开启(由于onnxruntime的版本太高，无cuda_provider_factory.h的头文件，加速可以使用onnxruntime V1.8的版本)
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	// ORT_ENABLE_ALL: 启用所有可能的优化
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //mmdetection
    MMCVRoiAlignCustomOp custom_op;
    GridSampleOp custom_op_2;

     // 创建定制算子域（CustomOpDomain）
    Ort::CustomOpDomain custom_op_domain("mmcv");
  // 在定制算子域中添加定制算子
    custom_op_domain.Add(&custom_op);
    custom_op_domain.Add(&custom_op_2);
    session_options.Add(custom_op_domain);

    printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_options);
	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;


	//model info
	// 获得模型又多少个输入和输出，一般是指对应网络层的数目
	// 一般输入只有图像的话input_nodes为1
	size_t num_input_nodes = session.GetInputCount();
	// 如果是多输出网络，就会是对应输出的数目
	size_t num_output_nodes = session.GetOutputCount();
	printf("Number of inputs = %zu\n", num_input_nodes);
	printf("Number of output = %zu\n", num_output_nodes);
	//获取输入name
	const char* input_name = session.GetInputName(0, allocator);               
	std::cout << "input_name:" << input_name << std::endl;
	//获取输出name
	const char* output_name = session.GetOutputName(2, allocator);
	std::cout << "output_name: " << output_name << std::endl;

    // 自动获取维度数量
	auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "input_dims:" << input_dims.size() << std::endl;
	std::cout << "output_dims:" << output_dims[0] << std::endl;

    
    std::vector<const char*> input_names{ input_name };
	std::vector<const char*> output_names = { output_name };
	std::vector<const char*> input_node_names = { "input" };
	std::vector<const char*> output_node_names = { "dets","labels","masks"};
	

    
	//加载图片
	Mat img = imread(pic_path);
    int64_t H = img.rows;
    int64_t W = img.cols;
    std::cout << img.size() << endl;
	Mat det1;
	//resize(img, det1, Size(256, 256), INTER_AREA);
	img.convertTo(img, CV_32FC3);
	PreProcess(img, det1);         //标准化处理
    std::cout << det1.size() << endl;
    
	Mat blob = dnn::blobFromImage(det1);
    // std::cout << blob.size() << endl;
    // cout<<"【Numpy风格】\n"<<format(det1,cv::Formatter::FMT_NUMPY)<<endl;
	printf("Load success!\n");

    //overwrite input dims
    input_dims[0] = 1;
    input_dims[1] = 3;
    input_dims[2] = H;
    input_dims[3] = W;

	clock_t startTime, endTime;
	//创建输入tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	std::vector<Ort::Value> input_tensors;
	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
	/*cout << int(input_dims.size()) << endl;*/

    startTime = clock();

	//推理(score model & input tensor, get back output tensor)
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_names.size(), output_node_names.data(), output_node_names.size());
	endTime = clock();
    
    std::cout << "The run time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
	printf("Done!\n");

    float* dets = output_tensors[0].GetTensorMutableData<float>();
    float* labels = output_tensors[1].GetTensorMutableData<float>();
    float* masks = output_tensors[2].GetTensorMutableData<float>();

    auto shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
    //int N = GetObjectNumber(dets);
    std::cout << shape[0] << shape[1] << shape[2] <<shape[3] << endl;
    

    vector<int> index; 
    //bboxes
    //TODO:
    vector<vector<float>> detecs;
    for(int i = 0; i < 100; ++i)
    {
        vector<float> detec;
        if(dets[4+5*i] > 0.5)
        {
            //record the index we need.
            index.push_back(i);
            //labels.push_back(labels[i]);
            for(int j = i*5;j <= 4+5*i; ++j)
            {
                detec.push_back(dets[j]);
            }
            detecs.push_back(detec);
        }
    }

    //masks
    vector<vector<vector<float>>> maskes;
    for(int i = 0;i <index.size();++i)
    {
        float * inde = &masks[index[i]*H*W];

        vector<vector<float>> mask;
        for (size_t j = 0; j < H; j++)
        {
            vector<float> mas;
            for (size_t k = 0; k < W; k++)
            {
                mas.push_back(inde[W*j+k]*255);
                //std::cout << inde[W*i+k]*255 << endl;
            }
            mask.push_back(mas);    
        }
        maskes.push_back(mask);
    }

    img.convertTo(img, CV_8UC3);
    //temp
    std::cout << detecs.size()<< endl;
    for(int i = 0;i <detecs.size();++i)
    {
        for (size_t j = 0; j < maskes[0].size(); j++)
        {
            for (size_t k = 0; k < maskes[0][0].size(); k++)
        {
            //TODO:colour
            if(maskes[i][j][k] != 0)
            {
                img.at<Vec3b>(j,k)[0]=0;
                img.at<Vec3b>(j,k)[2]=0;
            }
        }
        }
        cv::Point p3(detecs[i][0],detecs[i][1]), p4(detecs[i][2],detecs[i][3]);
	    cv::Scalar colorRectangle1(0, 255, 0);
	    int thicknessRectangle1 = 2;
        cv::rectangle(img,p3, p4, colorRectangle1, thicknessRectangle1);

        //contour
        Mat final_mask;
        vector2Mat(maskes[i],final_mask,4);
        final_mask.convertTo(final_mask,CV_8UC1);
        vector<vector<cv::Point>> contours;
		vector<cv::Vec4i> hierarchy;
		cv::findContours(final_mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
		cv::drawContours(img, contours, -1,cv::Scalar(0,0,255),2);
        
    }


    // Mat img_test;
    // vector2Mat(maskes[1],img_test,4);
    //img_test.convertTo(img_test, CV_8UC3);
    cv::imshow("",img);
    // cv::imshow("",img);
    cv::waitKey(0);

    return 0;





}
