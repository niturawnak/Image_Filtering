#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/quality/qualitymse.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string> 
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;






void CreateGaussianMask(const int& window_size, cv::Mat& mask, int& sum_mask, const double& sigma){
	
	cv::Size mask_size(window_size, window_size);
	mask = cv::Mat(mask_size, CV_8UC1);

	const double hw = window_size /2;
	//const double sigma = std::sqrt(2.0) * hw / 2.5; 
	const double sigmaSq = sigma * sigma ; 

	//rmax = 2.5 * sigma
	//sigma = rmax / 2.5

	for (int r = 0; r < window_size; ++r) {
		for (int c = 0; c < window_size; ++c) {
			//mask.at<uchar>(r, c) = 1; //box filter

			//TODO: implement Gaussian filter
			double r2 = (r - hw) * (r - hw) + (c - hw) * (c - hw); //distance squared from centre of the mask
			mask.at<uchar>(r, c) = 255 * std::exp(-r2 / (2*sigmaSq)); //0..1 -> 0..255
			std::cout << static_cast<int>(mask.at<uchar>(r,c)) << std::endl;	
		}
		std::cout << std::endl;
	}
	std::cout << "Gaussian Mask:" << std::endl;
	std::cout << mask << std::endl;

	for (int r = 0; r < window_size; ++r) {
		for (int c = 0; c < window_size; ++c) {
			sum_mask += static_cast<int>(mask.at<uchar>(r, c));
		}
	}

}



void BilateralFilter(const cv::Mat& input, cv::Mat& output, int window_size, double sigma_spatial, double sigma_range) {

	const auto width = input.cols;
	const auto height = input.rows;
	const double hw = window_size /2;
	//const double sigma = std::sqrt(2.0) * hw / 2.5; 
	const double sigma_s = sigma_spatial * hw / 2.5; 



	// TEMPORARY CODE
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	cv::Mat mask;
	int sum_mask = 0; //in order to normalize filtering

	CreateGaussianMask(window_size, mask, sum_mask, sigma_s);

	const float sigmaRange = sigma_range; //sigma big == more smoothing == wide gaussian curve == closer to gaussian filter//20
	const float sigmaRangeSq = sigmaRange * sigmaRange;

	float range_mask[256];
	//compute range kernel
	for(int diff = 0; diff < 256; ++diff){
		range_mask[diff] = std::exp(- diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			//TODO: get center intensity
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {

					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
					//compute range difference to centre pixel value
					int diff = std::abs(intensity_center - intensity);
					//compute the range kernel's value ...computing outside to save computation
					float weight_range = range_mask[diff];

					int weight_spatial = static_cast<int>(mask.at<uchar>(i + window_size/2, j + window_size/2));
					//....combine weights....
					float weight = weight_range * weight_spatial;
					sum += intensity * weight; //convolution happening ...
					sum_Bilateral_mask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_Bilateral_mask; //normalization

			

		}
	}
}



void JointBilateralFilter(const cv::Mat& input_RGB, const cv::Mat& input_depth, cv::Mat& output, int window_size, double sigma_spatial, double sigma_range) {

	const auto width = input_RGB.cols;
	const auto height = input_RGB.rows;

	const double hw = window_size /2;
	const double sigma_s = sigma_spatial * hw / 2.5; 

	cv::Mat mask;
	int sum_mask = 0; //in order to normalize filtering

	CreateGaussianMask(window_size, mask, sum_mask, sigma_s);

	const float sigmaRange = sigma_range; //sigma big == more smoothing == wide gaussian curve == closer to gaussian filter
	const float sigmaRangeSq = sigmaRange * sigmaRange;

	float range_mask[256];
	//compute range kernel
	for(int diff = 0; diff < 256; ++diff){
		range_mask[diff] = std::exp(- diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			//TODO: get center intensity
			int intensity_center = static_cast<int>(input_RGB.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {

					int intensity = static_cast<int>(input_RGB.at<uchar>(r + i, c + j));
					//compute range difference to centre pixel value
					int diff = std::abs(intensity_center - intensity);
					//compute the range kernel's value ...computing outside to save computation
					float weight_range = range_mask[diff];

					int weight_spatial = static_cast<int>(mask.at<uchar>(i + window_size/2, j + window_size/2));
					//....combine weights....
					float weight = weight_range * weight_spatial;
					sum += input_depth.at<uchar>(r + i, c + j) * weight; //convolution happening ...
					sum_Bilateral_mask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_Bilateral_mask; //normalization

		}
	}
}
void JointBilateralUpsampling(const cv::Mat& input_rgb, const cv::Mat& input_depth, cv::Mat& output, int window_size, double sigma_spatial, double sigma_range) {

	cv::Mat D = input_depth.clone(); 
	cv::Mat I = input_rgb.clone(); 
	cv::resize(D, D, input_rgb.size()); 
	JointBilateralFilter(input_rgb, D, output, window_size, sigma_spatial, sigma_range); 
}


void IterativeUpsampling(const cv::Mat& input_rgb, const cv::Mat& input_depth, cv::Mat& output, int window_size, double sigma_spatial, double sigma_range) {

	int uf = log2(input_rgb.rows / input_depth.rows); // upsample factor
	cv::Mat D = input_depth.clone(); // depth image
	cv::Mat I = input_rgb.clone(); // input image

	for (int i = 0; i < uf; ++i)
	{
		cv::resize(D, D, D.size() * 2); 
		cv::resize(I, I, D.size());	
		JointBilateralFilter(I, D, D, window_size, sigma_spatial, sigma_range); 

	}
	cv::resize(D, D, input_rgb.size()); 
	JointBilateralFilter(input_rgb, D, output, window_size, sigma_spatial, sigma_range); 

}

void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length)
{
  const auto& b = baseline ;
  const auto& f = focal_length ;
  std::stringstream out3d;
  out3d << output_file << ".xyz";
  std::ofstream outfile(out3d.str());
  for (int i = 0; i < height - window_size; ++i) {
    std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
    for (int j = 0; j < width - window_size; ++j) {
      if (disparities.at<uchar>(i, j) == 0) continue;

      const double d = static_cast<double>(disparities.at<uchar>(i, j) + dmin);

      const double Z = f * b / d;
      const double X = (i - width / 2) * b / d;
      const double Y = (j - height / 2) * b / d;
	  
      outfile << X << " " << Y << " " << Z << std::endl;
    }
  }
  std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
  std::cout << std::endl;
}


int main(int argc, char** argv) {


	// camera setup parameters
	const double focal_length = 3740;
	const double baseline = 160;

	// stereo estimation parameters
	const int dmin = 200; //for Art, Books, Dolls, Moebius
	//const int dmin = 230; //for laundry, Reindeer
	// const int dmin = 240; //for bowling
	//const int dmin = 251; //for flowerpots
	//const int dmin = 270; //for Aloe
	//const int dmin = 280; //for plastic
	//const int dmin = 290; //for cloth
	//const int dmin = 300; //for baby



	// Commandline arguments
	if (argc < 7) {
		std::cerr << "Usage: " << argv[0] << " RGB_IMAGE DEPTH_IMAGE OUTPUT_FILE WINDOW_SIZE SPATIAL_SIGMA RANGE_SIGMA" << std::endl;
		return 1;
	}

	cv::Mat im = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); //pixel format uchar.. 8bit 
	cv::Mat im_depth = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);//pixel format uchar.. 8bit 
	const std::string output_file = argv[3];
	const int window_size= atoi(argv[4]);
	const float sigma_s = atof(argv[5]);
	const float sigma_r = atof(argv[6]);


	if (!im.data) {
		std::cerr << "No image1 data" << std::endl;
		return EXIT_FAILURE;
	}

	if (!im_depth.data) {
		std::cerr << "No image2 data" << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "------------------ Parameters -------------------" << std::endl;
	std::cout << "focal_length = " << focal_length << std::endl;
	std::cout << "baseline = " << baseline << std::endl;
	std::cout << "window_size = " << window_size << std::endl;
	std::cout << "disparity added due to image cropping = " << dmin << std::endl;
	std::cout << "output filename = " << argv[3] << std::endl;
	std::cout << "Spatial Sigma = " << sigma_s << std::endl;
	std::cout << "Range Sigma = " << sigma_r << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;

	int height = im.size().height;
	int width = im.size().width;


	cv::Mat noise(im.size(), im.type());
	uchar mean = 0;
	uchar stddev = 25;
	cv::randn(noise, mean, stddev);

	//im.copyTo(...) // gt....
	cv::Mat GT, im_depth_downsampled;
    im.copyTo(GT);

	im += noise; //input with noise

	//cv::imshow("im", im);
	//cv::imshow("depth image", im_depth);
	//std::cout << "original size:" << im_depth.size() << std::endl;
	cv::resize(im_depth, im_depth_downsampled, im_depth.size() / 4, 0, 0, cv::INTER_LINEAR);

	//cv::imshow("depth image resized", im_depth_downsampled);
	//std::cout << "downsampled size:" << im_depth_downsampled.size() << std::endl;


	//processing time
	std::stringstream outTime;
	outTime << output_file << "_processing_time.txt";
	std::ofstream outfileTime(outTime.str());

	//cv::waitKey();
	std::stringstream out1, out2, out3, out4;
	// gaussian
	cv::Mat output, output_iter;
	output = cv::Mat::zeros(im.size().height, im.size().width, CV_8UC1);

	//Bilateral Filter
	BilateralFilter(im, output, window_size, sigma_s, sigma_r);


	out1 << output_file << "_bilateral.png";
  	cv::imwrite(out1.str(), output);
	cv::namedWindow("Bilateral Filter", cv::WINDOW_AUTOSIZE);
  	cv::imshow("Bilateral Filter", output);

	output = cv::Mat::zeros(im_depth.size().height, im_depth.size().width, CV_8UC1);

	
	//Implemented Guided Filter : JB
	JointBilateralFilter(im, im_depth, output, window_size, sigma_s, sigma_r);


  	out2 << output_file << "_JB.png";
  	cv::imwrite(out2.str(), output);
	cv::namedWindow("JB_Filter", cv::WINDOW_AUTOSIZE);
  	cv::imshow("JB_Filter", output);

	//Implemented Upsampling Filter : JBU
	double current_time;
	current_time = (double)cv::getTickCount();
	JointBilateralUpsampling(im, im_depth_downsampled, output, window_size, sigma_s, sigma_r);
  	current_time = ((double)cv::getTickCount() - current_time)/cv::getTickFrequency();
  	outfileTime << "JBU: " << current_time << " seconds" << std::endl;

  	out3 << output_file << "_JBU.png";
  	cv::imwrite(out3.str(), output);
	cv::namedWindow("JBU_Filter", cv::WINDOW_AUTOSIZE);
  	cv::imshow("JBU_Filter", output);


	output_iter = cv::Mat::zeros(im_depth.size().height, im_depth.size().width, CV_8UC1);
	//Implemented Upsampling Filter : Iterative Upsampling
	current_time = (double)cv::getTickCount();
	IterativeUpsampling(im, im_depth_downsampled, output_iter, window_size, sigma_s, sigma_r);
  	current_time = ((double)cv::getTickCount() - current_time)/cv::getTickFrequency();
  	outfileTime << "IterativeU: " << current_time << " seconds" << std::endl;

  	out4 << output_file << "_Iterative.png";
  	cv::imwrite(out4.str(), output_iter);
	cv::namedWindow("Iterative_Upsampling", cv::WINDOW_AUTOSIZE);
  	cv::imshow("Iterative_Upsampling", output_iter);

	//reconstruction
	Disparity2PointCloud(
		output_file,
		height, width, output,
		window_size, dmin, baseline, focal_length);

	cv::waitKey();

	return 0;
}