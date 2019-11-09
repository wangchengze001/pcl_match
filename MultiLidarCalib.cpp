#include<iostream>
#include<vector>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/filters/conditional_removal.h>
#include<pcl/filters/passthrough.h>
#include<pcl/registration/icp.h>
#include<opencv2/opencv.hpp>
#include<pcl/visualization/pcl_visualizer.h>
#include<opencv2/highgui.hpp>
#include"highgui.h"
#define ANGLE 5
using namespace cv;
using namespace std;

/*
判断矩阵是不是旋转矩阵
R:In, 输入要判断的矩阵
*/
bool isRotationMatrix(cv::Mat &R)
{
	cv::Mat Rt;
	cv::transpose(R, Rt);
	cv::Mat shouldBeIdentity = Rt*R;
	cv::Mat I = cv::Mat::eye(3,3,shouldBeIdentity.type());
	return norm(I, shouldBeIdentity) < 1e-6;
}

/*
旋转矩阵转欧拉角
R:In, 输入要旋转的矩阵
*/
Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{
	//std::cout <<"sdfafdasfdasdfadsf:"<<isRotationMatrix(R) << std::endl;
	//assert(isRotationMatrix(R));
	/*float sy = sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));
	bool singular = sy < 1e-6;
	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
		y = atan2(-R.at<float>(2, 0), sy);
		z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
	}
	 
	{
		x = atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
		y = atan2(-R.at<float>(2, 0), sy);
		z = 0;
	}*/
	float x, y, z;
	x = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
	y = atan2(-R.at<float>(2, 0), sqrt(R.at<float>(2, 1)*R.at<float>(2, 1) + R.at<float>(2, 2)*R.at<float>(2, 2)));
	z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
	return Vec3f(x, y, z);
}

/*
输出Euler角
Rotation:In, 输入旋转矩阵输出Euler角
*/
Vec3f printEulerAngle(cv::Mat &Rotation)
{
	cv::Mat rotationMatrix(cv::Size(3,3),CV_32F);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			rotationMatrix.at<float>(i, j) = Rotation.at<float>(i, j);
	Vec3f eulerAngle = rotationMatrixToEulerAngles(rotationMatrix);
	return eulerAngle;
}

/*
条件滤波
cloudIn:In, 输入要进行条件滤波的点云集
cloudOut:Out, 输出条件滤波后的点云集
range[4]:In, 滤波的范围
*/
int filter_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut,float range[6])
{
	pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
    //设置移除的范围
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::GT, range[0])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::LT, range[1])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::GT, range[2])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::LT, range[3])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GT, range[4])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, range[5])));
	//创建条件滤波对象
	pcl::ConditionalRemoval<pcl::PointXYZ> condrem(range_cond);
	condrem.setInputCloud(cloudIn);		//设置输出点云
	//让结构点云经过滤波操作之后，仍为结构点云
	condrem.setKeepOrganized(true);
	condrem.filter(*cloudOut);//执行滤波，保存过滤结果在cloudOut里
	return 0;
}

/*
直通滤波
cloud_in_out:In, 输入需要进行直通滤波的点云集
min:In, 最小值
max:In, 最大值
string s:In, 点云类型的字段
*/
int filter_passthrough(pcl::PointCloud<pcl::PointXYZ> &cloud_in_out, float min, float max, std::string s) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);	//处理过程中使用的中间变量
	//cloud = cloud_in_out;
	pcl::copyPointCloud(cloud_in_out, *cloud);

	//创建直通滤波对象
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_infiltered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud);            //设置输入点云
	pass.setFilterFieldName(s);         //设置过滤时所需要点云类型的字段
	pass.setFilterLimits(min, max);        //设置在过滤字段的范围
	pass.setFilterLimitsNegative (true);   //设置保留范围内还是过滤掉范围内
	pass.filter(*cloud_infiltered);            //执行滤波，保存过滤结果在cloud_filtered
	*cloud = *cloud_infiltered;

	pcl::copyPointCloud(*cloud, cloud_in_out);
	//cloud_in_out = cloud;
	return 1;
}

void PCL_trans1(pcl::PointCloud<pcl::PointXYZ>::Ptr source, const cv::Mat trans_final, pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud)
{
	//cout << "trans_final :" << endl << trans_final << endl;
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();//存储变换矩阵
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
			transform(i, j) = trans_final.at<float>(i, j);
	}
	cout << "Eigen trans :" << endl << transform << endl;
	pcl::transformPointCloud(*source, *transformed_cloud, transform);
}

void save2xml(cv::Mat trans)
{
	cv::FileStorage fs("trans.xml", cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		std::cout << "unable to open " << std::endl;
	}
	std::string s = "trans";
	fs << s << trans;
//	std::cout << trans << std::endl;
	std::cout << "结果已保存"<< std::endl;
	fs.release();
}

void save2xml1(const cv::Mat trans_final)
{
	cout << "trans_final :" << endl << trans_final << endl;
	cv::FileStorage fs("trans.xml", cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		std::cout << "unable to open " << std::endl;
	}
	std::string s = "trans";
	fs << s << trans_final;
	//	std::cout << trans << std::endl;
	std::cout << "结果已保存" << std::endl;
	fs.release();
}

/*
去除噪点
cloudIn:In, 需要去除噪点的点云集
*/
int removeNAN(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudIn)
{
	pcl::PointCloud<pcl::PointXYZ>::iterator it = cloudIn->points.begin();	//创建一个遍历点云集的迭代器
	while (it != cloudIn->points.end())
	{
		float x, y, z;	//获取每个点的x, y, z, rgb
		x = it->x;
		y = it->y;
		z = it->z;
	//	rgb = it->rgb;
		//cout << "x: " << x << "  y: " << y << "  z: " << z << "  rgb: " << rgb << endl;
		if (!pcl_isfinite(x) || !pcl_isfinite(y) || !pcl_isfinite(z))// || !pcl_isfinite(rgb))//pcl_isfinite()函数返回一个布尔值，检查一个值是不是正常数值
		{
			it = cloudIn->points.erase(it);//x,y,z三个数值里，有一个是非正常数值就擦除掉
		}
		else
			++it;
	}
	return 0;
}

/*
显示直通滤波和条件滤波处理后的点云集
cloud_16_transformed:In, 处理后的16线的点云
cloud_32:In, 处理后的32线点云
*/
void display(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_16_transformed, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_32)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("trans"));
	viewer->setBackgroundColor(0, 0, 0);	//黑色
	viewer->addPointCloud<pcl::PointXYZ>(cloud_32, "cloud_32");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_16_transformed, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_16_transformed, single_color, "cloud_16_transformed");
	viewer->addCoordinateSystem(1.0);	//在可视化窗口中坐标原点（0，0，0,）处添加一个红绿蓝的三维只是坐标轴，PCL点云库使用的是右手三维坐标系，
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

/*
ICP算法：两个点云集之间的刚体变换，即旋转和平移
source:In, 一个处理好的点云
dest:In, 另一个处理好的点云
trans_final:Out, 转为OpenCV中的Mat
*/
void PCL_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr dest, cv::Mat &trans_final)
{
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;//创建ICP的实例类
	pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>);
	icp.setInputSource(source);//设置输入源
	icp.setInputTarget(dest);//设置输入目标
    icp.setMaxCorrespondenceDistance(1);//设置对应点对之间的最大距离（此值对配准结果影响较大）
	icp.setMaxCorrespondenceDistance(0.25);
	icp.setTransformationEpsilon(1e-8);	//设置两次变化矩阵之间的差值（一般设置为1e-10即可）
	icp.setMaximumIterations(10000);//设置最大迭代次数
	icp.setEuclideanFitnessEpsilon(1e-8);//设置收敛条件是均方误差和小于阈值，停止迭代
	icp.align(*Final);
	Eigen::Matrix4f transform;
	transform = icp.getFinalTransformation().cast<float>();
	std::cout << transform << std::endl;

	std::cout << "has converged:" << icp.hasConverged() << ", score: " << icp.getFitnessScore() << std::endl;

	cv::Mat trans_temp(4, 4, CV_32F);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			trans_temp.at<float>(i, j) = transform(i, j);
		}
	}
	trans_final = trans_temp;

	//std::cout << "trans_final: " << std::endl << trans_final << std::endl;
}


//需要改变的是range_16[6],range_32[6]以及pcd路径
int main()
{
	std::string file16 = "./r1/r1_2/";
	std::string file32 = "./l1/l1_2/";

	//依次是x1,x2,y1,y2,选的区域在(x1,x2),(y1,y2)之间
	
	float range_16[6] = { -3.5,3.0,-1.4,1.4,-1.0,0.2 };  //l
	float range_32[6] = { -3.5,3.0,-1.4,1.4,-1.0,0.2 };  //r
	float min = -2.5, max = 2.0;

	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_16(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_32(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_16_temp(new pcl::PointCloud<pcl::PointXYZ>());//加载PCD文件临时存放的点云数据
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_32_temp(new pcl::PointCloud<pcl::PointXYZ>());//加载PCD文件临时存放的点云数据
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_filtered_16(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_filtered_32(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_filtered_16_changeCor(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_16_changeCor(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_16_transformed(new pcl::PointCloud<pcl::PointXYZ>());


	for (int i = 1; i <= 6; i++) {
		pcl::io::loadPCDFile(file16 + std::to_string(i) + ".pcd", *pointcloud_16_temp);
		pcl::io::loadPCDFile(file32 + std::to_string(i) + ".pcd", *pointcloud_32_temp);
		*pointcloud_16 += *pointcloud_16_temp;
		*pointcloud_32 += *pointcloud_32_temp;

	}
	std::cout << "点云加载完毕！" << std::endl;
	/*pcl::io::loadPCDFile("./16-3/33.pcd", *pointcloud_16);
	pcl::io::loadPCDFile("./32-3/32.pcd", *pointcloud_32);*/

	//直通滤波
	filter_passthrough(*pointcloud_16, min, max, "x");	//第四个参数x为需要过滤的字段
	filter_passthrough(*pointcloud_32, min, max, "x");
	std::cout << "直通滤波完毕！" << std::endl;
	display(pointcloud_16, pointcloud_32);

	//filter_passthrough(*pointcloud_16, min_ly1, max_ly1, "y");	//第四个参数x为需要过滤的字段
	//filter_passthrough(*pointcloud_32, min_ry2, max_ry2, "y");
	//display(pointcloud_16, pointcloud_32);

	//条件滤波
	filter_plane(pointcloud_16, pointcloud_filtered_16,range_16);//第四个参数是条件过滤的范围
	filter_plane(pointcloud_32, pointcloud_filtered_32,range_32);//过滤
	std::cout << "条件滤波完毕！" << std::endl;
	display(pointcloud_filtered_16,pointcloud_filtered_32);//显示过滤效果，变化后的16线涂成绿色

	std::cout << "正在配准……" << std::endl;
	cv::Mat trans_final(4, 4, CV_32F);
	
	removeNAN(pointcloud_filtered_16);
	removeNAN(pointcloud_filtered_32);            //去除噪点

	PCL_ICP(pointcloud_filtered_16, pointcloud_filtered_32, trans_final);//运行icp
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_filtered_16_changeCor_transformed(new pcl::PointCloud<pcl::PointXYZ>());
	PCL_trans1(pointcloud_filtered_16,trans_final, pointcloud_filtered_16_changeCor_transformed);

	std::cout << "EulerAngle：" << std::endl;
	std::cout <<printEulerAngle(trans_final) << std::endl;
	std::cout << "---------------------------------------------" << std::endl;
	display(pointcloud_filtered_16_changeCor_transformed, pointcloud_filtered_32);

	PCL_trans1(pointcloud_16, trans_final, pointcloud_16_transformed);//利用求出来的矩阵旋转16线数据
	
	save2xml1(trans_final);
	std::cout << "EulerAngle：" << std::endl;
	std::cout <<printEulerAngle(trans_final) << std::endl;
	std::cout << "配准完毕！" << std::endl;
	display(pointcloud_16_transformed, pointcloud_32);//将变化后的16线涂成绿色

	pcl::io::savePCDFileBinary("out_l.pcd", *pointcloud_16_transformed);
	return 0;
}