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
�жϾ����ǲ�����ת����
R:In, ����Ҫ�жϵľ���
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
��ת����תŷ����
R:In, ����Ҫ��ת�ľ���
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
���Euler��
Rotation:In, ������ת�������Euler��
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
�����˲�
cloudIn:In, ����Ҫ���������˲��ĵ��Ƽ�
cloudOut:Out, ��������˲���ĵ��Ƽ�
range[4]:In, �˲��ķ�Χ
*/
int filter_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut,float range[6])
{
	pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
    //�����Ƴ��ķ�Χ
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::GT, range[0])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::LT, range[1])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::GT, range[2])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::LT, range[3])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GT, range[4])));
	range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, range[5])));
	//���������˲�����
	pcl::ConditionalRemoval<pcl::PointXYZ> condrem(range_cond);
	condrem.setInputCloud(cloudIn);		//�����������
	//�ýṹ���ƾ����˲�����֮����Ϊ�ṹ����
	condrem.setKeepOrganized(true);
	condrem.filter(*cloudOut);//ִ���˲���������˽����cloudOut��
	return 0;
}

/*
ֱͨ�˲�
cloud_in_out:In, ������Ҫ����ֱͨ�˲��ĵ��Ƽ�
min:In, ��Сֵ
max:In, ���ֵ
string s:In, �������͵��ֶ�
*/
int filter_passthrough(pcl::PointCloud<pcl::PointXYZ> &cloud_in_out, float min, float max, std::string s) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);	//���������ʹ�õ��м����
	//cloud = cloud_in_out;
	pcl::copyPointCloud(cloud_in_out, *cloud);

	//����ֱͨ�˲�����
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_infiltered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud);            //�����������
	pass.setFilterFieldName(s);         //���ù���ʱ����Ҫ�������͵��ֶ�
	pass.setFilterLimits(min, max);        //�����ڹ����ֶεķ�Χ
	pass.setFilterLimitsNegative (true);   //���ñ�����Χ�ڻ��ǹ��˵���Χ��
	pass.filter(*cloud_infiltered);            //ִ���˲���������˽����cloud_filtered
	*cloud = *cloud_infiltered;

	pcl::copyPointCloud(*cloud, cloud_in_out);
	//cloud_in_out = cloud;
	return 1;
}

void PCL_trans1(pcl::PointCloud<pcl::PointXYZ>::Ptr source, const cv::Mat trans_final, pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud)
{
	//cout << "trans_final :" << endl << trans_final << endl;
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();//�洢�任����
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
	std::cout << "����ѱ���"<< std::endl;
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
	std::cout << "����ѱ���" << std::endl;
	fs.release();
}

/*
ȥ�����
cloudIn:In, ��Ҫȥ�����ĵ��Ƽ�
*/
int removeNAN(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudIn)
{
	pcl::PointCloud<pcl::PointXYZ>::iterator it = cloudIn->points.begin();	//����һ���������Ƽ��ĵ�����
	while (it != cloudIn->points.end())
	{
		float x, y, z;	//��ȡÿ�����x, y, z, rgb
		x = it->x;
		y = it->y;
		z = it->z;
	//	rgb = it->rgb;
		//cout << "x: " << x << "  y: " << y << "  z: " << z << "  rgb: " << rgb << endl;
		if (!pcl_isfinite(x) || !pcl_isfinite(y) || !pcl_isfinite(z))// || !pcl_isfinite(rgb))//pcl_isfinite()��������һ������ֵ�����һ��ֵ�ǲ���������ֵ
		{
			it = cloudIn->points.erase(it);//x,y,z������ֵ���һ���Ƿ�������ֵ�Ͳ�����
		}
		else
			++it;
	}
	return 0;
}

/*
��ʾֱͨ�˲��������˲������ĵ��Ƽ�
cloud_16_transformed:In, ������16�ߵĵ���
cloud_32:In, ������32�ߵ���
*/
void display(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_16_transformed, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_32)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("trans"));
	viewer->setBackgroundColor(0, 0, 0);	//��ɫ
	viewer->addPointCloud<pcl::PointXYZ>(cloud_32, "cloud_32");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_16_transformed, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_16_transformed, single_color, "cloud_16_transformed");
	viewer->addCoordinateSystem(1.0);	//�ڿ��ӻ�����������ԭ�㣨0��0��0,�������һ������������άֻ�������ᣬPCL���ƿ�ʹ�õ���������ά����ϵ��
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

/*
ICP�㷨���������Ƽ�֮��ĸ���任������ת��ƽ��
source:In, һ������õĵ���
dest:In, ��һ������õĵ���
trans_final:Out, תΪOpenCV�е�Mat
*/
void PCL_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr dest, cv::Mat &trans_final)
{
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;//����ICP��ʵ����
	pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>);
	icp.setInputSource(source);//��������Դ
	icp.setInputTarget(dest);//��������Ŀ��
    icp.setMaxCorrespondenceDistance(1);//���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ�
	icp.setMaxCorrespondenceDistance(0.25);
	icp.setTransformationEpsilon(1e-8);	//�������α仯����֮��Ĳ�ֵ��һ������Ϊ1e-10���ɣ�
	icp.setMaximumIterations(10000);//��������������
	icp.setEuclideanFitnessEpsilon(1e-8);//�������������Ǿ�������С����ֵ��ֹͣ����
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


//��Ҫ�ı����range_16[6],range_32[6]�Լ�pcd·��
int main()
{
	std::string file16 = "./r1/r1_2/";
	std::string file32 = "./l1/l1_2/";

	//������x1,x2,y1,y2,ѡ��������(x1,x2),(y1,y2)֮��
	
	float range_16[6] = { -3.5,3.0,-1.4,1.4,-1.0,0.2 };  //l
	float range_32[6] = { -3.5,3.0,-1.4,1.4,-1.0,0.2 };  //r
	float min = -2.5, max = 2.0;

	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_16(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_32(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_16_temp(new pcl::PointCloud<pcl::PointXYZ>());//����PCD�ļ���ʱ��ŵĵ�������
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_32_temp(new pcl::PointCloud<pcl::PointXYZ>());//����PCD�ļ���ʱ��ŵĵ�������
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
	std::cout << "���Ƽ�����ϣ�" << std::endl;
	/*pcl::io::loadPCDFile("./16-3/33.pcd", *pointcloud_16);
	pcl::io::loadPCDFile("./32-3/32.pcd", *pointcloud_32);*/

	//ֱͨ�˲�
	filter_passthrough(*pointcloud_16, min, max, "x");	//���ĸ�����xΪ��Ҫ���˵��ֶ�
	filter_passthrough(*pointcloud_32, min, max, "x");
	std::cout << "ֱͨ�˲���ϣ�" << std::endl;
	display(pointcloud_16, pointcloud_32);

	//filter_passthrough(*pointcloud_16, min_ly1, max_ly1, "y");	//���ĸ�����xΪ��Ҫ���˵��ֶ�
	//filter_passthrough(*pointcloud_32, min_ry2, max_ry2, "y");
	//display(pointcloud_16, pointcloud_32);

	//�����˲�
	filter_plane(pointcloud_16, pointcloud_filtered_16,range_16);//���ĸ��������������˵ķ�Χ
	filter_plane(pointcloud_32, pointcloud_filtered_32,range_32);//����
	std::cout << "�����˲���ϣ�" << std::endl;
	display(pointcloud_filtered_16,pointcloud_filtered_32);//��ʾ����Ч�����仯���16��Ϳ����ɫ

	std::cout << "������׼����" << std::endl;
	cv::Mat trans_final(4, 4, CV_32F);
	
	removeNAN(pointcloud_filtered_16);
	removeNAN(pointcloud_filtered_32);            //ȥ�����

	PCL_ICP(pointcloud_filtered_16, pointcloud_filtered_32, trans_final);//����icp
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_filtered_16_changeCor_transformed(new pcl::PointCloud<pcl::PointXYZ>());
	PCL_trans1(pointcloud_filtered_16,trans_final, pointcloud_filtered_16_changeCor_transformed);

	std::cout << "EulerAngle��" << std::endl;
	std::cout <<printEulerAngle(trans_final) << std::endl;
	std::cout << "---------------------------------------------" << std::endl;
	display(pointcloud_filtered_16_changeCor_transformed, pointcloud_filtered_32);

	PCL_trans1(pointcloud_16, trans_final, pointcloud_16_transformed);//����������ľ�����ת16������
	
	save2xml1(trans_final);
	std::cout << "EulerAngle��" << std::endl;
	std::cout <<printEulerAngle(trans_final) << std::endl;
	std::cout << "��׼��ϣ�" << std::endl;
	display(pointcloud_16_transformed, pointcloud_32);//���仯���16��Ϳ����ɫ

	pcl::io::savePCDFileBinary("out_l.pcd", *pointcloud_16_transformed);
	return 0;
}