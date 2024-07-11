#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread> // For std::this_thread::sleep_for
#include <chrono> // For std::chrono::milliseconds

using namespace std;
using namespace pcl;
using namespace io;

int main(int argc, char **argv) 
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file_path>" << endl;
        return -1;
    }

    string input_file_path = argv[1];
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);

    // Load the PCD file from the specified path
    if (loadPCDFile<PointXYZ>(input_file_path, *cloud) == -1) //* load the file
    {
        cerr << "Couldn't read file " << input_file_path << endl;
        return -1;
    }
    
    cout << "Loaded " << cloud->width * cloud->height << " data points from " << input_file_path << " with the following fields: " << endl;

    for (size_t i = 0; i < cloud->points.size(); ++i)
        cout << "    " << cloud->points[i].x << " " << cloud->points[i].y << " " << cloud->points[i].z << endl;

    // Create a PCLVisualizer object
    visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Add the point cloud to the viewer
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> single_color(cloud, 0, 255, 0);
    viewer->addPointCloud<PointXYZ>(cloud, single_color, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    // viewer->spin ();

    // Start the visualization loop
    int time = 10000;
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(time);
        std::this_thread::sleep_for(std::chrono::milliseconds(time));
    }

}
