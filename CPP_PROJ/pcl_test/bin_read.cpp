#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <chrono>

using namespace std;
namespace fs = std::filesystem;

void readAndDisplayPointCloud(const fs::path& filePath) {
    // Open the binary file in binary mode
    ifstream infile(filePath, ios::binary);

    // Check if the file was opened successfully
    if (!infile) {
        cerr << "Error opening file: " << filePath << endl;
        return;
    }

    // Read the binary data into two separate point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>());

    int pointCount = 0;
    while (true) {
        pcl::PointXYZ point;
        infile.read(reinterpret_cast<char*>(&point.x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&point.y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&point.z), sizeof(float));
        if (infile.gcount() == 0) {
            break; // End of file reached
        }
        if (infile.gcount() != sizeof(float)) {
            cerr << "Error reading point data from file: " << filePath << endl;
            break;
        }

        if (pointCount % 2 == 0) {
            cloud1->points.push_back(point);
        } else {
            cloud2->points.push_back(point);
        }

        if (pointCount < 10) { // Print first 10 points for inspection
            cout << "Point " << pointCount << ": (" << point.x << ", " << point.y << ", " << point.z << ")" << endl;
        }
        pointCount++;
    }

    cout << "Total number of points read: " << pointCount << endl;

    // Check if the read operation was successful
    if (infile.bad()) {
        cerr << "Error reading file: " << filePath << endl;
        infile.close();
        return;
    }

    // Close the file
    infile.close();

    // Set width and height of the point clouds
    cloud1->width = cloud1->points.size();
    cloud1->height = 1;
    cloud1->is_dense = true;

    cloud2->width = cloud2->points.size();
    cloud2->height = 1;
    cloud2->is_dense = true;

    // Visualize the point clouds
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_color(cloud1, 255, 0, 0); // Red
    viewer->addPointCloud<pcl::PointXYZ>(cloud1, cloud1_color, "cloud1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud2_color(cloud2, 0, 255, 0); // Green
    viewer->addPointCloud<pcl::PointXYZ>(cloud2, cloud2_color, "cloud2");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud2");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Set the camera position and orientation
    viewer->setCameraPosition(
        0, 0, -50,  // Camera position: 10 units back along the z-axis
        0, 0, 0,    // Viewpoint: looking at the origin
        0, -1, 0    // Up vector: positive y-axis
    );

    int open_time = 100000;
    while (!viewer->wasStopped()) {
        viewer->spinOnce(open_time);
        std::this_thread::sleep_for(std::chrono::milliseconds(open_time));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file_path>" << endl;
        return -1;
    }

    // Specify the file path
    string filePath = argv[1];

    try {
        // Check if the specified path is a valid file
        fs::path path(filePath);
        if (fs::is_regular_file(path) && path.extension() == ".bin") {
            readAndDisplayPointCloud(path);
        } else {
            cerr << "The specified path is not a valid .bin file: " << filePath << endl;
            return 1;
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
