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

// Function to read the point cloud from a binary file.
pcl::PointCloud<pcl::PointXYZI>::Ptr readPointCloud(const fs::path& filePath) {
    ifstream infile(filePath, ios::binary);
    if (!infile) {
        cerr << "Error opening file: " << filePath << endl;
        return nullptr;
    }

    infile.seekg(0, ios::end);
    size_t fileSize = infile.tellg();
    infile.seekg(0, ios::beg);
    cout << "File size: " << fileSize << " bytes" << endl;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    int pointCount = 0;

    while (!infile.eof()) {
        pcl::PointXYZI point;
        infile.read(reinterpret_cast<char*>(&point.x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&point.y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&point.z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&point.intensity), sizeof(float));
        
        if (infile.gcount() != sizeof(float)) {
            break;
        }

        // Filter out points with zero intensity
        if (point.intensity == 0.0f) {
            cout << "Zero intensity point found at (" << point.x << ", " << point.y << ", " << point.z << "), skipping..." << endl;
            continue; // Skip this point
        }

        cloud->points.push_back(point);

        if (pointCount < 10) {
            cout << "Point " << pointCount << ": (" << point.x << ", " << point.y << ", " << point.z << ", " << point.intensity << ")" << endl;
        }
        pointCount++;
    }

    cout << "Total number of points read: " << pointCount << endl;

    if (infile.bad()) {
        cerr << "Error reading file: " << filePath << endl;
        infile.close();
        return nullptr;
    }

    infile.close();
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

// Function to display the point cloud.
void displayPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(cloud, "intensity");
    viewer->addPointCloud<pcl::PointXYZI>(cloud, intensity_distribution, "cloud");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    viewer->setCameraPosition(
        0, 0, 50, 
        0, 0, 0, 
        0, 1, 0
        );

    int open_time = 100000;
    while (!viewer->wasStopped()) {
        viewer->spinOnce(open_time);
        std::this_thread::sleep_for(std::chrono::milliseconds(open_time));
    }
}

// Function to write the point cloud to a text file.
void writePointCloudToFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, const string& filename) {
    ofstream outfile(filename);
    if (!outfile) {
        cerr << "Error opening output file: " << filename << endl;
        return;
    }

    outfile << "X,Y,Z,Intensity" << endl;
    for (const auto& point : cloud->points) {
        outfile << point.x << "," << point.y << "," << point.z << "," << point.intensity << endl;
    }

    outfile.close();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_bin_file_path>" << endl;
        return -1;
    } else if (argc < 3) {
        cerr << "Usage: " << argv[0] << " " << argv[1] << " <output_csv_file_path>" << endl;
        return -1;
    }

    string filePath = argv[1];
    string csvfileName  = argv[2];

    try {
        fs::path path(filePath);
        if (fs::is_regular_file(path) && path.extension() == ".bin") {
            auto cloud = readPointCloud(path);
            if (cloud) {
                writePointCloudToFile(cloud, csvfileName);
                displayPointCloud(cloud);
            }
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
