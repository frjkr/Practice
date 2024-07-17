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
#include <tuple>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

using namespace std;
namespace fs = std::filesystem;

// Function to read the point cloud from a binary file.
tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> readPointCloud(const fs::path& filePath) {
    ifstream infile(filePath, ios::binary);
    if (!infile) {
        cerr << "Error opening file: " << filePath << endl;
        return {nullptr, nullptr};
    }

    infile.seekg(0, ios::end);
    size_t fileSize = infile.tellg();
    infile.seekg(0, ios::beg);
    cout << "File size: " << fileSize << " bytes" << endl;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_zero_intensity(new pcl::PointCloud<pcl::PointXYZI>());
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
            cloud_zero_intensity->points.push_back(point);
            pointCount++;
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
        return {nullptr, nullptr};
    }
    infile.close();

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    cloud_zero_intensity->width = cloud_zero_intensity->points.size();
    cloud_zero_intensity->height = 1;
    cloud_zero_intensity->is_dense = true;

    return {cloud, cloud_zero_intensity};
}

// Function to display the point clouds in two different windows.
void displayPointClouds(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_zero_intensity) {
    // Window 1 for the main cloud (non-zero intensity)
    pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Main Cloud Viewer"));
    viewer1->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(cloud, "intensity");
    viewer1->addPointCloud<pcl::PointXYZI>(cloud, intensity_distribution, "cloud");
    viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    viewer1->addCoordinateSystem(1.0);
    viewer1->initCameraParameters();
    viewer1->setCameraPosition(
        0, 0, 50, 
        0, 0, 0, 
        0, 1, 0);

    // Window 2 for the zero intensity cloud
    pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Zero Intensity Cloud Viewer"));
    viewer2->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> zero_intensity_color(cloud_zero_intensity, 255, 0, 0);
    viewer2->addPointCloud<pcl::PointXYZI>(cloud_zero_intensity, zero_intensity_color, "cloud_zero_intensity");
    viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_zero_intensity");
    viewer2->addCoordinateSystem(1.0);
    viewer2->initCameraParameters();
    viewer2->setCameraPosition(
        0, 0, 50, 
        0, 0, 0, 
        0, 1, 0);

    // Loop to keep both windows open
    int open_time = 100000;
    while (!viewer1->wasStopped() && !viewer2->wasStopped()) {
        viewer1->spinOnce(open_time);
        viewer2->spinOnce(open_time);
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
    string fileName  = argv[2];
    string Intensity = fileName + ".csv";
    string zeroIntensity = fileName + "_zero_intensity.csv";

    try {
        fs::path path(filePath);
        if (fs::is_regular_file(path) && path.extension() == ".bin") {
            auto [cloud, cloud_zero_intensity] = readPointCloud(path);
            if (cloud && cloud_zero_intensity) {
                writePointCloudToFile(cloud, Intensity);
                writePointCloudToFile(cloud_zero_intensity, zeroIntensity);
                displayPointClouds(cloud, cloud_zero_intensity);
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
