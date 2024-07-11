#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
using namespace std;
using namespace pcl;
using namespace io;

int main(int argc, char **argv) 
{
    // Check if the number of arguments is less than 2
    if (argc < 2) {
        // Print an error message to standard error
        cerr << "Usage: " << argv[0] << " <output_file_path>" << endl;
        // Return -1 to indicate an error and exit the program
        return -1;
    }

    // If we reach here, it means we have enough arguments
    // The second argument (argv[1]) is expected to be the output file path
    string output_file_path = argv[1];
    PointCloud<PointXYZ> cloud;

    // Fill in the cloud data
    cloud.width = 5;
    cloud.height = 2;
    cloud.is_dense = false;
    cloud.points.resize(cloud.width * cloud.height);

    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        cloud.points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud.points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud.points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
    }

    // Save the PCD file to the specified path
    savePCDFileASCII(output_file_path, cloud);
    cerr << "Saved " << cloud.points.size() << " data points to " << output_file_path << "." << endl;

    for (size_t i = 0; i < cloud.points.size(); ++i)
        cerr << "    " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << endl;

    return 0;
}