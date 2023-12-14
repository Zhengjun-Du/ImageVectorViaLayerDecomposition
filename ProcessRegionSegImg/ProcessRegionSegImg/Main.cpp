
#include "Utility.h"
#include <iostream>
#include <direct.h>
#include <io.h>
#include "RegionInfo.h"

int main() { 

    string main_dir = "../Data/";
    string cases[] = { "", "1-Syn1", "2-Syn2", "3-Syn3", "4-Syn4", "5-syn5", "6-Can", "7-Battery", "8-Phone", "9-Egg", "10-Cone" };
    int id = 1;

    for (int i = id; i < id+1; i++) {
        cout << "\n\nCase " << i << " : " << cases[i] << endl << endl;
        string data_dir = main_dir + cases[i];
        string input_img_path = data_dir + "/input.png";
        string input_seg_path = data_dir + "/seg.png";
        string input_mask = data_dir + "/mask.png";
        string output_region_path = data_dir + "/region.png";
        string output_param_path = data_dir + "/region_info.txt";
        string output_region_ind_path = data_dir + "/region_index.png";

        RegionInfo Ri(input_img_path, input_seg_path,input_mask);
        Ri.OutputRegionInfo_s1(output_region_path, output_param_path, output_region_ind_path);
        Ri.GetAdjacencyInfo(output_param_path);
        Ri.GetXjunctionInfo(output_param_path);
        Ri.OutputRegionInfo_s2(output_param_path);
    }
    return 0;
}