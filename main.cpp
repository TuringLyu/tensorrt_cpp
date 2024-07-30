#include <iostream>
#include <memory>
#include "model.hpp"
#include "utils.hpp"

using namespace std;



int main()
{
    const string enginePath = "F:/SLiM1100/Code/recon_slim1100/reconstruction/source/recon_model/ZZM60_ZU_ideal_epoch_800.engine";
    int input_w = 405;
    int output_w = 2025;
    Model model;
    model.initEngine(enginePath, input_w, output_w);
    
    vector<cv::Mat> imgs = readMultiframeTiff("Z:/2_Data/testDatas/Slim3.0/sample/Recon_v3.4.0.3/earthworm_60X_C2/Reconstruction/realign/earthworm_60X_S1_C2_T1_realign_merge.tif");
    
    vector<cv::Mat> res;
    for (int i = 0; i < 61; ++i) {
        cv::Mat mat(output_w, output_w, CV_16UC1);
        res.push_back(mat);
    }
    model.doInference(imgs, res);

    saveMatVectorToTiff(res, "Z:/2_Data/testDatas/Slim3.0/sample/Recon_v3.4.0.3/earthworm_60X_C2/Reconstruction/test.tif");
    system("pause");
}
