
#include <opencv2/opencv.hpp>
#include <iostream>

void printMatVals(cv::Mat mat, std::string name) {
    for(int i = 0; i < mat.rows; i++) {
        std::string rowString = "";
        if (mat.channels() == 1) {
            const float* matRow = mat.ptr<float>(i);
            for(int j = 0; j < mat.cols; j++) {
                rowString = rowString + "   " + std::to_string(matRow[j]);
            }
        }
        else {
            raise std::logic_error("Multi-channel printing not supported yet!");
        }
        std::cout << rowString << std::endl;
    }
}


int main() {
    // Load the forward model
    std::string forward_model_path = "C:\\Users\\U01\\Documents\\FunCoding\\small_experiments_opencv\\examples\\data\\models\\forward_model.pb"; //"./data/models/forward_model.pb";
    cv::dnn::Net forward_net = cv::dnn::readNetFromTensorflow(forward_model_path);

    if (forward_net.empty()) {
        std::cerr << "Error loading the forward model." << std::endl;
        return -1;
    }
    else {
        std::cout << "Model loaded correctly!" << std::endl;
        auto lns = forward_net.getLayerNames();
        for(const auto& ln: lns) {
            auto x = ln.c_str();
            std::cout << x << std::endl;
        }
    }

    int num_features = 227; // Hardcoded for now. Need to change that...

    // Create dummy input data.
    int batch_size = 1;
    // The CV_32F*C1* is important! Just having CV_32F by itself doesn't work!
    cv::Mat dummy_input(batch_size, num_features, CV_32FC1, cv::Scalar(1.0));
    std::cerr << "TODO: Make py model depth 32F instead of 64F?" << std::endl;


    // Prepare the blob for DNN module.
    auto ia = cv::dnn::blobFromImage(dummy_input);

    // Set the input blob for the net and run inference.
    forward_net.setInput(ia);
    auto output_blobs = forward_net.forward();

    printMatVals(output_blobs, "NN Output");

    return 0;
}
