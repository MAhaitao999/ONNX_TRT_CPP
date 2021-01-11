#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <numeric>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define CHECK(status) do {            \
    cudaError_t err = (status);       \
    if (err != cudaSuccess) {         \
        fprintf(stderr, "API error"   \
            "%s:%d Returned:%d\n",    \
            __FILE__, __LINE__, err); \
        exit(1);                      \
    }                                 \
} while(0)

// utilities ---------------------------------------------------------------------------------
// class to log errors, warnings, and other information during the build and inference phases.
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
} gLogger;

// destory TensorRT objects if something goes wrong
struct TRTDestroy {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (size_t i=0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }

    return size;
}

// get classes names
std::vector<std::string> getClassNames(const std::string& imagenet_classes) {
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good()) {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name)) {
        classes.push_back(class_name);
    }
    return classes;
}

// preprocessing stage ----------------------------------------------------------------------
void preprocessImage(const std::string& image_path, float* cpu_input, const nvinfer1::Dims& dims) {
    // read input image
    cv::Mat frame = cv::imread(image_path, 1);
    // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    if (frame.empty()) {
        std::cerr << "Input image " << image_path << " load failed\n";
    }
    auto input_width = dims.d[3];
    auto input_height = dims.d[2];
    auto channels = dims.d[1];
    std::cout << "(" << channels << ", " 
              << input_height << ", " 
              << input_width << ")" << std::endl;
    auto input_size = cv::Size(input_width, input_height);
    cv::Mat resized;
    cv::resize(frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    cv::Mat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // to tensor
    std::vector<cv::Mat> chw;
    for (size_t i=0; i < channels; ++i) {
        chw.emplace_back(cv::Mat(input_size, CV_32FC1, cpu_input + i * input_width * input_height));
    }
    cv::split(flt_image, chw);
}

// postprocessing stage ----------------------------------------------------------------------
void postprocessResults(float* cpu_output, const nvinfer1::Dims &dims, int batch_size) {
    // get class names
    auto classes = getClassNames("../imagenet_classes.txt");
    
    std::vector<int> output_array(cpu_output, cpu_output+1000);
    // calculate softmax
    std::transform(output_array.begin(), output_array.end(), output_array.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(output_array.begin(), output_array.end(), 0.0);
    // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims)*batch_size);
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&output_array](int i1, int i2) {return output_array[i1] > output_array[i2];});
    // print results
    int i = 0;
    while (output_array[indices[i]] / sum > 0.5) {
        if (classes.size() > indices[i]) {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * output_array[indices[i]] / sum << "% | index: " << indices[i] << "\n";
        ++i;
    }
}

// intialize TensorRT engine and parse ONNX model
void parseOnnxModel(const std::string& model_path, 
                    TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context,
                    std::string engine_name) {
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    nvinfer1::IHostMemory* trtModelStream{nullptr}; // 用于序列化engine

    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "ERROR: could not parse the model.\n";
        exit(1);
    }

    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);
    // generate TensorRT engine optimized for the target platform
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
    // 将转换好的TensorRT object序列化到内存中, trtModelStream是一块内存空间.
    // 这里也可以序列化到磁盘中.
    // Serialize engine and destroy it
    trtModelStream = engine->serialize();
    std::ofstream p(engine_name.c_str(), std::ios::binary);
    p.write((const char*)trtModelStream->data(), trtModelStream->size());
    p.close();
}

void deserializeEngineModel(const std::string engine_name, 
                            TRTUniquePtr<nvinfer1::ICudaEngine>& engine, 
                            TRTUniquePtr<nvinfer1::IExecutionContext>& context) {
    std::ifstream in_file(engine_name.c_str(), std::ios::in | std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "ERROR: fail to open file: " << engine_name.c_str() << std::endl;
        exit(1);
    }

    std::streampos begin, end;
    begin = in_file.tellg();
    in_file.seekg(0, std::ios::end);
    end = in_file.tellg();
    size_t size = end - begin;
    std::cout << "engine file size: " << size << " bytes" << std::endl;
    in_file.seekg(0, std::ios::beg);
    std::unique_ptr<unsigned char[]> engine_data(new unsigned char[size]);
    in_file.read((char*)engine_data.get(), size);
    in_file.close();

    // deserialize the engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    engine.reset(runtime->deserializeCudaEngine((const void*)engine_data.get(), size, nullptr));
    context.reset(engine->createExecutionContext());
    
}

inline bool exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

int main(int argc, char* argv[]) {

    // get class names
    // auto classes = getClassNames("../imagenet_classes.txt");
    /*
    for(auto iter=classes.begin(); iter < classes.end(); iter++) {
        std::cout << *iter << std::endl;
    }
    */

    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " model.onnx image.jpg\n";
        return -1;
    }

    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    int batch_size = 1;
    
    // initialize TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    std::string engine_name = model_path.substr(0, model_path.rfind(".")) + ".engine";
    if (!exists(engine_name)) {
        parseOnnxModel(model_path, engine, context, engine_name);
    }
    else {
        deserializeEngineModel(engine_name, engine, context);
    }

    // get sizes of input and output and allocate memory 
    // required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims;  // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output

    // std::cout << engine->getNbBindings() << std::endl; // 获取与这个engine相关的输入输出tensor的数量
    std::vector<void*> buffers(engine->getNbBindings()); // cpu buffers for input and output data
    std::vector<void*> gpu_buffers(engine->getNbBindings()); // gpu buffers for input and output data
    
    // 创建cuda流, 用于管理数据复制, 存取和计算的并发操作
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    for (size_t i=0; i < engine->getNbBindings(); ++i) {
        std::cout << i << " io" << std::endl;
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);        
        std::cout << "binding_size is: " << binding_size << std::endl;
        buffers[i] = (void*)malloc(binding_size);
        CHECK(cudaMalloc((void**)(&gpu_buffers[i]), binding_size));
        if (engine->bindingIsInput(i)) {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }

    if (input_dims.empty() || output_dims.empty()) {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }

    std::cout << "input_dims[0] is: " << input_dims[0].d[0] << ", "
                                      << input_dims[0].d[1] << ", "
                                      << input_dims[0].d[2] << ", "
                                      << input_dims[0].d[3] << ", " << std::endl;

    preprocessImage(image_path, (float*)buffers[0], input_dims[0]);
    
    // 从内存到显存, 从CPU到GPU, 将输入数据拷贝到显存
    // buffers[0]是读入内存中的数据; gpu_buffers[0]是显存上的存储区域, 用于存放输入数据
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(gpu_buffers[0], buffers[0], 
                          getSizeByDim(engine->getBindingDimensions(0)) * batch_size * sizeof(float), 
                          cudaMemcpyHostToDevice, stream));

    // 启动cuda核, 异步执行推理计算.
    context->enqueue(batch_size, gpu_buffers.data(), stream, nullptr);

    // 从显存到内存, 将计算结果拷贝回内存中.
    // buffers[1]是内存中的存储区域; gpu_buffers[1]是显存中的存储区域, 存放模型输出.
    CHECK(cudaMemcpyAsync(buffers[1], gpu_buffers[1], 
                          getSizeByDim(engine->getBindingDimensions(1)) * batch_size * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));

    // 这个是为了同步不同的流
    cudaStreamSynchronize(stream);

    /*
    // 测试输入数据
    std::cout << "输入数据: " << std::endl;
    float *p = (float*)buffers[0];
    for (size_t i=0; i < 1*3*224*224; i++) {
        std::cout << p[i] << std::endl;
    }
    
    // 测试输出数据
    std::cout << "输出数据: " << std::endl;
    p = (float*)buffers[1];
    for (size_t i=0; i < 1*1000; i++) {
        std::cout << p[i] << std::endl;   
    }

    // 测试
    void* pVoid = buffers[0];
    std::cout << "buffers[0] is: " << buffers[0] << std::endl;
    pVoid = buffers[1];
    std::cout << "buffers[1] is: " << buffers[1] << std::endl;
    */

    // postprocess results
    postprocessResults((float*)buffers[1], output_dims[0], batch_size);

    // 销毁流对象
    cudaStreamDestroy(stream);
    // 释放显存
    for (void* gpu_buf : gpu_buffers) {
        CHECK(cudaFree(gpu_buf));
    }
    // 释放内存
    for (void* buf : buffers) {
        free(buf);
    }

    return 0;
}
