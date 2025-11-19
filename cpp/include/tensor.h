#ifndef TENSOR_H
#define TENSOR_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>

enum class Device {
    CPU,
    CUDA
};

enum class DType {
    FLOAT32,
    FLOAT16,
    INT8,
    INT32
};

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int>& shape, Device device = Device::CPU, DType dtype = DType::FLOAT32);
    Tensor(const Eigen::MatrixXf& data);
    
    // Factory methods
    static Tensor zeros(const std::vector<int>& shape, Device device = Device::CPU);
    static Tensor ones(const std::vector<int>& shape, Device device = Device::CPU);
    static Tensor randn(const std::vector<int>& shape, Device device = Device::CPU);
    
    // Shape operations
    std::vector<int> shape() const { return m_shape; }
    int size(int dim) const { return m_shape[dim]; }
    int numel() const;  // Total number of elements
    Tensor transpose() const;  // Matrix transpose
    
    // Device operations
    Device device() const { return m_device; }
    DType dtype() const { return m_dtype; }
    Tensor to(Device device);  // Move to device
    Tensor to(DType dtype);    // Convert dtype
    
    // Data access (CPU only for now)
    float* data_ptr();
    const float* data_ptr() const;
    Eigen::MatrixXf& matrix() { return m_cpu_data; }
    const Eigen::MatrixXf& matrix() const { return m_cpu_data; }
    
    // Operations
    Tensor matmul(const Tensor& other) const;
    Tensor add(const Tensor& other) const;
    Tensor mul(float scalar) const;
    
    // Utility
    void print(int max_elements = 10) const;
    
private:
    std::vector<int> m_shape;
    Device m_device;
    DType m_dtype;
    
    // CPU storage
    Eigen::MatrixXf m_cpu_data;
    
    // CUDA storage (will implement later)
    void* m_cuda_data = nullptr;
    
    void allocate_cpu();
    void allocate_cuda();
};

#endif // TENSOR_H