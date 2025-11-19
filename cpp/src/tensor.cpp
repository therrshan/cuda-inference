#include "tensor.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <numeric>

// ==================== Constructors ====================

Tensor::Tensor() 
    : m_shape({0}), m_device(Device::CPU), m_dtype(DType::FLOAT32) {
}

Tensor::Tensor(const std::vector<int>& shape, Device device, DType dtype)
    : m_shape(shape), m_device(device), m_dtype(dtype) {
    
    if (device == Device::CPU) {
        allocate_cpu();
    } else {
        allocate_cuda();
    }
}

Tensor::Tensor(const Eigen::MatrixXf& data)
    : m_device(Device::CPU), m_dtype(DType::FLOAT32) {
    
    m_shape = {static_cast<int>(data.rows()), static_cast<int>(data.cols())};
    m_cpu_data = data;
}

// ==================== Factory Methods ====================

Tensor Tensor::zeros(const std::vector<int>& shape, Device device) {
    Tensor t(shape, device, DType::FLOAT32);
    if (device == Device::CPU) {
        t.m_cpu_data.setZero();
    }
    return t;
}

Tensor Tensor::ones(const std::vector<int>& shape, Device device) {
    Tensor t(shape, device, DType::FLOAT32);
    if (device == Device::CPU) {
        t.m_cpu_data.setOnes();
    }
    return t;
}

Tensor Tensor::randn(const std::vector<int>& shape, Device device) {
    Tensor t(shape, device, DType::FLOAT32);
    if (device == Device::CPU) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < t.m_cpu_data.rows(); i++) {
            for (int j = 0; j < t.m_cpu_data.cols(); j++) {
                t.m_cpu_data(i, j) = dist(gen);
            }
        }
    }
    return t;
}

// ==================== Shape Operations ====================

int Tensor::numel() const {
    return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<int>());
}

Tensor Tensor::transpose() const {
    if (m_device != Device::CPU) {
        throw std::runtime_error("CUDA transpose not yet implemented");
    }
    
    if (m_shape.size() != 2) {
        throw std::runtime_error("transpose requires 2D tensor");
    }
    
    Eigen::MatrixXf transposed = m_cpu_data.transpose();
    return Tensor(transposed);
}

// ==================== Device Operations ====================

Tensor Tensor::to(Device device) {
    if (device == m_device) {
        return *this;  // Already on the right device
    }
    
    // TODO: Implement CPU <-> CUDA transfer
    throw std::runtime_error("Device transfer not yet implemented");
}

Tensor Tensor::to(DType dtype) {
    if (dtype == m_dtype) {
        return *this;
    }
    
    // TODO: Implement dtype conversion
    throw std::runtime_error("Dtype conversion not yet implemented");
}

// ==================== Data Access ====================

float* Tensor::data_ptr() {
    if (m_device != Device::CPU) {
        throw std::runtime_error("data_ptr() only works for CPU tensors");
    }
    return m_cpu_data.data();
}

const float* Tensor::data_ptr() const {
    if (m_device != Device::CPU) {
        throw std::runtime_error("data_ptr() only works for CPU tensors");
    }
    return m_cpu_data.data();
}

// ==================== Operations ====================

Tensor Tensor::matmul(const Tensor& other) const {
    if (m_device != Device::CPU || other.m_device != Device::CPU) {
        throw std::runtime_error("CUDA matmul not yet implemented");
    }
    
    if (m_shape.size() != 2 || other.m_shape.size() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }
    
    if (m_shape[1] != other.m_shape[0]) {
        throw std::runtime_error("Incompatible dimensions for matmul");
    }
    
    Eigen::MatrixXf result = m_cpu_data * other.m_cpu_data;
    return Tensor(result);
}

Tensor Tensor::add(const Tensor& other) const {
    if (m_device != Device::CPU || other.m_device != Device::CPU) {
        throw std::runtime_error("CUDA add not yet implemented");
    }
    
    if (m_shape != other.m_shape) {
        throw std::runtime_error("Incompatible shapes for add");
    }
    
    Eigen::MatrixXf result = m_cpu_data + other.m_cpu_data;
    return Tensor(result);
}

Tensor Tensor::mul(float scalar) const {
    if (m_device != Device::CPU) {
        throw std::runtime_error("CUDA mul not yet implemented");
    }
    
    Eigen::MatrixXf result = m_cpu_data * scalar;
    return Tensor(result);
}

// ==================== Utility ====================

void Tensor::print(int max_elements) const {
    if (m_device != Device::CPU) {
        std::cout << "Tensor on CUDA device (cannot print)" << std::endl;
        return;
    }
    
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < m_shape.size(); i++) {
        std::cout << m_shape[i];
        if (i < m_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], device=CPU, dtype=FLOAT32)" << std::endl;
    
    if (m_shape.size() == 2) {
        int rows_to_print = std::min(max_elements, static_cast<int>(m_cpu_data.rows()));
        int cols_to_print = std::min(max_elements, static_cast<int>(m_cpu_data.cols()));
        
        for (int i = 0; i < rows_to_print; i++) {
            std::cout << "  [";
            for (int j = 0; j < cols_to_print; j++) {
                std::cout << m_cpu_data(i, j);
                if (j < cols_to_print - 1) std::cout << ", ";
            }
            if (cols_to_print < static_cast<int>(m_cpu_data.cols())) {
                std::cout << ", ...";
            }
            std::cout << "]" << std::endl;
        }
        if (rows_to_print < static_cast<int>(m_cpu_data.rows())) {
            std::cout << "  ..." << std::endl;
        }
    }
}

// ==================== Private Methods ====================

void Tensor::allocate_cpu() {
    if (m_shape.size() == 0 || m_shape.size() > 2) {
        throw std::runtime_error("Only 1D and 2D tensors supported for now");
    }
    
    int rows = m_shape[0];
    int cols = (m_shape.size() == 1) ? 1 : m_shape[1];
    
    m_cpu_data = Eigen::MatrixXf(rows, cols);
    m_cpu_data.setZero();
}

void Tensor::allocate_cuda() {
    // TODO: Implement CUDA allocation
    throw std::runtime_error("CUDA allocation not yet implemented");
}