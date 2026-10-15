// Headless correctness runner. File transfer/readback is deliberately not a latency benchmark.
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "shader_source.h"

using Microsoft::WRL::ComPtr;

namespace {
void check(HRESULT result, const char* action) {
    if (FAILED(result)) {
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(result));
    }
}

struct Frame {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> rgba;
};

struct Parameters {
    float knee = 0.2f;
    float ceiling = 0.55f;
    float strength = 1.0f;
    float reserved = 0;
};
static_assert(sizeof(Parameters) == 16);

float parseNumber(const wchar_t* text) {
    std::wstring value(text);
    size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error("Expected finite numeric parameters");
    }
    return parsed;
}

Frame readFrame(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    std::array<char, 4> magic{};
    Frame frame;
    stream.read(magic.data(), 4);
    stream.read(reinterpret_cast<char*>(&frame.width), 4);
    stream.read(reinterpret_cast<char*>(&frame.height), 4);
    if (!stream || magic != std::array<char, 4>{'S', 'G', 'F', '1'} ||
        frame.width == 0 || frame.height == 0 || frame.width > 8192 || frame.height > 8192 ||
        uint64_t(frame.width) * frame.height > 16'000'000) {
        throw std::runtime_error("Invalid SGF1 header or dimensions");
    }
    const auto count = size_t(frame.width) * frame.height * 4;
    if (std::filesystem::file_size(path) != 12 + count * sizeof(float)) {
        throw std::runtime_error("SGF1 payload size mismatch");
    }
    frame.rgba.resize(count);
    stream.read(reinterpret_cast<char*>(frame.rgba.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!stream) throw std::runtime_error("Could not read SGF1 payload");
    for (float value : frame.rgba) {
        if (!std::isfinite(value) || value < 0 || value > 1) {
            throw std::runtime_error("Expected finite normalized SDR RGBA");
        }
    }
    return frame;
}

void writeFrame(const std::filesystem::path& path, const Frame& frame) {
    // CREATE_NEW prevents a race from overwriting another file after the CLI precheck.
    const HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Could not create output exclusively");
    auto write = [handle](const void* data, DWORD size) {
        DWORD written = 0;
        return WriteFile(handle, data, size, &written, nullptr) && written == size;
    };
    const bool success = write("SGF1", 4) && write(&frame.width, 4) && write(&frame.height, 4) &&
        write(frame.rgba.data(), static_cast<DWORD>(frame.rgba.size() * sizeof(float)));
    CloseHandle(handle);
    if (!success) throw std::runtime_error("Incomplete output write");
}

Frame compute(const Frame& input, const Parameters& parameters, D3D_DRIVER_TYPE driver) {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    const D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL actual{};
    check(D3D11CreateDevice(nullptr, driver, nullptr, 0, requested, 1, D3D11_SDK_VERSION,
                           &device, &actual, &context), "D3D11CreateDevice");
    ComPtr<IDXGIDevice> dxgiDevice;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC adapterDescription{};
    check(device.As(&dxgiDevice), "Query DXGI device");
    check(dxgiDevice->GetAdapter(&adapter), "Get adapter");
    check(adapter->GetDesc(&adapterDescription), "Get adapter description");

    ComPtr<ID3DBlob> bytecode, errors;
    const HRESULT compiled = D3DCompile(highlightSource, sizeof(highlightSource) - 1, "highlight.hlsl",
        nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0, &bytecode, &errors);
    if (FAILED(compiled) && errors) {
        throw std::runtime_error(static_cast<const char*>(errors->GetBufferPointer()));
    }
    check(compiled, "D3DCompile");
    ComPtr<ID3D11ComputeShader> shader;
    check(device->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader),
          "CreateComputeShader");

    D3D11_TEXTURE2D_DESC description{};
    description.Width = input.width;
    description.Height = input.height;
    description.MipLevels = 1;
    description.ArraySize = 1;
    description.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    description.SampleDesc.Count = 1;
    description.Usage = D3D11_USAGE_IMMUTABLE;
    description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial{};
    initial.pSysMem = input.rgba.data();
    initial.SysMemPitch = input.width * 4 * sizeof(float);
    ComPtr<ID3D11Texture2D> source, destination, staging;
    check(device->CreateTexture2D(&description, &initial, &source), "Create source texture");
    description.Usage = D3D11_USAGE_DEFAULT;
    description.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    check(device->CreateTexture2D(&description, nullptr, &destination), "Create destination texture");
    description.Usage = D3D11_USAGE_STAGING;
    description.BindFlags = 0;
    description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    check(device->CreateTexture2D(&description, nullptr, &staging), "Create readback texture");
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
    check(device->CreateShaderResourceView(source.Get(), nullptr, &srv), "Create SRV");
    check(device->CreateUnorderedAccessView(destination.Get(), nullptr, &uav), "Create UAV");

    D3D11_BUFFER_DESC bufferDescription{};
    bufferDescription.ByteWidth = sizeof(Parameters);
    bufferDescription.Usage = D3D11_USAGE_IMMUTABLE;
    bufferDescription.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA bufferData{};
    bufferData.pSysMem = &parameters;
    ComPtr<ID3D11Buffer> buffer;
    check(device->CreateBuffer(&bufferDescription, &bufferData, &buffer), "Create parameters");
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, srv.GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, uav.GetAddressOf(), nullptr);
    context->CSSetConstantBuffers(0, 1, buffer.GetAddressOf());
    context->Dispatch((input.width + 7) / 8, (input.height + 7) / 8, 1);
    ID3D11UnorderedAccessView* empty = nullptr;
    context->CSSetUnorderedAccessViews(0, 1, &empty, nullptr);
    context->CopyResource(staging.Get(), destination.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{};
    check(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map output");
    Frame output{input.width, input.height, std::vector<float>(input.rgba.size())};
    for (uint32_t row = 0; row < input.height; ++row) {
        std::memcpy(output.rgba.data() + size_t(row) * input.width * 4,
                    static_cast<const uint8_t*>(mapped.pData) + size_t(row) * mapped.RowPitch,
                    size_t(input.width) * 4 * sizeof(float));
    }
    context->Unmap(staging.Get(), 0);
    std::cout << "backend=" << (driver == D3D_DRIVER_TYPE_WARP ? "warp" : "hardware")
              << " feature_level=" << static_cast<unsigned>(actual)
              << " adapter_vendor=" << adapterDescription.VendorId
              << " adapter_device=" << adapterDescription.DeviceId << '\n';
    return output;
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 7) {
            std::cerr << "Usage: sunglasses-compute INPUT.sgf OUTPUT.sgf KNEE CEILING STRENGTH warp|hardware\n";
            return 2;
        }
        Parameters parameters{parseNumber(argv[3]), parseNumber(argv[4]), parseNumber(argv[5]), 0};
        if (!(0 < parameters.knee && parameters.knee < parameters.ceiling && parameters.ceiling <= 1 &&
              0 <= parameters.strength && parameters.strength <= 1)) {
            throw std::runtime_error("Invalid filter parameters");
        }
        const std::wstring backend(argv[6]);
        if (backend != L"warp" && backend != L"hardware") throw std::runtime_error("Invalid backend");
        if (std::filesystem::exists(argv[2])) throw std::runtime_error("Output already exists");
        auto output = compute(readFrame(argv[1]), parameters,
                              backend == L"warp" ? D3D_DRIVER_TYPE_WARP : D3D_DRIVER_TYPE_HARDWARE);
        writeFrame(argv[2], output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
