// Offline sequence correctness runner. Synchronous I/O/readback is confined to this headless tool.
#include "session.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sunglasses;
namespace {
struct FrameHeader {
    uint32_t width, height;
    uint64_t id;
    double timestamp;
    uint32_t flags, reserved;
};
static_assert(sizeof(FrameHeader) == 32);

template<class T> void read(std::ifstream& stream, T& value) {
    if (!stream.read(reinterpret_cast<char*>(&value), sizeof(value))) throw std::runtime_error("truncated_input");
}
void readPixels(std::ifstream& stream, std::vector<float>& pixels) {
    if (!stream.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size() * sizeof(float))))
        throw std::runtime_error("truncated_pixels");
    for (float value : pixels)
        if (!std::isfinite(value) || value < 0 || value > 1) throw std::runtime_error("invalid_sdr_rgba");
}

class Input {
public:
    explicit Input(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
        std::array<char, 4> magic{};
        read(stream_, magic); read(stream_, count);
        if (magic != std::array<char, 4>{'S', 'G', 'A', '1'} || !count || count > 7200)
            throw std::runtime_error("invalid_sequence_header");
        uint64_t total = 0;
        // Validate the entire bounded input before creating output. Reuse one frame allocation.
        for (uint32_t index = 0; index < count; ++index) {
            FrameHeader header{};
            read(stream_, header);
            total += uint64_t(header.width) * header.height;
            if (!header.width || !header.height || header.width > 8192 || header.height > 8192 ||
                uint64_t(header.width) * header.height > 16'000'000 || total > 20'000'000 ||
                !std::isfinite(header.timestamp) || header.timestamp < 0 || header.flags > 1 || header.reserved)
                throw std::runtime_error("invalid_frame_header_or_budget");
            pixels.resize(size_t(header.width) * header.height * 4);
            readPixels(stream_, pixels);
            headers_.push_back(header);
        }
        if (stream_.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing_input");
        stream_.clear(); stream_.seekg(8);
    }
    FrameHeader next() {
        FrameHeader header{};
        read(stream_, header);
        if (index_ >= headers_.size() || std::memcmp(&header, &headers_[index_++], sizeof(header)))
            throw std::runtime_error("input_changed");
        pixels.resize(size_t(header.width) * header.height * 4);
        readPixels(stream_, pixels);
        return header;
    }
    uint32_t count = 0;
    std::vector<float> pixels;
private:
    std::ifstream stream_;
    std::vector<FrameHeader> headers_;
    size_t index_ = 0;
};

class Output {
public:
    explicit Output(const std::filesystem::path& path) {
        handle_ = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) throw std::runtime_error("output_create_new_failed");
    }
    ~Output() { if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_); }
    Output(const Output&) = delete;
    Output& operator=(const Output&) = delete;
    void write(const void* data, DWORD size) {
        DWORD written = 0;
        if (!WriteFile(handle_, data, size, &written, nullptr) || written != size) throw std::runtime_error("output_write_failed");
    }
    template<class T> void value(const T& item) { write(&item, sizeof(item)); }
private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

ComPtr<ID3D11Buffer> stagingBuffer(ID3D11Device* device, ID3D11Buffer* source) {
    D3D11_BUFFER_DESC desc{};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = desc.MiscFlags = desc.StructureByteStride = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> result;
    check(device->CreateBuffer(&desc, nullptr, &result), "Create buffer readback");
    return result;
}

void writeBuffer(Output& output, ID3D11DeviceContext* context, ID3D11Buffer* source, ID3D11Buffer* staging) {
    D3D11_BUFFER_DESC desc{};
    source->GetDesc(&desc);
    context->CopyResource(staging, source);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    check(context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped), "Map buffer");
    try { output.write(mapped.pData, desc.ByteWidth); }
    catch (...) { context->Unmap(staging, 0); throw; }
    context->Unmap(staging, 0);
}

float number(const wchar_t* value) {
    std::wstring text(value);
    size_t consumed = 0;
    float result = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(result)) throw std::runtime_error("invalid_parameter");
    return result;
}

void run(Input& input, const std::filesystem::path& target, AutoParameters parameters, bool warp) {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    const D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL actual{};
    check(D3D11CreateDevice(nullptr, warp ? D3D_DRIVER_TYPE_WARP : D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                           requested, 1, D3D11_SDK_VERSION, &device, &actual, &context), "Create device");
    ComPtr<IDXGIDevice> dxgi;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC adapterDesc{};
    check(device.As(&dxgi), "Get DXGI device");
    check(dxgi->GetAdapter(&adapter), "Get adapter");
    check(adapter->GetDesc(&adapterDesc), "Get adapter description");
    AutomaticSession processor(device.Get(), context.Get(), parameters);
    auto stateReadback = stagingBuffer(device.Get(), processor.diagnostics().stateBuffer());
    ComPtr<ID3D11Buffer> blocksReadback;
    ComPtr<ID3D11Texture2D> source, imageReadback;
    ComPtr<ID3D11ShaderResourceView> sourceView;
    uint32_t width = 0, height = 0;
    Output output(target);
    output.write("SGO1", 4); output.value(input.count);
    for (uint32_t i = 0; i < input.count; ++i) {
        const auto frame = input.next();
        const bool resized = width != frame.width || height != frame.height;
        if (resized) {
            width = frame.width; height = frame.height;
            source.Reset(); sourceView.Reset(); imageReadback.Reset();
            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = width; desc.Height = height;
            desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
            desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            check(device->CreateTexture2D(&desc, nullptr, &source), "Create input");
            check(device->CreateShaderResourceView(source.Get(), nullptr, &sourceView), "Create input view");
            desc.Usage = D3D11_USAGE_STAGING;
            desc.BindFlags = 0; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            check(device->CreateTexture2D(&desc, nullptr, &imageReadback), "Create image readback");
        }
        context->UpdateSubresource(source.Get(), 0, nullptr, input.pixels.data(), width * 16, 0);
        auto result = processor.process(sourceView.Get(), {frame.id, frame.timestamp, (frame.flags & 1) != 0});
        const auto& diagnostics = processor.diagnostics();
        if (resized) blocksReadback = stagingBuffer(device.Get(), diagnostics.blockBuffer());
        output.value(frame); output.value(result.timing.continuity); output.value(diagnostics.blockCount());
        writeBuffer(output, context.Get(), diagnostics.stateBuffer(), stateReadback.Get());
        writeBuffer(output, context.Get(), diagnostics.blockBuffer(), blocksReadback.Get());
        context->CopyResource(imageReadback.Get(), result.texture);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        check(context->Map(imageReadback.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map output");
        try {
            for (uint32_t row = 0; row < height; ++row)
                output.write(static_cast<const uint8_t*>(mapped.pData) + size_t(row) * mapped.RowPitch, width * 16);
        } catch (...) { context->Unmap(imageReadback.Get(), 0); throw; }
        context->Unmap(imageReadback.Get(), 0);
    }
    std::cout << "backend=" << (warp ? "warp" : "hardware") << " feature_level=" << static_cast<unsigned>(actual)
              << " adapter_vendor=" << adapterDesc.VendorId << " adapter_device=" << adapterDesc.DeviceId
              << " frames=" << input.count << '\n';
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 8 && argc != 9) {
            std::cerr << "usage: sunglasses-auto-compute INPUT.sga OUTPUT.sgo STRENGTH HOLD RELEASE REPEAT_0_OR_1 warp|hardware [highlight|legacy]\n";
            return 2;
        }
        AutoParameters parameters{number(argv[3]), number(argv[4]), number(argv[5]), false};
        const std::wstring repeat(argv[6]), backend(argv[7]);
        if (parameters.strength < 0 || parameters.strength > 1 || parameters.hold < 0 || parameters.hold > 5 ||
            parameters.release < .01f || parameters.release > 10 || (repeat != L"0" && repeat != L"1") ||
            (backend != L"warp" && backend != L"hardware")) throw std::runtime_error("invalid_parameters");
        parameters.repeat = repeat == L"1";
        const std::wstring policy = argc == 9 ? argv[8] : L"highlight";
        if (policy != L"highlight" && policy != L"legacy") throw std::runtime_error("invalid_policy");
        parameters.highlightOnly = policy == L"highlight";
        if (std::filesystem::exists(argv[2])) throw std::runtime_error("output_exists");
        Input input(argv[1]);
        run(input, argv[2], parameters, backend == L"warp");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error=" << error.what() << '\n';
        return 2;
    }
}
