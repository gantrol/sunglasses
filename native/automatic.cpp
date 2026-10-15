#include "automatic.h"
#include "automatic_source.h"
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sunglasses {
void check(HRESULT result, const char* action) {
    if (FAILED(result)) throw std::runtime_error(std::string(action) + ": " + std::to_string(result));
}

Timing FrameClock::accept(uint64_t id, double timestamp, uint32_t width, uint32_t height, bool discontinuity) {
    if (!std::isfinite(timestamp) || timestamp < 0) throw std::runtime_error("invalid_timestamp");
    if (initialized_ && id == id_ && timestamp == timestamp_ && width == width_ && height == height_ && !discontinuity)
        return {0, false, false, true, 4};
    const double dt = initialized_ ? timestamp - timestamp_ : 0;
    bool reset = !initialized_ || discontinuity || width != width_ || height != height_ || dt <= 0 ||
                 id_ == std::numeric_limits<uint64_t>::max() || id != id_ + 1;
    if (!reset && intervals_.size() >= 3) {
        std::vector<double> sorted(intervals_.begin(), intervals_.end());
        std::sort(sorted.begin(), sorted.end());
        const size_t n = sorted.size();
        const double median = (sorted[n / 2] + sorted[(n - 1) / 2]) / 2;
        reset = dt > 1.5 * median + .000001;
    }
    const bool first = !initialized_;
    if (reset) intervals_.clear();
    else {
        if (intervals_.size() == 8) intervals_.pop_front();
        intervals_.push_back(dt);
    }
    initialized_ = true;
    id_ = id; timestamp_ = timestamp; width_ = width; height_ = height;
    // Very large warmup intervals cannot overflow shader floats. All time constants <=10s.
    const float boundedDt = reset ? 0 : static_cast<float>(std::min(dt, 1000.0));
    return {boundedDt, first, reset, false, first ? 0u : reset ? 3u : intervals_.size() >= 3 ? 2u : 1u};
}

namespace {
struct Constants {
    uint32_t width, height, columns, rows;
    float dt, strength, hold, release;
    uint32_t reset, first, repeat, highlightOnly;
};
static_assert(sizeof(Constants) == 48);

void structured(ID3D11Device* device, uint32_t stride, uint32_t count, const void* initial,
                ComPtr<ID3D11Buffer>& buffer, ComPtr<ID3D11UnorderedAccessView>& view) {
    buffer.Reset(); view.Reset();
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = stride * count;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA data{};
    data.pSysMem = initial;
    check(device->CreateBuffer(&desc, initial ? &data : nullptr, &buffer), "Create structured buffer");
    check(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &view), "Create structured UAV");
}
} // namespace

AutomaticGpu::AutomaticGpu(ID3D11Device* device, ID3D11DeviceContext* context) : device_(device), context_(context) {
    const char* entries[] = {"Analyze", "Control", "Apply"};
    for (size_t i = 0; i < shaders_.size(); ++i) {
        ComPtr<ID3DBlob> bytecode, errors;
        const HRESULT result = D3DCompile(automaticSource, sizeof(automaticSource) - 1, "automatic.hlsl", nullptr, nullptr,
            entries[i], "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_WARNINGS_ARE_ERRORS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0, &bytecode, &errors);
        if (FAILED(result) && errors) throw std::runtime_error(static_cast<const char*>(errors->GetBufferPointer()));
        check(result, "Compile automatic shader");
        check(device_->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shaders_[i]), "Create shader");
    }
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = sizeof(Constants);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    check(device_->CreateBuffer(&desc, nullptr, &constants_), "Create constants");
    const AutoState initial{};
    structured(device_.Get(), sizeof(AutoState), 1, &initial, state_, stateUav_);
}

void AutomaticGpu::resize(uint32_t width, uint32_t height) {
    if (!width || !height || width > 8192 || height > 8192 || uint64_t(width) * height > 16'000'000)
        throw std::runtime_error("invalid_dimensions");
    if (width_ == width && height_ == height) return;
    width_ = width; height_ = height;
    columns_ = std::min(32u, width);
    rows_ = std::min(height, std::clamp(static_cast<uint32_t>(std::floor(32.0 * height / width + .5)), 1u, 64u));
    output_.Reset(); outputUav_.Reset();
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width; desc.Height = height;
    desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    check(device_->CreateTexture2D(&desc, nullptr, &output_), "Create auto output");
    check(device_->CreateUnorderedAccessView(output_.Get(), nullptr, &outputUav_), "Create output UAV");
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    for (size_t i = 0; i < y_.size(); ++i) {
        y_[i].Reset(); ySrv_[i].Reset(); yUav_[i].Reset();
        check(device_->CreateTexture2D(&desc, nullptr, &y_[i]), "Create history texture");
        check(device_->CreateShaderResourceView(y_[i].Get(), nullptr, &ySrv_[i]), "Create history SRV");
        check(device_->CreateUnorderedAccessView(y_[i].Get(), nullptr, &yUav_[i]), "Create history UAV");
    }
    structured(device_.Get(), sizeof(BlockState), blockCount(), nullptr, blocks_, blocksUav_);
    previous_ = 0;
    resized_ = true;
}

ID3D11Texture2D* AutomaticGpu::process(ID3D11ShaderResourceView* source, Timing timing, AutoParameters parameters) {
    if (!output_ || !source || !std::isfinite(parameters.strength) || !std::isfinite(parameters.hold) ||
        !std::isfinite(parameters.release) || parameters.strength < 0 || parameters.strength > 1 ||
        parameters.hold < 0 || parameters.hold > 5 || parameters.release < .01f || parameters.release > 10 ||
        !std::isfinite(timing.dt) || timing.dt < 0) throw std::runtime_error("invalid_auto_parameters");
    if (timing.duplicate && !resized_ && processed_) return output_.Get();
    const bool first = timing.first || !processed_;
    const bool reset = first || timing.reset || resized_;
    const Constants values{width_, height_, columns_, rows_, reset ? 0 : timing.dt,
                           parameters.strength, parameters.hold, parameters.release,
                           reset ? 1u : 0u, first ? 1u : 0u, parameters.repeat ? 1u : 0u,
                           parameters.highlightOnly ? 1u : 0u};
    context_->UpdateSubresource(constants_.Get(), 0, nullptr, &values, 0, 0);
    ID3D11ShaderResourceView* srvs[] = {source, ySrv_[previous_].Get()};
    ID3D11UnorderedAccessView* uavs[] = {yUav_[1 - previous_].Get(), blocksUav_.Get(), stateUav_.Get(), outputUav_.Get()};
    context_->CSSetConstantBuffers(0, 1, constants_.GetAddressOf());
    context_->CSSetShaderResources(0, 2, srvs);
    context_->CSSetUnorderedAccessViews(0, 4, uavs, nullptr);
    context_->CSSetShader(shaders_[0].Get(), nullptr, 0);
    context_->Dispatch(columns_, rows_, 1);
    context_->CSSetShader(shaders_[1].Get(), nullptr, 0);
    context_->Dispatch(1, 1, 1);
    context_->CSSetShader(shaders_[2].Get(), nullptr, 0);
    context_->Dispatch((width_ + 7) / 8, (height_ + 7) / 8, 1);
    // Explicitly release views before readback, reusing input, or swapping history roles.
    ID3D11ShaderResourceView* emptySrv[2]{};
    ID3D11UnorderedAccessView* emptyUav[4]{};
    context_->CSSetShaderResources(0, 2, emptySrv);
    context_->CSSetUnorderedAccessViews(0, 4, emptyUav, nullptr);
    previous_ = 1 - previous_;
    resized_ = false; processed_ = true;
    return output_.Get();
}
} // namespace sunglasses
