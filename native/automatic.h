#pragma once
#include <d3d11.h>
#include <wrl/client.h>
#include <array>
#include <cstdint>
#include <deque>

namespace sunglasses {
using Microsoft::WRL::ComPtr;

struct Timing {
    float dt = 0;
    bool first = false, reset = false, duplicate = false;
    uint32_t continuity = 0; // 0 first, 1 warmup, 2 continuous, 3 discontinuity, 4 duplicate
};

class FrameClock {
public:
    Timing accept(uint64_t id, double timestamp, uint32_t width, uint32_t height, bool discontinuity);
private:
    bool initialized_ = false;
    uint64_t id_ = 0;
    double timestamp_ = 0;
    uint32_t width_ = 0, height_ = 0;
    std::deque<double> intervals_;
};

struct AutoParameters {
    float strength = 1, hold = .2f, release = .5f;
    bool repeat = true;
    bool highlightOnly = true;
};
struct AutoState {
    float amount, bright, rise, repeat;
    float hold;
    uint32_t latch, reasons, reserved;
};
struct BlockState {
    float mean, area, excess, positive, negative, anchor, extreme;
    int32_t direction;
    std::array<float, 8> ages;
    uint32_t count;
    float repeat, bright, rise;
};
static_assert(sizeof(AutoState) == 32 && sizeof(BlockState) == 80);

// Call on the owner of the immediate context. No capture, CPU readback, or file I/O here.
// Source is same-sized, linear SDR RGBA, finite [0,1]; caller owns color validation.
class AutomaticGpu {
public:
    AutomaticGpu(ID3D11Device* device, ID3D11DeviceContext* context);
    void resize(uint32_t width, uint32_t height);
    ID3D11Texture2D* process(ID3D11ShaderResourceView* source, Timing timing, AutoParameters parameters);
    ID3D11Buffer* stateBuffer() const { return state_.Get(); }
    ID3D11Buffer* blockBuffer() const { return blocks_.Get(); }
    uint32_t blockCount() const { return columns_ * rows_; }
private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>, 3> shaders_;
    ComPtr<ID3D11Buffer> constants_, state_, blocks_;
    ComPtr<ID3D11UnorderedAccessView> stateUav_, blocksUav_, outputUav_;
    ComPtr<ID3D11Texture2D> output_;
    std::array<ComPtr<ID3D11Texture2D>, 2> y_;
    std::array<ComPtr<ID3D11ShaderResourceView>, 2> ySrv_;
    std::array<ComPtr<ID3D11UnorderedAccessView>, 2> yUav_;
    uint32_t width_ = 0, height_ = 0, columns_ = 0, rows_ = 0, previous_ = 0;
    bool resized_ = true, processed_ = false;
};
void check(HRESULT result, const char* action);
} // namespace sunglasses
