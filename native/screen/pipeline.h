#pragma once
#include "../automatic.h"
#include "algorithm.h"
#include "apple_vfm.h"
#include <memory>

namespace sunglasses {
struct ScreenTelemetry {
    uint64_t drawId=0, frameId=0;
    int64_t queuedQpc=0;
    double captureSeconds=0;
    float strength=0, elapsed=0;
    bool reset=false, refresh=false;
    // load, mean luminance, event peak, event age, broad gate, three reserved floats.
    std::array<float,8> event{};
    ScreenAlgorithm algorithm=ScreenAlgorithm::SuddenHighlights;
};
// The caller owns the immediate context and source lifetime.
// source is SDR BGRA8 UNORM; target accepts premultiplied RGBA. No capture or UI.
class ScreenPipeline {
public:
    ScreenPipeline(ID3D11Device* device, ID3D11DeviceContext* context,
                   ScreenAlgorithm algorithm=ScreenAlgorithm::SuddenHighlights);
    void render(ID3D11Texture2D* source, ID3D11RenderTargetView* target,
                double timestamp, uint64_t id, float strength);
    // Redraw the last mask at its true age. Does not create a new capture or event.
    void refresh(ID3D11RenderTargetView* target, float strength, double secondsSinceCapture);
    // Opt-in metadata only: four 32-byte readbacks, never wait for the GPU.
    bool enableTelemetry() noexcept;
    void disableTelemetry() noexcept;
    bool takeTelemetry(ScreenTelemetry& sample) noexcept;
    uint64_t telemetryDropped() const noexcept { return telemetryDropped_; }
    HRESULT telemetryError() const noexcept { return telemetryError_; }
private:
    void resize(uint32_t width, uint32_t height);
    void draw(ID3D11RenderTargetView* target, float strength, float elapsed);
    void queueTelemetry(float strength,float elapsed) noexcept;
    struct Readback {
        ComPtr<ID3D11Buffer> buffer;
        ComPtr<ID3D11Query> ready;
        ScreenTelemetry sample;
    };
    std::array<Readback,4> readbacks_{};
    size_t readbackHead_=0, readbackCount_=0;
    uint64_t drawId_=0, telemetryDropped_=0;
    HRESULT telemetryError_=S_OK;
    bool telemetryEnabled_=false, lastReset_=false, refreshing_=false;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> convert_, control_;
    ComPtr<ID3D11VertexShader> vertex_;
    ComPtr<ID3D11PixelShader> overlay_;
    std::unique_ptr<AppleVfm> apple_;
    ComPtr<ID3D11Buffer> constants_, totals_, event_;
    ComPtr<ID3D11Texture2D> captured_, linear_;
    ComPtr<ID3D11ShaderResourceView> captureView_, linearView_, eventView_;
    ComPtr<ID3D11UnorderedAccessView> linearUav_, totalsUav_, eventUav_;
    uint32_t width_ = 0, height_ = 0, columns_ = 0, rows_ = 0;
    bool initialized_ = false;
    uint64_t lastId_ = 0;
    double lastTimestamp_ = 0;
    float lastElapsed_ = 0;
};
}
