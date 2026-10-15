#pragma once
#include "automatic.h"

namespace sunglasses {
struct FrameInfo {
    uint64_t id;
    double timestampSeconds;
    bool discontinuity = false;
};
struct FrameResult {
    ID3D11Texture2D* texture; // Borrowed: content changes on the next process call.
    Timing timing;
};

// One instance per source/owner thread. Encapsulates clock, resizing and GPU state.
class AutomaticSession {
public:
    AutomaticSession(ID3D11Device* device, ID3D11DeviceContext* context, AutoParameters parameters = {});
    FrameResult process(ID3D11ShaderResourceView* source, FrameInfo frame);
    void reset(AutoParameters parameters);
    const AutomaticGpu& diagnostics() const { return gpu_; }
private:
    AutoParameters parameters_;
    FrameClock clock_;
    AutomaticGpu gpu_;
};
} // namespace sunglasses
