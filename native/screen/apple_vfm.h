#pragma once
#include "../automatic.h"

namespace sunglasses {
// Fixed 60 Hz MATLAB-reference core; 100-nit/1265.625-deg2 assumptions.
// Image reduction, temporal history and mapping stay on the GPU.
class AppleVfm {
public:
    AppleVfm(ID3D11Device* device,ID3D11DeviceContext* context);
    void process(ID3D11ShaderResourceView* source,uint32_t width,uint32_t height,
                 double timestamp,bool newCapture,bool reset);
    void bindOverlay(float strength);
    ID3D11Buffer* telemetry() const { return telemetry_.Get(); }
private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> reduce_,control_;
    ComPtr<ID3D11PixelShader> overlay_;
    ComPtr<ID3D11Buffer> constants_,kernels_,totals_,history_,state_,telemetry_;
    ComPtr<ID3D11ShaderResourceView> stateView_;
    ComPtr<ID3D11UnorderedAccessView> totalsUav_,historyUav_,stateUav_,telemetryUav_;
    uint32_t width_=0,height_=0;
    double nextTick_=0,lastTime_=0;
    bool initialized_=false;
};
}
