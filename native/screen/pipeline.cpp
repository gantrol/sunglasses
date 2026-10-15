#include "pipeline.h"
#include "screen_source.h"
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace sunglasses {
namespace {
struct Constants {
    uint32_t width,height,columns,rows;
    float dt,strength,elapsed;
    uint32_t reset;
};
static_assert(sizeof(Constants)==32);
}
ScreenPipeline::ScreenPipeline(ID3D11Device* device, ID3D11DeviceContext* context,ScreenAlgorithm algorithm)
    : device_(device), context_(context) {
    auto compile = [](const char* entry, const char* profile) {
        ComPtr<ID3DBlob> code, errors;
        const D3D_SHADER_MACRO macros[]{{"SCREEN_COMPUTE",profile[0]=='c'?"1":"0"},{nullptr,nullptr}};
        const HRESULT hr = D3DCompile(screenSource, sizeof(screenSource)-1, "screen.hlsl", macros, nullptr,
            entry, profile, D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_WARNINGS_ARE_ERRORS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0, &code, &errors);
        if (FAILED(hr) && errors) throw std::runtime_error(static_cast<const char*>(errors->GetBufferPointer()));
        check(hr, "compile screen shader");
        return code;
    };
    auto code=compile("Convert","cs_5_0");
    check(device_->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&convert_),"convert shader");
    code=compile("Control","cs_5_0");
    check(device_->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&control_),"event shader");
    code=compile("Vertex","vs_5_0");
    check(device_->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vertex_),"vertex shader");
    code=compile("Overlay","ps_5_0");
    check(device_->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&overlay_),"overlay shader");
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth=sizeof(Constants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    check(device_->CreateBuffer(&desc,nullptr,&constants_),"screen constants");
    desc.ByteWidth=32*64*8; desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=8;
    check(device_->CreateBuffer(&desc,nullptr,&totals_),"screen totals");
    check(device_->CreateUnorderedAccessView(totals_.Get(),nullptr,&totalsUav_),"screen totals UAV");
    desc.ByteWidth=32; desc.StructureByteStride=16; desc.BindFlags|=D3D11_BIND_SHADER_RESOURCE;
    const float zero[8]{}; D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=zero;
    check(device_->CreateBuffer(&desc,&initial,&event_),"screen event");
    check(device_->CreateUnorderedAccessView(event_.Get(),nullptr,&eventUav_),"screen event UAV");
    check(device_->CreateShaderResourceView(event_.Get(),nullptr,&eventView_),"screen event view");
    if (algorithm==ScreenAlgorithm::AppleVfm) apple_=std::make_unique<AppleVfm>(device,context);
}

void ScreenPipeline::resize(uint32_t width, uint32_t height) {
    if (width==width_ && height==height_) return;
    if (!width || !height || width>8192 || height>8192 || uint64_t(width)*height>16'000'000)
        throw std::runtime_error("screen_size");
    captureView_.Reset(); linearView_.Reset(); linearUav_.Reset(); captured_.Reset(); linear_.Reset();
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width=width; desc.Height=height; desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
    desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM; desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    check(device_->CreateTexture2D(&desc,nullptr,&captured_),"screen texture");
    check(device_->CreateShaderResourceView(captured_.Get(),nullptr,&captureView_),"screen view");
    if (!apple_) {
        desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; desc.BindFlags|=D3D11_BIND_UNORDERED_ACCESS;
        check(device_->CreateTexture2D(&desc,nullptr,&linear_),"linear texture");
        check(device_->CreateShaderResourceView(linear_.Get(),nullptr,&linearView_),"linear view");
        check(device_->CreateUnorderedAccessView(linear_.Get(),nullptr,&linearUav_),"linear UAV");
    }
    width_=width; height_=height;
    columns_=std::min(32u,width);
    rows_=std::min(height,std::clamp(static_cast<uint32_t>(std::floor(32.0*height/width+.5)),1u,64u));
    initialized_=false;
}

void ScreenPipeline::render(ID3D11Texture2D* source, ID3D11RenderTargetView* target,
                            double timestamp, uint64_t id, float strength) {
    if (!source || !target || !std::isfinite(strength) || strength<0 || strength>1 ||
        !std::isfinite(timestamp) || timestamp<0) throw std::runtime_error("screen_parameters");
    D3D11_TEXTURE2D_DESC desc{}; source->GetDesc(&desc);
    if (desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM || desc.ArraySize!=1 || desc.MipLevels!=1 || desc.SampleDesc.Count!=1)
        throw std::runtime_error("screen_format");
    resize(desc.Width,desc.Height);
    if (initialized_ && id==lastId_ && timestamp==lastTimestamp_) {
        refreshing_=true; draw(target,strength,lastElapsed_); return;
    }
    const bool reset=!initialized_ || timestamp<=lastTimestamp_ || id<=lastId_ || id!=lastId_+1;
    lastReset_=reset; refreshing_=false;
    if (apple_) {
        context_->CopyResource(captured_.Get(),source);
        apple_->process(captureView_.Get(),width_,height_,timestamp,true,reset);
        initialized_=true; lastTimestamp_=timestamp; lastId_=id; lastElapsed_=0;
        draw(target,strength,0); return;
    }
    const Constants values{width_,height_,columns_,rows_,reset?0:static_cast<float>(std::min(1.0,timestamp-lastTimestamp_)),
                           strength,0,reset?1u:0u};
    context_->UpdateSubresource(constants_.Get(),0,nullptr,&values,0,0);
    context_->CopyResource(captured_.Get(),source);
    context_->CSSetConstantBuffers(0,1,constants_.GetAddressOf());
    context_->CSSetShaderResources(0,1,captureView_.GetAddressOf());
    ID3D11UnorderedAccessView* uavs[]{linearUav_.Get(),totalsUav_.Get(),eventUav_.Get()};
    context_->CSSetUnorderedAccessViews(0,3,uavs,nullptr);
    context_->CSSetShader(convert_.Get(),nullptr,0); context_->Dispatch(columns_,rows_,1);
    context_->CSSetShader(control_.Get(),nullptr,0); context_->Dispatch(1,1,1);
    ID3D11ShaderResourceView* noView{}; ID3D11UnorderedAccessView* noUav[3]{};
    context_->CSSetShaderResources(0,1,&noView); context_->CSSetUnorderedAccessViews(0,3,noUav,nullptr);
    initialized_=true; lastTimestamp_=timestamp; lastId_=id; lastElapsed_=0;
    draw(target,strength,0);
}

void ScreenPipeline::draw(ID3D11RenderTargetView* target,float strength,float elapsed) {
    const Constants values{width_,height_,columns_,rows_,0,strength,elapsed,0};
    context_->UpdateSubresource(constants_.Get(),0,nullptr,&values,0,0);
    D3D11_VIEWPORT viewport{0,0,static_cast<float>(width_),static_cast<float>(height_),0,1};
    context_->RSSetViewports(1,&viewport);
    context_->RSSetState(nullptr); context_->OMSetBlendState(nullptr,nullptr,0xffffffff);
    context_->OMSetDepthStencilState(nullptr,0); context_->OMSetRenderTargets(1,&target,nullptr);
    context_->IASetInputLayout(nullptr); context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context_->VSSetShader(vertex_.Get(),nullptr,0);
    if (apple_) {
        apple_->bindOverlay(strength);
        context_->PSSetShaderResources(0,1,captureView_.GetAddressOf());
    } else {
        context_->PSSetShader(overlay_.Get(),nullptr,0);
        context_->PSSetConstantBuffers(0,1,constants_.GetAddressOf());
        ID3D11ShaderResourceView* views[]{captureView_.Get(),linearView_.Get(),eventView_.Get()};
        context_->PSSetShaderResources(0,3,views);
    }
    context_->Draw(3,0);
    ID3D11ShaderResourceView* noView[3]{};
    context_->PSSetShaderResources(0,3,noView); context_->OMSetRenderTargets(0,nullptr,nullptr);
    queueTelemetry(strength,elapsed);
}

void ScreenPipeline::refresh(ID3D11RenderTargetView* target,float strength,double secondsSinceCapture) {
    if (!initialized_) throw std::runtime_error("screen_no_cached_frame");
    if (!target || !std::isfinite(strength) || strength<0 || strength>1 ||
        !std::isfinite(secondsSinceCapture) || secondsSinceCapture<0) throw std::runtime_error("screen_parameters");
    // Bounded fade even when capture emits no frame. Never retrigger or reset the baseline on slider changes.
    lastElapsed_=std::max(lastElapsed_,static_cast<float>(std::min(apple_?86400.0:1.0,secondsSinceCapture)));
    if (apple_) apple_->process(captureView_.Get(),width_,height_,lastTimestamp_+lastElapsed_,false,false);
    refreshing_=true;
    draw(target,strength,lastElapsed_);
}

bool ScreenPipeline::enableTelemetry() noexcept {
    if (telemetryEnabled_) return true;
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth=32; desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0};
    for (auto& slot:readbacks_) {
        telemetryError_=device_->CreateBuffer(&desc,nullptr,&slot.buffer);
        if (SUCCEEDED(telemetryError_)) telemetryError_=device_->CreateQuery(&query,&slot.ready);
        if (FAILED(telemetryError_)) { disableTelemetry(); return false; }
    }
    telemetryEnabled_=true;
    return true;
}

void ScreenPipeline::disableTelemetry() noexcept {
    telemetryEnabled_=false;
    telemetryDropped_+=readbackCount_;
    readbackHead_=readbackCount_=0;
    for (auto& slot:readbacks_) { slot.buffer.Reset(); slot.ready.Reset(); }
}

void ScreenPipeline::queueTelemetry(float strength,float elapsed) noexcept {
    if (!telemetryEnabled_) return;
    ++drawId_;
    if (readbackCount_==readbacks_.size()) { ++telemetryDropped_; return; }
    auto& slot=readbacks_[(readbackHead_+readbackCount_)%readbacks_.size()];
    LARGE_INTEGER now{}; QueryPerformanceCounter(&now);
    slot.sample={drawId_,lastId_,now.QuadPart,lastTimestamp_,strength,elapsed,lastReset_,refreshing_,{}};
    slot.sample.algorithm=apple_?ScreenAlgorithm::AppleVfm:ScreenAlgorithm::SuddenHighlights;
    context_->CopyResource(slot.buffer.Get(),apple_?apple_->telemetry():event_.Get());
    context_->End(slot.ready.Get());
    ++readbackCount_;
}

bool ScreenPipeline::takeTelemetry(ScreenTelemetry& sample) noexcept {
    if (!telemetryEnabled_ || !readbackCount_) return false;
    auto& slot=readbacks_[readbackHead_];
    HRESULT hr=context_->GetData(slot.ready.Get(),nullptr,0,D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (hr==S_FALSE) return false;
    if (FAILED(hr)) { telemetryError_=hr; disableTelemetry(); return false; }
    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr=context_->Map(slot.buffer.Get(),0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
    if (hr==DXGI_ERROR_WAS_STILL_DRAWING) return false;
    if (FAILED(hr)) { telemetryError_=hr; disableTelemetry(); return false; }
    sample=slot.sample;
    std::memcpy(sample.event.data(),mapped.pData,sizeof(float)*sample.event.size());
    context_->Unmap(slot.buffer.Get(),0);
    if (!std::all_of(sample.event.begin(),sample.event.end(),[](float value) { return std::isfinite(value); })) {
        telemetryError_=E_UNEXPECTED; disableTelemetry(); return false;
    }
    readbackHead_=(readbackHead_+1)%readbacks_.size();
    --readbackCount_;
    return true;
}
}
