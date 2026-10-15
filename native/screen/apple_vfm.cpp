#include "apple_vfm.h"
#include "apple_vfm_source.h"
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sunglasses {
namespace {
struct Parameters {
    uint32_t width,height,columns,rows;
    uint32_t oldSamples,currentSample,newCapture,reset;
    float strength;
    float padding[3]{};
};
static_assert(sizeof(Parameters)==48);
// HLSL arrays of float4 have the same 16-byte stride as these records.
struct KernelData {
    float contrast[5*16][4]{}; // x=coefficient; y on first record=length
    float gamma[64][4]{};
};
KernelData kernelData() {
    KernelData result;
    // Parsing embedded text is bounded startup CPU work on the capture thread;
    // no disk I/O or model calls. Select the official 45-degree, 60-Hz rows.
    std::istringstream input(appleVfmKernelData);
    std::string line;
    unsigned lineIndex=0,selected=0;
    while (std::getline(input,line)) {
        if (lineIndex>=70 && (lineIndex-70)%7==4 && selected<5) {
            std::replace(line.begin(),line.end(),',',' ');
            std::istringstream row(line);
            float value=0; unsigned count=0;
            while (row>>value) {
                if (count>=16 || !std::isfinite(value)) throw std::runtime_error("apple_vfm_kernel");
                result.contrast[selected*16+count++][0]=value;
            }
            if (!count || !row.eof()) throw std::runtime_error("apple_vfm_kernel");
            result.contrast[selected*16][1]=static_cast<float>(count);
            ++selected;
        }
        ++lineIndex;
    }
    if (lineIndex!=105 || selected!=5) throw std::runtime_error("apple_vfm_kernel_count");
    double total=0;
    unsigned count=0;
    while (total/60<=.99) {
        if (count>=64) throw std::runtime_error("apple_vfm_gamma");
        const double t=double(count+1)/60;
        const double value=std::exp(-t/.15)*t/(.15*.15);
        result.gamma[count++][0]=static_cast<float>(value);
        total+=value;
    }
    if (count!=61) throw std::runtime_error("apple_vfm_gamma_length");
    return result;
}
}

AppleVfm::AppleVfm(ID3D11Device* device,ID3D11DeviceContext* context):device_(device),context_(context) {
    auto compile=[&](const char* entry,const char* profile) {
        ComPtr<ID3DBlob> code,errors;
        const D3D_SHADER_MACRO macros[]{{"APPLE_COMPUTE",profile[0]=='c'?"1":"0"},{nullptr,nullptr}};
        const HRESULT hr=D3DCompile(appleVfmSource,sizeof(appleVfmSource)-1,"apple_vfm.hlsl",macros,nullptr,entry,profile,
            D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_WARNINGS_ARE_ERRORS|D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
        if (FAILED(hr) && errors) throw std::runtime_error(static_cast<const char*>(errors->GetBufferPointer()));
        check(hr,"compile Apple VFM"); return code;
    };
    auto code=compile("Reduce","cs_5_0");
    check(device_->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&reduce_),"Apple reduction");
    code=compile("Control","cs_5_0");
    check(device_->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&control_),"Apple temporal core");
    code=compile("Overlay","ps_5_0");
    check(device_->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&overlay_),"Apple overlay");
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth=sizeof(Parameters); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    check(device_->CreateBuffer(&desc,nullptr,&constants_),"Apple parameters");
    const auto data=kernelData(); D3D11_SUBRESOURCE_DATA initial{&data,0,0};
    desc.ByteWidth=sizeof(KernelData); desc.Usage=D3D11_USAGE_IMMUTABLE;
    check(device_->CreateBuffer(&desc,&initial,&kernels_),"Apple kernels");
    auto buffer=[&](UINT bytes,UINT stride,ComPtr<ID3D11Buffer>& result,ComPtr<ID3D11UnorderedAccessView>& uav,bool view=false) {
        D3D11_BUFFER_DESC d{};
        d.ByteWidth=bytes; d.Usage=D3D11_USAGE_DEFAULT; d.StructureByteStride=stride;
        d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        d.BindFlags=D3D11_BIND_UNORDERED_ACCESS|(view?D3D11_BIND_SHADER_RESOURCE:0);
        check(device_->CreateBuffer(&d,nullptr,&result),"Apple buffer");
        check(device_->CreateUnorderedAccessView(result.Get(),nullptr,&uav),"Apple UAV");
    };
    buffer(32*64*4,4,totals_,totalsUav_);
    buffer((80+5*61*2)*4,4,history_,historyUav_);
    buffer(32,16,state_,stateUav_,true);
    buffer(32,16,telemetry_,telemetryUav_);
    check(device_->CreateShaderResourceView(state_.Get(),nullptr,&stateView_),"Apple state view");
}

void AppleVfm::process(ID3D11ShaderResourceView* source,uint32_t width,uint32_t height,
                       double timestamp,bool newCapture,bool reset) {
    const bool restart=reset || !initialized_ || width!=width_ || height!=height_ || timestamp-lastTime_>.5;
    // Late arriving capture after an idle redraw does not rewind the model.
    const double now=restart?timestamp:std::max(timestamp,lastTime_);
    uint32_t oldSamples=0,currentSample=1;
    if (restart) nextTick_=now;
    else {
        oldSamples=static_cast<uint32_t>(std::max(0.0,std::ceil((now-nextTick_)*60-1e-7)));
        currentSample=std::abs(now-(nextTick_+double(oldSamples)/60))<1e-7?1u:0u;
        if (oldSamples>30) throw std::runtime_error("apple_vfm_clock");
    }
    nextTick_+=double(oldSamples+currentSample)/60;
    lastTime_=now; width_=width; height_=height; initialized_=true;
    const uint32_t columns=std::min(32u,width);
    const uint32_t rows=std::min(height,std::clamp(static_cast<uint32_t>(std::floor(32.0*height/width+.5)),1u,64u));
    const Parameters values{width,height,columns,rows,oldSamples,currentSample,newCapture?1u:0u,restart?1u:0u,0,{}};
    context_->UpdateSubresource(constants_.Get(),0,nullptr,&values,0,0);
    ID3D11Buffer* buffers[]{constants_.Get(),kernels_.Get()};
    context_->CSSetConstantBuffers(0,2,buffers);
    context_->CSSetShaderResources(0,1,&source);
    ID3D11UnorderedAccessView* uavs[]{totalsUav_.Get(),historyUav_.Get(),stateUav_.Get(),telemetryUav_.Get()};
    context_->CSSetUnorderedAccessViews(0,4,uavs,nullptr);
    if (newCapture) { context_->CSSetShader(reduce_.Get(),nullptr,0); context_->Dispatch(columns,rows,1); }
    context_->CSSetShader(control_.Get(),nullptr,0); context_->Dispatch(1,1,1);
    ID3D11ShaderResourceView* none{}; ID3D11UnorderedAccessView* noUav[4]{};
    context_->CSSetShaderResources(0,1,&none); context_->CSSetUnorderedAccessViews(0,4,noUav,nullptr);
}

void AppleVfm::bindOverlay(float strength) {
    const Parameters values{width_,height_,0,0,0,0,0,0,strength,{}};
    context_->UpdateSubresource(constants_.Get(),0,nullptr,&values,0,0);
    context_->PSSetShader(overlay_.Get(),nullptr,0);
    context_->PSSetConstantBuffers(0,1,constants_.GetAddressOf());
    context_->PSSetShaderResources(1,1,stateView_.GetAddressOf());
}
}
