#include "recovery.h"
#include "recovery_source.h"
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sunglasses {
ScreenRecovery::ScreenRecovery(ID3D11Device* device,ID3D11DeviceContext* context)
    : device_(device),context_(context) {
    static_assert(sizeof(Constants)==80);
    const char* entries[]{"RecoveryControl","Solve","ReduceTiles","SpatialCone","Base",
        "SurroundRows","Candidates","MotionScores","ChooseMotion","PrepareDetail",
        "ChooseNew","Budget","ChooseBudget","Finish"};
    auto compile=[](const char* entry,const char* profile) {
        ComPtr<ID3DBlob> code,errors;
        const D3D_SHADER_MACRO macros[]{{"RECOVERY_COMPUTE",profile[0]=='c'?"1":"0"},{nullptr,nullptr}};
        HRESULT hr=D3DCompile(recoverySource,sizeof(recoverySource)-1,"screen_recovery.hlsl",macros,nullptr,
            entry,profile,D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_WARNINGS_ARE_ERRORS|D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,&code,&errors);
        if (FAILED(hr) && errors) throw std::runtime_error(static_cast<const char*>(errors->GetBufferPointer()));
        check(hr,"compile screen recovery");
        return code;
    };
    for (size_t i=0;i<stages_.size();++i) {
        auto code=compile(entries[i],"cs_5_0");
        check(device_->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&stages_[i]),"recovery shader");
    }
    auto code=compile("RecoveryOverlay","ps_5_0");
    check(device_->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&overlay_),"recovery overlay");
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth=sizeof(Constants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    check(device_->CreateBuffer(&desc,nullptr,&constants_),"recovery constants");
}

void ScreenRecovery::resize(uint32_t width,uint32_t height) {
    if (values_.width==width && values_.height==height) return;
    if (!width || !height || width>8192 || height>8192 || uint64_t(width)*height>16'000'000)
        throw std::runtime_error("screen_size");
    // The same assumed physical profile as the offline reference. Capture dimensions
    // determine aspect ratio; no unverified EDID luminance/distance is inferred.
    const double scale=27*.0254/std::hypot(double(width),double(height));
    const double horizontal=2*std::atan(width*scale/(2*.65))*180/3.141592653589793;
    const double vertical=2*std::atan(height*scale/(2*.65))*180/3.141592653589793;
    Constants next{};
    next.width=width; next.height=height;
    next.columns=std::min({width,96u,static_cast<uint32_t>(std::ceil(horizontal/4))});
    next.rows=std::min({height,64u,static_cast<uint32_t>(std::ceil(vertical/4))});
    next.radiusX=std::max(1u,static_cast<uint32_t>(std::nearbyint(.5*width/horizontal)));
    next.radiusY=std::max(1u,static_cast<uint32_t>(std::nearbyint(.5*height/vertical)));
    next.motionX=std::max(1u,static_cast<uint32_t>(std::ceil(.25*width/horizontal)));
    next.motionY=std::max(1u,static_cast<uint32_t>(std::ceil(.25*height/vertical)));
    next.groupsX=(width+7)/8; next.groupsY=(height+7)/8;
    next.horizontalStep=static_cast<float>(.02*horizontal/next.columns);
    next.verticalStep=static_cast<float>(.02*vertical/next.rows);
    // Bounded allocation, no per-frame creation and no image readback. Allocation
    // failures propagate to the capture owner, which removes the overlay.
    auto texture=[&](Texture& target) {
        target={};
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width=width; desc.Height=height; desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        check(device_->CreateTexture2D(&desc,nullptr,&target.resource),"recovery texture");
        check(device_->CreateShaderResourceView(target.resource.Get(),nullptr,&target.view),"recovery texture view");
        check(device_->CreateUnorderedAccessView(target.resource.Get(),nullptr,&target.uav),"recovery texture UAV");
        const float zero[4]{}; context_->ClearUnorderedAccessViewFloat(target.uav.Get(),zero);
    };
    for (auto& target:history_) texture(target);
    for (auto& target:detail_) texture(target);
    for (auto& target:work_) texture(target);
    texture(base_); texture(output_);
    auto buffer=[&](Buffer& target,uint32_t count,uint32_t stride) {
        target={};
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth=count*stride; desc.StructureByteStride=stride;
        desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        check(device_->CreateBuffer(&desc,nullptr,&target.resource),"recovery buffer");
        check(device_->CreateShaderResourceView(target.resource.Get(),nullptr,&target.view),"recovery buffer view");
        check(device_->CreateUnorderedAccessView(target.resource.Get(),nullptr,&target.uav),"recovery buffer UAV");
        const UINT zero[4]{}; context_->ClearUnorderedAccessViewUint(target.uav.Get(),zero);
    };
    buffer(required_,next.columns*next.rows,16); buffer(tiles_,next.columns*next.rows,16);
    buffer(scores_,std::max(next.groupsX*next.groupsY,(2*next.motionX+1)*(2*next.motionY+1)),16);
    buffer(control_,3,16); buffer(indices_,width*height+1,4);
    values_=next; previous_=0;
}

void ScreenRecovery::dispatch(size_t stage,uint32_t x,uint32_t y,
    const std::array<ID3D11ShaderResourceView*,12>& inputs,
    const std::array<ID3D11UnorderedAccessView*,8>& outputs) {
    context_->CSSetShaderResources(0,static_cast<UINT>(inputs.size()),inputs.data());
    context_->CSSetUnorderedAccessViews(0,static_cast<UINT>(outputs.size()),outputs.data(),nullptr);
    context_->CSSetShader(stages_[stage].Get(),nullptr,0); context_->Dispatch(x,y,1);
    const std::array<ID3D11ShaderResourceView*,12> noViews{};
    const std::array<ID3D11UnorderedAccessView*,8> noOutputs{};
    context_->CSSetShaderResources(0,static_cast<UINT>(noViews.size()),noViews.data());
    context_->CSSetUnorderedAccessViews(0,static_cast<UINT>(noOutputs.size()),noOutputs.data(),nullptr);
}

void ScreenRecovery::process(ID3D11ShaderResourceView* encoded,ID3D11ShaderResourceView* linear,
                            ID3D11ShaderResourceView* detector,float dt,float strength,bool reset) {
    values_.dt=reset?0:dt; values_.strength=strength; values_.reset=reset?1u:0u;
    values_.cachedStrength=strength; values_.elapsed=0;
    context_->UpdateSubresource(constants_.Get(),0,nullptr,&values_,0,0);
    context_->CSSetConstantBuffers(0,1,constants_.GetAddressOf());
    const auto next=1-previous_;
    using Inputs=std::array<ID3D11ShaderResourceView*,12>;
    using Outputs=std::array<ID3D11UnorderedAccessView*,8>;
    auto run=[&](size_t stage,uint32_t x,uint32_t y,Inputs inputs,Outputs outputs) { dispatch(stage,x,y,inputs,outputs); };
    run(0,1,1,{nullptr,nullptr,detector},{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,control_.uav.Get()});
    run(1,values_.groupsX,values_.groupsY,{encoded,linear,nullptr,history_[previous_].view.Get(),nullptr,nullptr,nullptr,nullptr,nullptr,control_.view.Get()},
        {work_[0].uav.Get()});
    run(2,values_.columns,values_.rows,{nullptr,nullptr,nullptr,nullptr,nullptr,work_[0].view.Get()},
        {nullptr,nullptr,nullptr,nullptr,required_.uav.Get()});
    run(3,values_.columns,values_.rows,{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,required_.view.Get()},
        {nullptr,nullptr,nullptr,nullptr,tiles_.uav.Get()});
    run(4,values_.groupsX,values_.groupsY,{encoded,linear,nullptr,nullptr,nullptr,work_[0].view.Get(),nullptr,tiles_.view.Get(),nullptr,control_.view.Get()},
        {base_.uav.Get(),history_[next].uav.Get()});
    run(5,values_.groupsX,values_.groupsY,{nullptr,linear},{work_[0].uav.Get()});
    const UINT zero[4]{}; context_->ClearUnorderedAccessViewUint(indices_.uav.Get(),zero);
    run(6,values_.groupsX,values_.groupsY,{nullptr,linear,nullptr,nullptr,detail_[previous_].view.Get(),work_[0].view.Get()},
        {work_[1].uav.Get(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,indices_.uav.Get()});
    run(7,2*values_.motionX+1,2*values_.motionY+1,{nullptr,nullptr,nullptr,history_[previous_].view.Get(),nullptr,work_[1].view.Get(),nullptr,nullptr,nullptr,nullptr,indices_.view.Get()},
        {nullptr,nullptr,nullptr,nullptr,nullptr,scores_.uav.Get()});
    run(8,1,1,{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,tiles_.view.Get(),scores_.view.Get(),nullptr,indices_.view.Get()},
        {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,control_.uav.Get()});
    run(9,values_.groupsX,values_.groupsY,{nullptr,linear,nullptr,history_[previous_].view.Get(),detail_[previous_].view.Get(),work_[1].view.Get(),base_.view.Get(),nullptr,nullptr,control_.view.Get()},
        {work_[0].uav.Get(),nullptr,nullptr,nullptr,nullptr,scores_.uav.Get()});
    run(10,1,1,{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,scores_.view.Get()},
        {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,control_.uav.Get()});
    run(11,values_.groupsX,values_.groupsY,{nullptr,nullptr,nullptr,nullptr,nullptr,work_[0].view.Get(),nullptr,nullptr,nullptr,control_.view.Get()},
        {work_[1].uav.Get(),nullptr,nullptr,nullptr,nullptr,scores_.uav.Get()});
    run(12,1,1,{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,scores_.view.Get()},
        {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,control_.uav.Get()});
    run(13,values_.groupsX,values_.groupsY,{encoded,linear,nullptr,history_[previous_].view.Get(),detail_[previous_].view.Get(),work_[1].view.Get(),base_.view.Get(),nullptr,nullptr,control_.view.Get()},
        {output_.uav.Get(),nullptr,detail_[next].uav.Get()});
    previous_=next;
}

void ScreenRecovery::bindOverlay(float strength,float elapsed) {
    values_.strength=strength; values_.elapsed=elapsed;
    context_->UpdateSubresource(constants_.Get(),0,nullptr,&values_,0,0);
    context_->PSSetConstantBuffers(0,1,constants_.GetAddressOf());
    context_->PSSetShader(overlay_.Get(),nullptr,0);
    context_->PSSetShaderResources(11,1,output_.view.GetAddressOf());
}
void ScreenRecovery::unbindOverlay() {
    ID3D11ShaderResourceView* none{}; context_->PSSetShaderResources(11,1,&none);
}
}
