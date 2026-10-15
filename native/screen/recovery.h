#pragma once
#include "../automatic.h"

namespace sunglasses {
// GPU implementation of pixel-cone recovery and motion-propagated detail protection.
// Uses a nominal 27-inch SDR display at 0.65 m, 200/0.2 nit; not monitor calibration.
// All image state stays on the caller's D3D11 immediate context.
class ScreenRecovery {
public:
    ScreenRecovery(ID3D11Device* device, ID3D11DeviceContext* context);
    void resize(uint32_t width,uint32_t height);
    void process(ID3D11ShaderResourceView* encoded, ID3D11ShaderResourceView* linear,
                 ID3D11ShaderResourceView* detector, float dt,float strength,bool reset);
    void bindOverlay(float strength,float elapsed);
    void unbindOverlay();
private:
    struct Texture {
        ComPtr<ID3D11Texture2D> resource;
        ComPtr<ID3D11ShaderResourceView> view;
        ComPtr<ID3D11UnorderedAccessView> uav;
    };
    struct Buffer {
        ComPtr<ID3D11Buffer> resource;
        ComPtr<ID3D11ShaderResourceView> view;
        ComPtr<ID3D11UnorderedAccessView> uav;
    };
    struct Constants {
        uint32_t width=0,height=0,columns=0,rows=0;
        uint32_t radiusX=0,radiusY=0,motionX=0,motionY=0;
        uint32_t groupsX=0,groupsY=0,reset=0,reserved=0;
        float dt=0,strength=0,horizontalStep=0,verticalStep=0;
        float cachedStrength=0,elapsed=0,padding[2]{};
    } values_;
    void dispatch(size_t stage,uint32_t x,uint32_t y,
                  const std::array<ID3D11ShaderResourceView*,12>& inputs,
                  const std::array<ID3D11UnorderedAccessView*,8>& outputs);
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11Buffer> constants_;
    std::array<ComPtr<ID3D11ComputeShader>,14> stages_;
    ComPtr<ID3D11PixelShader> overlay_;
    std::array<Texture,2> history_,detail_,work_;
    Texture base_,output_;
    Buffer required_,tiles_,scores_,control_,indices_;
    uint32_t previous_=0;
};
}
