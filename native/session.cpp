#include "session.h"
#include <cmath>
#include <stdexcept>

namespace sunglasses {
namespace {
AutoParameters validate(AutoParameters parameters) {
    if (!std::isfinite(parameters.strength) || !std::isfinite(parameters.hold) || !std::isfinite(parameters.release) ||
        parameters.strength < 0 || parameters.strength > 1 || parameters.hold < 0 || parameters.hold > 5 ||
        parameters.release < .01f || parameters.release > 10) throw std::runtime_error("invalid_auto_parameters");
    return parameters;
}
}
AutomaticSession::AutomaticSession(ID3D11Device* device, ID3D11DeviceContext* context, AutoParameters parameters)
    : parameters_(validate(parameters)), gpu_(device, context) {}

void AutomaticSession::reset(AutoParameters parameters) {
    parameters_ = validate(parameters);
    clock_ = FrameClock{};
}

FrameResult AutomaticSession::process(ID3D11ShaderResourceView* source, FrameInfo frame) {
    if (!source) throw std::runtime_error("missing_source");
    D3D11_SHADER_RESOURCE_VIEW_DESC view{};
    source->GetDesc(&view);
    if (view.Format != DXGI_FORMAT_R32G32B32A32_FLOAT || view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D ||
        view.Texture2D.MostDetailedMip != 0 || view.Texture2D.MipLevels != 1) throw std::runtime_error("invalid_source_view");
    ComPtr<ID3D11Resource> resource;
    source->GetResource(&resource);
    ComPtr<ID3D11Texture2D> texture;
    check(resource.As(&texture), "Source texture");
    D3D11_TEXTURE2D_DESC description{};
    texture->GetDesc(&description);
    if (description.ArraySize != 1 || description.SampleDesc.Count != 1) throw std::runtime_error("invalid_source_texture");
    // Validate metadata before mutating the accepted clock. A GPU failure requires host recovery.
    FrameClock next = clock_;
    const auto timing = next.accept(frame.id, frame.timestampSeconds, description.Width, description.Height, frame.discontinuity);
    gpu_.resize(description.Width, description.Height);
    auto result = gpu_.process(source, timing, parameters_);
    clock_ = std::move(next);
    return {result, timing};
}
} // namespace sunglasses
