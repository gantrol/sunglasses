#include "capture.h"
#include "pipeline.h"
#include "diagnostics.h"
#include <dxgi1_6.h>
#include <dcomp.h>
#include <dwmapi.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Security.Authorization.AppCapabilityAccess.h>
#include <winrt/Windows.UI.h>
#include <windows.ui.interop.h>
#include <chrono>
#include <mutex>
#include <stdexcept>

using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;

namespace sunglasses {
bool captureTargetValid(const CaptureTarget& target) {
    MONITORINFO current{sizeof(current)};
    return target.monitor && GetMonitorInfoW(target.monitor,&current) && EqualRect(&current.rcMonitor,&target.bounds);
}
struct ScreenCapture::Shared {
    HWND overlay, owner;
    UINT message;
    CaptureTarget target;
    unsigned generation;
    ScreenDiagnostics* diagnostics;
    HANDLE wake=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    std::atomic<bool> quit=false, closed=false, overlayVisible=false;
    std::atomic<float> strength=1;
    mutable std::mutex mutex;
    ScreenCounters stats;
    Shared(HWND view,HWND control,UINT msg,const CaptureTarget& source,unsigned serial,float value,ScreenDiagnostics* log)
        : overlay(view),owner(control),message(msg),target(source),generation(serial),diagnostics(log),strength(value) {
        if (!wake) throw std::runtime_error("capture_event");
    }
    ~Shared() { CloseHandle(wake); }
    void post(ScreenStatus status) const {
        if (diagnostics) diagnostics->event(DiagnosticEvent::CaptureStatus,generation,static_cast<int64_t>(status));
        PostMessageW(owner,message,generation,static_cast<LPARAM>(status));
    }
};

namespace {
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::duration value) { return std::chrono::duration<double,std::milli>(value).count(); }
int64_t qpc() { LARGE_INTEGER value{}; QueryPerformanceCounter(&value); return value.QuadPart; }
struct WindowIdInterop {
    // Optional OS exports: resolve on the capture worker, without a hard import
    // that could prevent even hidden-recording/manual modes from starting.
    HMODULE module=LoadLibraryExW(L"Windows.UI.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    decltype(&::GetWindowIdFromWindow) fromWindow=module?
        reinterpret_cast<decltype(&::GetWindowIdFromWindow)>(GetProcAddress(module,"GetWindowIdFromWindow")):nullptr;
    decltype(&::GetWindowFromWindowId) toWindow=module?
        reinterpret_cast<decltype(&::GetWindowFromWindowId)>(GetProcAddress(module,"GetWindowFromWindowId")):nullptr;
    WindowIdInterop()=default;
    WindowIdInterop(const WindowIdInterop&)=delete;
    WindowIdInterop& operator=(const WindowIdInterop&)=delete;
    ~WindowIdInterop() { if (module) FreeLibrary(module); }
    winrt::Windows::UI::WindowId id(HWND window) const {
        if (!fromWindow || !toWindow || !IsWindow(window)) throw ScreenStatus::CaptureExclusionUnavailable;
        ABI::Windows::UI::WindowId value{};
        HWND restored=nullptr;
        if (FAILED(fromWindow(window,&value)) || !value.Value ||
            FAILED(toWindow(value,&restored)) || restored!=window)
            throw ScreenStatus::CaptureExclusionUnavailable;
        return {value.Value};
    }
};
uint64_t excludeAnalysisWindows(const GraphicsCaptureSession& session,const std::array<HWND,3>& windows) {
    try {
        const auto display=session.try_as<IDisplayGraphicsCaptureSession>();
        if (!display) throw ScreenStatus::CaptureExclusionUnavailable;
        const WindowIdInterop interop;
        auto excluded=winrt::single_threaded_vector<winrt::Windows::UI::WindowId>();
        for (const auto window:windows) excluded.Append(interop.id(window));
        const auto iteration=display.SetWindowExclusionList(excluded);
        const auto applied=display.GetWindowExclusionList();
        // Round-trip/list/configuration checks confirm the request, not that
        // composed overlay pixels are actually absent from the captured image.
        for (const auto required:excluded) {
            bool found=false;
            for (const auto actual:applied) if (actual.Value==required.Value) { found=true; break; }
            if (!found) throw ScreenStatus::CaptureExclusionUnavailable;
        }
        return iteration;
    } catch (const winrt::hresult_error&) { throw ScreenStatus::CaptureExclusionUnavailable; }
}
struct CaptureResources {
    Direct3D11CaptureFramePool pool{nullptr};
    GraphicsCaptureSession session{nullptr};
    ~CaptureResources() {
        // Events only signal a weak Shared reference; Close cannot race a D3D context operation.
        try { if (session) session.Close(); } catch (...) {}
        try { if (pool) pool.Close(); } catch (...) {}
    }
};
}

ScreenCapture::ScreenCapture(HWND overlay, HWND owner, UINT message,const CaptureTarget& target,unsigned generation,float strength,
                             ScreenDiagnostics* diagnostics)
    : shared_(std::make_shared<Shared>(overlay,owner,message,target,generation,strength,diagnostics)) {
    thread_=std::thread([state=shared_] {
        ScreenStatus terminal=ScreenStatus::Stopped;
        try {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            // Scope all WinRT/D3D objects before uninitializing the apartment.
            auto run=[&] {
                if (!GraphicsCaptureSession::IsSupported()) throw ScreenStatus::Unsupported;
                const std::weak_ptr<Shared> weak=state;
                // Obtain borderless access before creating/starting capture. Setting only
                // IsBorderRequired(false) can otherwise be silently ignored by Windows.
                try {
                    using winrt::Windows::Foundation::AsyncStatus;
                    using winrt::Windows::Security::Authorization::AppCapabilityAccess::AppCapabilityAccessStatus;
                    auto access=GraphicsCaptureAccess::RequestAccessAsync(GraphicsCaptureAccessKind::Borderless);
                    access.Completed([weak](auto const&,AsyncStatus) {
                        if (auto s=weak.lock()) SetEvent(s->wake);
                    });
                    while (access.Status()==AsyncStatus::Started) {
                        if (state->quit) { access.Cancel(); return; }
                        WaitForSingleObject(state->wake,50);
                    }
                    if (state->quit) return;
                    if (access.Status()==AsyncStatus::Canceled) throw ScreenStatus::BorderlessRequestFailed;
                    // GetResults also surfaces the HRESULT of a failed operation.
                    switch (access.GetResults()) {
                    case AppCapabilityAccessStatus::Allowed: break;
                    case AppCapabilityAccessStatus::DeniedByUser: throw ScreenStatus::BorderlessDeniedByUser;
                    case AppCapabilityAccessStatus::DeniedBySystem: throw ScreenStatus::BorderlessDeniedBySystem;
                    case AppCapabilityAccessStatus::NotDeclaredByApp: throw ScreenStatus::BorderlessNotDeclared;
                    default: throw ScreenStatus::BorderlessRequestFailed;
                    }
                } catch (const winrt::hresult_error& error) {
                    if (error.code()==E_NOINTERFACE || error.code()==E_NOTIMPL || error.code()==REGDB_E_CLASSNOTREG)
                        throw ScreenStatus::BorderlessUnavailable;
                    if (error.code()==HRESULT_FROM_WIN32(APPMODEL_ERROR_NO_PACKAGE))
                        throw ScreenStatus::BorderlessNotDeclared;
                    throw ScreenStatus::BorderlessRequestFailed;
                }
                ComPtr<IDXGIFactory1> factory;
                check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"DXGI factory");
                ComPtr<IDXGIAdapter1> adapter, selected;
                ComPtr<IDXGIOutput6> selectedOutput;
                for (UINT i=0;factory->EnumAdapters1(i,&adapter)==S_OK;++i,adapter.Reset()) {
                    ComPtr<IDXGIOutput> output;
                    for (UINT j=0;adapter->EnumOutputs(j,&output)==S_OK;++j,output.Reset()) {
                        DXGI_OUTPUT_DESC desc{}; check(output->GetDesc(&desc),"output desc");
                        if (desc.Monitor!=state->target.monitor) continue;
                        ComPtr<IDXGIOutput6> output6;
                        check(output.As(&output6),"SDR validation");
                        DXGI_OUTPUT_DESC1 color{}; check(output6->GetDesc1(&color),"output color");
                        if (color.ColorSpace!=DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) throw ScreenStatus::Hdr;
                        selected=adapter; selectedOutput=output6; break;
                    }
                    if (selected) break;
                }
                if (!selected) throw ScreenStatus::DisplayChanged;
                ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context;
                const D3D_FEATURE_LEVEL levels[]{D3D_FEATURE_LEVEL_11_0};
                check(D3D11CreateDevice(selected.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                    levels,1,D3D11_SDK_VERSION,&device,nullptr,&context),"screen D3D11 device");
                ComPtr<IDXGIDevice> dxgi; check(device.As(&dxgi),"DXGI device");
                winrt::com_ptr<IInspectable> inspectable;
                winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.Get(),inspectable.put()));
                auto runtimeDevice=inspectable.as<IDirect3DDevice>();
                auto interop=winrt::get_activation_factory<GraphicsCaptureItem,IGraphicsCaptureItemInterop>();
                GraphicsCaptureItem item{nullptr};
                if (!captureTargetValid(state->target)) throw ScreenStatus::DisplayChanged;
                winrt::check_hresult(interop->CreateForMonitor(state->target.monitor,winrt::guid_of<GraphicsCaptureItem>(),winrt::put_abi(item)));
                const auto size=item.Size();
                if (size.Width<=0 || size.Height<=0 || size.Width>8192 || size.Height>8192 ||
                    int64_t(size.Width)*size.Height>16'000'000) throw ScreenStatus::Unsupported;
                if (size.Width!=state->target.bounds.right-state->target.bounds.left ||
                    size.Height!=state->target.bounds.bottom-state->target.bounds.top) throw ScreenStatus::DisplayChanged;
                ComPtr<IDXGIFactory2> factory2; check(factory.As(&factory2),"composition factory");
                DXGI_SWAP_CHAIN_DESC1 description{};
                description.Width=static_cast<UINT>(size.Width); description.Height=static_cast<UINT>(size.Height);
                description.Format=DXGI_FORMAT_B8G8R8A8_UNORM; description.SampleDesc.Count=1;
                description.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT; description.BufferCount=2;
                description.Scaling=DXGI_SCALING_STRETCH; description.SwapEffect=DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
                description.AlphaMode=DXGI_ALPHA_MODE_PREMULTIPLIED;
                ComPtr<IDXGISwapChain1> swap;
                check(factory2->CreateSwapChainForComposition(device.Get(),&description,nullptr,&swap),"composition swapchain");
                ComPtr<IDXGIDevice1> latency; check(dxgi.As(&latency),"latency device");
                check(latency->SetMaximumFrameLatency(1),"frame latency");
                ComPtr<ID3D11Texture2D> back;
                check(swap->GetBuffer(0,IID_PPV_ARGS(&back)),"composition buffer");
                ComPtr<ID3D11RenderTargetView> target;
                check(device->CreateRenderTargetView(back.Get(),nullptr,&target),"composition target");
                const float clear[4]{}; context->ClearRenderTargetView(target.Get(),clear);
                check(swap->Present(1,0),"initial present");
                ComPtr<IDCompositionDevice> composition;
                check(DCompositionCreateDevice(dxgi.Get(),IID_PPV_ARGS(&composition)),"composition device");
                ComPtr<IDCompositionTarget> destination; ComPtr<IDCompositionVisual> visual;
                check(composition->CreateTargetForHwnd(state->overlay,TRUE,&destination),"composition window");
                check(composition->CreateVisual(&visual),"composition visual");
                check(visual->SetContent(swap.Get()),"visual content");
                check(destination->SetRoot(visual.Get()),"visual root");
                check(composition->Commit(),"composition commit");
                const bool apple=state->target.algorithm==ScreenAlgorithm::AppleVfm;
                ScreenPipeline pipeline(device.Get(),context.Get(),state->target.algorithm);
                auto* log=state->diagnostics;
                if (log && log->enabled() && !pipeline.enableTelemetry())
                    log->event(DiagnosticEvent::TelemetryUnavailable,state->generation,pipeline.telemetryError());
                uint64_t reportedTelemetryDrops=0;
                HRESULT reportedTelemetryError=pipeline.telemetryError();
                auto collectTelemetry=[&] {
                    if (!log) return;
                    if (!log->enabled()) pipeline.disableTelemetry();
                    ScreenTelemetry sample;
                    for (unsigned i=0;i<4 && pipeline.takeTelemetry(sample);++i) log->gpu(state->generation,sample);
                    if (pipeline.telemetryDropped()!=reportedTelemetryDrops) {
                        reportedTelemetryDrops=pipeline.telemetryDropped();
                        log->event(DiagnosticEvent::TelemetryDropped,state->generation,static_cast<int64_t>(reportedTelemetryDrops));
                    }
                    if (pipeline.telemetryError()!=reportedTelemetryError) {
                        reportedTelemetryError=pipeline.telemetryError();
                        log->event(DiagnosticEvent::TelemetryUnavailable,state->generation,reportedTelemetryError);
                    }
                };
                CaptureResources capture;
                capture.pool=Direct3D11CaptureFramePool::CreateFreeThreaded(runtimeDevice,
                    DirectXPixelFormat::B8G8R8A8UIntNormalized,2,size);
                capture.pool.FrameArrived([weak](auto const&,auto const&) {
                    if (auto s=weak.lock()) SetEvent(s->wake);
                });
                item.Closed([weak](auto const&,auto const&) {
                    if (auto s=weak.lock()) { s->closed=true; SetEvent(s->wake); }
                });
                capture.session=capture.pool.CreateCaptureSession(item);
                capture.session.IsCursorCaptureEnabled(false);
                if (state->target.includeInRecordings) {
                    // Previously exclusion was configured while the overlay was
                    // hidden; it entered the composition tree only after capture
                    // had started. Keep the surface transparent through setup.
                    check(composition->WaitForCommitCompletion(),"initial transparent composition");
                    state->post(ScreenStatus::OverlayReady);
                    const auto deadline=Clock::now()+std::chrono::seconds(5);
                    while (!state->overlayVisible) {
                        if (state->quit) return;
                        if (Clock::now()>=deadline) throw ScreenStatus::CaptureExclusionUnavailable;
                        WaitForSingleObject(state->wake,50);
                    }
                    if (state->quit) return;
                    if (!IsWindowVisible(state->overlay) || FAILED(DwmFlush()))
                        throw ScreenStatus::CaptureExclusionUnavailable;
                }
                // Our display capture omits the filter and controls; other recorders can
                // still include them. Hidden recordings use the existing global affinity.
                const uint64_t exclusionIteration=state->target.includeInRecordings?
                    excludeAnalysisWindows(capture.session,state->target.excludedWindows):0;
                if (state->target.includeInRecordings && log) {
                    int64_t visible=0;
                    for (size_t i=0;i<state->target.excludedWindows.size();++i)
                        if (IsWindowVisible(state->target.excludedWindows[i])) visible|=int64_t{1}<<i;
                    log->event(DiagnosticEvent::CaptureExclusionConfigured,state->generation,
                        static_cast<int64_t>(exclusionIteration),visible);
                }
                try {
                    capture.session.IsBorderRequired(false);
                    if (capture.session.IsBorderRequired()) throw ScreenStatus::BorderlessUnavailable;
                } catch (const winrt::hresult_error&) { throw ScreenStatus::BorderlessUnavailable; }
                if (state->quit) return;
                capture.session.StartCapture();
                const auto started=Clock::now();
                auto checkedDisplay=started;
                auto lastReceived=started;
                auto lastIdleReport=started;
                bool expired=false;
                uint64_t frameId=0, skipped=0, repeated=0;
                int64_t lastTicks=-1;
                float appliedStrength=-1;
                LARGE_INTEGER frequency{}; QueryPerformanceFrequency(&frequency);
                auto refreshMask=[&] {
                    if (!frameId) return;
                    const float requested=state->strength.load();
                    const double elapsed=milliseconds(Clock::now()-lastReceived)/1000;
                    if (apple || !expired || requested!=appliedStrength) {
                        const auto began=Clock::now();
                        const auto submitQpc=log?qpc():0;
                        pipeline.refresh(target.Get(),requested,elapsed);
                        const auto submitted=Clock::now();
                        if (state->quit) return;
                        const auto presentBegin=log?qpc():0;
                        const HRESULT hr=swap->Present(1,0);
                        const auto presentEnd=log?qpc():0;
                        if (log) log->frame(state->generation,{frameId-1,skipped,repeated,lastTicks,submitQpc,presentBegin,presentEnd,
                            1000.0*(double(submitQpc)/double(frequency.QuadPart)-double(lastTicks)/1e7),
                            milliseconds(submitted-began),1000.0*double(presentEnd-presentBegin)/double(frequency.QuadPart),elapsed,hr,true});
                        collectTelemetry();
                        check(hr,"refresh screen mask");
                        appliedStrength=requested;
                        expired=elapsed>=.32;
                    }
                };
                while (!state->quit) {
                    WaitForSingleObject(state->wake,apple?16:50);
                    if (state->quit) break;
                    collectTelemetry();
                    if (state->closed || !captureTargetValid(state->target)) throw ScreenStatus::DisplayChanged;
                    if (!frameId && Clock::now()-started>std::chrono::seconds(5)) throw ScreenStatus::NoFrames;
                    if (Clock::now()-checkedDisplay>std::chrono::seconds(1)) {
                        DXGI_OUTPUT_DESC1 current{};
                        check(selectedOutput->GetDesc1(&current),"monitor state");
                        if (current.ColorSpace!=DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) throw ScreenStatus::Hdr;
                        checkedDisplay=Clock::now();
                    }
                    auto frame=capture.pool.TryGetNextFrame();
                    if (!frame) {
                        if (log && Clock::now()-lastIdleReport>=std::chrono::milliseconds(500)) {
                            log->event(DiagnosticEvent::CaptureIdle,state->generation,
                                static_cast<int64_t>(milliseconds(Clock::now()-lastReceived)),static_cast<int64_t>(frameId));
                            lastIdleReport=Clock::now();
                        }
                        refreshMask();
                        continue;
                    }
                    // At most two queued frames; prefer the most recent and release the stale one.
                    if (auto newest=capture.pool.TryGetNextFrame()) { frame.Close(); frame=std::move(newest); ++skipped; }
                    if (state->target.includeInRecordings) {
                        try {
                            const auto configured=frame.try_as<IDirect3D11CaptureFrame3>();
                            if (!configured) throw ScreenStatus::CaptureExclusionUnavailable;
                            const auto frameIteration=configured.ConfigurationIteration();
                            if (frameIteration<exclusionIteration) {
                                ++skipped; frame.Close(); refreshMask(); continue;
                            }
                            if (!frameId && log)
                                log->event(DiagnosticEvent::CaptureExclusionFrame,state->generation,
                                    static_cast<int64_t>(frameIteration),static_cast<int64_t>(exclusionIteration));
                        } catch (const winrt::hresult_error&) { throw ScreenStatus::CaptureExclusionUnavailable; }
                    }
                    const auto content=frame.ContentSize();
                    if (content.Width!=size.Width || content.Height!=size.Height) throw ScreenStatus::DisplayChanged;
                    const int64_t ticks=frame.SystemRelativeTime().count();
                    if (ticks==lastTicks) { ++repeated; frame.Close(); refreshMask(); continue; }
                    lastTicks=ticks;
                    auto access=frame.Surface().as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
                    ComPtr<ID3D11Texture2D> texture;
                    winrt::check_hresult(access->GetInterface(IID_PPV_ARGS(&texture)));
                    LARGE_INTEGER now{}; QueryPerformanceCounter(&now);
                    const double age=1000.0*(double(now.QuadPart)/double(frequency.QuadPart)-double(ticks)/1e7);
                    const auto submitStart=Clock::now();
                    const auto submitQpc=log?qpc():0;
                    lastReceived=submitStart; expired=false;
                    appliedStrength=state->strength.load();
                    pipeline.render(texture.Get(),target.Get(),double(ticks)/1e7,frameId,appliedStrength);
                    const auto submitted=Clock::now();
                    if (state->quit) break;
                    const auto presentBegin=log?qpc():0;
                    const HRESULT presentResult=swap->Present(1,0);
                    const auto presentEnd=log?qpc():0;
                    const auto presented=Clock::now();
                    if (log) log->frame(state->generation,{frameId,skipped,repeated,ticks,submitQpc,presentBegin,presentEnd,
                        age,milliseconds(submitted-submitStart),1000.0*double(presentEnd-presentBegin)/double(frequency.QuadPart),0,presentResult,false});
                    collectTelemetry();
                    check(presentResult,"screen present");
                    frame.Close();
                    ++frameId;
                    {
                        std::lock_guard lock(state->mutex);
                        state->stats={frameId,skipped,repeated,ticks,age,milliseconds(submitted-submitStart),milliseconds(presented-submitted)};
                    }
                    if (frameId==1) state->post(ScreenStatus::Active);
                }
                // Transparent last surface before resources close; UI also hides immediately on Stop.
                collectTelemetry(); pipeline.disableTelemetry(); collectTelemetry();
                context->ClearRenderTargetView(target.Get(),clear);
                swap->Present(0,DXGI_PRESENT_DO_NOT_WAIT);
                destination->SetRoot(nullptr); composition->Commit();
            };
            try { run(); }
            catch (...) { winrt::uninit_apartment(); throw; }
            winrt::uninit_apartment();
        } catch (ScreenStatus status) { terminal=status; }
          catch (const winrt::hresult_error& error) {
              terminal=ScreenStatus::CaptureFailed;
              if (state->diagnostics) state->diagnostics->event(DiagnosticEvent::CaptureError,state->generation,error.code().value);
          }
          catch (...) { terminal=ScreenStatus::DeviceFailed; }
        state->post(state->quit ? ScreenStatus::Stopped : terminal);
    });
}
ScreenCapture::~ScreenCapture() { stop(); if (thread_.joinable()) thread_.join(); }
void ScreenCapture::stop() { shared_->quit=true; SetEvent(shared_->wake); }
void ScreenCapture::overlayShown() { shared_->overlayVisible=true; SetEvent(shared_->wake); }
void ScreenCapture::strength(float value) { shared_->strength=value; SetEvent(shared_->wake); }
ScreenCounters ScreenCapture::counters() const { std::lock_guard lock(shared_->mutex); return shared_->stats; }
}
