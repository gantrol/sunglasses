#pragma once
#include "pipeline.h"
#include <memory>

namespace sunglasses {
enum class DiagnosticEvent {
    AppStart, AppStop, OverlayCreated, OverlayShow, OverlayHide, OverlayDestroyed,
    FilterStart, FilterStop, StrengthChanged, CaptureStatus, CaptureError,
    DisplayChanged, SessionChanged, PowerSuspend, Cloaked, Uncloaked, ForegroundChanged,
    Hooks, TelemetryUnavailable, TelemetryDropped, CaptureIdle,
    CaptureExclusionConfigured, CaptureExclusionFrame
};
struct DiagnosticFrame {
    uint64_t frameId=0, skipped=0, repeated=0;
    int64_t captureTicks=0, submitQpc=0, presentBeginQpc=0, presentEndQpc=0;
    double captureAgeMs=0, submitMs=0, presentWaitMs=0, idleSeconds=0;
    HRESULT presentResult=S_OK;
    bool refresh=false;
};

// Enable with SUNGLASSES_DIAGNOSTICS=1 before launch. All records contain only
// metadata. Construct/destroy on the UI thread; destroy after capture and UI exit.
class ScreenDiagnostics {
public:
    ScreenDiagnostics() noexcept;
    ~ScreenDiagnostics();
    ScreenDiagnostics(const ScreenDiagnostics&)=delete;
    ScreenDiagnostics& operator=(const ScreenDiagnostics&)=delete;
    bool enabled() const noexcept;
    void watch(HWND overlay,unsigned generation) noexcept;
    void event(DiagnosticEvent kind,unsigned generation,int64_t value=0,int64_t detail=0) noexcept;
    void frame(unsigned generation,const DiagnosticFrame& value) noexcept;
    void gpu(unsigned generation,const ScreenTelemetry& value) noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
