#pragma once
#include <windows.h>
#include <atomic>
#include <array>
#include <memory>
#include <thread>
#include "algorithm.h"

namespace sunglasses {
class ScreenDiagnostics;
// Append values: existing numbers also appear in diagnostic logs.
enum class ScreenStatus : unsigned { Starting, Active, Stopped, Unsupported, Hdr, CaptureFailed, DeviceFailed, DisplayChanged, NoFrames,
    BorderlessUnavailable, CaptureExclusionUnavailable, BorderlessDeniedByUser, BorderlessDeniedBySystem,
    BorderlessNotDeclared, BorderlessRequestFailed, OverlayReady };
struct CaptureTarget {
    HMONITOR monitor=nullptr;
    RECT bounds{};
    bool includeInRecordings=false;
    ScreenAlgorithm algorithm=ScreenAlgorithm::SuddenHighlights;
    // Stable app windows excluded only from our analysis when recordings include them.
    std::array<HWND,3> excludedWindows{};
};
bool captureTargetValid(const CaptureTarget& target);
struct ScreenCounters {
    uint64_t frames=0, skipped=0, repeated=0;
    int64_t captureTicks=0;
    double captureAgeMs=0, submitMs=0, presentWaitMs=0;
};
class ScreenCapture {
public:
    ScreenCapture(HWND overlay, HWND owner, UINT message, const CaptureTarget& target, unsigned generation, float strength,
                  ScreenDiagnostics* diagnostics=nullptr);
    ~ScreenCapture();
    ScreenCapture(const ScreenCapture&)=delete;
    ScreenCapture& operator=(const ScreenCapture&)=delete;
    void stop();
    // UI acknowledgment, after showing the worker's transparent initial surface.
    void overlayShown();
    void strength(float value);
    ScreenCounters counters() const;
private:
    struct Shared;
    std::shared_ptr<Shared> shared_;
    std::thread thread_;
};
}
