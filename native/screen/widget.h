#pragma once
#include <windows.h>
#include <memory>
#include <string>

namespace sunglasses {
inline constexpr UINT WidgetStrengthMessage=WM_APP+41;
inline constexpr UINT WidgetTrayUnavailableMessage=WM_APP+47;
inline constexpr int DefaultScreenStrength=50;
struct WidgetState {
    int strength=DefaultScreenStrength;
    bool on=false, busy=false, canToggle=false, error=false, pinned=false;
    std::wstring title, toggle, status, detail, strengthName, percent;
    std::wstring menu, settings, hide, restore, restoreCase, stop, exit;
    // Keep size labels distinct from the Windows SDK's small macro.
    std::wstring pin, size, sizeSmall, sizeNormal, sizeLarge;
};

// Native controls supply keyboard and accessibility semantics; GDI supplies the pixel skin.
// This is a control window, separate from the capture/DirectComposition overlay.
class GlassesWidget {
public:
    GlassesWidget();
    ~GlassesWidget();
    GlassesWidget(const GlassesWidget&)=delete;
    GlassesWidget& operator=(const GlassesWidget&)=delete;
    bool create(HINSTANCE instance,HWND controller);
    void update(const WidgetState& state);
    void show();
    void hide();
    void dock();
    void showFromCase(RECT bounds,bool activate);
    RECT bounds() const;
    bool processMessage(MSG& message);
    void fitOnScreen();
    bool hasTray() const;
    HWND window() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
