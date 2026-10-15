#pragma once
#include <windows.h>
#include <memory>

namespace sunglasses {
inline constexpr UINT CaseHiddenMessage=WM_APP+43;
inline constexpr UINT CaseOpenedMessage=WM_APP+44;
// Skins the existing modeless settings dialog; it does not own capture or settings values.
class GlassesCase {
public:
    GlassesCase();
    ~GlassesCase();
    GlassesCase(const GlassesCase&)=delete;
    GlassesCase& operator=(const GlassesCase&)=delete;
    bool attach(HWND dialog);
    void fitOnScreen();
    void open(RECT glasses,bool on,bool activate=true);
    void close(SIZE glasses,bool on);
    RECT glassesBounds() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
