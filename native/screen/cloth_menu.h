#pragma once
#include <windows.h>
#include <memory>
#include <string>
#include <vector>

namespace sunglasses {
inline constexpr UINT ClothCommandMessage=WM_APP+45,ClothClosedMessage=WM_APP+46;
struct ClothAction {
    UINT command;
    std::wstring label;
    bool enabled=true,checked=false,compact=false,separator=false,checkable=false;
};
struct ClothState {
    std::wstring title,size;
    std::vector<ClothAction> actions;
};
class ClothMenu {
public:
    ClothMenu();
    ~ClothMenu();
    ClothMenu(const ClothMenu&)=delete;
    ClothMenu& operator=(const ClothMenu&)=delete;
    bool show(HWND owner,POINT point,const ClothState& state,bool animate);
    void dismiss(bool restoreFocus=false);
    bool processMessage(MSG& message);
    bool isVisible() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
