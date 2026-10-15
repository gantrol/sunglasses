#include "cloth_menu.h"
#include "pixel_cloth.h"
#include "cloth_pull.h"
#include "pixel_controls.h"
#include <commctrl.h>
#include <dwmapi.h>
#include <uxtheme.h>
#include <windowsx.h>
#include <algorithm>
#include <cmath>

namespace sunglasses {
namespace {
constexpr wchar_t ClothClass[]=L"Sunglasses.LensCloth";
constexpr wchar_t PullClass[]=L"Sunglasses.LensCloth.Pull";
constexpr UINT_PTR PullTimer=0x5349;
void borderless(HWND window) {
    const DWMNCRENDERINGPOLICY policy=DWMNCRP_DISABLED;
    const COLORREF border=DWMWA_COLOR_NONE;
    const BOOL disabled=TRUE;
    DwmSetWindowAttribute(window,DWMWA_NCRENDERING_POLICY,&policy,sizeof(policy));
    DwmSetWindowAttribute(window,DWMWA_BORDER_COLOR,&border,sizeof(border));
    DwmSetWindowAttribute(window,DWMWA_TRANSITIONS_FORCEDISABLED,&disabled,sizeof(disabled));
}
}
struct ClothMenu::Impl {
    HWND hwnd=nullptr,pullWindow=nullptr,owner=nullptr,previousFocus=nullptr,previousForeground=nullptr;
    HFONT font=nullptr;
    HBRUSH surfaceBrush=nullptr;
    ClothState state;
    std::vector<HWND> controls;
    std::vector<RECT> placements;
    RECT sizeLabel{};
    POINT position{};
    int scale=100,width=300,height=320;
    int anchor=0;
    bool visible=false,pulling=false,positioning=false,handingOff=false;
    ClothPull pull;
    ULONGLONG started=0;
    int px(int value) const { return MulDiv(value,scale,100); }
    ~Impl() {
        if (IsWindow(pullWindow)) DestroyWindow(pullWindow);
        if (IsWindow(hwnd)) DestroyWindow(hwnd);
        if (font) DeleteObject(font);
        if (surfaceBrush) DeleteObject(surfaceBrush);
    }
    void dismiss(bool restore) {
        if (!visible) return;
        const bool active=GetForegroundWindow()==hwnd || GetForegroundWindow()==pullWindow;
        visible=false; pulling=false; KillTimer(hwnd,PullTimer);
        if (pullWindow) ShowWindow(pullWindow,SW_HIDE);
        ShowWindow(hwnd,SW_HIDE);
        pull.reset();
        PostMessageW(owner,ClothClosedMessage,0,0);
        if (restore && active && IsWindow(previousForeground) && IsWindowVisible(previousForeground)) {
            SetForegroundWindow(previousForeground);
            if (IsWindow(previousFocus)) SetFocus(previousFocus);
        }
    }
    void focusFirst() const {
        for (auto control:controls) if (IsWindowEnabled(control)) { SetFocus(control); return; }
    }
    void navigate(HWND from,int direction) const {
        const auto current=std::find(controls.begin(),controls.end(),from);
        int index=current==controls.end()?0:static_cast<int>(current-controls.begin());
        for (size_t i=0;i<controls.size();++i) {
            index=(index+direction+static_cast<int>(controls.size()))%static_cast<int>(controls.size());
            if (IsWindowEnabled(controls[static_cast<size_t>(index)])) { SetFocus(controls[static_cast<size_t>(index)]); return; }
        }
    }
    static LRESULT CALLBACK buttonProc(HWND control,UINT message,WPARAM wp,LPARAM lp,UINT_PTR,DWORD_PTR data) {
        auto& self=*reinterpret_cast<Impl*>(data);
        if (message==WM_GETDLGCODE) return DefSubclassProc(control,message,wp,lp)|DLGC_WANTARROWS|DLGC_WANTMESSAGE;
        if (message==WM_KEYDOWN) {
            if (wp==VK_ESCAPE) { self.dismiss(true); return 0; }
            if (wp==VK_RETURN) { SendMessageW(control,BM_CLICK,0,0); return 0; }
            if (wp==VK_TAB || wp==VK_UP || wp==VK_DOWN || wp==VK_LEFT || wp==VK_RIGHT) {
                const int direction=(wp==VK_UP || wp==VK_LEFT || (wp==VK_TAB && (GetKeyState(VK_SHIFT)&0x8000)))?-1:1;
                self.navigate(control,direction); return 0;
            }
        }
        if (message==WM_MOUSEMOVE) {
            TRACKMOUSEEVENT track{sizeof(track),TME_LEAVE,control,0}; TrackMouseEvent(&track);
            InvalidateRect(control,nullptr,FALSE);
        }
        if (message==WM_MOUSELEAVE || message==WM_SETFOCUS || message==WM_KILLFOCUS || message==WM_UPDATEUISTATE)
            InvalidateRect(control,nullptr,FALSE);
        return DefSubclassProc(control,message,wp,lp);
    }
    bool measure() {
        NONCLIENTMETRICSW metrics{sizeof(metrics)};
        const UINT dpi=GetDpiForWindow(owner);
        if (!SystemParametersInfoForDpi(SPI_GETNONCLIENTMETRICS,sizeof(metrics),&metrics,0,dpi)) return false;
        const auto next=CreateFontIndirectW(&metrics.lfMenuFont);
        if (!next) return false;
        if (font) DeleteObject(font); font=next;
        scale=std::max(MulDiv(static_cast<int>(dpi),100,96),MulDiv(static_cast<int>(std::abs(metrics.lfMenuFont.lfHeight)),100,14));
        const auto dc=GetDC(owner); const auto old=SelectObject(dc,font);
        width=px(280);
        for (const auto& action:state.actions) {
            SIZE size{}; GetTextExtentPoint32W(dc,action.label.c_str(),static_cast<int>(action.label.size()),&size);
            width=std::max(width,static_cast<int>(size.cx)*(action.compact?3:1)+px(action.compact?76:72));
        }
        SelectObject(dc,old); ReleaseDC(owner,dc);
        const int margin=px(18),row=px(32),gap=px(4);
        int top=px(46),column=0;
        placements.clear();
        for (const auto& action:state.actions) {
            if (action.separator) top+=px(9);
            if (action.compact) {
                if (column==0) { sizeLabel={margin,top,width-margin,top+px(22)}; top+=px(22); }
                const int cell=(width-margin*2-gap*2)/3;
                placements.push_back({margin+column*(cell+gap),top,margin+column*(cell+gap)+cell,top+row});
                if (++column==3) { column=0; top+=row+gap; }
            } else {
                placements.push_back({margin,top,width-margin,top+row}); top+=row+gap;
            }
        }
        height=top+px(20); return true;
    }
    void shape() {
        RECT rect{}; GetClientRect(hwnd,&rect);
        auto region=pixel::clothRegion(rect,std::max(1,px(3)));
        if (region && !SetWindowRgn(hwnd,region,FALSE)) DeleteObject(region);
    }
    void finishPull() {
        const bool activate=GetForegroundWindow()==pullWindow;
        KillTimer(hwnd,PullTimer); pulling=false; handingOff=true;
        // Paint the complete menu under the last opaque fabric frame before
        // removing the animation surface; never expose an unpainted window.
        for (auto control:controls) ShowWindow(control,SW_SHOWNA);
        ShowWindow(hwnd,SW_SHOWNOACTIVATE);
        RedrawWindow(hwnd,nullptr,nullptr,RDW_INVALIDATE|RDW_ALLCHILDREN|RDW_UPDATENOW);
        if (activate) { SetForegroundWindow(hwnd); focusFirst(); }
        if (pullWindow) ShowWindow(pullWindow,SW_HIDE);
        handingOff=false; pull.reset();
    }
    void frame() {
        const float progress=std::clamp(static_cast<float>(GetTickCount64()-started)/360.0f,0.0f,1.0f);
        if (!pull.present(pullWindow,position,anchor,std::max(1,px(3)),progress) || progress>=1) finishPull();
    }
    bool preparePull(DWORD affinity) {
        if (!pullWindow) {
            WNDCLASSW klass{}; klass.lpfnWndProc=pullProcedure; klass.hInstance=GetModuleHandleW(nullptr);
            klass.lpszClassName=PullClass; klass.hCursor=LoadCursorW(nullptr,IDC_ARROW);
            if (!RegisterClassW(&klass) && GetLastError()!=ERROR_CLASS_ALREADY_EXISTS) return false;
            pullWindow=CreateWindowExW(WS_EX_TOOLWINDOW|WS_EX_LAYERED,PullClass,state.title.c_str(),WS_POPUP,
                position.x,position.y,width,height,hwnd,nullptr,klass.hInstance,this);
            if (!pullWindow) return false;
            borderless(pullWindow);
        }
        if (!SetWindowDisplayAffinity(pullWindow,affinity) || !pull.prepare(width,height)) return false;
        positioning=true;
        const bool ready=pull.present(pullWindow,position,anchor,std::max(1,px(3)),0);
        positioning=false;
        return ready;
    }
    bool show(HWND target,POINT point,const ClothState& next,bool animate) {
        dismiss(false); owner=target; state=next;
        previousFocus=GetFocus(); previousForeground=GetForegroundWindow();
        if (!hwnd) {
            WNDCLASSW klass{}; klass.lpfnWndProc=procedure; klass.hInstance=GetModuleHandleW(nullptr);
            klass.lpszClassName=ClothClass; klass.hCursor=LoadCursorW(nullptr,IDC_ARROW);
            if (!RegisterClassW(&klass) && GetLastError()!=ERROR_CLASS_ALREADY_EXISTS) return false;
            hwnd=CreateWindowExW(WS_EX_TOOLWINDOW|WS_EX_CONTROLPARENT,ClothClass,L"",WS_POPUP|WS_CLIPCHILDREN,
                point.x,point.y,1,1,owner,nullptr,klass.hInstance,this);
            if (!hwnd) return false;
        }
        borderless(hwnd);
        for (auto control:controls) DestroyWindow(control);
        controls.clear();
        if (!measure()) return false;
        MONITORINFO monitor{sizeof(monitor)};
        if (!GetMonitorInfoW(MonitorFromPoint(point,MONITOR_DEFAULTTONEAREST),&monitor)) return false;
        const auto& work=monitor.rcWork;
        // Native menus remain the fallback when large system text cannot fit on this monitor.
        if (width>work.right-work.left || height>work.bottom-work.top) return false;
        const int left=std::clamp(point.x-width+px(24),work.left,work.right-width);
        const int top=std::clamp(point.y,work.top,work.bottom-height);
        position={left,top}; anchor=point.x-left;
        for (size_t i=0;i<state.actions.size();++i) {
            const auto& action=state.actions[i]; const auto& rect=placements[i];
            const DWORD type=action.compact?BS_AUTORADIOBUTTON:action.checkable?BS_AUTOCHECKBOX:BS_OWNERDRAW;
            const DWORD group=action.compact && (i==0 || !state.actions[i-1].compact)?WS_GROUP:0;
            const auto button=CreateWindowExW(0,WC_BUTTONW,action.label.c_str(),WS_CHILD|type|group|WS_TABSTOP,
                rect.left,rect.top,rect.right-rect.left,rect.bottom-rect.top,hwnd,
                reinterpret_cast<HMENU>(static_cast<UINT_PTR>(action.command)),GetModuleHandleW(nullptr),nullptr);
            if (!button) { for (auto child:controls) DestroyWindow(child); controls.clear(); return false; }
            controls.push_back(button); SendMessageW(button,WM_SETFONT,reinterpret_cast<WPARAM>(font),FALSE);
            if (action.checkable || action.compact) SendMessageW(button,BM_SETCHECK,action.checked?BST_CHECKED:BST_UNCHECKED,0);
            EnableWindow(button,action.enabled); SetWindowSubclass(button,buttonProc,1,reinterpret_cast<DWORD_PTR>(this));
            if (!pixel::skinButton(button)) {
                for (auto child:controls) DestroyWindow(child);
                controls.clear(); return false;
            }
            SetWindowTheme(button,L"",L"");
        }
        SetWindowTextW(hwnd,state.title.c_str());
        const bool hideFocus=(SendMessageW(owner,WM_QUERYUISTATE,0,0)&UISF_HIDEFOCUS)!=0;
        SendMessageW(hwnd,WM_UPDATEUISTATE,MAKEWPARAM(hideFocus?UIS_SET:UIS_CLEAR,UISF_HIDEFOCUS),0);
        DWORD affinity=WDA_NONE;
        if (GetWindowDisplayAffinity(owner,&affinity) && !SetWindowDisplayAffinity(hwnd,affinity)) return false;
        BOOL enabled=FALSE; const auto colors=pixel::palette();
        if (surfaceBrush) DeleteObject(surfaceBrush);
        surfaceBrush=CreateSolidBrush(pixel::clothPalette(colors).surface);
        pulling=animate && !colors.highContrast && SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION,0,&enabled,0) && enabled;
        positioning=true;
        SetWindowPos(hwnd,nullptr,left,top,width,height,SWP_NOZORDER|SWP_NOACTIVATE);
        positioning=false;
        shape();
        visible=true;
        // A clamped menu near a screen edge opens directly rather than pulling
        // fabric from a point disconnected from the temple.
        pulling=pulling && std::abs(top-point.y)<=px(8) && anchor>=0 && anchor<=width;
        if (pulling && preparePull(affinity)) {
            ShowWindow(pullWindow,SW_SHOWNORMAL); SetForegroundWindow(pullWindow); SetFocus(pullWindow);
            started=GetTickCount64();
            if (!SetTimer(hwnd,PullTimer,15,nullptr)) finishPull();
        } else {
            finishPull(); SetForegroundWindow(hwnd); focusFirst();
        }
        return true;
    }
    void paint() const {
        PAINTSTRUCT paint{}; const auto target=BeginPaint(hwnd,&paint);
        RECT rect{}; GetClientRect(hwnd,&rect);
        pixel::paintBuffered(target,rect,[&](HDC output) {
            const auto colors=pixel::palette();
            const auto cloth=pixel::clothPalette(colors);
            pixel::fill(output,rect,cloth.surface);
            pixel::drawCloth(output,rect,colors,std::max(1,px(3)));
            const auto fontBefore=SelectObject(output,font); SetBkMode(output,TRANSPARENT);
            SetTextColor(output,cloth.ink);
            RECT title{px(20),px(12),width-px(20),px(40)};
            DrawTextW(output,state.title.c_str(),-1,&title,DT_SINGLELINE|DT_VCENTER|DT_NOPREFIX);
            auto label=sizeLabel; DrawTextW(output,state.size.c_str(),-1,&label,DT_SINGLELINE|DT_VCENTER|DT_NOPREFIX);
            SelectObject(output,fontBefore);
        });
        EndPaint(hwnd,&paint);
    }
    void button(const DRAWITEMSTRUCT& draw) const {
        const auto found=std::find_if(state.actions.begin(),state.actions.end(),[&](const auto& a) { return a.command==draw.CtlID; });
        if (found==state.actions.end()) return;
        const auto colors=pixel::palette(); const auto cloth=pixel::clothPalette(colors); const auto& action=*found;
        POINT cursor{}; GetCursorPos(&cursor);
        const bool hover=WindowFromPoint(cursor)==draw.hwndItem;
        const bool selected=action.enabled && (action.checked || (draw.itemState&ODS_SELECTED)!=0 || hover);
        const COLORREF background=selected?cloth.active:cloth.surface;
        const COLORREF foreground=!action.enabled?cloth.muted:colors.highContrast && selected?
            GetSysColor(COLOR_HIGHLIGHTTEXT):cloth.ink;
        RECT rect=draw.rcItem; pixel::fill(draw.hDC,rect,background);
        if (action.compact) {
            const auto brush=CreateSolidBrush(action.checked?cloth.ink:cloth.hem); FrameRect(draw.hDC,&rect,brush); DeleteObject(brush);
        }
        const auto previous=SelectObject(draw.hDC,font);
        SetBkMode(draw.hDC,TRANSPARENT); SetTextColor(draw.hDC,foreground);
        RECT label=rect; label.left+=px(action.compact?3:24); label.right-=px(6);
        DrawTextW(draw.hDC,action.label.c_str(),-1,&label,DT_SINGLELINE|DT_VCENTER|DT_END_ELLIPSIS|DT_NOPREFIX|(action.compact?DT_CENTER:DT_LEFT));
        if (action.checked && !action.compact) {
            const int p=std::max(1,px(2)),x=px(7),y=(rect.top+rect.bottom)/2;
            pixel::fill(draw.hDC,{x,y,x+p,y+p*2},foreground);
            pixel::fill(draw.hDC,{x+p,y+p,x+p*2,y+p*3},foreground);
            pixel::fill(draw.hDC,{x+p*2,y-p*2,x+p*3,y+p*2},foreground);
        }
        SelectObject(draw.hDC,previous);
        if ((draw.itemState&ODS_FOCUS)!=0 && (SendMessageW(draw.hwndItem,WM_QUERYUISTATE,0,0)&UISF_HIDEFOCUS)==0) {
            InflateRect(&rect,-px(2),-px(2)); DrawFocusRect(draw.hDC,&rect);
        }
    }
    static LRESULT CALLBACK pullProcedure(HWND window,UINT message,WPARAM wp,LPARAM lp) {
        auto self=reinterpret_cast<Impl*>(GetWindowLongPtrW(window,GWLP_USERDATA));
        if (message==WM_NCCREATE) {
            self=static_cast<Impl*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);
            SetWindowLongPtrW(window,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(self));
        }
        if (!self) return DefWindowProcW(window,message,wp,lp);
        switch (message) {
        case WM_ERASEBKGND: return 1;
        case WM_NCCALCSIZE: case WM_NCPAINT: return 0;
        case WM_NCACTIVATE: return TRUE;
        case WM_ACTIVATE:
            if (LOWORD(wp)==WA_INACTIVE && !self->handingOff) self->dismiss(false);
            return 0;
        case WM_LBUTTONUP: if (self->visible && self->pulling) self->finishPull(); return 0;
        case WM_CLOSE: self->dismiss(true); return 0;
        case WM_THEMECHANGED: case WM_SETTINGCHANGE: self->dismiss(false); return 0;
        case WM_DPICHANGED: if (!self->positioning) self->dismiss(false); return 0;
        case WM_NCDESTROY: self->pullWindow=nullptr; break;
        }
        return DefWindowProcW(window,message,wp,lp);
    }
    static LRESULT CALLBACK procedure(HWND window,UINT message,WPARAM wp,LPARAM lp) {
        auto self=reinterpret_cast<Impl*>(GetWindowLongPtrW(window,GWLP_USERDATA));
        if (message==WM_NCCREATE) {
            self=static_cast<Impl*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);
            self->hwnd=window; SetWindowLongPtrW(window,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(self));
        }
        if (!self) return DefWindowProcW(window,message,wp,lp);
        switch (message) {
        case WM_ERASEBKGND: return 1;
        case WM_NCCALCSIZE: case WM_NCPAINT: return 0;
        case WM_NCACTIVATE: return TRUE;
        case WM_PAINT: self->paint(); return 0;
        case WM_SIZE: self->shape(); return 0;
        case WM_TIMER: if (wp==PullTimer && self->pulling) { self->frame(); return 0; } break;
        case WM_DPICHANGED:
            if (self->positioning) return 0;
            self->dismiss(false);
            return 0;
        case WM_THEMECHANGED: case WM_SETTINGCHANGE: self->dismiss(false); return 0;
        case WM_ACTIVATE:
            if (LOWORD(wp)==WA_INACTIVE && !self->handingOff && !self->pulling) self->dismiss(false);
            return 0;
        case WM_KEYDOWN: if (wp==VK_ESCAPE) { self->dismiss(true); return 0; } break;
        case WM_CLOSE: self->dismiss(true); return 0;
        case WM_DRAWITEM:
            pixel::drawBufferedItem(*reinterpret_cast<DRAWITEMSTRUCT*>(lp),[&](const auto& item) { self->button(item); });
            return TRUE;
        case WM_CTLCOLORBTN:
            SetBkColor(reinterpret_cast<HDC>(wp),pixel::clothPalette(pixel::palette()).surface);
            return reinterpret_cast<LRESULT>(self->surfaceBrush);
        case WM_COMMAND:
            if (LOWORD(wp)==IDCANCEL) { self->dismiss(true); return 0; }
            for (const auto& action:self->state.actions) if (action.command==LOWORD(wp) && action.enabled) {
                self->dismiss(true); PostMessageW(self->owner,ClothCommandMessage,action.command,0); break;
            }
            return 0;
        case WM_DESTROY: KillTimer(window,PullTimer); return 0;
        case WM_NCDESTROY: self->hwnd=nullptr; break;
        }
        return DefWindowProcW(window,message,wp,lp);
    }
};
ClothMenu::ClothMenu():impl_(std::make_unique<Impl>()) {}
ClothMenu::~ClothMenu()=default;
bool ClothMenu::show(HWND owner,POINT point,const ClothState& state,bool animate) { return impl_->show(owner,point,state,animate); }
void ClothMenu::dismiss(bool restoreFocus) { impl_->dismiss(restoreFocus); }
bool ClothMenu::processMessage(MSG& message) {
    if (!impl_->visible) return false;
    const bool fromPull=message.hwnd==impl_->pullWindow;
    if (!fromPull && message.hwnd!=impl_->hwnd && !IsChild(impl_->hwnd,message.hwnd)) return false;
    if (message.message==WM_KEYDOWN || message.message==WM_SYSKEYDOWN)
        SendMessageW(impl_->hwnd,WM_CHANGEUISTATE,MAKEWPARAM(UIS_CLEAR,UISF_HIDEFOCUS),0);
    if (impl_->pulling && (message.message==WM_KEYDOWN || message.message==WM_SYSKEYDOWN)) {
        if (message.wParam==VK_ESCAPE) { impl_->dismiss(true); return true; }
        impl_->finishPull(); message.hwnd=GetFocus();
    } else if (fromPull) return false;
    return IsDialogMessageW(impl_->hwnd,&message);
}
bool ClothMenu::isVisible() const { return impl_->visible; }
}
