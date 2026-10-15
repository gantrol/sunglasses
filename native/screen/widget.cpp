#include "widget.h"
#include "resource.h"
#include "pixel_theme.h"
#include "pixel_controls.h"
#include "pixel_menu.h"
#include "pixel_glasses.h"
#include "pixel_cloth.h"
#include "cloth_menu.h"
#include <windowsx.h>
#include <commctrl.h>
#include <shellapi.h>
#include <uxtheme.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace sunglasses {
namespace {
constexpr wchar_t WidgetClass[]=L"Sunglasses.PixelGlasses";
constexpr UINT TrayMessage=WM_APP+42;
constexpr UINT_PTR TrayRetryTimer=0x534A;
constexpr int Toggle=401, Level=402, Value=403, Menu=404, Hide=405;
constexpr int Restore=410;
constexpr int SizeSmall=420,SizeNormal=421,SizeLarge=422;
constexpr int CanvasWidth=pixel::GlassesWidth,CanvasHeight=pixel::GlassesHeight;
using pixel::fill;
}

struct GlassesWidget::Impl {
    HWND hwnd=nullptr,controller=nullptr;
    HWND toggle=nullptr,slider=nullptr,value=nullptr,menu=nullptr,hide=nullptr,tooltip=nullptr;
    HFONT font=nullptr,numberFont=nullptr;
    HICON icon=nullptr;
    HICON windowSmallIcon=nullptr,windowLargeIcon=nullptr;
    UINT windowIconDpi=0;
    WidgetState state;
    UINT taskbarMessage=RegisterWindowMessageW(L"TaskbarCreated");
    bool tray=false,trayVersion4=false,hover=false,menuOpen=false,highContrast=false;
    int width=480,height=160,minWidth=360;
    COLORREF frame{},lens{},ink{},accent{};
    ClothMenu cloth;
    bool placingFromCase=false;

    ~Impl() {
        if (IsWindow(hwnd)) DestroyWindow(hwnd);
        if (font) DeleteObject(font);
        if (numberFont) DeleteObject(numberFont);
        if (icon) DestroyIcon(icon);
        if (windowSmallIcon) DestroyIcon(windowSmallIcon);
        if (windowLargeIcon) DestroyIcon(windowLargeIcon);
    }
    int x(int coordinate) const { return MulDiv(coordinate,width,CanvasWidth); }
    int y(int coordinate) const { return MulDiv(coordinate,height,CanvasHeight); }
    RECT rect(int left,int top,int right,int bottom) const { return {x(left),y(top),x(right),y(bottom)}; }
    bool revealed() const {
        const auto focus=GetFocus(); const auto captured=GetCapture();
        return hover || menuOpen || state.error || state.busy || highContrast ||
            focus==hwnd || IsChild(hwnd,focus) || captured==hwnd || IsChild(hwnd,captured);
    }
    void repaint() const {
        RedrawWindow(hwnd,nullptr,nullptr,RDW_INVALIDATE|RDW_ALLCHILDREN);
    }
    void pointer(HWND target,bool entered) {
        if (entered) {
            TRACKMOUSEEVENT tracking{sizeof(tracking),TME_LEAVE,target,0}; TrackMouseEvent(&tracking);
            if (!hover) { hover=true; repaint(); }
        } else {
            POINT p{}; GetCursorPos(&p); const auto under=WindowFromPoint(p);
            hover=under==hwnd || IsChild(hwnd,under); repaint();
        }
    }
    static LRESULT CALLBACK childProc(HWND window,UINT message,WPARAM wp,LPARAM lp,UINT_PTR,DWORD_PTR data) {
        auto& self=*reinterpret_cast<Impl*>(data);
        if (message==WM_GETDLGCODE && wp==VK_RETURN)
            return DefSubclassProc(window,message,wp,lp)|DLGC_WANTMESSAGE;
        if (message==WM_KEYDOWN && wp==VK_RETURN) {
            if (window!=self.slider && window!=self.value) SendMessageW(window,BM_CLICK,0,0);
            return 0; // Enter on the strength control must not toggle the filter.
        }
        if (message==WM_MOUSEMOVE) self.pointer(window,true);
        if (message==WM_MOUSELEAVE) self.pointer(window,false);
        if (message==WM_SETFOCUS || message==WM_KILLFOCUS || message==WM_CAPTURECHANGED) {
            const auto result=DefSubclassProc(window,message,wp,lp); self.repaint(); return result;
        }
        if (message==WM_KEYDOWN && wp==VK_ESCAPE) { PostMessageW(self.controller,WM_COMMAND,IDC_STOP,0); return 0; }
        if (message==WM_KEYDOWN && wp==VK_F10 && (GetKeyState(VK_SHIFT)&0x8000)) { self.popup(); return 0; }
        if (message==WM_MOUSEWHEEL && window==self.slider) {
            const int change=GET_WHEEL_DELTA_WPARAM(wp)>0?5:-5;
            const int next=std::clamp(self.state.strength+change,0,100);
            SendMessageW(self.controller,WidgetStrengthMessage,static_cast<WPARAM>(next),0); return 0;
        }
        return DefSubclassProc(window,message,wp,lp);
    }
    void nativeControl(HWND& child,const wchar_t* klass,int id,DWORD style) {
        child=CreateWindowExW(0,klass,L"",WS_CHILD|WS_VISIBLE|style,0,0,1,1,hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),GetModuleHandleW(nullptr),nullptr);
        if (child) SetWindowSubclass(child,childProc,1,reinterpret_cast<DWORD_PTR>(this));
    }
    void colors() {
        const auto theme=pixel::palette(state.on);
        highContrast=theme.highContrast; frame=theme.frame; lens=theme.lens; ink=theme.ink; accent=theme.accent;
    }
    void layout() {
        RECT client{}; GetClientRect(hwnd,&client); width=std::max(1L,client.right); height=std::max(1L,client.bottom);
        auto region=pixel::glassesPolygon(client,pixel::GlassesOutline);
        if (region && !SetWindowRgn(hwnd,region,TRUE)) DeleteObject(region); // Windows owns successful regions.
        const auto move=[&](HWND child,RECT r) { SetWindowPos(child,nullptr,r.left,r.top,r.right-r.left,r.bottom-r.top,SWP_NOACTIVATE|SWP_NOZORDER); };
        move(toggle,rect(80,52,226,136)); move(value,rect(380,51,526,95));
        move(slider,rect(375,101,534,141)); move(menu,rect(555,29,594,66)); move(hide,rect(6,29,45,66));
        NONCLIENTMETRICSW metrics{sizeof(metrics)};
        SystemParametersInfoForDpi(SPI_GETNONCLIENTMETRICS,sizeof(metrics),&metrics,0,GetDpiForWindow(hwnd));
        // Respect the user's message-font size as a lower bound when scaling the shaped window.
        minWidth=std::max(MulDiv(360,static_cast<int>(GetDpiForWindow(hwnd)),96),static_cast<int>(std::abs(metrics.lfMessageFont.lfHeight))*28);
        auto logfont=metrics.lfMessageFont;
        logfont.lfHeight=-std::max(static_cast<int>(std::abs(logfont.lfHeight)),y(19));
        auto replacement=CreateFontIndirectW(&logfont);
        if (replacement) {
            for (auto child:{toggle,slider,value,menu,hide}) SendMessageW(child,WM_SETFONT,reinterpret_cast<WPARAM>(replacement),FALSE);
            if (font) DeleteObject(font); font=replacement;
        }
        logfont.lfHeight=-std::max(static_cast<int>(std::abs(logfont.lfHeight)),y(32));
        auto number=CreateFontIndirectW(&logfont);
        if (number) { if (numberFont) DeleteObject(numberFont); numberFont=number; }
        SendMessageW(slider,TBM_SETTHUMBLENGTH,std::max(12,y(24)),0);
        repaint();
    }
    HICON makeIcon() const {
        BITMAPINFO bitmap{}; bitmap.bmiHeader={sizeof(BITMAPINFOHEADER),32,-32,1,32,BI_RGB,0,0,0,0,0};
        void* memory=nullptr; const auto color=CreateDIBSection(nullptr,&bitmap,DIB_RGB_COLORS,&memory,nullptr,0);
        if (!color) return nullptr;
        auto pixels=static_cast<DWORD*>(memory); std::fill(pixels,pixels+32*32,0u);
        const auto argb=[](COLORREF colorValue) {
            return 0xff000000u|(static_cast<DWORD>(GetRValue(colorValue))<<16)|
                (static_cast<DWORD>(GetGValue(colorValue))<<8)|GetBValue(colorValue);
        };
        const auto theme=pixel::palette(state.on);
        for (int row=10;row<22;++row) for (int col=2;col<30;++col) {
            const bool left=col<=13,right=col>=18;
            if ((!left&&!right&&row>13) || ((col<4||col>27)&&row>16)) continue;
            const bool inside=row>=13 && row<=18 && ((col>=5&&col<=10)||(col>=21&&col<=26));
            pixels[row*32+col]=argb(inside?(state.on?theme.accent:theme.frame):theme.edge);
        }
        const std::array<BYTE,128> maskBits{};
        const auto mask=CreateBitmap(32,32,1,1,maskBits.data());
        ICONINFO info{TRUE,0,0,mask,color}; const auto result=CreateIconIndirect(&info);
        if (mask) DeleteObject(mask); DeleteObject(color); return result;
    }
    NOTIFYICONDATAW trayData() const {
        NOTIFYICONDATAW data{sizeof(data)}; data.hWnd=hwnd; data.uID=1; return data;
    }
    void updateWindowIcons() {
        const UINT dpi=std::max(GetDpiForWindow(hwnd),GetDpiForWindow(controller));
        if (windowIconDpi==dpi && windowSmallIcon && windowLargeIcon) return;
        HICON smallIcon=nullptr,largeIcon=nullptr;
        const auto instance=GetModuleHandleW(nullptr);
        const auto resource=MAKEINTRESOURCEW(IDI_SUNGLASSES);
        const auto smallResult=LoadIconWithScaleDown(instance,resource,GetSystemMetricsForDpi(SM_CXSMICON,dpi),
            GetSystemMetricsForDpi(SM_CYSMICON,dpi),&smallIcon);
        const auto largeResult=LoadIconWithScaleDown(instance,resource,GetSystemMetricsForDpi(SM_CXICON,dpi),
            GetSystemMetricsForDpi(SM_CYICON,dpi),&largeIcon);
        if (FAILED(smallResult) || FAILED(largeResult)) {
            if (smallIcon) DestroyIcon(smallIcon); if (largeIcon) DestroyIcon(largeIcon); return;
        }
        for (const auto target:{hwnd,controller}) if (IsWindow(target)) {
            SendMessageW(target,WM_SETICON,ICON_SMALL,reinterpret_cast<LPARAM>(smallIcon));
            SendMessageW(target,WM_SETICON,ICON_BIG,reinterpret_cast<LPARAM>(largeIcon));
        }
        if (windowSmallIcon) DestroyIcon(windowSmallIcon);
        if (windowLargeIcon) DestroyIcon(windowLargeIcon);
        windowSmallIcon=smallIcon; windowLargeIcon=largeIcon; windowIconDpi=dpi;
    }
    void updateTray(bool rebuild=false) {
        updateWindowIcons();
        auto replacement=makeIcon();
        if (replacement) {
            if (icon) DestroyIcon(icon); icon=replacement;
        }
        auto data=trayData(); data.uFlags=NIF_MESSAGE|NIF_ICON|NIF_TIP|NIF_SHOWTIP;
        data.uCallbackMessage=TrayMessage; data.hIcon=icon?icon:LoadIconW(nullptr,IDI_APPLICATION);
        const auto tip=state.title+L" · "+state.detail;
        wcsncpy_s(data.szTip,tip.c_str(),_TRUNCATE);
        if (rebuild) tray=false;
        if (tray && !Shell_NotifyIconW(NIM_MODIFY,&data)) tray=false;
        if (!tray) {
            trayVersion4=false;
            tray=data.hIcon && (Shell_NotifyIconW(NIM_ADD,&data)!=FALSE || Shell_NotifyIconW(NIM_MODIFY,&data)!=FALSE);
            if (tray) {
                data.uVersion=NOTIFYICON_VERSION_4;
                trayVersion4=Shell_NotifyIconW(NIM_SETVERSION,&data)!=FALSE;
            }
        }
        if (tray) KillTimer(hwnd,TrayRetryTimer);
        else {
            // Explorer may broadcast TaskbarCreated before its notification area is ready.
            SetTimer(hwnd,TrayRetryTimer,3000,nullptr);
            PostMessageW(controller,WidgetTrayUnavailableMessage,0,0);
        }
    }
    void show() {
        cloth.dismiss(false);
        ShowWindow(hwnd,SW_RESTORE); fit(); SetForegroundWindow(hwnd); SetFocus(toggle); repaint();
    }
    void conceal() {
        cloth.dismiss(false);
        // Check the recovery entry before hiding a window excluded from task switching.
        updateTray();
        if (tray) ShowWindow(hwnd,SW_HIDE);
        else PostMessageW(controller,WM_COMMAND,IDC_WIDGET_SETTINGS,0);
    }
    bool showing() const { return IsWindowVisible(hwnd) && !IsIconic(hwnd); }
    const std::wstring& restoreLabel() const { return IsWindowVisible(controller)?state.restoreCase:state.restore; }
    void fit() {
        if (IsIconic(hwnd)) return;
        RECT r{}; GetWindowRect(hwnd,&r); MONITORINFO info{sizeof(info)};
        if (!GetMonitorInfoW(MonitorFromWindow(hwnd,MONITOR_DEFAULTTONEAREST),&info)) return;
        const auto& work=info.rcWork;
        const int w=std::min({r.right-r.left,work.right-work.left,(work.bottom-work.top)*3}),h=std::max(1,w/3);
        SetWindowPos(hwnd,nullptr,std::clamp(r.left,work.left,work.right-w),std::clamp(r.top,work.top,work.bottom-h),w,h,SWP_NOZORDER|SWP_NOACTIVATE);
    }
    std::array<int,3> presetWidths() const {
        const int normal=std::max(minWidth,MulDiv(480,static_cast<int>(GetDpiForWindow(hwnd)),96));
        return {std::max(minWidth,normal*3/4),normal,normal*5/4};
    }
    void popup(bool atCursor=false) {
        POINT point{}; GetCursorPos(&point);
        if (!atCursor) {
            RECT button{}; GetWindowRect(menu,&button);
            point={(button.left+button.right)/2,button.top+(button.bottom-button.top)*4/5};
        }
        const auto widths=presetWidths();
        ClothState content{state.menu,state.size,{
            {IDC_SWITCH,state.toggle,state.canToggle},
            {IDC_STOP,state.stop,state.on||state.busy},
            {IDC_WIDGET_SETTINGS,state.settings,true,false,false,true},
            {IDC_WIDGET_PIN,state.pin,true,state.pinned,false,false,true},
            {SizeSmall,state.sizeSmall,widths[0]<widths[1],width==widths[0] && widths[0]<widths[1],true},
            {SizeNormal,state.sizeNormal,true,width==widths[1],true},
            {SizeLarge,state.sizeLarge,true,width==widths[2],true},
            {static_cast<UINT>(showing()?Hide:Restore),showing()?state.hide:restoreLabel(),true,false,false,true},
            {IDC_WIDGET_EXIT,state.exit}
        }};
        menuOpen=true; repaint();
        if (cloth.show(hwnd,point,content,IsWindowVisible(hwnd)!=FALSE && !atCursor)) return;
        nativePopup(atCursor);
    }
    void nativePopup(bool atCursor) {
        pixel::Menu context(hwnd); if (!context.handle()) return;
        context.add(IDC_SWITCH,state.toggle,state.canToggle); context.add(IDC_STOP,state.stop,state.on||state.busy);
        context.add(0,L"");
        context.add(IDC_WIDGET_SETTINGS,state.settings);
        context.add(IDC_WIDGET_PIN,state.pin,true,state.pinned);
        if (const auto sizes=context.submenu(state.size)) {
            const auto widths=presetWidths();
            context.add(SizeSmall,state.sizeSmall,widths[0]<widths[1],width==widths[0] && widths[0]<widths[1],sizes);
            context.add(SizeNormal,state.sizeNormal,true,width==widths[1],sizes);
            context.add(SizeLarge,state.sizeLarge,true,width==widths[2],sizes);
        }
        context.add(showing()?Hide:Restore,showing()?state.hide:restoreLabel());
        context.add(0,L""); context.add(IDC_WIDGET_EXIT,state.exit);
        POINT point{}; GetCursorPos(&point);
        if (!atCursor) {
            RECT button{}; GetWindowRect(menu,&button); point={button.left,button.bottom};
        }
        menuOpen=true; repaint(); SetForegroundWindow(hwnd);
        const UINT command=static_cast<UINT>(TrackPopupMenu(context.handle(),TPM_RETURNCMD|TPM_RIGHTBUTTON,point.x,point.y,0,hwnd,nullptr));
        menuOpen=false;
        commandSelected(command);
        PostMessageW(hwnd,WM_NULL,0,0); pointer(hwnd,false);
    }
    void commandSelected(UINT command) {
        if (command==Restore) PostMessageW(controller,WM_COMMAND,IDC_WIDGET_RESTORE,0);
        else if (command==Hide) conceal();
        else if (command>=SizeSmall && command<=SizeLarge) {
            const int next=presetWidths()[command-SizeSmall];
            RECT current{}; GetWindowRect(hwnd,&current);
            SetWindowPos(hwnd,nullptr,current.left+(width-next)/2,current.top+(height-next/3)/2,next,next/3,SWP_NOZORDER|SWP_NOACTIVATE);
            fit();
        }
        else if (command) PostMessageW(controller,WM_COMMAND,command,0);
    }
    void tool(HWND target,const std::wstring& label) const {
        TOOLINFOW info{sizeof(info)}; info.uFlags=TTF_IDISHWND|TTF_SUBCLASS;
        info.hwnd=hwnd; info.uId=reinterpret_cast<UINT_PTR>(target); info.lpszText=const_cast<wchar_t*>(label.c_str());
        SendMessageW(tooltip,TTM_UPDATETIPTEXTW,0,reinterpret_cast<LPARAM>(&info));
    }
    void update(const WidgetState& next) {
        cloth.dismiss(false);
        const bool errorChanged=next.error && (!state.error || state.detail!=next.detail);
        const bool trayChanged=state.on!=next.on || state.detail!=next.detail || state.title!=next.title;
        state=next;
        SetWindowTextW(hwnd,state.title.c_str()); SetWindowTextW(toggle,state.toggle.c_str());
        SetWindowTextW(menu,state.menu.c_str()); SetWindowTextW(hide,state.hide.c_str());
        SetWindowTextW(slider,state.strengthName.c_str()); SetWindowTextW(value,state.percent.c_str());
        EnableWindow(toggle,state.canToggle);
        if (SendMessageW(slider,TBM_GETPOS,0,0)!=state.strength) SendMessageW(slider,TBM_SETPOS,TRUE,state.strength);
        tool(toggle,state.detail); tool(slider,state.strengthName); tool(menu,state.menu); tool(hide,state.hide);
        colors(); repaint();
        if (trayChanged || !tray) updateTray();
        if (errorChanged) {
            auto data=trayData(); data.uFlags=NIF_INFO; data.dwInfoFlags=NIIF_ERROR;
            wcsncpy_s(data.szInfoTitle,state.title.c_str(),_TRUNCATE); wcsncpy_s(data.szInfo,state.detail.c_str(),_TRUNCATE);
            if (!tray || !Shell_NotifyIconW(NIM_MODIFY,&data)) PostMessageW(controller,WM_COMMAND,IDC_WIDGET_SETTINGS,0);
        }
    }
    void paint() {
        PAINTSTRUCT ps{}; const auto dc=BeginPaint(hwnd,&ps);
        RECT client{}; GetClientRect(hwnd,&client);
        pixel::paintBuffered(dc,client,[&](HDC memory) {
            fill(memory,client,frame);
            pixel::drawGlasses(memory,client,pixel::palette(state.on));
        });
        EndPaint(hwnd,&ps);
    }
    void draw(const DRAWITEMSTRUCT& item) const {
        const bool hinge=item.CtlID==Menu || item.CtlID==Hide;
        fill(item.hDC,item.rcItem,hinge?frame:lens);
        if (!revealed() && item.CtlID!=Menu) return;
        const auto dc=item.hDC; const int w=item.rcItem.right,h=item.rcItem.bottom;
        const auto textColor=hinge && highContrast?GetSysColor(COLOR_WINDOW):ink;
        SetBkMode(dc,TRANSPARENT); SetTextColor(dc,textColor);
        const auto previous=SelectObject(dc,item.CtlID==Value?numberFont:font);
        if (item.CtlID==Toggle) {
            const int unit=std::max(2,y(4)),cx=w/2,cy=h/3,radius=std::max(8,y(17));
            const auto pen=CreatePen(PS_SOLID,unit,textColor); const auto oldPen=SelectObject(dc,pen);
            const auto oldBrush=SelectObject(dc,GetStockObject(NULL_BRUSH));
            // A compact power symbol; there are no flashing state animations.
            Arc(dc,cx-radius,cy-radius,cx+radius,cy+radius,cx-radius/2,cy-radius,cx+radius/2,cy-radius);
            MoveToEx(dc,cx,cy-radius-unit,nullptr); LineTo(dc,cx,cy);
            SelectObject(dc,oldBrush); SelectObject(dc,oldPen); DeleteObject(pen);
            RECT label{0,h*2/3,w,h}; DrawTextW(dc,state.status.c_str(),-1,&label,DT_CENTER|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS|DT_NOPREFIX);
        } else if (item.CtlID==Value) {
            RECT label=item.rcItem; DrawTextW(dc,state.percent.c_str(),-1,&label,DT_CENTER|DT_VCENTER|DT_SINGLELINE|DT_NOPREFIX);
        } else if (item.CtlID==Menu) {
            RECT clothRect{w/5,h/5,w*4/5,h*4/5};
            pixel::drawCloth(dc,clothRect,pixel::palette(state.on),std::max(1,x(2)),false);
        } else if (item.CtlID==Hide) {
            fill(dc,{w/3,h/2-std::max(1,y(2)),w*2/3,h/2+std::max(1,y(2))},textColor);
        }
        SelectObject(dc,previous);
        if ((item.itemState&ODS_FOCUS)!=0 && (item.itemState&ODS_NOFOCUSRECT)==0) {
            RECT focus=item.rcItem; InflateRect(&focus,-2,-2); DrawFocusRect(dc,&focus);
        }
    }
    LRESULT drawSlider(NMCUSTOMDRAW& draw) const {
        if (draw.dwDrawStage!=CDDS_PREPAINT) return CDRF_DODEFAULT;
        RECT client{}; GetClientRect(slider,&client); fill(draw.hdc,client,lens);
        if (!revealed()) return CDRF_SKIPDEFAULT;
        RECT channel{},thumb{}; SendMessageW(slider,TBM_GETCHANNELRECT,0,reinterpret_cast<LPARAM>(&channel));
        SendMessageW(slider,TBM_GETTHUMBRECT,0,reinterpret_cast<LPARAM>(&thumb));
        const int center=(thumb.top+thumb.bottom)/2,half=std::max(2,y(5));
        const int extent=channel.right-channel.left;
        const auto strength=SendMessageW(slider,TBM_GETPOS,0,0);
        for (int i=0;i<10;++i) {
            const int a=channel.left+MulDiv(i,extent,10),b=channel.left+MulDiv(i+1,extent,10)-std::max(1,x(2));
            fill(draw.hdc,{a,center-half,b,center+half},i*10<strength?accent:frame);
        }
        fill(draw.hdc,thumb,ink);
        if (GetFocus()==slider) { InflateRect(&client,-1,-1); DrawFocusRect(draw.hdc,&client); }
        return CDRF_SKIPDEFAULT;
    }
    static LRESULT CALLBACK procedure(HWND window,UINT message,WPARAM wp,LPARAM lp) {
        auto self=reinterpret_cast<Impl*>(GetWindowLongPtrW(window,GWLP_USERDATA));
        if (message==WM_NCCREATE) {
            self=static_cast<Impl*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);
            self->hwnd=window; SetWindowLongPtrW(window,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(self));
        }
        if (!self) return DefWindowProcW(window,message,wp,lp);
        if (self->taskbarMessage && message==self->taskbarMessage) { self->updateTray(true); return 0; }
        switch (message) {
        case WM_TIMER:
            if (wp==TrayRetryTimer) { self->updateTray(); return 0; }
            break;
        case WM_CREATE: {
            self->nativeControl(self->toggle,WC_BUTTONW,Toggle,BS_OWNERDRAW|WS_TABSTOP);
            self->nativeControl(self->value,WC_STATICW,Value,SS_OWNERDRAW);
            self->nativeControl(self->slider,TRACKBAR_CLASSW,Level,TBS_NOTICKS|TBS_FIXEDLENGTH|WS_TABSTOP);
            self->nativeControl(self->menu,WC_BUTTONW,Menu,BS_OWNERDRAW|WS_TABSTOP);
            self->nativeControl(self->hide,WC_BUTTONW,Hide,BS_OWNERDRAW|WS_TABSTOP);
            if (!self->toggle || !self->value || !self->slider || !self->menu || !self->hide) return -1;
            for (auto button:{self->toggle,self->menu,self->hide}) if (!pixel::skinButton(button)) return -1;
            if (!pixel::skinStatic(self->value)) return -1;
            SetWindowTheme(self->slider,L"",L"");
            if (!pixel::skinSlider(self->slider)) return -1;
            SendMessageW(self->slider,TBM_SETRANGE,TRUE,MAKELPARAM(0,100));
            SendMessageW(self->slider,TBM_SETPAGESIZE,0,5);
            self->tooltip=CreateWindowExW(WS_EX_TOPMOST,TOOLTIPS_CLASSW,nullptr,WS_POPUP|TTS_ALWAYSTIP|TTS_NOPREFIX,
                CW_USEDEFAULT,CW_USEDEFAULT,CW_USEDEFAULT,CW_USEDEFAULT,window,nullptr,GetModuleHandleW(nullptr),nullptr);
            SendMessageW(self->tooltip,TTM_SETMAXTIPWIDTH,0,360);
            for (auto child:{self->toggle,self->slider,self->menu,self->hide}) {
                TOOLINFOW tool{sizeof(tool)}; tool.uFlags=TTF_IDISHWND|TTF_SUBCLASS;
                tool.hwnd=window; tool.uId=reinterpret_cast<UINT_PTR>(child); tool.lpszText=const_cast<wchar_t*>(L"");
                SendMessageW(self->tooltip,TTM_ADDTOOLW,0,reinterpret_cast<LPARAM>(&tool));
            }
            self->colors(); self->layout(); return 0;
        }
        case WM_NCCALCSIZE: return 0;
        case WM_NCHITTEST: {
            POINT p{GET_X_LPARAM(lp),GET_Y_LPARAM(lp)}; ScreenToClient(window,&p);
            const int edge=std::max(4,self->x(6));
            if (p.x<edge) return HTLEFT;
            if (p.x>=self->width-edge) return HTRIGHT;
            if (p.y<self->y(16)) return HTTOP;
            if (p.y>self->y(175)) return HTBOTTOM;
            const auto child=ChildWindowFromPointEx(window,p,CWP_SKIPINVISIBLE);
            return child && child!=window?HTCLIENT:HTCAPTION;
        }
        case WM_NCMOUSEMOVE: {
            if (!self->hover) { self->hover=true; self->repaint(); }
            TRACKMOUSEEVENT track{sizeof(track),TME_LEAVE|TME_NONCLIENT,window,0}; TrackMouseEvent(&track); break;
        }
        case WM_MOUSEMOVE: self->pointer(window,true); return 0;
        case WM_MOUSELEAVE: case WM_NCMOUSELEAVE: self->pointer(window,false); return 0;
        case WM_ERASEBKGND: return 1;
        case WM_PAINT: self->paint(); return 0;
        case WM_SIZE: if (wp!=SIZE_MINIMIZED) self->layout(); return 0;
        case WM_SIZING: {
            auto& r=*reinterpret_cast<RECT*>(lp);
            const int width=std::max(self->minWidth,(wp==WMSZ_TOP || wp==WMSZ_BOTTOM)?static_cast<int>((r.bottom-r.top)*3):static_cast<int>(r.right-r.left));
            if (wp==WMSZ_LEFT || wp==WMSZ_TOPLEFT || wp==WMSZ_BOTTOMLEFT) r.left=r.right-width;
            else r.right=r.left+width;
            if (wp==WMSZ_TOP || wp==WMSZ_TOPLEFT || wp==WMSZ_TOPRIGHT) r.top=r.bottom-width/3;
            else r.bottom=r.top+width/3;
            return TRUE;
        }
        case WM_EXITSIZEMOVE: self->fit(); return 0;
        case WM_GETMINMAXINFO: {
            auto& size=*reinterpret_cast<MINMAXINFO*>(lp);
            size.ptMinTrackSize={self->minWidth,self->minWidth/3};
            return 0;
        }
        case WM_DPICHANGED: {
            self->cloth.dismiss(false);
            self->updateWindowIcons();
            if (self->placingFromCase) { self->layout(); return 0; }
            const auto& r=*reinterpret_cast<RECT*>(lp);
            SetWindowPos(window,nullptr,r.left,r.top,r.right-r.left,r.bottom-r.top,SWP_NOACTIVATE|SWP_NOZORDER);
            self->fit(); return 0;
        }
        case WM_SETTINGCHANGE: case WM_THEMECHANGED:
            self->cloth.dismiss(false);
            self->colors(); self->layout(); self->updateTray();
            if (self->width<self->minWidth) SetWindowPos(window,nullptr,0,0,self->minWidth,self->minWidth/3,SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE);
            self->fit(); return 0;
        case WM_SETFOCUS: SetFocus(self->toggle); return 0;
        case WM_ACTIVATE: self->repaint(); break;
        case ClothClosedMessage: self->menuOpen=self->cloth.isVisible(); self->pointer(window,false); return 0;
        case ClothCommandMessage: self->commandSelected(static_cast<UINT>(wp)); return 0;
        case WM_MEASUREITEM:
            if (pixel::Menu::measure(*reinterpret_cast<MEASUREITEMSTRUCT*>(lp))) return TRUE;
            break;
        case WM_CTLCOLORSTATIC: case WM_CTLCOLORBTN: {
            const auto dc=reinterpret_cast<HDC>(wp);
            const auto child=reinterpret_cast<HWND>(lp);
            const bool hinge=child==self->menu || child==self->hide;
            const auto background=hinge?self->frame:self->lens;
            SetBkColor(dc,background); SetTextColor(dc,self->ink); SetDCBrushColor(dc,background);
            return reinterpret_cast<LRESULT>(GetStockObject(DC_BRUSH));
        }
        case WM_DRAWITEM:
            pixel::drawBufferedItem(*reinterpret_cast<DRAWITEMSTRUCT*>(lp),[&](const auto& item) {
                if (!pixel::Menu::draw(item)) self->draw(item);
            });
            return TRUE;
        case WM_NOTIFY: {
            const auto header=reinterpret_cast<NMHDR*>(lp);
            if (header->hwndFrom==self->slider && header->code==NM_CUSTOMDRAW) return self->drawSlider(*reinterpret_cast<NMCUSTOMDRAW*>(lp));
            break;
        }
        case WM_HSCROLL:
            SendMessageW(self->controller,WidgetStrengthMessage,SendMessageW(self->slider,TBM_GETPOS,0,0),0); return 0;
        case WM_COMMAND:
            if (LOWORD(wp)==Toggle || LOWORD(wp)==IDOK) PostMessageW(self->controller,WM_COMMAND,IDC_SWITCH,0);
            else if (LOWORD(wp)==Menu) self->popup();
            else if (LOWORD(wp)==Hide) self->conceal();
            else if (LOWORD(wp)==IDCANCEL) PostMessageW(self->controller,WM_COMMAND,IDC_STOP,0);
            return 0;
        case WM_CONTEXTMENU: self->popup(lp!=-1); return 0;
        case TrayMessage:
            if (LOWORD(lp)==NIN_SELECT || LOWORD(lp)==NIN_KEYSELECT || (!self->trayVersion4 && LOWORD(lp)==WM_LBUTTONUP))
                PostMessageW(self->controller,WM_COMMAND,IDC_WIDGET_RESTORE,0);
            else if (LOWORD(lp)==NIN_BALLOONUSERCLICK) PostMessageW(self->controller,WM_COMMAND,IDC_WIDGET_SETTINGS,0);
            else if (LOWORD(lp)==WM_CONTEXTMENU || (!self->trayVersion4 && LOWORD(lp)==WM_RBUTTONUP)) self->popup(true);
            return 0;
        case WM_SYSCOMMAND:
            // Show Desktop must not become an explicit request to hide in the tray.
            if ((wp&0xfff0)==SC_MINIMIZE && self->state.pinned) return 0;
            if ((wp&0xfff0)==SC_MAXIMIZE) return 0;
            break;
        case WM_CLOSE: PostMessageW(self->controller,WM_COMMAND,IDC_WIDGET_EXIT,0); return 0;
        case WM_DESTROY: {
            KillTimer(window,TrayRetryTimer);
            self->cloth.dismiss(false);
            auto data=self->trayData(); Shell_NotifyIconW(NIM_DELETE,&data); self->tray=false;
            if (self->tooltip) { DestroyWindow(self->tooltip); self->tooltip=nullptr; }
            return 0;
        }
        case WM_NCDESTROY: self->hwnd=nullptr; SetWindowLongPtrW(window,GWLP_USERDATA,0); break;
        }
        return DefWindowProcW(window,message,wp,lp);
    }
};

GlassesWidget::GlassesWidget():impl_(std::make_unique<Impl>()) {}
GlassesWidget::~GlassesWidget()=default;
bool GlassesWidget::create(HINSTANCE instance,HWND controller) {
    impl_->controller=controller;
    WNDCLASSW klass{}; klass.hInstance=instance; klass.lpszClassName=WidgetClass;
    klass.lpfnWndProc=Impl::procedure; klass.hCursor=LoadCursorW(nullptr,IDC_ARROW);
    klass.hIcon=LoadIconW(instance,MAKEINTRESOURCEW(IDI_SUNGLASSES));
    if (!RegisterClassW(&klass) && GetLastError()!=ERROR_CLASS_ALREADY_EXISTS) return false;
    const UINT dpi=GetDpiForSystem(); const int width=MulDiv(480,static_cast<int>(dpi),96);
    // A floating control, with no task-switching entry or minimize button. Keep it
    // ownerless so minimizing the case cannot hide it through owner propagation.
    const auto window=CreateWindowExW(WS_EX_TOOLWINDOW|WS_EX_CONTROLPARENT,WidgetClass,L"",WS_POPUP|WS_THICKFRAME|WS_SYSMENU|WS_CLIPCHILDREN,
        100,100,width,width/3,nullptr,nullptr,instance,impl_.get());
    if (window && impl_->width<impl_->minWidth)
        SetWindowPos(window,nullptr,0,0,impl_->minWidth,impl_->minWidth/3,SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE);
    return window!=nullptr;
}
void GlassesWidget::update(const WidgetState& state) { if (impl_->hwnd) impl_->update(state); }
void GlassesWidget::show() { if (impl_->hwnd) impl_->show(); }
void GlassesWidget::hide() { if (impl_->hwnd) impl_->conceal(); }
void GlassesWidget::dock() {
    if (impl_->hwnd) { impl_->cloth.dismiss(false); ShowWindow(impl_->hwnd,SW_HIDE); }
}
void GlassesWidget::showFromCase(RECT bounds,bool activate) {
    if (!impl_->hwnd) return;
    impl_->placingFromCase=true;
    SetWindowPos(impl_->hwnd,HWND_TOP,bounds.left,bounds.top,bounds.right-bounds.left,bounds.bottom-bounds.top,SWP_NOACTIVATE);
    impl_->placingFromCase=false;
    if (activate) impl_->show();
    else { ShowWindow(impl_->hwnd,SW_SHOWNOACTIVATE); impl_->fit(); impl_->repaint(); }
}
RECT GlassesWidget::bounds() const {
    RECT rect{}; if (impl_->hwnd) GetWindowRect(impl_->hwnd,&rect); return rect;
}
bool GlassesWidget::processMessage(MSG& message) { return impl_->cloth.processMessage(message); }
void GlassesWidget::fitOnScreen() { if (impl_->hwnd) impl_->fit(); }
bool GlassesWidget::hasTray() const { return impl_->tray; }
HWND GlassesWidget::window() const { return impl_->hwnd; }
}
