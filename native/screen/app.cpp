// Screen filter controls. Starts disabled. Never saves captured pixels or uploads data.
#include <windows.h>
#include <commctrl.h>
#include <wtsapi32.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.System.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "resource.h"
#include "capture.h"
#include "diagnostics.h"
#include "widget.h"
#include "glasses_case.h"
#include "single_instance.h"
#include "desktop_strings.h"

namespace {
constexpr UINT CaptureMessage=WM_APP+40;
constexpr UINT PermissionSettingsMessage=WM_APP+49;
constexpr int ToggleHotkey=1,StopHotkey=2;
constexpr wchar_t OverlayClass[]=L"Sunglasses.ScreenOverlay";
constexpr wchar_t ManualClass[]=L"Sunglasses.ManualDimmer";
struct Monitor { HMONITOR handle; RECT bounds; std::wstring name; bool primary; };
enum class Phase { Off,Starting,On,Stopping };
using sunglasses::DiagnosticEvent;
std::wstring localized(std::wstring_view id,bool chinese) {
    for (const auto& entry:sunglasses::strings::entries)
        if (entry.id==id) return std::wstring(chinese?entry.chinese:entry.english);
    return std::wstring(id);
}
int startupError(std::wstring_view key) {
    const bool chinese=PRIMARYLANGID(GetUserDefaultUILanguage())==LANG_CHINESE;
    MessageBoxW(nullptr,localized(key,chinese).c_str(),localized(L"desktop.widget_title",chinese).c_str(),MB_OK|MB_ICONERROR);
    return 2;
}
struct ComApartment {
    HRESULT result=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
    ~ComApartment() { if (SUCCEEDED(result)) CoUninitialize(); }
};
struct App {
    // Declared before capture so the diagnostic writer outlives its producers.
    sunglasses::ScreenDiagnostics diagnostics;
    HWND window=nullptr,overlay=nullptr;
    bool chinese=PRIMARYLANGID(GetUserDefaultUILanguage())==LANG_CHINESE,hotkeys=false;
    bool manual=false,notified=false,closing=false,pinned=true;
    unsigned generation=0;
    Phase phase=Phase::Off;
    std::wstring state=L"desktop.screen_off";
    std::vector<Monitor> monitors;
    std::unique_ptr<sunglasses::ScreenCapture> capture;
    sunglasses::GlassesWidget widget;
    sunglasses::GlassesCase glassesCase;
    sunglasses::SingleInstance* instance=nullptr;
    winrt::Windows::Foundation::IAsyncOperation<bool> settingsRequest{nullptr};
    std::wstring text(std::wstring_view id) const {
        return localized(id,chinese);
    }
    void set(int id,std::wstring_view value) const { SetDlgItemTextW(window,id,std::wstring(value).c_str()); }
    void status(std::wstring_view key) { state=key; set(IDC_STATE,text(key)); controls(); }
    bool permissionRecovery() const {
        return state==L"desktop.screen_permission_denied" || state==L"desktop.screen_permission_system" ||
            state==L"desktop.screen_settings_failed";
    }
    void registerHotkeys() {
        UnregisterHotKey(window,ToggleHotkey); UnregisterHotKey(window,StopHotkey);
        const bool toggle=RegisterHotKey(window,ToggleHotkey,MOD_CONTROL|MOD_ALT|MOD_NOREPEAT,'S')!=FALSE;
        const bool stop=RegisterHotKey(window,StopHotkey,MOD_CONTROL|MOD_ALT|MOD_NOREPEAT,VK_ESCAPE)!=FALSE;
        hotkeys=toggle&&stop;
        // A failed attempt must not keep one shortcut reserved.
        if (!hotkeys) { UnregisterHotKey(window,ToggleHotkey); UnregisterHotKey(window,StopHotkey); }
    }
    void recoverOrStop() {
        if (phase!=Phase::Off) { stop(); return; }
        if (!hotkeys) {
            registerHotkeys(); status(hotkeys?L"desktop.screen_off":L"desktop.screen_hotkey_failed");
            return;
        }
        if (!permissionRecovery() || settingsRequest) return;
        try {
            using namespace winrt::Windows::Foundation;
            settingsRequest=winrt::Windows::System::Launcher::LaunchUriAsync(Uri(L"ms-settings:privacy-graphicscapturewithoutborder"));
            settingsRequest.Completed([owner=window](const IAsyncOperation<bool>& operation,AsyncStatus result) {
                bool opened=false;
                try { if (result==AsyncStatus::Completed) opened=operation.GetResults(); } catch (...) {}
                PostMessageW(owner,PermissionSettingsMessage,opened,0);
            });
            controls();
        } catch (...) {
            settingsRequest=nullptr; status(L"desktop.screen_settings_failed");
        }
    }
    int level() const { return static_cast<int>(SendDlgItemMessageW(window,IDC_LEVEL,TBM_GETPOS,0,0)); }
    bool recordable() const { return SendDlgItemMessageW(window,IDC_RECORDING,CB_GETCURSEL,0,0)==1; }
    std::wstring percent() const {
        const auto number=std::to_wstring(level()); wchar_t formatted[32]{};
        NUMBERFMTW format{0,1,0,const_cast<wchar_t*>(L"."),const_cast<wchar_t*>(L","),1};
        GetNumberFormatEx(chinese?L"zh-CN":L"en-US",0,number.c_str(),&format,formatted,32);
        auto pattern=text(L"desktop.widget_percent"); const auto offset=pattern.find(L"{value}");
        if (offset!=std::wstring::npos) pattern.replace(offset,7,formatted[0]?formatted:number);
        return pattern;
    }
    void syncWidget() {
        sunglasses::WidgetState model;
        model.strength=level(); model.on=phase==Phase::On;
        model.busy=phase==Phase::Starting || phase==Phase::Stopping;
        model.canToggle=hotkeys && !closing && phase!=Phase::Stopping && !monitors.empty();
        model.error=state!=L"desktop.screen_off" && state!=L"desktop.screen_on" &&
            state!=L"desktop.screen_starting" && state!=L"desktop.screen_stopping";
        const auto shortState=model.error?L"desktop.widget_error":phase==Phase::Starting?L"desktop.widget_starting":
            phase==Phase::Stopping?L"desktop.widget_stopping":phase==Phase::On?L"desktop.screen_on":L"desktop.screen_off";
        model.title=text(L"desktop.widget_title"); model.toggle=text(phase==Phase::Off?L"desktop.screen_enable":L"desktop.screen_disable");
        model.status=text(shortState); model.detail=text(state); model.strengthName=text(L"desktop.strength");
        model.percent=percent(); model.menu=text(L"desktop.widget_menu"); model.settings=text(L"desktop.widget_settings");
        model.hide=text(L"desktop.widget_hide"); model.restore=text(L"desktop.widget_restore");
        model.restoreCase=text(L"desktop.widget_restore_case");
        model.stop=text(L"desktop.screen_stop"); model.exit=text(L"desktop.widget_exit");
        model.pinned=pinned; model.pin=text(L"desktop.widget_pin"); model.size=text(L"desktop.widget_size");
        model.sizeSmall=text(L"desktop.widget_size_small"); model.sizeNormal=text(L"desktop.widget_size_normal");
        model.sizeLarge=text(L"desktop.widget_size_large");
        widget.update(model);
    }
    void controls() {
        const bool off=phase==Phase::Off;
        EnableWindow(GetDlgItem(window,IDC_SCREEN),off);
        EnableWindow(GetDlgItem(window,IDC_MODE),off);
        EnableWindow(GetDlgItem(window,IDC_RECORDING),off);
        EnableWindow(GetDlgItem(window,IDC_SWITCH),hotkeys && !closing && phase!=Phase::Stopping && !monitors.empty());
        const bool permissions=off && permissionRecovery();
        const bool retry=off && !hotkeys;
        set(IDC_STOP,text(permissions?L"desktop.screen_permissions":retry?L"desktop.screen_retry_hotkeys":L"desktop.screen_stop"));
        EnableWindow(GetDlgItem(window,IDC_STOP),!off || retry || (permissions && !settingsRequest));
        set(IDC_SWITCH,text(off?L"desktop.screen_enable":L"desktop.screen_disable"));
        syncWidget();
    }
    void localize() {
        SetWindowTextW(window,text(L"desktop.widget_settings_title").c_str());
        for (const auto& [id,key]:std::initializer_list<std::pair<int,const wchar_t*>>{
            {IDC_MODE_LABEL,L"desktop.screen_mode"},{IDC_RECORDING_LABEL,L"desktop.recording"},
            {IDC_LEVEL_LABEL,L"desktop.strength"},{IDC_LOCALE_LABEL,L"desktop.language"},{IDC_STOP,L"desktop.screen_stop"},
            {IDC_CASE_TITLE,L"desktop.case_title"},{IDC_CASE_CLOSE,L"desktop.case_close"}})
            set(id,text(key));
        int mode=static_cast<int>(SendDlgItemMessageW(window,IDC_MODE,CB_GETCURSEL,0,0));
        SendDlgItemMessageW(window,IDC_MODE,CB_RESETCONTENT,0,0);
        for (auto key:{L"desktop.screen_highlights",L"desktop.screen_manual",L"desktop.screen_apple_vfm"}) {
            auto value=text(key); SendDlgItemMessageW(window,IDC_MODE,CB_ADDSTRING,0,reinterpret_cast<LPARAM>(value.c_str()));
        }
        SendDlgItemMessageW(window,IDC_MODE,CB_SETCURSEL,mode<0?0:mode,0);
        const auto recording=SendDlgItemMessageW(window,IDC_RECORDING,CB_GETCURSEL,0,0);
        SendDlgItemMessageW(window,IDC_RECORDING,CB_RESETCONTENT,0,0);
        for (auto key:{L"desktop.recording_local",L"desktop.recording_visible"}) {
            const auto value=text(key); SendDlgItemMessageW(window,IDC_RECORDING,CB_ADDSTRING,0,reinterpret_cast<LPARAM>(value.c_str()));
        }
        SendDlgItemMessageW(window,IDC_RECORDING,CB_SETCURSEL,recording<0?0:recording,0);
        set(IDC_SCREEN_LABEL,text(L"desktop.screen_monitor"));
        set(IDC_STATE,text(state)); controls();
    }
    void enumerateTargets(HMONITOR previous) {
        SendDlgItemMessageW(window,IDC_SCREEN,CB_RESETCONTENT,0,0);
        int selected=-1,primary=0;
        for (size_t i=0;i<monitors.size();++i) {
            const auto& m=monitors[i];
            auto label=m.name+L"  "+std::to_wstring(m.bounds.right-m.bounds.left)+L" × "+std::to_wstring(m.bounds.bottom-m.bounds.top);
            SendDlgItemMessageW(window,IDC_SCREEN,CB_ADDSTRING,0,reinterpret_cast<LPARAM>(label.c_str()));
            if (m.primary) primary=static_cast<int>(i);
            if (m.handle==previous) selected=static_cast<int>(i);
        }
        SendDlgItemMessageW(window,IDC_SCREEN,CB_SETCURSEL,selected<0?primary:selected,0);
        set(IDC_SCREEN_LABEL,text(L"desktop.screen_monitor"));
        controls();
    }
    void enumerate() {
        const auto index=static_cast<size_t>(SendDlgItemMessageW(window,IDC_SCREEN,CB_GETCURSEL,0,0));
        const auto previous=index<monitors.size()?monitors[index].handle:nullptr;
        monitors.clear();
        EnumDisplayMonitors(nullptr,nullptr,[](HMONITOR monitor,HDC,LPRECT,LPARAM value)->BOOL {
            auto& list=*reinterpret_cast<std::vector<Monitor>*>(value);
            MONITORINFOEXW info{}; info.cbSize=sizeof(info);
            if (GetMonitorInfoW(monitor,&info)) list.push_back({monitor,info.rcMonitor,info.szDevice,(info.dwFlags&MONITORINFOF_PRIMARY)!=0});
            return TRUE;
        },reinterpret_cast<LPARAM>(&monitors));
        enumerateTargets(previous);
    }
    void applyPin() const {
        if (IsWindow(widget.window()))
            SetWindowPos(widget.window(),pinned?HWND_TOPMOST:HWND_NOTOPMOST,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE|SWP_NOACTIVATE);
    }
    void destroyOverlay() {
        if (overlay) {
            diagnostics.event(DiagnosticEvent::OverlayHide,generation);
            diagnostics.watch(nullptr,generation);
            ShowWindow(overlay,SW_HIDE); DestroyWindow(overlay); overlay=nullptr;
            diagnostics.event(DiagnosticEvent::OverlayDestroyed,generation);
        }
        if (IsWindow(window)) {
            SetWindowDisplayAffinity(window,WDA_NONE);
        }
        if (widget.window()) {
            SetWindowDisplayAffinity(widget.window(),WDA_NONE);
        }
        applyPin();
    }
    void settings(bool activate=true) {
        if (closing) return;
        if (IsWindowVisible(window)) {
            // A tray restore preserves the open case, its selected target and focused control.
            if (IsIconic(window)) ShowWindow(window,activate?SW_RESTORE:SW_SHOWNOACTIVATE);
            glassesCase.open(widget.bounds(),phase==Phase::On,activate);
            if (activate) SetForegroundWindow(window);
            return;
        }
        const auto source=widget.bounds();
        glassesCase.open(source,phase==Phase::On,activate); widget.dock();
        if (activate) SetForegroundWindow(window);
    }
    void restoreControls() {
        if (closing) return;
        if (IsWindowVisible(window)) settings();
        else widget.show();
    }
    void closeCase() {
        if (closing) return;
        if (!IsWindowVisible(window)) { widget.show(); return; }
        if (IsIconic(window)) ShowWindow(window,SW_RESTORE);
        SetForegroundWindow(window);
        const auto source=widget.bounds();
        const int width=MulDiv(source.right-source.left,static_cast<int>(GetDpiForWindow(window)),static_cast<int>(GetDpiForWindow(widget.window())));
        glassesCase.close({width,width/3},phase==Phase::On);
    }
    void changeLevel(int value) {
        diagnostics.event(DiagnosticEvent::StrengthChanged,generation,value);
        SendDlgItemMessageW(window,IDC_LEVEL,TBM_SETPOS,TRUE,std::clamp(value,0,100));
        set(IDC_LEVEL_VALUE,std::to_wstring(level()));
        if (capture) capture->strength(level()/100.0f);
        else if (overlay && manual) SetLayeredWindowAttributes(overlay,0,static_cast<BYTE>(level()*153/100),LWA_ALPHA);
        syncWidget();
    }
    void exit() {
        if (instance) instance->stopAccepting();
        closing=true; stop();
        // Keep pumping messages until capture has released its GPU/WGC resources.
        if (!capture) { DestroyWindow(window); PostQuitMessage(0); }
    }
    void stop() {
        diagnostics.event(DiagnosticEvent::FilterStop,generation);
        if (overlay) diagnostics.event(DiagnosticEvent::OverlayHide,generation);
        if (overlay) ShowWindow(overlay,SW_HIDE); // Immediate UI operation, never waits for GPU/capture.
        if (capture) {
            capture->stop(); phase=Phase::Stopping; status(L"desktop.screen_stopping");
        } else {
            destroyOverlay(); phase=Phase::Off;
            status(hotkeys || closing?L"desktop.screen_off":L"desktop.screen_hotkey_failed");
        }
        controls();
    }
    void start() {
        if (closing || phase==Phase::Stopping) return;
        if (phase!=Phase::Off) { stop(); return; }
        if (!hotkeys) return;
        const auto index=static_cast<size_t>(SendDlgItemMessageW(window,IDC_SCREEN,CB_GETCURSEL,0,0));
        const auto mode=SendDlgItemMessageW(window,IDC_MODE,CB_GETCURSEL,0,0);
        manual=mode==1;
        sunglasses::CaptureTarget target;
        target.algorithm=mode==2?sunglasses::ScreenAlgorithm::AppleVfm:sunglasses::ScreenAlgorithm::SuddenHighlights;
        if (index>=monitors.size()) { status(L"desktop.screen_changed"); return; }
        target.monitor=monitors[index].handle; target.bounds=monitors[index].bounds;
        target.includeInRecordings=recordable();
        ++generation; // Also identify manual-mode overlay lifetimes in diagnostics.
        diagnostics.event(DiagnosticEvent::FilterStart,generation,mode,level());
        const DWORD extended=WS_EX_TOPMOST|WS_EX_TOOLWINDOW|WS_EX_NOACTIVATE|WS_EX_TRANSPARENT|WS_EX_LAYERED|
                             (manual?0:WS_EX_NOREDIRECTIONBITMAP);
        overlay=CreateWindowExW(extended,manual?ManualClass:OverlayClass,L"",WS_POPUP,target.bounds.left,target.bounds.top,
            target.bounds.right-target.bounds.left,target.bounds.bottom-target.bounds.top,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
        if (!overlay) {
            diagnostics.event(DiagnosticEvent::OverlayCreated,generation,0,GetLastError());
            status(L"desktop.screen_failed"); return;
        }
        diagnostics.watch(overlay,generation);
        diagnostics.event(DiagnosticEvent::OverlayCreated,generation,1,target.includeInRecordings);
        target.excludedWindows={overlay,window,widget.window()};
        // Layered hit testing requests pass-through; the automatic window has no GDI backdrop.
        if (!SetLayeredWindowAttributes(overlay,0,manual?static_cast<BYTE>(level()*153/100):255,LWA_ALPHA)) {
            destroyOverlay(); status(L"desktop.screen_failed"); return;
        }
        const DWORD affinity=recordable()?WDA_NONE:WDA_EXCLUDEFROMCAPTURE;
        if (!SetWindowDisplayAffinity(overlay,affinity) || !SetWindowDisplayAffinity(window,affinity) ||
            !SetWindowDisplayAffinity(widget.window(),affinity)) {
            destroyOverlay(); status(L"desktop.screen_exclusion_failed"); return;
        }
        if (manual) {
            diagnostics.event(DiagnosticEvent::OverlayShow,generation);
            phase=Phase::On; ShowWindow(overlay,SW_SHOWNOACTIVATE); status(L"desktop.screen_on");
        } else {
            // OS check avoids old WDA_MONITOR behavior (a black captured rectangle).
            using VersionFn=LONG(WINAPI*)(OSVERSIONINFOW*);
            auto version=reinterpret_cast<VersionFn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"),"RtlGetVersion"));
            OSVERSIONINFOW info{}; info.dwOSVersionInfoSize=sizeof(info);
            if (!version || version(&info)!=0 || info.dwBuildNumber<22000) {
                destroyOverlay(); status(L"desktop.screen_unsupported"); return;
            }
            try {
                capture=std::make_unique<sunglasses::ScreenCapture>(overlay,window,CaptureMessage,target,generation,level()/100.0f,
                    diagnostics.enabled()?&diagnostics:nullptr);
                phase=Phase::Starting; status(L"desktop.screen_starting");
            } catch (...) { destroyOverlay(); status(L"desktop.screen_failed"); }
        }
        applyPin();
        controls();
    }
};

LRESULT CALLBACK overlayProc(HWND window,UINT message,WPARAM wparam,LPARAM lparam) {
    if (message==WM_NCHITTEST) return HTTRANSPARENT;
    if (message==WM_MOUSEACTIVATE) return MA_NOACTIVATE;
    if (message==WM_PAINT) { PAINTSTRUCT ps{}; BeginPaint(window,&ps); EndPaint(window,&ps); return 0; }
    return DefWindowProcW(window,message,wparam,lparam);
}
INT_PTR CALLBACK dialog(HWND window,UINT message,WPARAM wparam,LPARAM lparam) {
    auto app=reinterpret_cast<App*>(GetWindowLongPtrW(window,DWLP_USER));
    if (message==WM_INITDIALOG) {
        app=reinterpret_cast<App*>(lparam); app->window=window; SetWindowLongPtrW(window,DWLP_USER,lparam);
        if (!app->glassesCase.attach(window)) { DestroyWindow(window); return TRUE; }
        SendDlgItemMessageW(window,IDC_LEVEL,TBM_SETRANGE,TRUE,MAKELPARAM(0,100));
        SendDlgItemMessageW(window,IDC_LEVEL,TBM_SETPOS,TRUE,sunglasses::DefaultScreenStrength);
        for (const auto key:{L"desktop.english",L"desktop.chinese"}) {
            const auto language=app->text(key); SendDlgItemMessageW(window,IDC_LOCALE,CB_ADDSTRING,0,reinterpret_cast<LPARAM>(language.c_str()));
        }
        SendDlgItemMessageW(window,IDC_LOCALE,CB_SETCURSEL,app->chinese?1:0,0);
        app->registerHotkeys();
        app->notified=WTSRegisterSessionNotification(window,NOTIFY_FOR_THIS_SESSION)!=FALSE;
        app->localize(); app->enumerate();
        if (!app->hotkeys) app->status(L"desktop.screen_hotkey_failed");
        return TRUE;
    }
    if (!app) return FALSE;
    switch (message) {
    case sunglasses::RestoreInstanceMessage: app->restoreControls(); return TRUE;
    case PermissionSettingsMessage:
        if (!app->settingsRequest) return TRUE;
        app->settingsRequest=nullptr;
        if (!wparam && app->phase==Phase::Off && app->permissionRecovery()) app->status(L"desktop.screen_settings_failed");
        app->controls(); return TRUE;
    case WM_HSCROLL: app->changeLevel(app->level()); return TRUE;
    case sunglasses::WidgetStrengthMessage: app->changeLevel(static_cast<int>(wparam)); return TRUE;
    case sunglasses::WidgetTrayUnavailableMessage:
        // The case supplies a normal taskbar entry if all other controls are hidden.
        if (!app->widget.hasTray() && !IsWindowVisible(window) &&
            (!IsWindowVisible(app->widget.window()) || IsIconic(app->widget.window()))) app->settings(false);
        return TRUE;
    case sunglasses::CaseHiddenMessage:
        if (!app->closing) app->widget.showFromCase(app->glassesCase.glassesBounds(),wparam!=0); return TRUE;
    case sunglasses::CaseOpenedMessage:
        if (!app->closing && IsWindowVisible(window) && GetForegroundWindow()==window)
            SetFocus(GetDlgItem(window,app->phase==Phase::Off?IDC_SCREEN:IDC_LEVEL));
        return TRUE;
    case WM_HOTKEY:
        if (wparam==StopHotkey) app->stop();
        else if (wparam==ToggleHotkey && app->phase!=Phase::Stopping) app->start();
        return TRUE;
    case CaptureMessage: {
        if (wparam!=app->generation || !app->capture) return TRUE;
        using sunglasses::ScreenStatus;
        const auto status=static_cast<ScreenStatus>(lparam);
        if (status==ScreenStatus::OverlayReady) {
            if (app->phase!=Phase::Starting) return TRUE;
            // The worker has committed a transparent surface. Show it before
            // configuring per-session exclusion; no filtered frame exists yet.
            app->diagnostics.event(DiagnosticEvent::OverlayShow,app->generation);
            ShowWindow(app->overlay,SW_SHOWNOACTIVATE);
            app->capture->overlayShown();
        } else if (status==ScreenStatus::Active) {
            if (app->phase!=Phase::Starting) return TRUE;
            if (!IsWindowVisible(app->overlay)) {
                app->diagnostics.event(DiagnosticEvent::OverlayShow,app->generation);
                ShowWindow(app->overlay,SW_SHOWNOACTIVATE);
            }
            app->phase=Phase::On;
            app->applyPin();
            app->status(L"desktop.screen_on");
        } else {
            app->destroyOverlay(); app->capture.reset(); app->phase=Phase::Off;
            const wchar_t* key=L"desktop.screen_failed";
            switch (status) {
            case ScreenStatus::Stopped: key=L"desktop.screen_off"; break;
            case ScreenStatus::Unsupported: key=L"desktop.screen_unsupported"; break;
            case ScreenStatus::Hdr: key=L"desktop.screen_hdr"; break;
            case ScreenStatus::DisplayChanged: key=L"desktop.screen_changed"; break;
            case ScreenStatus::NoFrames: key=L"desktop.screen_no_frames"; break;
            case ScreenStatus::BorderlessUnavailable: key=L"desktop.screen_borderless_unavailable"; break;
            case ScreenStatus::CaptureExclusionUnavailable: key=L"desktop.screen_capture_exclusion_unavailable"; break;
            case ScreenStatus::BorderlessDeniedByUser: key=L"desktop.screen_permission_denied"; break;
            case ScreenStatus::BorderlessDeniedBySystem: key=L"desktop.screen_permission_system"; break;
            case ScreenStatus::BorderlessNotDeclared: key=L"desktop.screen_permission_missing"; break;
            case ScreenStatus::BorderlessRequestFailed: key=L"desktop.screen_permission_failed"; break;
            default: break;
            }
            app->status(key); SetWindowDisplayAffinity(window,WDA_NONE);
            if (app->closing) { app->exit(); return TRUE; }
            // Keep the full error and available recovery action visible without
            // relying on a tray notification. Do not take focus from another app.
            if (status!=ScreenStatus::Stopped)
                app->settings(GetForegroundWindow()==window || GetForegroundWindow()==app->widget.window());
            if (GetForegroundWindow()==window) SetFocus(GetDlgItem(window,IDC_SWITCH));
        }
        app->controls(); return TRUE;
    }
    case WM_DISPLAYCHANGE:
        app->diagnostics.event(DiagnosticEvent::DisplayChanged,app->generation);
        app->stop(); app->enumerate(); app->widget.fitOnScreen(); app->glassesCase.fitOnScreen(); return TRUE;
    case WM_WTSSESSION_CHANGE:
        app->diagnostics.event(DiagnosticEvent::SessionChanged,app->generation,static_cast<int64_t>(wparam));
        if (wparam==WTS_SESSION_LOCK || wparam==WTS_CONSOLE_DISCONNECT || wparam==WTS_REMOTE_DISCONNECT) app->stop();
        return TRUE;
    case WM_POWERBROADCAST:
        if (wparam==PBT_APMSUSPEND) { app->diagnostics.event(DiagnosticEvent::PowerSuspend,app->generation); app->stop(); }
        return TRUE;
    case WM_COMMAND:
        if (LOWORD(wparam)==IDC_SWITCH || LOWORD(wparam)==IDOK) app->start();
        else if (LOWORD(wparam)==IDC_STOP) app->recoverOrStop();
        else if (LOWORD(wparam)==IDCANCEL || LOWORD(wparam)==IDC_CASE_CLOSE) app->closeCase();
        else if (LOWORD(wparam)==IDC_WIDGET_SETTINGS) app->settings();
        else if (LOWORD(wparam)==IDC_WIDGET_RESTORE) app->restoreControls();
        else if (LOWORD(wparam)==IDC_WIDGET_EXIT) app->exit();
        else if (LOWORD(wparam)==IDC_WIDGET_PIN) { app->pinned=!app->pinned; app->applyPin(); app->syncWidget(); }
        else if (LOWORD(wparam)==IDC_LOCALE && HIWORD(wparam)==CBN_SELCHANGE) {
            app->chinese=SendDlgItemMessageW(window,IDC_LOCALE,CB_GETCURSEL,0,0)==1; app->localize();
        }
        return TRUE;
    case WM_CLOSE:
        app->closeCase(); return TRUE;
    case WM_DESTROY:
        if (app->settingsRequest) {
            try { app->settingsRequest.Cancel(); } catch (...) {}
            app->settingsRequest=nullptr;
        }
        UnregisterHotKey(window,ToggleHotkey); UnregisterHotKey(window,StopHotkey);
        if (app->notified) WTSUnRegisterSessionNotification(window);
        return TRUE;
    }
    return FALSE;
}
}

int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,PWSTR,int) {
    sunglasses::SingleInstance single;
    const auto role=single.acquire();
    if (role==sunglasses::SingleInstance::Result::Redirected) return 0;
    if (role!=sunglasses::SingleInstance::Result::Primary) return startupError(L"desktop.screen_instance_unavailable");
    ComApartment apartment;
    if (FAILED(apartment.result)) return startupError(L"desktop.screen_start_failed");
    INITCOMMONCONTROLSEX controls{sizeof(controls),ICC_STANDARD_CLASSES|ICC_BAR_CLASSES};
    if (!InitCommonControlsEx(&controls)) return startupError(L"desktop.screen_start_failed");
    WNDCLASSW klass{}; klass.lpfnWndProc=overlayProc; klass.hInstance=instance;
    klass.lpszClassName=OverlayClass; // DComp owns all automatic-mode pixels, including alpha zero.
    if (!RegisterClassW(&klass)) return startupError(L"desktop.screen_start_failed");
    klass.lpszClassName=ManualClass; klass.hbrBackground=static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    if (!RegisterClassW(&klass)) return startupError(L"desktop.screen_start_failed");
    App app; app.instance=&single;
    const auto settings=CreateDialogParamW(instance,MAKEINTRESOURCEW(IDD_SCREEN),nullptr,dialog,reinterpret_cast<LPARAM>(&app));
    if (!IsWindow(settings) || !app.widget.create(instance,settings) || !single.publish(settings)) {
        single.stopAccepting();
        if (IsWindow(settings)) DestroyWindow(settings);
        return startupError(L"desktop.screen_start_failed");
    }
    app.syncWidget(); app.applyPin(); app.widget.show();
    if (!app.hotkeys) app.settings();
    MSG message{};
    while (GetMessageW(&message,nullptr,0,0)>0) {
        if (app.widget.processMessage(message)) continue;
        if ((message.hwnd==settings || IsChild(settings,message.hwnd)) && IsDialogMessageW(settings,&message)) continue;
        if ((message.hwnd==app.widget.window() || IsChild(app.widget.window(),message.hwnd)) &&
            IsDialogMessageW(app.widget.window(),&message)) continue;
        TranslateMessage(&message); DispatchMessageW(&message);
    }
    single.stopAccepting(); app.capture.reset(); app.destroyOverlay();
    return static_cast<int>(message.wParam);
}
