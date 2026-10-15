#include "single_instance.h"
#include <sddl.h>
#include <vector>

namespace sunglasses {
namespace {
constexpr wchar_t EndpointClass[]=L"Sunglasses.Screen.Instance";
constexpr UINT RestoreRequest=WM_APP+1;
struct TokenHandle {
    HANDLE value=nullptr;
    ~TokenHandle() { if (value) CloseHandle(value); }
};
struct SidString {
    PWSTR value=nullptr;
    ~SidString() { if (value) LocalFree(value); }
};
std::wstring userKey() {
    TokenHandle token;
    if (!OpenProcessToken(GetCurrentProcess(),TOKEN_QUERY,&token.value)) return {};
    DWORD bytes=0;
    GetTokenInformation(token.value,TokenUser,nullptr,0,&bytes);
    if (!bytes) return {};
    std::vector<BYTE> buffer(bytes);
    if (!GetTokenInformation(token.value,TokenUser,buffer.data(),bytes,&bytes)) return {};
    SidString sid;
    if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(buffer.data())->User.Sid,&sid.value)) return {};
    return std::wstring(L"Local\\Sunglasses.Screen.")+sid.value;
}
LRESULT CALLBACK endpointProcedure(HWND window,UINT message,WPARAM wparam,LPARAM lparam) {
    if (message==WM_NCCREATE) {
        const auto create=reinterpret_cast<CREATESTRUCTW*>(lparam);
        SetWindowLongPtrW(window,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }
    if (message==RestoreRequest) {
        const auto controller=reinterpret_cast<HWND>(GetWindowLongPtrW(window,GWLP_USERDATA));
        return IsWindow(controller) && PostMessageW(controller,RestoreInstanceMessage,0,0);
    }
    return DefWindowProcW(window,message,wparam,lparam);
}
}

SingleInstance::~SingleInstance() {
    stopAccepting();
    if (owned_) ReleaseMutex(mutex_);
    if (mutex_) CloseHandle(mutex_);
}

SingleInstance::Result SingleInstance::acquire() noexcept {
    try {
        key_=userKey();
        if (key_.empty()) return Result::Unavailable;
        mutex_=CreateMutexW(nullptr,FALSE,key_.c_str());
        if (!mutex_) return Result::Unavailable;
        // This bounded wait runs before creating any UI. Never wait for another
        // process from the controller's message loop or start a competing filter.
        const auto deadline=GetTickCount64()+3000;
        do {
            const auto wait=WaitForSingleObject(mutex_,0);
            if (wait==WAIT_OBJECT_0 || wait==WAIT_ABANDONED) {
                owned_=true;
                return Result::Primary;
            }
            if (wait!=WAIT_TIMEOUT) return Result::Unavailable;
            const auto existing=FindWindowExW(HWND_MESSAGE,nullptr,EndpointClass,key_.c_str());
            if (existing) {
                DWORD process=0;
                GetWindowThreadProcessId(existing,&process);
                if (process) AllowSetForegroundWindow(process);
                DWORD_PTR accepted=0;
                if (SendMessageTimeoutW(existing,RestoreRequest,0,0,SMTO_ABORTIFHUNG|SMTO_BLOCK,
                                        250,&accepted) && accepted) return Result::Redirected;
            }
            // A primary still starting has not published its window; an exiting
            // primary has withdrawn it. Its mutex also signals crash/exit recovery.
            const auto retry=WaitForSingleObject(mutex_,50);
            if (retry==WAIT_OBJECT_0 || retry==WAIT_ABANDONED) {
                owned_=true;
                return Result::Primary;
            }
            if (retry!=WAIT_TIMEOUT) return Result::Unavailable;
        } while (GetTickCount64()<deadline);
    } catch (...) {}
    return Result::Unavailable;
}

bool SingleInstance::publish(HWND controller) {
    if (!owned_ || !IsWindow(controller)) return false;
    WNDCLASSW type{};
    type.hInstance=GetModuleHandleW(nullptr); type.lpszClassName=EndpointClass;
    type.lpfnWndProc=endpointProcedure;
    if (!RegisterClassW(&type) && GetLastError()!=ERROR_CLASS_ALREADY_EXISTS) return false;
    endpoint_=CreateWindowExW(0,EndpointClass,key_.c_str(),0,0,0,0,0,HWND_MESSAGE,nullptr,type.hInstance,controller);
    return endpoint_!=nullptr;
}

void SingleInstance::stopAccepting() {
    if (endpoint_) { DestroyWindow(endpoint_); endpoint_=nullptr; }
}
}
