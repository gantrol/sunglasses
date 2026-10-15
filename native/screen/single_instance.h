#pragma once
#include <windows.h>
#include <string>

namespace sunglasses {
inline constexpr UINT RestoreInstanceMessage=WM_APP+48;

// One screen controller per user and Windows session, including portable copies.
// No command line, path or filter command is accepted from another instance.
class SingleInstance {
public:
    enum class Result { Primary, Redirected, Unavailable };
    SingleInstance()=default;
    ~SingleInstance();
    SingleInstance(const SingleInstance&)=delete;
    SingleInstance& operator=(const SingleInstance&)=delete;
    Result acquire() noexcept;
    bool publish(HWND controller);
    void stopAccepting();
private:
    HANDLE mutex_=nullptr;
    HWND endpoint_=nullptr;
    bool owned_=false;
    std::wstring key_;
};
}
