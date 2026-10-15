// Native offline host: platform controls and a single isolated background worker.
#include <windows.h>
#include <commctrl.h>
#include <shobjidl.h>
#include <shlobj.h>
#include <shellapi.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include "resource.h"
#include "worker_process.h"
#include "desktop_strings.h"

using Microsoft::WRL::ComPtr;
namespace {
constexpr UINT WorkerMessage = WM_APP + 1;
struct App {
    HWND window = nullptr;
    bool chinese = PRIMARYLANGID(GetUserDefaultUILanguage()) == LANG_CHINESE;
    bool busy = false, terminal = false, cancelling = false;
    std::wstring resultFolder;
    sunglasses::WorkerProcess worker;
    std::map<int, RECT> layout;
    int baselineWidth = 0, baselineHeight = 0;

    std::wstring text(std::wstring_view id) const {
        for (const auto& entry : sunglasses::strings::entries)
            if (entry.id == id) return std::wstring(chinese ? entry.chinese : entry.english);
        return std::wstring(id);
    }
    void set(int id, const std::wstring& value) { SetDlgItemTextW(window, id, value.c_str()); }
    std::wstring get(int id) const {
        const HWND control = GetDlgItem(window, id);
        std::wstring value(static_cast<size_t>(GetWindowTextLengthW(control)) + 1, L'\0');
        value.resize(static_cast<size_t>(GetWindowTextW(control, value.data(), static_cast<int>(value.size()))));
        return value;
    }
    int task() const { return static_cast<int>(SendDlgItemMessageW(window, IDC_TASK, CB_GETCURSEL, 0, 0)); }
    void status(std::wstring_view key) { set(IDC_STATUS, text(key)); }
    void controls() {
        for (int id : {IDC_TASK, IDC_OUTPUT, IDC_BROWSE_OUTPUT, IDC_STRENGTH, IDC_LANGUAGE, IDC_RUN})
            EnableWindow(GetDlgItem(window, id), !busy);
        for (int id : {IDC_INPUT, IDC_BROWSE_INPUT, IDC_ASSUME})
            EnableWindow(GetDlgItem(window, id), !busy && task() == 0);
        EnableWindow(GetDlgItem(window, IDC_CANCEL), busy);
        EnableWindow(GetDlgItem(window, IDC_OPEN), !busy && !resultFolder.empty());
    }
    void localize() {
        SetWindowTextW(window, text(L"desktop.title").c_str());
        const std::pair<int, const wchar_t*> labels[] = {
            {IDC_TASK_LABEL,L"desktop.task"}, {IDC_INPUT_LABEL,L"desktop.input"}, {IDC_OUTPUT_LABEL,L"desktop.output"},
            {IDC_BROWSE_INPUT,L"desktop.browse"}, {IDC_BROWSE_OUTPUT,L"desktop.browse"}, {IDC_STRENGTH_LABEL,L"desktop.strength"},
            {IDC_ASSUME,L"desktop.assume"}, {IDC_RUN,L"desktop.run"}, {IDC_CANCEL,L"desktop.cancel"}, {IDC_OPEN,L"desktop.open"},
            {IDC_LANGUAGE_LABEL,L"desktop.language"}};
        for (const auto& [id,key] : labels) set(id, text(key));
        int previous = task();
        SendDlgItemMessageW(window, IDC_TASK, CB_RESETCONTENT, 0, 0);
        for (auto key : {L"desktop.image", L"desktop.quick", L"desktop.full"}) {
            auto label = text(key);
            SendDlgItemMessageW(window, IDC_TASK, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(label.c_str()));
        }
        SendDlgItemMessageW(window, IDC_TASK, CB_SETCURSEL, previous < 0 ? 0 : previous, 0);
        status(L"desktop.ready");
        controls();
    }
};

std::wstring fromUtf8(const std::string& value) {
    int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (!length) return {};
    std::wstring result(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), result.data(), length);
    return result;
}
std::filesystem::path applicationFolder() {
    std::vector<wchar_t> path(32768);
    const DWORD count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    return std::filesystem::path(std::wstring(path.data(), count)).parent_path();
}
std::wstring choose(HWND owner, bool folder, const std::wstring& pngLabel) {
    ComPtr<IFileOpenDialog> dialog;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return {};
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
    COMDLG_FILTERSPEC type{pngLabel.c_str(), L"*.png"};
    if (!folder) dialog->SetFileTypes(1, &type);
    if (FAILED(dialog->Show(owner))) return {};
    ComPtr<IShellItem> item;
    PWSTR path = nullptr;
    if (FAILED(dialog->GetResult(&item)) || FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) return {};
    std::wstring result(path);
    CoTaskMemFree(path);
    return result;
}

void rememberLayout(App& app) {
    const int dpi = static_cast<int>(GetDpiForWindow(app.window));
    RECT client{};
    GetClientRect(app.window, &client);
    app.baselineWidth = MulDiv(client.right, 96, dpi);
    app.baselineHeight = MulDiv(client.bottom, 96, dpi);
    for (int id = IDC_TASK_LABEL; id <= IDC_LANGUAGE; ++id) {
        HWND control = GetDlgItem(app.window, id);
        RECT rect{};
        GetWindowRect(control, &rect);
        MapWindowPoints(nullptr, app.window, reinterpret_cast<POINT*>(&rect), 2);
        app.layout[id] = {MulDiv(rect.left, 96, dpi), MulDiv(rect.top, 96, dpi),
                          MulDiv(rect.right, 96, dpi), MulDiv(rect.bottom, 96, dpi)};
    }
}
void resize(App& app) {
    if (app.layout.empty()) return;
    RECT client{};
    GetClientRect(app.window, &client);
    const int dpi = static_cast<int>(GetDpiForWindow(app.window));
    const int dw = MulDiv(client.right, 96, dpi) - app.baselineWidth;
    const int dh = MulDiv(client.bottom, 96, dpi) - app.baselineHeight;
    for (const auto& [id, original] : app.layout) {
        auto rect = original;
        switch (id) {
        case IDC_TASK: case IDC_INPUT: case IDC_OUTPUT: case IDC_STRENGTH: case IDC_ASSUME: case IDC_STATUS: case IDC_PROGRESS:
            rect.right += dw; break;
        case IDC_BROWSE_INPUT: case IDC_BROWSE_OUTPUT: case IDC_STRENGTH_VALUE: case IDC_RUN: case IDC_CANCEL: case IDC_OPEN:
            rect.left += dw; rect.right += dw; break;
        }
        if (id == IDC_RUN || id == IDC_CANCEL || id == IDC_OPEN || id == IDC_PROGRESS) {
            rect.top += dh; rect.bottom += dh;
        }
        if (id == IDC_STATUS) rect.bottom += dh;
        SetWindowPos(GetDlgItem(app.window, id), nullptr, MulDiv(rect.left, dpi, 96), MulDiv(rect.top, dpi, 96),
                     MulDiv(rect.right - rect.left, dpi, 96), MulDiv(rect.bottom - rect.top, dpi, 96), SWP_NOZORDER | SWP_NOACTIVATE);
    }
}

void run(App& app) {
    if (app.busy) return;
    std::filesystem::path input(app.get(IDC_INPUT)), base(app.get(IDC_OUTPUT));
    if (app.task() == 0 && (input.empty() || !input.is_absolute())) { app.status(L"desktop.missing_input"); return; }
    if (base.empty() || !base.is_absolute()) { app.status(L"desktop.invalid_output"); return; }
    GUID guid{};
    if (FAILED(CoCreateGuid(&guid))) { app.status(L"desktop.start_failed"); return; }
    wchar_t identifier[40]{};
    StringFromGUID2(guid, identifier, 40);
    std::wstring suffix(identifier + 1, identifier + 37);
    auto output = base / (app.task() == 0 ? L"filtered-" + suffix + L".png" : L"experiment-" + suffix);
    const int strength = static_cast<int>(SendDlgItemMessageW(app.window, IDC_STRENGTH, TBM_GETPOS, 0, 0));
    std::vector<std::wstring> arguments{L"--language", app.chinese ? L"zh-CN" : L"en", L"--strength", std::to_wstring(strength / 100.0)};
    if (app.task() == 0) {
        arguments.insert(arguments.end(), {L"image", input.wstring(), output.wstring()});
        if (IsDlgButtonChecked(app.window, IDC_ASSUME) == BST_CHECKED) arguments.push_back(L"--assume-srgb");
    } else {
        arguments.insert(arguments.end(), {L"experiment", output.wstring(), L"--preset", app.task() == 1 ? L"quick" : L"full"});
    }
    const auto executable = applicationFolder() / L"runtime" / L"sunglasses-worker.exe";
    app.terminal = false;
    app.cancelling = false;
    app.resultFolder.clear();
    SendDlgItemMessageW(app.window, IDC_PROGRESS, PBM_SETPOS, 0, 0);
    if (!app.worker.start(executable.wstring(), arguments, [window=app.window] { PostMessageW(window, WorkerMessage, 0, 0); })) {
        app.status(L"desktop.start_failed"); app.controls(); return;
    }
    app.busy = true;
    app.status(L"desktop.running"); app.controls();
    SetFocus(GetDlgItem(app.window, IDC_CANCEL));
}

INT_PTR CALLBACK dialog(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    auto app = reinterpret_cast<App*>(GetWindowLongPtrW(window, DWLP_USER));
    if (message == WM_INITDIALOG) {
        app = reinterpret_cast<App*>(lparam);
        app->window = window;
        SetWindowLongPtrW(window, DWLP_USER, lparam);
        SendDlgItemMessageW(window, IDC_STRENGTH, TBM_SETRANGE, TRUE, MAKELPARAM(0, 100));
        SendDlgItemMessageW(window, IDC_STRENGTH, TBM_SETPOS, TRUE, 100);
        SendDlgItemMessageW(window, IDC_PROGRESS, PBM_SETRANGE32, 0, 100);
        for (auto key : {L"desktop.english",L"desktop.chinese"}) {
            auto label = app->text(key);
            SendDlgItemMessageW(window, IDC_LANGUAGE, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(label.c_str()));
        }
        SendDlgItemMessageW(window, IDC_LANGUAGE, CB_SETCURSEL, app->chinese ? 1 : 0, 0);
        PWSTR local = nullptr;
        if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_DEFAULT, nullptr, &local))) {
            app->set(IDC_OUTPUT, (std::filesystem::path(local) / L"Sunglasses" / L"results").wstring());
            CoTaskMemFree(local);
        }
        SendDlgItemMessageW(window, IDC_INPUT, EM_SETLIMITTEXT, 32760, 0);
        SendDlgItemMessageW(window, IDC_OUTPUT, EM_SETLIMITTEXT, 32760, 0);
        app->localize(); rememberLayout(*app);
        return TRUE;
    }
    if (!app) return FALSE;
    switch (message) {
    case WM_SIZE: resize(*app); return TRUE;
    case WM_GETMINMAXINFO: {
        if (!app->baselineWidth) break;
        const UINT dpi = GetDpiForWindow(window);
        RECT minimum{0,0,MulDiv(app->baselineWidth, static_cast<int>(dpi),96),MulDiv(app->baselineHeight,static_cast<int>(dpi),96)};
        AdjustWindowRectExForDpi(&minimum, static_cast<DWORD>(GetWindowLongPtrW(window,GWL_STYLE)), FALSE,
                                static_cast<DWORD>(GetWindowLongPtrW(window,GWL_EXSTYLE)), dpi);
        auto info = reinterpret_cast<MINMAXINFO*>(lparam);
        info->ptMinTrackSize = {minimum.right-minimum.left,minimum.bottom-minimum.top};
        return TRUE;
    }
    case WM_HSCROLL:
        app->set(IDC_STRENGTH_VALUE,std::to_wstring(SendDlgItemMessageW(window,IDC_STRENGTH,TBM_GETPOS,0,0)));
        return TRUE;
    case WorkerMessage:
        for (const auto& event : app->worker.takeEvents()) {
            if (event.kind == "progress") {
                try { SendDlgItemMessageW(window,IDC_PROGRESS,PBM_SETPOS, std::clamp(std::stoi(event.value),0,100),0); }
                catch (...) { /* Malformed diagnostics do not execute host actions. */ }
            } else if (event.kind == "error") {
                app->set(IDC_STATUS,fromUtf8(event.value)); app->terminal = true;
            } else if (event.kind == "cancelled") {
                app->status(L"desktop.cancelled"); app->terminal = true;
            } else if (event.kind == "done") {
                app->resultFolder = std::filesystem::path(fromUtf8(event.value)).parent_path().wstring();
                app->status(L"desktop.done"); app->terminal = true;
                SendDlgItemMessageW(window,IDC_PROGRESS,PBM_SETPOS,100,0);
            } else if (event.kind == "exit") {
                if (!app->terminal || (event.value != "0" && !app->resultFolder.empty())) {
                    app->resultFolder.clear(); app->status(L"desktop.failed");
                }
                app->busy = false; app->controls(); SetFocus(GetDlgItem(window,IDC_RUN));
            }
        }
        return TRUE;
    case WM_COMMAND:
        switch (LOWORD(wparam)) {
        case IDC_TASK: if (HIWORD(wparam)==CBN_SELCHANGE) app->controls(); break;
        case IDC_LANGUAGE:
            if (HIWORD(wparam)==CBN_SELCHANGE) { app->chinese = SendDlgItemMessageW(window,IDC_LANGUAGE,CB_GETCURSEL,0,0)==1; app->localize(); }
            break;
        case IDC_BROWSE_INPUT: case IDC_BROWSE_OUTPUT: {
            const bool folder = LOWORD(wparam)==IDC_BROWSE_OUTPUT;
            auto selected = choose(window,folder,app->text(L"desktop.png"));
            if (!selected.empty()) app->set(folder ? IDC_OUTPUT : IDC_INPUT,selected);
            break;
        }
        case IDOK: case IDC_RUN: run(*app); break;
        case IDCANCEL:
            if (!app->busy) { EndDialog(window,0); break; }
            [[fallthrough]];
        case IDC_CANCEL:
            if (app->busy && !app->cancelling) {
                app->cancelling = true; app->worker.cancel(); app->status(L"desktop.cancelling");
                EnableWindow(GetDlgItem(window,IDC_CANCEL),FALSE);
            }
            break;
        case IDC_OPEN:
            if (!app->busy && !app->resultFolder.empty() && reinterpret_cast<INT_PTR>(ShellExecuteW(window,L"open",app->resultFolder.c_str(),nullptr,nullptr,SW_SHOWNORMAL))<=32)
                app->status(L"desktop.open_failed");
            break;
        }
        return TRUE;
    case WM_CLOSE:
        EndDialog(window,0); return TRUE; // Destructor terminates the child after the window closes.
    }
    return FALSE;
}
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED))) return 2;
    INITCOMMONCONTROLSEX controls{sizeof(controls),ICC_STANDARD_CLASSES | ICC_BAR_CLASSES | ICC_PROGRESS_CLASS};
    InitCommonControlsEx(&controls);
    int result = 2;
    {
        App app;
        result = static_cast<int>(DialogBoxParamW(instance,MAKEINTRESOURCEW(IDD_MAIN),nullptr,dialog,reinterpret_cast<LPARAM>(&app)));
    }
    CoUninitialize();
    return result;
}
