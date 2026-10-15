#include "worker_process.h"
#include <array>
#include <utility>

namespace sunglasses {
namespace {
struct Handle {
    HANDLE value = nullptr;
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    HANDLE release() { return std::exchange(value, nullptr); }
};
std::wstring quote(const std::wstring& argument) {
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (wchar_t c : argument) {
        if (c == L'\\') { ++slashes; continue; }
        result.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
        result += c;
        slashes = 0;
    }
    result.append(2 * slashes, L'\\');
    return result + L'"';
}
}

WorkerProcess::~WorkerProcess() { stop(); }
void WorkerProcess::stop() {
    if (input_) { CloseHandle(input_); input_ = nullptr; }
    if (job_) { CloseHandle(job_); job_ = nullptr; }
    if (thread_.joinable()) thread_.join();
}
void WorkerProcess::cancel() {
    // One seven-byte control message per task, never file or frame data.
    if (input_) { DWORD written = 0; WriteFile(input_, "cancel\n", 7, &written, nullptr); }
}
std::deque<WorkerEvent> WorkerProcess::takeEvents() {
    std::lock_guard lock(mutex_);
    std::deque<WorkerEvent> result;
    result.swap(events_);
    return result;
}
void WorkerProcess::publish(WorkerEvent event) {
    {
        std::lock_guard lock(mutex_);
        if (events_.size() == 32) events_.pop_front();
        events_.push_back(std::move(event));
    }
    notify_();
}

bool WorkerProcess::start(const std::wstring& executable, const std::vector<std::wstring>& arguments, std::function<void()> notify) {
    stop();
    { std::lock_guard lock(mutex_); events_.clear(); }
    notify_ = std::move(notify);
    SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
    Handle childIn, parentIn, parentOut, childOut;
    if (!CreatePipe(&childIn.value, &parentIn.value, &security, 0) ||
        !CreatePipe(&parentOut.value, &childOut.value, &security, 0)) return false;
    if (!SetHandleInformation(parentIn.value, HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(parentOut.value, HANDLE_FLAG_INHERIT, 0)) return false;
    Handle job{CreateJobObjectW(nullptr, nullptr)};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job.value || !SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) return false;
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<unsigned char> storage(bytes);
    auto attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if (!InitializeProcThreadAttributeList(attributes, 1, 0, &bytes)) return false;
    HANDLE inherited[] = {childIn.value, childOut.value};
    const bool updated = UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                                   inherited, sizeof(inherited), nullptr, nullptr) != FALSE;
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    startup.StartupInfo.wShowWindow = SW_HIDE;
    startup.StartupInfo.hStdInput = childIn.value;
    startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = childOut.value;
    startup.lpAttributeList = attributes;
    std::wstring command = quote(executable);
    for (const auto& argument : arguments) command += L" " + quote(argument);
    PROCESS_INFORMATION process{};
    const bool launched = updated && CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo, &process);
    DeleteProcThreadAttributeList(attributes);
    if (!launched) return false;
    Handle processHandle{process.hProcess}, processThread{process.hThread};
    if (!AssignProcessToJobObject(job.value, process.hProcess) || ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
        TerminateProcess(process.hProcess, 2);
        return false;
    }
    input_ = parentIn.release();
    job_ = job.release();
    HANDLE output = parentOut.release(), running = processHandle.release();
    thread_ = std::thread([this, output, running] {
        Handle outputHandle{output}, runningHandle{running};
        std::string pending;
        std::array<char, 4096> buffer{};
        DWORD count = 0;
        bool tooLong = false;
        while (ReadFile(output, buffer.data(), static_cast<DWORD>(buffer.size()), &count, nullptr) && count) {
            pending.append(buffer.data(), count);
            size_t end = 0;
            while ((end = pending.find('\n')) != std::string::npos) {
                if (end > 8192) { tooLong = true; break; }
                std::string line = pending.substr(0, end);
                pending.erase(0, end + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                const auto tab = line.find('\t');
                if (tab != std::string::npos) publish({line.substr(0, tab), line.substr(tab + 1)});
            }
            if (tooLong || pending.size() > 8192) { TerminateProcess(running, 2); break; }
        }
        WaitForSingleObject(running, INFINITE);
        DWORD exitCode = 2;
        GetExitCodeProcess(running, &exitCode);
        publish({"exit", std::to_string(exitCode)});
    });
    return true;
}
} // namespace sunglasses
