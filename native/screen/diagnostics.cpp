#include "diagnostics.h"
#include <dwmapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <locale>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sunglasses {
namespace {
int64_t qpc() noexcept { LARGE_INTEGER value{}; QueryPerformanceCounter(&value); return value.QuadPart; }
const char* eventName(DiagnosticEvent value) noexcept {
    static constexpr const char* names[]{"app_start","app_stop","overlay_created","overlay_show_requested",
        "overlay_hide_requested","overlay_destroyed","filter_start","filter_stop","strength_changed",
        "capture_status","capture_error","display_changed","session_changed","power_suspend",
        "cloaked","uncloaked","foreground_changed","hooks","telemetry_unavailable","telemetry_dropped","capture_idle",
        "capture_exclusion_configured","capture_exclusion_frame"};
    const auto index=static_cast<size_t>(value);
    return index<std::size(names)?names[index]:"unknown";
}
std::ostringstream json() {
    std::ostringstream stream; stream.imbue(std::locale::classic()); stream<<std::setprecision(17)<<std::boolalpha;
    return stream;
}
std::string guidText(const GUID& value) {
    auto stream=json();
    stream<<std::hex<<std::setfill('0')<<std::setw(8)<<value.Data1<<'-'<<std::setw(4)<<value.Data2
        <<'-'<<std::setw(4)<<value.Data3<<'-';
    for (size_t i=0;i<8;++i) { if (i==2) stream<<'-'; stream<<std::setw(2)<<unsigned(value.Data4[i]); }
    return stream.str();
}
float smooth(float lo,float hi,float value) noexcept {
    const float x=std::clamp((value-lo)/(hi-lo),0.0f,1.0f); return x*x*(3-2*x);
}
struct WriterLease {
    HANDLE handle=INVALID_HANDLE_VALUE;
    explicit WriterLease(const std::filesystem::path& directory) {
        const auto lock=directory/L"screen-diagnostics.lock";
        handle=CreateFileW(lock.c_str(),GENERIC_WRITE|DELETE,0,nullptr,OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
    }
    ~WriterLease() { if (handle!=INVALID_HANDLE_VALUE) CloseHandle(handle); }
};
struct ComApartment {
    HRESULT result=CoInitializeEx(nullptr,COINIT_MULTITHREADED);
    ~ComApartment() { if (SUCCEEDED(result)) CoUninitialize(); }
};
class LogFile {
public:
    LogFile(std::filesystem::path directory,std::string header) : directory_(std::move(directory)),header_(std::move(header)) {
        std::filesystem::create_directories(directory_); rotate();
    }
    void write(const std::string& line) {
        if (bytes_+line.size()>8*1024*1024) rotate();
        file_<<line; bytes_+=line.size();
        if (!file_) throw std::runtime_error("diagnostic_write");
    }
    void flush() { file_.flush(); if (!file_) throw std::runtime_error("diagnostic_flush"); }
private:
    std::filesystem::path path(unsigned index) const {
        return directory_/(L"screen-diagnostics."+std::to_wstring(index)+L".jsonl");
    }
    void rotate() {
        if (file_.is_open()) { flush(); file_.close(); }
        std::filesystem::remove(path(3));
        for (unsigned i=3;i>0;--i) if (std::filesystem::exists(path(i-1))) std::filesystem::rename(path(i-1),path(i));
        file_.open(path(0),std::ios::binary|std::ios::trunc);
        if (!file_) throw std::runtime_error("diagnostic_open");
        file_<<header_; bytes_=header_.size();
    }
    std::filesystem::path directory_;
    std::string header_;
    std::ofstream file_;
    size_t bytes_=0;
};
}

struct ScreenDiagnostics::Impl {
    enum class Kind { Event, Frame, Gpu };
    struct Record {
        Kind kind=Kind::Event;
        int64_t at=0,value=0,detail=0;
        unsigned generation=0;
        DiagnosticEvent event=DiagnosticEvent::AppStart;
        DiagnosticFrame frame;
        ScreenTelemetry gpu;
    };
    struct Watched { HWND window=nullptr; unsigned generation=0; };
    static_assert(sizeof(Record)<=256); // At most 512 KiB in the producer queue.
    static thread_local Impl* hookOwner;
    std::atomic<bool> active=true;
    std::atomic<uint64_t> dropped=0;
    std::mutex mutex;
    std::condition_variable wake;
    std::array<Record,2048> queue{};
    size_t head=0,count=0;
    bool stopping=false;
    Watched watched; // Only copy this while holding mutex; no COM/I/O under mutex.
    Watched hookWindow; // UI thread only, like the hook callbacks.
    HWINEVENTHOOK cloakHook=nullptr,foregroundHook=nullptr;
    std::thread writer;

    void push(Record record) noexcept {
        if (!active.load(std::memory_order_relaxed)) return;
        record.at=qpc();
        std::unique_lock lock(mutex,std::try_to_lock);
        if (!lock.owns_lock()) { ++dropped; return; }
        if (stopping || count==queue.size()) { ++dropped; return; }
        queue[(head+count)%queue.size()]=record; ++count;
        lock.unlock(); wake.notify_one();
    }
    void signal(DiagnosticEvent event,unsigned generation,int64_t value=0,int64_t detail=0) noexcept {
        Record record; record.event=event; record.generation=generation; record.value=value; record.detail=detail; push(record);
    }
    static void CALLBACK winEvent(HWINEVENTHOOK,DWORD event,HWND window,LONG object,LONG child,DWORD,DWORD time) {
        auto* self=hookOwner;
        if (!self || !self->active.load(std::memory_order_relaxed) || !self->hookWindow.window) return;
        if (event==EVENT_SYSTEM_FOREGROUND) {
            // Do not store foreground HWNDs, titles, process names or user input.
            self->signal(DiagnosticEvent::ForegroundChanged,self->hookWindow.generation,time);
        } else if (window==self->hookWindow.window && object==OBJID_WINDOW && child==CHILDID_SELF) {
            self->signal(event==EVENT_OBJECT_CLOAKED?DiagnosticEvent::Cloaked:DiagnosticEvent::Uncloaked,
                self->hookWindow.generation,time);
        }
    }
    void installHooks() noexcept {
        hookOwner=this;
        // Shell can originate cloak events. Filter to our HWND in the callback.
        cloakHook=SetWinEventHook(EVENT_OBJECT_CLOAKED,EVENT_OBJECT_UNCLOAKED,nullptr,winEvent,0,0,WINEVENT_OUTOFCONTEXT);
        foregroundHook=SetWinEventHook(EVENT_SYSTEM_FOREGROUND,EVENT_SYSTEM_FOREGROUND,nullptr,winEvent,0,0,WINEVENT_OUTOFCONTEXT);
        signal(DiagnosticEvent::Hooks,0,cloakHook!=nullptr,foregroundHook!=nullptr);
    }
    void stop() {
        if (cloakHook) UnhookWinEvent(cloakHook);
        if (foregroundHook) UnhookWinEvent(foregroundHook);
        hookOwner=nullptr;
        signal(DiagnosticEvent::AppStop,0);
        { std::lock_guard lock(mutex); stopping=true; }
        wake.notify_one();
        // Called after the window message loop and capture worker have finished.
        if (writer.joinable()) writer.join();
    }
    static void writeRecord(LogFile& file,const Record& record) {
        auto out=json();
        out<<"{\"qpc\":"<<record.at<<",\"generation\":"<<record.generation;
        if (record.kind==Kind::Event) {
            out<<",\"type\":\"event\",\"event\":\""<<eventName(record.event)<<"\",\"value\":"<<record.value<<",\"detail\":"<<record.detail;
        } else if (record.kind==Kind::Frame) {
            const auto& value=record.frame;
            out<<",\"type\":\"frame\",\"frame_id\":"<<value.frameId<<",\"refresh\":"<<value.refresh
                <<",\"capture_ticks_100ns\":"<<value.captureTicks<<",\"submit_qpc\":"<<value.submitQpc
                <<",\"present_begin_qpc\":"<<value.presentBeginQpc<<",\"present_end_qpc\":"<<value.presentEndQpc
                <<",\"capture_age_ms\":"<<value.captureAgeMs<<",\"submit_ms\":"<<value.submitMs
                <<",\"present_wait_ms\":"<<value.presentWaitMs<<",\"idle_seconds\":"<<value.idleSeconds
                <<",\"skipped\":"<<value.skipped<<",\"repeated\":"<<value.repeated<<",\"present_hr\":"<<value.presentResult;
        } else {
            const auto& value=record.gpu;
            const auto& state=value.event;
            if (value.algorithm==ScreenAlgorithm::AppleVfm) {
                out<<",\"type\":\"gpu\",\"algorithm\":\"apple_matlab_vfm_60hz_v1\",\"frame_id\":"<<value.frameId
                    <<",\"draw_id\":"<<value.drawId<<",\"queued_qpc\":"<<value.queuedQpc
                    <<",\"capture_seconds\":"<<value.captureSeconds<<",\"strength\":"<<value.strength
                    <<",\"refresh\":"<<value.refresh<<",\"elapsed_seconds\":"<<value.elapsed
                    <<",\"apl_relative\":"<<state[0]<<",\"adaptation_nits\":"<<state[1]
                    <<",\"response_energy\":"<<state[2]<<",\"normalized_step_energy\":"<<state[3]
                    <<",\"unmasked_score\":"<<state[4]<<",\"score\":"<<state[5]
                    <<",\"applied_mitigation\":"<<state[6]<<",\"next_mitigation\":"<<state[7]
                    <<",\"reference_hz\":60,\"assumed_peak_nits\":100,\"assumed_area_deg2\":1265.625}\n";
                file.write(out.str()); return;
            }
            const float age=state[3]+value.elapsed;
            const float gain=std::min(2*value.strength,1.0f)*state[2]*(1-smooth(.04f,.32f,age))*smooth(.01f,.04f,state[0]);
            out<<",\"type\":\"gpu\",\"frame_id\":"<<value.frameId<<",\"draw_id\":"<<value.drawId
                <<",\"queued_qpc\":"<<value.queuedQpc<<",\"capture_seconds\":"<<value.captureSeconds
                <<",\"strength\":"<<value.strength<<",\"refresh\":"<<value.refresh<<",\"baseline_reset\":"<<value.reset
                <<",\"elapsed_seconds\":"<<value.elapsed<<",\"highlight_load\":"<<state[0]<<",\"mean_luminance\":"<<state[1]
                <<",\"event_peak\":"<<state[2]<<",\"event_age_seconds\":"<<state[3]<<",\"broad_gate\":"<<state[4]
                <<",\"triggered\":"<<(!value.reset && !value.refresh && state[2]>0 && state[3]==0)
                <<",\"estimated_mask_gain\":"<<gain;
        }
        out<<"}\n"; file.write(out.str());
    }
    static void writeWindow(LogFile& file,IVirtualDesktopManager* desktops,HRESULT managerResult,Watched target) {
        if (!target.window) return;
        const auto began=qpc();
        const bool valid=IsWindow(target.window)!=FALSE;
        DWORD cloak=0;
        const HRESULT cloakResult=DwmGetWindowAttribute(target.window,DWMWA_CLOAKED,&cloak,sizeof(cloak));
        BOOL current=FALSE;
        GUID desktop{},foregroundDesktop{};
        const HRESULT currentResult=desktops?desktops->IsWindowOnCurrentVirtualDesktop(target.window,&current):managerResult;
        const HRESULT desktopResult=desktops?desktops->GetWindowDesktopId(target.window,&desktop):managerResult;
        const HWND foreground=GetForegroundWindow();
        const HRESULT foregroundResult=desktops && foreground?desktops->GetWindowDesktopId(foreground,&foregroundDesktop):E_FAIL;
        const bool visible=IsWindowVisible(target.window)!=FALSE,iconic=IsIconic(target.window)!=FALSE;
        const bool topmost=(GetWindowLongPtrW(target.window,GWL_EXSTYLE)&WS_EX_TOPMOST)!=0;
        const auto ended=qpc();
        auto out=json();
        out<<"{\"type\":\"window\",\"qpc\":"<<began<<",\"query_end_qpc\":"<<ended<<",\"generation\":"<<target.generation
            <<",\"valid\":"<<valid<<",\"visible_style\":"<<visible<<",\"iconic\":"<<iconic<<",\"topmost\":"<<topmost
            <<",\"cloak_hr\":"<<cloakResult<<",\"cloak_flags\":"<<cloak<<",\"current_desktop_hr\":"<<currentResult
            <<",\"on_current_desktop\":"<<(current!=FALSE)<<",\"desktop_hr\":"<<desktopResult<<",\"desktop_id\":\""<<guidText(desktop)
            <<"\",\"foreground_desktop_hr\":"<<foregroundResult<<",\"foreground_desktop_id\":\""<<guidText(foregroundDesktop)<<"\"}\n";
        file.write(out.str());
    }
    void run() noexcept {
        try {
            ComApartment apartment;
            ComPtr<IVirtualDesktopManager> desktops;
            HRESULT managerResult=apartment.result;
            if (SUCCEEDED(managerResult)) managerResult=CoCreateInstance(CLSID_VirtualDesktopManager,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&desktops));
            PWSTR raw=nullptr;
            const HRESULT folderResult=SHGetKnownFolderPath(FOLDERID_LocalAppData,0,nullptr,&raw);
            std::unique_ptr<wchar_t,decltype(&CoTaskMemFree)> folder(raw,CoTaskMemFree);
            if (FAILED(folderResult)) throw std::runtime_error("diagnostic_directory");
            const auto directory=std::filesystem::path(folder.get())/L"Sunglasses"/L"Logs";
            std::filesystem::create_directories(directory);
            WriterLease lease(directory);
            if (lease.handle==INVALID_HANDLE_VALUE) throw std::runtime_error("diagnostic_writer_unavailable");
            LARGE_INTEGER frequency{}; QueryPerformanceFrequency(&frequency);
            FILETIME utc{}; GetSystemTimePreciseAsFileTime(&utc);
            const uint64_t utcTicks=(uint64_t(utc.dwHighDateTime)<<32)|utc.dwLowDateTime;
            auto header=json();
            header<<"{\"type\":\"session\",\"schema\":1,\"pid\":"<<GetCurrentProcessId()<<",\"qpc\":"<<qpc()
                <<",\"qpc_frequency\":"<<frequency.QuadPart<<",\"utc_filetime_100ns\":"<<utcTicks
                <<",\"virtual_desktop_manager_hr\":"<<managerResult<<",\"queue_capacity\":"<<queue.size()
                <<",\"window_poll_ms\":100,\"gpu_slots\":4,\"gpu_bytes_per_sample\":32}\n";
            // Synchronous filesystem calls are isolated to this writer. Producers only
            // enqueue bounded POD records; each drain handles at most 128 records.
            LogFile file(directory,header.str());
            std::array<Record,128> batch;
            auto lastWindow=std::chrono::steady_clock::now(),lastFlush=lastWindow;
            uint64_t reportedDrops=0;
            for (;;) {
                size_t available=0;
                Watched target;
                bool finished=false;
                {
                    std::unique_lock lock(mutex);
                    wake.wait_for(lock,std::chrono::milliseconds(100),[&] { return stopping || count; });
                    available=std::min(count,batch.size());
                    for (size_t i=0;i<available;++i) batch[i]=queue[(head+i)%queue.size()];
                    head=(head+available)%queue.size(); count-=available;
                    target=watched; finished=stopping && count==0;
                }
                for (size_t i=0;i<available;++i) writeRecord(file,batch[i]);
                const auto lost=dropped.load();
                if (lost!=reportedDrops) {
                    auto out=json(); out<<"{\"type\":\"log_dropped\",\"qpc\":"<<qpc()<<",\"total\":"<<lost<<"}\n";
                    file.write(out.str()); reportedDrops=lost;
                }
                const auto now=std::chrono::steady_clock::now();
                if (!finished && now-lastWindow>=std::chrono::milliseconds(100)) {
                    writeWindow(file,desktops.Get(),managerResult,target); lastWindow=now;
                }
                if (finished || now-lastFlush>=std::chrono::milliseconds(250)) { file.flush(); lastFlush=now; }
                if (finished) break;
            }
        } catch (...) {
            // Logging failures do not stop capture or change the filter. No dialog.
            OutputDebugStringW(L"Sunglasses diagnostics stopped: log unavailable or another diagnostic writer is active.\n");
        }
        active=false;
    }
};
thread_local ScreenDiagnostics::Impl* ScreenDiagnostics::Impl::hookOwner=nullptr;

ScreenDiagnostics::ScreenDiagnostics() noexcept {
    wchar_t value[2]{};
    if (GetEnvironmentVariableW(L"SUNGLASSES_DIAGNOSTICS",value,2)!=1 || value[0]!=L'1') return;
    try {
        impl_=std::make_unique<Impl>();
        impl_->writer=std::thread([state=impl_.get()] { state->run(); });
        impl_->installHooks();
        impl_->signal(DiagnosticEvent::AppStart,0);
    } catch (...) { impl_.reset(); }
}
ScreenDiagnostics::~ScreenDiagnostics() { if (impl_) impl_->stop(); }
bool ScreenDiagnostics::enabled() const noexcept { return impl_ && impl_->active.load(std::memory_order_relaxed); }
void ScreenDiagnostics::watch(HWND overlay,unsigned generation) noexcept {
    if (!impl_) return;
    impl_->hookWindow={overlay,generation};
    std::lock_guard lock(impl_->mutex); impl_->watched={overlay,generation};
}
void ScreenDiagnostics::event(DiagnosticEvent kind,unsigned generation,int64_t value,int64_t detail) noexcept {
    if (impl_) impl_->signal(kind,generation,value,detail);
}
void ScreenDiagnostics::frame(unsigned generation,const DiagnosticFrame& value) noexcept {
    if (!impl_) return;
    Impl::Record record; record.kind=Impl::Kind::Frame; record.generation=generation; record.frame=value; impl_->push(record);
}
void ScreenDiagnostics::gpu(unsigned generation,const ScreenTelemetry& value) noexcept {
    if (!impl_) return;
    Impl::Record record; record.kind=Impl::Kind::Gpu; record.generation=generation; record.gpu=value; impl_->push(record);
}
}
