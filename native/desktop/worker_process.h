#pragma once
#include <windows.h>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sunglasses {
struct WorkerEvent { std::string kind, value; };

// One child, bounded output queue, no shell. All reading/waiting is on the background thread.
class WorkerProcess {
public:
    ~WorkerProcess();
    bool start(const std::wstring& executable, const std::vector<std::wstring>& arguments, std::function<void()> notify);
    void cancel();
    void stop();
    std::deque<WorkerEvent> takeEvents();
private:
    void publish(WorkerEvent event);
    HANDLE input_ = nullptr, job_ = nullptr;
    std::thread thread_;
    std::mutex mutex_;
    std::deque<WorkerEvent> events_;
    std::function<void()> notify_;
};
} // namespace sunglasses
