// Headless integration probe for the same process host used by the native dialog.
#include "../native/desktop/worker_process.h"
#include <chrono>
#include <iostream>
#include <memory>

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) return 2;
    std::unique_ptr<void, decltype(&CloseHandle)> event(CreateEventW(nullptr, FALSE, FALSE, nullptr), &CloseHandle);
    if (!event) return 2;
    sunglasses::WorkerProcess worker;
    std::vector<std::wstring> arguments;
    for (int i = 2; i < argc; ++i) arguments.emplace_back(argv[i]);
    if (!worker.start(argv[1], arguments, [&] { SetEvent(event.get()); })) return 2;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(40);
    bool cancelled = false;
    while (std::chrono::steady_clock::now() < deadline) {
        WaitForSingleObject(event.get(), 50);
        for (const auto& item : worker.takeEvents()) {
            std::cout << item.kind << '\t' << item.value << '\n' << std::flush;
            if (item.kind == "exit") return std::stoi(item.value);
        }
        if (!cancelled) {
            DWORD available = 0;
            if (PeekNamedPipe(GetStdHandle(STD_INPUT_HANDLE), nullptr, 0, nullptr, &available, nullptr) && available) {
                char command[32]{};
                DWORD count = 0;
                if (ReadFile(GetStdHandle(STD_INPUT_HANDLE), command, 31, &count, nullptr) && std::string(command).starts_with("cancel")) {
                    worker.cancel(); cancelled = true;
                }
            }
        }
    }
    return 2; // Worker destructor closes its job and joins the reader.
}
