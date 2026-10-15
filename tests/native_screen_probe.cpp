// Authorized synthetic sequence runner. No HWND, screen capture or user pixels.
// Blocking file I/O/readback is confined to this offline command, bounded to 256 MiB output.
#include "../native/screen/pipeline.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

template<class T> T read(std::ifstream& file) {
    T value{}; file.read(reinterpret_cast<char*>(&value),sizeof(value));
    if (!file) throw std::runtime_error("truncated input");
    return value;
}
int wmain(int argc,wchar_t** argv) {
    using namespace sunglasses;
    try {
        if (argc!=4 && argc!=5) throw std::runtime_error("input output warp|hardware [apple-vfm]");
        const bool apple=argc==5;
        if (apple && std::wstring(argv[4])!=L"apple-vfm") throw std::runtime_error("algorithm");
        std::ifstream file(argv[1],std::ios::binary);
        char magic[4]{}; file.read(magic,4);
        const auto width=read<uint32_t>(file),height=read<uint32_t>(file),steps=read<uint32_t>(file);
        if (std::memcmp(magic,"SGB2",4) || !width || !height || width>2048 || height>2048 || !steps || steps>600 ||
            uint64_t(width)*height*steps*16>256*1024*1024 || std::filesystem::file_size(argv[1])>256*1024*1024)
            throw std::runtime_error("input format");
        const std::wstring backend(argv[3]);
        if (backend!=L"warp" && backend!=L"hardware") throw std::runtime_error("backend");
        ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context;
        const D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0;
        check(D3D11CreateDevice(nullptr,backend==L"warp"?D3D_DRIVER_TYPE_WARP:D3D_DRIVER_TYPE_HARDWARE,
            nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,&level,1,D3D11_SDK_VERSION,&device,nullptr,&context),"device");
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width=width; desc.Height=height; desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM; desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        ComPtr<ID3D11Texture2D> source,target,staging;
        check(device->CreateTexture2D(&desc,nullptr,&source),"input texture");
        desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; desc.BindFlags=D3D11_BIND_RENDER_TARGET;
        check(device->CreateTexture2D(&desc,nullptr,&target),"output texture");
        ComPtr<ID3D11RenderTargetView> view;
        check(device->CreateRenderTargetView(target.Get(),nullptr,&view),"target view");
        desc.Usage=D3D11_USAGE_STAGING; desc.BindFlags=0; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        check(device->CreateTexture2D(&desc,nullptr,&staging),"staging");
        ScreenPipeline pipeline(device.Get(),context.Get(),apple?ScreenAlgorithm::AppleVfm:ScreenAlgorithm::SuddenHighlights);
        const size_t frameValues=size_t(width)*height*4;
        std::vector<unsigned char> bgra(frameValues);
        std::vector<float> values(frameValues*steps);
        for (uint32_t step=0;step<steps;++step) {
            const auto operation=read<uint32_t>(file);
            const auto time=read<double>(file);
            const auto id=read<uint64_t>(file);
            const auto strength=read<float>(file);
            if (operation==0) {
                file.read(reinterpret_cast<char*>(bgra.data()),static_cast<std::streamsize>(bgra.size()));
                if (!file) throw std::runtime_error("truncated pixels");
                context->UpdateSubresource(source.Get(),0,nullptr,bgra.data(),width*4,0);
                pipeline.render(source.Get(),view.Get(),time,id,strength);
            } else if (operation==1) pipeline.refresh(view.Get(),strength,time);
            else throw std::runtime_error("operation");
            context->CopyResource(staging.Get(),target.Get());
            D3D11_MAPPED_SUBRESOURCE mapped{};
            check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"read synthetic mask");
            for (uint32_t y=0;y<height;++y) std::memcpy(values.data()+size_t(step)*frameValues+size_t(y)*width*4,
                static_cast<const unsigned char*>(mapped.pData)+size_t(y)*mapped.RowPitch,size_t(width)*16);
            context->Unmap(staging.Get(),0);
        }
        if (file.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing input");
        HANDLE output=CreateFileW(argv[2],GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        if (output==INVALID_HANDLE_VALUE) throw std::runtime_error("output exists or unavailable");
        DWORD written=0;
        bool ok=WriteFile(output,values.data(),static_cast<DWORD>(values.size()*4),&written,nullptr)!=FALSE;
        CloseHandle(output);
        if (!ok || written!=values.size()*4) throw std::runtime_error("output write");
        std::cout<<(apple?"synthetic_mask_only apple_matlab_vfm_60hz_v1\n":"synthetic_mask_only\n");
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
