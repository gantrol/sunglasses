#pragma once
#include "pixel_cloth.h"
#include <algorithm>
#include <cmath>

namespace sunglasses {
// One bounded ARGB surface for the short pull animation; no native children or
// changing HWND regions are exposed while the fabric is moving.
class ClothPull {
public:
    ~ClothPull() { reset(); }
    ClothPull()=default;
    ClothPull(const ClothPull&)=delete;
    ClothPull& operator=(const ClothPull&)=delete;
    void reset() {
        if (dc_ && previous_) SelectObject(dc_,previous_);
        if (bitmap_) DeleteObject(bitmap_);
        if (dc_) DeleteDC(dc_);
        dc_=nullptr; bitmap_=nullptr; previous_=nullptr; pixels_=nullptr;
    }
    bool prepare(int width,int height) {
        reset();
        // Extremely large text/work areas use the immediate native menu.
        if (width<=0 || height<=0 || static_cast<size_t>(width)*height>1'000'000) return false;
        width_=width; height_=height;
        dc_=CreateCompatibleDC(nullptr);
        BITMAPINFO info{};
        info.bmiHeader={sizeof(BITMAPINFOHEADER),width,-height,1,32,BI_RGB,0,0,0,0,0};
        void* memory=nullptr;
        bitmap_=CreateDIBSection(nullptr,&info,DIB_RGB_COLORS,&memory,nullptr,0);
        if (!dc_ || !bitmap_) { reset(); return false; }
        previous_=SelectObject(dc_,bitmap_); pixels_=static_cast<DWORD*>(memory);
        return true;
    }
    bool present(HWND window,POINT position,int anchor,int unit,float progress) {
        if (!pixels_) return false;
        const auto ease=[](float t) { t=std::clamp(t,0.0f,1.0f); return t*t*(3-2*t); };
        const auto mix=[](int a,int b,float t) { return a+static_cast<int>(std::lround((b-a)*t)); };
        const float length=ease(progress/.65f),spread=ease((progress-.42f)/.58f),release=ease((progress-.70f)/.30f);
        const int half=std::min(unit*9,width_/4);
        anchor=std::clamp(anchor,half,width_-half);
        const int bottom=mix(std::min(unit*8,height_),height_,length);
        const RECT fabric{mix(anchor-half,0,spread),0,mix(anchor+half,width_,spread),bottom};
        const int neckLeft=mix(anchor-half/2,0,release),neckRight=mix(anchor+half/2,width_,release);
        const int shoulder=std::min(bottom/3,mix(unit*8,0,release));
        const POINT outline[]={{neckLeft,0},{neckRight,0},{fabric.right,shoulder},
            {fabric.right,bottom},{fabric.left,bottom},{fabric.left,shoulder}};
        const auto silhouette=CreatePolygonRgn(outline,6,WINDING);
        if (!silhouette) return false;
        const auto hem=pixel::clothRegion(fabric,unit);
        if (!hem) { DeleteObject(silhouette); return false; }
        CombineRgn(silhouette,silhouette,hem,RGN_AND); DeleteObject(hem);
        std::fill_n(pixels_,static_cast<size_t>(width_)*height_,0u);
        const int saved=SaveDC(dc_); SelectClipRgn(dc_,silhouette);
        const auto colors=pixel::palette();
        pixel::drawCloth(dc_,fabric,colors,unit);
        // Two quiet crease lines travel with the fabric and relax as it opens.
        const int crease=static_cast<int>(std::lround(unit*2*(1-release)));
        if (crease>0 && bottom>unit*10) {
            const auto pen=CreatePen(PS_SOLID,crease,pixel::clothPalette(colors).fold);
            if (pen) {
                const auto old=SelectObject(dc_,pen);
                MoveToEx(dc_,neckLeft+unit,unit*2,nullptr);
                LineTo(dc_,fabric.left+(fabric.right-fabric.left)/3,bottom-unit*5);
                MoveToEx(dc_,neckRight-unit,unit*2,nullptr);
                LineTo(dc_,fabric.right-(fabric.right-fabric.left)/4,bottom-unit*5);
                SelectObject(dc_,old); DeleteObject(pen);
            }
        }
        RestoreDC(dc_,saved); DeleteObject(silhouette);
        // GDI writes RGB but not alpha. The dark fabric is opaque; cleared pixels
        // stay transparent. Flush the DIB writes before touching its memory.
        GdiFlush();
        for (size_t i=0,count=static_cast<size_t>(width_)*height_;i<count;++i)
            if (pixels_[i]&0x00ffffffu) pixels_[i]|=0xff000000u;
        POINT source{}; SIZE size{width_,height_};
        BLENDFUNCTION blend{AC_SRC_OVER,0,255,AC_SRC_ALPHA};
        return UpdateLayeredWindow(window,nullptr,&position,&size,dc_,&source,0,&blend,ULW_ALPHA)!=FALSE;
    }
private:
    HDC dc_=nullptr;
    HBITMAP bitmap_=nullptr;
    HGDIOBJ previous_=nullptr;
    DWORD* pixels_=nullptr;
    int width_=0,height_=0;
};
}
