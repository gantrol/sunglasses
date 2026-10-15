#pragma once
#include "pixel_theme.h"
#include <algorithm>

namespace sunglasses::pixel {
struct ClothPalette { COLORREF surface,active,ink,muted,hem,fold; };
inline ClothPalette clothPalette(const Palette& colors) {
    if (colors.highContrast)
        return {colors.lens,colors.accent,colors.ink,GetSysColor(COLOR_GRAYTEXT),colors.ink,colors.ink};
    return {colors.lens,RGB(57,57,57),colors.accent,colors.edge,RGB(76,76,76),RGB(64,64,64)};
}
inline HRGN clothRegion(RECT rect,int tooth) {
    const auto region=CreateRectRgnIndirect(&rect);
    if (!region) return nullptr;
    tooth=std::max(1,tooth);
    const auto notch=[&](RECT cut) {
        const auto part=CreateRectRgnIndirect(&cut);
        if (part) { CombineRgn(region,region,part,RGN_DIFF); DeleteObject(part); }
    };
    for (int x=rect.left;x<rect.right;x+=tooth*4) {
        notch({x,rect.top,x+tooth,rect.top+tooth});
        notch({x+tooth*2,rect.bottom-tooth,x+tooth*3,rect.bottom});
    }
    for (int y=rect.top;y<rect.bottom;y+=tooth*4) {
        notch({rect.left,y,rect.left+tooth,y+tooth});
        notch({rect.right-tooth,y+tooth*2,rect.right,y+tooth*3});
    }
    return region;
}
inline void drawCloth(HDC dc,RECT rect,const Palette& colors,int unit,bool detail=true) {
    unit=std::max(1,unit);
    const auto cloth=clothPalette(colors);
    const auto region=clothRegion(rect,unit);
    const auto brush=CreateSolidBrush(cloth.surface);
    if (region && brush) FillRgn(dc,region,brush);
    if (!detail && region) {
        const auto outline=CreateSolidBrush(colors.edge);
        if (outline) { FrameRgn(dc,region,outline,unit,unit); DeleteObject(outline); }
    }
    if (region) DeleteObject(region); if (brush) DeleteObject(brush);
    if (detail) {
        RECT inset=rect; InflateRect(&inset,-unit*3,-unit*3);
        for (int x=inset.left;x<inset.right-unit;x+=unit*3) {
            fill(dc,{x,inset.top,x+unit,inset.top+unit},cloth.hem);
            fill(dc,{x,inset.bottom-unit,x+unit,inset.bottom},cloth.hem);
        }
        for (int y=inset.top;y<inset.bottom-unit;y+=unit*3) {
            fill(dc,{inset.left,y,inset.left+unit,y+unit},cloth.hem);
            fill(dc,{inset.right-unit,y,inset.right,y+unit},cloth.hem);
        }
    }
    // A stepped folded corner, deliberately without gradients or shadows.
    const int fold=std::min({unit*7,static_cast<int>((rect.right-rect.left)/3),static_cast<int>((rect.bottom-rect.top)/3)});
    for (int y=0;y<fold;y+=unit)
        fill(dc,{rect.right-fold+y,rect.bottom-fold,rect.right,rect.bottom-fold+y+unit},cloth.fold);
}
}
