#pragma once
#include <windows.h>

namespace sunglasses::pixel {
struct Palette {
    bool highContrast;
    COLORREF frame,lens,ink,accent,field,edge;
};
inline Palette palette(bool on=true) {
    HIGHCONTRASTW hc{sizeof(hc)}; SystemParametersInfoW(SPI_GETHIGHCONTRAST,sizeof(hc),&hc,0);
    if ((hc.dwFlags&HCF_HIGHCONTRASTON)!=0)
        return {true,GetSysColor(COLOR_WINDOWTEXT),GetSysColor(COLOR_WINDOW),GetSysColor(COLOR_WINDOWTEXT),
            GetSysColor(COLOR_HIGHLIGHT),GetSysColor(COLOR_WINDOW),GetSysColor(COLOR_WINDOWTEXT)};
    return {false,RGB(23,23,23),on?RGB(43,43,43):RGB(36,36,36),RGB(242,242,242),
        RGB(199,199,199),RGB(32,32,32),RGB(130,130,130)};
}
inline void fill(HDC dc,RECT rect,COLORREF color) {
    const auto brush=CreateSolidBrush(color); FillRect(dc,&rect,brush); DeleteObject(brush);
}
}
