#pragma once
#include "pixel_theme.h"
#include <array>

namespace sunglasses::pixel {
inline constexpr int GlassesWidth=600,GlassesHeight=200;
inline constexpr POINT GlassesOutline[]={
    {0,28},{20,28},{20,18},{48,18},{48,10},{210,10},{210,16},{240,16},
    {240,24},{264,24},{264,32},{336,32},{336,24},{360,24},{360,16},
    {390,16},{390,10},{552,10},{552,18},{580,18},{580,28},{600,28},
    {600,68},{580,68},{580,106},{570,106},{570,138},{554,138},{554,162},
    {530,162},{530,180},{378,180},{378,170},{354,170},{354,154},{340,154},
    {340,130},{330,130},{330,98},{322,98},{322,70},{278,70},{278,98},
    {270,98},{270,130},{260,130},{260,154},{246,154},{246,170},{222,170},
    {222,180},{70,180},{70,162},{46,162},{46,138},{30,138},{30,106},
    {20,106},{20,68},{0,68}
};
inline constexpr POINT GlassesLens[]={
    {50,42},{70,42},{70,32},{212,32},{212,40},{236,40},{236,50},{250,50},
    {250,102},{240,102},{240,130},{226,130},{226,148},{206,148},{206,158},
    {84,158},{84,148},{64,148},{64,130},{50,130}
};
template<size_t N> HRGN glassesPolygon(RECT rect,const POINT (&points)[N],bool mirror=false) {
    std::array<POINT,N> scaled{};
    for (size_t i=0;i<N;++i) scaled[i]={
        rect.left+MulDiv(mirror?GlassesWidth-points[i].x:points[i].x,rect.right-rect.left,GlassesWidth),
        rect.top+MulDiv(points[i].y,rect.bottom-rect.top,GlassesHeight)};
    return CreatePolygonRgn(scaled.data(),static_cast<int>(N),WINDING);
}
inline void drawGlasses(HDC dc,RECT rect,const Palette& colors) {
    const auto draw=[&](HRGN region,COLORREF color) {
        const auto brush=CreateSolidBrush(color);
        if (region && brush) FillRgn(dc,region,brush);
        if (brush) DeleteObject(brush); if (region) DeleteObject(region);
    };
    draw(glassesPolygon(rect,GlassesOutline),colors.frame);
    for (bool mirror:{false,true}) draw(glassesPolygon(rect,GlassesLens,mirror),colors.lens);
}
}
