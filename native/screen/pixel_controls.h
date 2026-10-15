#pragma once
#include <windows.h>
#include <functional>

namespace sunglasses::pixel {
// Keep native input/accessibility; replace only painting. Sliders still send NM_CUSTOMDRAW.
bool skinSlider(HWND control);
bool skinCombo(HWND control);
// Buttons/statics still use WM_DRAWITEM; native focus, press and accessibility remain intact.
bool skinButton(HWND control);
bool skinStatic(HWND control);
// Draw the complete background/content before copying to the target, preserving rect coordinates.
void paintBuffered(HDC target,RECT rect,const std::function<void(HDC)>& draw);
void drawBufferedItem(DRAWITEMSTRUCT item,const std::function<void(const DRAWITEMSTRUCT&)>& draw);
void drawComboItem(const DRAWITEMSTRUCT& item);
}
