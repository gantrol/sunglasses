#pragma once
#include "pixel_theme.h"
#include <algorithm>
#include <deque>
#include <string>

namespace sunglasses::pixel {
// A native popup keeps system menu navigation/dismissal and text for accessibility.
class Menu {
    struct Item { Menu* owner; std::wstring text; bool separator,submenu; };
    HMENU root_=CreatePopupMenu();
    HFONT font_=nullptr;
    HBRUSH brush_=nullptr;
    UINT dpi_;
    Palette colors_=palette();
    std::deque<Item> items_;
    int px(int value) const { return MulDiv(value,static_cast<int>(dpi_),96); }
    void background(HMENU menu) const {
        MENUINFO info{sizeof(info)}; info.fMask=MIM_BACKGROUND; info.hbrBack=brush_;
        SetMenuInfo(menu,&info);
    }
public:
    explicit Menu(HWND owner):dpi_(GetDpiForWindow(owner)) {
        NONCLIENTMETRICSW metrics{sizeof(metrics)};
        if (SystemParametersInfoForDpi(SPI_GETNONCLIENTMETRICS,sizeof(metrics),&metrics,0,dpi_))
            font_=CreateFontIndirectW(&metrics.lfMenuFont);
        brush_=CreateSolidBrush(colors_.field); if (root_) background(root_);
    }
    ~Menu() { if (root_) DestroyMenu(root_); if (font_) DeleteObject(font_); if (brush_) DeleteObject(brush_); }
    Menu(const Menu&)=delete;
    Menu& operator=(const Menu&)=delete;
    HMENU handle() const { return root_; }
    void add(UINT command,const std::wstring& text,bool enabled=true,bool checked=false,HMENU parent=nullptr,HMENU submenu=nullptr) {
        if (!parent) parent=root_;
        if (!parent) return;
        items_.push_back({this,text,command==0 && !submenu,submenu!=nullptr});
        auto& item=items_.back();
        MENUITEMINFOW info{sizeof(info)};
        info.fMask=MIIM_ID|MIIM_FTYPE|MIIM_STATE|MIIM_DATA|MIIM_STRING;
        info.wID=command; info.fType=MFT_OWNERDRAW|(item.separator?MFT_SEPARATOR:0);
        info.fState=(enabled?MFS_ENABLED:MFS_DISABLED)|(checked?MFS_CHECKED:0);
        info.dwTypeData=item.text.data(); info.dwItemData=reinterpret_cast<ULONG_PTR>(&item);
        if (submenu) { info.fMask|=MIIM_SUBMENU; info.hSubMenu=submenu; }
        InsertMenuItemW(parent,static_cast<UINT>(GetMenuItemCount(parent)),TRUE,&info);
    }
    HMENU submenu(const std::wstring& text) {
        const auto child=CreatePopupMenu(); if (!child) return nullptr;
        background(child); add(0,text,true,false,nullptr,child); return child;
    }
    static bool measure(MEASUREITEMSTRUCT& draw) {
        if (draw.CtlType!=ODT_MENU || !draw.itemData) return false;
        const auto& item=*reinterpret_cast<Item*>(draw.itemData); const auto& self=*item.owner;
        const auto dc=GetDC(nullptr);
        const auto previous=SelectObject(dc,self.font_?self.font_:GetStockObject(DEFAULT_GUI_FONT));
        SIZE extent{}; GetTextExtentPoint32W(dc,item.text.c_str(),static_cast<int>(item.text.size()),&extent);
        TEXTMETRICW metrics{}; GetTextMetricsW(dc,&metrics);
        SelectObject(dc,previous); ReleaseDC(nullptr,dc);
        draw.itemWidth=static_cast<UINT>(extent.cx+self.px(64));
        draw.itemHeight=static_cast<UINT>(item.separator?self.px(9):std::max(self.px(32),static_cast<int>(metrics.tmHeight)+self.px(14)));
        return true;
    }
    static bool draw(const DRAWITEMSTRUCT& draw) {
        if (draw.CtlType!=ODT_MENU || !draw.itemData) return false;
        const auto& item=*reinterpret_cast<Item*>(draw.itemData); const auto& self=*item.owner;
        const bool selected=(draw.itemState&ODS_SELECTED)!=0,disabled=(draw.itemState&ODS_DISABLED)!=0;
        const auto& colors=self.colors_; const auto dc=draw.hDC;
        fill(dc,draw.rcItem,colors.field);
        RECT rect=draw.rcItem;
        if (item.separator) {
            rect.left+=self.px(12); rect.right-=self.px(12); rect.top=(rect.top+rect.bottom)/2; rect.bottom=rect.top+self.px(1);
            fill(dc,rect,colors.edge); return true;
        }
        InflateRect(&rect,-self.px(3),-self.px(2));
        if (selected && !disabled) fill(dc,rect,colors.accent);
        const auto foreground=disabled?GetSysColor(COLOR_GRAYTEXT):selected?
            (colors.highContrast?GetSysColor(COLOR_HIGHLIGHTTEXT):colors.frame):colors.ink;
        const int saved=SaveDC(dc);
        SelectObject(dc,self.font_?self.font_:GetStockObject(DEFAULT_GUI_FONT));
        SetTextColor(dc,foreground); SetBkMode(dc,TRANSPARENT);
        rect.left+=self.px(29); rect.right-=self.px(25);
        DrawTextW(dc,item.text.c_str(),-1,&rect,DT_SINGLELINE|DT_VCENTER|DT_NOPREFIX);
        const int p=std::max(1,self.px(2)),cy=(rect.top+rect.bottom)/2;
        if ((draw.itemState&ODS_CHECKED)!=0) {
            const int x=draw.rcItem.left+self.px(12);
            fill(dc,{x,cy-p,x+p*2,cy+p},foreground);
            fill(dc,{x+p*2,cy,x+p*3,cy+p*2},foreground);
            fill(dc,{x+p*3,cy-p*3,x+p*4,cy+p},foreground);
        }
        if (item.submenu) {
            const int x=draw.rcItem.right-self.px(17);
            for (int i=0;i<3;++i) fill(dc,{x+i*p,cy-(3-i)*p,x+(i+1)*p,cy+(3-i)*p},foreground);
        }
        RestoreDC(dc,saved); return true;
    }
};
}
