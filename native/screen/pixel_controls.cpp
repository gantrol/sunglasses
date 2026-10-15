#include "pixel_controls.h"
#include "pixel_theme.h"
#include <commctrl.h>
#include <algorithm>
#include <string>

namespace sunglasses::pixel {
void paintBuffered(HDC target,RECT rect,const std::function<void(HDC)>& draw) {
    const int width=rect.right-rect.left,height=rect.bottom-rect.top;
    if (!target || width<=0 || height<=0) return;
    const auto memory=CreateCompatibleDC(target);
    const auto bitmap=memory?CreateCompatibleBitmap(target,width,height):nullptr;
    const auto previous=bitmap?SelectObject(memory,bitmap):nullptr;
    if (previous && previous!=HGDI_ERROR) {
        // Owner-drawn list/menu items need not start at (0,0).
        SetViewportOrgEx(memory,-rect.left,-rect.top,nullptr);
        draw(memory);
        BitBlt(target,rect.left,rect.top,width,height,memory,rect.left,rect.top,SRCCOPY);
        SelectObject(memory,previous);
    } else draw(target); // Resource failure retains the themed drawing path.
    if (bitmap) DeleteObject(bitmap);
    if (memory) DeleteDC(memory);
}
void drawBufferedItem(DRAWITEMSTRUCT item,const std::function<void(const DRAWITEMSTRUCT&)>& draw) {
    paintBuffered(item.hDC,item.rcItem,[&](HDC dc) { item.hDC=dc; draw(item); });
}
namespace {
enum class ControlKind : DWORD_PTR { Slider,Combo,Button,Static };
int px(HWND window,int value) { return MulDiv(value,static_cast<int>(GetDpiForWindow(window)),96); }
std::wstring comboText(HWND control,int index) {
    if (index<0) return {};
    const auto size=SendMessageW(control,CB_GETLBTEXTLEN,index,0);
    if (size==CB_ERR) return {};
    std::wstring text(static_cast<size_t>(size)+1,L'\0');
    SendMessageW(control,CB_GETLBTEXT,index,reinterpret_cast<LPARAM>(text.data()));
    return text;
}
void comboLabel(HWND control,HDC dc,RECT rect,int index,bool selected,bool disabled) {
    const auto colors=palette();
    fill(dc,rect,selected?colors.accent:colors.field);
    const auto old=SelectObject(dc,reinterpret_cast<HFONT>(SendMessageW(control,WM_GETFONT,0,0)));
    SetBkMode(dc,TRANSPARENT);
    SetTextColor(dc,disabled?GetSysColor(COLOR_GRAYTEXT):selected?
        (colors.highContrast?GetSysColor(COLOR_HIGHLIGHTTEXT):colors.frame):colors.ink);
    rect.left+=px(control,8); rect.right-=px(control,8);
    const auto text=comboText(control,index);
    DrawTextW(dc,text.c_str(),-1,&rect,DT_SINGLELINE|DT_VCENTER|DT_END_ELLIPSIS|DT_NOPREFIX);
    SelectObject(dc,old);
}
void paintControl(HWND control,HDC dc,ControlKind kind) {
    if (kind==ControlKind::Button || kind==ControlKind::Static) {
        DRAWITEMSTRUCT item{};
        item.CtlType=kind==ControlKind::Button?ODT_BUTTON:ODT_STATIC;
        item.CtlID=static_cast<UINT>(GetDlgCtrlID(control)); item.itemAction=ODA_DRAWENTIRE;
        item.hwndItem=control; item.hDC=dc; GetClientRect(control,&item.rcItem);
        if (GetFocus()==control) item.itemState|=ODS_FOCUS;
        if (!IsWindowEnabled(control)) item.itemState|=ODS_DISABLED;
        const auto ui=SendMessageW(control,WM_QUERYUISTATE,0,0);
        if (ui&UISF_HIDEFOCUS) item.itemState|=ODS_NOFOCUSRECT;
        if (ui&UISF_HIDEACCEL) item.itemState|=ODS_NOACCEL;
        if (kind==ControlKind::Button) {
            if (SendMessageW(control,BM_GETSTATE,0,0)&BST_PUSHED) item.itemState|=ODS_SELECTED;
            if (SendMessageW(control,BM_GETCHECK,0,0)==BST_CHECKED) item.itemState|=ODS_CHECKED;
        }
        // The parent buffers WM_DRAWITEM too: native focus/press processing can send
        // it synchronously, outside this subclass's WM_PAINT handler.
        SendMessageW(GetParent(control),WM_DRAWITEM,item.CtlID,reinterpret_cast<LPARAM>(&item));
        return;
    }
    if (kind==ControlKind::Slider) {
        NMCUSTOMDRAW draw{};
        draw.hdr={control,static_cast<UINT_PTR>(GetDlgCtrlID(control)),NM_CUSTOMDRAW};
        draw.dwDrawStage=CDDS_PREPAINT; draw.hdc=dc; GetClientRect(control,&draw.rc);
        SendMessageW(GetParent(control),WM_NOTIFY,draw.hdr.idFrom,reinterpret_cast<LPARAM>(&draw));
        return;
    }
    RECT client{}; GetClientRect(control,&client);
    const auto colors=palette(); const bool focused=GetFocus()==control,enabled=IsWindowEnabled(control)!=FALSE;
    fill(dc,client,focused?colors.ink:colors.edge);
    RECT inner=client; InflateRect(&inner,-px(control,2),-px(control,2));
    fill(dc,inner,colors.field);
    COMBOBOXINFO info{sizeof(info)}; GetComboBoxInfo(control,&info);
    const int arrowWidth=std::max(px(control,28),static_cast<int>(info.rcButton.right-info.rcButton.left));
    RECT arrow{std::max(inner.left,inner.right-arrowWidth),inner.top,inner.right,inner.bottom};
    RECT label=inner; label.right=arrow.left;
    comboLabel(control,dc,label,static_cast<int>(SendMessageW(control,CB_GETCURSEL,0,0)),false,!enabled);
    fill(dc,arrow,colors.lens);
    const int cx=(arrow.left+arrow.right)/2,cy=(arrow.top+arrow.bottom)/2,p=std::max(1,px(control,2));
    const bool open=SendMessageW(control,CB_GETDROPPEDSTATE,0,0)!=0;
    for (int i=0;i<3;++i) {
        const int y=cy+(open?1-i:i-1)*p;
        fill(dc,{cx-(3-i)*p,y,cx+(3-i)*p,y+p},enabled?colors.ink:GetSysColor(COLOR_GRAYTEXT));
    }
    if (focused && (SendMessageW(control,WM_QUERYUISTATE,0,0)&UISF_HIDEFOCUS)==0) {
        InflateRect(&label,-px(control,2),-px(control,2)); DrawFocusRect(dc,&label);
    }
}
LRESULT CALLBACK procedure(HWND control,UINT message,WPARAM wp,LPARAM lp,UINT_PTR id,DWORD_PTR data) {
    const auto kind=static_cast<ControlKind>(data);
    const auto render=[&](HDC dc) {
        if (kind==ControlKind::Button || kind==ControlKind::Static) paintControl(control,dc,kind);
        else {
            RECT rect{}; GetClientRect(control,&rect);
            paintBuffered(dc,rect,[&](HDC memory) { paintControl(control,memory,kind); });
        }
    };
    if (message==WM_ERASEBKGND) return 1;
    if (message==WM_PAINT) {
        // Native trackbars invalidate only the old/new thumb. Our segmented fill changes
        // elsewhere too: expand the update region BEFORE BeginPaint establishes its clip.
        InvalidateRect(control,nullptr,FALSE);
        PAINTSTRUCT paint{}; const auto dc=BeginPaint(control,&paint);
        render(dc);
        EndPaint(control,&paint); return 0;
    }
    if (message==WM_PRINTCLIENT) { render(reinterpret_cast<HDC>(wp)); return 0; }
    if (message==WM_NCDESTROY) RemoveWindowSubclass(control,procedure,id);
    const auto result=DefSubclassProc(control,message,wp,lp);
    switch (message) {
    case WM_SETFOCUS: case WM_KILLFOCUS: case WM_ENABLE: case WM_UPDATEUISTATE:
    case WM_KEYDOWN: case WM_KEYUP: case WM_MOUSEWHEEL: case WM_LBUTTONDOWN: case WM_LBUTTONUP:
    case WM_CAPTURECHANGED: case CB_SETCURSEL: case CB_SHOWDROPDOWN: case TBM_SETPOS:
    case WM_SETTEXT: case BM_SETCHECK: case BM_SETSTATE:
        InvalidateRect(control,nullptr,FALSE); break;
    case WM_MOUSEMOVE: if (GetCapture()==control) InvalidateRect(control,nullptr,FALSE); break;
    }
    return result;
}
}
bool skinSlider(HWND control) { return SetWindowSubclass(control,procedure,0x534c,static_cast<DWORD_PTR>(ControlKind::Slider))!=FALSE; }
bool skinCombo(HWND control) { return SetWindowSubclass(control,procedure,0x5343,static_cast<DWORD_PTR>(ControlKind::Combo))!=FALSE; }
bool skinButton(HWND control) { return SetWindowSubclass(control,procedure,0x5342,static_cast<DWORD_PTR>(ControlKind::Button))!=FALSE; }
bool skinStatic(HWND control) { return SetWindowSubclass(control,procedure,0x5354,static_cast<DWORD_PTR>(ControlKind::Static))!=FALSE; }
void drawComboItem(const DRAWITEMSTRUCT& item) {
    const bool selected=(item.itemState&ODS_SELECTED)!=0,disabled=(item.itemState&ODS_DISABLED)!=0;
    comboLabel(item.hwndItem,item.hDC,item.rcItem,static_cast<int>(item.itemID),selected,disabled);
    if ((item.itemState&ODS_FOCUS)!=0 && (item.itemState&ODS_NOFOCUSRECT)==0) {
        RECT focus=item.rcItem; InflateRect(&focus,-2,-2); DrawFocusRect(item.hDC,&focus);
    }
}
}
