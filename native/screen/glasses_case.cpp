#include "glasses_case.h"
#include "pixel_theme.h"
#include "pixel_controls.h"
#include "pixel_glasses.h"
#include "resource.h"
#include <windowsx.h>
#include <commctrl.h>
#include <uxtheme.h>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>

namespace sunglasses {
namespace {
constexpr int Controls[]={IDC_SCREEN_LABEL,IDC_SCREEN,IDC_MODE_LABEL,IDC_MODE,IDC_LEVEL_LABEL,IDC_LEVEL,
    IDC_LEVEL_VALUE,IDC_LOCALE_LABEL,IDC_LOCALE,IDC_STATE,IDC_STOP,IDC_SWITCH,IDC_CASE_TITLE,IDC_CASE_CLOSE,
    IDC_RECORDING_LABEL,IDC_RECORDING};
constexpr UINT_PTR MotionTimer=0x5347;
using pixel::fill;
// Both halves use the same stepped shell, without bitmap stretching.
HRGN stepped(RECT r,int step) {
    const POINT points[]={
        {r.left+step*2,r.top},{r.right-step*2,r.top},{r.right-step*2,r.top+step},
        {r.right-step,r.top+step},{r.right-step,r.top+step*2},{r.right,r.top+step*2},
        {r.right,r.bottom-step*2},{r.right-step,r.bottom-step*2},{r.right-step,r.bottom-step},
        {r.right-step*2,r.bottom-step},{r.right-step*2,r.bottom},{r.left+step*2,r.bottom},
        {r.left+step*2,r.bottom-step},{r.left+step,r.bottom-step},{r.left+step,r.bottom-step*2},
        {r.left,r.bottom-step*2},{r.left,r.top+step*2},{r.left+step,r.top+step*2},
        {r.left+step,r.top+step},{r.left+step*2,r.top+step}
    };
    return CreatePolygonRgn(points,static_cast<int>(std::size(points)),WINDING);
}
void panel(HDC dc,RECT r,int step,COLORREF color) {
    const auto region=stepped(r,step);
    const auto brush=CreateSolidBrush(color);
    if (region && brush) FillRgn(dc,region,brush);
    if (brush) DeleteObject(brush); if (region) DeleteObject(region);
}
RECT inset(RECT r,int amount) { InflateRect(&r,-amount,-amount); return r; }
struct CaseGeometry {
    RECT lid,base;
    RECT hinges[2];
};
float ease(float value) { value=std::clamp(value,0.0f,1.0f); return value*value*(3-2*value); }
RECT interpolate(RECT from,RECT to,float progress) {
    const auto mix=[&](LONG a,LONG b) { return a+static_cast<LONG>(std::lround((b-a)*progress)); };
    return {mix(from.left,to.left),mix(from.top,to.top),mix(from.right,to.right),mix(from.bottom,to.bottom)};
}
}

struct GlassesCase::Impl {
    HWND hwnd=nullptr;
    HFONT font=nullptr,titleFont=nullptr;
    HBRUSH liningBrush=nullptr,fieldBrush=nullptr;
    pixel::Palette colors=pixel::palette();
    UINT dpi=96;
    int textScale=100,width=620,height=552;
    bool layingOut=false;
    enum class Motion { None,Opening,Closing };
    Motion motion=Motion::None;
    ULONGLONG motionStarted=0;
    float progress=1,startedProgress=1;
    float anchorX=.5f,anchorY=.75f;
    int glassesWidth=480;
    pixel::Palette glassesColors=pixel::palette(false);

    ~Impl() {
        if (IsWindow(hwnd)) {
            KillTimer(hwnd,MotionTimer);
            RemoveWindowSubclass(hwnd,procedure,1);
            for (int id:{IDC_STOP,IDC_SWITCH,IDC_CASE_CLOSE,IDC_LEVEL}) RemoveWindowSubclass(GetDlgItem(hwnd,id),buttonProcedure,1);
        }
        if (font) DeleteObject(font); if (titleFont) DeleteObject(titleFont);
        if (liningBrush) DeleteObject(liningBrush); if (fieldBrush) DeleteObject(fieldBrush);
    }
    int px(int value) const { return MulDiv(value,static_cast<int>(dpi),96); }
    int unit(int value) const { return MulDiv(px(value),textScale,100); }
    bool motionAllowed() const {
        BOOL enabled=FALSE;
        return !colors.highContrast && SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION,0,&enabled,0) && enabled;
    }
    void endMotion(bool opening) {
        KillTimer(hwnd,MotionTimer);
        // Hand off at the exact drawn rectangle BEFORE removing the case window.
        // One clock paints both objects; no independently flying/fading HWND remains.
        if (!opening) {
            SendMessageW(hwnd,CaseHiddenMessage,GetForegroundWindow()==hwnd,0);
            ShowWindow(hwnd,SW_HIDE);
        }
        motion=Motion::None; progress=1;
        for (int id:Controls) ShowWindow(GetDlgItem(hwnd,id),SW_SHOWNA);
        layout();
        if (opening) ShowWindow(hwnd,SW_SHOWNOACTIVATE);
        if (opening) PostMessageW(hwnd,CaseOpenedMessage,0,0);
    }
    void motionFrame() {
        const bool opening=motion==Motion::Opening;
        const float elapsed=static_cast<float>(GetTickCount64()-motionStarted)/420.0f;
        progress=std::clamp(startedProgress+(opening?elapsed:-elapsed),0.0f,1.0f);
        if ((opening && progress>=1) || (!opening && progress<=0)) { endMotion(opening); return; }
        updateRegion(); InvalidateRect(hwnd,nullptr,FALSE);
    }
    void settleForMinimize() {
        // A timer must not finish opening/closing after the shell has minimized us.
        KillTimer(hwnd,MotionTimer);
        motion=Motion::None; progress=1;
        for (int id:Controls) ShowWindow(GetDlgItem(hwnd,id),SW_SHOWNA);
    }
    void animate(bool opening,bool activate=true) {
        if ((motion==Motion::Opening && opening) || (motion==Motion::Closing && !opening)) return;
        if (motion==Motion::None) {
            if (opening && IsWindowVisible(hwnd)) {
                ShowWindow(hwnd,activate?SW_RESTORE:SW_SHOWNOACTIVATE); return;
            }
            if (!opening && !IsWindowVisible(hwnd)) return;
            fit(); progress=opening?0.0f:1.0f;
        }
        if (!activate || !motionAllowed()) { endMotion(opening); return; }
        for (int id:Controls) {
            if (id==IDC_SCREEN || id==IDC_MODE || id==IDC_RECORDING || id==IDC_LOCALE)
                SendDlgItemMessageW(hwnd,id,CB_SHOWDROPDOWN,FALSE,0);
            ShowWindow(GetDlgItem(hwnd,id),SW_HIDE);
        }
        SetFocus(hwnd);
        startedProgress=progress; motionStarted=GetTickCount64(); motion=opening?Motion::Opening:Motion::Closing;
        motionFrame();
        if (opening) {
            ShowWindow(hwnd,SW_SHOWNOACTIVATE);
            SetWindowPos(hwnd,HWND_TOP,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE|SWP_NOACTIVATE);
            RedrawWindow(hwnd,nullptr,nullptr,RDW_INVALIDATE|RDW_UPDATENOW);
        }
        if (motion!=Motion::None && !SetTimer(hwnd,MotionTimer,15,nullptr)) endMotion(opening);
    }
    CaseGeometry geometry() const {
        // An odd spare pixel belongs to the hinge gap, so the shells stay exactly equal.
        const int half=(height-unit(12))/2;
        CaseGeometry result{{0,0,width,half},{0,height-half,width,height},{}};
        int i=0;
        for (int center:{width/4,width-width/4})
            result.hinges[i++]={center-unit(22),half-unit(6),center+unit(22),height-half+unit(6)};
        return result;
    }
    RECT glassesRect() const {
        const int margin=0;
        const int w=std::max(3,std::min({glassesWidth,width-margin*2,(height-margin*2)*3})),h=w/3;
        const int left=std::clamp(static_cast<int>(std::lround(width*anchorX))-w/2,margin,std::max(margin,width-margin-w));
        const int top=std::clamp(static_cast<int>(std::lround(height*anchorY))-h/2,margin,std::max(margin,height-margin-h));
        return {left,top,left+w,top+h};
    }
    void prepare(RECT source) {
        if (motion!=Motion::None || IsWindowVisible(hwnd)) return;
        MONITORINFO monitor{sizeof(monitor)};
        if (!GetMonitorInfoW(MonitorFromRect(&source,MONITOR_DEFAULTTONEAREST),&monitor)) return;
        const auto& work=monitor.rcWork;
        const int w=std::min(std::max(width,static_cast<int>(source.right-source.left)+unit(32)),static_cast<int>(work.right-work.left));
        const int h=std::min(height,static_cast<int>(work.bottom-work.top));
        const int left=std::clamp((source.left+source.right)/2-w/2,work.left,work.right-w);
        const int top=std::clamp((source.top+source.bottom)/2-h*3/4,work.top,work.bottom-h);
        SetWindowPos(hwnd,nullptr,left,top,w,h,SWP_NOACTIVATE|SWP_NOZORDER);
        // DPI changes may have adjusted the window while positioning it.
        RECT actual{}; GetWindowRect(hwnd,&actual);
        glassesWidth=source.right-source.left;
        anchorX=static_cast<float>((source.left+source.right)/2-actual.left)/std::max(1,width);
        anchorY=static_cast<float>((source.top+source.bottom)/2-actual.top)/std::max(1,height);
    }
    CaseGeometry animatedGeometry() const {
        auto result=geometry();
        if (motion==Motion::None) return result;
        RECT wrapped=glassesRect();
        const int padding=static_cast<int>(std::lround(unit(12)*ease(progress/.22f)));
        InflateRect(&wrapped,padding,padding);
        const float unfold=ease((progress-.45f)/.55f);
        result.base=interpolate(wrapped,result.base,unfold);
        const auto openLid=result.lid;
        const int hinge=result.base.top;
        result.lid=result.base;
        if (unfold>=.5f) {
            const float extent=(unfold-.5f)*2;
            result.lid.top=hinge-static_cast<int>(std::lround((hinge-openLid.top)*extent));
            result.lid.bottom=std::max(result.lid.top+unit(6),static_cast<LONG>(hinge-std::lround((hinge-openLid.bottom)*extent)));
        } else {
            result.lid.top=hinge;
            result.lid.bottom=hinge+std::max(unit(6),static_cast<int>(std::lround((result.base.bottom-hinge)*(1-unfold*2))));
        }
        return result;
    }
    void updateRegion() {
        const auto shape=animatedGeometry();
        const int lidStep=std::max(1,std::min(unit(10),static_cast<int>((shape.lid.bottom-shape.lid.top)/4)));
        const auto region=stepped(shape.lid,lidStep),base=stepped(shape.base,unit(10));
        if (region && base) {
            CombineRgn(region,region,base,RGN_OR);
            if (motion==Motion::None || progress>.85f) for (const auto& hinge:shape.hinges) {
                const auto joint=CreateRectRgnIndirect(&hinge);
                if (joint) { CombineRgn(region,region,joint,RGN_OR); DeleteObject(joint); }
            }
            if (motion!=Motion::None && progress<.22f) {
                const auto glasses=glassesRect();
                const int center=(shape.base.top+shape.base.bottom)/2;
                const int extent=static_cast<int>(std::lround((shape.base.bottom-shape.base.top)*ease(progress/.22f)/2));
                const auto envelope=CreateRectRgn(shape.base.left,center-extent,shape.base.right,center+extent);
                const auto silhouette=pixel::glassesPolygon(glasses,pixel::GlassesOutline);
                if (envelope) { CombineRgn(region,region,envelope,RGN_AND); DeleteObject(envelope); }
                if (silhouette) { CombineRgn(region,region,silhouette,RGN_OR); DeleteObject(silhouette); }
            }
            if (!SetWindowRgn(hwnd,region,TRUE)) DeleteObject(region);
        } else if (region) DeleteObject(region);
        if (base) DeleteObject(base);
    }
    void refreshTheme() {
        colors=pixel::palette();
        if (liningBrush) DeleteObject(liningBrush); if (fieldBrush) DeleteObject(fieldBrush);
        liningBrush=CreateSolidBrush(colors.lens); fieldBrush=CreateSolidBrush(colors.field);
    }
    void refreshFont() {
        const auto previousDpi=dpi;
        dpi=GetDpiForWindow(hwnd);
        glassesWidth=MulDiv(glassesWidth,static_cast<int>(dpi),static_cast<int>(previousDpi));
        NONCLIENTMETRICSW metrics{sizeof(metrics)};
        if (!SystemParametersInfoForDpi(SPI_GETNONCLIENTMETRICS,sizeof(metrics),&metrics,0,dpi)) return;
        // Layout grows with the user's system message font as well as monitor DPI.
        const int fontPixels=std::max(px(14),static_cast<int>(std::abs(metrics.lfMessageFont.lfHeight)));
        textScale=std::max(100,MulDiv(fontPixels,100,std::max(1,px(14))));
        metrics.lfMessageFont.lfHeight=-fontPixels;
        const auto next=CreateFontIndirectW(&metrics.lfMessageFont);
        if (next) {
            for (int id:Controls) SendDlgItemMessageW(hwnd,id,WM_SETFONT,reinterpret_cast<WPARAM>(next),FALSE);
            if (font) DeleteObject(font); font=next;
        }
        metrics.lfMessageFont.lfHeight=-unit(20); metrics.lfMessageFont.lfWeight=FW_SEMIBOLD;
        const auto heading=CreateFontIndirectW(&metrics.lfMessageFont);
        if (heading) {
            SendDlgItemMessageW(hwnd,IDC_CASE_TITLE,WM_SETFONT,reinterpret_cast<WPARAM>(heading),FALSE);
            if (titleFont) DeleteObject(titleFont); titleFont=heading;
        }
        for (int id:{IDC_SCREEN,IDC_MODE,IDC_LOCALE,IDC_RECORDING}) {
            SendDlgItemMessageW(hwnd,id,CB_SETITEMHEIGHT,static_cast<WPARAM>(-1),unit(24));
            SendDlgItemMessageW(hwnd,id,CB_SETITEMHEIGHT,0,unit(24));
        }
        SendDlgItemMessageW(hwnd,IDC_LEVEL,TBM_SETTHUMBLENGTH,unit(22),0);
    }
    void layout() {
        if (layingOut || IsIconic(hwnd)) return;
        layingOut=true;
        RECT client{}; GetClientRect(hwnd,&client); width=client.right; height=client.bottom;
        const auto shape=geometry();
        updateRegion();
        const auto place=[&](int id,int x,int y,int w,int h) {
            SetWindowPos(GetDlgItem(hwnd,id),nullptr,x,y,std::max(1,w),std::max(1,h),SWP_NOZORDER|SWP_NOACTIVATE);
        };
        const int left=unit(34),content=unit(154),right=width-unit(34),row=unit(40);
        const int top=shape.base.top+((shape.base.bottom-shape.base.top)-unit(194))/2;
        place(IDC_CASE_TITLE,unit(104),unit(30),right-unit(104),unit(28));
        const int stateTop=unit(84),closeWidth=unit(112),gap=unit(16);
        const int actionsTop=shape.lid.bottom-unit(66),buttonWidth=(right-left-gap)/2;
        place(IDC_STATE,left,stateTop+unit(6),right-left-closeWidth-gap,actionsTop-stateTop-unit(20));
        place(IDC_CASE_CLOSE,right-closeWidth,stateTop,closeWidth,unit(34));
        place(IDC_STOP,left,actionsTop,buttonWidth,unit(36));
        place(IDC_SWITCH,left+buttonWidth+gap,actionsTop,right-left-buttonWidth-gap,unit(36));
        int i=0;
        for (const auto id:{IDC_RECORDING_LABEL,IDC_SCREEN_LABEL,IDC_MODE_LABEL,IDC_LEVEL_LABEL,IDC_LOCALE_LABEL}) {
            place(id,left,top+row*i+unit(4),content-left-unit(10),unit(26)); ++i;
        }
        place(IDC_RECORDING,content,top,right-content,unit(130));
        place(IDC_SCREEN,content,top+row,right-content,unit(180));
        place(IDC_MODE,content,top+row*2,right-content,unit(130));
        place(IDC_LEVEL,content,top+row*3,right-content-unit(72),unit(30));
        place(IDC_LEVEL_VALUE,right-unit(62),top+row*3+unit(4),unit(62),unit(24));
        place(IDC_LOCALE,content,top+row*4,right-content,unit(130));
        RedrawWindow(hwnd,nullptr,nullptr,RDW_INVALIDATE|RDW_ALLCHILDREN);
        layingOut=false;
    }
    void fit() {
        if (!IsWindow(hwnd) || IsIconic(hwnd)) return;
        RECT r{}; GetWindowRect(hwnd,&r); MONITORINFO monitor{sizeof(monitor)};
        if (!GetMonitorInfoW(MonitorFromWindow(hwnd,MONITOR_DEFAULTTONEAREST),&monitor)) return;
        const auto& work=monitor.rcWork;
        const int w=std::min(std::max(r.right-r.left,static_cast<LONG>(unit(560))),work.right-work.left);
        const int h=std::min(std::max(r.bottom-r.top,static_cast<LONG>(unit(528))),work.bottom-work.top);
        SetWindowPos(hwnd,nullptr,std::clamp(r.left,work.left,work.right-w),std::clamp(r.top,work.top,work.bottom-h),w,h,SWP_NOZORDER|SWP_NOACTIVATE);
    }
    void draw(HDC dc) const {
        RECT client{}; GetClientRect(hwnd,&client); fill(dc,client,colors.frame);
        const auto shape=animatedGeometry();
        // Matching rims and a cut-out waist read as two open halves, rather than a title bar.
        for (const auto& shell:{shape.base,shape.lid}) {
            const int depth=shell.bottom-shell.top;
            if (depth>px(24)) {
                const int step=std::max(1,std::min(unit(10),(depth-px(20))/4));
                panel(dc,inset(shell,px(6)),step,colors.edge);
                panel(dc,inset(shell,px(10)),step,colors.lens);
            } else fill(dc,shell,colors.edge);
        }
        if (motion==Motion::None || progress>.85f) for (const auto& hinge:shape.hinges) {
            fill(dc,hinge,colors.frame);
            fill(dc,inset(hinge,px(3)),colors.edge);
            const int middle=(hinge.top+hinge.bottom)/2;
            fill(dc,{hinge.left+px(3),middle-px(2),hinge.right-px(3),middle+px(2)},colors.frame);
        }
        if (motion!=Motion::None && progress<.45f) {
            const RECT glasses=glassesRect();
            // The shell wraps a stationary silhouette, then covers it from top to bottom.
            // Closing reverses that cover, leaving the final glasses fully visible in place.
            const int saved=SaveDC(dc);
            const int covered=static_cast<int>(std::lround((glasses.bottom-glasses.top)*ease((progress-.22f)/.23f)));
            IntersectClipRect(dc,glasses.left,glasses.top+covered,glasses.right,glasses.bottom);
            pixel::drawGlasses(dc,glasses,glassesColors); RestoreDC(dc,saved);
        }
        // Embossing and clasps belong to the fully expanded settings case.
        if (motion==Motion::None) {
            const int x=unit(34),y=shape.lid.top+unit(34),p=unit(3);
            fill(dc,{x,y,x+p*18,y+p*2},colors.ink);
            fill(dc,{x+p,y+p*2,x+p*8,y+p*6},colors.ink);
            fill(dc,{x+p*10,y+p*2,x+p*17,y+p*6},colors.ink);
            fill(dc,{x+p*2,y+p*6,x+p*7,y+p*7},colors.ink);
            fill(dc,{x+p*11,y+p*6,x+p*16,y+p*7},colors.ink);
            // Opposing clasp halves on the outside edges reinforce the open-case silhouette.
            for (const int yLatch:{static_cast<int>(shape.lid.top)+px(4),height-px(16)}) {
                fill(dc,{width/2-unit(18),yLatch,width/2+unit(18),yLatch+px(12)},colors.edge);
                fill(dc,{width/2-unit(10),yLatch+px(4),width/2+unit(10),yLatch+px(8)},colors.frame);
            }
        }
    }
    void paint() const {
        PAINTSTRUCT ps{}; const auto target=BeginPaint(hwnd,&ps);
        pixel::paintBuffered(target,{0,0,width,height},[&](HDC dc) { draw(dc); });
        EndPaint(hwnd,&ps);
    }
    void button(const DRAWITEMSTRUCT& item) const {
        const bool enabled=(item.itemState&ODS_DISABLED)==0,pressed=(item.itemState&ODS_SELECTED)!=0;
        const bool primary=item.CtlID==IDC_SWITCH;
        const auto dc=item.hDC; RECT r=item.rcItem;
        fill(dc,r,colors.lens);
        const COLORREF border=colors.highContrast?colors.ink:colors.edge;
        fill(dc,r,border); InflateRect(&r,-px(2),-px(2));
        const COLORREF background=primary&&enabled?colors.accent:colors.field;
        fill(dc,r,background);
        if (pressed) OffsetRect(&r,px(1),px(1));
        SetBkMode(dc,TRANSPARENT);
        SetTextColor(dc,!enabled?GetSysColor(COLOR_GRAYTEXT):primary?(colors.highContrast?GetSysColor(COLOR_HIGHLIGHTTEXT):colors.frame):colors.ink);
        const auto old=SelectObject(dc,font);
        const int length=GetWindowTextLengthW(item.hwndItem);
        std::wstring label(static_cast<size_t>(length)+1,L'\0'); GetWindowTextW(item.hwndItem,label.data(),length+1);
        DrawTextW(dc,label.c_str(),-1,&r,DT_CENTER|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS|DT_NOPREFIX);
        SelectObject(dc,old);
        if ((item.itemState&ODS_FOCUS)!=0 && (item.itemState&ODS_NOFOCUSRECT)==0) {
            InflateRect(&r,-px(2),-px(2)); DrawFocusRect(dc,&r);
        }
    }
    LRESULT slider(NMCUSTOMDRAW& draw) const {
        if (draw.dwDrawStage!=CDDS_PREPAINT) return CDRF_DODEFAULT;
        const auto control=GetDlgItem(hwnd,IDC_LEVEL);
        RECT client{},channel{},thumb{}; GetClientRect(control,&client); fill(draw.hdc,client,colors.lens);
        SendMessageW(control,TBM_GETCHANNELRECT,0,reinterpret_cast<LPARAM>(&channel));
        SendMessageW(control,TBM_GETTHUMBRECT,0,reinterpret_cast<LPARAM>(&thumb));
        const int center=(thumb.top+thumb.bottom)/2,half=unit(4),extent=channel.right-channel.left;
        const auto strength=SendMessageW(control,TBM_GETPOS,0,0);
        for (int i=0;i<10;++i) {
            const int a=channel.left+MulDiv(i,extent,10),b=channel.left+MulDiv(i+1,extent,10)-px(2);
            fill(draw.hdc,{a,center-half,b,center+half},i*10<strength?colors.accent:colors.field);
        }
        fill(draw.hdc,thumb,colors.ink);
        if (GetFocus()==control) { InflateRect(&client,-1,-1); DrawFocusRect(draw.hdc,&client); }
        return CDRF_SKIPDEFAULT;
    }
    static LRESULT CALLBACK buttonProcedure(HWND control,UINT message,WPARAM wp,LPARAM lp,UINT_PTR,DWORD_PTR) {
        if (message==WM_GETDLGCODE && wp==VK_RETURN) return DefSubclassProc(control,message,wp,lp)|DLGC_WANTMESSAGE;
        if (message==WM_KEYDOWN && wp==VK_RETURN) {
            if (GetDlgCtrlID(control)!=IDC_LEVEL) SendMessageW(control,BM_CLICK,0,0);
            return 0;
        }
        return DefSubclassProc(control,message,wp,lp);
    }
    static LRESULT CALLBACK procedure(HWND dialog,UINT message,WPARAM wp,LPARAM lp,UINT_PTR,DWORD_PTR data) {
        auto& self=*reinterpret_cast<Impl*>(data);
        switch (message) {
        case WM_NCCALCSIZE: return 0;
        case WM_NCHITTEST: {
            if (self.motion!=Motion::None) return HTCLIENT;
            POINT p{GET_X_LPARAM(lp),GET_Y_LPARAM(lp)}; ScreenToClient(dialog,&p);
            const int edge=self.px(8),corner=self.px(24);
            if (p.y<edge) return p.x<corner?HTTOPLEFT:p.x>self.width-corner?HTTOPRIGHT:HTTOP;
            if (p.y>self.height-edge) return p.x<corner?HTBOTTOMLEFT:p.x>self.width-corner?HTBOTTOMRIGHT:HTBOTTOM;
            if (p.x<edge) return HTLEFT; if (p.x>self.width-edge) return HTRIGHT;
            if (p.y<self.geometry().lid.bottom) {
                POINT screen=p; ClientToScreen(dialog,&screen);
                for (int id:{IDC_CASE_CLOSE,IDC_STOP,IDC_SWITCH}) {
                    RECT button{}; GetWindowRect(GetDlgItem(dialog,id),&button);
                    if (PtInRect(&button,screen)) return HTCLIENT;
                }
                return HTCAPTION;
            }
            return HTCLIENT;
        }
        case WM_ERASEBKGND: return 1;
        case WM_PAINT: self.paint(); return 0;
        case WM_SIZE:
            if (wp==SIZE_MINIMIZED) self.settleForMinimize();
            else self.layout();
            return 0;
        case WM_TIMER:
            if (wp==MotionTimer && self.motion!=Motion::None) { self.motionFrame(); return 0; }
            break;
        case WM_COMMAND:
            if (LOWORD(wp)==IDOK && self.motion!=Motion::None) return TRUE;
            break;
        case WM_GETMINMAXINFO: {
            auto& info=*reinterpret_cast<MINMAXINFO*>(lp); info.ptMinTrackSize={self.unit(560),self.unit(528)}; return 0;
        }
        case WM_DPICHANGED: {
            if (self.motion!=Motion::None) self.endMotion(self.motion==Motion::Opening);
            // Keep the dialog manager's per-monitor control/font bookkeeping, then apply our layout.
            const auto result=DefSubclassProc(dialog,message,wp,lp);
            self.refreshFont(); self.fit(); self.layout(); return result;
        }
        case WM_SETTINGCHANGE: case WM_THEMECHANGED: {
            if (self.motion!=Motion::None) self.endMotion(self.motion==Motion::Opening);
            const auto result=DefSubclassProc(dialog,message,wp,lp);
            self.refreshTheme(); self.refreshFont(); self.fit(); self.layout(); return result;
        }
        case WM_CTLCOLORSTATIC: case WM_CTLCOLORDLG: case WM_CTLCOLORBTN:
            SetTextColor(reinterpret_cast<HDC>(wp),message==WM_CTLCOLORSTATIC && !IsWindowEnabled(reinterpret_cast<HWND>(lp))?
                GetSysColor(COLOR_GRAYTEXT):self.colors.ink);
            SetBkColor(reinterpret_cast<HDC>(wp),self.colors.lens);
            return reinterpret_cast<LRESULT>(self.liningBrush);
        case WM_CTLCOLORLISTBOX: case WM_CTLCOLOREDIT:
            SetTextColor(reinterpret_cast<HDC>(wp),self.colors.ink); SetBkColor(reinterpret_cast<HDC>(wp),self.colors.field);
            return reinterpret_cast<LRESULT>(self.fieldBrush);
        case WM_DRAWITEM: {
            pixel::drawBufferedItem(*reinterpret_cast<DRAWITEMSTRUCT*>(lp),[&](const auto& item) {
                if (item.CtlType==ODT_COMBOBOX) pixel::drawComboItem(item);
                else self.button(item);
            });
            return TRUE;
        }
        case WM_NOTIFY: {
            const auto header=reinterpret_cast<NMHDR*>(lp);
            if (header->idFrom==IDC_LEVEL && header->code==NM_CUSTOMDRAW)
                return self.slider(*reinterpret_cast<NMCUSTOMDRAW*>(lp));
            break;
        }
        case WM_NCLBUTTONDBLCLK: return 0; // A lid double-click does not maximize the case.
        case WM_SYSCOMMAND:
            if ((wp&0xfff0)==SC_MINIMIZE) { self.settleForMinimize(); self.layout(); }
            if ((wp&0xfff0)==SC_MAXIMIZE) return 0;
            break;
        case WM_NCDESTROY:
            KillTimer(dialog,MotionTimer);
            RemoveWindowSubclass(dialog,procedure,1); self.hwnd=nullptr; break;
        }
        return DefSubclassProc(dialog,message,wp,lp);
    }
};

GlassesCase::GlassesCase():impl_(std::make_unique<Impl>()) {}
GlassesCase::~GlassesCase()=default;
bool GlassesCase::attach(HWND dialog) {
    impl_->hwnd=dialog;
    if (!SetWindowSubclass(dialog,Impl::procedure,1,reinterpret_cast<DWORD_PTR>(impl_.get()))) { impl_->hwnd=nullptr; return false; }
    impl_->refreshTheme(); impl_->refreshFont();
    for (int id:{IDC_SCREEN,IDC_MODE,IDC_LEVEL,IDC_LOCALE,IDC_RECORDING}) SetWindowTheme(GetDlgItem(dialog,id),L"",L"");
    if (!pixel::skinSlider(GetDlgItem(dialog,IDC_LEVEL))) return false;
    for (int id:{IDC_SCREEN,IDC_MODE,IDC_LOCALE,IDC_RECORDING})
        if (!pixel::skinCombo(GetDlgItem(dialog,id))) return false;
    for (int id:{IDC_STOP,IDC_SWITCH,IDC_CASE_CLOSE})
        if (!pixel::skinButton(GetDlgItem(dialog,id))) return false;
    for (int id:{IDC_STOP,IDC_SWITCH,IDC_CASE_CLOSE,IDC_LEVEL})
        SetWindowSubclass(GetDlgItem(dialog,id),Impl::buttonProcedure,1,0);
    SetWindowPos(dialog,nullptr,0,0,impl_->unit(620),impl_->unit(552),SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE|SWP_FRAMECHANGED);
    impl_->layout(); return true;
}
void GlassesCase::fitOnScreen() { if (impl_->hwnd) impl_->fit(); }
void GlassesCase::open(RECT glasses,bool on,bool activate) {
    if (impl_->hwnd) { impl_->glassesColors=pixel::palette(on); impl_->prepare(glasses); impl_->animate(true,activate); }
}
void GlassesCase::close(SIZE glasses,bool on) {
    if (impl_->hwnd) {
        impl_->glassesColors=pixel::palette(on);
        if (impl_->motion==Impl::Motion::None) impl_->glassesWidth=glasses.cx;
        impl_->animate(false);
    }
}
RECT GlassesCase::glassesBounds() const {
    if (!impl_->hwnd) return {};
    RECT anchor=impl_->glassesRect();
    MapWindowPoints(impl_->hwnd,nullptr,reinterpret_cast<POINT*>(&anchor),2); return anchor;
}
}
