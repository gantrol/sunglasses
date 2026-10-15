// Pixel-cone recovery + motion-propagated stable details, SDR 8-bit output.
// Reference: recovery.PixelLogRiseLimiter and detail_protection.StableDetailProtector.
// Nominal profile is 200/0.2 nit, 27 inches at 0.65 m. GPU equivalence/performance
// requires a freshly built headless probe; CPU experiment results are not GPU evidence.
cbuffer RecoveryConstants : register(b0) {
    uint width,height,columns,rows;
    uint radiusX,radiusY,motionX,motionY;
    uint groupsX,groupsY,reset,reserved;
    float dt,strength,horizontalStep,verticalStep;
    float cachedStrength,elapsed; float2 padding;
};
Texture2D<float4> encoded : register(t0);
Texture2D<float4> source : register(t1);
StructuredBuffer<float4> detector : register(t2);
Texture2D<float4> previousHistory : register(t3); // source log, base ceiling, base log, unused
Texture2D<float4> previousDetail : register(t4); // age, strength, final output log, active envelope
Texture2D<float4> work : register(t5);
Texture2D<float4> base : register(t6);
StructuredBuffer<float4> tiles : register(t7);
StructuredBuffer<float4> scores : register(t8);
StructuredBuffer<float4> control : register(t9);
StructuredBuffer<uint> indices : register(t10);
Texture2D<float4> output : register(t11);
static const float3 luma=float3(.2126,.7152,.0722);
static const float stableDelta=.009950330853168;
float smooth(float lo,float hi,float x) { x=saturate((x-lo)/(hi-lo)); return x*x*(3-2*x); }
// Clamp each pow operand locally so strict FXC compilation can establish a
// nonnegative base, including when conditional expressions are flattened.
float decodeChannel(float x) { return x<=.04045?x/12.92:pow(max((x+.055)/1.055,0),2.4); }
float encodeChannel(float x) { return x<=.0031308?12.92*x:1.055*pow(max(x,0),1/2.4)-.055; }
float3 decode(float3 x) { return float3(decodeChannel(x.x),decodeChannel(x.y),decodeChannel(x.z)); }
float3 encode(float3 x) { x=max(x,0); return float3(encodeChannel(x.x),encodeChannel(x.y),encodeChannel(x.z)); }
float logY(float3 rgb) { return log(.7+199.8*dot(rgb,luma)); }
float3 quantize(float3 rgb,float3 original) { return min(original,round(saturate(encode(rgb))*255)/255); }
float3 curve(float3 rgb,float3 original,float amount,float broad) {
    float y=dot(rgb,luma),knee=.65-.20*broad,span=.15-.05*broad;
    if (y<=knee || amount<=0) return original;
    float mapped=knee+span*(1-exp(-(y-knee)/span));
    return quantize(rgb*(1+amount*(mapped/y-1)),original);
}
bool inside(int2 p) { return all(p>=0) && p.x<int(width) && p.y<int(height); }
int reflect(int p,int size) {
    if (size<=1) return 0;
    // Reflection is symmetric around zero; coordinates are bounded by the
    // texture size plus its surround radius, so abs cannot overflow here.
    uint period=2u*(uint(size)-1u),folded=uint(abs(p))%period;
    return int(folded<uint(size)?folded:period-folded);
}
#if RECOVERY_COMPUTE
RWTexture2D<float4> workOut : register(u0);
RWTexture2D<float4> historyOut : register(u1);
RWTexture2D<float4> detailOut : register(u2);
RWStructuredBuffer<float4> tilesOut : register(u4);
RWStructuredBuffer<float4> scoresOut : register(u5);
RWStructuredBuffer<float4> controlOut : register(u6);
RWStructuredBuffer<uint> indicesOut : register(u7);
groupshared float4 reduction[64];
void sumGroup(uint lane,float4 value) {
    reduction[lane]=value; GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride=32;stride>0;stride/=2) {
        if (lane<stride) reduction[lane]+=reduction[lane+stride];
        GroupMemoryBarrierWithGroupSync();
    }
}
[numthreads(1,1,1)]
void RecoveryControl(uint3 id:SV_DispatchThreadID) {
    float4 event=detector[0];
    float amount=min(2*strength,1)*event.z*(1-smooth(.04,.32,event.w))*smooth(.01,.04,event.x);
    float broad=smooth(.5,1,strength)*detector[1].x;
    if (!reset && controlOut[0].z>0) broad=max(broad,controlOut[0].y);
    controlOut[0]=float4(amount,broad,0,0);
}
[numthreads(8,8,1)]
void Solve(uint3 id:SV_DispatchThreadID) {
    if (id.x>=width || id.y>=height) return;
    uint2 p=id.xy; float3 rgb=source[p].rgb,original=encoded[p].rgb;
    float4 parameters=control[0]; float sourceLog=logY(rgb);
    float desiredLog=logY(decode(curve(rgb,original,parameters.x,parameters.y)));
    float ceiling=desiredLog,selected=parameters.x;
    if (!reset) {
        float4 previous=previousHistory[p];
        ceiling=min(sourceLog,previous.y+max(sourceLog-previous.x,0)+.5*dt);
        if (desiredLog>ceiling) {
            float lo=selected,hi=1;
            [loop] for (uint step=0;step<18;++step) {
                float mid=(lo+hi)*.5;
                if (logY(decode(curve(rgb,original,mid,parameters.y)))>ceiling) lo=mid;
                else hi=mid;
            }
            selected=hi;
        }
    }
    workOut[p]=float4(selected,ceiling,0,0);
}
[numthreads(8,8,1)]
void ReduceTiles(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID,uint lane:SV_GroupIndex) {
    uint2 begin=group.xy*uint2(width,height)/uint2(columns,rows);
    uint2 end=(group.xy+1)*uint2(width,height)/uint2(columns,rows);
    float value=0;
    [loop] for (uint y=begin.y+thread.y;y<end.y;y+=8)
        [loop] for (uint x=begin.x+thread.x;x<end.x;x+=8) value=max(value,work[uint2(x,y)].x);
    reduction[lane]=float4(value,0,0,0); GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride=32;stride>0;stride/=2) {
        if (lane<stride) reduction[lane]=max(reduction[lane],reduction[lane+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (!lane) tilesOut[group.y*columns+group.x]=reduction[0];
}
[numthreads(1,1,1)]
void SpatialCone(uint3 id:SV_DispatchThreadID) {
    float required=tiles[id.y*columns+id.x].x,amount=required;
    [loop] for (uint y=0;y<rows;++y) [loop] for (uint x=0;x<columns;++x)
        amount=max(amount,tiles[y*columns+x].x-abs(float(x)-float(id.x))*horizontalStep-abs(float(y)-float(id.y))*verticalStep);
    tilesOut[id.y*columns+id.x]=float4(amount,required,0,0);
}
[numthreads(8,8,1)]
void Base(uint3 id:SV_DispatchThreadID) {
    if (id.x>=width || id.y>=height) return;
    uint2 p=id.xy;
    // Invert floor(i*size/count) partitions, including uneven and single-pixel cells.
    uint2 cell=((p+1)*uint2(columns,rows)-1)/uint2(width,height);
    float2 amounts=tiles[cell.y*columns+cell.x].xy;
    float3 original=encoded[p].rgb,rgb=source[p].rgb;
    float3 value=curve(rgb,original,amounts.x,control[0].y);
    float outputLog=logY(decode(value));
    float credit=logY(decode(min(original,value+1/255.0)))-outputLog;
    float ceiling=min(work[p].y,outputLog+max(credit,0));
    if (reset || amounts.x>amounts.y+1e-7 || outputLog>work[p].y) ceiling=outputLog;
    historyOut[p]=float4(logY(rgb),ceiling,outputLog,0);
    workOut[p]=float4(decode(value),1);
}
[numthreads(8,8,1)]
void SurroundRows(uint3 id:SV_DispatchThreadID) {
    if (id.x>=width || id.y>=height) return;
    float sum=0;
    [loop] for (int dx=-int(radiusX);dx<=int(radiusX);++dx)
        sum+=logY(source[int2(reflect(int(id.x)+dx,int(width)),id.y)].rgb);
    workOut[id.xy]=float4(sum/float(2*radiusX+1),0,0,0);
}
[numthreads(8,8,1)]
void Candidates(uint3 id:SV_DispatchThreadID) {
    if (id.x>=width || id.y>=height) return;
    uint2 p=id.xy; float sum=0;
    [loop] for (int dy=-int(radiusY);dy<=int(radiusY);++dy)
        sum+=work[int2(id.x,reflect(int(id.y)+dy,int(height)))].x;
    float sourceLog=logY(source[p].rgb);
    bool candidate=dot(source[p].rgb,luma)>=.65 && sourceLog-sum/float(2*radiusY+1)>=.22314355131421;
    workOut[p]=float4(sourceLog,candidate?1:0,0,0);
    if (!reset && previousDetail[p].y>0) {
        uint index; InterlockedAdd(indicesOut[0],1,index);
        indicesOut[index+1]=id.y*width+id.x;
    }
}
[numthreads(64,1,1)]
void MotionScores(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    int2 shift=int2(group.xy)-int2(motionX,motionY);
    uint count=indices[0]; float matched=0,error=0;
    if (count>=4 && !reset) {
        [loop] for (uint i=lane;i<count;i+=64) {
            uint index=indices[i+1]; int2 p=int2(index%width,index/width),q=p+shift;
            if (inside(q)) {
                float difference=abs(previousHistory[p].x-work[q].x);
                if (work[q].y>0 && difference<=stableDelta) { matched+=1; error+=difference; }
            }
        }
    }
    sumGroup(lane,float4(matched,error,0,0));
    if (!lane) scoresOut[group.y*(2*motionX+1)+group.x]=reduction[0];
}
bool better(float2 score,int2 shift,float2 best,int2 bestShift) {
    if (score.x!=best.x) return score.x>best.x;
    float error=score.x>0?score.y/score.x:1e20;
    float bestError=best.x>0?best.y/best.x:1e20;
    if (error!=bestError) return error<bestError;
    int distance=abs(shift.x)+abs(shift.y),bestDistance=abs(bestShift.x)+abs(bestShift.y);
    if (distance!=bestDistance) return distance<bestDistance;
    return shift.y!=bestShift.y?shift.y>bestShift.y:shift.x>bestShift.x;
}
[numthreads(1,1,1)]
void ChooseMotion(uint3 id:SV_DispatchThreadID) {
    uint count=indices[0]; int2 bestShift=int2(0,0);
    float2 zero=scores[motionY*(2*motionX+1)+motionX].xy,best=zero;
    [loop] for (uint y=0;y<=2*motionY;++y) [loop] for (uint x=0;x<=2*motionX;++x) {
        int2 shift=int2(x,y)-int2(motionX,motionY); float2 score=scores[y*(2*motionX+1)+x].xy;
        if (better(score,shift,best,bestShift)) { best=score; bestShift=shift; }
    }
    if (reset || count<4 || best.x<.85*count || best.x<zero.x+.10*count) bestShift=int2(0,0);
    controlOut[1]=float4(bestShift,0,0);
    float maximum=0;
    [loop] for (uint i=0;i<rows*columns;++i) maximum=max(maximum,tiles[i].x);
    float4 state=controlOut[0]; state.z=maximum; controlOut[0]=state;
}
[numthreads(8,8,1)]
void PrepareDetail(uint3 id:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    float existing=0,fresh=0;
    if (id.x<width && id.y<height) {
        uint2 p=id.xy; float4 candidate=work[p]; float age=0,weight=0;
        if (!reset) {
            int2 shift=int2(control[1].xy),q=int2(p)-shift;
            bool moved=any(shift!=0);
            bool matched=inside(q) && candidate.y>0 && abs(previousHistory[q].x-candidate.x)<=stableDelta;
            if (!moved || matched) {
                float4 previous=previousDetail[q];
                age=abs(previousHistory[q].x-candidate.x)<=stableDelta?min(previous.x+dt,1):0;
                weight=previous.y;
            } else age=dt;
        }
        bool eligible=age>=.20 && candidate.y>0;
        existing=eligible && weight>0?1:0;
        fresh=eligible && weight==0 && dot(base[p].rgb,luma)>=dot(source[p].rgb,luma)-.000303526984?1:0;
        workOut[p]=float4(age,weight,existing,fresh);
    }
    sumGroup(lane,float4(existing,fresh,0,0));
    if (!lane) scoresOut[group.y*groupsX+group.x]=reduction[0];
}
[numthreads(1,1,1)]
void ChooseNew(uint3 id:SV_DispatchThreadID) {
    float2 counts=0;
    [loop] for (uint i=0;i<groupsX*groupsY;++i) counts+=scores[i].xy;
    float maximum=max(1,floor(.05*float(width*height)));
    float4 state=controlOut[1]; state.z=counts.y<=max(0,maximum-counts.x)?1:0; controlOut[1]=state;
}
[numthreads(8,8,1)]
void Budget(uint3 id:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    float remaining=0,requested=0;
    if (id.x<width && id.y<height) {
        float4 previous=work[id.xy];
        float target=previous.z>0 || (previous.w>0 && control[1].z>0)?1:0;
        remaining=target<previous.y?max(target,previous.y-dt/.10):previous.y;
        requested=target>previous.y?min(target,previous.y+dt/.20)-previous.y:0;
        workOut[id.xy]=float4(previous.x,remaining,requested,0);
    }
    sumGroup(lane,float4(remaining,requested,0,0));
    if (!lane) scoresOut[group.y*groupsX+group.x]=reduction[0];
}
[numthreads(1,1,1)]
void ChooseBudget(uint3 id:SV_DispatchThreadID) {
    float2 sums=0;
    [loop] for (uint i=0;i<groupsX*groupsY;++i) sums+=scores[i].xy;
    float maximum=max(1,floor(.05*float(width*height)));
    float scale=sums.y>0?min(1,max(0,maximum-sums.x)/sums.y):0;
    float total=sums.x+scale*sums.y;
    controlOut[2]=float4(scale,total>maximum?maximum/total:1,0,0);
}
[numthreads(8,8,1)]
void Finish(uint3 id:SV_DispatchThreadID) {
    if (id.x>=width || id.y>=height) return;
    uint2 p=id.xy; float4 prepared=work[p],parameters=control[2];
    float weight=saturate((prepared.y+prepared.z*parameters.x)*parameters.y);
    float3 original=encoded[p].rgb,sourceRGB=source[p].rgb,baseRGB=base[p].rgb;
    float3 value=quantize(lerp(baseRGB,sourceRGB,weight),original);
    float sourceLog=logY(sourceRGB),outputLog=logY(decode(value));
    bool active=weight>0 || (!reset && previousDetail[p].w>0);
    if (!reset && active) {
        // Envelope remains in screen coordinates when detail state moves.
        float ceiling=min(sourceLog,previousDetail[p].z+max(sourceLog-previousHistory[p].x,0)+.5*dt);
        if (outputLog>ceiling) {
            float3 rgb=decode(value);
            float targetY=saturate((exp(ceiling)-.7)/199.8);
            float ratio=min(1,targetY/max(dot(rgb,luma),1e-20));
            value=min(original,floor(saturate(encode(rgb*ratio))*255)/255);
            outputLog=logY(decode(value));
        }
    }
    bool differs=any(value!=quantize(baseRGB,original));
    detailOut[p]=float4(prepared.x,weight,outputLog,active && (weight>0 || differs)?1:0);
    workOut[p]=float4(value,1);
}
#else
float4 RecoveryOverlay(float4 position:SV_Position):SV_Target {
    uint2 p=uint2(position.xy); float3 original=encoded[p].rgb;
    // UNORM sampling and arithmetic division can represent the same 8-bit code
    // a float ULP apart. Unchanged codes must yield exactly zero mask opacity.
    if (all(round(output[p].rgb*255)==round(original*255))) return 0;
    float scale=cachedStrength>0?min(1,strength/cachedStrength):0;
    scale*=1-smooth(.04,.32,elapsed);
    if (scale<=0) return 0;
    float3 desired=lerp(original,min(original,output[p].rgb),scale);
    float3 ratio=float3(original.x>0?desired.x/original.x:1,original.y>0?desired.y/original.y:1,
                        original.z>0?desired.z/original.z:1);
    float alpha=saturate(1-min(ratio.x,min(ratio.y,ratio.z)));
    return float4(clamp(desired-(1-alpha)*original,0,alpha),alpha);
}
#endif
