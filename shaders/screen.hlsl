// SDR correction overlay. Only a new increase in highlight load can start an event.
cbuffer ScreenParameters : register(b0) {
    uint width, height, columns, rows;
    float dt, userStrength, elapsed;
    uint resetHistory;
};
Texture2D<float4> encoded : register(t0);
Texture2D<float4> linearInput : register(t1);
StructuredBuffer<float4> eventInput : register(t2);
RWTexture2D<float4> linearOutput : register(u0);
RWStructuredBuffer<float2> totals : register(u1);
// Entry 0: load, mean luminance, event peak and age. Entry 1.x: held broad-event gate.
RWStructuredBuffer<float4> eventState : register(u2);
float decode(float x) { return x <= .04045 ? x / 12.92 : pow((x + .055) / 1.055, 2.4); }
float encode(float x) { return x <= .0031308 ? 12.92 * x : 1.055 * pow(max(x,0), 1 / 2.4) - .055; }
float smooth(float lo, float hi, float x) {
    float z = saturate((x-lo)/(hi-lo));
    return z*z*(3-2*z);
}
float envelope(float age) { return 1-smooth(.04, .32, age); }

#if SCREEN_COMPUTE
groupshared float2 sums[64];
[numthreads(8,8,1)]
void Convert(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint lane : SV_GroupIndex) {
    uint2 begin = group.xy * uint2(width,height) / uint2(columns,rows);
    uint2 end = (group.xy+1) * uint2(width,height) / uint2(columns,rows);
    float2 sum = 0;
    for (uint y=begin.y+local.y; y<end.y; y+=8) {
        for (uint x=begin.x+local.x; x<end.x; x+=8) {
            float3 rgb = saturate(encoded[uint2(x,y)].rgb);
            float3 decoded = float3(decode(rgb.r),decode(rgb.g),decode(rgb.b));
            linearOutput[uint2(x,y)] = float4(decoded,1);
            float value = dot(decoded,float3(.2126,.7152,.0722));
            sum += float2(max(value-.65,0)/.35,value);
        }
    }
    sums[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride=32; stride>0; stride>>=1) {
        if (lane<stride) sums[lane] += sums[lane+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0) totals[group.y*columns+group.x] = sums[0];
}

[numthreads(1,1,1)]
void Control() {
    float2 sum = 0;
    for (uint i=0; i<columns*rows; ++i) sum += totals[i];
    float2 current = sum/(width*height);
    if (resetHistory) { eventState[0]=float4(current,0,0); eventState[1]=0; return; }
    float4 previous = eventState[0];
    float age = min(.32,previous.w+dt);
    float broad = age>=.32 ? 0 : eventState[1].x;
    // Net changes reject stationary bright content, small fluctuations and bright
    // objects moving without increasing the scene's total brightness/load.
    float demand = smooth(.015,.08,current.x-previous.x) * smooth(.01,.08,current.y-previous.y);
    float peak = previous.z;
    if (demand>0) { peak=max(demand,peak*envelope(age)); age=0; }
    eventState[0] = float4(current,age>=.32 ? 0 : peak,age);
    // Keep the shoulder fixed as a flash contracts; the existing envelope owns
    // release. Recomputing it from declining load would add a second bright rebound.
    eventState[1] = float4(age>=.32 || peak<=0 ? 0 : max(broad,smooth(.08,.25,current.x)),0,0,0);
}
#endif

float4 Vertex(uint id : SV_VertexID) : SV_Position {
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * float2(2,-2) + float2(-1,1), 0, 1);
}
float4 Overlay(float4 position : SV_Position) : SV_Target {
    float4 event = eventInput[0];
    // 50% is the previous full-strength result. Above it, widen the shoulder
    // only for large highlight events; detection and expiry remain unchanged.
    float amount = min(2*userStrength,1) * event.z * envelope(event.w+elapsed) * smooth(.01,.04,event.x);
    if (amount<=0) return 0; // Exact transparency, including startup and strength zero.
    float extra = smooth(.5,1,userStrength) * eventInput[1].x;
    float knee = lerp(.65,.45,extra);
    float span = lerp(.15,.10,extra);
    uint2 p = uint2(position.xy);
    float3 rgb = linearInput[p].rgb;
    float value = dot(rgb,float3(.2126,.7152,.0722));
    if (value<=knee) return 0; // Balanced preserves Y<=.65; every strength preserves Y<=.45.
    float mapped = knee + span*(1-exp(-(value-knee)/span));
    rgb *= 1 + amount*(mapped/value-1);
    float3 original = saturate(encoded[p].rgb);
    float3 desired = min(original,float3(encode(rgb.r),encode(rgb.g),encode(rgb.b)));
    float3 ratio = float3(original.r>0 ? desired.r/original.r : 1,
                         original.g>0 ? desired.g/original.g : 1,
                         original.b>0 ? desired.b/original.b : 1);
    float alpha = saturate(1-min(ratio.r,min(ratio.g,ratio.b)));
    if (alpha<.000001) return 0;
    return float4(clamp(desired-(1-alpha)*original,0,alpha),alpha);
}
