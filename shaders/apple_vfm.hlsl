// Semantic port of Apple VideoFlashingReduction MATLAB passes 0/1/2 at
// 6e9604d92446c8e6763bc53c5e12ccc4e0046796. See the distributed MIT notice:
// sunglasses/data/APPLE_VIDEO_FLASHING_REDUCTION_NOTICE.md.
// Fixed assumptions: 60 Hz, 100 nit, equivalent size 45 degrees. Not calibrated.
cbuffer Parameters : register(b0) {
    uint width,height,columns,rows;
    uint oldSamples,currentSample,newCapture,resetHistory;
    float userStrength;
    float3 padding;
};
cbuffer Kernels : register(b1) {
    float4 contrastKernels[80];
    float4 gammaKernel[64];
};
Texture2D<float4> encoded : register(t0);
StructuredBuffer<float4> stateInput : register(t1);

#if APPLE_COMPUTE
RWStructuredBuffer<float> totals : register(u0);
RWStructuredBuffer<float> history : register(u1);
// state[0]: adaptation nit, next mitigation, applied mitigation, output adaptation nit.
// state[1]: last sampled APL, initialized, response energy, normalized step energy.
RWStructuredBuffer<float4> state : register(u2);
// APL, adaptation, energy, step energy; unmasked score, score, applied, next.
RWStructuredBuffer<float4> telemetry : register(u3);
groupshared float sums[64];
[numthreads(8,8,1)]
void Reduce(uint3 group:SV_GroupID,uint3 local:SV_GroupThreadID,uint lane:SV_GroupIndex) {
    uint2 begin=group.xy*uint2(width,height)/uint2(columns,rows);
    uint2 end=(group.xy+1)*uint2(width,height)/uint2(columns,rows);
    float sum=0;
    for (uint y=begin.y+local.y;y<end.y;y+=8) {
        for (uint x=begin.x+local.x;x<end.x;x+=8) {
            float3 rgb=pow(max(saturate(encoded[uint2(x,y)].rgb),0),2.2);
            sum+=dot(rgb,float3(.299,.587,.114));
        }
    }
    sums[lane]=sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride=32;stride>0;stride>>=1) {
        if (lane<stride) sums[lane]+=sums[lane+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0) totals[group.y*columns+group.x]=sums[0];
}

void sampleAPL(float apl) {
    float4 previous=state[0];
    float average=100*apl;
    float outputAdaptation=state[1].y!=0?previous.x:average;
    float adaptation=state[1].y!=0?lerp(previous.x,average,1-exp(-1.0/60)):average;
    // Explicit numeric extension for black input; avoids upstream 0/0.
    float safeAdaptation=max(adaptation,1e-6);
    float contrast=(average-safeAdaptation)/safeAdaptation;
    float pooled[5];
    float normalizedPool=0;
    [loop] for (uint level=0;level<5;++level) {
        uint base=level*16;
        uint length=(uint)contrastKernels[base].y;
        [loop] for (uint j=0;j+1<length;++j) history[base+j]=history[base+j+1];
        history[base+length-1]=contrast;
        float response=0,reverseResponse=0,bufferSquare=0,kernelSquare=0;
        [loop] for (uint k=0;k<length;++k) {
            float value=history[base+k],weight=contrastKernels[base+k].x;
            response+=value*contrastKernels[base+length-1-k].x;
            reverseResponse+=value*weight;
            bufferSquare+=value*value; kernelSquare+=weight*weight;
        }
        float normalized=reverseResponse/sqrt(bufferSquare*kernelSquare+1e-7);
        uint energyBase=80+level*61,normalizedBase=80+305+level*61;
        [loop] for (uint n=0;n<60;++n) {
            history[energyBase+n]=history[energyBase+n+1];
            history[normalizedBase+n]=history[normalizedBase+n+1];
        }
        history[energyBase+60]=response*response;
        history[normalizedBase+60]=normalized*normalized;
        float energy=0,normalEnergy=0;
        [loop] for (uint p=0;p<61;++p) {
            energy+=history[energyBase+p]*gammaKernel[60-p].x;
            normalEnergy+=history[normalizedBase+p]*gammaKernel[60-p].x;
        }
        pooled[level]=sqrt(max(energy,0));
        if (level==4) normalizedPool=sqrt(max(normalEnergy,0));
    }
    const float standard[5]={.2,1,10,150,500};
    float logAdaptation=log10(safeAdaptation);
    float energy=pooled[0];
    [unroll] for (uint a=0;a<4;++a) {
        if (safeAdaptation>=standard[a])
            energy=lerp(pooled[a],pooled[a+1],saturate((logAdaptation-log10(standard[a]))/
                (log10(standard[a+1])-log10(standard[a]))));
    }
    energy/=sqrt(60.0);
    float excess=max((energy-33)/200,0);
    // Negate the complete cube: FXC fails to converge on (-x)*x*x under /O3.
    float unmasked=100*(1-exp(-(excess*excess*excess)));
    float score=normalizedPool>1.8?unmasked:0;
    float mitigation=score>0?saturate(log10(max(score,1e-20))*.5):0;
    if (mitigation<previous.y) {
        mitigation=lerp(previous.y,mitigation,1-exp(-1.0/120));
        if (mitigation<.01) mitigation=0;
    }
    state[0]=float4(adaptation,mitigation,previous.y,outputAdaptation);
    state[1]=float4(apl,1,energy,normalizedPool);
    telemetry[0]=float4(apl,adaptation,energy,normalizedPool);
    telemetry[1]=float4(unmasked,score,previous.y,mitigation);
}

[numthreads(1,1,1)]
void Control() {
    if (resetHistory) {
        float heldAPL=0;
        if (!newCapture) heldAPL=state[1].x;
        [loop] for (uint i=0;i<690;++i) history[i]=0;
        state[0]=0; state[1]=float4(heldAPL,0,0,0); telemetry[0]=0; telemetry[1]=0;
    }
    [loop] for (uint tick=0;tick<oldSamples;++tick) sampleAPL(state[1].x);
    if (newCapture) {
        float sum=0;
        [loop] for (uint tile=0;tile<columns*rows;++tile) sum+=totals[tile];
        float4 latest=state[1]; latest.x=saturate(sum/(width*height)); state[1]=latest;
    }
    if (currentSample) sampleAPL(state[1].x);
}
#endif

float4 Overlay(float4 position:SV_Position):SV_Target {
    float mitigation=saturate(stateInput[0].z*userStrength);
    if (mitigation==0) return 0;
    float3 original=saturate(encoded[uint2(position.xy)].rgb);
    float3 rgb=pow(max(original,0),2.2);
    float scale=1-mitigation,adaptation=stateInput[0].w/100;
    float3 mapped=saturate(scale*scale*(rgb-adaptation)+scale*adaptation);
    float3 desired=pow(max(mapped,0),1.0/2.2);
    // The reference contrast mapping may raise dark pixels. Represent the full
    // mapping with a premultiplied correction, not a black-only approximation.
    float3 darken=saturate((original-desired)/max(original,1e-7));
    float3 brighten=saturate((desired-original)/max(1-original,1e-7));
    float3 required=max(darken,brighten);
    float alpha=max(required.r,max(required.g,required.b));
    if (alpha<1e-7) return 0;
    return float4(clamp(desired-(1-alpha)*original,0,alpha),alpha);
}
