// Linear SDR input. Temporal history controls the curve; RGB frames are never blended.
cbuffer Parameters : register(b0) {
    uint width, height, columns, rows;
    float dt, userStrength, holdSeconds, releaseSeconds;
    uint resetHistory, firstFrame, repeatEnabled, highlightOnly;
};
struct Block {
    float meanY, area, excess, positive, negative;
    float anchor, extreme;
    int direction;
    float ages[8];
    uint count;
    float repeatDemand, brightDemand, riseDemand;
};
struct State {
    float amount, bright, rise, repeatDemand;
    float holdRemaining;
    uint latch, reasons, reserved;
};
Texture2D<float4> source : register(t0);
Texture2D<float> previousY : register(t1);
RWTexture2D<float> nextY : register(u0);
RWStructuredBuffer<Block> blocks : register(u1);
RWStructuredBuffer<State> state : register(u2);
RWTexture2D<float4> output : register(u3);
groupshared float4 sums[64];
groupshared float negatives[64];

float smooth(float lo, float hi, float value) {
    float z = saturate((value - lo) / (hi - lo));
    return z * z * (3 - 2 * z);
}

[numthreads(8, 8, 1)]
void Analyze(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint lane : SV_GroupIndex) {
    uint2 begin = uint2(group.x * width / columns, group.y * height / rows);
    uint2 end = uint2((group.x + 1) * width / columns, (group.y + 1) * height / rows);
    float4 sum = 0;
    float negative = 0;
    for (uint y = begin.y + local.y; y < end.y; y += 8) {
        for (uint x = begin.x + local.x; x < end.x; x += 8) {
            uint2 pos = uint2(x, y);
            float value = dot(source[pos].rgb, float3(.2126, .7152, .0722));
            float change = resetHistory ? 0 : value - previousY[pos];
            sum += float4(value, value > .65 ? 1 : 0, max(value - .65, 0), max(change, 0));
            negative += max(-change, 0);
            nextY[pos] = value;
        }
    }
    sums[lane] = sum;
    negatives[lane] = negative;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            negatives[lane] += negatives[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane != 0) return;
    uint index = group.y * columns + group.x;
    float count = (end.x - begin.x) * (end.y - begin.y);
    float4 mean = sums[0] / count;
    Block block;
    if (resetHistory) {
        block = (Block)0;
        block.anchor = block.extreme = mean.x;
        [unroll] for (uint initialAge = 0; initialAge < 8; ++initialAge) block.ages[initialAge] = 1000;
    } else {
        block = blocks[index];
    }
    block.meanY = mean.x;
    block.area = mean.y;
    block.excess = mean.z;
    block.positive = mean.w;
    block.negative = negatives[0] / count;
    // Fixed slots, newest first; avoids dynamically indexing local struct-array writes in SM5.
    block.count = 0;
    [unroll] for (uint i = 0; i < 8; ++i) {
        block.ages[i] = min(1000, block.ages[i] + dt);
        if (block.ages[i] <= .750001) block.count++;
    }
    bool reversal = false;
    if (block.direction == 0) {
        if (abs(block.meanY - block.anchor) >= .08) {
            block.direction = block.meanY > block.anchor ? 1 : -1;
            block.extreme = block.meanY;
        }
    } else if (block.direction > 0) {
        block.extreme = max(block.extreme, block.meanY);
        if (block.extreme - block.meanY >= .08) {
            block.direction = -1;
            reversal = true;
        }
    } else {
        block.extreme = min(block.extreme, block.meanY);
        if (block.meanY - block.extreme >= .08) {
            block.direction = 1;
            reversal = true;
        }
    }
    if (reversal) {
        block.extreme = block.meanY;
        [unroll] for (uint j = 7; j > 0; --j) block.ages[j] = block.ages[j - 1];
        block.ages[0] = 0;
        block.count = min(8, block.count + 1);
    }
    block.repeatDemand = repeatEnabled ? smooth(2, 4, block.count) : 0;
    block.brightDemand = smooth(.01, .05, block.excess);
    block.riseDemand = smooth(.02, .15, block.positive) * smooth(.20, .45, block.meanY);
    blocks[index] = block;
}

[numthreads(1, 1, 1)]
void Control() {
    if (highlightOnly) {
        float excessSum = 0;
        for (uint index = 0; index < columns * rows; ++index) {
            uint x = index % columns, y = index / columns;
            uint count = ((x + 1) * width / columns - x * width / columns) *
                         ((y + 1) * height / rows - y * height / rows);
            excessSum += blocks[index].excess * count;
        }
        State gated = (State)0;
        gated.amount = gated.bright = smooth(.01, .04, excessSum / (width * height) / .35);
        gated.latch = gated.reasons = gated.amount > 0 ? 1 : 0;
        state[0] = gated;
        return;
    }
    State current = state[0];
    if (firstFrame) current = (State)0;
    if (resetHistory) {
        current.latch = 0;
        current.holdRemaining = firstFrame ? 0 : holdSeconds;
    }
    current.bright = current.rise = current.repeatDemand = 0;
    for (uint i = 0; i < columns * rows; ++i) {
        current.bright = max(current.bright, blocks[i].brightDemand);
        current.rise = max(current.rise, blocks[i].riseDemand);
        current.repeatDemand = max(current.repeatDemand, blocks[i].repeatDemand);
    }
    float demand = max(current.bright, max(current.rise, current.repeatDemand));
    if (!current.latch && demand >= .60) current.latch = 1;
    if (current.latch && demand <= .30) current.latch = 0;
    float target = current.latch ? demand : 0;
    float releaseDt = max(0, dt - current.holdRemaining);
    current.holdRemaining = max(0, current.holdRemaining - dt);
    if (current.latch && target >= current.amount) {
        current.amount = target;
        current.holdRemaining = holdSeconds;
    } else if (current.holdRemaining <= .000001) {
        current.amount = target + (current.amount - target) * exp(-releaseDt / releaseSeconds);
    }
    if (!current.latch && current.holdRemaining <= .000001 && current.amount < .001) current.amount = 0;
    current.reasons = (current.bright > 0 ? 1 : 0) | (current.rise > 0 ? 2 : 0) | (current.repeatDemand > 0 ? 4 : 0);
    state[0] = current;
}

float gain(float value, float knee, float ceiling) {
    if (value <= knee) return 1;
    float mapped = knee + (ceiling - knee) * (1 - exp(-(value - knee) / (ceiling - knee)));
    return saturate(mapped / value);
}

[numthreads(8, 8, 1)]
void Apply(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    float4 pixel = source[id.xy];
    float value = dot(pixel.rgb, float3(.2126, .7152, .0722));
    if (highlightOnly) {
        float multiplier = saturate(1 + userStrength * state[0].amount * (gain(value, .65, .80) - 1));
        output[id.xy] = float4(pixel.rgb * multiplier, pixel.a);
        return;
    }
    float light = gain(value, .65, .90);
    float strong = gain(value, .20, .55);
    float multiplier = saturate(1 + userStrength * ((light - 1) + state[0].amount * (strong - light)));
    output[id.xy] = float4(pixel.rgb * multiplier, pixel.a);
}
