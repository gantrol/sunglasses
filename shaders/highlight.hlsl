// Both resources contain linear-light, normalized SDR RGB (alpha is passed through).
// The future capture/present host must perform transfer decoding/encoding exactly once.
Texture2D<float4> sourceFrame : register(t0);
RWTexture2D<float4> destinationFrame : register(u0);

cbuffer HighlightParameters : register(b0)
{
    float knee;
    float ceiling;
    float strength;
    float reserved;
};

[numthreads(8, 8, 1)]
void main(uint3 position : SV_DispatchThreadID)
{
    uint width, height;
    destinationFrame.GetDimensions(width, height);
    if (position.x >= width || position.y >= height)
        return;

    float4 pixel = sourceFrame.Load(int3(position.xy, 0));
    float y = dot(pixel.rgb, float3(0.2126, 0.7152, 0.0722));
    if (y > knee && strength > 0.0)
    {
        float span = ceiling - knee;
        float shoulder = knee + span * (1.0 - exp(-(y - knee) / span));
        float mapped = lerp(y, shoulder, strength);
        pixel.rgb *= saturate(mapped / y);
    }
    destinationFrame[position.xy] = pixel;
}
