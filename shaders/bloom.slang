/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Dual-kawase bloom: 13-tap downsample with Karis average + 9-tap tent upsample
// Operates on R16G16B16A16_SFLOAT off-screen HDR color

// Set 0, binding 0: source mip to downsample or upsample
[[vk::binding(0, 0)]] Sampler2D srcImage;
// Set 0, binding 1: higher-resolution mip to add during upsample pass
[[vk::binding(1, 0)]] Sampler2D addImage;

struct BloomPush {
    float2 texelSize; // 1.0 / srcImage resolution for current pass
    int    mipLevel;  // 0 = first downsample (brightpass + Karis average)
    float  threshold; // luminance threshold applied at mip 0 only
    float  strength;  // additive blend weight for upsample pass
    float3 _pad;      // pad to 32 bytes
};
[[vk::push_constant]] BloomPush push;

// ---------------------------------------------------------------------------
// Shared vertex shader: fullscreen triangle (no vertex buffer required)
// ---------------------------------------------------------------------------

struct VSOut {
    float4 Position : SV_POSITION;
    float2 UV       : TEXCOORD0;
};

[shader("vertex")]
VSOut VSMain(uint vid : SV_VertexID)
{
    // Three vertices covering the entire clip-space [-1,1]^2
    float2 pos = float2( (vid == 2) ? 3.0 : -1.0,
                         (vid == 1) ? 3.0 : -1.0 );
    float2 uv  = float2( (vid == 2) ? 2.0 :  0.0,
                         (vid == 1) ? 2.0 :  0.0 );
    VSOut o;
    o.Position = float4(pos, 0.0, 1.0);
    o.UV = uv;
    return o;
}

// ---------------------------------------------------------------------------
// Helper: perceptual luminance
// ---------------------------------------------------------------------------

float Luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Karis average: weight a color sample by 1/(1+luminance) to suppress fireflies
float3 KarisAverage(float3 c) {
    return c / (1.0 + Luminance(c));
}

// ---------------------------------------------------------------------------
// DownsamplePS — 13-tap downsample with optional Karis average
//
// Tap layout (relative to sample UV, offsets in texel units):
//   4 corner 2x2 box averages at (-1,-1), (+1,-1), (-1,+1), (+1,+1)
//   5 centre cross taps: (0,0), (±1,0), (0,±1)   [weight 0.5 each for the arms]
//   The 4 inner 2x2 quad centres at (±0.5, ±0.5)
//
// Reference: "Next Generation Post Processing in Call of Duty: Advanced Warfare"
// (Jimenez 2014), slides 162-163.
// ---------------------------------------------------------------------------

[shader("fragment")]
float4 DownsamplePS(VSOut i) : SV_TARGET
{
    float2 uv = i.UV;
    float2 ts = push.texelSize; // texel size in UV space

    // --- 13-tap sample set ---

    // Centre sample
    float3 s00 = srcImage.Sample(uv).rgb;

    // 4 cardinal neighbours at ±1 texel (weight 2 each in the standard sum)
    float3 s10 = srcImage.Sample(uv + float2( ts.x,  0.0 )).rgb;
    float3 s01 = srcImage.Sample(uv + float2( 0.0,   ts.y)).rgb;
    float3 sm1 = srcImage.Sample(uv + float2(-ts.x,  0.0 )).rgb;
    float3 s0m = srcImage.Sample(uv + float2( 0.0,  -ts.y)).rgb;

    // 4 inner quad centres (2x2 bilinear boxes at ±0.5 texel)
    float3 q00 = srcImage.Sample(uv + float2(-ts.x, -ts.y) * 0.5).rgb;
    float3 q10 = srcImage.Sample(uv + float2( ts.x, -ts.y) * 0.5).rgb;
    float3 q01 = srcImage.Sample(uv + float2(-ts.x,  ts.y) * 0.5).rgb;
    float3 q11 = srcImage.Sample(uv + float2( ts.x,  ts.y) * 0.5).rgb;

    // 4 outer corner taps at ±1 texel diagonals
    float3 c00 = srcImage.Sample(uv + float2(-ts.x, -ts.y)).rgb;
    float3 c10 = srcImage.Sample(uv + float2( ts.x, -ts.y)).rgb;
    float3 c01 = srcImage.Sample(uv + float2(-ts.x,  ts.y)).rgb;
    float3 c11 = srcImage.Sample(uv + float2( ts.x,  ts.y)).rgb;

    float3 result;

    if (push.mipLevel == 0)
    {
        // First downsample: brightpass threshold + Karis average to suppress fireflies.
        // Weight each 2x2 group with Karis weighting before averaging the groups.

        // Inner quad (weight 0.5 in final blend)
        float3 innerGroup = KarisAverage(q00) + KarisAverage(q10)
                          + KarisAverage(q01) + KarisAverage(q11);
        innerGroup *= 0.25; // average of 4

        // Four corner 2x2 boxes (weight 0.125 each = 0.5 total)
        float3 boxTL = KarisAverage(c00) + KarisAverage(s0m) + KarisAverage(sm1) + KarisAverage(s00);
        float3 boxTR = KarisAverage(s0m) + KarisAverage(c10) + KarisAverage(s00) + KarisAverage(s10);
        float3 boxBL = KarisAverage(sm1) + KarisAverage(s00) + KarisAverage(c01) + KarisAverage(s01);
        float3 boxBR = KarisAverage(s00) + KarisAverage(s10) + KarisAverage(s01) + KarisAverage(c11);
        boxTL *= 0.25;
        boxTR *= 0.25;
        boxBL *= 0.25;
        boxBR *= 0.25;

        // Blend: inner gets weight 0.5, each corner box gets 0.125
        result = innerGroup * 0.5
               + (boxTL + boxTR + boxBL + boxBR) * 0.125;

        // Brightpass: soft-knee threshold
        float lum = Luminance(result);
        float knee = push.threshold * 0.5;
        float rq = clamp(lum - push.threshold + knee, 0.0, 2.0 * knee);
        rq = (rq * rq) / (4.0 * knee + 1e-5);
        result *= max(rq, lum - push.threshold) / max(lum, 1e-5);
    }
    else
    {
        // Subsequent mips: plain weighted average (no Karis, no threshold)
        // Inner quad (weight 0.5)
        float3 innerGroup = q00 + q10 + q01 + q11;
        innerGroup *= 0.25;

        // Corner boxes (weight 0.5 total, 0.125 each)
        float3 boxTL = c00 + s0m + sm1 + s00;
        float3 boxTR = s0m + c10 + s00 + s10;
        float3 boxBL = sm1 + s00 + c01 + s01;
        float3 boxBR = s00 + s10 + s01 + c11;
        boxTL *= 0.25;
        boxTR *= 0.25;
        boxBL *= 0.25;
        boxBR *= 0.25;

        result = innerGroup * 0.5
               + (boxTL + boxTR + boxBL + boxBR) * 0.125;
    }

    return float4(result, 1.0);
}

// ---------------------------------------------------------------------------
// AddPS — simple passthrough for additively blending bloom into the HDR scene.
// The pipeline uses srcFactor=One / dstFactor=One; this shader outputs the
// bloom contribution (srcImage * strength) and the GPU performs the addition.
// ---------------------------------------------------------------------------

[shader("fragment")]
float4 AddPS(VSOut i) : SV_TARGET
{
    float3 bloom = srcImage.Sample(i.UV).rgb;
    return float4(bloom * push.strength, 0.0);
}

// ---------------------------------------------------------------------------
// UpsamplePS — 9-tap tent filter (bilinear 3x3 kernel)
//
// Blends the bilinearly upscaled current mip with the higher-resolution mip
// (addImage) using additive blending weighted by push.strength.
// ---------------------------------------------------------------------------

[shader("fragment")]
float4 UpsamplePS(VSOut i) : SV_TARGET
{
    float2 uv = i.UV;
    float2 ts = push.texelSize;

    // 9-tap 3x3 tent filter — bilinear hardware samples at ±1 texel offsets
    // Weights: corners=1, edges=2, centre=4  (sum=16)
    float3 t  = float3(0.0, 0.0, 0.0);

    t += srcImage.Sample(uv + float2(-ts.x, -ts.y)).rgb * 1.0;
    t += srcImage.Sample(uv + float2( 0.0,  -ts.y)).rgb * 2.0;
    t += srcImage.Sample(uv + float2( ts.x, -ts.y)).rgb * 1.0;

    t += srcImage.Sample(uv + float2(-ts.x,  0.0 )).rgb * 2.0;
    t += srcImage.Sample(uv + float2( 0.0,   0.0 )).rgb * 4.0;
    t += srcImage.Sample(uv + float2( ts.x,  0.0 )).rgb * 2.0;

    t += srcImage.Sample(uv + float2(-ts.x,  ts.y)).rgb * 1.0;
    t += srcImage.Sample(uv + float2( 0.0,   ts.y)).rgb * 2.0;
    t += srcImage.Sample(uv + float2( ts.x,  ts.y)).rgb * 1.0;

    t *= (1.0 / 16.0);

    // Additively blend upsampled bloom into the higher-res mip
    float3 hi = addImage.Sample(uv).rgb;
    float3 result = hi + t * push.strength;

    return float4(result, 1.0);
}
