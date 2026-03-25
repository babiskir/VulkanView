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

// Histogram-based auto-exposure [Ch18]
// Two compute passes:
//   BuildHistogramCS  — scatter log-luminance into a 256-bin histogram
//   ReduceHistogramCS — prefix-sum + weighted average, EV100 with eye adaptation

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// Binding 0: HDR scene color (read-only)
[[vk::binding(0, 0)]] Sampler2D sceneColor;

// Binding 1: 256-entry log-luminance histogram (read/write, atomics)
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> histogram;

// Binding 2: single float exposure multiplier written by ReduceHistogramCS
[[vk::binding(2, 0)]] RWStructuredBuffer<float> exposureBuffer;

// ---------------------------------------------------------------------------
// Push constants
// ---------------------------------------------------------------------------

struct BuildHistogramPush {
    float  minLogLum;    // minimum log2 luminance mapped to bin 0
    float  logLumRange;  // log2 luminance range (bin 255 = minLogLum + logLumRange)
    float2 texelSize;    // 1.0 / resolution, used to convert DTid to UV
};
[[vk::push_constant]] BuildHistogramPush buildPush;

struct ReduceHistogramPush {
    float minLogLum;    // same range as above
    float logLumRange;
    float targetLum;    // desired average scene luminance (linear)
    float adaptSpeed;   // adaptation speed (blending coefficient per second)
    float deltaTime;    // frame delta time in seconds
    float minExposure;  // clamp output exposure
    float maxExposure;
    float _pad;         // pad to 32 bytes
};

// Slang does not allow two push-constant blocks in the same module to have
// different types simultaneously; the correct pattern is a single struct
// placed at the top and used by whichever entry point needs it.  We instead
// use a union-style struct that covers both cases and let each entry point
// read the relevant fields.
struct ExposurePush {
    float  minLogLum;
    float  logLumRange;
    float2 texelSize;    // only used by BuildHistogramCS
    float  targetLum;    // only used by ReduceHistogramCS
    float  adaptSpeed;   // only used by ReduceHistogramCS
    float  deltaTime;    // only used by ReduceHistogramCS
    float  minExposure;  // only used by ReduceHistogramCS
    float  maxExposure;  // only used by ReduceHistogramCS
    float  baseEV;       // EV100 bias added to the computed target
    float2 _pad;         // pad to 48 bytes
};
[[vk::push_constant]] ExposurePush push;

// ---------------------------------------------------------------------------
// Shared memory for ReduceHistogramCS prefix sum
// ---------------------------------------------------------------------------

groupshared float sums[256];

// ---------------------------------------------------------------------------
// BuildHistogramCS
//
// Each 16x16 thread group covers a 16x16 tile of the HDR image.
// Each thread reads one texel, computes log2 luminance, maps it to a
// histogram bin [0..255] and atomically increments that bin.
// ---------------------------------------------------------------------------

[numthreads(16, 16, 1)]
[shader("compute")]
void BuildHistogramCS(uint3 DTid : SV_DispatchThreadID)
{
    // Reconstruct UV from dispatch thread ID and texel size
    float2 uv = (float2(DTid.xy) + float2(0.5, 0.5)) * push.texelSize;

    // Clamp to valid UV range to avoid sampling outside the image
    if (uv.x > 1.0 || uv.y > 1.0)
        return;

    float3 color = sceneColor.SampleLevel(uv, 0).rgb;

    // Perceptual luminance; clamp to avoid log(0)
    float lum = max(dot(color, float3(0.2126, 0.7152, 0.0722)), 1e-4);

    // Map log2 luminance to [0..255]
    float logLum = log2(lum);
    float t = (logLum - push.minLogLum) / push.logLumRange;
    t = saturate(t);
    uint bin = (uint)(t * 255.0 + 0.5);
    bin = clamp(bin, 0u, 255u);

    InterlockedAdd(histogram[bin], 1u);
}

// ---------------------------------------------------------------------------
// ReduceHistogramCS
//
// One thread group of 256 threads processes the entire 256-bin histogram.
// Computes a weighted average log luminance using a running prefix sum,
// derives a target EV100, exponentially smooths toward it, and writes
// the resulting linear exposure multiplier.  Clears the histogram for
// the next frame.
// ---------------------------------------------------------------------------

[numthreads(256, 1, 1)]
[shader("compute")]
void ReduceHistogramCS(uint3 GTid : SV_GroupThreadID)
{
    uint bin = GTid.x; // 0..255

    // Load bin count into shared memory as a float weight
    float count = (float)histogram[bin];
    sums[bin] = count;

    GroupMemoryBarrierWithGroupSync();

    // Parallel prefix sum (inclusive scan) to compute total pixel count
    // and weighted sum of log luminances.
    // We do a standard up-sweep / down-sweep in shared memory.

    // Up-sweep (reduce)
    [unroll]
    for (uint stride = 1; stride < 256; stride <<= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((bin & (2 * stride - 1)) == (2 * stride - 1))
        {
            sums[bin] += sums[bin - stride];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Total pixel count is in sums[255] after the reduction
    float totalPixels = sums[255];

    // Reset sums to weighted log-luminance for a second pass
    GroupMemoryBarrierWithGroupSync();

    // Re-load bin count (cleared sums was needed for total only)
    float binCount = (float)histogram[bin];

    // Log luminance value represented by this bin (bin centre)
    float t = ((float)bin + 0.5) / 255.0;
    float logLumBin = push.minLogLum + t * push.logLumRange;

    sums[bin] = binCount * logLumBin;

    GroupMemoryBarrierWithGroupSync();

    // Second reduction: sum of (count * logLum) across all bins
    [unroll]
    for (uint stride2 = 1; stride2 < 256; stride2 <<= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((bin & (2 * stride2 - 1)) == (2 * stride2 - 1))
        {
            sums[bin] += sums[bin - stride2];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Only thread 255 writes the result
    if (bin == 255)
    {
        float weightedLogLumSum = sums[255];
        float avgLogLum = (totalPixels > 0.0)
                        ? (weightedLogLumSum / totalPixels)
                        : push.minLogLum;

        float avgLum = exp2(avgLogLum);

        // Target EV100 to expose avgLum as targetLum
        float targetEV = log2(push.targetLum / max(avgLum, 1e-6)) + push.baseEV;

        // Convert target EV100 to linear exposure multiplier
        // exposure = 1 / (2^EV100)  with EV100 = log2(avgLum/targetLum) inverted
        float targetExposure = push.targetLum / max(avgLum, 1e-6);
        targetExposure = clamp(targetExposure, push.minExposure, push.maxExposure);

        // Exponential smoothing toward target (eye adaptation)
        float current = exposureBuffer[0];
        float adapted = lerp(current, targetExposure, saturate(push.adaptSpeed * push.deltaTime));
        exposureBuffer[0] = adapted;
    }

    // All threads clear their histogram bin for the next frame
    histogram[bin] = 0u;
}
