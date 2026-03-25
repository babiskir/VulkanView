// water.slang — Water surface shader
// Ported from WaterSurfaceRendering demo (MIT License)
// Copyright (c) 2025 Holochip Corporation

#define M_PI 3.14159265358979323846f
#define ONE_OVER_PI (1.0f / M_PI)
#define WATER_IOR 1.33477f
#define AIR_IOR   1.0f
#define IOR_AIR2WATER (AIR_IOR / WATER_IOR)

// ---------------------------------------------------------------------------
// Uniform buffer objects

struct WaterVertexUBO {
    column_major float4x4 model;
    column_major float4x4 view;
    column_major float4x4 proj;
    float WSHeightAmp;
    float WSChoppy;
    float texScale;
    float _pad;
};

struct WaterFragUBO {
    float3 camPos;
    float  height;
    float3 absorpCoef;
    float  _p0;
    float3 scatterCoef;
    float  _p1;
    float3 backscatterCoef;
    float  _p2;
    float3 terrainColor;
    float  _p3;
    float  skyIntensity;
    float  specularIntensity;
    float  specularHighlights;
    uint   useEnvMap;    // 1 = sample EnvMap for sky reflections
    float3 sunDirection;
    float  sunIntensity;
    // Preetham sky parameters
    float3 skyA;
    float  turbidity;
    float3 skyB;
    float  _p4;
    float3 skyC;
    float  _p5;
    float3 skyD;
    float  _p6;
    float3 skyE;
    float  _p7;
    float3 ZenithLum;
    float  _p8;
    float3 ZeroThetaSun;
    float  _p9;
};

[[vk::binding(0, 0)]] ConstantBuffer<WaterVertexUBO> vertUBO;
[[vk::binding(1, 0)]] ConstantBuffer<WaterFragUBO>   fragUBO;
[[vk::binding(2, 0)]] Sampler2D DisplacementMap;
[[vk::binding(3, 0)]] Sampler2D NormalMap;
[[vk::binding(4, 0)]] Sampler2D EnvMap;   // equirectangular HDR sky (reflections)

// Direction -> equirectangular UV
float2 dirToEquirect(float3 dir) {
    float phi   = atan2(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0f, 1.0f));
    return float2(phi / (2.0f * M_PI) + 0.5f, 0.5f - theta / M_PI);
}

// ---------------------------------------------------------------------------
// Vertex shader

struct VSIn {
    float3 pos : POSITION;
    float2 uv  : TEXCOORD0;
};

struct VSOut {
    float4 clipPos  : SV_POSITION;
    float4 worldPos : TEXCOORD0;   // xyz = world pos, w = jacobian
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
};

[[shader("vertex")]]
VSOut VSMain(VSIn vin)
{
    VSOut vout;
    float2 suv = vin.uv * vertUBO.texScale;

    float4 D = DisplacementMap.SampleLevel(suv, 0);
    D.y *= vertUBO.WSHeightAmp;

    float3 displaced = vin.pos + D.xyz;
    vout.worldPos = float4(displaced, D.w);

    float4 worldPos4 = mul(vertUBO.model, float4(displaced, 1.0));
    float4 viewPos   = mul(vertUBO.view,  worldPos4);
    vout.clipPos     = mul(vertUBO.proj,  viewPos);

    float4 slope = NormalMap.SampleLevel(suv, 0);
    float3 n = float3(
        -(slope.x / (1.0f + vertUBO.WSChoppy * slope.z)),
        1.0f,
        -(slope.y / (1.0f + vertUBO.WSChoppy * slope.w))
    );
    vout.normal = normalize(n);
    vout.uv = vin.uv;

    return vout;
}

// ---------------------------------------------------------------------------
// Fragment shader helpers

float3 SkyLuminance(float3 sunDir, float3 viewDir);

float FresnelFull(float theta_i, float theta_t)
{
    if (theta_i <= 0.00001f) {
        float at0 = (WATER_IOR - 1.0f) / (WATER_IOR + 1.0f);
        return at0 * at0;
    }
    float t1 = sin(theta_i - theta_t) / sin(theta_i + theta_t);
    float t2 = tan(theta_i - theta_t) / tan(theta_i + theta_t);
    return 0.5f * (t1 * t1 + t2 * t2);
}

float3 Attenuate(float dist, float depth)
{
    const float kScale = 0.1f;
    return exp(-fragUBO.absorpCoef  * kScale * dist
               -fragUBO.scatterCoef * kScale * depth);
}

// Flat terrain at y=0
float IntersectTerrain(float3 rayOrg, float3 rayDir)
{
    // plane: y = 0, normal = (0,1,0), d=0
    return -rayOrg.y / rayDir.y;
}

float3 TerrainColor(float2 p)
{
    return fragUBO.terrainColor;
}

float3 ComputeTerrainRadiance(float3 p_g, float3 incidentDir)
{
    float3 terrainNormal = float3(0.0f, 1.0f, 0.0f);
    return TerrainColor(p_g.xz) * max(dot(terrainNormal, -incidentDir), 0.0f);
}

float3 ComputeWaterSurfaceColor(float3 rayOrg, float3 rayDir, float3 p_w, float3 kNormal)
{
    float3 kLightDir = normalize(fragUBO.sunDirection);

    float3 kReflectDir  = reflect(rayDir, kNormal);
    float3 kSkyReflected = SkyLuminance(kLightDir, kReflectDir);

    // Diffuse atmospheric skylight
    float3 L_a;
    {
        float kNdL = max(dot(kNormal, kLightDir), 0.0f);
        L_a = fragUBO.skyIntensity * kNdL * kSkyReflected;
    }

    float3 kViewDir = -rayDir;

    // Blinn-Phong specular
    float3 L_s;
    {
        float3 kHalfWay = normalize(kLightDir + kViewDir);
        float specular = fragUBO.specularIntensity *
            clamp(pow(max(dot(kNormal, kHalfWay), 0.0f), fragUBO.specularHighlights), 0.0f, 1.0f);
        L_s = fragUBO.sunIntensity * specular * kSkyReflected;
    }

    float3 kRefractDir = refract(-kLightDir, kNormal, IOR_AIR2WATER);

    // Subsurface / underwater radiance
    float3 L_u;
    {
        float3 E_d0  = M_PI * L_a;
        float3 L_df0 = (0.33f * fragUBO.backscatterCoef) / fragUBO.absorpCoef * (E_d0 * ONE_OVER_PI);

        float t_g = IntersectTerrain(p_w, kRefractDir);
        float3 p_g = p_w + t_g * kRefractDir;
        float3 L_g = ComputeTerrainRadiance(p_g, kRefractDir);

        float depth = abs(p_w.y - p_g.y);
        L_u = Attenuate(t_g, 0.0f) * L_g
            + (1.0f - Attenuate(t_g, depth)) * L_df0;
    }

    float cosI = dot(kNormal, kViewDir);
    float cosT = dot(-kNormal, kRefractDir);
    float F_r  = FresnelFull(acos(clamp(cosI, 0.0f, 1.0f)), acos(clamp(cosT, 0.0f, 1.0f)));

    return F_r * (L_s + L_a) + (1.0f - F_r) * L_u;
}

// ---------------------------------------------------------------------------
// Preetham sky (simplified from water demo)

float SaturateDot(float3 v, float3 u) { return max(dot(v, u), 0.0f); }

float3 ComputePerezLuminanceYxy(float theta, float gamma)
{
    const float kBias = 1e-3f;
    return (1.0f + fragUBO.skyA * exp(fragUBO.skyB / (cos(theta) + kBias)))
         * (1.0f + fragUBO.skyC * exp(fragUBO.skyD * gamma)
                 + fragUBO.skyE * cos(gamma) * cos(gamma));
}

float3 ComputeSkyLuminance(float3 kSunDir, float3 kViewDir)
{
    float thetaView = acos(SaturateDot(kViewDir, float3(0.0f, 1.0f, 0.0f)));
    float gammaView = acos(SaturateDot(kSunDir, kViewDir));

    float3 f = ComputePerezLuminanceYxy(thetaView, gammaView);
    return fragUBO.ZenithLum * (f / max(fragUBO.ZeroThetaSun, float3(1e-6f)));
}

float3 YxyToXYZ(float3 Yxy)
{
    float Y = Yxy.r, x = Yxy.g, y = Yxy.b;
    float X = x * (Y / max(y, 1e-6f));
    float Z = (1.0f - x - y) * (Y / max(y, 1e-6f));
    return float3(X, Y, Z);
}

float3 XYZToRGB(float3 XYZ)
{
    return float3(
         2.3706743f * XYZ.x - 0.9000405f * XYZ.y - 0.4706338f * XYZ.z,
        -0.5138850f * XYZ.x + 1.4253036f * XYZ.y + 0.0885814f * XYZ.z,
         0.0052982f * XYZ.x - 0.0146949f * XYZ.y + 1.0093968f * XYZ.z
    );
}

float3 YxyToRGB(float3 Yxy) { return XYZToRGB(YxyToXYZ(Yxy)); }

float3 SkyLuminance(float3 kSunDir, float3 kViewDir)
{
    if (fragUBO.useEnvMap != 0u) {
        // Sample the HDR equirectangular env map for accurate sky reflections
        float2 uv = dirToEquirect(kViewDir);
        return EnvMap.Sample(uv).rgb;
    }
    float kSunDisk = smoothstep(0.997f, 1.0f, SaturateDot(kViewDir, kSunDir));
    float3 kSkyLum = YxyToRGB(ComputeSkyLuminance(kSunDir, kViewDir));
    return kSkyLum * 0.05f + kSunDisk * kSkyLum;
}

// ---------------------------------------------------------------------------
// Fragment shader

struct FSIn {
    float4 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
};

[[shader("fragment")]]
float4 PSMain(FSIn fin) : SV_Target
{
    float3 rayOrg = fragUBO.camPos;
    float3 rayDir = normalize(fin.worldPos.xyz - fragUBO.camPos);
    float3 kNormal = normalize(fin.normal);

    float3 p_w = float3(fin.worldPos.x, fin.worldPos.y + fragUBO.height, fin.worldPos.z);

    float3 color = ComputeWaterSurfaceColor(rayOrg, rayDir, p_w, kNormal);

    // Foam on crests (negative jacobian)
    if (fin.worldPos.w < 0.0f)
        color = float3(1.0f, 1.0f, 1.0f);

    float4 fragColor = float4(color, 1.0f);

    // Tone mapping (Reinhard-like)
    fragColor = float4(1.0f) - exp(-fragColor * 2.0f);

    return fragColor;
}
