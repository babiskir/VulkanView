// sky.slang — Procedural Preetham sky + sun disc
// Copyright (c) 2025 Holochip Corporation

struct SkyPushConstants {
    column_major float4x4 invViewProj;   // inverse(proj * view)
    float3   sunDirection;               // normalized, pointing TOWARD sun (world space)
    float    turbidity;                  // 1.0–10.0
    float3   groundColor;               // horizon/ground color
    float    sunIntensity;              // brightness multiplier
    float    sunDiscAngle;             // cos of sun disc half-angle
    uint     useEnvMap;                // 1 = sample env cubemap, 0 = Preetham procedural
    float2   _pad;
};
[[vk::push_constant]] SkyPushConstants sky;

// Equirectangular environment map (binding 0, set 0)
[[vk::binding(0, 0)]] Sampler2D envMap;

// Convert a direction vector to equirectangular UV coordinates
float2 dirToEquirect(float3 dir) {
    float phi   = atan2(dir.z, dir.x);           // [-pi, pi]
    float theta = asin(clamp(dir.y, -1.0, 1.0)); // [-pi/2, pi/2]
    return float2(phi / (2.0 * 3.14159265) + 0.5, 0.5 - theta / 3.14159265);
}

// Preetham model: compute sky luminance distribution
float3 preetham(float3 viewDir, float3 sunDir, float turbidity) {
    float sunTheta = acos(clamp(sunDir.y, 0.0, 1.0));
    float theta = acos(clamp(viewDir.y, -1.0, 1.0));
    float gamma = acos(clamp(dot(viewDir, sunDir), -1.0, 1.0));

    // Zenith luminance (Perez model coefficients)
    float t = turbidity;
    float t2 = t * t;
    float chi = (4.0/9.0 - t/120.0) * (3.14159 - 2.0 * sunTheta);
    float Yz = (4.0453*t - 4.9710) * tan(chi) - 0.2155*t + 2.4192;
    Yz = max(Yz, 0.0);

    // CIE x,y chromaticity at zenith
    float3 theta3 = float3(sunTheta*sunTheta*sunTheta, sunTheta*sunTheta, sunTheta);
    float3 Yx = float3(0.00166,-0.00375,0.00209) * t2 + float3(-0.02903,0.06377,-0.03202) * t + float3(0.11693,-0.21196,0.06052);
    float Xz = dot(Yx, theta3) + (t2 * 0.0 - 0.00515 * t + 0.25886);
    float3 Yy = float3(0.00275,-0.00610,0.00317) * t2 + float3(-0.04214,0.08970,-0.04153) * t + float3(0.15346,-0.26756,0.06669);
    float Zz = dot(Yy, theta3) + (-0.00043 * t2 + 0.00516 * t + 0.26688);

    // Perez distribution function coefficients
    float A = -0.0193 * t - 0.2592;
    float B = -0.0665 * t + 0.0008;
    float C =  0.0000 * t + 0.2125;
    float D = -0.0670 * t - 0.8989;
    float E =  0.0000 * t + 0.0452;

    float Ax = -0.0141*t - 0.4286; float Bx = -0.0670*t + 0.0272; float Cx = -0.0567*t + 0.6887; float Dx = -0.1294*t - 0.3468; float Ex = -0.0290*t + 0.0166;
    float Ay = -0.0117*t - 0.2903; float By = -0.0587*t + 0.0228; float Cy =  0.2187*t - 0.2969; float Dy = -0.1655*t + 0.0572; float Ey = -0.0255*t + 0.0003;

    float cosGamma = cos(gamma);
    float cosSun   = cos(sunTheta);

    float F  = (1.0 + A*exp(B/max(cos(theta),1e-4))) * (1.0 + C*exp(D*gamma) + E*cosGamma*cosGamma);
    float F0 = (1.0 + A*exp(B)) * (1.0 + C*exp(D*sunTheta) + E*cosSun*cosSun);

    float Fx  = (1.0 + Ax*exp(Bx/max(cos(theta),1e-4))) * (1.0 + Cx*exp(Dx*gamma) + Ex*cosGamma*cosGamma);
    float Fx0 = (1.0 + Ax*exp(Bx)) * (1.0 + Cx*exp(Dx*sunTheta) + Ex*cosSun*cosSun);

    float Fy  = (1.0 + Ay*exp(By/max(cos(theta),1e-4))) * (1.0 + Cy*exp(Dy*gamma) + Ey*cosGamma*cosGamma);
    float Fy0 = (1.0 + Ay*exp(By)) * (1.0 + Cy*exp(Dy*sunTheta) + Ey*cosSun*cosSun);

    float Y = Yz * F  / max(F0,  1e-6);
    float x = Xz * Fx / max(Fx0, 1e-6);
    float y = Zz * Fy / max(Fy0, 1e-6);

    // CIE xyY to XYZ
    float X = (Y / max(y, 1e-6)) * x;
    float Z = (Y / max(y, 1e-6)) * (1.0 - x - y);

    // XYZ to linear sRGB
    float3 rgb;
    rgb.r =  3.2406*X - 1.5372*Y - 0.4986*Z;
    rgb.g = -0.9689*X + 1.8758*Y + 0.0415*Z;
    rgb.b =  0.0557*X - 0.2040*Y + 1.0570*Z;
    // Scale Preetham's physical luminance into a workable HDR range.
    // ~0.04 puts zenith luminance around 0.3–1.5 linear, which survives Uncharted2 tonemapping
    // without washing out into white. Increase exposure slider if the sky looks dark.
    return max(rgb * 0.04, float3(0.0, 0.0, 0.0));
}

struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

[[shader("vertex")]]
VSOut VSMain(uint vid : SV_VertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.uv  = uv;
    o.pos = float4(uv * 2.0 - 1.0, 1.0, 1.0);
    return o;
}

[[shader("fragment")]]
float4 PSMain(VSOut input) : SV_TARGET {
    float4 clipPos = float4(input.uv * 2.0 - 1.0, 1.0, 1.0);
    float4 worldPos = mul(sky.invViewProj, clipPos);
    float3 viewDir = normalize(worldPos.xyz / worldPos.w);

    float3 color;

    if (sky.useEnvMap != 0u) {
        // Sample equirectangular HDR environment map
        float2 uv = dirToEquirect(viewDir);
        color = envMap.Sample(uv).rgb * sky.sunIntensity;
    } else {
        float3 sampleDir = viewDir;
        bool belowHorizon = sampleDir.y < 0.0;
        if (belowHorizon) sampleDir.y = 0.001;
        sampleDir = normalize(sampleDir);

        color = preetham(sampleDir, normalize(sky.sunDirection), sky.turbidity);
        color *= sky.sunIntensity;

        if (belowHorizon) {
            float t = clamp(-viewDir.y * 8.0, 0.0, 1.0);
            color = lerp(color, sky.groundColor, t);
        }

        // Sun disc: sharp bright core that fades at the edge (corona glow)
        float cosAngle = dot(viewDir, normalize(sky.sunDirection));
        if (cosAngle > sky.sunDiscAngle) {
            // Normalise how deep inside the disc we are (0 = edge, 1 = centre)
            float t = (cosAngle - sky.sunDiscAngle) / (1.0 - sky.sunDiscAngle);
            float disc = smoothstep(0.0, 1.0, t);
            color = lerp(color, float3(12.0, 10.0, 7.0) * sky.sunIntensity, disc);
        }
    }

    return float4(color, 1.0);
}
