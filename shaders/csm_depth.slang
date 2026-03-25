// csm_depth.slang — depth-only pass for cascaded shadow maps
// Copyright (c) 2025 Holochip Corporation

struct CsmDepthUBO {
    column_major float4x4 lightSpaceMatrices[3];
};
[[vk::binding(0, 0)]] ConstantBuffer<CsmDepthUBO> csmUbo;

struct CsmDepthPushConstants {
    column_major float4x4 modelMatrix;
    int cascadeIndex;
    float _pad0;
    float _pad1;
    float _pad2;
};
[[vk::push_constant]] CsmDepthPushConstants pushConsts;

[[shader("vertex")]]
float4 VSMain(float3 inPos : POSITION) : SV_POSITION {
    return mul(csmUbo.lightSpaceMatrices[pushConsts.cascadeIndex], mul(pushConsts.modelMatrix, float4(inPos, 1.0)));
}
