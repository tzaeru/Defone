Shader "Hidden/DepthVisualize"
{
    Properties
    {
        _MaxDistance ("Max Distance", Float) = 100.0
    }

    SubShader
    {
        Tags { "RenderPipeline" = "HDRenderPipeline" }
        ZWrite Off
        ZTest Always
        Cull Off

        Pass
        {
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.high-definition/Runtime/ShaderLibrary/ShaderVariables.hlsl"

            float _MaxDistance;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 texcoord   : TEXCOORD0;
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
                output.texcoord   = GetFullScreenTriangleTexCoord(input.vertexID);
                return output;
            }

            float4 Frag(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;
                // Sample the camera depth texture (raw depth)
                float rawDepth = LoadCameraDepth(uint2(uv * _ScreenSize.xy));
                // Convert to linear eye depth
                float linearDepth = LinearEyeDepth(rawDepth, _ZBufferParams);
                // Normalize to 0-1 range using max distance (near=white, far=black)
                float normalized = saturate(linearDepth / _MaxDistance);
                float grayscale = 1.0 - normalized;
                return float4(grayscale, grayscale, grayscale, 1.0);
            }
            ENDHLSL
        }
    }
}
