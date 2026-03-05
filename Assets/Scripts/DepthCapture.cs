using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Captures the depth buffer from the Main Camera after rendering and blits it
/// as a linearized grayscale image into the depthView RenderTexture.
/// Attach to any active GameObject (e.g. PythonTalker) and assign fields in Inspector.
/// </summary>
public class DepthCapture : MonoBehaviour
{
    [Tooltip("Camera whose depth buffer to capture.")]
    public Camera sourceCamera;

    [Tooltip("RenderTexture to write the depth visualization into.")]
    public RenderTexture depthView;

    [Tooltip("Maximum visible distance in meters. Depth beyond this is black.")]
    public float maxDistance = 100f;

    private Material _depthMaterial;

    void OnEnable()
    {
        Shader shader = Shader.Find("Hidden/DepthVisualize");
        if (shader == null)
        {
            Debug.LogError("[DepthCapture] Could not find Hidden/DepthVisualize shader.");
            enabled = false;
            return;
        }
        _depthMaterial = new Material(shader);

        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
    }

    void OnDisable()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        if (_depthMaterial != null)
        {
            Destroy(_depthMaterial);
            _depthMaterial = null;
        }
    }

    private void OnEndCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        if (camera != sourceCamera || depthView == null || _depthMaterial == null)
            return;

        _depthMaterial.SetFloat("_MaxDistance", maxDistance);

        var cmd = CommandBufferPool.Get("DepthVisualize");
        cmd.SetRenderTarget(depthView);
        cmd.DrawProcedural(Matrix4x4.identity, _depthMaterial, 0, MeshTopology.Triangles, 3);
        context.ExecuteCommandBuffer(cmd);
        context.Submit();
        CommandBufferPool.Release(cmd);
    }
}
