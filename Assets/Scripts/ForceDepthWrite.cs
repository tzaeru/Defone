using UnityEngine;

/// <summary>
/// Forces all Renderer materials on this GameObject and its children to write to the depth buffer.
/// Fixes glTF-imported models that use transparent surface type in HDRP and don't appear in depth maps.
/// Attach to any GameObject with renderers (e.g. drone root).
/// </summary>
public class ForceDepthWrite : MonoBehaviour
{
    [Tooltip("If true, also switches materials from Transparent to Opaque surface type (HDRP).")]
    public bool forceOpaque = true;

    void Start()
    {
        var renderers = GetComponentsInChildren<Renderer>(true);
        int fixedCount = 0;

        foreach (var r in renderers)
        {
            foreach (var mat in r.materials)
            {
                if (mat == null) continue;
                bool changed = false;

                // HDRP Lit / StackLit / Unlit use _SurfaceType: 0=Opaque, 1=Transparent
                if (mat.HasProperty("_SurfaceType"))
                {
                    float surfaceType = mat.GetFloat("_SurfaceType");
                    if (surfaceType > 0.5f && forceOpaque)
                    {
                        mat.SetFloat("_SurfaceType", 0f);
                        changed = true;
                    }
                }

                // Force depth write on
                if (mat.HasProperty("_ZWrite"))
                {
                    if (mat.GetFloat("_ZWrite") < 0.5f)
                    {
                        mat.SetFloat("_ZWrite", 1f);
                        changed = true;
                    }
                }

                // HDRP transparent depth prepass/postpass
                if (mat.HasProperty("_TransparentDepthPrepassEnable"))
                {
                    mat.SetFloat("_TransparentDepthPrepassEnable", 1f);
                    mat.SetFloat("_TransparentDepthPostpassEnable", 1f);
                    changed = true;
                }

                // Set render queue to geometry (opaque) if it was in transparent range
                if (forceOpaque && mat.renderQueue >= 3000)
                {
                    mat.renderQueue = 2000;
                    changed = true;
                }

                // Update HDRP keywords after property changes
                if (changed)
                {
                    // Disable transparent keywords, enable opaque ones
                    mat.DisableKeyword("_SURFACE_TYPE_TRANSPARENT");
                    mat.EnableKeyword("_SURFACE_TYPE_OPAQUE");
                    // Disable alpha blending
                    if (mat.HasProperty("_BlendMode"))
                        mat.SetFloat("_BlendMode", 0f);
                    if (mat.HasProperty("_AlphaCutoffEnable"))
                        mat.SetFloat("_AlphaCutoffEnable", 0f);
                    // Set blend factors for opaque
                    mat.SetOverrideTag("RenderType", "Opaque");
                    mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                    mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
                    fixedCount++;
                }
            }
        }

        if (fixedCount > 0)
            Debug.Log($"[ForceDepthWrite] Fixed {fixedCount} materials on '{gameObject.name}' to write depth.");
    }
}
