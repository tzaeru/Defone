using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Sends mainCameraView to the ControlPy TCP server for drone detection,
/// receives bounding boxes, and writes mainCameraView + drawn boxes to augmentedCameraView.
/// </summary>
public class PythonCommunicator : MonoBehaviour
{
    [Header("Textures")]
    public RenderTexture mainCameraView;
    public RenderTexture augmentedCameraView;

    [Header("Connection")]
    public string host = "127.0.0.1";
    public int port = 5555;

    [Header("Detection")]
    [Tooltip("Seconds between sending a frame for detection.")]
    public float detectionInterval = 0.2f;
    [Tooltip("Minimum confidence for drawn boxes (0–1). Higher reduces false positives (e.g. branches).")]
    [Range(0f, 1f)]
    public float minConfidence = 0.4f;
    [Tooltip("Color of bounding box lines.")]
    public Color boxColor = Color.green;
    [Tooltip("Line width in pixels.")]
    public int lineWidth = 2;
    [Tooltip("Minimum seconds between printing received bounding boxes to the console.")]
    public float boxLogInterval = 2f;

    // Shared state between main thread and worker
    private readonly object _lock = new object();
    private byte[] _pendingImage;
    private List<BoundingBox> _receivedBoxes = new List<BoundingBox>();
    private bool _requestInFlight;
    private bool _pendingBoxLog;
    private float _lastBoxLogTime = -999f;
    private float _lastRequestTime;
    private Texture2D _readbackTexture;
    private Material _lineMaterial;
    private bool _lineMaterialFailed;

    [Serializable]
    public class BoundingBoxDto
    {
        public float x1, y1, x2, y2;
        public float confidence;
        public string className;
    }

    [Serializable]
    public class BoxesResponseDto
    {
        public BoundingBoxDto[] boxes;
    }

    public struct BoundingBox
    {
        public float x1, y1, x2, y2;
        public float confidence;
        public string className;
    }

    void Start()
    {
        if (mainCameraView != null && augmentedCameraView != null &&
            (mainCameraView.width != augmentedCameraView.width || mainCameraView.height != augmentedCameraView.height))
            Debug.LogWarning("[PythonCommunicator] mainCameraView and augmentedCameraView have different sizes; boxes may not align.");
    }

    void OnDestroy()
    {
        if (_lineMaterial != null)
            Destroy(_lineMaterial);
    }

    void Update()
    {
        if (mainCameraView == null || augmentedCameraView == null) return;

        // 1) Always write mainCameraView + last known boxes to augmentedCameraView
        BlitMainAndDrawBoxes();

        // 2) Print received boxes at most once every boxLogInterval seconds
        if (boxLogInterval > 0)
        {
            lock (_lock)
            {
                if (_pendingBoxLog && Time.time - _lastBoxLogTime >= boxLogInterval)
                {
                    _lastBoxLogTime = Time.time;
                    _pendingBoxLog = false;
                    int n = _receivedBoxes.Count;
                    if (n == 0)
                        Debug.Log("[PythonCommunicator] Received 0 bounding boxes.");
                    else
                    {
                        var sb = new StringBuilder();
                        sb.AppendFormat("[PythonCommunicator] Received {0} bounding box(es): ", n);
                        for (int i = 0; i < n; i++)
                        {
                            var b = _receivedBoxes[i];
                            if (i > 0) sb.Append(" | ");
                            sb.AppendFormat("({0:F0},{1:F0})-({2:F0},{3:F0}) conf={4:F2} class={5}",
                                b.x1, b.y1, b.x2, b.y2, b.confidence, b.className);
                        }
                        Debug.Log(sb.ToString());
                    }
                }
            }
        }

        // 3) Periodically send a frame for detection (one request at a time)
        if (detectionInterval > 0 && Time.time - _lastRequestTime >= detectionInterval)
        {
            lock (_lock)
            {
                if (!_requestInFlight && _pendingImage == null)
                {
                    byte[] jpg = EncodeMainCameraToJpg();
                    if (jpg != null && jpg.Length > 0)
                    {
                        _requestInFlight = true;
                        _lastRequestTime = Time.time;
                        byte[] copy = jpg;
                        Thread t = new Thread(() => SendImageAndReceiveBoxes(copy));
                        t.IsBackground = true;
                        t.Start();
                    }
                }
            }
        }
    }

    private byte[] EncodeMainCameraToJpg()
    {
        if (mainCameraView == null) return null;
        int w = mainCameraView.width;
        int h = mainCameraView.height;
        if (w <= 0 || h <= 0) return null;

        if (_readbackTexture == null || _readbackTexture.width != w || _readbackTexture.height != h)
        {
            if (_readbackTexture != null) Destroy(_readbackTexture);
            _readbackTexture = new Texture2D(w, h, TextureFormat.RGB24, false);
        }

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = mainCameraView;
        _readbackTexture.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        _readbackTexture.Apply();
        RenderTexture.active = prev;

        byte[] jpg = _readbackTexture.EncodeToJPG(85);
        return jpg;
    }

    private void SendImageAndReceiveBoxes(byte[] jpgBytes)
    {
        try
        {
            using (var client = new TcpClient())
            {
                client.Connect(host, port);
                client.ReceiveTimeout = 30000;
                client.SendTimeout = 10000;
                using (NetworkStream stream = client.GetStream())
                {
                    ControlPyClient.SendMessage(stream, "detect", jpgBytes);
                    string replyType, payload;
                    if (ControlPyClient.TryReadMessage(stream, out replyType, out payload) && replyType == "boxes")
                    {
                        List<BoundingBox> boxes = ParseBoxes(payload);
                        lock (_lock)
                        {
                            _receivedBoxes = boxes;
                            _pendingBoxLog = true;
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[PythonCommunicator] Detect request failed: " + ex.Message);
        }
        finally
        {
            lock (_lock) { _requestInFlight = false; }
        }
    }

    private static List<BoundingBox> ParseBoxes(string json)
    {
        var list = new List<BoundingBox>();
        if (string.IsNullOrEmpty(json)) return list;
        try
        {
            var dto = JsonUtility.FromJson<BoxesResponseDto>(json);
            if (dto?.boxes == null) return list;
            foreach (var b in dto.boxes)
            {
                if (b == null) continue;
                list.Add(new BoundingBox
                {
                    x1 = b.x1, y1 = b.y1, x2 = b.x2, y2 = b.y2,
                    confidence = b.confidence,
                    className = b.className ?? ""
                });
            }
        }
        catch (Exception)
        {
            // ignore parse errors
        }
        return list;
    }

    private void BlitMainAndDrawBoxes()
    {
        Graphics.Blit(mainCameraView, augmentedCameraView);

        List<BoundingBox> boxes;
        lock (_lock) { boxes = new List<BoundingBox>(_receivedBoxes); }

        if (boxes.Count == 0) return;

        EnsureLineMaterial();
        if (_lineMaterial == null) return;

        int dstW = augmentedCameraView.width;
        int dstH = augmentedCameraView.height;
        int srcW = mainCameraView.width;
        int srcH = mainCameraView.height;
        // Scale from detection image space (mainCameraView when we sent the frame) to display RT
        float scaleX = (float)dstW / Mathf.Max(1, srcW);
        float scaleY = (float)dstH / Mathf.Max(1, srcH);

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = augmentedCameraView;

        GL.PushMatrix();
        // LoadPixelMatrix(left, right, bottom, top): (0,0) top-left, y down; so bottom=dstH, top=0
        GL.LoadPixelMatrix(0, dstW, dstH, 0);
        _lineMaterial.SetPass(0);
        _lineMaterial.color = boxColor;
        GL.Begin(GL.QUADS);
        float half = Mathf.Max(1f, lineWidth / 2f);

        foreach (var b in boxes)
        {
            if (b.confidence < minConfidence) continue;
            // Normalize to min/max (Python may return either order) and scale to display size
            float xMin = Mathf.Min(b.x1, b.x2) * scaleX;
            float xMax = Mathf.Max(b.x1, b.x2) * scaleX;
            float yMin = Mathf.Min(b.y1, b.y2) * scaleY;
            float yMax = Mathf.Max(b.y1, b.y2) * scaleY;
            // Clamp to view (with margin so border stays visible)
            xMin = Mathf.Clamp(xMin, -half, dstW + half);
            xMax = Mathf.Clamp(xMax, -half, dstW + half);
            yMin = Mathf.Clamp(yMin, -half, dstH + half);
            yMax = Mathf.Clamp(yMax, -half, dstH + half);
            DrawRectQuads(xMin, yMin, xMax, yMax, half);
        }

        GL.End();
        GL.PopMatrix();
        RenderTexture.active = prev;
    }

    private static void DrawRectQuads(float xMin, float yMin, float xMax, float yMax, float half)
    {
        // In LoadPixelMatrix(0,w,h,0): y=0 is top, y=h is bottom. So "top" of box = smaller y = yMin, "bottom" = yMax.
        // Draw four edge quads (each as a thin rectangle).
        // Top edge (at yMin)
        GL.Vertex3(xMin - half, yMin - half, 0);
        GL.Vertex3(xMax + half, yMin - half, 0);
        GL.Vertex3(xMax + half, yMin + half, 0);
        GL.Vertex3(xMin - half, yMin + half, 0);
        // Bottom edge (at yMax)
        GL.Vertex3(xMin - half, yMax - half, 0);
        GL.Vertex3(xMax + half, yMax - half, 0);
        GL.Vertex3(xMax + half, yMax + half, 0);
        GL.Vertex3(xMin - half, yMax + half, 0);
        // Left edge (at xMin)
        GL.Vertex3(xMin - half, yMin - half, 0);
        GL.Vertex3(xMin + half, yMin - half, 0);
        GL.Vertex3(xMin + half, yMax + half, 0);
        GL.Vertex3(xMin - half, yMax + half, 0);
        // Right edge (at xMax)
        GL.Vertex3(xMax - half, yMin - half, 0);
        GL.Vertex3(xMax + half, yMin - half, 0);
        GL.Vertex3(xMax + half, yMax + half, 0);
        GL.Vertex3(xMax - half, yMax + half, 0);
    }

    private void EnsureLineMaterial()
    {
        if (_lineMaterial != null || _lineMaterialFailed) return;
        Shader sh = Shader.Find("Unlit/Color");
        if (sh == null) sh = Shader.Find("Sprites/Default");
        if (sh == null)
        {
            _lineMaterialFailed = true;
            return;
        }
        _lineMaterial = new Material(sh);
        _lineMaterial.color = boxColor;
    }
}
