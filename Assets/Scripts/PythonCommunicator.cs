using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Sends mainCameraView to the ControlPy TCP server for drone detection,
/// receives bounding boxes, and writes mainCameraView + drawn boxes to augmentedCameraView.
/// Uses AsyncGPUReadback to avoid stalling the render pipeline.
/// Keeps a persistent TCP connection to avoid per-frame connect overhead.
/// </summary>
public class PythonCommunicator : MonoBehaviour
{
    [Header("Textures")]
    public RenderTexture mainCameraView;
    public RenderTexture augmentedCameraView;
    [Tooltip("Depth visualization RT from DepthCapture. If set, depth is sent alongside color.")]
    public RenderTexture depthView;

    [Header("Connection")]
    public string host = "127.0.0.1";
    public int port = 5555;

    [Header("Detection")]
    [Tooltip("Seconds between sending a frame for detection.")]
    public float detectionInterval = 0.2f;
    [Tooltip("Minimum confidence for drawn boxes (0-1).")]
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
    private List<BoundingBox> _receivedBoxes = new List<BoundingBox>();
    private bool _requestInFlight;
    private bool _pendingBoxLog;
    private float _lastBoxLogTime = -999f;
    private float _lastRequestTime;
    private Material _lineMaterial;
    private bool _lineMaterialFailed;

    // Pre-allocated textures for encoding (avoids GC alloc per frame)
    private Texture2D _colorEncodeTexture;
    private Texture2D _depthEncodeTexture;

    // Async readback coordination
    private byte[] _asyncColorJpg;
    private byte[] _asyncDepthJpg;
    private bool _asyncColorDone;
    private bool _asyncDepthDone;
    private bool _asyncReadbackActive;

    // Persistent TCP connection
    private TcpClient _tcpClient;
    private NetworkStream _tcpStream;
    private readonly object _tcpLock = new object();
    private long _lastConnectAttemptMs;
    private const long RECONNECT_INTERVAL_MS = 2000;

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
        // Force Performant quality level at runtime (index 2) for usable frame rates
        if (QualitySettings.GetQualityLevel() != 2)
            QualitySettings.SetQualityLevel(2, true);

        if (mainCameraView != null && augmentedCameraView != null &&
            (mainCameraView.width != augmentedCameraView.width || mainCameraView.height != augmentedCameraView.height))
            Debug.LogWarning("[PythonCommunicator] mainCameraView and augmentedCameraView have different sizes; boxes may not align.");
    }

    void OnDestroy()
    {
        if (_lineMaterial != null) Destroy(_lineMaterial);
        if (_colorEncodeTexture != null) Destroy(_colorEncodeTexture);
        if (_depthEncodeTexture != null) Destroy(_depthEncodeTexture);
        CloseConnection();
    }

    void Update()
    {
        if (mainCameraView == null || augmentedCameraView == null) return;

        BlitMainAndDrawBoxes();

        // Print received boxes at most once every boxLogInterval seconds
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

        // Periodically kick off async readback for detection
        if (detectionInterval > 0 && Time.time - _lastRequestTime >= detectionInterval)
        {
            bool canStart;
            lock (_lock) { canStart = !_requestInFlight && !_asyncReadbackActive; }
            if (canStart)
            {
                _lastRequestTime = Time.time;
                StartAsyncReadback();
            }
        }
    }

    private void StartAsyncReadback()
    {
        _asyncReadbackActive = true;
        _asyncColorDone = false;
        _asyncDepthDone = depthView == null;
        _asyncColorJpg = null;
        _asyncDepthJpg = null;

        AsyncGPUReadback.Request(mainCameraView, 0, TextureFormat.RGB24, OnColorReadbackComplete);

        if (depthView != null)
            AsyncGPUReadback.Request(depthView, 0, TextureFormat.RGB24, OnDepthReadbackComplete);
    }

    private void OnColorReadbackComplete(AsyncGPUReadbackRequest req)
    {
        if (req.hasError)
        {
            _asyncColorDone = true;
            _asyncReadbackActive = false;
            return;
        }

        EnsureEncodeTexture(ref _colorEncodeTexture, req.width, req.height);
        _colorEncodeTexture.LoadRawTextureData(req.GetData<byte>());
        _colorEncodeTexture.Apply(false, false);
        _asyncColorJpg = _colorEncodeTexture.EncodeToJPG(85);
        _asyncColorDone = true;

        TryDispatchToNetwork();
    }

    private void OnDepthReadbackComplete(AsyncGPUReadbackRequest req)
    {
        if (req.hasError)
        {
            _asyncDepthDone = true;
            TryDispatchToNetwork();
            return;
        }

        EnsureEncodeTexture(ref _depthEncodeTexture, req.width, req.height);
        _depthEncodeTexture.LoadRawTextureData(req.GetData<byte>());
        _depthEncodeTexture.Apply(false, false);
        _asyncDepthJpg = _depthEncodeTexture.EncodeToJPG(85);
        _asyncDepthDone = true;

        TryDispatchToNetwork();
    }

    private void EnsureEncodeTexture(ref Texture2D tex, int w, int h)
    {
        if (tex == null || tex.width != w || tex.height != h)
        {
            if (tex != null) Destroy(tex);
            tex = new Texture2D(w, h, TextureFormat.RGB24, false);
        }
    }

    private void TryDispatchToNetwork()
    {
        if (!_asyncColorDone || !_asyncDepthDone) return;
        _asyncReadbackActive = false;

        byte[] colorJpg = _asyncColorJpg;
        byte[] depthJpg = _asyncDepthJpg;
        _asyncColorJpg = null;
        _asyncDepthJpg = null;

        if (colorJpg == null || colorJpg.Length == 0) return;

        lock (_lock) { _requestInFlight = true; }

        Thread t = new Thread(() => SendAndReceive(colorJpg, depthJpg));
        t.IsBackground = true;
        t.Start();
    }

    private void SendAndReceive(byte[] jpgBytes, byte[] depthJpgBytes)
    {
        try
        {
            NetworkStream stream = GetOrCreateConnection();
            if (stream == null) return;

            try
            {
                if (depthJpgBytes != null && depthJpgBytes.Length > 0)
                {
                    int colorLen = jpgBytes.Length;
                    byte[] combined = new byte[4 + colorLen + depthJpgBytes.Length];
                    combined[0] = (byte)(colorLen >> 24);
                    combined[1] = (byte)(colorLen >> 16);
                    combined[2] = (byte)(colorLen >> 8);
                    combined[3] = (byte)colorLen;
                    Buffer.BlockCopy(jpgBytes, 0, combined, 4, colorLen);
                    Buffer.BlockCopy(depthJpgBytes, 0, combined, 4 + colorLen, depthJpgBytes.Length);
                    ControlPyClient.SendMessage(stream, "detect_depth", combined);
                }
                else
                {
                    ControlPyClient.SendMessage(stream, "detect", jpgBytes);
                }

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
            catch (Exception)
            {
                CloseConnection();
                throw;
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

    private NetworkStream GetOrCreateConnection()
    {
        lock (_tcpLock)
        {
            if (_tcpClient != null && _tcpClient.Connected && _tcpStream != null)
                return _tcpStream;

            long nowMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            if (nowMs - _lastConnectAttemptMs < RECONNECT_INTERVAL_MS)
                return null;
            _lastConnectAttemptMs = nowMs;

            CloseConnectionLocked();
            try
            {
                _tcpClient = new TcpClient();
                _tcpClient.Connect(host, port);
                _tcpClient.ReceiveTimeout = 30000;
                _tcpClient.SendTimeout = 10000;
                _tcpClient.NoDelay = true;
                _tcpStream = _tcpClient.GetStream();
                return _tcpStream;
            }
            catch (Exception)
            {
                CloseConnectionLocked();
                return null;
            }
        }
    }

    private void CloseConnection()
    {
        lock (_tcpLock) { CloseConnectionLocked(); }
    }

    private void CloseConnectionLocked()
    {
        try { _tcpStream?.Close(); } catch { }
        try { _tcpClient?.Close(); } catch { }
        _tcpStream = null;
        _tcpClient = null;
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
        float scaleX = (float)dstW / Mathf.Max(1, srcW);
        float scaleY = (float)dstH / Mathf.Max(1, srcH);

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = augmentedCameraView;

        GL.PushMatrix();
        GL.LoadPixelMatrix(0, dstW, dstH, 0);
        _lineMaterial.SetPass(0);
        _lineMaterial.color = boxColor;
        GL.Begin(GL.QUADS);
        float half = Mathf.Max(1f, lineWidth / 2f);

        foreach (var b in boxes)
        {
            if (b.confidence < minConfidence) continue;
            float xMin = Mathf.Min(b.x1, b.x2) * scaleX;
            float xMax = Mathf.Max(b.x1, b.x2) * scaleX;
            float yMin = Mathf.Min(b.y1, b.y2) * scaleY;
            float yMax = Mathf.Max(b.y1, b.y2) * scaleY;
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
        GL.Vertex3(xMin - half, yMin - half, 0);
        GL.Vertex3(xMax + half, yMin - half, 0);
        GL.Vertex3(xMax + half, yMin + half, 0);
        GL.Vertex3(xMin - half, yMin + half, 0);

        GL.Vertex3(xMin - half, yMax - half, 0);
        GL.Vertex3(xMax + half, yMax - half, 0);
        GL.Vertex3(xMax + half, yMax + half, 0);
        GL.Vertex3(xMin - half, yMax + half, 0);

        GL.Vertex3(xMin - half, yMin - half, 0);
        GL.Vertex3(xMin + half, yMin - half, 0);
        GL.Vertex3(xMin + half, yMax + half, 0);
        GL.Vertex3(xMin - half, yMax + half, 0);

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
