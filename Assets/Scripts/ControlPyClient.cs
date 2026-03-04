using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Unity behaviour that connects to the ControlPy TCP server on load and sends a "Hello World" message (echo type).
/// Ensure the Python server is running (python ControlPy/main.py) before entering Play mode.
/// </summary>
public class ControlPyClient : MonoBehaviour
{
    [Tooltip("Host address of the ControlPy server.")]
    public string host = "127.0.0.1";

    [Tooltip("Port of the ControlPy server.")]
    public int port = 5555;

    [Tooltip("If true, attempt connection when this component is loaded (Start).")]
    public bool connectOnLoad = true;

    private void Start()
    {
        if (connectOnLoad)
        {
            Thread thread = new Thread(ConnectAndSendHello);
            thread.IsBackground = true;
            thread.Start();
        }
    }

    private void ConnectAndSendHello()
    {
        try
        {
            using (var client = new TcpClient())
            {
                client.Connect(host, port);
                using (NetworkStream stream = client.GetStream())
                {
                    SendMessage(stream, "echo", "Hello World");
                    // Optionally read echo response
                    string replyType;
                    string replyPayload;
                    if (TryReadMessage(stream, out replyType, out replyPayload))
                        Debug.Log($"[ControlPyClient] Echo reply: {replyPayload}");
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[ControlPyClient] Could not connect to {host}:{port}: {ex.Message}");
        }
    }

    /// <summary>
    /// Encodes and sends one message: [4-byte length][1-byte type length][type][payload]. Big-endian.
    /// </summary>
    public static void SendMessage(NetworkStream stream, string msgType, string payload)
    {
        SendMessage(stream, msgType, Encoding.UTF8.GetBytes(payload));
    }

    /// <summary>
    /// Encodes and sends one message with binary payload (e.g. image bytes). Big-endian.
    /// </summary>
    public static void SendMessage(NetworkStream stream, string msgType, byte[] payloadBytes)
    {
        byte[] typeBytes = Encoding.UTF8.GetBytes(msgType);
        if (typeBytes.Length > 255)
            throw new ArgumentException("Message type must be at most 255 bytes.", nameof(msgType));
        int bodyLength = 1 + typeBytes.Length + payloadBytes.Length;
        byte[] packet = new byte[4 + bodyLength];
        packet[0] = (byte)(bodyLength >> 24);
        packet[1] = (byte)(bodyLength >> 16);
        packet[2] = (byte)(bodyLength >> 8);
        packet[3] = (byte)bodyLength;
        packet[4] = (byte)typeBytes.Length;
        Buffer.BlockCopy(typeBytes, 0, packet, 5, typeBytes.Length);
        Buffer.BlockCopy(payloadBytes, 0, packet, 5 + typeBytes.Length, payloadBytes.Length);
        stream.Write(packet, 0, packet.Length);
    }

    /// <summary>
    /// Reads one length-prefixed message from the stream. Returns true if a full message was read.
    /// </summary>
    public static bool TryReadMessage(NetworkStream stream, out string msgType, out string payload)
    {
        msgType = null;
        payload = null;
        byte[] lenBuf = new byte[4];
        if (ReadExactly(stream, lenBuf, 0, 4) != 4)
            return false;
        int bodyLength = (lenBuf[0] << 24) | (lenBuf[1] << 16) | (lenBuf[2] << 8) | lenBuf[3];
        byte[] body = new byte[bodyLength];
        if (ReadExactly(stream, body, 0, bodyLength) != bodyLength)
            return false;
        int typeLen = body[0];
        msgType = Encoding.UTF8.GetString(body, 1, typeLen);
        payload = Encoding.UTF8.GetString(body, 1 + typeLen, bodyLength - 1 - typeLen);
        return true;
    }

    private static int ReadExactly(NetworkStream stream, byte[] buffer, int offset, int count)
    {
        int read = 0;
        while (read < count)
        {
            int n = stream.Read(buffer, offset + read, count - read);
            if (n == 0)
                break;
            read += n;
        }
        return read;
    }
}
