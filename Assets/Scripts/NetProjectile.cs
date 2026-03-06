using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A physics-based net made of small sphere nodes connected by SpringJoints.
/// Spawned by NetGun. The net is a grid of nodes that expand after launch
/// and can wrap around objects via collisions and springs.
/// </summary>
public class NetProjectile : MonoBehaviour
{
    [Header("Net Shape")]
    [Tooltip("Number of nodes along one edge of the net grid.")]
    public int gridSize = 5;
    [Tooltip("Spacing between nodes at rest (meters).")]
    public float nodeSpacing = 0.3f;
    [Tooltip("Radius of each node sphere.")]
    public float nodeRadius = 0.05f;
    [Tooltip("Mass of each node.")]
    public float nodeMass = 0.02f;

    [Header("Spring Settings")]
    [Tooltip("Spring force connecting nodes.")]
    public float springForce = 200f;
    [Tooltip("Damper on springs.")]
    public float springDamper = 5f;

    [Header("Launch")]
    [Tooltip("How many seconds after spawn the net starts expanding.")]
    public float expandDelay = 0.3f;
    [Tooltip("Outward expansion force applied to corner/edge nodes.")]
    public float expandForce = 3f;

    [Header("Lifetime")]
    [Tooltip("Destroy the net after this many seconds (0 = never).")]
    public float lifetime = 15f;

    // All node Rigidbodies, indexed [row * gridSize + col]
    private List<Rigidbody> _nodes = new List<Rigidbody>();
    private float _spawnTime;
    private bool _expanded;

    /// <summary>
    /// Build the net grid. Call after instantiation and before launch.
    /// </summary>
    public void BuildNet()
    {
        _spawnTime = Time.time;
        Vector3 origin = transform.position;
        float halfSize = (gridSize - 1) * nodeSpacing * 0.5f;

        // Create node spheres in a grid
        for (int row = 0; row < gridSize; row++)
        {
            for (int col = 0; col < gridSize; col++)
            {
                Vector3 localPos = new Vector3(
                    col * nodeSpacing - halfSize,
                    row * nodeSpacing - halfSize,
                    0f
                );
                // Transform to world space using net's orientation
                Vector3 worldPos = origin + transform.rotation * localPos;

                GameObject node = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                node.name = $"NetNode_{row}_{col}";
                node.transform.position = worldPos;
                node.transform.localScale = Vector3.one * nodeRadius * 2f;
                node.transform.parent = transform;

                // Physics
                Rigidbody rb = node.AddComponent<Rigidbody>();
                rb.mass = nodeMass;
                rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
                rb.interpolation = RigidbodyInterpolation.Interpolate;

                // Small drag to simulate air resistance on the net
                rb.linearDamping = 0.5f;
                rb.angularDamping = 0.5f;

                // Use a simple unlit material (white)
                var renderer = node.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material = new Material(Shader.Find("Unlit/Color"));
                    renderer.material.color = new Color(0.9f, 0.9f, 0.85f);
                }

                _nodes.Add(rb);
            }
        }

        // Connect adjacent nodes with springs (horizontal + vertical + diagonal)
        for (int row = 0; row < gridSize; row++)
        {
            for (int col = 0; col < gridSize; col++)
            {
                int idx = row * gridSize + col;
                // Right neighbor
                if (col < gridSize - 1)
                    AddSpring(idx, idx + 1, nodeSpacing);
                // Bottom neighbor
                if (row < gridSize - 1)
                    AddSpring(idx, idx + gridSize, nodeSpacing);
                // Diagonal (bottom-right)
                if (col < gridSize - 1 && row < gridSize - 1)
                    AddSpring(idx, idx + gridSize + 1, nodeSpacing * 1.414f);
                // Diagonal (bottom-left)
                if (col > 0 && row < gridSize - 1)
                    AddSpring(idx, idx + gridSize - 1, nodeSpacing * 1.414f);
            }
        }

        if (lifetime > 0f)
            Destroy(gameObject, lifetime);
    }

    private void AddSpring(int idxA, int idxB, float restLength)
    {
        SpringJoint spring = _nodes[idxA].gameObject.AddComponent<SpringJoint>();
        spring.connectedBody = _nodes[idxB];
        spring.spring = springForce;
        spring.damper = springDamper;
        spring.minDistance = 0f;
        spring.maxDistance = restLength * 1.5f;
        spring.autoConfigureConnectedAnchor = true;
        spring.enableCollision = true;
    }

    void FixedUpdate()
    {
        // After a short delay, apply outward force to expand the net
        if (!_expanded && Time.time - _spawnTime > expandDelay)
        {
            _expanded = true;
            ExpandNet();
        }
    }

    private void ExpandNet()
    {
        float halfSize = (gridSize - 1) * 0.5f;
        for (int row = 0; row < gridSize; row++)
        {
            for (int col = 0; col < gridSize; col++)
            {
                int idx = row * gridSize + col;
                // Direction from center of grid outward
                float dx = col - halfSize;
                float dy = row - halfSize;
                float dist = Mathf.Sqrt(dx * dx + dy * dy);
                if (dist < 0.01f) continue;

                Vector3 outDir = transform.rotation * new Vector3(dx / dist, dy / dist, 0f);
                _nodes[idx].AddForce(outDir * expandForce, ForceMode.Impulse);
            }
        }
    }

    /// <summary>
    /// Launch the entire net with a velocity.
    /// </summary>
    public void Launch(Vector3 velocity)
    {
        foreach (var rb in _nodes)
        {
            rb.linearVelocity = velocity;
        }
    }
}
