using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// A simple net gun that fires a NetProjectile toward a target or in the forward direction.
/// Attach to a GameObject that represents the gun barrel/muzzle.
/// Press left mouse or call Fire() from code to launch.
/// </summary>
public class NetGun : MonoBehaviour
{
    [Header("Target")]
    [Tooltip("Optional target to aim at. If null, fires in the gun's forward direction.")]
    public Transform target;

    [Header("Launch")]
    [Tooltip("Speed of the net projectile (m/s).")]
    public float launchSpeed = 25f;
    [Tooltip("Minimum seconds between shots.")]
    public float cooldown = 2f;

    [Header("Net Settings")]
    [Tooltip("Grid size of the net (NxN nodes).")]
    public int netGridSize = 5;
    [Tooltip("Node spacing in the net.")]
    public float netNodeSpacing = 0.4f;
    [Tooltip("Spring force for net connections.")]
    public float netSpringForce = 200f;
    [Tooltip("Outward expansion force.")]
    public float netExpandForce = 4f;

    private float _lastFireTime = -999f;

    void Update()
    {
        if (Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame &&
            Time.time - _lastFireTime >= cooldown)
        {
            Fire();
        }
    }

    /// <summary>
    /// Fire a net projectile from this gun's position.
    /// </summary>
    public void Fire()
    {
        _lastFireTime = Time.time;

        // Calculate aim direction
        Vector3 aimDir;
        if (target != null)
            aimDir = (target.position - transform.position).normalized;
        else
            aimDir = transform.forward;

        // Create the net container
        GameObject netObj = new GameObject("NetProjectile");
        netObj.transform.position = transform.position + aimDir * 0.5f; // spawn slightly in front
        netObj.transform.rotation = Quaternion.LookRotation(aimDir);

        // Configure and build the net
        NetProjectile net = netObj.AddComponent<NetProjectile>();
        net.gridSize = netGridSize;
        net.nodeSpacing = netNodeSpacing;
        net.springForce = netSpringForce;
        net.expandForce = netExpandForce;
        net.BuildNet();

        // Launch with velocity
        Vector3 velocity = aimDir * launchSpeed;
        net.Launch(velocity);

        Debug.Log($"[NetGun] Fired net toward {(target != null ? target.name : "forward")} at {launchSpeed} m/s");
    }
}
