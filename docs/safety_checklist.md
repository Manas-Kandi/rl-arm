# Hardware Safety Checklist

1. Verify emergency-stop circuitry and watchdog nodes are active before enabling velocity commands.
2. Run the policy through the ROS safety supervisor (`src/real/ros_interface.py`) with rate limiting enabled.
3. Begin with reduced command magnitudes (25% of nominal limits) and gradually scale once stable.
4. Monitor force/torque sensors and joint limits; abort immediately on sustained limit violations.
5. Record all telemetry (joint states, door hinge angle, policy outputs) for later offline analysis.
