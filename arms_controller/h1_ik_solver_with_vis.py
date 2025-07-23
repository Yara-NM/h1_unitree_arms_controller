
import os
import numpy as np
import casadi
import pinocchio as pin
import meshcat.geometry as mg
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper


class H1_IK_Arms:
    def __init__(self, visualize=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        base_path = os.path.abspath(os.path.dirname(__file__))
        urdf_path = os.path.join(base_path, "assets/h1/h1.urdf")
        mesh_dir = os.path.join(base_path, "assets/")

        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])

        # Joints to lock (legs, waist, fingers)
        self.joints_to_lock = [
           "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
            "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_ankle_joint", "right_ankle_joint",
            "torso_joint", 
              ]

        self.reduced_robot = self.robot.buildReducedRobot(self.joints_to_lock, 
                                                          np.zeros(self.robot.model.nq))

        self.reduced_robot.model.addFrame(pin.Frame(
            'L_ee',
            self.reduced_robot.model.getJointId('left_elbow_joint'),
            pin.SE3(np.eye(3), np.array([0.36, 0, -0.02])),
            pin.FrameType.OP_FRAME
        ))
        self.reduced_robot.model.addFrame(pin.Frame(
            'R_ee',
            self.reduced_robot.model.getJointId('right_elbow_joint'),
            pin.SE3(np.eye(3), np.array([0.36, 0, -0.02])),
            pin.FrameType.OP_FRAME
        ))

        self.reduced_robot.data = self.reduced_robot.model.createData()
        self.L_ee_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_ee_id = self.reduced_robot.model.getFrameId("R_ee")
        self.nq = self.reduced_robot.model.nq

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.translational_error = casadi.Function("translational_error", [self.cq, self.cTf_l, self.cTf_r], [
            casadi.vertcat(
                self.cdata.oMf[self.L_ee_id].translation - self.cTf_l[:3, 3],
                self.cdata.oMf[self.R_ee_id].translation - self.cTf_r[:3, 3]
            )
        ])
        self.rotational_error = casadi.Function("rotational_error", [self.cq, self.cTf_l, self.cTf_r], [
            casadi.vertcat(
                cpin.log3(self.cdata.oMf[self.L_ee_id].rotation @ self.cTf_l[:3, :3].T),
                cpin.log3(self.cdata.oMf[self.R_ee_id].rotation @ self.cTf_r[:3, :3].T)
            )
        ])

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.var_q_last = self.opti.parameter(self.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)

        
        translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        rotational_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        reg_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit
        ))
        self.opti.minimize(50 * translational_cost + 0.008 * rotational_cost + 0.02 * reg_cost + 0.1 * smooth_cost)

        opts = {'ipopt': {'print_level': 0, 'max_iter': 50, 'tol': 1e-6}, 'print_time': False}
        self.opti.solver("ipopt", opts)
        self.init_data = np.zeros(self.nq)

        # Meshcat visualization
        self.visualizer = None
        if visualize:
            try:
                from pinocchio.visualize import MeshcatVisualizer
                self.visualizer = MeshcatVisualizer(self.reduced_robot.model,
                                                    self.reduced_robot.collision_model,
                                                    self.reduced_robot.visual_model)
                self.visualizer.initViewer(open=True)
                self.visualizer.loadViewerModel("pinocchio")
                self.visualizer.displayFrames(True, frame_ids=[105, 106], axis_length = 0.15, axis_width = 5)
                self.visualizer.display(pin.neutral(self.reduced_robot.model))
                # Enable the display of end effector target frames with short axis lengths and greater width.
                frame_viz_names = ['L_ee_frame', 'R_ee_frame']
                FRAME_AXIS_POSITIONS = (
                    np.array([[0, 0, 0], [1, 0, 0],
                            [0, 0, 0], [0, 1, 0],
                            [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
                )
                FRAME_AXIS_COLORS = (
                    np.array([[1.0, 0.3, 0.3], [1.0, 0.7, 0.7],
                            [0.3, 1.0, 0.5], [0.7, 1.0, 0.8],
                            [0.3, 0.8, 1.0], [0.7, 0.9, 1.0]]).astype(np.float32).T
                )
                axis_length = 0.1
                axis_width = 4
                for frame_viz_name in frame_viz_names:
                    self.visualizer.viewer[frame_viz_name].set_object(
                        mg.LineSegments(
                            mg.PointsGeometry(
                                position=axis_length * FRAME_AXIS_POSITIONS,
                                color=FRAME_AXIS_COLORS,
                            ),
                            mg.LineBasicMaterial(
                                linewidth=axis_width,
                                vertexColors=True,
                            ),
                        )
                    )
            except Exception as e:
                print("[WARN] Meshcat not initialized:", e)
                self.visualizer = None

    def display_configuration(self, q):
        if self.visualizer:
            self.visualizer.display(q)
            pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
            pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

            L_pose = self.reduced_robot.data.oMf[self.L_ee_id]
            R_pose = self.reduced_robot.data.oMf[self.R_ee_id]

            self.visualizer.viewer["L_ee_frame"].set_transform(L_pose.homogeneous)
            self.visualizer.viewer["R_ee_frame"].set_transform(R_pose.homogeneous)
        def modify_end_effector_frame(self, side: str, new_offset: np.ndarray):
            """
            Modify the end-effector frame translation offset (relative to its parent link).

            Parameters:
                side (str): 'L' for left arm or 'R' for right arm.
                new_offset (np.ndarray): 3-element vector (x, y, z) in meters.
            """
            assert new_offset.shape == (3,), "new_offset must be a 3-element vector"
            
            if side.upper() == "L":
                frame_id = self.L_ee_id
            elif side.upper() == "R":
                frame_id = self.R_ee_id
            else:
                raise ValueError("Side must be 'L' or 'R'")
            
            self.reduced_robot.model.frames[frame_id].placement.translation = new_offset
            print(f"[INFO] {side.upper()} end-effector frame offset updated to {new_offset}")


    def solve_ik(self, left_tf, right_tf, q_init=None, dq=None):
        if q_init is not None:
            self.init_data = q_init
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.var_q_last, self.init_data)
        self.opti.set_value(self.param_tf_l, left_tf)
        self.opti.set_value(self.param_tf_r, right_tf)

        try:
            sol = self.opti.solve()
            q_sol = self.opti.value(self.var_q)
            self.init_data = q_sol
            dq = dq if dq is not None else np.zeros_like(q_sol)
            tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, q_sol, dq, np.zeros_like(dq))
            return q_sol, tauff
        except Exception as e:
            print(f"[IK Error] {e}")
            return self.init_data, np.zeros(self.nq)
        
    def forward_kinematics(self, q_dict):
        """Returns the current SE3 transform of left and right end effectors."""
        q = np.zeros(self.nq)
        for j in range(self.reduced_robot.model.njoints):
            idx_q = self.reduced_robot.model.joints[j].idx_q
            if idx_q >= 0:
                joint_name = self.reduced_robot.model.names[j]
                q[idx_q] = q_dict.get(joint_name, 0.0)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        L_pose = self.reduced_robot.data.oMf[self.L_ee_id]
        R_pose = self.reduced_robot.data.oMf[self.R_ee_id]
        return {"L_ee": L_pose, "R_ee": R_pose}
    

    def joint_dict_to_q(self, joint_dict):
        """Converts a joint dictionary back to a q vector."""
        q = np.zeros(self.nq)
        for j in range(self.reduced_robot.model.njoints):
            idx_q = self.reduced_robot.model.joints[j].idx_q
            if idx_q >= 0:
                joint_name = self.reduced_robot.model.names[j]
                if joint_name in joint_dict:
                    q[idx_q] = joint_dict[joint_name]
        return q
    def _q_to_dict(self, q_vec):
        """Internal utility to convert joint vector to name-keyed dictionary."""
        q_dict = {}
        for j in range(self.reduced_robot.model.njoints):
            idx_q = self.reduced_robot.model.joints[j].idx_q
            if idx_q >= 0:
                name = self.reduced_robot.model.names[j]
                q_dict[name] = q_vec[idx_q]
        return q_dict
    
    def ik_both_pose(self, left_tf, right_tf, q_init=None, dq=None):
        q, tau = self.solve_ik(left_tf, right_tf, q_init=q_init, dq=dq)
        return self._q_to_dict(q), self._q_to_dict(tau)
 

if __name__ == "__main__":

     
    from pinocchio import SE3, Quaternion

    ik_solver = H1_IK_Arms(visualize=True)

    # Start from zero joint configuration
    q0 = np.zeros(ik_solver.nq)
    ik_solver.display_configuration(q0)
    input("\n[0] Displaying initial zero configuration. Press ENTER to begin tests...")

    # Get initial end-effector pose
    pin.forwardKinematics(ik_solver.reduced_robot.model, ik_solver.reduced_robot.data, q0)
    pin.updateFramePlacements(ik_solver.reduced_robot.model, ik_solver.reduced_robot.data)
    init_L_pose = ik_solver.reduced_robot.data.oMf[ik_solver.L_ee_id]
    init_R_pose = ik_solver.reduced_robot.data.oMf[ik_solver.R_ee_id]  # unchanged

    # Directions to test (x, y, z) ±10 cm
    directions = {
        "+x": np.array([0.10, 0.00, 0.00]),
        "-x": np.array([-0.10, 0.00, 0.00]),
        "+y": np.array([0.00, 0.10, 0.00]),
        "-y": np.array([0.00, -0.10, 0.00]),
        "+z": np.array([0.00, 0.00, 0.10]),
        "-z": np.array([0.00, 0.00, -0.10]),
    }

    for label, delta in directions.items():
        print(f"\n[{label}] Moving Left EE by {delta} from initial pose...")
        input("Press ENTER to solve IK and display result...")

        # Build new target pose for left EE
        target_L = SE3(init_L_pose.rotation, init_L_pose.translation + delta)
        target_R = init_R_pose  # Keep right arm unchanged

        q_dict, _ = ik_solver.ik_both_pose(target_L.homogeneous, target_R.homogeneous)
        q_vec = ik_solver.joint_dict_to_q(q_dict)
        ik_solver.display_configuration(q_vec)

        # Measure achieved pose
        achieved_pose = ik_solver.forward_kinematics(q_dict)["L_ee"]
        error = achieved_pose.translation - target_L.translation

        print(f"  ➤ Target translation:   {target_L.translation}")
        print(f"  ➤ Achieved translation: {achieved_pose.translation}")
        print(f"  ➤ Error:                {error}")

        input("Press ENTER to return to initial pose...")

        ik_solver.display_configuration(q0)

    print("\n✅ Finished all movement tests.")