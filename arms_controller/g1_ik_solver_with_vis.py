
import os
import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper


class G1_IK_Arms:
    def __init__(self, visualize=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        base_path = os.path.abspath(os.path.dirname(__file__))
        urdf_path = os.path.join(base_path, "assets/g1/g1_body29_hand14.urdf")
        mesh_dir = os.path.join(base_path, "assets/g1")

        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])

        # Joints to lock (legs, waist, fingers)
        self.joints_to_lock = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_index_0_joint", "left_hand_index_1_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"
        ]

        self.reduced_robot = self.robot.buildReducedRobot(self.joints_to_lock, np.zeros(self.robot.model.nq))

        self.reduced_robot.model.addFrame(pin.Frame(
            'L_ee',
            self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
            pin.SE3(np.eye(3), np.array([0.13, 0, 0])),
            pin.FrameType.OP_FRAME
        ))
        self.reduced_robot.model.addFrame(pin.Frame(
            'R_ee',
            self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
            pin.SE3(np.eye(3), np.array([0.13, 0, 0])),
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

        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit
        ))
        translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        rotational_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        reg_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.minimize(50 * translational_cost + rotational_cost + 0.02 * reg_cost + 0.1 * smooth_cost)

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
                self.visualizer.display(pin.neutral(self.reduced_robot.model))
            except Exception as e:
                print("[WARN] Meshcat not initialized:", e)
                self.visualizer = None

    def display_configuration(self, q):
        if self.visualizer:
            self.visualizer.display(q)

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
    ik_solver = G1_IK_Arms()

    print("\n[TEST] Display zero configuration")
    ik_solver.display_configuration(np.zeros(ik_solver.nq))
    input("Press ENTER to run ik_both_pose...")

    print("\n[TEST] ik_both_pose")
    L_pose = SE3(Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.3, 0.2, 0.3]))
    R_pose = SE3(Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.3, -0.2, 0.3]))
    q_dict, tau_dict = ik_solver.ik_both_pose(L_pose.homogeneous, R_pose.homogeneous)
    q_vec = ik_solver.joint_dict_to_q(q_dict)
    print("Joint angles (dict):", q_dict)
    print("Feedforward torques (dict):", tau_dict)
    ik_solver.display_configuration(q_vec)



    input("Press ENTER to exit.")
    
    # from pinocchio import SE3, Quaternion

    # ik_solver = G1_IK_Arms()

    # # 1. Display neutral (zero) configuration
    # print("\n[STEP 1] Displaying zero configuration...")
    # ik_solver.display_configuration(np.zeros(ik_solver.nq))
    # input("Press ENTER to continue to IK example...")

    # # 2. Solve IK for target reaching
    # print("\n[STEP 2] Solving IK for arm reaching pose...")
    # L_pose = SE3(Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.45, 0.25, 0.25]))
    # R_pose = SE3(Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.45, -0.25, 0.25]))
    # q_sol, tau_sol = ik_solver.solve_ik(L_pose.homogeneous, R_pose.homogeneous)

    # print("[RESULT] IK joint solution:")
    # print(q_sol)
    # print("[RESULT] Estimated feedforward torques:")
    # print(tau_sol)

    # ik_solver.display_configuration(q_sol)
    # input("Press ENTER to continue to IK example...")

    # # 3. Solve IK for target reaching
    # print("\n[STEP 2] Solving IK for arm reaching pose...")
    # L_pose = SE3(Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.25, 0.25, 0.0]))
    # R_pose = SE3(Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.25, -0.25, 0.0]))
    # q_sol, tau_sol = ik_solver.solve_ik(L_pose.homogeneous, R_pose.homogeneous)

    # print("[RESULT] IK joint solution:")
    # print(q_sol)
    # print("[RESULT] Estimated feedforward torques:")
    # print(tau_sol)

    # ik_solver.display_configuration(q_sol)
    # input("Press ENTER to continue to IK example...")
    # # 4. Solve IK for target reaching
    # print("\n[STEP 2] Solving IK for arm reaching pose...")
    # L_pose = SE3(Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.25, 0.0, 0.0]))
    # R_pose = SE3(Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.25, -0.0, 0.0]))
    # q_sol, tau_sol = ik_solver.solve_ik(L_pose.homogeneous, R_pose.homogeneous)

    # print("[RESULT] IK joint solution:")
    # print(q_sol)
    # print("[RESULT] Estimated feedforward torques:")
    # print(tau_sol)

    # ik_solver.display_configuration(q_sol)
    # # input("Press ENTER to continue to IK example...")
    # input("Press ENTER to exit visualization...")
