import numpy as np


# ======================================================================
# COMMON ANT CONTROLLER
# ======================================================================

class AntController:
    """
    Base controller interface for MuJoCo Ant-v5.

    Ant-v5 has:
        8 actuators
        8 joint positions
        8 joint velocities

    Controller comparison:

        Direct:
            Open-loop actuator trajectory.
            Uses only time and intention.

        CPG + PD:
            Fixed oscillator -> desired joint positions -> fixed PD.
            Uses joint position/velocity feedback.
            Does not use morphology parameters.

        Morphology-aware PPO:
            Learned policy.
            Uses the complete observation containing:
                Ant state + morphology + intention.
    """

    ACTION_DIM = 8

    # ------------------------------------------------------------------
    # Ant-v5 observation layout
    # ------------------------------------------------------------------

    # Standard Ant-v5 observation:
    #
    # qpos[2:] contains 13 values:
    #
    #   0       torso z
    #   1:5     torso quaternion
    #   5:13    eight joint positions
    #
    # qvel contains 14 values:
    #
    #   0:3     torso linear velocity
    #   3:6     torso angular velocity
    #   6:14    eight joint velocities
    #
    # Therefore, in the standard Ant observation:
    #
    #   joint positions = obs[5:13]
    #   joint velocities = obs[19:27]
    #
    # The project environment appends morphology and intention
    # information after the standard Ant observation.
    # ------------------------------------------------------------------

    JOINT_POSITION_START = 5
    JOINT_POSITION_END = 13

    JOINT_VELOCITY_START = 19
    JOINT_VELOCITY_END = 27

    MIN_ANT_OBSERVATION_DIM = 27

    # ------------------------------------------------------------------
    # Physical actuator ordering
    # ------------------------------------------------------------------
    #
    # FIXED: the previous mapping here was wrong for 3 of 4 legs. It
    # was empirically re-verified by walking each actuator's joint up
    # the MuJoCo body tree to its aux_N attachment point and reading
    # that body's actual (x, y) position (see diagnose_leg_sides.py).
    # Ground truth, confirmed on the installed Ant-v5 model:
    #
    #   0,1 -> aux_4 -> front-right leg   (was mislabeled back-right)
    #   2,3 -> aux_1 -> front-left leg    (was already correct)
    #   4,5 -> aux_2 -> back-left leg     (was mislabeled front-right)
    #   6,7 -> aux_3 -> back-right leg    (was mislabeled back-left)
    #
    # Only actuators 2,3 (front-left) were correctly labeled before.
    # This mismatch is what caused LEFT_LEGS / RIGHT_LEGS below to
    # each contain one genuine left-side leg and one genuine
    # right-side leg, which in turn caused the "Right" intention to
    # produce a net LEFT turn instead of a right turn (confirmed via
    # test_intentions.py: Right showed the same positive yaw sign as
    # Left instead of the opposite sign).
    #
    # Each pair contains:
    #
    #   first actuator  -> hip
    #   second actuator -> ankle
    #
    # Physical arrangement (verified):
    #
    #                    FRONT
    #
    #          2,3                    0,1
    #       front-left             front-right
    #
    #
    #          4,5                    6,7
    #        back-left              back-right
    #
    # ------------------------------------------------------------------

    LEG_GROUPS = (
        np.array([2, 3], dtype=np.int64),  # leg 0: front-left  (aux_1)
        np.array([0, 1], dtype=np.int64),  # leg 1: front-right (aux_4)
        np.array([4, 5], dtype=np.int64),  # leg 2: back-left   (aux_2)
        np.array([6, 7], dtype=np.int64),  # leg 3: back-right  (aux_3)
    )

    # Physical sides:
    #
    #   Left  = front-left + back-left
    #   Right = front-right + back-right

    LEFT_LEGS = (0, 2)
    RIGHT_LEGS = (1, 3)

    # ------------------------------------------------------------------
    # Diagonal gait phases
    # ------------------------------------------------------------------
    #
    # Front-left  + back-right -> phase 0
    # Front-right + back-left  -> phase pi
    #
    # This creates the alternating diagonal gait.
    # ------------------------------------------------------------------

    LEG_PHASE = np.array(
        [
            0.0,       # front-left
            np.pi,     # front-right
            np.pi,     # back-left
            0.0,       # back-right
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # Physical ankle orientation
    # ------------------------------------------------------------------
    #
    # The ankle actuators have opposite physical orientation on the
    # left and right sides of the Ant model.
    #
    # These signs are applied consistently to the nominal gait and
    # CPG desired ankle trajectories.
    # ------------------------------------------------------------------

    ANKLE_SIGN = {
        0: -1.0,      # front-left
        1:  1.0,      # front-right
        2: -1.0,      # back-left
        3:  1.0,      # back-right
    }

    def __init__(self):
        self.previous_action = np.zeros(
            self.ACTION_DIM,
            dtype=np.float32,
        )

    def reset(self):
        self.previous_action[:] = 0.0

    def act(self, intention, step, obs=None):
        raise NotImplementedError


# ======================================================================
# DIRECT TRAJECTORY CONTROLLER
# ======================================================================

class DirectTrajectoryController(AntController):
    """
    Open-loop trajectory controller.

    The controller generates a fixed actuator trajectory using only:
        - time
        - intention

    It does NOT use:
        - joint positions
        - joint velocities
        - morphology
        - feedback
        - learned parameters

    Therefore, it represents the least adaptive baseline.
    """

    def __init__(
        self,
        amplitude=0.45,
        frequency=1.5,
        dt=0.01,
    ):
        super().__init__()

        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.dt = float(dt)

    def reset(self):
        super().reset()

    def _base_gait(self, step):
        """
        Generate the fixed nominal actuator trajectory.

        The ankle direction is corrected for the physical
        Ant-v5 actuator orientation.

        Returns:
            action: shape (8,)
        """

        time = float(step) * self.dt

        phase = (
            2.0
            * np.pi
            * self.frequency
            * time
        )

        action = np.zeros(
            self.ACTION_DIM,
            dtype=np.float32,
        )

        for leg_id, leg_indices in enumerate(
            self.LEG_GROUPS
        ):

            leg_phase = (
                phase
                + self.LEG_PHASE[leg_id]
            )

            hip_index = leg_indices[0]
            ankle_index = leg_indices[1]

            hip_signal = np.sin(
                leg_phase
            )

            ankle_signal = np.sin(
                leg_phase + np.pi
            )

            action[hip_index] = (
                self.amplitude
                * hip_signal
            )

            action[ankle_index] = (
                self.ANKLE_SIGN[leg_id]
                * 0.65
                * self.amplitude
                * ankle_signal
            )

        return action

    def act(
        self,
        intention,
        step,
        obs=None,
    ):
        """
        Generate an open-loop actuator command.

        The observation is deliberately ignored.
        """

        action = self._base_gait(step)

        # --------------------------------------------------------------
        # Forward
        # --------------------------------------------------------------

        if intention == 0:

            # Use the nominal gait.
            pass

        # --------------------------------------------------------------
        # Left
        # --------------------------------------------------------------

        elif intention == 1:

            # FIXED comment (behavior unchanged): under this project's
            # frame convention (x=forward, y=left, z=up), POSITIVE yaw
            # is the left-turn direction. Slowing propulsion on the
            # left side while the right side keeps pushing at full
            # strength pivots the body toward positive (left) yaw,
            # which is exactly the physically correct behavior for a
            # left turn. No logic change needed here.

            for leg_id in self.LEFT_LEGS:

                indices = self.LEG_GROUPS[leg_id]

                action[indices] *= 0.55

            for leg_id in self.RIGHT_LEGS:

                indices = self.LEG_GROUPS[leg_id]

                action[indices] *= 1.0

        # --------------------------------------------------------------
        # Right
        # --------------------------------------------------------------

        elif intention == 2:

            # FIXED comment (behavior unchanged): NEGATIVE yaw is the
            # right-turn direction under this project's frame
            # convention. Slowing the right side pivots the body
            # toward negative (right) yaw. No logic change needed
            # here.

            for leg_id in self.LEFT_LEGS:

                indices = self.LEG_GROUPS[leg_id]

                action[indices] *= 1.0

            for leg_id in self.RIGHT_LEGS:

                indices = self.LEG_GROUPS[leg_id]

                action[indices] *= 0.55

        # --------------------------------------------------------------
        # Stop
        # --------------------------------------------------------------

        elif intention == 3:

            # The open-loop controller cannot actively stabilize.
            # It simply removes almost all actuation.

            action *= 0.05

        else:

            raise ValueError(
                f"Invalid intention: {intention}"
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        ).astype(np.float32)

        self.previous_action[:] = action

        return action


# ======================================================================
# FIXED CPG + LOCAL PD CONTROLLER
# ======================================================================

class FixedCPGController(AntController):
    """
    Fixed central-pattern-generator controller with local PD feedback.

    Architecture:

        fixed oscillator
              |
              v
        desired joint positions
              |
              v
        fixed local PD controller
              |
              v
        normalized actuator command

    The controller uses:
        - time
        - intention
        - joint positions
        - joint velocities

    The controller does NOT use:
        - mass
        - limb length
        - joint strength
        - friction
        - morphology vector
        - learned policy

    The same oscillator frequency and PD gains are used for every
    morphology condition.

    The controller therefore adapts to instantaneous joint state
    through local feedback, but does not adapt its parameters to
    morphology.

    Gain tuning note
    -----------------
    A multi-seed sweep (see tune_cpg.py / tune_cpg_round2.py /
    robust_check.py in the project root) initially suggested
    frequency=1.30, hip_amplitude=0.35, kd=0.50 as an improvement,
    based on Forward-intention distance alone. Evaluating that
    candidate across ALL FOUR intentions (eval_all_intentions.py)
    showed it was overfit to Forward: it did walk ~2.5x further
    going straight, but Left and Right fell 4-5x sooner (avg
    fall_step ~40-47 vs ~170-194), and Stop drifted more than twice
    as far while falling twice as often. The ORIGINAL defaults below
    are the better-balanced controller once all four intentions are
    weighed, so they have been kept rather than replaced. This
    controller is not tuned to be maximally stable in any one
    direction -- it is meant to remain a reasonably-behaved,
    non-adaptive baseline that the morphology-aware PPO controller is
    expected to outperform under perturbation.
    """

    def __init__(
        self,
        frequency=1.5,
        hip_amplitude=0.65,
        ankle_amplitude=0.40,
        kp=1.8,
        kd=0.20,
        action_scale=2.0,
        dt=0.01,
    ):
        super().__init__()

        self.frequency = float(frequency)

        self.hip_amplitude = float(
            hip_amplitude
        )

        self.ankle_amplitude = float(
            ankle_amplitude
        )

        self.kp = float(kp)
        self.kd = float(kd)

        self.action_scale = float(
            action_scale
        )

        self.dt = float(dt)

    def reset(self):
        super().reset()

    # ------------------------------------------------------------------
    # Observation extraction
    # ------------------------------------------------------------------

    def _extract_joint_state(self, obs):
        """
        Extract the eight joint positions and eight joint velocities
        from the Ant-v5 observation.

        Returns:
            joint_position: shape (8,)
            joint_velocity: shape (8,)
        """

        if obs is None:

            raise ValueError(
                "FixedCPGController requires "
                "the current Ant observation."
            )

        obs = np.asarray(
            obs,
            dtype=np.float32,
        ).reshape(-1)

        if obs.shape[0] < self.MIN_ANT_OBSERVATION_DIM:

            raise ValueError(
                "Unexpected Ant observation size. "
                f"Expected at least "
                f"{self.MIN_ANT_OBSERVATION_DIM} values, "
                f"got {obs.shape[0]}."
            )

        joint_position = obs[
            self.JOINT_POSITION_START:
            self.JOINT_POSITION_END
        ]

        joint_velocity = obs[
            self.JOINT_VELOCITY_START:
            self.JOINT_VELOCITY_END
        ]

        if joint_position.shape[0] != 8:

            raise ValueError(
                "Expected 8 Ant joint positions, "
                f"got {joint_position.shape[0]}."
            )

        if joint_velocity.shape[0] != 8:

            raise ValueError(
                "Expected 8 Ant joint velocities, "
                f"got {joint_velocity.shape[0]}."
            )

        return (
            joint_position.astype(np.float32),
            joint_velocity.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Fixed oscillator
    # ------------------------------------------------------------------

    def _oscillator_phase(self, step):
        """
        Return the global oscillator phase.
        """

        time = float(step) * self.dt

        return (
            2.0
            * np.pi
            * self.frequency
            * time
        )

    def _oscillator_targets(
        self,
        step,
        intention,
    ):
        """
        Generate desired joint positions.

        Returns:
            desired_position: shape (8,)
        """

        phase = self._oscillator_phase(
            step
        )

        desired = np.zeros(
            self.ACTION_DIM,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Forward
        # --------------------------------------------------------------

        if intention == 0:

            for leg_id, leg_indices in enumerate(
                self.LEG_GROUPS
            ):

                leg_phase = (
                    phase
                    + self.LEG_PHASE[leg_id]
                )

                hip_index = leg_indices[0]
                ankle_index = leg_indices[1]

                desired[hip_index] = (
                    self.hip_amplitude
                    * np.sin(leg_phase)
                )

                desired[ankle_index] = (
                    self.ANKLE_SIGN[leg_id]
                    * self.ankle_amplitude
                    * np.sin(
                        leg_phase + np.pi
                    )
                )

        # --------------------------------------------------------------
        # Left
        # --------------------------------------------------------------

        elif intention == 1:

            for leg_id, leg_indices in enumerate(
                self.LEG_GROUPS
            ):

                leg_phase = (
                    phase
                    + self.LEG_PHASE[leg_id]
                )

                # FIXED comment (behavior unchanged): POSITIVE yaw =
                # left under this project's frame convention. Keep the
                # normal gait direction and phase; only reduce the
                # amplitude of the left side to pivot toward positive
                # (left) yaw.

                side_scale = (
                    0.55
                    if leg_id in self.LEFT_LEGS
                    else 1.0
                )

                hip_index = leg_indices[0]
                ankle_index = leg_indices[1]

                desired[hip_index] = (
                    side_scale
                    * self.hip_amplitude
                    * np.sin(leg_phase)
                )

                desired[ankle_index] = (
                    side_scale
                    * self.ANKLE_SIGN[leg_id]
                    * self.ankle_amplitude
                    * np.sin(
                        leg_phase + np.pi
                    )
                )

        # --------------------------------------------------------------
        # Right
        # --------------------------------------------------------------

        elif intention == 2:

            for leg_id, leg_indices in enumerate(
                self.LEG_GROUPS
            ):

                leg_phase = (
                    phase
                    + self.LEG_PHASE[leg_id]
                )

                # FIXED comment (behavior unchanged): NEGATIVE yaw =
                # right under this project's frame convention. Keep
                # the normal gait direction and phase; only reduce the
                # amplitude of the right side to pivot toward negative
                # (right) yaw.

                side_scale = (
                    1.0
                    if leg_id in self.LEFT_LEGS
                    else 0.55
                )

                hip_index = leg_indices[0]
                ankle_index = leg_indices[1]

                desired[hip_index] = (
                    side_scale
                    * self.hip_amplitude
                    * np.sin(leg_phase)
                )

                desired[ankle_index] = (
                    side_scale
                    * self.ANKLE_SIGN[leg_id]
                    * self.ankle_amplitude
                    * np.sin(
                        leg_phase + np.pi
                    )
                )

        # --------------------------------------------------------------
        # Stop
        # --------------------------------------------------------------

        elif intention == 3:

            # Zero desired joint positions.
            #
            # The PD controller therefore attempts to bring the
            # joints toward rest while retaining local feedback.

            desired[:] = 0.0

        else:

            raise ValueError(
                f"Invalid intention: {intention}"
            )

        return desired

    # ------------------------------------------------------------------
    # Local PD controller
    # ------------------------------------------------------------------

    def _pd_control(
        self,
        desired_position,
        joint_position,
        joint_velocity,
    ):
        """
        Fixed-gain local PD control.

        tau = Kp * position_error - Kd * velocity

        The same Kp and Kd are used for every morphology.

        The resulting command is normalized to [-1, 1].
        """

        position_error = (
            desired_position
            - joint_position
        )

        pd_command = (
            self.kp
            * position_error
            - self.kd
            * joint_velocity
        )

        action = (
            pd_command
            / self.action_scale
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        return action.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Controller step
    # ------------------------------------------------------------------

    def act(
        self,
        intention,
        step,
        obs=None,
    ):
        """
        Generate a fixed-CPG + local-PD actuator command.
        """

        (
            joint_position,
            joint_velocity,
        ) = self._extract_joint_state(
            obs
        )

        desired_position = (
            self._oscillator_targets(
                step=step,
                intention=intention,
            )
        )

        action = self._pd_control(
            desired_position=desired_position,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
        )

        self.previous_action[:] = action

        return action


# ======================================================================
# MORPHOLOGY-AWARE PPO CONTROLLER
# ======================================================================

class MorphologyAwarePPOController(AntController):
    """
    Morphology-aware PPO controller.

    Expected observation:

        105-dimensional Ant state
        +
        4 morphology parameters
        +
        4-dimensional intention one-hot
        --------------------------------
        113 dimensions

    The policy therefore receives:

        1. high-level motor intention
        2. current morphology
        3. current Ant state

    The policy learns to transform the same intention into appropriate
    motor commands under different physical conditions.
    """

    EXPECTED_OBSERVATION_DIM = 113

    def __init__(self, model):
        super().__init__()

        self.model = model

    def reset(self):
        super().reset()

    def act(
        self,
        intention,
        step,
        obs=None,
    ):
        """
        Predict the actuator command from the complete observation.

        The intention and step arguments are retained so that all
        controllers share the same interface.

        For PPO, the actual intention and morphology information must
        already be present inside obs.
        """

        if obs is None:

            raise ValueError(
                "MorphologyAwarePPOController requires "
                "the complete 113-dimensional observation."
            )

        obs = np.asarray(
            obs,
            dtype=np.float32,
        ).reshape(-1)

        if obs.shape[0] != (
            self.EXPECTED_OBSERVATION_DIM
        ):

            raise ValueError(
                "Unexpected PPO observation size. "
                f"Expected "
                f"{self.EXPECTED_OBSERVATION_DIM}, "
                f"got {obs.shape[0]}."
            )

        action, _ = self.model.predict(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape[0] != self.ACTION_DIM:

            raise ValueError(
                "Unexpected PPO action size. "
                f"Expected {self.ACTION_DIM}, "
                f"got {action.shape[0]}."
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        ).astype(np.float32)

        self.previous_action[:] = action

        return action